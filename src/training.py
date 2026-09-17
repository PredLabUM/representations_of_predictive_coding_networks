import copy
import os

import tensorflow as tf
import keras


def get_inv_activation(activation):
    """Returns the inverse of an activation if an appropriate inverse exists."""
    if activation == tf.keras.activations.linear:
        inv_activation = tf.keras.activations.get("linear")
    elif activation == tf.keras.activations.relu:
        # Heuristically maps values that were lower than 0 before to 0.
        inv_activation = tf.keras.activations.get("linear")
    elif activation == tf.keras.activations.get("softplus"):
        # More stable implementation of the softplus inverse.
        inv_activation = lambda x: x + tf.math.log(1. - tf.math.exp(-x))
    elif activation == tf.keras.activations.get("tanh"):
        inv_activation = lambda x: tf.math.atanh(tf.clip_by_value(x, -0.999999, 0.999999))
    elif activation == tf.keras.activations.get("sigmoid"):
        inv_activation = lambda x: -tf.math.log(1. / tf.clip_by_value(x, 1e-7, 1 - 1e-7) - 1.)
    elif activation == tf.keras.activations.get("leaky_relu"):  # assumes an alpha of 0.2
        inv_activation = lambda x: tf.where(x >= 0, x, x / 0.2)
    else:
        raise ValueError("Activation Function not invertible")
    return inv_activation


####################
# Training functions
####################

def model_fit_layers(model, train_data, validation_data=None, progressive=True, **kwargs):
    """Progressive=False uses the original y as target. Progressive=True uses the previous layer output as input."""
    # Get optimizer config.
    if isinstance(model.loss, FFLossWithThreshold):
        progressive = False
    else:
        progressive = True

    # Loop through all layers.
    histories = []
    for i, layer in enumerate(model.layers):
        for x, y in train_data.take(1):
            layer.build(x.shape)

        if layer.trainable and isinstance(layer, (tf.keras.layers.SimpleRNN,)):
            # Compile current layer.
            if len(layer.trainable_variables) > 0:
                layer_model = tf.keras.models.Sequential([layer])

                # Adapt the loss to the current layer when necessary.
                if isinstance(model.loss, PredPriorLoss):
                    layer_loss = PredPriorLoss(layer_model)
                elif isinstance(model.loss, FFLossWithThreshold):
                    layer_loss = FFLossWithThreshold(10., average=False)
                else:
                    layer_loss = model.loss

                layer_model.compile(optimizer=copy.copy(model.optimizer), loss=layer_loss)

                # Add a layer indicator for saving the model, so we can stitch them together later.
                if "callbacks" in kwargs:
                    for cb in kwargs["callbacks"]:
                        if isinstance(cb, SafeModelCheckpoint):
                            parts = cb.filepath_template.rsplit(".", 1)
                            cb.filepath_template = f"{parts[0]}_layer{i}.{parts[1]}"

                # Fit the layer.
                layer_hist = layer_model.fit(train_data, validation_data=validation_data, **kwargs)
                histories.append(layer_hist)

        # Functions to map the input through the layer.
        @tf.function
        def map_progressive(x, y):
            """Encodes input and target through the corresponding layer."""
            x_transformed = layer(x)
            return x_transformed, x_transformed

        @tf.function
        def map_non_progressive(x, y):
            """Encodes only input through the corresponding layer."""
            x_transformed = layer(x)
            return x_transformed, y

        if progressive:
            train_data = train_data.map(map_progressive)  # The target is also encoded.
        else:
            train_data = train_data.map(map_non_progressive)  # Only the input is encoded.

        if validation_data is not None:
            if progressive:
                validation_data = validation_data.map(map_progressive)
            else:
                validation_data = validation_data.map(map_non_progressive)

    return histories


def model_fit(model, train_data, validation_data=None, progressive=False, **kwargs):
    """Do a standard model fit."""
    return model.fit(train_data, validation_data=validation_data, **kwargs)


def model_do_not_fit(model, train_data, validation_data=None, progressive=False, **kwargs):
    """Do a standard model fit."""
    kwargs["epochs"] = 0  # We do not train.
    return model.fit(train_data, validation_data=validation_data, **kwargs)


####################
# Loss functions
####################

def layer_predict_backtrack(h_t, h_t_prev, layer, pinv_dict=None):
    """Recreate priors and predictions for a single layer using output and layer weights."""
    if isinstance(layer, tf.keras.layers.SimpleRNN):
        inv_activation = get_inv_activation(layer.cell.activation)

        input_kernel = layer.cell.kernel
        recurrent_kernel = layer.cell.recurrent_kernel
        bias = layer.cell.bias

        drive = inv_activation(h_t)
        drive = tf.debugging.check_numerics(drive, message="drive (after activation) blew up")
        unbiased_drive = drive - bias[None, :]

        input_drive = unbiased_drive - tf.einsum("bth,hk->btk", h_t_prev, recurrent_kernel)

        if pinv_dict == None:
            inv_kernel = tf.linalg.inv(tf.transpose(layer.cell.kernel) @ layer.cell.kernel) @ tf.transpose(layer.cell.kernel)
        else:
            inv_kernel = pinv_dict[layer.name]

        input_prev = tf.einsum("bth,hk->btk", input_drive, inv_kernel)

    elif isinstance(layer, tf.keras.layers.Reshape):
        input_prev = tf.reshape(h_t, (-1,) + layer.input_shape[1:])
    elif isinstance(layer, tf.keras.layers.Activation):
        input_prev = get_inv_activation(layer.activation)(h_t)
    else:
        input_prev = h_t
    return input_prev


def model_reconstruct_prediction(model, state, pinv_dict=None):
    """Reconstruct the input data from a model and its output data."""
    states = []
    for layer in model.layers[::-1]:
        h_t_prev = tf.concat([tf.zeros_like(state)[:, :1], state[:, :-1]], axis=1)
        state = layer_predict_backtrack(state, h_t_prev, layer, pinv_dict)
        states.append(state)
    return states[::-1]


class PredPriorLoss(tf.keras.losses.Loss):
    """Optimizes the latent kernel to predict the input. Only works for SimpleRNN."""

    def __init__(self, model, **kwargs):
        super(PredPriorLoss, self).__init__(**kwargs)
        self.model = model

    def call(self, y_true, y_pred):
        # Compute pinv(kernel) only once per layer.
        pinv_dict = {layer.name: tf.linalg.inv(tf.transpose(layer.cell.kernel) @ layer.cell.kernel) @ tf.transpose(layer.cell.kernel)
                     for layer in self.model.layers if isinstance(layer, tf.keras.layers.SimpleRNN)}

        # Project output state.
        y_pred = y_pred @ self.model.layers[-1].cell.recurrent_kernel

        # Reconstruct input.
        prediction_at_input = model_reconstruct_prediction(self.model, y_pred, pinv_dict)[0]

        # Compute loss.
        pred_loss = keras.losses.mean_squared_error(-y_true[:, 1:], prediction_at_input[:, :-1])

        return pred_loss

    def get_config(self):
        # Include the necessary information to recreate the object.
        base_config = super(PredPriorLoss, self).get_config()
        config = {"model": self.model}
        return dict(list(base_config.items()) + list(config.items()))


class FFLossWithThreshold():
    """Forward Forward-based contrastive loss according to Hinton (2022)."""

    def __init__(self, threshold, average=False, **kwargs):
        """Return an FF loss function with a predefined threshold."""
        self.threshold = threshold
        self.average = average

    def __call__(self, y_true, y_pred):
        """Return the FF loss following the standard Keras loss structure."""
        # Cast y_true and y_pred into the correct data types.
        y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)

        # Square the model output, then sum across all output neurons.
        g = keras.ops.power(y_pred, 2.)
        g = keras.ops.sum(g, axis=-1)

        # Subtract the threshold.
        g = g - self.threshold

        # Convert activation sums into a 0-1 space representing the probability of a negative sample.
        p_negative = tf.nn.sigmoid(g)

        # Calculate Binary Crossentropy from the predicted and actual probability of negativity.
        loss = tf.keras.losses.BinaryCrossentropy()(y_true, p_negative)

        # Average the loss if necessary.
        if self.average:
            loss = tf.reduce_mean(loss)
        return loss

    def get_config(self):
        # Include the necessary information to recreate the object.
        config = {"threshold": self.threshold, "average": self.average}
        return config


####################
# Callbacks
####################

class SafeModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath_template):
        super().__init__()
        self.filepath_template = filepath_template

    def on_epoch_end(self, epoch, logs=None):
        filepath = self.filepath_template.format(epoch=epoch + 1)

        # Ensure directory exists.
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Clone the model.
        model_clone = tf.keras.models.clone_model(self.model)
        model_clone.set_weights(self.model.get_weights())

        # Compile it with no loss to avoid saving train config that causes recursion.
        model_clone.compile(optimizer=self.model.optimizer, loss=None)

        # Save it.
        model_clone.save(filepath)


def rebind_callbacks_for_layer_fit(callbacks, full_model):
    """Reconnect all callbacks to the main model rather than the layer models."""
    patched = []
    for cb in callbacks:
        if hasattr(cb, "set_model"):
            cb = copy.copy(cb)  # Do not mutate the original callback outside.
            cb.set_model(full_model)
        patched.append(cb)
    return patched