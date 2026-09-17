# imports
import joblib
import os.path as op
import tensorflow as tf
import matplotlib.pyplot as plt

tf.config.experimental_run_functions_eagerly(True)

from src.load_dataset import (create_paired_dataset, repeat_pair_images, probabilistic_binary_labels, label_dict,
    supervised_dict, supervised_labels, transition_likelihoods, flatten, transition_target, img_preproc,
    autoregressive_target, sharpened_probabilistic_binary_labels, paired_augment)
from src.training import (PredPriorLoss, FFLossWithThreshold, model_fit, model_fit_layers, model_do_not_fit,
    SafeModelCheckpoint)
from src.visualization import plot_training_history

# Allow GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
print(f"Available GPUs: {gpus}")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Paths & Hyperparameters
data_dir = "/scratch/guetlid95/datasets/mcdermott_2024/stimuli_2"
results_dir = "/scratch/guetlid95/datasets/mcdermott_2024/models_isi_fi"

train_leading_dir = op.join(data_dir, "train", "leading")
train_trailing_dir = op.join(data_dir, "train", "trailing")
test_leading_dir = op.join(data_dir, "test", "leading")
test_trailing_dir = op.join(data_dir, "test", "trailing")

batch_size = 128
n_steps = 15
image_size = (40, 40)


# Dataset creation
train_dataset = create_paired_dataset(
    leading_dir=train_leading_dir, trailing_dir=train_trailing_dir,
    transition_dict=label_dict, batch_size=batch_size, image_size=image_size,
    shuffle_buffer_size=250)

test_dataset = create_paired_dataset(
    leading_dir=test_leading_dir, trailing_dir=test_trailing_dir,
    transition_dict=label_dict, batch_size=batch_size, image_size=image_size,
    shuffle_buffer_size=250)


def repeat_images(x, y, n_steps=n_steps, inter_stim_interval=True):
    """Repeat and optionally insert ISI gaps between paired images."""
    if inter_stim_interval:
        x, y = repeat_pair_images(x, y, n_steps // 2)
        flip = n_steps // 15
        isi_shape = tf.concat([[batch_size, 8 * flip], tf.shape(x)[2:]], axis=0)
        isi = tf.fill(isi_shape, tf.constant(0.5, x.dtype))
        x = tf.concat([isi[:, :flip], x[:, :flip], isi, x[:, -flip:], isi[:, :4*flip]], axis=1)
        y_repeated = tf.repeat(y[:, :flip], repeats=10, axis=1)
        y = tf.concat([y_repeated, y[:, -5*flip:]], axis=1)
    else:
        reps = n_steps // 2 if (n_steps % 2 == 0) else (n_steps + 1) // 2
        x, y = repeat_pair_images(x, y, reps)
        if not (n_steps % 2 == 0):
            x, y = x[:, :-1], y[:, :-1]
    return x, y


def autoregressive_target(x, y, inter_stim_interval=True):
    """Return future data as target; time shift applied in loss."""
    if inter_stim_interval:
        return x, tf.stack([x[:, -1], x[:, -1]], axis=1)
    else:
        return x, x


def shuffled_sparse_ce_from_logits(y_true, y_pred):
    """Sparse categorical crossentropy with shuffled labels."""
    y_true = tf.random.shuffle(y_true)
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred)


target_preproc = {
    "Predictive global": autoregressive_target,
    "Predictive local": autoregressive_target,
    "Contrastive global": transition_target,
    "Contrastive local": transition_target,
    "Supervised": supervised_labels,
    "Supervised shuffled": supervised_labels,
    "Untrained": supervised_labels,
}

training_functions = {
    "Predictive global": model_fit,
    "Predictive local": model_fit_layers,
    "Contrastive global": model_fit,
    "Contrastive local": model_fit_layers,
    "Supervised": model_fit,
    "Supervised shuffled": model_fit,
    "Untrained": model_do_not_fit,
}


def create_loss(model, condition):
    """Create loss function based on training condition."""
    if condition in ("Contrastive global", "Contrastive local"):
        return FFLossWithThreshold(10., average=False)
    elif condition in ("Predictive global", "Predictive local"):
        return PredPriorLoss(model)
    elif condition == "Supervised":
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    elif condition == "Untrained":
        return None
    elif condition == "Supervised shuffled":
        return shuffled_sparse_ce_from_logits


def create_model(condition):
    """Create model architecture based on condition."""
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(256, activation="leaky_relu", return_sequences=True, unroll=True),
        tf.keras.layers.SimpleRNN(128, activation="leaky_relu", return_sequences=True, unroll=True),])
    if condition in ("Supervised", "Supervised shuffled"):
        model.add(tf.keras.layers.Dense(9, activation="linear"))
    return model


conditions = list(training_functions.keys())


# Run and save models
for condition in [i for i in target_preproc.keys()]:
    print(f"Now training {condition}")

    model = create_model(condition)
    model.build([None, n_steps, image_size[0]*image_size[1]*3])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), metrics=["accuracy"],
                  loss=create_loss(model, condition))
    model.summary()

    save_callback = SafeModelCheckpoint(op.join(results_dir, condition, condition + "_{epoch}.keras"))
    history = training_functions[condition](
        model,
        train_dataset.map(img_preproc).map(target_preproc[condition]).map(repeat_images).prefetch(tf.data.AUTOTUNE),
        validation_data=test_dataset.map(img_preproc).map(target_preproc[condition]).map(repeat_images).prefetch(tf.data.AUTOTUNE),
        steps_per_epoch=3196//batch_size, validation_steps=800//batch_size, epochs=100,
        callbacks=[save_callback])

    history = [h.history for h in history] if isinstance(history, list) else history.history
    joblib.dump(history, op.join(results_dir, f"{condition}.pkl"))

    model.compile(optimizer=model.optimizer, loss=None)
    model.save(op.join(results_dir, f"{condition}_final.keras"))
    joblib.dump(history, op.join(results_dir, f"{condition}.pkl"))