import tensorflow as tf
import numpy as np
import os
import random
from pathlib import Path

AUTOTUNE = tf.data.AUTOTUNE

label_dict = {
    ("barn", "church"): 0.75,
    ("barn", "conference_room"): 0.25,
    ("barn", "castle"): 0.,
    ("barn", "forest"): 0.,
    ("beach", "church"): 0.75,
    ("beach", "conference_room"): 0.25,
    ("beach", "castle"): 0.0,
    ("beach", "forest"): 0.0,
    ("library", "church"): 0.25,
    ("library", "conference_room"): 0.75,
    ("library", "castle"): 0.,
    ("library", "forest"): 0.,
    ("restaurant", "church"): 0.25,
    ("restaurant", "conference_room"): 0.75,
    ("restaurant", "castle"): 0.,
    ("restaurant", "forest"): 0.,
    ("cave", "church"): 0.,
    ("cave", "conference_room"): 0.,
    ("cave", "castle"): 0.5,
    ("cave", "forest"): 0.5,
}

supervised_dict = {
    'church': 0,
    'barn': 1,
    'beach': 2,
    'castle': 3,
    'cave': 4,
    'conference_room': 5,
    'forest': 6,
    'library': 7,
    'restaurant': 8
}

# Dataset and preprocessing functions
augmenter = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomBrightness(0.1, value_range=(0, 1)),
])

def paired_augment(x, y):
    augmented_lead = augmenter(x[:, 0])
    augmented_trail = augmenter(x[:, 1])
    x = tf.stack([augmented_lead, augmented_trail], axis=1)
    return x, y

def get_image_paths_by_class(base_dir):
    """Scans folders and returns dict[class_name] = list of image file paths."""
    class_dirs = {p.name: list(p.glob("*.jpg")) + list(p.glob("*.png")) for p in Path(base_dir).iterdir() if p.is_dir()}
    return class_dirs

def parse_image(filename, image_size=(224, 224)):
    """Reads and preprocesses image."""
    image = tf.io.read_file(filename)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0  # normalize to [0, 1]
    return image

def build_pair_generator(leading_paths, trailing_paths, transition_dict, image_size):
    """Yields (lead_img_path, trail_img_path, lead_class, trail_class)."""
    leading_classes = list(leading_paths.keys())
    trailing_classes = list(trailing_paths.keys())

    # Precompute probability distributions and lookup for transition probs
    trailing_probs_by_leading = {}
    prob_lookup = {}

    for lead_class in leading_classes:
        trailing_targets = []
        trailing_probs = []
        for trail_class in trailing_classes:
            p = transition_dict.get((lead_class, trail_class), 0.0)
            if p > 0.0:
                trailing_targets.append(trail_class)
                trailing_probs.append(p)
                prob_lookup[(lead_class, trail_class)] = p
        total = sum(trailing_probs)
        trailing_probs = [p / total for p in trailing_probs]
        trailing_probs_by_leading[lead_class] = (trailing_targets, trailing_probs)

    def generator():
        while True:
            lead_class = random.choice(leading_classes)
            lead_img_path = random.choice(leading_paths[lead_class])
            trail_classes, probs = trailing_probs_by_leading[lead_class]
            trail_class = np.random.choice(trail_classes, p=probs)
            trail_img_path = random.choice(trailing_paths[trail_class])
            yield str(lead_img_path), str(trail_img_path), str(lead_class), str(trail_class)

    return generator


def create_paired_dataset(leading_dir, trailing_dir, transition_dict, batch_size=32,
                         image_size=(224, 224), shuffle_buffer_size=1000):
    """Creates paired dataset with image loading and batching."""
    leading_paths = get_image_paths_by_class(leading_dir)
    trailing_paths = get_image_paths_by_class(trailing_dir)
    generator = build_pair_generator(leading_paths, trailing_paths, transition_dict, image_size)

    output_signature = (
        tf.TensorSpec(shape=(), dtype=tf.string),  # lead path
        tf.TensorSpec(shape=(), dtype=tf.string),  # trail path
        tf.TensorSpec(shape=(), dtype=tf.string),  # lead class
        tf.TensorSpec(shape=(), dtype=tf.string),  # trail class
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    def load_image(lead_path, trail_path, lead_class, trail_class):
        lead_img = parse_image(lead_path, image_size)
        trail_img = parse_image(trail_path, image_size)
        x = tf.stack([lead_img, trail_img], axis=0)  # (2, H, W, C)
        y = tf.stack([lead_class, trail_class], axis=0)
        return x, y

    dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


def supervised_labels(x, y, supervised_dict=supervised_dict):
    """Maps string labels to integer IDs via dictionary lookup."""
    keys = tf.constant(list(supervised_dict.keys()), dtype=tf.string)
    values = tf.constant(list(supervised_dict.values()), dtype=tf.int32)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values), default_value=-1)
    y_mapped = table.lookup(y)
    return x, y_mapped


def transition_likelihoods(x, y, label_dict=label_dict, separator=", "):
    """Lookup float values for tuple keys given y of shape (batch, 2)."""
    combined_keys = [f"{k[0]}{separator}{k[1]}" for k in label_dict.keys()]
    combined_values = list(label_dict.values())
    keys_tensor = tf.constant(combined_keys, dtype=tf.string)
    values_tensor = tf.constant(combined_values, dtype=tf.float32)
    initializer = tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor)
    lookup_table = tf.lookup.StaticHashTable(initializer, default_value=-1.0)
    combined_y = tf.strings.join([y[:, 0], y[:, 1]], separator=separator)
    y_mapped = lookup_table.lookup(combined_y)
    y_mapped = tf.stack([y_mapped, y_mapped], axis=1)
    return x, y_mapped


def probabilistic_binary_labels(x, y):
    """Converts float probs to binary labels via Bernoulli sampling."""
    y_bin = tf.cast(tf.random.uniform(tf.shape(y)) < y, tf.float32)
    return x, y_bin


def sharpened_probabilistic_binary_labels(y, x):
    """Clamps to [0.25, 0.75], maps to [0, 1], samples binary labels."""
    x = tf.clip_by_value(x, 0.25, 0.75)
    probs = (x - 0.25) * 2.0
    return y, tf.cast(tf.random.uniform(tf.shape(x)) < probs, tf.float32)


def autoregressive_target(x, y):
    """Returns future data as target; time shift applied in loss."""
    return x, x


def repeat_pair_images(x, y, repeats=2):
    """Repeats pair images along axis 1."""
    x = tf.repeat(x, repeats, axis=1)
    y = tf.repeat(y, repeats, axis=1)
    return x, y

def flatten(x, y=None):
    """Flattens spatial dimensions of x."""
    shape = tf.shape(x)
    x = tf.reshape(x, [shape[0], shape[1], -1])
    return x, y

def transition_target(x, y):
    """Computes transition targets: lookup likelihoods and sample labels."""
    x, y = transition_likelihoods(x, y)
    x, y = sharpened_probabilistic_binary_labels(x, y)
    return x, y

def img_preproc(x, y):
    """Augments and flattens paired images."""
    x, y = paired_augment(x, y)
    x, y = flatten(x, y)
    return x, y