"""
Stage 1 — Loading and preprocessing the Fashion-MNIST dataset.

Fashion-MNIST is 70,000 grayscale images (28x28 pixels) of clothing items,
each belonging to one of 10 classes. It's a drop-in replacement for the classic
MNIST digits, but a bit more interesting. It ships with Keras and downloads from
Google's fast CDN in about a second.

(We originally planned CIFAR-10, but its host was throttling us badly. The only
real differences: Fashion-MNIST is 28x28 GRAYSCALE, so 1 color channel instead
of 3. When CIFAR finishes downloading in the background we can swap back by
changing the dataset import and the channel count -- see the notes below.)

This module's only job is to hand back clean, model-ready arrays.
"""

import numpy as np
from tensorflow.keras.datasets import fashion_mnist

# The 10 class names, in label order (0-9). Lets us turn a label like 3 back
# into the human-readable "Dress".
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def load_data():
    """Load Fashion-MNIST and return preprocessed train/test splits.

    Returns:
        (x_train, y_train), (x_test, y_test)
        - x_* : float32 arrays of shape (N, 28, 28, 1), pixels scaled to [0, 1]
        - y_* : int arrays of shape (N,) with values 0-9
    """
    # Keras hands us two splits already separated. We train on one and only ever
    # judge final performance on the other -- never let the model "see" the test
    # set during training, or we'd be fooling ourselves.
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # --- Preprocessing step 1: normalize the pixels --------------------------
    # Raw pixels are integers 0-255. Neural nets train far more smoothly on
    # small, centered numbers, so we rescale to [0, 1]. One line, big effect.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # --- Preprocessing step 2: add a channel dimension -----------------------
    # Fashion-MNIST images arrive shaped (N, 28, 28) -- no channel axis, because
    # they're grayscale. But Conv2D always expects (height, width, channels), so
    # we add a trailing axis of size 1: (N, 28, 28) -> (N, 28, 28, 1).
    # (A color dataset like CIFAR already has 3 there, so it skips this step.)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # --- Preprocessing step 3: tidy the label shape --------------------------
    # Keep labels as plain integers 0-9 (not one-hot); we'll pair them with the
    # "sparse" cross-entropy loss later. Fashion-MNIST labels are already (N,),
    # but flatten() is a harmless no-op that keeps this robust.
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    return (x_train, y_train), (x_test, y_test)


# Running this file directly is a quick sanity check: does the data load, and
# are the shapes what we expect? Always confirm shapes before building a model.
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()
    print("Training images:", x_train.shape, x_train.dtype)
    print("Training labels:", y_train.shape, y_train.dtype)
    print("Test images:    ", x_test.shape, x_test.dtype)
    print("Pixel range:     min =", x_train.min(), " max =", x_train.max())
    print("First 10 labels:", y_train[:10],
          "->", [CLASS_NAMES[i] for i in y_train[:10]])
