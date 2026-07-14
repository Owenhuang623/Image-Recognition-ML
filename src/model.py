"""
Stage 1 — The convolutional neural network itself.

This is the small 3-block CNN we sketched out. Read it top to bottom like a
pipeline: an image enters at the top, gets transformed block by block, and
exits the bottom as 10 probabilities (one per class).

The signature shape to notice: spatial size SHRINKS (28 -> 14 -> 7 -> 3) while
channel depth GROWS (1 -> 32 -> 64 -> 64). We're gradually trading "where things
are" for "what things are" — the hallmark funnel of every CNN.

(Input is 28x28x1 for grayscale Fashion-MNIST. To switch back to CIFAR-10 later,
change this one line to (32, 32, 3) — the rest of the network is unchanged.)
"""

from tensorflow.keras import layers, models


def build_model():
    """Build and return the (uncompiled) CNN."""
    model = models.Sequential([
        layers.Input((28, 28, 1)),   # 28x28 grayscale: 1 channel

        # --- Block 1 --------------------------------------------------------
        # 32 filters, each a 3x3 grid of weights that START RANDOM and get
        # tuned by training. What emerges is edge/color detectors — we never
        # hand-design them.
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        # MaxPooling keeps the strongest response in each 2x2 window, halving
        # the width and height. This is our dimensionality reduction.
        layers.MaxPooling2D((2, 2)),                        # 28x28 -> 14x14

        # --- Block 2 --------------------------------------------------------
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),                        # 14x14 -> 7x7

        # --- Block 3 --------------------------------------------------------
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),                        # 7x7 -> 3x3 (odd sizes round down)

        # --- Classifier head ------------------------------------------------
        # Convolutions found spatial features; now flatten that 3x3x64 grid
        # into a plain vector and use ordinary dense layers to decide.
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        # 10 outputs + softmax = your "10-branch softmax". Softmax turns raw
        # scores into 10 probabilities that sum to 1.
        layers.Dense(10, activation="softmax"),
    ])
    return model


if __name__ == "__main__":
    # Printing the summary is a great way to SEE the funnel shape and confirm
    # the output-shape arithmetic in the comments above is actually correct.
    model = build_model()
    model.summary()
