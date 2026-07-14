"""
A from-scratch CNN for the 10 household classes — the control group.

This is the SAME kind of network as Stage 1, scaled up for 160x160 color input.
Crucially, NOTHING here is pretrained: every one of these ~240k parameters
starts random and must be learned from our ~500 images alone. That's the whole
point of the experiment — can it learn to "see" real photos from scratch?

To keep the comparison fair, it uses the same regularization as the transfer
model (data augmentation + dropout) and the same input size, so the ONLY
difference vs. transfer_model.py is: random conv stack (trainable) instead of a
pretrained MobileNet backbone (frozen).
"""

from tensorflow.keras import layers, models

IMG_SHAPE = (160, 160, 3)


def build_model(num_classes=10):
    # Same augmentation as the transfer model — fair fight, and it helps the
    # from-scratch model resist overfitting as much as possible.
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ], name="data_augmentation")

    inputs = layers.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)          # normalize pixels to [0, 1]

    # Four conv blocks: the funnel again (160 -> 80 -> 40 -> 20 -> 10), channels
    # growing 32 -> 64 -> 128 -> 128. These filters start RANDOM and must learn
    # edges/textures/parts from our tiny dataset — the hard "learn to see" job.
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)


if __name__ == "__main__":
    build_model().summary()
