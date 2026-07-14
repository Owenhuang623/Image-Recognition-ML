"""
Stage 3 — A transfer-learning model for the 10 household classes.

Instead of training a CNN from scratch, we stand on the shoulders of MobileNetV2,
which was already trained on 1.28M ImageNet photos. We KEEP its convolutional
backbone (a world-class feature extractor) FROZEN, and train only a small new
head on top. That's why this is fast even on a CPU: ~2.2M backbone params never
change; we only learn the ~13k params in the head.

Read the model top to bottom as a pipeline for one 160x160 photo:

    input (160x160x3, pixels 0-255)
      -> data augmentation   (random flips/rotations -- fights overfitting)
      -> preprocess_input    (scale pixels to [-1, 1], what MobileNet expects)
      -> MobileNetV2 backbone (FROZEN) -> 5x5x1280 feature maps
      -> GlobalAveragePooling -> 1280-vector (one number per feature map)
      -> Dropout             (randomly zero 20% -- more overfitting defense)
      -> Dense(10, softmax)  -> 10 class probabilities
"""

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SHAPE = (160, 160, 3)


def build_model(num_classes=10, hidden_units=None):
    """Build the transfer model.

    hidden_units: if None (default), the head is a single Dense(softmax) layer --
    a linear classifier over MobileNet's features. If an int (e.g. 128), we
    insert one hidden Dense(relu) layer first, making the head a deeper 2-layer
    fully-connected network. This is the knob for the "does a hidden layer help?"
    experiment.
    """
    # --- The pretrained backbone -------------------------------------------
    # include_top=False drops MobileNet's original 1000-class ImageNet head --
    # we only want the feature extractor. weights="imagenet" downloads the
    # pretrained weights (~9MB, from Google's fast CDN, one time).
    base_model = MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    # FREEZE it: mark all ~2.2M params non-trainable so training leaves the
    # learned features untouched. This is the heart of transfer learning.
    base_model.trainable = False

    # --- Data augmentation -------------------------------------------------
    # With only ~500 training images, the model could easily memorize them.
    # These layers randomly flip/rotate each image every epoch, so the model
    # effectively sees "new" variations -- a cheap, strong defense against the
    # overfitting we watched happen in Stage 1. (Active only during training;
    # Keras automatically turns them off at inference.)
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ], name="data_augmentation")

    # --- Assemble with the functional API ----------------------------------
    inputs = layers.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = preprocess_input(x)                 # 0-255  ->  [-1, 1]
    # training=False keeps the backbone's BatchNorm layers in inference mode --
    # important when the backbone is frozen, or its running stats would drift.
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)  # 5x5x1280 -> 1280, cheaply
    x = layers.Dropout(0.2)(x)
    # Optional hidden layer: turns the head from a linear classifier into a
    # 2-layer fully-connected net. More capacity, but more params to overfit.
    if hidden_units:
        x = layers.Dense(hidden_units, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
    # The telling line in the summary: "Trainable params" should be tiny
    # (~13k) versus "Non-trainable params" (~2.2M, the frozen backbone).
