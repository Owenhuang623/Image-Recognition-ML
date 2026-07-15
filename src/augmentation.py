"""
The data-augmentation pipeline — the heart of this experiment.

Data augmentation manufactures realistic *variations* of our training images on
the fly, so the model sees far more variety than our ~500 real photos actually
contain. This fights overfitting and, more importantly for us, teaches the model
to tolerate the tilt / framing / lighting differences that REAL phone photos have
but clean Caltech photos don't.

THE GOLDEN RULE: every augmentation must produce an image that could plausibly
be a real photo of the SAME class. If it wouldn't, it teaches the model lies.

These layers are special: they run ONLY during training (model.fit). At
inference/validation time Keras automatically turns them into no-ops, so we
always evaluate and predict on clean, un-augmented images.
"""

from tensorflow.keras import layers, models


def build_augmentation():
    """Return a Sequential of label-preserving, realistic augmentations."""
    return models.Sequential(
        [
            # Mirror left<->right. A mirrored cup is still a cup. We do NOT use
            # "vertical" (upside-down) flips: you never photograph a laptop
            # upside down, so that would be an unrealistic example.
            layers.RandomFlip("horizontal"),

            # Tilt by up to ~+/-36deg (0.1 of a full turn). Real photos are
            # rarely perfectly level. Kept modest on purpose -- a huge rotation
            # would look unnatural for household objects.
            layers.RandomRotation(0.1),

            # Zoom in/out by up to 20%. Simulates the object being nearer/farther
            # from the camera (scale variation).
            layers.RandomZoom(0.2),

            # Shift the object up to 10% horizontally/vertically, so it is NOT
            # always dead-centre -- exactly the off-centre framing real photos
            # have. (This is your "different positions" idea.)
            layers.RandomTranslation(0.1, 0.1),

            # Vary contrast +/-20%: some photos are flat, some punchy.
            layers.RandomContrast(0.2),

            # Vary brightness +/-20%: simulates different lighting. Note the
            # value_range=(0,255): augmentation runs BEFORE the model's MobileNet
            # preprocessing, while pixels are still 0-255.
            layers.RandomBrightness(0.2, value_range=(0, 255)),

            # --- Deliberately EXCLUDED (would HURT this task) ------------------
            # layers.RandomFlip("vertical")   -> upside-down objects: unrealistic
            # color inversion / negative       -> destroys real color cues
            # pixel/patch scrambling           -> destroys the object's structure
            # (A realistic cousin of "scrambling" is random-erasing/cutout, which
            #  simulates occlusion -- that one IS valid, but we keep this set
            #  simple and clearly realistic.)
        ],
        name="data_augmentation",
    )
