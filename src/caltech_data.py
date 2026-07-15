"""
Stage 2/3 — Loading real photos of household items from folders.

Unlike Fashion-MNIST (a tidy array bundled with Keras), this is how you load
REAL image data: a directory with one sub-folder per class, each holding that
class's photos. Keras reads the folder names as the labels automatically. This
is the exact same setup you'd use for your OWN photos later.

    data/household10/
        watch/      image_0001.jpg ...
        laptop/     image_0001.jpg ...
        ...

The photos are all different sizes, so the key job here is making them uniform:
every image gets resized to 160x160x3 so the network sees a consistent shape.
"""

import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

DATA_DIR = "data/household10"
IMG_SIZE = (224, 224)   # MobileNetV2's native ImageNet resolution (more detail)
BATCH_SIZE = 32
SEED = 123              # fixing the seed makes the train/val split reproducible


def load_datasets():
    """Return (train_ds, val_ds, class_names).

    train_ds / val_ds are tf.data.Dataset objects that yield batches of
    (images, labels), where images are float tensors shaped (batch, 160, 160, 3)
    with pixel values still in [0, 255] (the model itself will do the final
    MobileNet-specific scaling).
    """
    # image_dataset_from_directory does a LOT for us in one call: reads every
    # file, decodes the JPEG, resizes to IMG_SIZE, batches, and infers integer
    # labels from the sub-folder names. validation_split + subset carve off a
    # held-out 20% -- using the SAME seed for both so no image lands in both.
    train_ds = image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",          # integer labels -> sparse crossentropy later
    )
    val_ds = image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    # class_names comes from the folder names, sorted alphabetically. We grab it
    # BEFORE the performance tweaks below, because those return a new dataset
    # object that no longer carries the attribute.
    class_names = train_ds.class_names

    # Performance: cache decoded images in RAM after the first epoch, and
    # prefetch the next batch while the current one trains. Pure speed; no
    # effect on the math. AUTOTUNE lets TF pick the buffer sizes.
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    return train_ds, val_ds, class_names


if __name__ == "__main__":
    train_ds, val_ds, class_names = load_datasets()
    print("\nClasses:", class_names)
    # Peek at one batch to confirm shapes and the pixel range.
    images, labels = next(iter(train_ds))
    print("One batch of images:", images.shape, images.dtype)
    print("Pixel range: min =", float(tf.reduce_min(images)),
          " max =", float(tf.reduce_max(images)))
    print("Labels in batch:", labels.numpy())
