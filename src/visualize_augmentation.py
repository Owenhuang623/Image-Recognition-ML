"""
SEE what augmentation actually does — run this to build intuition.

It takes ONE real training image and applies the augmentation pipeline several
times, so you can watch the same chair (or whatever) show up tilted, shifted,
zoomed, brighter, mirrored... every version still clearly the same object. That
"still clearly a chair" is the golden rule made visible.

Run:  ./venv/bin/python src/visualize_augmentation.py
Saves a grid to  outputs/augmentation_demo.png
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from augmentation import build_augmentation

IMG_SIZE = (224, 224)


def main(save_path="outputs/augmentation_demo.png"):
    # Grab one real image (first chair we find) to demonstrate on.
    sample = sorted(glob.glob("data/household10/chair/*.jpg"))[0]
    img = tf.keras.utils.load_img(sample, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)          # (224,224,3), 0-255
    batch = np.expand_dims(arr, 0)

    augment = build_augmentation()

    plt.figure(figsize=(9, 9))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        if i == 0:
            # top-left: the ORIGINAL, for comparison
            shown = arr.astype("uint8")
            plt.title("original", fontsize=9)
        else:
            # training=True forces the augmentation layers to actually fire
            # (at inference they'd be no-ops). Each call gives a new random mix.
            out = augment(batch, training=True)[0].numpy()
            shown = np.clip(out, 0, 255).astype("uint8")
            plt.title(f"augmented #{i}", fontsize=9)
        plt.imshow(shown)
        plt.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    print(f"Saved augmentation demo to {save_path}")


if __name__ == "__main__":
    main()
