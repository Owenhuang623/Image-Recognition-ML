"""
Stage 1 — Look at the data before doing anything else.

Rule of thumb in machine learning: never train on data you haven't eyeballed.
Plotting a handful of images with their labels catches a whole class of bugs
(wrong color channels, mislabeled data, bad preprocessing) in about 5 seconds,
before you waste time training on garbage.

Run:  ./venv/bin/python src/visualize.py
It saves a grid of sample images to  outputs/sample_images.png
"""

import os
import matplotlib.pyplot as plt

from data import load_data, CLASS_NAMES


def show_samples(n=25, save_path="outputs/sample_images.png"):
    (x_train, y_train), _ = load_data()

    # Arrange the first n images in a square-ish grid.
    cols = 5
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols * 1.6, rows * 1.6))

    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        # x[i] is shape (28, 28, 1); squeeze() drops the size-1 channel axis to
        # (28, 28) so imshow can render it. cmap="gray" because it's grayscale.
        plt.imshow(x_train[i].squeeze(), cmap="gray")
        plt.title(CLASS_NAMES[y_train[i]], fontsize=8)
        plt.axis("off")                  # hide tick marks — we just want the picture

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    print(f"Saved {n} sample images to {save_path}")


if __name__ == "__main__":
    show_samples()
