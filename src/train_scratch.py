"""
The experiment: train the from-scratch CNN on the SAME data as the transfer
model, and compare — to see for ourselves why transfer learning matters.

Run:  ./venv/bin/python src/train_scratch.py

NO LEAKAGE: we call the same load_datasets() (seed=123), so this model trains on
the identical 510 images and is evaluated on the identical 127 held-out images
that the transfer model used. Neither model ever trains on a validation image.

We give from-scratch every fair advantage: same augmentation, and MORE epochs
(40 vs the transfer model's 10), since a randomly-initialized network learns far
more slowly than one standing on pretrained features.
"""

import os
import matplotlib.pyplot as plt

from caltech_data import load_datasets
from scratch_model import build_model

EPOCHS = 40


def main():
    # Identical split to the transfer experiment -> fair, leak-free comparison.
    train_ds, val_ds, class_names = load_datasets()
    print("\nClasses:", class_names)

    model = build_model(num_classes=len(class_names))
    model.summary()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # Report BOTH the final and the BEST validation accuracy over all epochs --
    # the "best" is the most generous number we can honestly give from-scratch.
    val_acc_curve = history.history["val_accuracy"]
    final_val = val_acc_curve[-1]
    best_val = max(val_acc_curve)
    print(f"\nFrom-scratch final val accuracy: {final_val:.3f}")
    print(f"From-scratch BEST  val accuracy: {best_val:.3f}  (epoch {val_acc_curve.index(best_val)+1})")
    print("Compare to transfer learning:    0.976")

    plot_history(history)


def plot_history(history, save_path="outputs/scratch_curves.png"):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    plt.figure(figsize=(6, 4))
    plt.plot(acc, label="training accuracy")
    plt.plot(val_acc, label="validation accuracy")
    plt.axhline(0.976, color="green", linestyle="--", label="transfer learning (0.976)")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("From-scratch CNN on 10 household items (same data)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    print(f"Saved learning curves to {save_path}")


if __name__ == "__main__":
    main()
