"""
Stage 3 — Train the transfer-learning head on the 10 household classes.

Run:  ./venv/bin/python src/train_transfer.py

Because only the small head trains (the backbone is frozen), this is quick even
on a CPU. Expect high accuracy (often 90%+) from very little data -- that's the
payoff of standing on a pretrained backbone.
"""

import os
import matplotlib.pyplot as plt

from caltech_data import load_datasets
from transfer_model import build_model

EPOCHS = 10


def main():
    # 1. Data ----------------------------------------------------------------
    train_ds, val_ds, class_names = load_datasets()
    print("\nClasses:", class_names)

    # 2. Model ---------------------------------------------------------------
    model = build_model(num_classes=len(class_names))
    model.summary()

    # 3. Compile -------------------------------------------------------------
    # Same recipe as Stage 1: integer labels -> sparse crossentropy, adam,
    # report accuracy.
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 4. Train (only the head learns; the backbone is frozen) -----------------
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # 5. Report final validation accuracy ------------------------------------
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\nFinal validation accuracy: {val_acc:.3f}")

    # 6. Plot learning curves ------------------------------------------------
    plot_history(history)

    # 7. Save the trained model (self-contained: it includes the preprocessing,
    #    so later we can feed it a raw photo directly).
    os.makedirs("outputs", exist_ok=True)
    model.save("outputs/household10_transfer.keras")
    print("Saved trained model to outputs/household10_transfer.keras")


def plot_history(history, save_path="outputs/transfer_curves.png"):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    plt.figure(figsize=(6, 4))
    plt.plot(acc, label="training accuracy")
    plt.plot(val_acc, label="validation accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Transfer-learning curves (10 household items)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    print(f"Saved learning curves to {save_path}")


if __name__ == "__main__":
    main()
