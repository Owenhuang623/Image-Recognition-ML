"""
Experiment: does adding a hidden layer to the head help?

We train the SAME transfer model but with a hidden Dense(128, relu) layer added
to the head, on the SAME leak-free split, and compare to the single-layer head's
97.6%. Prediction: on only ~500 images, the extra capacity likely helps little
or even hurts (more params to overfit).

Run:  ./venv/bin/python src/train_deep_head.py
"""

import matplotlib.pyplot as plt

from caltech_data import load_datasets
from transfer_model import build_model

EPOCHS = 10
HIDDEN_UNITS = 128


def main():
    train_ds, val_ds, class_names = load_datasets()

    model = build_model(num_classes=len(class_names), hidden_units=HIDDEN_UNITS)
    trainable = sum(int(w.shape.num_elements()) for w in model.trainable_weights)
    print(f"\nHead now has a hidden Dense({HIDDEN_UNITS}); trainable params = {trainable:,}")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    val_curve = history.history["val_accuracy"]
    print(f"\nDeep-head final val accuracy: {val_curve[-1]:.3f}")
    print(f"Deep-head BEST  val accuracy: {max(val_curve):.3f}")
    print("Single-layer head (baseline): 0.976")

    plt.figure(figsize=(6, 4))
    plt.plot(history.history["accuracy"], label="training accuracy")
    plt.plot(val_curve, label="validation accuracy")
    plt.axhline(0.976, color="green", linestyle="--", label="single-layer head (0.976)")
    plt.xlabel("epoch"); plt.ylabel("accuracy")
    plt.title(f"Transfer head WITH hidden Dense({HIDDEN_UNITS})")
    plt.legend(); plt.tight_layout()
    plt.savefig("outputs/deep_head_curves.png", dpi=120)
    print("Saved curves to outputs/deep_head_curves.png")
    model.save("outputs/household10_transfer_deep.keras")
    print("Saved model to outputs/household10_transfer_deep.keras")


if __name__ == "__main__":
    main()
