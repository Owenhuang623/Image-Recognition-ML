"""
Stage 1 — Train the CNN and see how it does.

This ties everything together: load data -> build model -> compile -> train ->
evaluate -> plot the learning curves. Run it with:

    ./venv/bin/python src/train.py

On a laptop CPU this takes a few minutes. Expect ~70% test accuracy — far
above the 10% you'd get by random guessing across 10 classes.
"""

import os
import matplotlib.pyplot as plt

from data import load_data
from model import build_model

EPOCHS = 10          # one epoch = one full pass over the training data
BATCH_SIZE = 64      # how many images the model looks at before each weight update


def main():
    # 1. Data ----------------------------------------------------------------
    (x_train, y_train), (x_test, y_test) = load_data()

    # 2. Model ---------------------------------------------------------------
    model = build_model()
    model.summary()

    # 3. Compile -------------------------------------------------------------
    # "Compiling" tells Keras HOW to train:
    #   optimizer = the rule for updating weights ('adam' is the safe default)
    #   loss      = the number we minimize. sparse_categorical_crossentropy is
    #               the right loss for integer labels + softmax outputs.
    #   metrics   = extra numbers to report; accuracy is the intuitive one.
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 4. Train ---------------------------------------------------------------
    # validation_split=0.1 carves 10% off the training data as a VALIDATION set
    # the model doesn't learn from. Comparing training vs. validation accuracy
    # is how we detect overfitting (memorizing instead of generalizing).
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
    )

    # 5. Evaluate on the held-out test set -----------------------------------
    # This is the honest score: data the model has never touched in any way.
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nFinal test accuracy: {test_acc:.3f}")

    # 6. Plot the learning curves --------------------------------------------
    # If the training curve keeps climbing while validation flattens or dips,
    # that gap IS overfitting — the thing we'll fight in later stages.
    plot_history(history)

    # 7. Save the trained model so we don't have to retrain to reuse it.
    os.makedirs("outputs", exist_ok=True)
    model.save("outputs/fashion_mnist_cnn.keras")
    print("Saved trained model to outputs/fashion_mnist_cnn.keras")


def plot_history(history, save_path="outputs/training_curves.png"):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    plt.figure(figsize=(6, 4))
    plt.plot(acc, label="training accuracy")
    plt.plot(val_acc, label="validation accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Learning curves")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    print(f"Saved learning curves to {save_path}")


if __name__ == "__main__":
    main()
