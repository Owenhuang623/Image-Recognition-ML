"""
Use the trained model to classify a photo of a household item.

This is the goal we set out for: point it at an image, get a prediction.

Run:
    ./venv/bin/python src/classify.py path/to/photo.jpg

The saved model is self-contained -- it bundles the resize-time preprocessing,
so we just hand it a raw image and read off the probabilities.
"""

import os
import sys
import numpy as np
import tensorflow as tf

MODEL_PATH = "outputs/household10_transfer.keras"
DATA_DIR = "data/household10"
IMG_SIZE = (160, 160)


def get_class_names():
    """Recover the class names in the SAME order the model was trained on.

    Keras used the alphabetically-sorted sub-folder names as labels 0..9, so we
    reproduce that here. (In a production project you'd save this list to a file
    at training time rather than depend on the data folder still being present.)
    """
    return sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    )


def classify(image_path, top_k=3):
    class_names = get_class_names()
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load the image and resize to what the model expects. load_img handles any
    # common format (jpg/png) and any input size; we standardize to 160x160.
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)          # (160, 160, 3), floats 0-255
    batch = np.expand_dims(arr, axis=0)             # models expect a batch: (1, 160, 160, 3)

    # predict() returns the 10 softmax probabilities for our one image.
    probs = model.predict(batch, verbose=0)[0]

    # Sort classes by probability, highest first, and show the top few.
    order = np.argsort(probs)[::-1][:top_k]
    print(f"\nPrediction for: {image_path}")
    print("-" * 40)
    for rank, i in enumerate(order, 1):
        bar = "#" * int(probs[i] * 30)
        print(f"{rank}. {class_names[i]:12s} {probs[i]*100:5.1f}%  {bar}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./venv/bin/python src/classify.py path/to/photo.jpg")
        sys.exit(1)
    classify(sys.argv[1])
