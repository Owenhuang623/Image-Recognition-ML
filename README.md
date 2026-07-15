# Image-Recognition-ML

An educational, hands-on project for learning how convolutional neural networks (CNNs)
do image classification — built up from first principles to a working real-world classifier
you can test from your phone.

The project deliberately walks through **two stages**, so each core idea is learned in
isolation before being combined:

1. **A CNN trained from scratch** on a simple dataset (Fashion-MNIST) — to learn the
   mechanics: convolution, pooling, the training loop, and overfitting.
2. **A transfer-learning classifier** for **10 common household items** (real photos) —
   the practical way to get high accuracy from little data, plus data augmentation and a
   phone-accessible web interface.

---

## What it can do

- Train a small CNN from scratch and read its learning curves.
- Build a **10-class household-item classifier** (camera, ceiling fan, cellphone, chair,
  cup, lamp, laptop, stapler, umbrella, watch) by fine-tuning a pretrained MobileNetV2 —
  reaching ~98–99% validation accuracy from only ~500 images.
- Classify an image from the command line, or from your **phone** via a QR-code web app.
- Demonstrate *why* transfer learning matters (a controlled from-scratch vs. transfer
  experiment) and *how* data augmentation works (with a visualizer).

---

## Setup

TensorFlow does not yet support Python 3.14, so the project uses a **Python 3.12**
virtual environment.

```bash
# create the venv (uses Python 3.12) and install dependencies
python3.12 -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements.txt
```

Run every script with `./venv/bin/python <script>` from the project root.

> **macOS note:** if dataset downloads fail with an SSL certificate error, run
> `/Applications/Python 3.12/Install Certificates.command` once.

---

## Stage 1 — a CNN from scratch (Fashion-MNIST)

Learn the fundamentals: a 3-block conv/pool CNN classifying 28×28 grayscale clothing images.

```bash
./venv/bin/python src/data.py         # load + preprocess, print shapes (sanity check)
./venv/bin/python src/visualize.py    # save a grid of sample images (look before you train)
./venv/bin/python src/model.py        # print the model summary (see the funnel shape)
./venv/bin/python src/train.py        # train, evaluate (~91%), plot learning curves
```

| File | Role |
|------|------|
| `src/data.py` | Load and preprocess Fashion-MNIST (normalize pixels, tidy labels) |
| `src/visualize.py` | Save a grid of sample images to inspect before training |
| `src/model.py` | The from-scratch CNN (Conv → Pool blocks → dense head) |
| `src/train.py` | Compile, train, evaluate, and plot the learning curves |

---

## Stage 2 — a real household-item classifier (transfer learning)

Instead of training from scratch, we freeze a **MobileNetV2** backbone (pretrained on
ImageNet's 1.28M photos) and train only a small classification head on ~500 real photos of
10 household items, curated from the Caltech-101 dataset.

```bash
./venv/bin/python src/caltech_data.py    # load the 10-class folder dataset, print shapes
./venv/bin/python src/train_transfer.py  # train the head (~98% val accuracy)
./venv/bin/python src/classify.py path/to/photo.jpg   # classify a single image
```

| File | Role |
|------|------|
| `src/caltech_data.py` | Load real photos from `data/household10/<class>/`, resize to 224×224 |
| `src/transfer_model.py` | Frozen MobileNetV2 backbone + augmentation + trainable head |
| `src/train_transfer.py` | Train the head, evaluate, plot, and save the model |
| `src/classify.py` | Command-line classifier for one image |

### The dataset (`data/household10/`)

Real photos in the standard "one folder per class" layout (this folder is git-ignored and
re-buildable). Each class is capped at 100 images for balance:

```
data/household10/
    camera/   ceiling_fan/  cellphone/  chair/  cup/
    lamp/     laptop/       stapler/    umbrella/  watch/
```

The images come from **Caltech-101**
(`https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip`).

---

## Phone web app

Classify photos from your phone: it starts a small local server, prints a URL, and saves a
QR code. Your phone and Mac must be on the **same WiFi**.

```bash
./venv/bin/python src/app.py
# -> open the printed URL on your phone, or scan outputs/qr.png
# stop with Ctrl-C
```

The page lets you take or upload a photo and shows the top-3 predictions with confidence
bars. `src/app.py` loads the trained model once and serves a `/predict` endpoint.

---

## Learning experiments

These scripts exist to teach specific concepts by demonstration:

| File | What it demonstrates |
|------|----------------------|
| `src/scratch_model.py` + `src/train_scratch.py` | Trains a from-scratch CNN on the **same** household data (leak-free, same split) — reaches only ~53% vs transfer learning's ~98%, proving *why* transfer learning is needed |
| `src/train_deep_head.py` | Adds a hidden layer to the head — tests whether a deeper classifier helps (it's within noise on this data) |
| `src/augmentation.py` | The data-augmentation pipeline (flip, rotate, zoom, translate, contrast, brightness) with the *golden rule* and deliberately-excluded transforms documented |
| `src/visualize_augmentation.py` | Saves a grid showing one image augmented many ways — sanity-check that augmentations stay realistic |

---

## Key concepts covered

- **Convolution & pooling** — how a CNN turns pixels into features (the shrinking-spatial,
  growing-channel "funnel").
- **The training loop** — forward pass → loss → backpropagation → optimizer step; what is
  actually *trained* (the weights) vs. *chosen* (the architecture).
- **Overfitting** — reading train vs. validation curves; fighting it with dropout and
  augmentation.
- **Transfer learning** — freezing a pretrained backbone (learn to *see*) and training only
  a small head (learn to *name*); why ~500 images is enough.
- **Data augmentation** — manufacturing realistic variations; the golden rule that every
  augmentation must remain a plausible photo of the same class.
- **Preprocessing** — resizing/normalizing so all inputs match the network's fixed input.
- **Honest evaluation** — leak-free train/val splits, and the gap between a saturated
  metric and real-world (phone-photo) performance.

---

## Project layout

```
Image-Recognition-ML/
├── requirements.txt
├── src/
│   ├── data.py, visualize.py, model.py, train.py        # Stage 1 (Fashion-MNIST)
│   ├── caltech_data.py, transfer_model.py, train_transfer.py   # Stage 2 (transfer learning)
│   ├── augmentation.py, visualize_augmentation.py       # data augmentation
│   ├── scratch_model.py, train_scratch.py               # from-scratch control experiment
│   ├── train_deep_head.py                               # deeper-head experiment
│   ├── classify.py                                      # CLI classifier
│   └── app.py                                           # phone web app
├── data/household10/     # real image data (git-ignored, re-buildable)
├── outputs/              # trained models, plots, QR code (git-ignored)
└── venv/                 # Python 3.12 virtual environment (git-ignored)
```

## Tech stack

TensorFlow / Keras · NumPy · Matplotlib · Flask · MobileNetV2 (ImageNet-pretrained)
