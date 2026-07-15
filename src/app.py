"""
A tiny phone-friendly web app to test the classifier.

Flow:
  1. This starts a small web server on your Mac, reachable over your WiFi.
  2. It prints a URL and saves a QR code (outputs/qr.png).
  3. Scan the QR with your phone (same WiFi) -> opens the upload page.
  4. Take/upload a photo of a household item -> see the classification.

Run:  ./venv/bin/python src/app.py
Stop: Ctrl-C  (or stop the background task)

Note: your Mac and phone must be on the SAME WiFi network. macOS may pop up a
"allow incoming connections" firewall prompt the first time -- click Allow.
"""

import io
import json
import socket

import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf

MODEL_PATH = "outputs/household10_transfer_deep.keras"
CLASS_NAMES = json.load(open("outputs/class_names.json"))
IMG_SIZE = (224, 224)
PORT = 8000

# Load the trained model ONCE at startup (not per request -- loading is slow).
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

app = Flask(__name__)

# Mobile-first page: a big "take/choose photo" button, a preview, and results.
# accept="image/*" capture="environment" makes phones offer the rear camera.
PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Household Item Classifier</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0;
         padding: 24px; max-width: 520px; margin: 0 auto; }
  h1 { font-size: 1.4rem; }
  p.sub { opacity: 0.7; margin-top: -8px; }
  label.btn { display: block; text-align: center; padding: 18px; border-radius: 14px;
        background: #2563eb; color: #fff; font-size: 1.1rem; font-weight: 600;
        cursor: pointer; margin: 20px 0; }
  #preview { width: 100%; border-radius: 14px; display: none; margin: 12px 0; }
  input[type=file] { display: none; }
  .row { display: flex; align-items: center; gap: 10px; margin: 10px 0; }
  .bar { height: 20px; background: #2563eb; border-radius: 6px; }
  .name { width: 110px; font-weight: 600; }
  .pct { width: 56px; text-align: right; opacity: 0.8; }
  #status { text-align: center; opacity: 0.7; }
</style>
</head>
<body>
  <h1>🏠 Household Item Classifier</h1>
  <p class="sub">Take or upload a photo of one of: camera, ceiling fan, cellphone,
  chair, cup, lamp, laptop, stapler, umbrella, watch.</p>

  <label class="btn" for="file">📷 Take / Choose Photo</label>
  <input id="file" type="file" accept="image/*" capture="environment">

  <img id="preview">
  <div id="status"></div>
  <div id="results"></div>

<script>
const fileInput = document.getElementById('file');
const preview = document.getElementById('preview');
const status = document.getElementById('status');
const results = document.getElementById('results');

fileInput.addEventListener('change', async () => {
  const file = fileInput.files[0];
  if (!file) return;

  // Show a local preview immediately.
  preview.src = URL.createObjectURL(file);
  preview.style.display = 'block';
  results.innerHTML = '';
  status.textContent = 'Classifying…';

  // Send the image to the server.
  const form = new FormData();
  form.append('image', file);
  try {
    const resp = await fetch('/predict', { method: 'POST', body: form });
    const data = await resp.json();
    status.textContent = '';
    results.innerHTML = data.predictions.map(p => `
      <div class="row">
        <div class="name">${p.label}</div>
        <div class="bar" style="width:${Math.max(2, p.prob*260)}px"></div>
        <div class="pct">${(p.prob*100).toFixed(1)}%</div>
      </div>`).join('');
  } catch (e) {
    status.textContent = 'Error: ' + e;
  }
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify(error="no image"), 400

    # Read the uploaded bytes, fix phone EXIF rotation, force RGB, resize to 160.
    raw = request.files["image"].read()
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img).convert("RGB").resize(IMG_SIZE)

    # To a (1, 160, 160, 3) float array in [0,255] -- the model does the rest
    # of the preprocessing (MobileNet scaling) internally.
    arr = np.expand_dims(np.asarray(img, dtype="float32"), axis=0)
    probs = model.predict(arr, verbose=0)[0]

    # Return the top 3 as JSON.
    order = np.argsort(probs)[::-1][:3]
    preds = [{"label": CLASS_NAMES[i], "prob": float(probs[i])} for i in order]
    return jsonify(predictions=preds)


def local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


if __name__ == "__main__":
    import qrcode
    url = f"http://{local_ip()}:{PORT}/"
    qrcode.make(url).save("outputs/qr.png")
    print("\n" + "=" * 46)
    print(f"  Open on your phone (same WiFi):  {url}")
    print(f"  Or scan the QR code:             outputs/qr.png")
    print("=" * 46 + "\n")
    # host=0.0.0.0 makes it reachable from other devices on the LAN (your phone).
    app.run(host="0.0.0.0", port=PORT, debug=False)
