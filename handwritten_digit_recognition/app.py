from flask import Flask, request, jsonify, render_template

# import pandas as pd
import tensorflow as tf
import numpy as np 
import cv2 as cv
import base64, re
from io import BytesIO
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


model = tf.keras.models.load_model("models/handwritten_digit_recognition.keras")


def preprocess_image(image_data):
    image_str = re.search(r'base64,(.*)', image_data).group(1)
    image_bytes = base64.b64decode(image_str)

    # with open("raw_input.png", "wb") as f:
    #     f.write(image_bytes)

    img = Image.open(BytesIO(image_bytes)).convert('L')  # Grayscale

    # Resize to 28x28 and convert to NumPy
    img = img.resize((28, 28), resample=Image.Resampling.LANCZOS)
    img_np = np.array(img)

    # Invert if background is white
    if np.mean(img_np) > 127:
        img_np = 255 - img_np

    # Optional: binarize to remove noise
    img_np = (img_np > 30).astype(np.uint8) * 255

    # Normalize to [0, 1]
    img_np = img_np.astype('float32') / 255.0

    # Reshape to match model input
    img_np = img_np.reshape(1, 28, 28, 1)

    return img_np



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        processed = preprocess_image(data["image"])
        prediction = model.predict(processed)
        predicted_label = int(np.argmax(prediction))

        return jsonify({"prediction": predicted_label})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)