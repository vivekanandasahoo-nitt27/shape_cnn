import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model once
model = load_model("best_shape_model.h5")

# 🔒 FINAL CLASS LIST (your correct order)
class_names = ['CIRCLE', 'DIMOND', 'HEART', 'HEXAGON', 'PENTAGON', 'RECTANGLE', 'STAR', 'TRIANGLE']


def predict_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = img.reshape(1, 64, 64, 1)

    pred = model.predict(img)
    index = np.argmax(pred)

    return class_names[index], float(np.max(pred))


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    img_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            pred, conf = predict_image(filepath)

            prediction = pred
            confidence = round(conf * 100, 2)
            img_path = filepath

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           img_path=img_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)