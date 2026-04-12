import os
import cv2
import numpy as np
from flask import Flask, render_template, request

import requests
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER




# 🔒 FINAL CLASS LIST (your correct order)
class_names = ['CIRCLE', 'DIMOND', 'HEART', 'HEXAGON', 'PENTAGON', 'RECTANGLE', 'STAR', 'TRIANGLE']


def predict_image(img_path):
    url = "http://64.227.158.176/"

    with open(img_path, "rb") as f:
        response = requests.post(url, files={"file": f})

    html = response.text

    # DEBUG (optional)
    print(html[:500])

    
    try:
        
        import re

        pred_match = re.search(r'Prediction:\s*(\w+)', html)
        conf_match = re.search(r'Confidence:\s*([\d\.]+)', html)

        prediction = pred_match.group(1) if pred_match else "Not Found"
        confidence = float(conf_match.group(1))/100 if conf_match else 0

        return prediction, confidence

    except:
        return "Error parsing response", 0

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