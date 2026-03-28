import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load model
model = load_model("best_shape_model.h5")

# IMPORTANT: match this with training output
class_names = ['CIRCLE', 'DIMOND', 'HEART', 'HEXAGON', 'PENTAGON', 'RECTANGLE', 'STAR', 'TRIANGLE']

# ===== CHANGE THIS PATH =====
img_path = "D:/DATASET/SHAPES/DIMOND/1.jpg"

# Check file exists
if not os.path.exists(img_path):
    print("❌ Image not found:", img_path)
    exit()

# Load image
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Check if loaded properly
if img is None:
    print("❌ Failed to load image")
    exit()

# Preprocess
img = cv2.resize(img, (64, 64))
img = img / 255.0
img = img.reshape(1, 64, 64, 1)

# Predict
prediction = model.predict(img)
index = np.argmax(prediction)

# Output
print("\n✅ Predicted Shape:", class_names[index])
print(index)
