import cv2
import numpy as np
import os
import random

input_dir = "D:/DATASET/SHAPES"
output_dir = "D:/DATASET/AUG_SHAPES"

IMG_SIZE = 64
TARGET_COUNT = 500

os.makedirs(output_dir, exist_ok=True)

# Only allow image files
valid_ext = [".png", ".jpg", ".jpeg"]


def random_transform(img):
    rows, cols = img.shape

    # Rotation
    angle = random.randint(-60, 60)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img = cv2.warpAffine(img, M, (cols, rows), borderValue=255)

    # Scaling (with center crop)
    scale = random.uniform(0.6, 1.4)
    scaled = cv2.resize(img, None, fx=scale, fy=scale)

    # Center crop/pad to maintain shape position
    h, w = scaled.shape
    if h > IMG_SIZE:
        start = (h - IMG_SIZE) // 2
        scaled = scaled[start:start+IMG_SIZE, start:start+IMG_SIZE]
    else:
        pad = IMG_SIZE - h
        scaled = cv2.copyMakeBorder(
            scaled, pad//2, pad//2, pad//2, pad//2,
            cv2.BORDER_CONSTANT, value=255
        )

    img = cv2.resize(scaled, (IMG_SIZE, IMG_SIZE))

    # Translation
    tx = random.randint(-8, 8)
    ty = random.randint(-8, 8)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), borderValue=255)

    return img


def thickness_variation(img):
    kernel_size = random.choice([1, 2])
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if random.random() > 0.5:
        img = cv2.erode(img, kernel, iterations=1)
    else:
        img = cv2.dilate(img, kernel, iterations=1)

    return img


def add_noise(img):
    noise = np.random.randint(0, 15, img.shape, dtype='uint8')
    img = cv2.subtract(img, noise)
    return img


for cls in os.listdir(input_dir):
    class_path = os.path.join(input_dir, cls)

    # Skip non-folders
    if not os.path.isdir(class_path):
        continue

    save_path = os.path.join(output_dir, cls)
    os.makedirs(save_path, exist_ok=True)

    # Filter only images
    images = [f for f in os.listdir(class_path)
              if os.path.splitext(f)[1].lower() in valid_ext]

    if len(images) == 0:
        print(f"⚠️ No images in {cls}")
        continue

    count = 0

    while count < TARGET_COUNT:
        img_name = random.choice(images)
        img_path = os.path.join(class_path, img_name)

        img = cv2.imread(img_path, 0)

        # Skip if image not loaded
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        img = random_transform(img)
        img = thickness_variation(img)
        img = add_noise(img)

        # Flip (avoid for HEART)
        if cls not in ["HEART"] and random.random() > 0.5:
            img = cv2.flip(img, 1)

        # Final threshold (keep shape clear)
        _, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)

        cv2.imwrite(os.path.join(save_path, f"{count}.png"), img)
        count += 1

    print(f"✅ {cls} done (500 images)")