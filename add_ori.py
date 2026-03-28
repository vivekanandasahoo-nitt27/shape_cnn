import cv2
import os
import random

input_dir = "D:/DATASET/SHAPES"
output_dir = "D:/DATASET/AUG_SHAPES"

IMG_SIZE = 64
ADD_COUNT = 70

valid_ext = [".png", ".jpg", ".jpeg"]

for cls in os.listdir(input_dir):
    class_path = os.path.join(input_dir, cls)
    aug_class_path = os.path.join(output_dir, cls)

    if not os.path.isdir(class_path):
        continue

    if not os.path.exists(aug_class_path):
        print(f"⚠️ Missing AUG folder for {cls}")
        continue

    # Get original images
    images = [f for f in os.listdir(class_path)
              if os.path.splitext(f)[1].lower() in valid_ext]

    if len(images) == 0:
        print(f"⚠️ No images in {cls}")
        continue

    # Get current count in AUG folder
    existing = len(os.listdir(aug_class_path))

    count = 0
    index = existing  # start from next number

    while count < ADD_COUNT:
        img_name = random.choice(images)
        img_path = os.path.join(class_path, img_name)

        img = cv2.imread(img_path, 0)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        save_path = os.path.join(aug_class_path, f"{index}.png")
        cv2.imwrite(save_path, img)

        index += 1
        count += 1

    print(f"✅ {cls}: Added 70 originals (Total ≈ {existing + 70})")