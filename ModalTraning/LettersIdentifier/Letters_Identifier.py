import os
import numpy as np
from PIL import Image

# Define dataset path and target size for images
dataset_path = "C:\Users\PCS\Desktop\Signin language datasets\4"
image_size = (64, 64)  # Resize all images to 64x64

# Loop through each folder (label)
for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    if os.path.isdir(label_path):  # Ensure it's a directory
        label_data = []  # Initialize list for this label's images
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            try:
                # Load image, convert to RGB, and resize
                img = Image.open(img_path).convert("RGB")
                img = img.resize(image_size)
                img_array = np.array(img)

                # Append image to label_data
                label_data.append(img_array)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

        # Convert label_data to numpy array and save as .npy
        label_data = np.array(label_data)
        np.save(f"{label}.npy", label_data)
        print(f"Saved {label}.npy with {len(label_data)} images.")
