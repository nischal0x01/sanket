import cv2
import os
import matplotlib.pyplot as plt

# Path to one category (start with Plain Background)
folder = "/Users/umangarayamajhi/Desktop/sanket/NSL/Plain Background"

# Loop through a few subfolders and display some images
for subfolder in os.listdir(folder):
    subfolder_path = os.path.join(folder, subfolder)
    if os.path.isdir(subfolder_path):
        print(f"Showing images from folder: {subfolder}")
        for filename in os.listdir(subfolder_path)[:3]:  # Show first 3 images per folder
            if filename.endswith(".jpg"):
                img_path = os.path.join(subfolder_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    # Convert BGR (OpenCV format) to RGB (Matplotlib format)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.figure(figsize=(5, 5))
                    plt.title(f"Character {subfolder}")
                    plt.imshow(img_rgb)
                    plt.axis("off")
                    plt.show(block=False)
                    plt.pause(1)  # Show for 1 second
                    plt.close()
                else:
                    print(f"Failed to load image: {img_path}")