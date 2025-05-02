import cv2
import numpy as np
import os
from sklearn.utils import class_weight

# Path to dataset
folder = "/Users/umangarayamajhi/Desktop/sanket/NSL/Plain Background"  # Replace with your path
images = []
labels = []

# Load images and labels
for subfolder in os.listdir(folder):
    subfolder_path = os.path.join(folder, subfolder)
    if os.path.isdir(subfolder_path):
        for filename in os.listdir(subfolder_path):
            if filename.endswith(".jpg"):
                img_path = os.path.join(subfolder_path, filename)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (64, 64))  # Resize to 64x64
                img = img / 255.0  # Normalize
                images.append(img)
                labels.append(int(subfolder))  # Use folder number as label (0-35)

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Ensure the dataset is balanced
class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)
class_weights_dict = dict(enumerate(class_weights))

# Save preprocessed data
np.save("nsl_images.npy", images)
np.save("nsl_labels.npy", labels)

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Class weights:", class_weights_dict)
