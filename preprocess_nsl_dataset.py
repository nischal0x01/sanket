import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt

def preprocess_nsl_dataset(dataset_path, target_size=(64, 64)):
    """
    Preprocess the NSL dataset images and save them as numpy arrays.
    
    Args:
        dataset_path: Path to the NSL dataset directory
        target_size: Target size for resizing images (height, width)
    """
    # List all class directories
    print(f"Loading dataset from: {dataset_path}")
    class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    class_dirs.sort()  # Sort to ensure consistent class ordering
    
    images = []
    labels = []
    label_map = {}  # To store mapping of class names to numeric labels
    
    print(f"Found {len(class_dirs)} classes: {class_dirs}")
    
    # Process each class directory
    for class_idx, class_dir in enumerate(class_dirs):
        label_map[class_dir] = class_idx
        class_path = os.path.join(dataset_path, class_dir)
        print(f"Processing class {class_dir} ({class_idx}):")
        
        # Get all image files in the class directory
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"  Found {len(image_files)} images")
        
        # Process each image
        for image_file in tqdm(image_files, desc=f"Class {class_dir}"):
            image_path = os.path.join(class_path, image_file)
            
            # Read and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}")
                continue
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, target_size)
            
            # Add to dataset
            images.append(img)
            labels.append(class_idx)
    
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Dataset shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    # Save the preprocessed dataset
    np.save("nsl_images.npy", images)
    np.save("nsl_labels.npy", labels)
    
    # Save the label mapping for reference
    with open("nsl_label_map.txt", "w") as f:
        for class_name, idx in label_map.items():
            f.write(f"{idx}: {class_name}\n")
    
    # Visualize class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(12, 6))
    plt.bar(unique_labels, counts)
    plt.title('Class Distribution')
    plt.xlabel('Class Index')
    plt.ylabel('Number of Samples')
    plt.savefig('class_distribution.png')
    plt.show()
    
    # Display sample images from each class
    plt.figure(figsize=(15, 15))
    for i, class_idx in enumerate(unique_labels):
        if i >= 36:  # Display max 36 classes
            break
        
        # Get random image from this class
        class_indices = np.where(labels == class_idx)[0]
        if len(class_indices) > 0:
            img_idx = np.random.choice(class_indices)
            img = images[img_idx]
            
            plt.subplot(6, 6, i + 1)
            plt.imshow(img)
            plt.title(f"Class {class_idx}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.show()
    
    return images, labels, label_map

if __name__ == "__main__":
    # Path to the NSL dataset directory
    dataset_path = os.path.join("NSL", "Random Background")
    
    # Check if the dataset directory exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory {dataset_path} not found!")
        print("Please ensure the NSL dataset is correctly placed in the specified directory.")
        exit(1)
    
    # Preprocess the dataset
    images, labels, label_map = preprocess_nsl_dataset(dataset_path)
    
    print("Preprocessing complete!")
    print(f"Saved preprocessed data to nsl_images.npy and nsl_labels.npy")
    print(f"Saved label mapping to nsl_label_map.txt")