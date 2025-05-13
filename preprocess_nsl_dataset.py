import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def augment_minority_classes(images, labels, target_size=500):
    """
    Augment minority classes to balance the dataset
    """
    # Create image data generator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Count samples per class
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_count = max(counts)
    
    # Augment each class that has fewer samples than target_size
    augmented_images = []
    augmented_labels = []
    
    for label in unique_labels:
        # Get images for this class
        class_indices = np.where(labels == label)[0]
        class_images = images[class_indices]
        class_count = len(class_images)
        
        # Add original images
        augmented_images.extend(class_images)
        augmented_labels.extend([label] * class_count)
        
        # If we need more samples
        if class_count < target_size:
            # Calculate how many more samples we need
            num_to_generate = target_size - class_count
            
            # Generate augmented images
            aug_images = []
            for img in class_images:
                img = img.reshape((1,) + img.shape)
                for batch in datagen.flow(img, batch_size=1):
                    aug_images.append(batch[0])
                    if len(aug_images) >= num_to_generate:
                        break
            
            # Add augmented images
            augmented_images.extend(aug_images)
            augmented_labels.extend([label] * len(aug_images))
    
    return np.array(augmented_images), np.array(augmented_labels)

def preprocess_nsl_dataset(dataset_path, target_size=(64, 64)):
    """
    Preprocess the NSL dataset images and save them as numpy arrays.
    
    Args:
        dataset_path: Path to the NSL dataset directory
        target_size: Target size for resizing images (height, width)
    """
    # List all background directories (Plain Background, Random Background)
    print(f"Processing NSL dataset...")
    background_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    background_dirs.sort()
    
    images = []
    labels = []
    label_map = {}  # To store mapping of class names to numeric labels
    
    # Mapping for Nepali letters - Devanagari characters only
    nepali_labels = {
        '0': 'क', '1': 'ख', '2': 'ग', '3': 'घ', '4': 'ङ',
        '5': 'च', '6': 'छ', '7': 'ज', '8': 'झ', '9': 'ञ',
        '10': 'ट', '11': 'ठ', '12': 'ड', '13': 'ढ', '14': 'ण', # Retroflex consonants
        '15': 'त', '16': 'थ', '17': 'द', '18': 'ध', '19': 'न', # Dental consonants
        '20': 'प', '21': 'फ', '22': 'ब', '23': 'भ', '24': 'म',
        '25': 'य', '26': 'र', '27': 'ल', '28': 'व', '29': 'श', 
        '30': 'ष', '31': 'स', '32': 'ह', '33': 'क्ष', '34': 'त्र', '35': 'ज्ञ'
    }
    
    # Process each background directory
    for background_dir in background_dirs:
        background_path = os.path.join(dataset_path, background_dir)
        
        # Check if background directory exists
        if not os.path.isdir(background_path):
            print(f"Warning: Background directory {background_path} not found, skipping.")
            continue
        
        # List all class directories (0-35)
        class_dirs = [d for d in os.listdir(background_path) if os.path.isdir(os.path.join(background_path, d))]
        class_dirs.sort()  # Sort to ensure consistent class ordering
        
        # Process each class directory
        for class_dir in class_dirs:
            try:
                class_idx = int(class_dir)  # Convert folder name to integer for class index
                nepali_name = nepali_labels.get(class_dir, f"Class {class_dir}")
                label_map[class_idx] = nepali_name
                
                class_path = os.path.join(background_path, class_dir)
                
                # Check if class directory exists
                if not os.path.isdir(class_path):
                    continue
                
                # Get all image files in the class directory
                image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                # Process each image
                for image_file in image_files:
                    image_path = os.path.join(class_path, image_file)
                    
                    # Read and preprocess image
                    img = cv2.imread(image_path)
                    if img is None:
                        continue
                    
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize image
                    img = cv2.resize(img, target_size)
                    
                    # Add to dataset
                    images.append(img)
                    labels.append(class_idx)
            except ValueError:
                continue
    
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Initial dataset size: {len(images)} images across {len(np.unique(labels))} classes")
    
    # Balance dataset through augmentation
    print("Balancing dataset through augmentation...")
    images, labels = augment_minority_classes(images, labels)
    
    print(f"Final dataset size after augmentation: {len(images)} images")
    
    # Normalize images to [0,1] range
    images = images.astype('float32') / 255.0
    
    # Save the preprocessed dataset
    np.save("nsl_images.npy", images)
    np.save("nsl_labels.npy", labels)
    
    # Save the label mapping for reference
    with open("nsl_label_map.txt", "w") as f:
        for class_idx, nepali_name in sorted(label_map.items()):
            f.write(f"{class_idx}: {nepali_name}\n")
    
    # Visualize final class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(15, 5))
    plt.bar(unique_labels, counts)
    plt.title('Class Distribution After Augmentation')
    plt.xlabel('Class Index')
    plt.ylabel('Number of Samples')
    plt.xticks(unique_labels)
    plt.savefig('class_distribution.png')
    plt.close()
    
    # Display sample images from each class
    plt.figure(figsize=(15, 15))
    for i, label in enumerate(unique_labels):
        if i >= 36:  # Only show first 36 classes
            break
        idx = np.where(labels == label)[0][0]
        plt.subplot(6, 6, i + 1)
        plt.imshow(images[idx])
        plt.title(f'{label}: {nepali_labels[str(label)]}')
        plt.axis('off')
    plt.savefig('sample_images.png')
    plt.close()
    
    return images, labels

if __name__ == "__main__":
    # Path to the NSL dataset directory
    dataset_path = "/Users/umangarayamajhi/Desktop/sanket/NSL"
    
    # Check if the dataset directory exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory not found!")
        exit(1)
    
    # Preprocess the dataset
    images, labels = preprocess_nsl_dataset(dataset_path)
    
    print("Preprocessing complete! Files saved: nsl_images.npy, nsl_labels.npy, nsl_label_map.txt")