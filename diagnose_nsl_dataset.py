import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import cv2
from collections import Counter
import random

def diagnose_dataset():
    """
    Diagnose potential issues with the NSL dataset
    """
    # Check if the data files exist
    if not os.path.exists("nsl_images.npy") or not os.path.exists("nsl_labels.npy"):
        print("Error: nsl_images.npy or nsl_labels.npy not found!")
        return
    
    # Load the data
    print("Loading data...")
    images = np.load("nsl_images.npy")
    labels = np.load("nsl_labels.npy")
    
    print(f"Dataset shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Basic statistics
    print("\n===== BASIC STATISTICS =====")
    num_classes = len(np.unique(labels))
    print(f"Number of classes: {num_classes}")
    print(f"Min pixel value: {images.min()}, Max pixel value: {images.max()}")
    print(f"Mean pixel value: {images.mean():.4f}, Std: {images.std():.4f}")
    
    # Check for NaN or infinity values
    if np.isnan(images).any():
        print("WARNING: Dataset contains NaN values!")
    if np.isinf(images).any():
        print("WARNING: Dataset contains infinity values!")
    
    # Check label distribution
    print("\n===== LABEL DISTRIBUTION =====")
    class_counts = Counter(labels)
    print(f"Class distribution: {class_counts.most_common()}")
    
    # Check for class imbalance
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    print(f"Class imbalance ratio (max/min): {max_count/min_count:.2f}")
    
    # Plot class distribution
    plt.figure(figsize=(14, 6))
    plt.bar(range(num_classes), [class_counts[i] for i in range(num_classes)])
    plt.title('Class Distribution')
    plt.xlabel('Class Index')
    plt.ylabel('Number of Samples')
    plt.savefig('class_distribution.png')
    plt.close()
    
    # Check for blank/empty images
    print("\n===== CHECKING FOR BLANK/EMPTY IMAGES =====")
    dark_images = 0
    bright_images = 0
    low_contrast_images = 0
    
    for i in range(len(images)):
        img = images[i]
        avg_intensity = np.mean(img)
        contrast = np.std(img)
        
        if avg_intensity < 20:  # Very dark image
            dark_images += 1
        elif avg_intensity > 235:  # Very bright image
            bright_images += 1
        
        if contrast < 10:  # Low contrast
            low_contrast_images += 1
    
    print(f"Number of very dark images: {dark_images} ({dark_images/len(images):.2%})")
    print(f"Number of very bright images: {bright_images} ({bright_images/len(images):.2%})")
    print(f"Number of low contrast images: {low_contrast_images} ({low_contrast_images/len(images):.2%})")
    
    # Visualize some random images from each class
    print("\n===== VISUALIZING RANDOM SAMPLES =====")
    num_samples = min(5, min(class_counts.values()))
    num_classes_to_show = min(10, num_classes)
    
    plt.figure(figsize=(15, num_classes_to_show * 2))
    for c_idx, c in enumerate(range(num_classes_to_show)):
        class_indices = np.where(labels == c)[0]
        sample_indices = random.sample(list(class_indices), num_samples)
        
        for i, idx in enumerate(sample_indices):
            plt.subplot(num_classes_to_show, num_samples, c_idx * num_samples + i + 1)
            plt.imshow(images[idx])
            plt.title(f"Class {c}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images_by_class.png')
    plt.close()
    
    # Try normalizing the data and check if any image becomes completely black or white
    print("\n===== CHECKING NORMALIZATION EFFECTS =====")
    normalized_images = images.astype('float32') / 255.0
    
    # Check for images that become too uniform after normalization
    uniform_after_norm = 0
    for img in normalized_images:
        if np.std(img) < 0.05:  # Very uniform image after normalization
            uniform_after_norm += 1
    
    print(f"Images becoming too uniform after normalization: {uniform_after_norm} ({uniform_after_norm/len(images):.2%})")
    
    # Randomly visualize some normalized images
    plt.figure(figsize=(15, 10))
    for i in range(15):
        idx = random.randint(0, len(images) - 1)
        plt.subplot(3, 5, i + 1)
        plt.imshow(normalized_images[idx])
        plt.title(f"Class {labels[idx]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('normalized_samples.png')
    plt.close()
    
    # Feature visualization using PCA
    print("\n===== FEATURE ANALYSIS =====")
    # Flatten images for PCA
    flattened_images = images.reshape(images.shape[0], -1).astype('float32') / 255.0
    
    # Apply PCA for dimensionality reduction
    print("Running PCA...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(flattened_images)
    
    # Plot PCA results
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='tab20', s=10, alpha=0.6)
    plt.colorbar(scatter)
    plt.title('PCA of Image Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig('pca_visualization.png')
    plt.close()
    
    # Apply t-SNE on a subset for better visualization
    print("Running t-SNE on a subset...")
    # Take a random subset to make t-SNE computation faster
    subset_size = min(2000, len(images))
    random_indices = np.random.choice(len(images), subset_size, replace=False)
    subset_images = flattened_images[random_indices]
    subset_labels = labels[random_indices]
    
    tsne = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=42)
    tsne_result = tsne.fit_transform(subset_images)
    
    # Plot t-SNE results
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=subset_labels, cmap='tab20', s=20, alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE of Image Features')
    plt.savefig('tsne_visualization.png')
    plt.close()
    
    print("\nDiagnosis complete! Check the generated images for visual insights.")

if __name__ == "__main__":
    diagnose_dataset()