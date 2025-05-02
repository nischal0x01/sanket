import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# First, try to import sklearn, install if not available
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("sklearn not found. Installing scikit-learn...")
    os.system("pip install scikit-learn")
    from sklearn.model_selection import train_test_split

# Load preprocessed data
try:
    images = np.load("nsl_images.npy")
    labels = np.load("nsl_labels.npy")
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Load the model
    model = tf.keras.models.load_model("nsl_model.h5")
    
    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")
    
except Exception as e:
    print(f"An error occurred: {e}")
    print("\nTroubleshooting tips:")
    print("1. Make sure 'nsl_images.npy' and 'nsl_labels.npy' files exist in the current directory")
    print("2. Make sure 'nsl_model.h5' exists in the current directory")
    print("3. If you're still seeing sklearn errors, try manually installing it with:")
    print("   pip install scikit-learn")