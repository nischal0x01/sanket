import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Nepali character mapping with Devanagari only (matching preprocessing)
nepali_chars = ["क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ", "ट", "ठ", "ड", "ढ", "ण", 
                "त", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श", 
                "ष", "स", "ह", "क्ष", "त्र", "ज्ञ"]

def load_data():
    """
    Load preprocessed data from numpy files
    """
    try:
        images = np.load("nsl_images.npy")
        labels = np.load("nsl_labels.npy")
        print(f"Loaded preprocessed data: {images.shape} images, {labels.shape} labels")
        return images, labels
    except FileNotFoundError:
        print("Error: Preprocessed data not found. Please run preprocess_nsl_dataset.py first.")
        return None, None

def build_model(input_shape, num_classes):
    """
    Build a CNN for NSL recognition with additional regularization
    """
    model = models.Sequential([
        # First convolutional block with L2 regularization
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape,
                     kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                     kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),  # Increased dropout
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation="relu", padding="same",
                     kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same",
                     kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation="relu", padding="same",
                     kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same",
                     kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Fully connected layers
        layers.Flatten(),
        layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    
    # Compile the model with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Reduced learning rate
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=30):
    """
    Train the model with class weights and additional callbacks
    """
    # Data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)
    
    # Calculate class weights to handle imbalance
    class_weights = {}
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    
    for cls, count in zip(unique_classes, class_counts):
        # Inverse frequency weighting
        class_weights[cls] = total_samples / (len(unique_classes) * count)
    
    # Early stopping with longer patience
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce learning rate on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Model checkpoint to save best model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train with data augmentation, class weights, and callbacks
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=epochs,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )
    
    return history

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    """
    # Accuracy plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data
    """
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Confusion matrix & per-class accuracy
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get per-class accuracy
    class_accuracies = {}
    for i in range(len(nepali_chars)):
        class_indices = np.where(y_test == i)[0]
        if len(class_indices) > 0:
            class_accuracy = np.mean(predicted_classes[class_indices] == i)
            class_accuracies[i] = class_accuracy
            print(f"Class {i} ({nepali_chars[i]}): {class_accuracy:.4f} accuracy")
    
    return test_accuracy, class_accuracies

def main():
    # Load data
    X, y = load_data()
    
    if X is None or y is None:
        return
    
    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Print class distribution
    print("\nClass distribution in training set:")
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique_classes, class_counts):
        print(f"Class {cls} ({nepali_chars[cls]}): {count} samples")
    
    # Build the model
    input_shape = X_train.shape[1:]
    num_classes = len(nepali_chars)
    
    model = build_model(input_shape, num_classes)
    model.summary()
    
    # Train the model
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=50)  # Increased epochs
    
    # Plot training history
    plot_training_history(history)
    
    # Load the best model for evaluation
    if os.path.exists('best_model.h5'):
        model = tf.keras.models.load_model('best_model.h5')
        print("Loaded best model for evaluation")
    
    # Evaluate on test set
    test_accuracy, class_accuracies = evaluate_model(model, X_test, y_test)
    
    # Save the model
    model.save("nsl_model.h5")
    print("Model saved to nsl_model.h5")
    
    # Check for problematic classes
    problem_classes = {i: acc for i, acc in class_accuracies.items() if acc < 0.8}
    if problem_classes:
        print("\nWarning: Some classes have low accuracy:")
        for class_idx, accuracy in problem_classes.items():
            print(f"  Class {class_idx} ({nepali_chars[class_idx]}): {accuracy:.4f}")
        print("\nConsider collecting more data for these classes or improving the model.")

if __name__ == "__main__":
    main()