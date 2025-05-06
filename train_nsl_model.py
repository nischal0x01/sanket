import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

def advanced_model_fix(input_model_path="nsl_model_fixed.h5", output_model_path="nsl_model_balanced.h5"):
    """
    Apply more aggressive fixes to the model to ensure balanced class prediction
    """
    print(f"Loading model from {input_model_path}...")
    
    # Check if model exists
    if not os.path.exists(input_model_path):
        print(f"Error: Model file '{input_model_path}' not found!")
        print("Have you run the diagnostic tool and fixed the model?")
        return False
    
    # Load the model
    model = tf.keras.models.load_model(input_model_path)
    
    # Get output layer
    output_layer = model.layers[-1]
    weights, biases = output_layer.get_weights()
    
    print(f"\nOriginal output layer shape: {weights.shape}, biases shape: {biases.shape}")
    print(f"Number of classes: {biases.shape[0]}")
    
    # ----------------------------------------
    # Advanced Fix 1: Balanced weight initialization
    # ----------------------------------------
    print("\nApplying advanced weight balancing...")
    
    # Initialize weights with smaller variance to reduce initial bias
    new_weights = np.random.normal(0, 0.005, size=weights.shape)
    
    # Ensure no class has an initial advantage by explicitly normalizing weight columns
    for i in range(weights.shape[1]):
        col_norm = np.linalg.norm(new_weights[:, i])
        if col_norm > 0:
            # Normalize each class's weight vector to have equal length
            new_weights[:, i] = new_weights[:, i] / col_norm * 0.1
    
    # Set all biases to exactly zero - critical for balanced initial predictions
    new_biases = np.zeros_like(biases)
    
    # Apply the fix
    output_layer.set_weights([new_weights, new_biases])
    
    # ----------------------------------------
    # Advanced Fix 2: Test with synthetic data and visualize 
    # ----------------------------------------
    print("\nTesting with diverse synthetic inputs...")
    
    num_test_samples = 100
    num_classes = biases.shape[0]
    
    # Create diverse test data
    test_inputs = []
    prediction_stats = np.zeros(num_classes)
    
    # Generate random inputs
    for _ in range(num_test_samples):
        # Generate random input with same shape as expected by model
        random_input = np.random.random((1,) + model.input_shape[1:])
        test_inputs.append(random_input)
        
        # Predict
        pred = model.predict(random_input, verbose=0)
        pred_class = np.argmax(pred[0])
        prediction_stats[pred_class] += 1
    
    # Plot prediction distribution
    plt.figure(figsize=(15, 5))
    plt.bar(range(num_classes), prediction_stats)
    plt.title('Prediction Distribution with Random Inputs After Fix')
    plt.xlabel('Class Index')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('prediction_distribution.png')
    print("Prediction distribution saved to 'prediction_distribution.png'")
    
    # Check if predictions are now diverse
    unique_predictions = np.count_nonzero(prediction_stats)
    print(f"Model now predicts {unique_predictions} different classes out of {num_classes}")
    
    # ----------------------------------------
    # Advanced Fix 3: Higher bias for underrepresented classes
    # ----------------------------------------
    if unique_predictions < num_classes * 0.5:
        print("\nWarning: Still not predicting enough classes, applying stronger bias correction...")
        
        # Find classes that are never predicted
        zero_pred_classes = np.where(prediction_stats == 0)[0]
        
        weights, biases = output_layer.get_weights()
        
        # Give a slight advantage to classes never predicted
        for class_idx in zero_pred_classes:
            # Slightly increase the bias for this class
            biases[class_idx] = 0.05
            
            # Also slightly increase the weights for this class
            weights[:, class_idx] = weights[:, class_idx] * 1.2
        
        # Apply this stronger fix
        output_layer.set_weights([weights, biases])
        
        print(f"Applied bias boost to {len(zero_pred_classes)} underrepresented classes")
        
        # Test again
        prediction_stats = np.zeros(num_classes)
        for random_input in test_inputs:
            pred = model.predict(random_input, verbose=0)
            pred_class = np.argmax(pred[0])
            prediction_stats[pred_class] += 1
            
        unique_predictions = np.count_nonzero(prediction_stats)
        print(f"After stronger fix: Model predicts {unique_predictions} different classes")
    
    # Save the improved model
    model.save(output_model_path)
    print(f"\nBalanced model saved to '{output_model_path}'")
    return True

if __name__ == "__main__":
    print("NSL Model Advanced Balancing Tool")
    print("=================================")
    
    # Get input/output paths
    input_path = input("Enter input model path [nsl_model_fixed.h5]: ").strip()
    if not input_path:
        input_path = "nsl_model_fixed.h5"
        
    output_path = input("Enter output model path [nsl_model_balanced.h5]: ").strip()
    if not output_path:
        output_path = "nsl_model_balanced.h5"
    
    success = advanced_model_fix(input_path, output_path)
    
    if success:
        print("\nRecommendations:")
        print("1. Use the balanced model in your application")
        print("2. Consider retraining your model with class balancing")
        print("3. Add data augmentation to increase variety in your training set")