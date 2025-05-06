import cv2
import numpy as np
import tensorflow as tf
import time
import os

def load_label_map():
    """Load the label mapping from file"""
    label_map = {}
    try:
        with open("nsl_label_map.txt", "r") as f:
            for line in f:
                idx, label = line.strip().split(": ", 1)
                label_map[int(idx)] = label
    except FileNotFoundError:
        print("Warning: nsl_label_map.txt not found. Using numeric indices as labels.")
        # Create a default label map with numbers 0-35
        label_map = {i: str(i) for i in range(36)}
    return label_map

def preprocess_frame(frame, target_size=(64, 64)):
    """
    Preprocess a webcam frame for prediction
    
    Args:
        frame: Input frame from webcam
        target_size: Target size for the model input
    
    Returns:
        Preprocessed frame ready for model prediction
    """
    # Convert to RGB (from BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize to target size
    resized_frame = cv2.resize(rgb_frame, target_size)
    
    # Normalize pixel values to [0, 1] - IMPORTANT: match training normalization
    normalized_frame = resized_frame / 255.0
    
    # Add batch dimension
    input_tensor = np.expand_dims(normalized_frame, axis=0)
    
    return input_tensor

def main():
    # Check if model exists
    if not os.path.exists("nsl_model.h5"):
        print("Error: Model file 'nsl_model.h5' not found!")
        return
    
    # Load the trained model
    print("Loading model...")
    model = tf.keras.models.load_model("nsl_model.h5")
    print("Model loaded successfully!")
    
    # Load label mapping
    label_map = load_label_map()
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam!")
        return
    
    print("Ready! Press 'q' to quit.")
    
    # Create a named window
    cv2.namedWindow('NSL Recognition', cv2.WINDOW_NORMAL)
    
    # For FPS calculation
    prev_frame_time = 0
    new_frame_time = 0
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame from webcam!")
            break
        
        # Calculate FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        fps_text = f"FPS: {int(fps)}"
        
        # Create display frame (original frame with info overlay)
        display_frame = frame.copy()
        
        # Preprocess the frame
        input_tensor = preprocess_frame(frame)
        
        # Make prediction
        prediction = model.predict(input_tensor, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        # Map class index to label
        predicted_label = label_map.get(predicted_class, str(predicted_class))
        
        # Add prediction and FPS info to display frame
        cv2.putText(display_frame, f"Prediction: {predicted_label}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Confidence: {confidence:.2f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, fps_text, (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('NSL Recognition', display_frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Release webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()