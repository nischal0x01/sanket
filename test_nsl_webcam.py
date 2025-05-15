import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
import time
import os

# Load the trained model
model = tf.keras.models.load_model("nsl_model.h5")
print(f"Model loaded successfully. Input shape: {model.input_shape}")

# Nepali character mapping
nepali_chars = ["क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ", "ट", "ठ", "ड", "ढ", "ण", 
                "त", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श", 
                "ष", "स", "ह", "क्ष", "त्र", "ज्ञ"]

# Create debug directory
debug_dir = "debug_frames"
os.makedirs(debug_dir, exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Parameters for prediction stability
previous_predictions = []
prediction_threshold = 3  # Reduced from 5 to 3 for faster response
confidence_threshold = 0.5  # Reduced from 0.7 to see more predictions

# Frame counter for debugging
frame_counter = 0

# Debug mode - save problematic frames
debug_mode = False  # Disable debug frame saving
debug_frame_interval = 20  # (value doesn't matter if debug_mode is False)

def preprocess_image_for_prediction(image):
    """
    Preprocess image consistently with training and inference
    """
    # Resize to model's expected input size
    resized = cv2.resize(image, (64, 64))
    
    # Convert to RGB (if not already)
    if len(image.shape) == 2 or image.shape[2] == 1:
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    else:
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0,1] range
    normalized = rgb_image / 255.0
    
    return normalized

def save_debug_info(frame, processed_img, prediction, label, confidence, frame_num):
    """
    Save debug information to analyze prediction issues
    """
    if not debug_mode:
        return

def main():
    global frame_counter
    
    # Display model summary
    print("Model Summary:")
    model.summary()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Create output window
    cv2.namedWindow("NSL Recognition", cv2.WINDOW_NORMAL)
    
    # Open log file
    with open("predictions.txt", "w") as log_file:
        log_file.write("NSL Recognition Log\n")
        log_file.write("Time, Label Index, Character, Confidence\n")
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            # Mirror frame for more intuitive interaction
            frame = cv2.flip(frame, 1)
            frame_counter += 1
            
            # Get frame dimensions - FIXED: defined here to be available in all code paths
            h, w, _ = frame.shape
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Convert frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            # Current time for logging
            current_time = time.strftime("%H:%M:%S")
            
            # Draw help text
            cv2.putText(display_frame, "Press 'q' to quit, 's' to save frame", (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # If hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_draw.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Get bounding box around the hand
                    x_min, y_min = w, h
                    x_max, y_max = 0, 0
                    
                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x_max, x)
                        y_max = max(y_max, y)
                    
                    # Add padding to the bounding box
                    padding = 30
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)
                    
                    # Draw bounding box
                    cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Crop the hand region
                    hand_img = frame[y_min:y_max, x_min:x_max]
                    
                    if hand_img.size != 0:
                        # Ensure consistent aspect ratio (square bounding box)
                        h_crop, w_crop, _ = hand_img.shape
                        size = max(h_crop, w_crop)
                        square_img = np.zeros((size, size, 3), dtype=np.uint8)
                        
                        # Center the hand in the square image
                        start_h = (size - h_crop) // 2
                        start_w = (size - w_crop) // 2
                        square_img[start_h:start_h+h_crop, start_w:start_w+w_crop] = hand_img
                        
                        # Display preprocessed hand
                        display_hand = cv2.resize(square_img, (128, 128))
                        display_frame[10:138, w-138:w-10] = display_hand
                        cv2.rectangle(display_frame, (w-138, 10), (w-10, 138), (0, 0, 255), 2)
                        cv2.putText(display_frame, "Input", (w-138, 160), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        
                        # Process for model prediction - using consistent preprocessing
                        processed_img = preprocess_image_for_prediction(square_img)
                        processed_img = np.expand_dims(processed_img, axis=0)
                        
                        # Save processed input visualization
                        processed_viz = (processed_img[0] * 255).astype(np.uint8)
                        processed_viz = cv2.resize(processed_viz, (128, 128))
                        processed_viz = cv2.cvtColor(processed_viz, cv2.COLOR_RGB2BGR)
                        display_frame[10:138, w-276:w-148] = processed_viz
                        cv2.rectangle(display_frame, (w-276, 10), (w-148, 138), (255, 0, 0), 2)
                        cv2.putText(display_frame, "Processed", (w-276, 160), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        
                        # Predict
                        prediction = model.predict(processed_img, verbose=0)
                        confidence = np.max(prediction)
                        label = np.argmax(prediction)
                        
                        # Display confidence on frame
                        cv2.putText(display_frame, f"Confidence: {confidence:.2f}", (10, 55), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Display all predictions (not just top 3)
                        top_indices = np.argsort(prediction[0])[-5:][::-1]
                        for i, idx in enumerate(top_indices):
                            pred_text = f"{i+1}. {nepali_chars[idx]}: {prediction[0][idx]:.3f}"
                            cv2.putText(display_frame, pred_text, (10, 90 + 30*i), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        
                        # Process prediction stability with lower threshold
                        if confidence > confidence_threshold:
                            # Add current prediction to history
                            previous_predictions.append(int(label))
                            
                            # Keep only recent predictions
                            if len(previous_predictions) > 10:
                                previous_predictions.pop(0)
                            
                            # Check if we have enough consistent predictions
                            if len(previous_predictions) >= prediction_threshold:
                                # Find most common prediction in recent history
                                prediction_counts = {}
                                for pred in previous_predictions[-prediction_threshold:]:
                                    if pred in prediction_counts:
                                        prediction_counts[pred] += 1
                                    else:
                                        prediction_counts[pred] = 1
                                
                                most_common = max(prediction_counts, key=prediction_counts.get)
                                stability = prediction_counts[most_common] / prediction_threshold
                                
                                # Only update if prediction is stable
                                if stability >= 0.6:  # Reduced from 0.8 to 0.6 for testing
                                    if 0 <= most_common < len(nepali_chars):
                                        # Display stable prediction
                                        char_text = f"Recognized: {nepali_chars[most_common]}"
                                        cv2.putText(display_frame, char_text, (10, h - 30), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                                        
                                        # Log prediction
                                        log_entry = f"{current_time}, {most_common}, {nepali_chars[most_common]}, {confidence:.4f}\n"
                                        log_file.write(log_entry)
                                        log_file.flush()
                        else:
                            # Show waiting message if confidence is low
                            cv2.putText(display_frame, "Waiting for clear gesture...", (10, h - 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 140, 255), 2)
            else:
                # No hand detected - FIXED: h is now defined at the beginning of the loop
                cv2.putText(display_frame, "No hand detected", (10, h - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow("NSL Recognition", display_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame for debugging
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(debug_dir, f"manual_save_{timestamp}.jpg")
                cv2.imwrite(save_path, frame)
                print(f"Frame saved to {save_path}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()