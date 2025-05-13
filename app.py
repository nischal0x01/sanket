import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify
import time
import os

app = Flask(__name__)

# Create debug directory
debug_dir = "debug_frames"
os.makedirs(debug_dir, exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model("nsl_model.h5")
print(f"Model loaded successfully. Input shape: {model.input_shape}")

# Nepali character mapping
nepali_chars = ["क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ", "ट", "ठ", "ड", "ढ", "ण", 
                "त", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श", 
                "ष", "स", "ह", "क्ष", "त्र", "ज्ञ"]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Global variables
latest_prediction = None
previous_predictions = []
prediction_threshold = 3  # Reduced from 5 to 3 for faster response
confidence_threshold = 0.5  # Reduced from 0.7 to see more predictions

# Frame counter for debugging
frame_counter = 0

# Debug mode - save problematic frames
debug_mode = True
debug_frame_interval = 20  # Save every 20th frame for analysis

def preprocess_image_for_prediction(image):
    """
    Preprocess image consistently with training data
    """
    # Resize to model's expected input size
    resized = cv2.resize(image, (64, 64))  # Ensure this matches training resolution
    
    # Convert to RGB (if not already)
    if len(image.shape) == 2 or image.shape[2] == 1:
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    else:
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0,1] range - matching training preprocessing
    normalized = rgb_image / 255.0  # Ensure this matches training normalization
    
    return normalized

def save_debug_info(frame, processed_img, prediction, label, confidence, frame_num):
    """
    Save debug information to analyze prediction issues
    """
    if not debug_mode:
        return
    
    # Create a debug frame
    debug_path = os.path.join(debug_dir, f"frame_{frame_num:04d}_pred_{nepali_chars[label]}_conf_{confidence:.2f}.jpg")
    
    # Save original frame
    cv2.imwrite(debug_path, frame)
    
    # Save processed input to model
    processed_debug = (processed_img[0] * 255).astype(np.uint8)  # Convert back to 0-255 range
    processed_path = os.path.join(debug_dir, f"frame_{frame_num:04d}_processed.jpg")
    cv2.imwrite(processed_path, cv2.cvtColor(processed_debug, cv2.COLOR_RGB2BGR))
    
    # Save prediction distribution
    with open(os.path.join(debug_dir, f"frame_{frame_num:04d}_predictions.txt"), "w") as f:
        f.write(f"Predicted Label: {nepali_chars[label]} (index {label})\n")
        f.write(f"Confidence: {confidence:.4f}\n\n")
        f.write("All Predictions:\n")
        for i, prob in enumerate(prediction[0]):
            f.write(f"{i}: {nepali_chars[i]} = {prob:.4f}\n")

# Webcam feed generator
def gen_frames():
    global latest_prediction, previous_predictions, frame_counter
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
            
        # Mirror frame for more intuitive interaction
        frame = cv2.flip(frame, 1)
        frame_counter += 1
        
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
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
                
                # Crop and preprocess the hand region
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
                    
                    # Process for model prediction - using consistent preprocessing
                    processed_img = preprocess_image_for_prediction(square_img)
                    processed_img = np.expand_dims(processed_img, axis=0)
                    
                    # Predict
                    prediction = model.predict(processed_img, verbose=0)
                    confidence = np.max(prediction)
                    label = np.argmax(prediction)
                    
                    # Save debug information periodically
                    if frame_counter % debug_frame_interval == 0:
                        save_debug_info(frame, processed_img, prediction, label, confidence, frame_counter)
                    
                    # Display confidence on frame
                    cv2.putText(display_frame, f"Confidence: {confidence:.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display top 3 predictions
                    top_indices = np.argsort(prediction[0])[-3:][::-1]
                    for i, idx in enumerate(top_indices):
                        pred_text = f"{i+1}. {nepali_chars[idx]}: {prediction[0][idx]:.3f}"
                        cv2.putText(display_frame, pred_text, (10, 60 + 30*i), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    # Check for "घ" bias
                    if 3 in top_indices[:2]:  # If "घ" is in top 2 predictions
                        cv2.putText(display_frame, "'घ' (gha) bias detected!", (10, 150), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Only accept prediction if confidence is high enough
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
                                    latest_prediction = {
                                        "label": int(most_common), 
                                        "character": nepali_chars[most_common],
                                        "confidence": float(confidence),
                                        "stability": float(stability),
                                        "top_predictions": [
                                            {"char": nepali_chars[idx], "conf": float(prediction[0][idx])} 
                                            for idx in top_indices
                                        ]
                                    }
                                    
                                    # Display on frame
                                    char_text = f"Character: {nepali_chars[most_common]}"
                                    cv2.putText(display_frame, char_text, (10, 180), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        # Low confidence
                        latest_prediction = {
                            "character": "Waiting for clear gesture...",
                            "confidence": float(confidence),
                            "stability": 0.0
                        }
        else:
            # No hand detected
            latest_prediction = {
                "character": "No hand detected",
                "confidence": 0.0,
                "stability": 0.0
            }
            
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', display_frame)
        if not ret:
            print("Error: Could not encode frame.")
            continue
            
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict')
def predict():
    global latest_prediction
    if latest_prediction is None:
        return jsonify({"error": "No prediction available"})
    return jsonify(latest_prediction)

@app.route('/debug')
def debug_info():
    """Return debug information about the model and prediction process"""
    model_info = {
        "input_shape": model.input_shape[1:],
        "output_shape": model.output_shape[1:],
        "num_classes": len(nepali_chars),
        "confidence_threshold": confidence_threshold,
        "prediction_threshold": prediction_threshold,
        "debug_mode": debug_mode,
        "debug_dir": debug_dir
    }
    return jsonify(model_info)

if __name__ == '__main__':
    # Print model summary before starting
    print("Model Summary:")
    model.summary()
    
    app.run(debug=True)