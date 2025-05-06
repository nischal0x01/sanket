import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify, request
import os
import time

app = Flask(__name__)

# Load the trained model
model_path = "nsl_model_fixed.h5"
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found!")
    model = None
else:
    model = tf.keras.models.load_model(model_path)
    
    # Fix the model if needed by resetting the output layer
    last_layer = model.layers[-1]
    weights, biases = last_layer.get_weights()
    if np.std(biases) > 0.05 or np.max(biases) > 0.1:
        print("Detected possible bias in output layer. Applying fix...")
        new_weights = np.random.normal(0, 0.01, size=weights.shape)
        new_biases = np.zeros_like(biases)
        last_layer.set_weights([new_weights, new_biases])
        print("Model output layer reset to prevent class dominance")

# Nepali character mapping (ensure this matches your training labels)
nepali_chars = ["क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ", "ट", "ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श", "ष", "स", "ह", "क्ष", "त्र", "ज्ञ"]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Global variables
latest_prediction = None
prediction_history = []  # To track prediction stability
last_prediction_time = 0
prediction_cooldown = 0.5  # seconds between predictions to avoid flickering

# Create directory for debug images if it doesn't exist
debug_dir = "debug_frames"
os.makedirs(debug_dir, exist_ok=True)
debug_enabled = False

def get_hand_roi(frame, hand_landmarks):
    """Extract ROI around hand with proper padding and aspect ratio"""
    h, w, _ = frame.shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    
    # Get bounding box around hand landmarks
    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)
    
    # Calculate center and size
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    # Get the larger dimension for a square crop
    half_size = max(x_max - x_min, y_max - y_min) // 2
    # Add padding (40% of hand size)
    half_size = int(half_size * 1.4)
    
    # Calculate new bounding box
    x_min = max(0, center_x - half_size)
    y_min = max(0, center_y - half_size)
    x_max = min(w, center_x + half_size)
    y_max = min(h, center_y + half_size)
    
    # Extract and return the ROI
    hand_roi = frame[y_min:y_max, x_min:x_max]
    return hand_roi, (x_min, y_min, x_max, y_max)

def preprocess_hand_image(hand_img):
    """Preprocess hand image for model prediction"""
    if hand_img.size == 0:
        return None
    
    # Convert to RGB if needed
    if len(hand_img.shape) == 3 and hand_img.shape[2] == 3:
        hand_img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    else:
        return None
    
    # Resize to model input size
    hand_img_resized = cv2.resize(hand_img_rgb, (64, 64))
    
    # Normalize pixel values
    hand_img_normalized = hand_img_resized / 255.0
    
    # Add batch dimension
    hand_img_batch = np.expand_dims(hand_img_normalized, axis=0)
    
    return hand_img_batch

def get_stable_prediction(new_pred):
    """Get stable prediction using history to avoid flickering"""
    global prediction_history
    
    # Add new prediction to history
    prediction_history.append(new_pred)
    
    # Keep only last 5 predictions
    if len(prediction_history) > 5:
        prediction_history.pop(0)
    
    # Count occurrences of each class
    counts = {}
    for pred in prediction_history:
        if pred in counts:
            counts[pred] += 1
        else:
            counts[pred] = 1
    
    # Find most common prediction
    max_count = 0
    stable_pred = new_pred
    
    for pred, count in counts.items():
        if count > max_count:
            max_count = count
            stable_pred = pred
    
    return stable_pred

# Webcam feed generator
def gen_frames():
    global latest_prediction, last_prediction_time
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_count += 1
        
        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        current_time = time.time()
        prediction_ready = current_time - last_prediction_time >= prediction_cooldown
        
        # If hands are detected, process for prediction
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get and process hand ROI
                hand_img, roi_coords = get_hand_roi(frame, hand_landmarks)
                
                # Draw ROI rectangle
                x_min, y_min, x_max, y_max = roi_coords
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                if prediction_ready and model is not None:
                    # Preprocess the hand image
                    processed_img = preprocess_hand_image(hand_img)
                    
                    if processed_img is not None:
                        # Save debug frame if enabled
                        if debug_enabled and frame_count % 30 == 0:
                            debug_path = os.path.join(debug_dir, f"hand_{int(time.time())}.jpg")
                            cv2.imwrite(debug_path, hand_img)
                        
                        # Make prediction
                        prediction = model.predict(processed_img, verbose=0)
                        
                        # Get predicted class and confidence
                        predicted_class = np.argmax(prediction[0])
                        confidence = float(prediction[0][predicted_class])
                        
                        # Get stable prediction
                        stable_class = get_stable_prediction(predicted_class)
                        
                        # Only update if confidence is reasonable
                        if confidence > 0.3:
                            # Update latest prediction
                            label = int(stable_class)
                            
                            if label >= 0 and label < len(nepali_chars):
                                latest_prediction = {
                                    "label": label,
                                    "character": nepali_chars[label],
                                    "confidence": float(confidence),
                                    "top_predictions": [
                                        {"label": int(i), 
                                         "character": nepali_chars[i] if i < len(nepali_chars) else "Unknown",
                                         "confidence": float(prediction[0][i])}
                                        for i in np.argsort(prediction[0])[-3:][::-1]
                                    ]
                                }
                                last_prediction_time = current_time
                                
                                # Show prediction on frame
                                cv2.putText(
                                    frame, 
                                    f"{nepali_chars[label]} ({confidence:.2f})", 
                                    (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, 
                                    (0, 255, 0), 
                                    2
                                )

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
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

@app.route('/toggle_debug')
def toggle_debug():
    global debug_enabled
    debug_enabled = not debug_enabled
    return jsonify({"debug_enabled": debug_enabled})

@app.route('/reset_model')
def reset_model():
    """Endpoint to reset model weights if needed"""
    global model
    if model is not None:
        # Reset the output layer
        last_layer = model.layers[-1]
        weights, biases = last_layer.get_weights()
        new_weights = np.random.normal(0, 0.01, size=weights.shape)
        new_biases = np.zeros_like(biases)
        last_layer.set_weights([new_weights, new_biases])
        return jsonify({"status": "Model reset successfully"})
    return jsonify({"error": "Model not loaded"})

if __name__ == '__main__':
    app.run(debug=True)