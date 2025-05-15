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

# Webcam feed generator
def gen_frames():
    global latest_prediction, previous_predictions, frame_counter
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)
        frame_counter += 1
        h, w, _ = frame.shape
        display_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw only landmarks, no bounding box or text
                mp_draw.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Get bounding box for cropping only (not for drawing)
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                padding = 30
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                hand_img = frame[y_min:y_max, x_min:x_max]
                if hand_img.size != 0:
                    h_crop, w_crop, _ = hand_img.shape
                    size = max(h_crop, w_crop)
                    square_img = np.zeros((size, size, 3), dtype=np.uint8)
                    start_h = (size - h_crop) // 2
                    start_w = (size - w_crop) // 2
                    square_img[start_h:start_h+h_crop, start_w:start_w+w_crop] = hand_img
                    processed_img = preprocess_image_for_prediction(square_img)
                    processed_img = np.expand_dims(processed_img, axis=0)
                    prediction = model.predict(processed_img, verbose=0)
                    confidence = np.max(prediction)
                    label = np.argmax(prediction)
                    # No overlay text or bounding box here
                    if confidence > confidence_threshold:
                        previous_predictions.append(int(label))
                        if len(previous_predictions) > 10:
                            previous_predictions.pop(0)
                        if len(previous_predictions) >= prediction_threshold:
                            prediction_counts = {}
                            for pred in previous_predictions[-prediction_threshold:]:
                                if pred in prediction_counts:
                                    prediction_counts[pred] += 1
                                else:
                                    prediction_counts[pred] = 1
                            most_common = max(prediction_counts, key=prediction_counts.get)
                            stability = prediction_counts[most_common] / prediction_threshold
                            if stability >= 0.6:
                                if 0 <= most_common < len(nepali_chars):
                                    latest_prediction = {
                                        "label": int(most_common),
                                        "character": nepali_chars[most_common],
                                        "confidence": float(confidence),
                                        "stability": float(stability),
                                        "top_predictions": [
                                            {"char": nepali_chars[idx], "conf": float(prediction[0][idx])}
                                            for idx in np.argsort(prediction[0])[-3:][::-1]
                                        ]
                                    }
                    else:
                        latest_prediction = {
                            "character": "Waiting for clear gesture...",
                            "confidence": float(confidence),
                            "stability": 0.0
                        }
        else:
            latest_prediction = {
                "character": "No hand detected",
                "confidence": 0.0,
                "stability": 0.0
            }
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