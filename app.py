import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("nsl_model.h5")

# Nepali character mapping
nepali_chars = ["क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ", "ट", "ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श", "ष", "स", "ह", "क्ष", "त्र", "ज्ञ"]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Global variable to store the latest prediction
latest_prediction = None

# Webcam feed generator
def gen_frames():
    global latest_prediction
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # If hands are detected, crop the hand region
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get bounding box around the hand
                h, w, _ = frame.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                # Add padding to the bounding box
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)

                # Crop and preprocess the hand region
                hand_img = frame[y_min:y_max, x_min:x_max]
                if hand_img.size != 0:
                    hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                    hand_img = cv2.resize(hand_img, (64, 64))
                    hand_img = hand_img / 255.0
                    hand_img = np.expand_dims(hand_img, axis=0)

                    # Predict
                    prediction = model.predict(hand_img)
                    label = np.argmax(prediction)
                    print(f"Predicted Label: {label}")
                    label = int(label)  # Convert to native Python int
                    if label < 0 or label >= len(nepali_chars):
                        latest_prediction = {"label": label, "character": "Invalid Label"}
                    else:
                        latest_prediction = {"label": label, "character": nepali_chars[label]}

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

if __name__ == '__main__':
    app.run(debug=True)