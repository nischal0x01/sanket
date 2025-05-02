import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("nsl_model.h5")

# Nepali character mapping
nepali_chars = ["क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ", "ट", "ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श", "ष", "स", "ह", "क्ष", "त्र", "ज्ञ"]

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Log file
with open("predictions.txt", "w") as f:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Preprocess the frame
        img = cv2.resize(frame, (64, 64))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = model.predict(img)
        label = np.argmax(prediction)
        nepali_char = nepali_chars[label]

        # Log the prediction
        print(f"Predicted Character: {nepali_char}")
        f.write(f"Predicted Character: {nepali_char}\n")
        f.flush()

        # Display prediction on the frame
        cv2.putText(frame, f"Character: {nepali_char}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("NSL Detection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()