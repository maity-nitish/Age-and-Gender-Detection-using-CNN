import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import cv2
import time

gender_model = load_model("gender_model.keras")
age_model = load_model("age_model.keras")

GENDER_LABELS = ['Male', 'Female']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def class_to_range(class_index):
    start_age = 1 + (class_index * 5)
    end_age = start_age + 4
    return f"{start_age}-{end_age}"

cap = cv2.VideoCapture(0)
last_prediction_time = 0
last_predictions = []  

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        current_time = time.time()
        if current_time - last_prediction_time >= 0.2:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            last_predictions = []  
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face_img, (200, 200))
                face_input = face_resized.astype("float32") / 255.0
                face_input = np.expand_dims(face_input, axis=0)
                gender_pred = gender_model.predict(face_input)[0][0]
                gender = GENDER_LABELS[int(gender_pred < 0.5)]

                age_pred = age_model.predict(face_input)[0]
                age_class = np.argmax(age_pred)
                age_range = class_to_range(age_class)
                last_predictions.append(((x, y, w, h), f"{gender}, Age: {age_range}"))

            last_prediction_time = current_time

        for (x, y, w, h), label in last_predictions:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Live Age & Gender Prediction", frame)
        key = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty("Live Age & Gender Prediction", cv2.WND_PROP_VISIBLE) < 1:
            print("Exiting...")
            break

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed.")




