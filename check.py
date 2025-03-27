import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import pyttsx3
import time

# Load the trained model
model = load_model("emotion_detection_model.h5")

# Define emotion categories
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speech rate for faster response

# OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Streamlit UI
st.set_page_config(page_title="Emotion Detection", layout="wide")
st.title("Real-Time Emotion Detection")
st.write("Live camera feed is running...")

# Start webcam
cap = cv2.VideoCapture(0)

# Streamlit live camera feed
FRAME_WINDOW = st.image([])
last_speech_time = time.time()
frame_skip = 5  # Only predict every 5 frames to reduce lag
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if frame_count % frame_skip == 0:  # Reduce model calls for faster performance
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)
            face = face / 255.0  # Normalize

            predictions = model.predict(face)
            emotion = emotion_labels[np.argmax(predictions)]

            # Display emotion label
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Speak emotion only once every 2 seconds to avoid overlaps
            if time.time() - last_speech_time > 2:
                engine.say(emotion)
                engine.runAndWait()
                last_speech_time = time.time()

    # Convert frame for Streamlit
    _, buffer = cv2.imencode('.jpg', frame)
    FRAME_WINDOW.image(buffer.tobytes(), channels="BGR", use_column_width=True)

    frame_count += 1  # Increase frame counter

cap.release()
cv2.destroyAllWindows()
