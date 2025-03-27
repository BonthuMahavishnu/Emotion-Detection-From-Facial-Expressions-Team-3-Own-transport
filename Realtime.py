import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(r"C:\Users\Mahavishnu\OneDrive\Desktop\P_1\emotion_detection_model.h5")

# Define emotion categories
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))  # Resize to match input size
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)
        face = face / 255.0  # Normalize pixel values

        predictions = model.predict(face)
        emotion = emotion_labels[np.argmax(predictions)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Emotion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

cap.release()
cv2.destroyAllWindows() 