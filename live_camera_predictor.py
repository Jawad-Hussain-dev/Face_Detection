from collections import deque, Counter
import cv2
import joblib
import numpy as np
from feature_extraction import extract_hog_features
from skimage.feature import hog


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

model = joblib.load("tree_model.pkl")
_, _, label_map = joblib.load("hog_dataset.pkl")
inverse_label_map = {v: k for k, v in label_map.items()}

cap = cv2.VideoCapture(0)
label_buffer = deque(maxlen=15)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

        mouth = mouth_cascade.detectMultiScale(roi_gray)
        for (mx, my, mw, mh) in mouth:
            if my > h // 2:  
                cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 255, 255), 1)
                break

        try:
            face_resized = cv2.resize(roi_gray, (256, 256))
            hog_features = hog(face_resized,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys',
                   visualize=False,
                   feature_vector=True)
            hog_features = hog_features.reshape(1, -1)
            prediction = model.predict(hog_features)[0]
            name = inverse_label_map.get(prediction, "Unknown")
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        except:
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Live Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
