import datetime
import os

import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")

if not os.path.exists('detections'):
    os.mkdir('detections')  # create directory if it doesn't exist

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        face_roi = frame[y:y + h, x:x + w]  # region of interest
        eyes = eye_cascade.detectMultiScale(face_roi)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_roi, (ex, ey),
                          (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
