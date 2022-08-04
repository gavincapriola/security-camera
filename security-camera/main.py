import datetime

import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")


class HomeSecuritySystem:
    def __init__(self):
        self.current_time = datetime.datetime.now().strftime("%d-%m-%Y")
        self.video_capture = cv2.VideoCapture(0)
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    def record(self):
        out = cv2.VideoWriter(
            f"{self.current_time}.mp4", self.fourcc, 24.0, (640, 480))
        while True:
            _, frame = self.video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                face_roi = frame[y:y + h, x:x + w]  # region of interest
                eyes = eye_cascade.detectMultiScale(face_roi)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(face_roi, (ex, ey),
                                  (ex + ew, ey + eh), (0, 255, 0), 2)

            if len(faces) > 0:
                out.write(frame)
                if len(faces) == 0:
                    out.release()

            cv2.imshow("Security Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_capture.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    home_security_system = HomeSecuritySystem()
    home_security_system.record()
