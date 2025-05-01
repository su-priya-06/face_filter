import cv2
import numpy as np

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the hat filter with alpha channel
import os
filter_path = os.path.join(os.path.dirname(__file__), 'filters', 'orange_beanie.png')
filter_img = cv2.imread(filter_path, cv2.IMREAD_UNCHANGED)
print("[DEBUG] Loading from:", filter_path)

def overlay_filter(frame, filter_img, x, y, w, h):
    filter_width = w
    filter_height = int(h * 0.6)  # Taller for hats
    filter_resized = cv2.resize(filter_img, (filter_width, filter_height))

    y_offset = y - int(h * 0.55)  # Place the hat *above* the detected face

    for i in range(filter_height):
        for j in range(filter_width):
            if 0 <= y_offset + i < frame.shape[0] and 0 <= x + j < frame.shape[1]:
                alpha = filter_resized[i, j, 3] / 255.0  # Transparency
                for c in range(3):  # BGR channels
                    frame[y_offset + i, x + j, c] = (
                        alpha * filter_resized[i, j, c] + (1 - alpha) * frame[y_offset + i, x + j, c]
                    )
    return frame

# Open webcam
cap = cv2.VideoCapture(0)
print("[INFO] Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        frame = overlay_filter(frame, filter_img, x, y, w, h)

    cv2.imshow('Face Filter App 🎩', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

