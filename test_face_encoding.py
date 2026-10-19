import cv2
import numpy as np
from face_recognition_module import encode_face

print("[TEST] Testing face encoding from webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open camera")
    exit(1)

success, frame = cap.read()
cap.release()

if not success:
    print("[ERROR] Cannot read frame")
    exit(1)

print("[INFO] Frame shape:", frame.shape)
enc = encode_face(frame)
if enc is not None:
    print("[SUCCESS] Face encoding works! Shape:", enc.shape)
else:
    print("[WARNING] No face detected in current frame")
