import os
import json
from typing import Optional
import numpy as np
from keras import layers, models

# Default EMOTIONS — overridden at runtime if models/class_indices.json exists
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

_face_cascade = None


def load_emotions_from_class_indices(path: Optional[str] = None) -> bool:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "models", "class_indices.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            idx_to_class = json.load(f)
        ordered = [idx_to_class[str(i)] for i in range(len(idx_to_class))]
        ordered = [c.capitalize() for c in ordered]
        EMOTIONS[:] = ordered
        print(f"[INFO] Loaded {len(ordered)} emotions from {path}: {ordered}")
        return True
    except Exception as e:
        print(f"[WARNING] Could not load class_indices.json: {e}")
        return False


def _get_cascade_detector():
    global _face_cascade
    if _face_cascade is not None:
        return _face_cascade
    try:
        import cv2
    except ImportError:
        return None
    cascade_path = None
    data_dir = getattr(cv2, "data", None)
    if data_dir is not None:
        cascade_path = os.path.join(getattr(data_dir, "haarcascades", ""),
                                    "haarcascade_frontalface_default.xml")
    if not cascade_path or not os.path.exists(cascade_path):
        cascade_path = os.path.join(os.path.dirname(cv2.__file__), "data",
                                    "haarcascades", "haarcascade_frontalface_default.xml")
    if os.path.exists(cascade_path):
        _face_cascade = cv2.CascadeClassifier(cascade_path)
        if not _face_cascade.empty():
            print(f"[INFO] Loaded face detector from {cascade_path}")
            return _face_cascade
    print("[WARN] Haar Cascade XML not found")
    return None


def detect_faces(image, min_size=(40, 40)):
    if image is None:
        return []

    import cv2
    
    # --- ATTEMPT 1: FAST HAAR CASCADE ---
    detector = _get_cascade_detector()
    if detector is not None:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(faces) > 0:
                return [(int(x), int(y), int(w), int(h)) for x, y, w, h in
                        sorted(faces, key=lambda r: r[2] * r[3], reverse=True)]
        except Exception as e:
            pass # Silently ignore and move to fallback

    # --- ATTEMPT 2: HIGHLY ACCURATE FACE_RECOGNITION (dlib) ---
    # If Haar Cascade fails (e.g. lighting issues, or OpenCV 5 bug), we use dlib
    try:
        import face_recognition
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Find faces using the HOG model
        locations = face_recognition.face_locations(rgb_image, model="hog")
        
        if locations:
            boxes = []
            for (top, right, bottom, left) in locations:
                w = right - left
                h = bottom - top
                boxes.append((left, top, w, h))
            # Return largest face first
            return sorted(boxes, key=lambda r: r[2] * r[3], reverse=True)
    except Exception as e:
        print(f"[WARN] Dlib face detection failed: {e}")

    return []


def build_simple_cnn(input_shape=(48, 48, 1), n_classes=7):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def preprocess_pixels(pixel_str):
    arr = np.fromstring(pixel_str, sep=' ', dtype=np.float32)
    arr = arr.reshape((48, 48))
    arr = arr / 255.0
    return arr


def preprocess_face(face_img, target_size=(224, 224)):
    import cv2
    rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, target_size)
    normalized = resized.astype('float32') / 255.0
    return np.expand_dims(normalized, axis=0)