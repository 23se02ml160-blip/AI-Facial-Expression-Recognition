import os
import numpy as np
import pickle
import cv2
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import face_recognition
except ImportError:
    print("[FATAL] face_recognition library not installed.")
    face_recognition = None


def encode_face(image_array) -> Optional[np.ndarray]:
    if face_recognition is None or image_array is None:
        return None
    if not hasattr(image_array, 'shape') or len(image_array.shape) < 2:
        return None
    try:
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        elif len(image_array.shape) == 2:
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = image_array
        
        rgb_image = cv2.resize(rgb_image, (150, 150))
        encodings = face_recognition.face_encodings(rgb_image)
        
        if len(encodings) > 0:
            return encodings[0]
        return None
    except Exception as e:
        print(f"[ERROR] encode_face failed: {e}")
        return None


def encoding_to_bytes(encoding: np.ndarray) -> Optional[bytes]:
    if encoding is None:
        return None
    try:
        return pickle.dumps(encoding)
    except Exception:
        return None


def bytes_to_encoding(data: bytes) -> Optional[np.ndarray]:
    if data is None:
        return None
    try:
        result = pickle.loads(data)
        if isinstance(result, np.ndarray):
            return result
        return None
    except Exception:
        return None


def compare_faces(encoding1: np.ndarray, encoding2: np.ndarray, tolerance: float = 0.6) -> Tuple[bool, float]:
    try:
        if encoding1 is None or encoding2 is None:
            return False, 1.0
        if encoding1.shape[0] != 128 or encoding2.shape[0] != 128:
            return False, 1.0
        
        distance = np.linalg.norm(encoding1 - encoding2)
        is_match = bool(distance < tolerance)
        return is_match, float(distance)
    except Exception:
        return False, 1.0


def find_matching_person(incoming_encoding: np.ndarray, stored_encodings_dict: dict, tolerance: float = 0.6) -> Tuple[Optional[str], float]:
    if incoming_encoding is None or not isinstance(incoming_encoding, np.ndarray):
        return None, 0.0
    if incoming_encoding.shape[0] != 128 or not stored_encodings_dict:
        return None, 0.0
    
    best_match = None
    best_distance = float('inf')
    
    for person_name, encodings_list in stored_encodings_dict.items():
        for stored_encoding in encodings_list:
            if stored_encoding is None or not isinstance(stored_encoding, np.ndarray):
                continue
            if stored_encoding.shape[0] != 128:
                continue
                
            is_match, distance = compare_faces(incoming_encoding, stored_encoding, tolerance)
            
            if distance < best_distance:
                best_distance = distance
                if is_match:
                    best_match = person_name
    
    if best_match is not None:
        # Calculate percentage: distance 0.0 = 100%, distance 0.6 = 0%
        confidence = max(0, min(100, (1 - (best_distance / 0.6)) * 100))
        return best_match, round(confidence, 2)
    
    return None, 0.0