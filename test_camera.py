import cv2
from utils import detect_faces, preprocess_face
from database import get_all_face_encodings
from face_recognition_module import encode_face, bytes_to_encoding, find_matching_person

print("=" * 60)
print("CAMERA + FACE DETECTION TEST")
print("=" * 60)

# 1. Check encodings in DB
encodings = get_all_face_encodings()
print(f"\n[1] Encodings in DB:")
for user, encs in encodings.items():
    print(f"    {user}: {len(encs)}")

# 2. Load encodings
print(f"\n[2] Loading encodings...")
family_encodings = {}
for user, enc_list in encodings.items():
    family_encodings[user] = []
    for e in enc_list:
        decoded = bytes_to_encoding(e)
        if decoded is not None:
            family_encodings[user].append(decoded)
print(f"    Total people loaded: {len(family_encodings)}")

# 3. Open camera
print(f"\n[3] Opening camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("    ❌ Camera not opened!")
    exit()

print(f"    ✅ Camera opened")
print(f"    Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

# 4. Read frames and detect
print(f"\n[4] Reading 10 frames and detecting faces...")
for i in range(10):
    ret, frame = cap.read()
    if not ret:
        print(f"    Frame {i+1}: ❌ Read failed")
        continue
    
    faces = detect_faces(frame)
    print(f"    Frame {i+1}: {len(faces)} face(s) detected")
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_crop = frame[y:y+h, x:x+w]
        encoding = encode_face(face_crop)
        
        if encoding is not None:
            print(f"        Encoding shape: {encoding.shape}")
            person, conf = find_matching_person(encoding, family_encodings, tolerance=0.65)
            print(f"        Recognized: '{person}' conf={conf:.3f}")
        else:
            print(f"        ❌ Failed to encode face")
    
    cv2.waitKey(100)

cap.release()
cv2.destroyAllWindows()
print(f"\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)