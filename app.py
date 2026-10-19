import os
import sys
import csv
import sqlite3
import shutil
import re
import json
import time
import threading
import base64
import io
from PIL import Image
from io import StringIO, BytesIO
from functools import wraps
from flask import make_response
from typing import Any, Tuple
import smtplib
import random
import string
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
load_dotenv()
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from camera_manager import camera_manager

CAMERA_RUNNING = False

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import cv2
import numpy as np

# ==========================
# Performance Cache
# ==========================
FRAME_COUNT = 0
LAST_EMOTION = "Detecting..."
LAST_PERSON = "Unknown Person"
LAST_CONFIDENCE = 0.0
LAST_EMOTION_TIME = 0
LAST_FACE_TIME = 0

_frame_counter = 0
_last_faces = []
_face_recognition_cache = {}

os.environ.setdefault('SKIP_DEEPFACE', '1')

from flask import Flask, render_template, Response, request, redirect, url_for, session
try:
    from tensorflow.keras.models import load_model
except Exception:
    try:
        from keras.models import load_model
    except Exception as exc:
        print("[ERROR] Could not import keras or tensorflow.keras:", exc)
        raise

import utils as _utils_module
from utils import preprocess_face, detect_faces
from person_data import find_person, register_person as pd_register_person, load_people
from database import (
    DB_FILE,
    ensure_db_exists,
    register_person as db_register_person,
    delete_person as db_delete_person,
    log_face_detection,
    get_face_detection_logs,
    get_person_logs,
    find_person_by_username,
    store_face_encoding,
    get_face_encodings_for_person,
    get_all_face_encodings,
    clear_face_encodings_for_person,
    get_total_people,
    get_total_face_encodings,
    get_total_logs,
    get_today_detection_count,
    get_unique_people_detected,
    get_logs_by_date,
    get_recent_logs,
    verify_user_password,
    update_user_password,
    create_user,
    is_user_admin,
    user_exists_in_users,
)
from face_recognition_module import (
    encode_face,
    encoding_to_bytes,
    bytes_to_encoding,
    find_matching_person,
)

BASE_PATH = getattr(sys, '_MEIPASS', os.path.abspath("."))
CAM_INDEX = 0

_MODEL_CANDIDATES = [
    os.path.join(BASE_PATH, "models", "fer_model.h5"),
    os.path.join(BASE_PATH, "models", "fer_model_local.h5"),
    os.path.join(BASE_PATH, "models", "fer_model_fast.h5"),
]
MODEL_PATH = next((p for p in _MODEL_CANDIDATES if os.path.exists(p)), _MODEL_CANDIDATES[0])

EMAIL_ADDRESS = os.getenv("aifacialexpressionrecognition@gmail.com", "")
EMAIL_PASSWORD = os.getenv("gikh crlt ewfy cuwf", "")


def send_otp_email(receiver_email, otp):
    subject = "Your OTP for AI Facial Expression Recognition"
    body = f"Hello, your OTP is: {otp}. It is valid for 5 minutes."

    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("[WARNING] EMAIL_ADDRESS/EMAIL_PASSWORD not set in .env")
        return False

    try:
        server = smtplib.SMTP('smtp.gmail.com', 465)
        
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"[SUCCESS] Real OTP sent to {receiver_email}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
        return False


app = Flask(
    __name__,
    template_folder=os.path.join(BASE_PATH, "Templates"),
    static_folder=os.path.join(BASE_PATH, "static")
)

app.secret_key = os.getenv("SECRET_KEY") or os.urandom(32).hex()
if not os.getenv("SECRET_KEY"):
    print("[WARNING] SECRET_KEY not set in .env — using random key (sessions reset on restart).")


def login_required(f):
    @wraps(f)
    def _wrapped(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return _wrapped


def admin_required(f):
    @wraps(f)
    def _wrapped(*args, **kwargs):
        if "user" not in session or not session.get('is_admin'):
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return _wrapped


_login_attempts = {}
LOGIN_WINDOW_SECONDS = 300
MAX_LOGIN_ATTEMPTS = 10


def _login_key():
    return (request.remote_addr or "unknown") + "|" + request.form.get('username', '')


def _login_rate_limited(key):
    now = time.time()
    attempts = [t for t in _login_attempts.get(key, []) if now - t < LOGIN_WINDOW_SECONDS]
    _login_attempts[key] = attempts
    return len(attempts) >= MAX_LOGIN_ATTEMPTS


def _record_login_attempt(key):
    _login_attempts.setdefault(key, []).append(time.time())


def load_emotion_model(path: str) -> Any:
    if os.path.exists(path):
        try:
            model = load_model(path)
            print("[INFO] Loaded emotion model:", path)
            return model
        except Exception as exc:
            print("[ERROR] Failed to load model:", exc)
    else:
        print("[WARNING] Model path not found:", path)
    return None


emotion_model = load_emotion_model(MODEL_PATH)

# Detect model input size dynamically
_MODEL_INPUT_SIZE = (224, 224)
if emotion_model is not None:
    try:
        shape = emotion_model.input_shape
        if shape and len(shape) >= 3 and shape[1] and shape[2]:
            _MODEL_INPUT_SIZE = (int(shape[1]), int(shape[2]))
            print(f"[INFO] Emotion model input size: {_MODEL_INPUT_SIZE}")
    except Exception:
        pass

if emotion_model is not None:
    _loaded = _utils_module.load_emotions_from_class_indices()
    if not _loaded:
        try:
            num_outputs = int(emotion_model.output_shape[-1])
            if num_outputs != len(_utils_module.EMOTIONS):
                print(f"[WARNING] Model outputs {num_outputs} classes but EMOTIONS has "
                      f"{len(_utils_module.EMOTIONS)}. Predictions may be mislabelled.")
        except Exception:
            pass

if emotion_model is not None:
    try:
        dummy = np.zeros((1, _MODEL_INPUT_SIZE[0], _MODEL_INPUT_SIZE[1], 3), dtype=np.float32)
        emotion_model.predict(dummy, verbose=0)
        print("[INFO] Emotion model pre-warmed")
    except Exception as e:
        print(f"[WARNING] Pre-warm failed: {e}")

last_detection_log = {}

TRAINING_MODE = False
TRAINING_USERNAME = ""
TRAINING_COUNT = 0
TRAINING_TOTAL = 15
DETECTION_LOG_INTERVAL = 60

family_encodings_dict = {}


def load_face_encodings_to_memory():
    global family_encodings_dict
    encodings_data = get_all_face_encodings()
    temp_dict = {}
    for username, encoding_bytes_list in encodings_data.items():
        temp_dict[username] = []
        for encoding_bytes in encoding_bytes_list:
            try:
                encoding = bytes_to_encoding(encoding_bytes)
                if encoding is not None:
                    temp_dict[username].append(encoding)
            except Exception as e:
                print(f"[ERROR] Failed to load encoding for {username}: {e}")
    family_encodings_dict = temp_dict


def identify_face(face_image) -> Tuple[str, float]:
    global LAST_PERSON, LAST_CONFIDENCE, LAST_FACE_TIME
    current_time = time.time()

    if LAST_PERSON != "Unknown Person" and current_time - LAST_FACE_TIME < 5.0:
        return LAST_PERSON, LAST_CONFIDENCE

    LAST_FACE_TIME = current_time

    if not family_encodings_dict:
        return "Unknown Person", 0.0

    try:
        encoding = encode_face(face_image)
        if encoding is None:
            return LAST_PERSON, LAST_CONFIDENCE

        person_name, confidence = find_matching_person(
            encoding, family_encodings_dict, tolerance=0.60
        )

        print(f"[MATCH] Best: '{person_name}' Conf: {confidence:.3f}")

        if person_name and confidence > 0.20:
            LAST_PERSON = person_name
            try:
                if confidence <= 1.0:
                    confidence = confidence * 100
                LAST_CONFIDENCE = round(float(confidence), 2)
            except Exception:
                LAST_CONFIDENCE = 0.0
            print(f"[OK MATCHED] {person_name} ({LAST_CONFIDENCE}%)")
        else:
            LAST_PERSON = "Unknown Person"
            LAST_CONFIDENCE = 0.0
            print(f"[REJECTED] Confidence too low")
    except Exception as e:
        print(f"[ERROR] Face identification failed: {e}")

    return LAST_PERSON, LAST_CONFIDENCE


def should_log_detection(display_name: str) -> bool:
    current_time = time.time()
    last_time = last_detection_log.get(display_name, 0)
    return (current_time - last_time) > DETECTION_LOG_INTERVAL


load_face_encodings_to_memory()

DEEPFACE_LOAD_STATUS = "disabled"


def get_emotion_label(face_image):
    global LAST_EMOTION, LAST_EMOTION_TIME
    current_time = time.time()
    if current_time - LAST_EMOTION_TIME < 1.0:
        return LAST_EMOTION
    LAST_EMOTION_TIME = current_time
    try:
        if emotion_model is None:
            LAST_EMOTION = "Neutral"
            return LAST_EMOTION
        processed = preprocess_face(face_image, target_size=_MODEL_INPUT_SIZE)
        preds = emotion_model.predict(processed, verbose=0)
        idx = int(np.argmax(preds))
        if 0 <= idx < len(_utils_module.EMOTIONS):
            LAST_EMOTION = _utils_module.EMOTIONS[idx]
        else:
            LAST_EMOTION = "Neutral"
    except Exception as e:
        print(f"[ERROR] Emotion detection failed: {e}")
        LAST_EMOTION = "Neutral"
    return LAST_EMOTION


def get_camera_capture():
    cap, index = camera_manager.get_camera()
    if cap is None or not cap.isOpened():
        return None, None
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    for _ in range(3):
        cap.read()
    return cap, index


def generate_frames(display_name="Unknown Person"):
    global CAMERA_RUNNING
    CAMERA_RUNNING = True
    last_detection_log.clear()

    print("[DEBUG] generate_frames: starting...")

    global LAST_EMOTION_TIME, LAST_FACE_TIME
    LAST_EMOTION_TIME = 0
    LAST_FACE_TIME = 0

    video_capture, camera_index = get_camera_capture()
    if video_capture is None:
        CAMERA_RUNNING = False
        placeholder = create_placeholder_frame("Camera not found")
        if placeholder is not None:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
        return

    stable_detection_count = 0
    try:
        while True:
            success, frame = video_capture.read()
            if not success:
                break
            frame_bytes = render_frame(frame, camera_index=camera_index)
            if frame_bytes is None:
                continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            if LAST_PERSON != "Unknown Person":
                if stable_detection_count == 0:
                    stable_detection_count = time.time()
                elif time.time() - stable_detection_count > 8:
                    print("[INFO] Stable detection achieved. Auto-stopping camera.")
                    break
            else:
                stable_detection_count = 0
    except Exception as e:
        print(f"[ERROR] generate_frames: {e}")
    finally:
        camera_manager.release()
        CAMERA_RUNNING = False

    final_frame = create_placeholder_frame("Detection Complete! Camera stopped.")
    if final_frame is not None:
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + final_frame + b'\r\n')


def get_active_display_name():
    selected_person = session.get('selected_person', '')
    if selected_person:
        person = find_person(selected_person)
        if person:
            return person['name']
    username = session.get('user', '')
    if username:
        person = find_person(username)
        if person:
            return person['name']
    people = load_people()
    if people:
        return people[0].get('name', 'Unknown Person')
    return 'Unknown Person'


def load_translations():
    with open('translations.json', 'r', encoding='utf-8') as f:
        return json.load(f)


@app.route('/set_language/<lang>')
def set_language(lang):
    session['lang'] = lang
    return redirect(request.referrer or url_for('index'))


@app.context_processor
def inject_translations():
    lang = session.get('lang', 'en')
    try:
        translations = load_translations()
        return dict(t=translations.get(lang, translations['en']), current_lang=lang)
    except Exception:
        return dict(t={}, current_lang='en')


def render_frame(frame, camera_index=None):
    global LAST_EMOTION, LAST_PERSON, LAST_CONFIDENCE, LAST_EMOTION_TIME, LAST_FACE_TIME
    global last_detection_log, _frame_counter, _last_faces, _face_recognition_cache

    if frame is None:
        return None

    frame = cv2.resize(frame, (480, 360))

    _frame_counter += 1
    faces = detect_faces(frame)
    _last_faces = faces

    current_time = time.time()

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        _face_recognition_cache.clear()
    else:
        active_keys = set()

        for idx, (x, y, w, h) in enumerate(faces):
            face = frame[y:y + h, x:x + w]
            cache_key = f"{x // 80}_{y // 80}"

            cached = _face_recognition_cache.get(cache_key)
            if cached and (current_time - cached['time']) < 3.0:
                person_name = cached['name']
                confidence = cached['confidence']
                emotion = cached['emotion']
            else:
                try:
                    emotion = get_emotion_label(face)
                except Exception:
                    emotion = "Neutral"

                person_name = "Unknown Person"
                confidence = 0.0

                try:
                    encoding = encode_face(face)
                    if encoding is not None and family_encodings_dict:
                        # Stricter tolerance (0.50 instead of 0.65) and higher confidence
                        matched_name, matched_conf = find_matching_person(
                            encoding, family_encodings_dict, tolerance=0.60
                        )
                        if matched_name and matched_conf > 20.0:
                            person_name = matched_name
                            try:
                                if matched_conf <= 1.0:
                                    matched_conf = matched_conf * 100
                                confidence = round(float(matched_conf), 2)
                            except Exception:
                                confidence = 0.0
                        print(f"[MULTI-FACE] Face {idx + 1}: {person_name} ({confidence}%)")
                except Exception as e:
                    print(f"[ERROR] Recognition failed for face {idx + 1}: {e}")

                if person_name != "Unknown Person":
                       _face_recognition_cache[cache_key] = {
                        'name': person_name,
                        'confidence': confidence,
                        'emotion': emotion,
                        'time': current_time
                    }

            active_keys.add(cache_key)

            if idx == 0:
                LAST_PERSON = person_name
                LAST_CONFIDENCE = confidence
                LAST_EMOTION = emotion

            if person_name == "Unknown Person":
                overlay_text = f"Unknown | {emotion}"
                text_color = (0, 0, 255)
            else:
                person_data = find_person_by_username(person_name)
                display_name = person_data["name"] if person_data else person_name
                overlay_text = f"{display_name} | {emotion}"
                text_color = (0, 255, 0)

                if should_log_detection(person_name):
                    person_data = find_person_by_username(person_name)
                    person_id = person_data["id"] if person_data else None
                    log_face_detection(person_name, emotion, person_id)
                    last_detection_log[person_name] = current_time

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, overlay_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        stale_keys = set(_face_recognition_cache.keys()) - active_keys
        for key in stale_keys:
            del _face_recognition_cache[key]

    camera_text = f"Camera Index : {camera_index}" if camera_index is not None else "Camera"
    cv2.putText(frame, camera_text, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ret:
        return None
    return buffer.tobytes()


def create_placeholder_frame(message: str = "Starting camera..."):
    h, w = 360, 640
    placeholder = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(placeholder, message, (20, int(h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    ret, buffer = cv2.imencode('.jpg', placeholder)
    return buffer.tobytes() if ret else None


def get_one_frame(display_name="Unknown Person"):
    video_capture, camera_index = get_camera_capture()
    if video_capture is None:
        return create_placeholder_frame("Camera not found")
    try:
        warmup_tries = 0
        success = False
        frame = None
        while warmup_tries < 10:
            success, frame = video_capture.read()
            if success and frame is not None:
                break
            warmup_tries += 1
        if not success or frame is None:
            return create_placeholder_frame("No camera frames")
        result = render_frame(frame, camera_index=camera_index)
        return result
    except Exception as e:
        print(f"[DEBUG] /frame: Exception occurred: {e}")
        return create_placeholder_frame("Frame error")
    finally:
        camera_manager.release()


def get_profile_pic(username):
    pic_path = os.path.join('static', 'profile_pics', f"{username}.jpg")
    if os.path.exists(pic_path):
        return f"profile_pics/{username}.jpg"
    return "images/default_profile.png"


@app.route('/video_feed')
def video_feed():
    display_name = get_active_display_name()
    response = Response(generate_frames(display_name=display_name), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


_frame_cache = {"time": 0.0, "bytes": None}
_frame_cache_lock = threading.Lock()
FRAME_CACHE_MS = 0.2

import base64
import io
from PIL import Image

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # 1. Get the Base64 image from the browser
        data = request.get_json()
        image_data = data['image']
        
        # 2. Decode the image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV BGR format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Resize to save bandwidth and processing time
        frame = cv2.resize(frame, (480, 360))
        
        results = []
        
        # 3. Run AI detection
        faces = detect_faces(frame)
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                
                # Emotion
                emotion = get_emotion_label(face)
                
                # Face Recognition
                person_name = "Unknown Person"
                encoding = encode_face(face)
                if encoding is not None and family_encodings_dict:
                    matched_name, matched_conf = find_matching_person(
                        encoding, family_encodings_dict, tolerance=0.60
                    )
                    if matched_name and matched_conf > 20.0:
                        person_name = matched_name
                        
                        # Log to database (throttled)
                        if should_log_detection(person_name):
                            p_data = find_person_by_username(person_name)
                            p_id = p_data["id"] if p_data else None
                            log_face_detection(person_name, emotion, p_id)
                            last_detection_log[person_name] = time.time()

                 # Add to results list
                results.append({
                    "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                    "name": person_name,
                    "emotion": emotion,
                    "confidence": matched_conf if person_name != "Unknown Person" else 0.0
                })
                
        return {"success": True, "faces": results}

    except Exception as e:
        print(f"Error in process_frame: {e}")
        return {"success": False, "error": str(e)}

@app.route('/frame')
def frame():
    display_name = get_active_display_name()
    now = time.time()
    jpeg_bytes = None
    with _frame_cache_lock:
        if _frame_cache["bytes"] and (now - _frame_cache["time"]) < FRAME_CACHE_MS:
            jpeg_bytes = _frame_cache["bytes"]

    if jpeg_bytes is None:
        jpeg_bytes = get_one_frame(display_name=display_name)
        if jpeg_bytes is None:
            return Response('Camera error', status=503)
        with _frame_cache_lock:
            _frame_cache["bytes"] = jpeg_bytes
            _frame_cache["time"] = now

    response = Response(jpeg_bytes, mimetype='image/jpeg')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/current_detection')
def current_detection():
    return {"person": LAST_PERSON, "emotion": LAST_EMOTION, "confidence": LAST_CONFIDENCE, "camera_running": CAMERA_RUNNING}


@app.route("/training_status")
def training_status():
    return {"current": TRAINING_COUNT, "total": TRAINING_TOTAL}


@app.route('/debug_status')
@admin_required
def debug_status():
    return {
        "deepface_status": DEEPFACE_LOAD_STATUS,
        "encodings_in_memory": len(family_encodings_dict),
        "people_in_memory": list(family_encodings_dict.keys()),
        "total_encodings": sum(len(encs) for encs in family_encodings_dict.values())
    }


@app.route('/model_status')
def model_status():
    return {"status": DEEPFACE_LOAD_STATUS}


@app.route('/')
def home():
    return render_template("login.html")


@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')

    key = _login_key()
    if _login_rate_limited(key):
        return render_template("login.html",
                               error="Too many attempts. Please wait a few minutes.")

    admin_user = os.getenv("ADMIN_USERNAME", "Maramreddy Narendrareddy")
    admin_pass = os.getenv("ADMIN_PASSWORD", "Narendrareddy@2005")

    if username == admin_user and password == admin_pass:
        session['user'] = username
        session['selected_person'] = username
        session['is_admin'] = True
        try:
            if not find_person_by_username(username):
                db_register_person(username=username, name="Administrator",
                                   email=EMAIL_ADDRESS or "admin@example.com",
                                   phone="7661826389")
        except Exception:
            pass
        return redirect(url_for('index'))

    person = find_person_by_username(username)
    if not person:
        _record_login_attempt(key)
        return render_template("login.html", error="Invalid Username or Password")

    is_valid, _migrated = verify_user_password(username, password)
    if is_valid:
        session['user'] = username
        session['selected_person'] = username
        session['is_admin'] = is_user_admin(username)
        return redirect(url_for('index'))

    _record_login_attempt(key)
    return render_template("login.html", error="Invalid Username or Password")


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fullname = request.form.get('fullname', '').strip()
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not fullname or not username or not email or not password or not phone:
            return render_template('signup.html', error="All fields are required")
        if password != confirm_password:
            return render_template('signup.html', error="Passwords do not match")
        if len(password) < 6:
            return render_template('signup.html', error="Password must be at least 6 characters")
        if not re.search(r'[A-Z]', password):
            return render_template('signup.html', error="Password must contain at least one uppercase letter")
        if not re.search(r'[a-z]', password):
            return render_template('signup.html', error="Password must contain at least one lowercase letter")
        if not re.search(r'[0-9]', password):
            return render_template('signup.html', error="Password must contain at least one number")

        try:
            conn = sqlite3.connect(str(DB_FILE))
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                conn.close()
                return render_template('signup.html',
                    error="You already have a login account with this username. Please log in instead.")

            cursor.execute("SELECT username FROM people WHERE email = ? AND username != ?", (email, username))
            if cursor.fetchone():
                conn.close()
                return render_template('signup.html',
                    error="This email is already registered by another user. Please use a different email.")

            cursor.execute("SELECT username FROM people WHERE phone = ? AND username != ?", (phone, username))
            if cursor.fetchone():
                conn.close()
                return render_template('signup.html',
                    error="This phone number is already registered by another user. Please use a different phone number.")

            conn.close()
        except Exception as e:
            print(f"[ERROR] Database check failed: {e}")

        otp = ''.join(random.choices(string.digits, k=6))
        session['pending_signup_data'] = {
            'fullname': fullname, 'username': username,
            'email': email, 'phone': phone, 'password': password
        }
        session['otp'] = otp

        if send_otp_email(email, otp):
            return render_template('signup.html', show_otp=True, email=email,
                                   status="OTP sent to your email! Please verify.")
        else:
            print(f"DEBUG: The OTP is {otp}", flush=True)
            return render_template('signup.html', show_otp=True, email=email,
                                   status="OTP generated! (Check terminal)")

    return render_template('signup.html')


@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    entered_otp = request.form.get('otp')
    if entered_otp == session.get('otp'):
        session.pop('otp', None)

        if 'pending_login_user' in session:
            username = session.pop('pending_login_user')
            session['user'] = username
            session['selected_person'] = username
            session['is_admin'] = False
            return redirect(url_for('index'))

        elif 'pending_signup_data' in session:
            data = session.pop('pending_signup_data')

            existing = find_person_by_username(data['username'])
            if existing:
                conn = sqlite3.connect(str(DB_FILE))
                cursor = conn.cursor()
                cursor.execute("UPDATE people SET email=?, phone=? WHERE username=?",
                               (data['email'], data['phone'], data['username']))
                conn.commit()
                conn.close()
            else:
                db_register_person(username=data['username'], name=data['fullname'],
                                   email=data['email'], phone=data['phone'])

            create_user(data['username'], data['password'], is_admin=0)

            session['user'] = data['username']
            session['selected_person'] = data['username']
            session['is_admin'] = False
            return redirect(url_for('index'))

        elif 'pending_person_data' in session:
            data = session.pop('pending_person_data')
            db_register_person(data['username'], data['name'], data['email'], data['phone'])
            return redirect(url_for('train_face', username=data['username']))
    else:
        if 'pending_person_data' in session:
            return render_template('register_person.html', show_otp=True, status="Invalid OTP. Please try again.")
        return render_template('signup.html', show_otp=True, status="Invalid OTP. Please try again.")
    return redirect(url_for('home'))


@app.route('/resend_otp', methods=['POST'])
def resend_otp():
    email = None
    if 'pending_signup_data' in session:
        email = session['pending_signup_data'].get('email')
    elif 'pending_person_data' in session:
        email = session['pending_person_data'].get('email')
    elif 'pending_login_user' in session:
        username = session['pending_login_user']
        person = find_person_by_username(username)
        email = person['email'] if person else None
    if not email:
        return {"error": "No email found in session"}, 400
    new_otp = ''.join(random.choices(string.digits, k=6))
    session['otp'] = new_otp
    if send_otp_email(email, new_otp):
        return {"success": True, "message": "OTP sent successfully to " + email}
    else:
        print(f"DEBUG: The OTP is {new_otp}", flush=True)
        return {"success": True, "message": "OTP generated! Check terminal."}


@app.route("/index")
@login_required
def index():
    username = session.get('user', '')
    is_admin = session.get('is_admin', False)
    active_name = get_active_display_name()
    profile_pic = get_profile_pic(username)

    people = []
    total_people = 0
    today_count = 0
    unique_people = 0
    total_faces = 0
    total_logs = 0
    recent_logs = []

    try:
        if is_admin:
            people = load_people()
            total_people = get_total_people()
            today_count = get_today_detection_count()
            unique_people = get_unique_people_detected()
            total_faces = get_total_face_encodings()
            total_logs = get_total_logs()
            recent_logs = get_recent_logs(5)
        else:
            conn = sqlite3.connect(str(DB_FILE))
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM people WHERE username = ?", (username,))
            res = cursor.fetchone()
            person_id = res[0] if res else None
            if person_id:
                cursor.execute("SELECT COUNT(*) FROM face_detections WHERE person_id = ?", (person_id,))
                total_logs = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM face_detections WHERE person_id = ? AND DATE(detected_at) = DATE('now')", (person_id,))
                today_count = cursor.fetchone()[0]
            conn.close()
            people = [p for p in load_people() if p['username'] == username]
            total_people = 1
            unique_people = 1
            recent_logs = get_person_logs(username, limit=5) if person_id else []
    except Exception as e:
        print(f"[ERROR] Index page caught: {e}")

    return render_template("index.html", people=people, active_name=active_name,
                           total_people=total_people, total_faces=total_faces,
                           total_logs=total_logs, today_count=today_count,
                           unique_people=unique_people, recent_logs=recent_logs,
                           profile_pic=profile_pic)


@app.route('/emotion_analytics')
@login_required
def emotion_analytics():
    username = session.get('user', '')
    is_admin = session.get('is_admin', False)

    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()

    if is_admin:
        cursor.execute("SELECT emotion, COUNT(*) FROM face_detections WHERE person_name != 'Unknown Person' GROUP BY emotion")
    else:
        cursor.execute("SELECT emotion, COUNT(*) FROM face_detections WHERE person_name = ? GROUP BY emotion", (username,))

    data = cursor.fetchall()
    conn.close()

    emotions = [str(row[0]) for row in data]
    counts = [int(row[1]) for row in data]

    return render_template("emotion_analytics.html", emotions=emotions, counts=counts)


@app.route('/download_report')
@login_required
def download_report():
    range_type = request.args.get('range', 'week')
    days = 7 if range_type == 'week' else 30

    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT person_name, emotion, detected_at
        FROM face_detections
        WHERE detected_at >= datetime('now', '-{days} days')
        AND person_name != 'Unknown Person'
        ORDER BY detected_at DESC
    """)
    logs = cursor.fetchall()
    conn.close()

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph(f"AI Facial Expression Recognition - {range_type.title()}ly Report", styles['Title']))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(f"Total Detections: {len(logs)}", styles['Normal']))
    elements.append(Spacer(1, 20))

    data = [['#', 'Person Name', 'Emotion', 'Detected At']]
    for i, log in enumerate(logs, 1):
        data.append([str(i), log[0], log[1], log[2]])

    table = Table(data, colWidths=[40, 180, 120, 180])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4ff')])
    ]))
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)

    return Response(
        buffer.getvalue(),
        mimetype='application/pdf',
        headers={'Content-Disposition': f'attachment;filename=emotion_report_{range_type}.pdf'}
    )


@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html")


@app.route("/change_password", methods=["GET", "POST"])
@login_required
def change_password():
    if request.method == "POST":
        current = request.form.get("current_password", "")
        new_pass = request.form.get("new_password", "")
        confirm = request.form.get("confirm_password", "")
        username = session.get('user', '')

        if not current or not new_pass or not confirm:
            return render_template("change_password.html", error="All fields are required")
        if new_pass != confirm:
            return render_template("change_password.html", error="New passwords do not match")
        if len(new_pass) < 6:
            return render_template("change_password.html", error="Password must be at least 6 characters")
        if not re.search(r'[A-Z]', new_pass):
            return render_template("change_password.html", error="Password must contain at least one uppercase letter")
        if not re.search(r'[a-z]', new_pass):
            return render_template("change_password.html", error="Password must contain at least one lowercase letter")
        if not re.search(r'[0-9]', new_pass):
            return render_template("change_password.html", error="Password must contain at least one number")

        try:
            is_valid, _ = verify_user_password(username, current)
            if not is_valid:
                return render_template("change_password.html", error="Current password is incorrect")
            update_user_password(username, new_pass)
            print(f"[SUCCESS] Password updated for {username}")
            return render_template("change_password.html", success="Password updated successfully!")
        except Exception as e:
            print(f"[ERROR] Password change failed: {e}")
            return render_template("change_password.html", error="Failed to update password")

    return render_template("change_password.html")


@app.route("/login_info")
@login_required
def login_info():
    return render_template("login_info.html")


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    if request.method == "POST":
        if 'profile_pic' not in request.files:
            return render_template("settings.html", error="No file selected")
        file = request.files['profile_pic']
        if file.filename == '':
            return render_template("settings.html", error="No file selected")
        username = session.get('user', '')
        if file:
            os.makedirs(os.path.join('static', 'profile_pics'), exist_ok=True)
            filename = f"{username}.jpg"
            file.save(os.path.join('static', 'profile_pics', filename))
            return redirect(url_for('settings'))
    username = session.get('user', '')
    profile_pic = get_profile_pic(username)
    return render_template("settings.html", profile_pic=profile_pic)


@app.route('/camera')
@login_required
def camera():
    return render_template("camera.html")


@app.route("/stop_camera")
def stop_camera():
    camera_manager.release()
    return redirect(url_for("index"))


@app.route("/registered_persons")
@admin_required
def registered_persons():
    people = load_people()
    search = request.args.get("search", "").strip().lower()
    if search:
        people = [person for person in people if (search in person.get("name", "").lower()
                  or search in person.get("username", "").lower()
                  or search in person.get("email", "").lower()
                  or search in person.get("phone", "").lower())]
    return render_template("registered_persons.html", people=people)


@app.route("/person/<username>")
@login_required
def person_details(username):
    person = find_person_by_username(username)
    if person is None:
        return "Person Not Found", 404
    photo = None
    folder = os.path.join("static", "trained_faces", username)
    if os.path.exists(folder):
        images = sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])
        if images:
            photo = f"trained_faces/{username}/{images[0]}"
    return render_template("person_details.html", person=person, photo=photo)


@app.route("/edit_person/<username>", methods=["GET", "POST"])
@admin_required
def edit_person(username):
    person = find_person_by_username(username)
    if person is None:
        return "Person Not Found", 404

    if request.method == "POST":
        new_name = request.form["name"].strip()
        new_email = request.form["email"].strip()
        new_phone = request.form["phone"].strip()

        try:
            conn = sqlite3.connect(str(DB_FILE))
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE people SET name = ?, email = ?, phone = ? WHERE username = ?",
                (new_name, new_email, new_phone, username)
            )
            conn.commit()
            conn.close()
            print(f"[SUCCESS] Updated {username} in database")
        except Exception as e:
            print(f"[ERROR] Failed to update person: {e}")
            return render_template("edit_person.html", person=person, error="Failed to save changes")

        return redirect(url_for("registered_persons"))

    return render_template("edit_person.html", person=person)


@app.route("/delete_person/<username>")
@admin_required
def delete_person_route(username):
    folder = os.path.join("static", "trained_faces", username)
    if os.path.exists(folder):
        shutil.rmtree(folder)
    db_delete_person(username)
    try:
        conn = sqlite3.connect(str(DB_FILE))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[ERROR] Failed to delete from users table: {e}")
    load_face_encodings_to_memory()
    return redirect(url_for("registered_persons"))


@app.route('/add_person', methods=['POST'])
@login_required
def add_person():
    username = request.form.get('username', '').strip()
    name = request.form.get('name', '').strip()
    if username:
        pd_register_person(username=username, name=name or username)
        session['selected_person'] = username
    return redirect(url_for('index'))


@app.route('/set_person', methods=['POST'])
@login_required
def set_person():
    selected_person = request.form.get('selected_person', '').strip()
    if selected_person:
        session['selected_person'] = selected_person
    return redirect(url_for('index'))


@app.route('/train_face/<username>', methods=['GET', 'POST'])
@login_required
def train_face(username):
    if not session.get('is_admin') and session.get('user') != username:
        return "Not authorized to train this person", 403
    person = find_person_by_username(username)
    if not person:
        return "Person not found", 404
    if request.method == 'POST':
        video_capture, camera_index = get_camera_capture()
        if video_capture is None:
            return render_template("train_face.html", username=username, error="Camera not found"), 400
        TARGET = 15
        faces_found = 0
        frames_tried = 0
        max_frames = 2000
        try:
            while frames_tried < max_frames and faces_found < TARGET:
                success, frame = video_capture.read()
                if not success:
                    frames_tried += 1
                    continue
                frames_tried += 1
                faces = detect_faces(frame)
                if len(faces) == 0:
                    continue
                faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
                for (x, y, w, h) in faces:
                    global TRAINING_MODE, TRAINING_USERNAME, TRAINING_COUNT
                    pad_w = int(w * 0.1)
                    pad_h = int(h * 0.1)
                    x1 = max(0, x - pad_w)
                    y1 = max(0, y - pad_h)
                    x2 = min(frame.shape[1], x + w + pad_w)
                    y2 = min(frame.shape[0], y + h + pad_h)
                    face_crop = frame[y1:y2, x1:x2]
                    encoding = encode_face(face_crop)
                    if encoding is not None:
                        encoding_bytes = encoding_to_bytes(encoding)
                        if encoding_bytes is not None:
                            stored_id = store_face_encoding(person['id'], encoding_bytes)
                            if stored_id > 0:
                                faces_found += 1
                                TRAINING_COUNT = faces_found
                                print(f"Captured {TRAINING_COUNT}/{TARGET}")
                                time.sleep(0.1)
                    if faces_found >= TARGET:
                        break
                    break  
                time.sleep(0.05)
        finally:
            camera_manager.release()
        if faces_found > 0:
            load_face_encodings_to_memory()
            print(f"[SUCCESS] Trained {username} with {faces_found} face(s)")
            TRAINING_COUNT = 0
            return render_template("training_success.html", username=username)
        return render_template("train_face.html", username=username, error="No face detected. Please try again."), 400
    return render_template("train_face.html", username=username)


@app.route('/test_recognition')
@login_required
def test_recognition():
    video_capture, camera_index = get_camera_capture()
    if video_capture is None:
        return {"error": "Camera not found"}, 404
    success, frame = video_capture.read()
    camera_manager.release()
    if not success or frame is None:
        return {"error": "Failed to capture frame"}, 400
    faces = detect_faces(frame)
    if len(faces) == 0:
        return {"result": "No face detected in frame"}, 200
    face = frame[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]]
    encoding = encode_face(face)
    if encoding is None:
        return {"result": "Failed to encode face"}, 200
    person_name, confidence = identify_face(face)
    return {"result": "Success", "person": person_name, "confidence": confidence}


@app.route('/test_encodings')
@admin_required
def test_encodings():
    return {"in_memory": {"people_count": len(family_encodings_dict),
                          "people": list(family_encodings_dict.keys()),
                          "total_encodings": sum(len(encs) for encs in family_encodings_dict.values())}}


@app.route('/reload_encodings')
@admin_required
def reload_encodings():
    load_face_encodings_to_memory()
    return {"status": "Encodings reloaded successfully"}


@app.route("/capture_more_images/<username>")
@login_required
def capture_more_images(username):
    return redirect(url_for("train_face", username=username))


@app.route('/view_logs')
@login_required
def view_logs():
    from_date = request.args.get("from_date")
    to_date = request.args.get("to_date")
    if from_date and to_date:
        logs = get_logs_by_date(from_date, to_date)
    else:
        logs = get_face_detection_logs(limit=200)
    return render_template("logs.html", logs=logs, from_date=from_date, to_date=to_date)


@app.route('/today_logs')
@login_required
def today_logs():
    from datetime import date
    today_str = date.today().strftime("%Y-%m-%d")
    try:
        logs = get_logs_by_date(today_str, today_str)
    except Exception:
        logs = []
    return render_template("today_logs.html", logs=logs)


@app.route("/export_logs")
@login_required
def export_logs():
    from_date = request.args.get("from_date")
    to_date = request.args.get("to_date")
    if from_date and to_date:
        logs = get_logs_by_date(from_date, to_date)
    else:
        logs = get_face_detection_logs(limit=10000)
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Person Name", "Emotion", "Detected At"])
    for log in logs:
        writer.writerow([log["id"], log["person_name"], log["emotion"], log["detected_at"]])
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=attendance_logs.csv"
    response.headers["Content-Type"] = "text/csv"
    return response


@app.route('/person_logs/<person_name>')
@login_required
def view_person_logs(person_name):
    logs = get_person_logs(person_name, limit=100)
    return render_template("person_logs.html", person_name=person_name, logs=logs)


@app.route('/face_encodings')
@admin_required
def face_encodings_view():
    all_encodings = get_all_face_encodings()
    data = []
    for user, encs in all_encodings.items():
        data.append({"username": user, "count": len(encs)})
    return render_template("encodings.html", data=data)


@app.route("/register_person", methods=["GET", "POST"])
@admin_required
def register_person_page():
    if request.method == "POST":
        username = request.form["username"]
        name = request.form["name"]
        email = request.form["email"]
        phone = request.form["phone"]
        otp = ''.join(random.choices(string.digits, k=6))
        session['pending_person_data'] = {'username': username, 'name': name, 'email': email, 'phone': phone}
        session['otp'] = otp
        if send_otp_email(email, otp):
            return render_template("register_person.html", show_otp=True, email=email, status="OTP sent to your email!")
        else:
            print(f"DEBUG: The OTP is {otp}", flush=True)
            return render_template("register_person.html", show_otp=True, email=email, status="OTP generated! Check terminal.")
    return render_template("register_person.html")


@app.route('/debug_translations')
def debug_translations():
    lang = session.get('lang', 'en')
    try:
        translations = load_translations()
        current = translations.get(lang, translations['en'])
        return {
            "session_lang": lang,
            "available_languages": list(translations.keys()),
            "pwd_length": current.get('pwd_length', 'MISSING'),
            "verify_continue": current.get('verify_continue', 'MISSING'),
            "total_keys": len(current),
        }
    except Exception as e:
        return {"error": str(e)}


@app.route("/admin_panel")
@admin_required
def admin_panel():
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, name, email, phone FROM people")
    all_users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    all_logs = [log for log in get_face_detection_logs(limit=100) if log["person_name"] != "Unknown Person"]
    for user in all_users:
        if not user.get('email'):
            user['email'] = "Not Provided"
        if not user.get('phone'):
            user['phone'] = "Not Provided"
    return render_template("admin_panel.html", all_users=all_users, all_logs=all_logs)


print("\nRegistered Routes:")
print(app.url_map)

if __name__ == "__main__":
    ensure_db_exists()
    import webbrowser
    from threading import Timer
    Timer(1, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host="127.0.0.1", port=5000, debug=False)