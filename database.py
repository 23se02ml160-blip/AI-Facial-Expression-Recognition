import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from werkzeug.security import generate_password_hash, check_password_hash

DB_FILE = Path(__file__).resolve().parent / "data" / "app.db"

IST = timezone(timedelta(hours=5, minutes=30))


def _get_ist_timestamp() -> str:
    return datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')


def ensure_db_exists():
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            name TEXT,
            email TEXT,
            phone TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            person_name TEXT NOT NULL,
            emotion TEXT,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES people (id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            encoding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES people (id)
        )
    """)

    conn.commit()
    conn.close()


# ---------------- People ----------------

def find_person_by_username(username: str) -> Optional[Dict]:
    if not username:
        return None
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, name, email, phone, created_at FROM people WHERE username = ?",
        (username,)
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def user_exists(username: str) -> bool:
    if not username:
        return False
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM people WHERE username = ?", (username,))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


def register_person(username: str, name: str, email: str = "", phone: str = "") -> Dict:
    clean_username = (username or "").strip()
    if not clean_username:
        return {"username": "", "name": "Unknown Person"}
    display_name = (name or clean_username).strip() or clean_username
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, name FROM people WHERE username = ?", (clean_username,))
    row = cursor.fetchone()
    if row:
        cursor.execute(
            "UPDATE people SET name = ?, email = ?, phone = ? WHERE username = ?",
            (display_name, email, phone, clean_username)
        )
        person_id = row["id"]
    else:
        cursor.execute(
            "INSERT INTO people (username, name, email, phone) VALUES (?, ?, ?, ?)",
            (clean_username, display_name, email, phone)
        )
        person_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"id": person_id, "username": clean_username, "name": display_name,
            "email": email, "phone": phone}


def load_all_people() -> List[Dict]:
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, name, email, phone, created_at FROM people ORDER BY name")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ---------------- Users (auth) ----------------

def create_user(username: str, password: str, is_admin: int = 0) -> bool:
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), is_admin)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def verify_user_password(username: str, password: str):
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    cursor.execute("SELECT id, password FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False, False

    stored = row[1]
    try:
        if check_password_hash(stored, password):
            conn.close()
            return True, False
    except Exception:
        pass

    # Legacy plaintext migration
    if stored == password:
        new_hash = generate_password_hash(password)
        cursor.execute("UPDATE users SET password = ? WHERE username = ?", (new_hash, username))
        conn.commit()
        conn.close()
        return True, True

    conn.close()
    return False, False


def update_user_password(username: str, new_password: str) -> bool:
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET password = ? WHERE username = ?",
        (generate_password_hash(new_password), username)
    )
    conn.commit()
    conn.close()
    return True


def user_exists_in_users(username: str) -> bool:
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,))
    ok = cursor.fetchone() is not None
    conn.close()
    return ok


def is_user_admin(username: str) -> bool:
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    cursor.execute("SELECT is_admin FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    return bool(row and row[0])


# ---------------- Detection logs ----------------

def log_face_detection(person_name, emotion, person_id=None):
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO face_detections (person_id, person_name, emotion, detected_at) VALUES (?, ?, ?, ?)",
        (person_id, person_name, emotion, _get_ist_timestamp())
    )
    conn.commit()
    conn.close()


def get_face_detection_logs(limit: int = 100) -> List[Dict]:
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, person_name, emotion, detected_at FROM face_detections "
        "ORDER BY detected_at DESC LIMIT ?", (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_person_logs(person_name: str, limit: int = 50) -> List[Dict]:
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, person_name, emotion, detected_at FROM face_detections "
        "WHERE person_name = ? ORDER BY detected_at DESC LIMIT ?",
        (person_name, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ---------------- Face encodings ----------------

def store_face_encoding(person_id: int, encoding_bytes: bytes) -> int:
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO face_encodings (person_id, encoding) VALUES (?, ?)",
            (person_id, encoding_bytes)
        )
        conn.commit()
        encoding_id = cursor.lastrowid
        if encoding_id is None:
            cursor.execute("SELECT last_insert_rowid()")
            encoding_id = cursor.fetchone()[0]
        conn.close()
        return encoding_id
    except Exception as e:
        print(f"[ERROR] Failed to store face encoding: {e}")
        conn.close()
        return 0


def get_face_encodings_for_person(person_id: int) -> List[bytes]:
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    cursor.execute("SELECT encoding FROM face_encodings WHERE person_id = ?", (person_id,))
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]


def get_all_face_encodings() -> dict:
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.id, p.username, fe.encoding
        FROM face_encodings fe
        JOIN people p ON fe.person_id = p.id
    """)
    rows = cursor.fetchall()
    conn.close()
    encodings_dict = {}
    for row in rows:
        username = row['username']
        encodings_dict.setdefault(username, []).append(row['encoding'])
    return encodings_dict


def clear_face_encodings_for_person(person_id: int) -> None:
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    cursor.execute("DELETE FROM face_encodings WHERE person_id = ?", (person_id,))
    conn.commit()
    conn.close()


def delete_person(username):
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM people WHERE username=?", (username,))
    row = cursor.fetchone()
    if row is None:
        conn.close()
        return
    person_id = row[0]
    cursor.execute("DELETE FROM face_encodings WHERE person_id=?", (person_id,))
    cursor.execute("DELETE FROM face_detections WHERE person_id=?", (person_id,))
    cursor.execute("DELETE FROM people WHERE id=?", (person_id,))
    conn.commit()
    conn.close()


# ---------------- Stats ----------------

def get_total_people():
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM people")
    total = cursor.fetchone()[0]
    conn.close()
    return total


def get_total_face_encodings():
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM face_encodings")
    total = cursor.fetchone()[0]
    conn.close()
    return total


def get_total_logs():
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM face_detections")
    total = cursor.fetchone()[0]
    conn.close()
    return total


def get_today_detection_count():
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    today_ist = datetime.now(IST).strftime('%Y-%m-%d')
    cursor.execute(
        "SELECT COUNT(*) FROM face_detections WHERE DATE(detected_at) = ?",
        (today_ist,)
    )
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_unique_people_detected():
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT person_name) FROM face_detections")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_logs_by_date(from_date, to_date):
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM face_detections WHERE DATE(detected_at) BETWEEN ? AND ? "
        "ORDER BY detected_at DESC",
        (from_date, to_date)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_recent_logs(limit=5):
    ensure_db_exists()
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM face_detections ORDER BY detected_at DESC LIMIT ?", (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]