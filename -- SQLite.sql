-- ============================================
-- AI Facial Expression Recognition — reset script
-- WARNING: This DROPS all tables and deletes all data.
-- ============================================
PRAGMA foreign_keys = OFF;

DROP TABLE IF EXISTS face_detections;
DROP TABLE IF EXISTS face_encodings;
DROP TABLE IF EXISTS people;
DROP TABLE IF EXISTS users;

CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    is_admin INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    email TEXT,
    phone TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE face_encodings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    encoding BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES people(id)
);

CREATE TABLE face_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER,
    person_name TEXT NOT NULL,
    emotion TEXT,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES people(id)
);

PRAGMA foreign_keys = ON;