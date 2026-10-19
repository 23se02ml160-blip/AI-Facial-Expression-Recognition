import sqlite3
import os
import json

print("=" * 60)
print("VERIFYING DATABASE AND TRANSLATIONS")
print("=" * 60)

# 1. Check database tables
print("\n[1] Database Tables:")
try:
    conn = sqlite3.connect('data/app.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]
    print(f"    Tables found: {tables}")
    
    for table in ['people', 'detections', 'face_encodings', 'users']:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"    {table}: {count} rows")
        except Exception as e:
            print(f"    {table}: MISSING or ERROR")
    
    conn.close()
except Exception as e:
    print(f"    ERROR: {e}")

# 2. Check translations.json
print("\n[2] Translations File:")
try:
    with open('translations.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"    Languages: {list(data.keys())}")
    print(f"    English welcome: {data['en']['welcome']}")
    print(f"    Hindi welcome: {data['hi']['welcome']}")
    print(f"    Gujarati welcome: {data['gu']['welcome']}")
except Exception as e:
    print(f"    ERROR: {e}")
    print(f"    Please save translations.json with UTF-8 encoding")

# 3. Check models
print("\n[3] Models Folder:")
if os.path.exists('models'):
    for f in os.listdir('models'):
        size_mb = os.path.getsize(os.path.join('models', f)) / (1024 * 1024)
        print(f"    {f} ({size_mb:.2f} MB)")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)