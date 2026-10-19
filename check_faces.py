import sqlite3
conn = sqlite3.connect('data/app.db')
cursor = conn.cursor()

print("=" * 50)
print("PEOPLE IN DATABASE")
print("=" * 50)
cursor.execute("SELECT id, username, name FROM people")
for row in cursor.fetchall():
    print(f"  ID={row[0]} | username='{row[1]}' | name='{row[2]}'")

print()
print("=" * 50)
print("ENCODINGS PER PERSON")
print("=" * 50)
cursor.execute("""
    SELECT p.username, COUNT(fe.id) 
    FROM people p 
    LEFT JOIN face_encodings fe ON p.id = fe.person_id 
    GROUP BY p.username
""")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} encodings")

print()
print("=" * 50)
cursor.execute("SELECT COUNT(*) FROM face_encodings")
print(f"Total encodings: {cursor.fetchone()[0]}")
print("=" * 50)

conn.close()