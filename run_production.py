from waitress import serve
from app import app
import webbrowser
from threading import Timer

# Open browser automatically
Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8080")).start()

# Start production server
print("Starting production server on http://127.0.0.1:8080")
serve(app, host='0.0.0.0', port=8080, threads=4)