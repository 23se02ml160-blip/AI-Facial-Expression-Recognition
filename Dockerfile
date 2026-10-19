FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and dlib
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Hugging Face requires the app to run on port 7860
EXPOSE 7860

# Start the app using Gunicorn (1 worker to save RAM)
CMD gunicorn --bind 0.0.0.0:$PORT --timeout 300 --workers 1 app:app