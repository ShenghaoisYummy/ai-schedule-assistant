# Stage 1: Build the Flutter Web App
# We use a Node image because it's a common base that includes git and other tools needed for the Flutter install script.
FROM node:18-alpine AS flutter-builder

# Install necessary tools for Flutter SDK
RUN apk add --no-cache git curl unzip bash

# Clone and install the Flutter SDK
RUN git clone https://github.com/flutter/flutter.git -b stable /flutter
ENV PATH="/flutter/bin:${PATH}"

# Prepare the Flutter build environment
WORKDIR /app/frontend

# Copy only the necessary files first to leverage Docker cache
COPY frontend/pubspec.yaml frontend/pubspec.lock ./
RUN flutter pub get

# Copy the rest of the frontend source code
COPY frontend/ .

# Build the Flutter web app for release
# This creates a highly optimized build in /app/frontend/build/web
RUN flutter build web --release --web-renderer html

# ---

# Stage 2: Create the Python Production Server
FROM python:3.9-slim

WORKDIR /app

# Install Python dependencies, including a production-grade server (gunicorn)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install gunicorn

# Copy all backend code (Flask app, models, etc.)
COPY backend/ .

# Copy the already-built Flutter app from the first stage
# This places the Flutter app into a 'static' folder that Flask can serve
COPY --from=flutter-builder /app/frontend/build/web ./static

# Create necessary directories that your app might need to write to
RUN mkdir -p models/intent_classifier models/ner/mobilebert-ner logs data

# Use COPY with a heredoc to create the combined Flask app.
# This is more robust than using 'RUN cat'.
COPY <<EOF /app/combined_app.py
from flask import Flask, jsonify, send_from_directory, request
import os
import traceback
from models import ModelManager
from utils import generate_response

# Initialize Flask to serve the Flutter app from the 'static' folder
app = Flask(__name__, static_folder='static')
MODELS_LOADED = False

# Try to load ML models, but don't crash if they fail
try:
    model_manager = ModelManager()
    MODELS_LOADED = True
    print("✅ ML models loaded successfully.")
except Exception as e:
    print(f"⚠️  Warning: Could not load ML models. The app will run without NLP features. Error: {e}")

# API route for analyzing text
@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    if not MODELS_LOADED:
        return jsonify({"response": "Sorry, the AI features are currently unavailable."})

    try:
        data = request.json
        text = data['text']
        intent = model_manager.predict_intent(text)
        entities = model_manager.predict_ner(text)
        response = generate_response(intent, entities)
        return jsonify({"response": response})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# This is the main route that serves your Flutter app
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_flutter_app(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        # Always serve index.html for any route not found, so Flutter's routing can take over
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # Railway provides the PORT environment variable
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
EOF

# Command to run the application
# Use gunicorn for a production-ready server
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "combined_app:app"]