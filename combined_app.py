from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import os
import traceback

# Initialize Flask to serve the Flutter app from the 'static' folder
app = Flask(__name__, static_folder='static')
# Enable CORS for your Vercel domain
CORS(app, resources={r"/api/*": {"origins": ["https://ai-schedule-assistant.vercel.app/"]}})
MODELS_LOADED = False

# Try to load ML models, but don't crash if they fail
try:
    from models import ModelManager
    from utils import generate_response
    model_manager = ModelManager()
    MODELS_LOADED = True
    print("✅ ML models loaded successfully.")
except Exception as e:
    print(f"⚠️  Warning: Could not load ML models. The app will run without NLP features. Error: {e}")

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

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
        return jsonify({"response": response, "intent": intent, "entities": entities})
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
    # Use environment variable for port or default to 8080
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port) 