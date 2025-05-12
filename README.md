# Chatbot Calendar Assistant

## Project Overview
This is a chatbot calendar assistant based on Flutter and Flask, supporting natural language processing to add, query, and delete calendar events.

## Features
- Natural language understanding and intent recognition
- Calendar event management (add, query, delete)
- Responsive UI design
- Offline data storage

## Technology Stack
- Frontend: Flutter
- Backend: Flask
- Database: SQLite
- NLP: Custom intent and entity recognition model

## Running Instructions
### Backend Setup
1. Navigate to backend directory: `cd backend`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the server: `python app.py`

### Frontend Setup
1. Navigate to frontend directory: `cd frontend`
2. Install dependencies: `flutter pub get`
3. Run the app: `flutter run`

## Usage Examples
- To add an event: "add a meeting on May 12th"
- To query events: "what do I have on May 12th?"
- To delete an event: "delete the meeting on May 12th"