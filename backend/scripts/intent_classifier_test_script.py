import os
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load model and label encoder
models_dir = "../models/intent_classifier"

model_filename = os.path.join(models_dir, "intent_classifier_SVM.pkl")
encoder_filename = os.path.join(models_dir, "label_encoder.pkl")

try:
    with open(model_filename, "rb") as f:
        classifier = pickle.load(f)
    
    with open(encoder_filename, "rb") as f:
        label_encoder = pickle.load(f)
    
    print(f"Successfully loaded model from {model_filename}")
    print(f"Successfully loaded label encoder from {encoder_filename}")
except FileNotFoundError as e:
    print(f"Error: Could not find the model files in {models_dir}. Details: {e}")
except Exception as e:
    print(f"Error loading model files: {e}")

# Function to get BERT embeddings
def get_embedding(text, model_name="distilbert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use [CLS] token representation
    embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return embedding

# Prediction function with confidence
def predict_intent_with_confidence(text):
    embedding = get_embedding(text)
    
    # Get prediction probabilities
    probabilities = classifier.predict_proba(embedding)[0]
    
    # Get the predicted class index
    prediction = classifier.predict(embedding)[0]
    
    # Get the intent name
    intent = label_encoder.inverse_transform([prediction])[0]
    
    # Get confidence (probability of the predicted class)
    confidence = probabilities[prediction]
    
    # Get all class probabilities with labels
    all_probs = {label_encoder.inverse_transform([i])[0]: prob 
                 for i, prob in enumerate(probabilities)}
    
    return intent, confidence, all_probs

# Test examples
test_examples = [
    "Schedule a meeting tomorrow at 2 PM",
    "i want to sleep now",
    "hey, i want to delete my meeting tomorrow",
    "i want to rescheldu my meeting time from 6pm to 4am on 8th May.",
    "hey i love you",
    "i want to go shopping at friday day"
]

for text in test_examples:
    intent, confidence, all_probs = predict_intent_with_confidence(text)
    print(f"\nText: {text}")
    print(f"Predicted intent: {intent}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print("All probabilities:")
    
    # Sort probabilities from highest to lowest
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    for intent_name, prob in sorted_probs:
        print(f"  {intent_name}: {prob:.4f} ({prob*100:.2f}%)")