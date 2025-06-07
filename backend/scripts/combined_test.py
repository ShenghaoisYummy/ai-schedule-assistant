import argparse
import os
import sys
import torch
import pickle
import numpy as np
import json
import datetime
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification

# ---- INTENT CLASSIFIER FUNCTIONS ----

def load_intent_classifier():
    """Load intent classifier model and label encoder"""
    models_dir = "../models/intent_classifier"
    
    model_filename = os.path.join(models_dir, "intent_classifier_SVM.pkl")
    encoder_filename = os.path.join(models_dir, "label_encoder.pkl")
    
    try:
        with open(model_filename, "rb") as f:
            classifier = pickle.load(f)
        
        with open(encoder_filename, "rb") as f:
            label_encoder = pickle.load(f)
        
        print(f"Successfully loaded intent classifier from {model_filename}")
        return classifier, label_encoder
    except FileNotFoundError as e:
        print(f"Error: Could not find the intent classifier files in {models_dir}. Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading intent classifier files: {e}")
        sys.exit(1)

def get_embedding(text, model_name="distilbert-base-uncased"):
    """Extract embeddings using BERT model"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use [CLS] token representation
    embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return embedding

def predict_intent(text, classifier, label_encoder):
    """Predict intent with confidence scores"""
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
    
    # Sort probabilities from highest to lowest
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    
    return intent, confidence, sorted_probs

# ---- NER MODEL FUNCTIONS ----

def load_ner_model(model_path):
    """Load NER model and tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        model.eval()
        
        # Get label mappings from model config
        id2label = model.config.id2label
        label2id = model.config.label2id
        
        print(f"Successfully loaded NER model from {model_path}")
        return model, tokenizer, id2label, label2id
    except Exception as e:
        print(f"Error loading NER model: {e}")
        sys.exit(1)

def predict_ner(text, model, tokenizer, id2label):
    """Predict NER tags for given text"""
    # Basic tokenization - split text into words
    words = text.split()
    
    # Process with the model's tokenizer
    inputs = tokenizer(words, truncation=True, is_split_into_words=True, return_tensors="pt")
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted labels
    predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
    
    # Align predictions with words
    word_ids = inputs.word_ids()
    predicted_labels = []
    current_word = None
    
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        
        if word_id != current_word:
            current_word = word_id
            predicted_labels.append(id2label[predictions[idx]])
    
    return list(zip(words, predicted_labels))

def format_ner_output(ner_results, output_format="json"):
    """Format NER output according to preference"""
    if output_format == "bio":
        return "\n".join([f"{word} {label}" for word, label in ner_results])
    elif output_format == "json":
        result = {"entities": []}
        current_entity = None
        
        for i, (word, label) in enumerate(ner_results):
            if label.startswith("B-"):
                if current_entity:
                    result["entities"].append(current_entity)
                entity_type = label[2:]  # Remove "B-"
                current_entity = {"type": entity_type, "text": word, "start": i}
            elif label.startswith("I-") and current_entity:
                current_entity["text"] += " " + word
            elif label == "O":
                if current_entity:
                    result["entities"].append(current_entity)
                    current_entity = None
        
        if current_entity:
            result["entities"].append(current_entity)
            
        return result
    else:  # Default to simple format
        return "\n".join([f"{word}: {label}" for word, label in ner_results])

# ---- COMBINED TEST FUNCTION ----

def run_combined_test(text, intent_classifier, label_encoder, ner_model, ner_tokenizer, id2label, output_format="json"):
    """Run both intent classification and NER on the input text"""
    # Intent classification
    intent, confidence, sorted_probs = predict_intent(text, intent_classifier, label_encoder)
    
    # NER tagging
    ner_results = predict_ner(text, ner_model, ner_tokenizer, id2label)
    ner_formatted = format_ner_output(ner_results, output_format)
    
    # Combine results
    if output_format == "json":
        combined_result = {
            "text": text,
            "intent": {
                "name": intent,
                "confidence": float(confidence),
                "all_intents": [{"name": name, "confidence": float(prob)} for name, prob in sorted_probs[:3]]
            },
            "entities": ner_formatted["entities"]
        }
        return json.dumps(combined_result, indent=2)
    else:
        result = f"Text: {text}\n\n"
        result += f"Intent: {intent} (Confidence: {confidence:.2%})\n"
        result += "Top 3 intents:\n"
        for name, prob in sorted_probs[:3]:
            result += f"  - {name}: {prob:.2%}\n"
        result += "\nEntities:\n"
        
        if output_format == "bio":
            result += ner_formatted
        else:  # Simple format
            if isinstance(ner_formatted, str):
                result += ner_formatted
            else:
                # If it's the parsed JSON entities
                for entity in ner_formatted["entities"]:
                    result += f"  - {entity['type']}: {entity['text']}\n"
                    
        return result

def get_schedule_test_cases():
    """Return a list of test cases related to schedule management"""
    today = datetime.datetime.now()
    tomorrow = today + datetime.timedelta(days=1)
    next_week = today + datetime.timedelta(days=7)
    
    today_str = today.strftime("%B %d")
    tomorrow_str = tomorrow.strftime("%B %d")
    next_week_str = next_week.strftime("%B %d")
    
    test_cases = [
        "Schedule a meeting with John Smith tomorrow at 2pm in Conference Room A",
        f"Remind me to call Sarah about the project on {today_str} at 4:30pm",
        f"Set up a team discussion for {next_week_str} at 10am with the marketing team",
        "Cancel my appointment with Dr. Johnson on Friday at 3pm",
        "Move my 9am meeting with Alex to 11am",
        "What meetings do I have scheduled for tomorrow?",
        "Add 'Prepare quarterly report' to my to-do list for Monday",
        f"Book a flight to New York for {next_week_str}",
        "Create a reminder to send the proposal to client XYZ by end of day",
        "Schedule lunch with Mark and Emily at Cafe Bella on Thursday at 12:30pm",
        f"Set up a video call with the remote team on {tomorrow_str} morning",
        "Reschedule my dentist appointment from Tuesday to Wednesday afternoon",
        "Add a weekly recurring meeting with the development team every Monday at 9am",
        "Block off 2 hours for project work tomorrow afternoon",
        "Schedule a product review meeting next Friday with all department heads"
    ]
    return test_cases

def main():
    parser = argparse.ArgumentParser(description="Combined Intent Classification and NER Testing")
    parser.add_argument("--ner_model_path", type=str, default="../models/mobilebert-ner", 
                        help="Path to saved NER model")
    parser.add_argument("--input", type=str, help="Input text for testing")
    parser.add_argument("--input_file", type=str, help="Input file path with one sentence per line")
    parser.add_argument("--format", type=str, choices=["simple", "bio", "json"], 
                        default="json", help="Output format")
    parser.add_argument("--test", action="store_true", 
                        help="Run with schedule management test cases")
    
    args = parser.parse_args()
    
    # Load intent classifier
    intent_classifier, label_encoder = load_intent_classifier()
    
    # Load NER model
    ner_model, ner_tokenizer, id2label, label2id = load_ner_model(args.ner_model_path)
    
    # Handle test cases
    if args.test:
        test_cases = get_schedule_test_cases()
        print("\n===== RUNNING COMBINED TEST CASES =====\n")
        for i, test_case in enumerate(test_cases):
            print(f"Test #{i+1}: \"{test_case}\"")
            result = run_combined_test(
                test_case, intent_classifier, label_encoder, 
                ner_model, ner_tokenizer, id2label, args.format
            )
            print("\nResults:")
            print(result)
            print("\n" + "-" * 80 + "\n")
        return
    
    # Get input text
    if args.input:
        text = args.input
        result = run_combined_test(
            text, intent_classifier, label_encoder, 
            ner_model, ner_tokenizer, id2label, args.format
        )
        print("\nResults:")
        print(result)
            
    elif args.input_file:
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print("\n===== PROCESSING INPUT FILE =====\n")
            for i, line in enumerate(lines):
                line = line.strip()
                if line:  # Skip empty lines
                    print(f"Line #{i+1}: \"{line}\"")
                    result = run_combined_test(
                        line, intent_classifier, label_encoder, 
                        ner_model, ner_tokenizer, id2label, args.format
                    )
                    print("\nResults:")
                    print(result)
                    print("\n" + "-" * 80 + "\n")
                
        except Exception as e:
            print(f"Error processing input file: {e}")
            sys.exit(1)
    else:
        print("No input provided. Use --input, --input_file, or --test.")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
    
    
# 运行预设的测试用例
# python combined_test.py --test

# 测试单个句子
# python combined_test.py --input "Schedule a meeting with John tomorrow at 3pm"

# 从文件中读取多个句子进行测试
# python combined_test.py --input_file test_sentences.txt

# 指定输出格式
# python combined_test.py --test --format simple
# python combined_test.py --test --format bio

# 指定NER模型路径（如果不是默认路径）
# python combined_test.py --ner_model_path "/path/to/your/ner_model" --input "Set up a call with marketing team"