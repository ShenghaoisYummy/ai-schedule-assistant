import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import sys
import datetime

def load_model(model_path):
    """Load NER model and tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        model.eval()
        
        # Get label mappings from model config
        id2label = model.config.id2label
        label2id = model.config.label2id
        
        return model, tokenizer, id2label, label2id
    except Exception as e:
        print(f"Error loading model: {e}")
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

def format_output(ner_results, output_format):
    """Format output according to user preference"""
    if output_format == "bio":
        return "\n".join([f"{word} {label}" for word, label in ner_results])
    elif output_format == "json":
        import json
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
            
        return json.dumps(result, indent=2)
    else:  # Default to simple format
        return "\n".join([f"{word}: {label}" for word, label in ner_results])

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
    parser = argparse.ArgumentParser(description="Test NER model")
    parser.add_argument("--model_path", type=str, default="../data/mobilebert-ner", 
                        help="Path to saved model")
    parser.add_argument("--input", type=str, help="Input text for NER tagging")
    parser.add_argument("--input_file", type=str, help="Input file path")
    parser.add_argument("--format", type=str, choices=["simple", "bio", "json"], 
                        default="simple", help="Output format")
    parser.add_argument("--test", action="store_true", 
                        help="Run with schedule management test cases")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, id2label, label2id = load_model(args.model_path)
    
    # Handle test cases
    if args.test:
        test_cases = get_schedule_test_cases()
        print("\n===== RUNNING SCHEDULE MANAGEMENT TEST CASES =====\n")
        for i, test_case in enumerate(test_cases):
            print(f"Test case #{i+1}: \"{test_case}\"")
            results = predict_ner(test_case, model, tokenizer, id2label)
            output = format_output(results, args.format)
            print("\nNER Results:")
            print(output)
            print("\n" + "-" * 60 + "\n")
        return
        
    # Get input text
    if args.input:
        text = args.input
        results = predict_ner(text, model, tokenizer, id2label)
        output = format_output(results, args.format)
        print("\nNER Results:")
        print(output)
            
    elif args.input_file:
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            results = predict_ner(text, model, tokenizer, id2label)
            output = format_output(results, args.format)
            print("\nNER Results:")
            print(output)
                
        except Exception as e:
            print(f"Error processing input file: {e}")
            sys.exit(1)
    else:
        print("No input provided. Use --input, --input_file, or --test.")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
    
    
    
# 运行日程管理测试用例
#python ner_test.py --test --format json

# 或者使用其他格式
#python ner_test.py --test --format simple
#python ner_test.py --test --format bio

# 仍然可以使用原来的方式
#python ner_test.py --input "Schedule a meeting with John tomorrow at 3pm" --format json