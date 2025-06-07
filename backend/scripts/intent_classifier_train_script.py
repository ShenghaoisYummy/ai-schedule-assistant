import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import torch
from transformers import AutoTokenizer, AutoModel
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

def load_and_prepare_data(file_path):
    """Load and preprocess data"""
    try:
        # Read JSON data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} examples")
        
        # Ensure text column is string type
        df['text'] = df['text'].astype(str)
        
        # View class distribution
        intent_counts = df['intent'].value_counts()
        logger.info(f"Intent distribution:\n{intent_counts}")
        
        # Encode labels
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['intent'])
        
        # Split dataset
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=SEED, stratify=df['intent']
        )
        
        logger.info(f"Training set size: {len(train_df)}, Test set size: {len(test_df)}")
        
        return train_df, test_df, label_encoder
    
    except Exception as e:
        logger.error(f"Data loading and preprocessing failed: {e}")
        raise

def get_bert_embeddings(texts, model_name="distilbert-base-uncased"):
    """Extract BERT embeddings for text"""
    try:
        # Ensure texts is a list of strings
        texts = [str(text) for text in texts]
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        
        # List to store embeddings
        embeddings = []
        
        # Process texts in batches to avoid memory issues
        batch_size = 16
        for i in tqdm(range(0, len(texts), batch_size), desc="Getting BERT embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            # Encode text
            encoded_input = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Move encodings to device
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            
            # No gradient calculation
            with torch.no_grad():
                # Get BERT outputs
                outputs = model(**encoded_input)
                
                # Use [CLS] token representation as text embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        # Convert to NumPy array
        embeddings = np.array(embeddings)
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        
        return embeddings
    
    except Exception as e:
        logger.error(f"Failed to get BERT embeddings: {e}")
        raise

def train_and_evaluate_classifier(X_train, y_train, X_test, y_test, label_encoder):
    """Train and evaluate classifiers"""
    try:
        # 1. Train SVM classifier
        logger.info("Training SVM classifier...")
        svm_params = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'rbf']
        }
        
        svm = GridSearchCV(
            SVC(probability=True, random_state=SEED),
            svm_params,
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        svm.fit(X_train, y_train)
        logger.info(f"SVM best parameters: {svm.best_params_}")
        
        # Evaluate SVM
        svm_preds = svm.predict(X_test)
        svm_report = classification_report(
            y_test, 
            svm_preds, 
            target_names=label_encoder.classes_
        )
        logger.info(f"SVM classification report:\n{svm_report}")
        
        # 2. Train Random Forest classifier
        logger.info("Training Random Forest classifier...")
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        rf = GridSearchCV(
            RandomForestClassifier(random_state=SEED),
            rf_params,
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        rf.fit(X_train, y_train)
        logger.info(f"Random Forest best parameters: {rf.best_params_}")
        
        # Evaluate Random Forest
        rf_preds = rf.predict(X_test)
        rf_report = classification_report(
            y_test, 
            rf_preds, 
            target_names=label_encoder.classes_
        )
        logger.info(f"Random Forest classification report:\n{rf_report}")
        
        # Return the best performing classifier
        if svm.best_score_ > rf.best_score_:
            logger.info("SVM performs better, returning SVM classifier")
            return svm, "SVM"
        else:
            logger.info("Random Forest performs better, returning Random Forest classifier")
            return rf, "RandomForest"
    
    except Exception as e:
        logger.error(f"Training and evaluating classifiers failed: {e}")
        raise

def test_on_examples(classifier, label_encoder, model_name="distilbert-base-uncased"):
    """Test classifier on some examples"""
    examples = [
        "Add a meeting with Zhang tomorrow at 3 PM",
        "Move my dentist appointment to next Thursday",
        "Cancel my flight to Shanghai",
        "What meetings do I have tomorrow?",
        "Hello, how are you today?"
    ]
    
    # Get BERT embeddings for examples
    example_embeddings = get_bert_embeddings(examples, model_name)
    
    # Predict intents
    predictions = classifier.predict(example_embeddings)
    intent_names = label_encoder.inverse_transform(predictions)
    
    # Print results
    logger.info("\nExample test results:")
    for example, intent in zip(examples, intent_names):
        logger.info(f"Text: {example}")
        logger.info(f"Predicted intent: {intent}\n")

def main():
    try:
        # 1. Load and prepare data
        logger.info("Loading and preparing data...")
        train_df, test_df, label_encoder = load_and_prepare_data("../data/intent_classification_data.json")
        
        # 2. Get BERT embeddings
        logger.info("Getting BERT embeddings for training set...")
        # Key fix: ensure we pass a list, not a pandas Series
        X_train = get_bert_embeddings(train_df['text'].tolist())
        
        logger.info("Getting BERT embeddings for test set...")
        X_test = get_bert_embeddings(test_df['text'].tolist())
        
        # 3. Train and evaluate classifiers
        logger.info("Training and evaluating classifiers...")
        best_classifier, classifier_name = train_and_evaluate_classifier(
            X_train, train_df['label'].values,
            X_test, test_df['label'].values,
            label_encoder
        )
        
        # 4. Test on examples
        test_on_examples(best_classifier, label_encoder)
        
        
        #import pickle
        #model_filename = f"intent_classifier_{classifier_name}.pkl"
        #with open(model_filename, 'wb') as f:
        #    pickle.dump(best_classifier, f)
        
        #encoder_filename = "label_encoder.pkl"
        #with open(encoder_filename, 'wb') as f:
        #    pickle.dump(label_encoder, f)
        
        #logger.info(f"Model saved as {model_filename}")
        #logger.info(f"Label encoder saved as {encoder_filename}")
        # 5. Save model


        # 5. Save model
        import pickle
        import os
        models_dir = "../models/intent_classifier"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            logger.info(f"Created directory: {models_dir}")

       
        model_filename = os.path.join(models_dir, f"intent_classifier_{classifier_name}.pkl")
        with open(model_filename, 'wb') as f:
            pickle.dump(best_classifier, f)

        encoder_filename = os.path.join(models_dir, "label_encoder.pkl")
        with open(encoder_filename, 'wb') as f:
            pickle.dump(label_encoder, f)

        logger.info(f"Model saved as {model_filename}")
        logger.info(f"Label encoder saved as {encoder_filename}")
        
        # 6. Provide example code for using the saved model
        logger.info("\nExample code for using the saved model:")
        logger.info("""
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load model and label encoder
with open("intent_classifier_SVM.pkl", "rb") as f:
    classifier = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

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

# Prediction function
def predict_intent(text):
    embedding = get_embedding(text)
    prediction = classifier.predict(embedding)
    intent = label_encoder.inverse_transform(prediction)[0]
    return intent

# Test
text = "Schedule a meeting tomorrow at 2 PM"
intent = predict_intent(text)
print(f"Text: {text}")
print(f"Predicted intent: {intent}")
        """)
        
    except Exception as e:
        logger.error(f"Main function execution failed: {e}")

if __name__ == "__main__":
    main()