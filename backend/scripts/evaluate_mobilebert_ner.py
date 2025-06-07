import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset, Dataset
from sklearn.metrics import classification_report
import numpy as np

# Load model and tokenizer
model_path = "../models/mobilebert-ner"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()

# Load BIO-formatted data
def read_bio_data(filepath):
    tokens, labels, current_tokens, current_labels = [], [], [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "":
                if current_tokens:
                    tokens.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens, current_labels = [], []
            else:
                token, label = line.strip().split()
                current_tokens.append(token)
                current_labels.append(label)
        if current_tokens:
            tokens.append(current_tokens)
            labels.append(current_labels)
    return Dataset.from_dict({"tokens": tokens, "ner_tags": labels})

dataset = read_bio_data("../data/ner_bio_format.txt")

# Get label mappings
unique_labels = sorted(set(tag for tags in dataset["ner_tags"] for tag in tags))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    word_ids = tokenized_inputs.word_ids()
    prev_word_id = None
    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
        elif word_id != prev_word_id:
            labels.append(label2id[example["ner_tags"][word_id]])
        else:
            labels.append(label2id[example["ner_tags"][word_id]])
        prev_word_id = word_id
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

encoded_dataset = dataset.map(tokenize_and_align_labels)

# Evaluation
all_preds = []
all_labels = []

for example in encoded_dataset:
    input_ids = torch.tensor([example["input_ids"]])
    attention_mask = torch.tensor([example["attention_mask"]])
    labels = example["labels"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

    true_labels = [id2label[label] for label in labels if label != -100]
    pred_labels = [id2label[pred] for pred, label in zip(predictions, labels) if label != -100]

    all_preds.extend(pred_labels)
    all_labels.extend(true_labels)

# Print classification report
print(classification_report(all_labels, all_preds))
