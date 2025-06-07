import os
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report
import torch

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

# Build label list and map
unique_labels = sorted(set(tag for tags in dataset["ner_tags"] for tag in tags))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

# Encode labels
def encode_labels(example):
    example["labels"] = [label2id[tag] for tag in example["ner_tags"]]
    return example

dataset = dataset.map(encode_labels)

# Load tokenizer and model
model_name = "google/mobilebert-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(unique_labels), id2label=id2label, label2id=label2id)

# Tokenization
def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    word_ids = tokenized_inputs.word_ids()
    prev_word_id = None
    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
        elif word_id != prev_word_id:
            labels.append(example["labels"][word_id])
        else:
            labels.append(example["labels"][word_id] if True else -100)  # Use True to label all subtokens
        prev_word_id = word_id
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=False)

# Training setup
args = TrainingArguments(
    output_dir="../models/mobilebert-ner",
    evaluation_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=1,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
)

data_collator = DataCollatorForTokenClassification(tokenizer)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained("../models/mobilebert-ner")
tokenizer.save_pretrained("../models/mobilebert-ner")
