import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import torch

# Load Data
df = pd.read_csv("data/processed/cleaned_reviews.csv")
df.dropna(subset = ["cleaned_text", "label"])
df["label"] = df["label"].astype(int)
df["cleaned_text"] = df["cleaned_text"].astype(str)

# Split Data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["cleaned_text"].to_list(), 
    df["label"].to_list(),
    test_size = 0.2,
    random_state = 42,
    stratify = df["label"]
)

# Tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(train_texts, truncation = True, padding = True)
test_encodings = tokenizer(test_texts, truncation = True, padding = True)

# Prepare Datasets
train_dataset = Dataset.from_dict({
    "input_ids" : train_encodings["input_ids"],
    "attention_mask" : train_encodings["attention_mask"],
    "labels" : train_labels
})
test_dataset = Dataset.from_dict({
    "input_ids" : test_encodings["input_ids"],
    "attention_mask" : test_encodings["attention_mask"],
    "labels" : test_labels
})

# Model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Setup
training_args = TrainingArguments(
    output_dir = "./models/distilbert-sentiment",
    eval_strategy = "epoch",
    save_strategy = "steps",
    save_steps = 500,
    save_total_limit = 3,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    num_train_epochs = 2,
    weight_decay = 0.01, 
    logging_dir = "./logs",
    logging_steps = 10,
    report_to = "none",
)

data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

# Trainer
trainer = Trainer(
    args = training_args,
    data_collator = data_collator,
    model = model,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    tokenizer = tokenizer,
)

# Train
trainer.train(resume_from_checkpoint = True)