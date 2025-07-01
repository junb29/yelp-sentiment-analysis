import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load model and tokenizer
model_path = "models/distilbert-sentiment/checkpoint-105182"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load and clean test data
test_df = pd.read_csv("data/processed/test_data.csv").dropna(subset=["cleaned_text", "label"])
test_df["label"] = test_df["label"].astype(int)
test_df["cleaned_text"] = test_df["cleaned_text"].astype(str)

test_texts = test_df["cleaned_text"].tolist()
test_labels = test_df["label"].tolist()

# Tokenize
encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors="pt")
input_ids = encodings["input_ids"]
attention_mask = encodings["attention_mask"]
labels = torch.tensor(test_labels)

# Create dataset and loader
dataset = TensorDataset(input_ids, attention_mask, labels)
loader = DataLoader(dataset, batch_size=16)

# Run prediction in batches
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics
print("Classification Report:")
print(classification_report(all_labels, all_preds, digits=4))

cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Incorrect predictions
wrong_indices = [i for i, (p, t) in enumerate(zip(all_preds, all_labels)) if p != t]
print("\nExamples of incorrect predictions:")
for i in wrong_indices[:5]:
    print(f"Review: {test_texts[i]}")
    print(f"True: {all_labels[i]}, Pred: {all_preds[i]}\n")

