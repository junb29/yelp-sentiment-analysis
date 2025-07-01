import pandas as pd
import re

# Load labeled data
df = pd.read_csv("data/processed/labeled_reviews.csv")

def clean_text(text):
    text = text.lower()                              # lowercase
    text = re.sub(r"http\S+", "", text)              # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)             # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()         # remove extra whitespace
    return text

df["cleaned_text"] = df["text"].astype(str).apply(clean_text)

# Drop old reviews
df.drop(columns=["text"], inplace=True)

# Save cleaned data
df.to_csv("data/processed/cleaned_reviews.csv", index=False)
print(f"Saved cleaned reviews to data/processed/cleaned_reviews.csv")