import pandas as pd
from sklearn.model_selection import train_test_split

# Load Data
df = pd.read_csv("data/processed/cleaned_reviews.csv")
df.dropna(subset = ["cleaned_text", "label"])
df["label"] = df["label"].astype(int)
df["cleaned_text"] = df["cleaned_text"].astype(str)

# Split Data
train_df, test_df = train_test_split(
    df, test_size = 0.2, random_state = 42, stratify=df["label"])

# Save Test Dataset
output_path = "data/processed/test_data.csv"
df = pd.DataFrame(test_df)
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} reviews to {output_path}")