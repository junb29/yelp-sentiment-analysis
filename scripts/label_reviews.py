import pandas as pd

# Load filtered reviews
df = pd.read_csv("data/processed/asian_us_reviews.csv")

# Drop neutral reviews (3 stars)
df = df[df["stars"] != 3]

# Map stars to binary sentiment
df["label"] = df["stars"].apply(lambda x: 1 if x >= 4 else 0)

# Calculate the percentage of negative reviews
neg_review = (df["label"] == 0).sum()
neg_review_rate = (neg_review / len(df)) * 100 # 20.25%

# Save labeled data
df.to_csv("data/processed/labeled_reviews.csv", index=False)
print(f"Saved {len(df)} labeled reviews to data/processed/labeled_reviews.csv")
print(f"{neg_review_rate:.2f}% of the reviews are negative")
