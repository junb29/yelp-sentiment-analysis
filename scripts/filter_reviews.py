import json
import pandas as pd
from tqdm import tqdm

# Load business_ids of Asian restaurants with 50+ reviews
asian_df = pd.read_csv("data/processed/asian_us_businesses.csv")
asian_ids = set(asian_df["business_id"].values)

# Input Yelp review file
review_path = "data/raw/yelp_academic_dataset_review.json"
filtered_reviews = []

with open(review_path, "r") as f:
    for line in tqdm(f, desc="Filtering reviews"):
        r = json.loads(line)
        if r["business_id"] in asian_ids:
            filtered_reviews.append({
                "business_id": r["business_id"],
                "stars": r["stars"],
                "text": r["text"]
            })

# Save to CSV
output_path = "data/processed/asian_us_reviews.csv"
df = pd.DataFrame(filtered_reviews)
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} reviews to {output_path}")
