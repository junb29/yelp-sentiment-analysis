import pandas as pd
import json

# Path to Yelp business dataset
business_path = "data/raw/yelp_academic_dataset_business.json"

# Store matching businesses
businesses = []

with open(business_path, "r") as f:
    for line in f:
        b = json.loads(line)
        categories = b.get("categories", "")
        if (
            categories
            and any(res_keyword in categories for res_keyword in ["Food", "Restaurants"])
            and any(food_keyword in categories for food_keyword in ["Asian", "Korean", "Chinese", "Vietnamese", "Japanese", "Thai", "Sushi", "Ramen", "Asian Fusion"])
            and b.get("review_count", 0) >= 50 
        ):
            businesses.append({
                "business_id": b["business_id"],
                "name": b["name"],
                "categories": b.get("categories"),
                "stars": b["stars"],
                "review_count": b["review_count"],
                "city": b.get("city"),
                "address": b.get("address")
            })

# Convert to DataFrame
df = pd.DataFrame(businesses)

# Save filtered list
df.to_csv("data/processed/asian_us_businesses.csv", index=False)
print(f"Saved {len(df)} Asian restaurants in US to data/processed/asian_ca_businesses.csv")
