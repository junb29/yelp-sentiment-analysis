# Yelp Review Sentiment Classification

This project predicts the sentiment of Yelp reviews for over 3,000 Asian restaurants across the U.S., using both traditional machine learning and fine-tuned transformer models.

---

## Project Overview

- **Goal:** Predict whether a Yelp review is positive or negative.
- **Dataset:** Over 500,000 reviews filtered from the official [Yelp Open Dataset](https://www.yelp.com/dataset).
- **Models Used:**
  - Baseline: TF-IDF + Logistic Regression
  - Advanced: Fine-tuned DistilBERT via HuggingFace Transformers

---

## Data Processing

1. **Filtering:**
   - Extracted Asian restaurants from the `yelp_academic_dataset_business.json` file using category keywords.
   - Filtered reviews only from those restaurants.
   - Limited to restaurants with ≥50 reviews.

2. **Labeling:**
   - Reviews with stars ≥4 → labeled **positive (1)**
   - Reviews with stars ≤2 → labeled **negative (0)**
   - 3-star reviews were excluded for clearer separation.

3. **Cleaning:**
   - Removed special characters, punctuation, excessive whitespace, and converted text to lowercase using regex.

---

## Models & Evaluation

### Baseline Model
- **Vectorization:** TF-IDF
- **Classifier:** Logistic Regression
- **Accuracy:** ~96%
- **Precision:** ~93% (Negative) / ~97% (Positive)
- **Recall:** ~89% (Negative) / ~98% (Positive)
- **Macro F1-score:** ~94%
- **Evaluation:** Confusion matrix, classification report
- **Limitation:** Can’t understand context, sarcasm, or nuance.

### Transformer Model
- **Model:** DistilBERT (fine-tuned on review text)
- **Framework:** HuggingFace + PyTorch
- **Accuracy:** ~98% / Higher contextual understanding and robustness
- **Precision:** ~95% (Negative) / ~99% (Positive)
- **Recall:** ~95% (Negative) / ~99% (Positive)
- **Macro F1-score:** ~97%
- **Evaluation:** Confusion matrix, classification report

* Both models use 20% of the entire dataset as test dataset (100,000 reviews)

* Precision indicates how many were actually correct within all predicted positives/negatives

* Recall indicates how many were correctly predicted within actual positives/negatives

### Misclassification Examples from Transformer Model

- **Misclassification:** Negative reivew but the model predicted positive

Review: love green mint if you love cilantro you will love the sandwich topped with an egg its the best fresh ingredients and delicious edit i dont know if there are new owners but this place has changed and not for the better the place is looking run down and dirty paint is chipping off the walls menu board is rewritten and shows a whole new drink menu seems their focus has shifted to a drink place some new menu items are available and i ordered the bao which tastes weird wish it would go back to the old place lack of freshness and attention to detail

- **Misclassification:** Negative reivew but the model predicted positive

Review: the lady and i decided to eat here with friends for the restaurant week menu and had a pretty enjoyable time overall i started off with the sopa azteca and she got the queso fundido both of which were pretty excellent appetizers i particularly enjoyed the blue corn tortilla chip strips that they poured my soup over since it offered a fun and crunchy world of taste my ceviche de salmon that came after that was well done too and i could tell it was all very fresh not a huge fan of foam in general but i didnt mind the corn foam the main course was the only real issue with my experience since i ordered the stuffed poblano pepper with ground beef and a creamy sauce it was delicious but as far as entrees are concerned i could have had three times their portion and still would have not been full the tres leches dessert saved it all though it was way better than ones ive had before and thats saying a lot the atmosphere could have been a little darker for being a pretty small dining area but no glaring complaints here

- **Misclassification:** Positive reivew but the model predicted negative

Review: food good not delicious and mouth watering like some people expect cmon its a takeout if you want that then go somewhere else duh delivery never over an hour but common sense folks if its a rainy friday night when everyones home netflixing and chilling dont expect your food in minutes even if they say minutes fry yourselves some eggs if you want quick service overall i like it here tho their terayki isnt the best so i stick with sushi this is one of my spots when i dont feel like cooking and dont feel like eating greasy shitty mcdonalds food

---

## Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/yelp-review-sentiment-classifier.git
cd yelp-review-sentiment-classifier
pip install -r requirements.txt
```

Make sure to place the Yelp dataset in a `data/raw/` directory before running.

---

## How to Run

**Filter Asian Restaurants:**
```bash
python scripts/filter_asian.py
```

**Filter Asian Restaurants Reviews:**
```bash
python scripts/filter_reviews.py
```

**Label Reviews (Positive/Negative):**
```bash
python scripts/label_reviews.py
```

**Clean Reviews:**
```bash  
python scripts/clean_reviews.py
```

**Train Baseline Model:**
```bash
python scripts/train_baseline.py
```

**Fine-tune Transformer:**
```bash
python scripts/train_transformer_sentiment.py
```

**Save Test Dataset:**
```bash
python scripts/build_test_dataset.py
```

**Predict & Evaluate:**
```bash
python scripts/evaluation.py
```

---

## Project Structure

```
yelp_rating_predictor/
│
├── data/
│   ├── raw/   # Original Yelp dataset (not included in repo)
│   ├── processed/   # Cleaned and labeled dataset
├── models/
│   ├── distilbert-sentiment/ # Checkpoints of the fine-tuned model
├── scripts/
│   ├── filter_asian.py
|   ├── filter_reivews.py
|   ├── label_reivews.py
│   ├── clean_reviews.py
│   ├── train_baseline.py
│   ├── train_transformer_sentiment.py
│   ├── build_test_dataset.py
│   ├── visualization.py
├── requirements.txt
└── README.md
```

Note: The full Yelp dataset is not included due to size and licensing. Processed samples can be regenerated using the provided scripts.

---

## Future Work

- Multi-class classification (predict star rating from 1 to 5)
- Topic clustering using BERTopic or LDA
- Streamlit app for live demo

---

## Requirements

```
pandas
scikit-learn
matplotlib
seaborn
tqdm
transformers
datasets
torch
```
