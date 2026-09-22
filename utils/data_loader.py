import os
import csv
import nltk
from sklearn.model_selection import train_test_split

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "movie_reviews.csv")

# Quick-fill examples for the UI
DEMO_SAMPLES = {
    "Positive example": "An absolute masterpiece with stunning visuals, brilliant acting, and an unforgettable story.",
    "Negative example": "A complete disaster and waste of time, with awful dialogue and terrible pacing.",
    "Subtle positive": "While the plot was somewhat predictable, the emotional depth and outstanding performances made it well worth watching.",
    "Subtle negative": "Great cinematography and talented actors, but ruined by a dull script and messy direction.",
}


def ensure_nltk_dataset():
    """Ensure the NLTK movie_reviews corpus is downloaded."""
    try:
        from nltk.corpus import movie_reviews
        movie_reviews.fileids()
    except (LookupError, AttributeError):
        nltk.download("movie_reviews", quiet=True)


def load_review_data(csv_path=CSV_PATH):
    """
    Load the movie reviews dataset (starts as the 2000-sample NLTK Movie Reviews
    corpus: 1000 pos, 1000 neg; grows as reviews are added via the Dataset Explorer).
    Uses a local CSV cache for near-instant loading (0.05s).
    Returns (texts, labels) where 1 = Positive, 0 = Negative.
    """
    if os.path.exists(csv_path):
        texts, labels = [], []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 2:
                    texts.append(row[0])
                    labels.append(int(row[1]))
        if len(texts) >= 2000:
            return texts, labels

    # Fallback to downloading/loading directly from NLTK corpus
    ensure_nltk_dataset()
    from nltk.corpus import movie_reviews
    fileids = movie_reviews.fileids()
    texts = [movie_reviews.raw(f) for f in fileids]
    labels = [1 if f.startswith("pos") else 0 for f in fileids]

    # Save to CSV cache for future instant startup
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["review", "sentiment"])
        for text, label in zip(texts, labels):
            writer.writerow([text, label])

    return texts, labels


def append_review(text, label, csv_path=CSV_PATH):
    """Append a single (review, label) row to the CSV cache."""
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([text, label])


def get_train_test_data(test_size=0.2, random_state=42):
    """
    Load dataset and return stratified train/test split.
    """
    texts, labels = load_review_data()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    return texts, labels, X_train, X_test, y_train, y_test


def get_dataset_stats(texts, labels, train_count=None, test_count=None):
    """Dataset counts and train/test breakdown for display in the UI."""
    stats = {
        "total_samples": len(texts),
        "positive": sum(1 for l in labels if l == 1),
        "negative": sum(1 for l in labels if l == 0),
    }
    if train_count is not None:
        stats["train_samples"] = train_count
    if test_count is not None:
        stats["test_samples"] = test_count
    return stats

