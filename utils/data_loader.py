# Hardcoded toy dataset — 1 = Positive, 0 = Negative
REVIEWS = [
    "Great movie",
    "I loved the acting",
    "Amazing storyline and visuals",
    "Best film I have seen",
    "Terrible acting",
    "Waste of time",
    "Awful plot and boring",
    "I hated this movie",
]

LABELS = [1, 1, 1, 1, 0, 0, 0, 0]

# Quick-fill examples for the UI
DEMO_SAMPLES = {
    "Positive example": "Great movie, I loved it",
    "Negative example": "Terrible movie, a waste of time",
}


def load_review_data():
    """Return the toy (texts, labels) dataset."""
    return REVIEWS, LABELS


def get_dataset_stats(texts, labels):
    """Basic dataset counts for display in the UI/README."""
    return {
        "total_samples": len(texts),
        "positive": sum(1 for l in labels if l == 1),
        "negative": sum(1 for l in labels if l == 0),
    }
