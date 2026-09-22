from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(model, vectorizer, texts, labels):
    """
    Evaluate the fitted model on the given evaluation (test) texts and labels.
    Returns standard classification metrics (accuracy, precision, recall, f1).
    """
    X = vectorizer.transform(texts)
    preds = model.predict(X)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }
