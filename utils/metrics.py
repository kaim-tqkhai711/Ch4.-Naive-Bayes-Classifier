from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(model, vectorizer, texts, labels):
    """
    Evaluate the fitted model on the given texts/labels.
    The toy dataset is too small to split, so this reports training-set fit
    (i.e. how well the model has memorized the demo reviews).
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
