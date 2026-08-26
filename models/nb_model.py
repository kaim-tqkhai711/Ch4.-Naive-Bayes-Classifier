import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def train_naive_bayes(texts, labels):
    """Fit CountVectorizer + MultinomialNB on the given texts/labels."""
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(texts)
    model = MultinomialNB()
    model.fit(X_train, labels)
    return model, vectorizer


def predict_sentiment(model, vectorizer, text):
    """Return (label, probabilities [neg, pos]) for a single input text."""
    X = vectorizer.transform([text])
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    return pred, proba


def get_top_words(model, vectorizer, top_n=5):
    """
    Most indicative words per class, ranked by log-probability difference.
    Mirrors "feature importance" for a Naive Bayes model.
    """
    vocab = np.array(vectorizer.get_feature_names_out())
    log_prob = model.feature_log_prob_  # shape: (n_classes, n_features)
    diff = log_prob[1] - log_prob[0]    # positive - negative

    pos_idx = np.argsort(diff)[::-1][:top_n]
    neg_idx = np.argsort(diff)[:top_n]

    return {
        "positive": list(vocab[pos_idx]),
        "negative": list(vocab[neg_idx]),
    }
