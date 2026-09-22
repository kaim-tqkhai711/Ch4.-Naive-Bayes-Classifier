import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB

VECTORIZERS = {
    "count": CountVectorizer,
    "tfidf": TfidfVectorizer,
}

NB_VARIANTS = {
    "multinomial": MultinomialNB,
    "bernoulli": BernoulliNB,
    "complement": ComplementNB,
}

# sklearn's default English stop-word list drops "not"/"no"/etc. before n-grams are
# built, which silently kills negation bigrams like "not good" -> "movie good".
# Keep negation words in the vocabulary so ngram_range=(1, 2) can actually capture them.
_NEGATION_WORDS = {"not", "no", "nor", "never", "none", "cannot", "without"}
NEGATION_SAFE_STOP_WORDS = sorted(ENGLISH_STOP_WORDS - _NEGATION_WORDS)


def train_naive_bayes(
    texts,
    labels,
    vectorizer_type="count",
    nb_variant="multinomial",
    ngram_range=(1, 1),
    stop_words=NEGATION_SAFE_STOP_WORDS,
    min_df=5,
):
    """Fit a (CountVectorizer|TfidfVectorizer) + (Multinomial|Bernoulli|Complement)NB pipeline."""
    vectorizer_cls = VECTORIZERS[vectorizer_type]
    nb_cls = NB_VARIANTS[nb_variant]

    vectorizer = vectorizer_cls(stop_words=stop_words, min_df=min_df, ngram_range=ngram_range)
    X_train = vectorizer.fit_transform(texts)
    model = nb_cls()
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


def explain_prediction(model, vectorizer, text, top_k_phrases=5):
    """
    Word/phrase attribution for a single prediction via feature ablation:
    for each n-gram present in the input, zero it out and measure how much
    the (log P(pos) - log P(neg)) score shifts. Model-agnostic across
    Multinomial/Bernoulli/Complement NB since it only relies on
    `_joint_log_likelihood`, which every sklearn discrete NB variant shares.

    Returns:
        {
            "unigram_contribution": {word: score, ...},
            "phrases": [(ngram_text, score), ...]  # multi-word ngrams, sorted by |score| desc
        }
    """
    X = vectorizer.transform([text]).tocsr()
    base_ll = model._joint_log_likelihood(X)[0]
    base_score = base_ll[1] - base_ll[0]

    vocab = vectorizer.get_feature_names_out()
    unigram_contribution = {}
    phrase_contribution = {}

    for idx in sorted(set(X.indices)):
        X_ablated = X.copy()
        X_ablated[0, idx] = 0
        X_ablated.eliminate_zeros()
        ll = model._joint_log_likelihood(X_ablated)[0]
        score = ll[1] - ll[0]
        contribution = base_score - score

        ngram = vocab[idx]
        if " " in ngram:
            phrase_contribution[ngram] = contribution
        else:
            unigram_contribution[ngram] = contribution

    phrases = sorted(phrase_contribution.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k_phrases]

    return {
        "unigram_contribution": unigram_contribution,
        "phrases": phrases,
    }
