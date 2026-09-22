import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def word_frequency(texts, top_n=10):
    """Word -> occurrence count across all texts, filtering common stopwords and punctuation."""
    freq = {}
    for text in texts:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        for word in words:
            if word not in ENGLISH_STOP_WORDS:
                freq[word] = freq.get(word, 0) + 1
    return sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


def word_frequency_by_label(texts, labels, top_n=30):
    """Word -> occurrence count, split by class label. Returns {"positive": [...], "negative": [...]}."""
    pos_texts = [t for t, l in zip(texts, labels) if l == 1]
    neg_texts = [t for t, l in zip(texts, labels) if l == 0]
    return {
        "positive": word_frequency(pos_texts, top_n=top_n),
        "negative": word_frequency(neg_texts, top_n=top_n),
    }

