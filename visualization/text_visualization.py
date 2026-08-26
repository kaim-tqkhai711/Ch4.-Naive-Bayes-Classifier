def word_frequency(texts, top_n=10):
    """Word -> occurrence count across all texts, sorted descending."""
    freq = {}
    for text in texts:
        for word in text.lower().split():
            freq[word] = freq.get(word, 0) + 1
    return sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
