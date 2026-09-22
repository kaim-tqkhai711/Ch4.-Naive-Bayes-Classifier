import threading

from sklearn.model_selection import train_test_split

from models.nb_model import train_naive_bayes, get_top_words
from utils.data_loader import load_review_data, append_review, get_dataset_stats
from utils.metrics import evaluate_model
from visualization.text_visualization import word_frequency_by_label

NGRAM_LABELS = {"1": (1, 1), "2": (1, 2)}
VECTORIZER_LABELS = {"count", "tfidf"}
NB_VARIANT_LABELS = {"multinomial", "bernoulli", "complement"}


class AppState:
    """
    Global, mutable app state: current model config, the fitted model/vectorizer,
    and all derived stats shown in the UI. Retraining is cheap (~2000 docs) so it
    happens synchronously whenever the config or the dataset changes.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.vectorizer_type = "count"
        self.nb_variant = "multinomial"
        self.ngram_label = "1"
        self._train()

    def _train(self):
        texts, labels = load_review_data()
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        model, vectorizer = train_naive_bayes(
            X_train,
            y_train,
            vectorizer_type=self.vectorizer_type,
            nb_variant=self.nb_variant,
            ngram_range=NGRAM_LABELS[self.ngram_label],
        )

        self.texts, self.labels = texts, labels
        self.model, self.vectorizer = model, vectorizer
        self.stats = get_dataset_stats(texts, labels, train_count=len(X_train), test_count=len(X_test))
        self.metrics = evaluate_model(model, vectorizer, X_test, y_test)
        self.top_words = get_top_words(model, vectorizer, top_n=5)
        self.word_cloud = word_frequency_by_label(texts, labels, top_n=30)

    def set_config(self, vectorizer_type, nb_variant, ngram_label):
        vectorizer_type = vectorizer_type if vectorizer_type in VECTORIZER_LABELS else self.vectorizer_type
        nb_variant = nb_variant if nb_variant in NB_VARIANT_LABELS else self.nb_variant
        ngram_label = ngram_label if ngram_label in NGRAM_LABELS else self.ngram_label

        if (vectorizer_type, nb_variant, ngram_label) == (self.vectorizer_type, self.nb_variant, self.ngram_label):
            return

        with self._lock:
            self.vectorizer_type = vectorizer_type
            self.nb_variant = nb_variant
            self.ngram_label = ngram_label
            self._train()

    def add_review(self, text, label):
        with self._lock:
            append_review(text, label)
            self._train()
