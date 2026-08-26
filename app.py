from flask import Flask, render_template, request

from models.nb_model import train_naive_bayes, predict_sentiment, get_top_words
from utils.data_loader import load_review_data, get_dataset_stats, DEMO_SAMPLES
from utils.metrics import evaluate_model
from visualization.text_visualization import word_frequency

app = Flask(__name__)

# ── Train once at startup ────────────────────────────────────────────────
texts, labels = load_review_data()
model, vectorizer = train_naive_bayes(texts, labels)

stats = get_dataset_stats(texts, labels)
metrics = evaluate_model(model, vectorizer, texts, labels)
top_words = get_top_words(model, vectorizer, top_n=5)
frequency = word_frequency(texts, top_n=10)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    review = ""
    proba = None

    if request.method == "POST":
        review = request.form["review"]
        pred, proba = predict_sentiment(model, vectorizer, review)
        result = "Positive" if pred == 1 else "Negative"

    return render_template(
        "index.html",
        result=result,
        review=review,
        proba=proba,
        stats=stats,
        metrics=metrics,
        top_words=top_words,
        frequency=frequency,
        demo_samples=DEMO_SAMPLES,
    )


if __name__ == "__main__":
    app.run(debug=True)
