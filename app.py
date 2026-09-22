import re

from flask import Flask, render_template, request, redirect, url_for

from app_state import AppState
from models.nb_model import predict_sentiment, explain_prediction
from utils.data_loader import DEMO_SAMPLES

app = Flask(__name__)
state = AppState()

DATASET_PAGE_SIZE = 50


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        state.set_config(
            request.args.get("vectorizer", state.vectorizer_type),
            request.args.get("nb", state.nb_variant),
            request.args.get("ngram", state.ngram_label),
        )

    result = None
    review = ""
    proba = None
    tokens = None
    phrases = None

    if request.method == "POST":
        review = request.form["review"]
        pred, proba = predict_sentiment(state.model, state.vectorizer, review)
        result = "Positive" if pred == 1 else "Negative"

        explanation = explain_prediction(state.model, state.vectorizer, review)
        tokens, phrases = build_highlight_tokens(review, explanation)

    return render_template(
        "index.html",
        result=result,
        review=review,
        proba=proba,
        tokens=tokens,
        phrases=phrases,
        stats=state.stats,
        metrics=state.metrics,
        top_words=state.top_words,
        word_cloud=state.word_cloud,
        demo_samples=DEMO_SAMPLES,
        vectorizer_type=state.vectorizer_type,
        nb_variant=state.nb_variant,
        ngram_label=state.ngram_label,
    )


def build_highlight_tokens(review, explanation):
    """Split raw text into display tokens, pairing word tokens with their attribution score."""
    unigram_contribution = explanation["unigram_contribution"]
    max_abs = max([abs(v) for v in unigram_contribution.values()], default=0) or 1.0

    tokens = []
    for chunk in re.findall(r"\w+|\W+", review):
        if chunk.strip() and re.match(r"^\w+$", chunk):
            score = unigram_contribution.get(chunk.lower())
            intensity = round(abs(score) / max_abs, 3) if score is not None else 0
            tokens.append({"text": chunk, "score": score, "intensity": intensity})
        else:
            tokens.append({"text": chunk, "score": None, "intensity": 0})

    phrases = [{"text": ngram, "score": score} for ngram, score in explanation["phrases"]]
    return tokens, phrases


@app.route("/dataset")
def dataset():
    label_filter = request.args.get("label", "all")
    query = request.args.get("q", "").strip().lower()
    page = max(int(request.args.get("page", 1)), 1)
    added = request.args.get("added") == "1"

    rows = list(enumerate(zip(state.texts, state.labels)))
    if label_filter == "pos":
        rows = [r for r in rows if r[1][1] == 1]
    elif label_filter == "neg":
        rows = [r for r in rows if r[1][1] == 0]
    if query:
        rows = [r for r in rows if query in r[1][0].lower()]

    total = len(rows)
    total_pages = max((total + DATASET_PAGE_SIZE - 1) // DATASET_PAGE_SIZE, 1)
    page = min(page, total_pages)
    start = (page - 1) * DATASET_PAGE_SIZE
    page_rows = rows[start:start + DATASET_PAGE_SIZE]

    reviews = [
        {"index": idx, "snippet": text[:150] + ("..." if len(text) > 150 else ""), "label": label}
        for idx, (text, label) in page_rows
    ]

    return render_template(
        "dataset.html",
        reviews=reviews,
        label_filter=label_filter,
        query=request.args.get("q", ""),
        page=page,
        total_pages=total_pages,
        total=total,
        stats=state.stats,
        added=added,
    )


@app.route("/dataset/add", methods=["POST"])
def dataset_add():
    text = request.form.get("review", "").strip()
    label = request.form.get("label")

    if text and label in ("0", "1"):
        state.add_review(text, int(label))
        return redirect(url_for("dataset", added="1"))

    return redirect(url_for("dataset"))


if __name__ == "__main__":
    app.run(debug=True)
