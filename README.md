# Movie Review Sentiment — Naive Bayes Demo

A Flask application for exploring Multinomial Naive Bayes text classification on a small, hardcoded movie-review dataset.

## Features

- Enter a movie review and classify it as Positive or Negative.
- Quick-fill Positive/Negative example links.
- See the predicted class probabilities (P(Negative) / P(Positive)).
- Inspect the most indicative words per class (highest log-probability difference).
- View model fit metrics: accuracy, precision, recall, F1-score.
- View raw word frequency counts across the training reviews.

## Dataset

| Property | Value |
| --- | --- |
| Samples | 8 movie reviews |
| Classes | Positive (1), Negative (0) |
| Source | Hardcoded in `utils/data_loader.py` |
| Vectorization | `CountVectorizer` (bag-of-words) |

## Run locally

Requirements: Python 3.10 or newer.

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install and run:

```bash
python -m pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000` in your browser.

## Project structure

```text
NB-demo/
├── app.py
├── requirements.txt
├── models/
│   ├── __init__.py
│   └── nb_model.py
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   └── metrics.py
└── visualization/
    ├── __init__.py
    └── text_visualization.py
```

## Data and reproducibility

`utils/data_loader.py` hardcodes 8 short English movie reviews (4 positive, 4 negative) so the demo trains instantly with no external files. The dataset is intentionally tiny for teaching purposes; `utils/metrics.py` reports fit on this same training set rather than a held-out test split, since 8 samples are too few to split meaningfully.

## Learning goals

1. Understand how Naive Bayes estimates class-conditional word probabilities.
2. See how `CountVectorizer` turns text into a bag-of-words feature matrix.
3. Interpret which words push a prediction toward Positive vs. Negative.
4. Recognize the limits of a very small training set (memorization vs. generalization).

## References

- [scikit-learn Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [scikit-learn CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
