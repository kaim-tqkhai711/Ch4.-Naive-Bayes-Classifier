# Movie Review Sentiment — Naive Bayes Demo

A Flask application for exploring Multinomial Naive Bayes text classification on a small, hardcoded movie-review dataset.

## Features

- Enter a movie review and classify it as Positive or Negative.
- **Explainability (XAI):** Color-coded word/phrase attribution highlighting which tokens pushed the prediction toward Positive or Negative.
- **Model Comparison:** Switch between `MultinomialNB`, `BernoulliNB`, and `ComplementNB`.
- **Feature Extraction:** Switch between `CountVectorizer` and `TF-IDF`, with Unigram or Unigram + Bigram support.
- **Interactive Visualizations:** Dual-color probability bar, confusion matrix heatmap (Chart.js), and word cloud.
- **Dataset Explorer:** Browse, filter, search all 2,000 reviews, add custom reviews, and retrain on the fly.
- Inspect top indicative words per class and test-set performance metrics (Accuracy, Precision, Recall, F1).


## Dataset

| Property | Value |
| --- | --- |
| Samples | 2,000 movie reviews (1,000 positive, 1,000 negative) |
| Classes | Positive (1), Negative (0) |
| Split | 80% Train (1,600 reviews) / 20% Test (400 reviews) |
| Source | NLTK `movie_reviews` corpus (cached to `data/movie_reviews.csv`) |
| Vectorization | `CountVectorizer` (`stop_words='english'`, `min_df=5`) |
| Test Accuracy | ~81 - 82% |

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
├── app_state.py
├── requirements.txt
├── data/
│   └── movie_reviews.csv
├── models/
│   ├── __init__.py
│   └── nb_model.py
├── static/
│   ├── app.js
│   └── style.css
├── templates/
│   ├── dataset.html
│   └── index.html
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   └── metrics.py
└── visualization/
    ├── __init__.py
    └── text_visualization.py
```


## Data and reproducibility
 
`utils/data_loader.py` loads the standard NLTK `movie_reviews` dataset containing 2,000 full-text English reviews (1,000 positive, 1,000 negative). The data is split into 1,600 training reviews and 400 test reviews (`test_size=0.2, random_state=42`). A local cache file `data/movie_reviews.csv` is created on first load to allow instant app startup (<0.1s). `utils/metrics.py` reports performance evaluated strictly on the 400 held-out test samples.

## Learning goals

1. Understand how Naive Bayes estimates class-conditional word probabilities.
2. See how `CountVectorizer` turns text into a bag-of-words feature matrix.
3. Interpret which words push a prediction toward Positive vs. Negative.
4. Recognize the limits of a very small training set (memorization vs. generalization).

## References

- [scikit-learn Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [scikit-learn CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
