from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from Labeling_data_ingestion.models.sklearn_models.transformer import MiniLmVectorizer
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def train_test(df, test_size, seed=40):
    X = df["title"] + " " + df["theme"]
    y = df["relevant"]
    return train_test_split(X, y, random_state=seed, test_size=test_size)


def train_tfidf(model, X_train, y_train):
    if X_train is None or y_train is None:
        raise ValueError("Training data is None")

    pipe = Pipeline([
        ("vectorizer", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)
    return pipe


def train_transformer(model, X_train, y_train):
    if X_train is None or y_train is None:
        raise ValueError("Training data is None")

    pipe = Pipeline([
        ("vectorizer", MiniLmVectorizer()),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)
    return pipe


def evaluate(pipe, X_test, y_test, threshold):
    proba  = pipe.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    return {
        "accuracy":         accuracy_score(y_test, y_pred),
        "precision":        precision_score(y_test, y_pred),
        "recall":           recall_score(y_test, y_pred),
        "f1":               f1_score(y_test, y_pred),
        "roc_auc":          roc_auc_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report":           classification_report(y_test, y_pred),
    }


def save_model(pipe, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)