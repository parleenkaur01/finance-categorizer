"""Load the trained categorizer and predict categories.

Hybrid strategy:
1. Keyword rules win when they match (fixes real merchants the model never saw).
2. Otherwise fall back to the TF-IDF + Logistic Regression model.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.categorizer import match_keyword_category

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"

_model = None


def load_model(path: str | Path | None = None):
    global _model
    model_path = Path(path) if path else MODEL_PATH
    if _model is None or path is not None:
        _model = joblib.load(model_path)
    return _model


def predict_categories(descriptions: pd.Series, model=None) -> pd.Series:
    pipe = model or load_model()
    texts = descriptions.astype(str)
    ml_preds = pipe.predict(texts)

    final = []
    sources = []
    for text, ml_label in zip(texts, ml_preds):
        rule_label = match_keyword_category(text)
        if rule_label is not None:
            final.append(rule_label)
            sources.append("rule")
        else:
            final.append(ml_label)
            sources.append("model")

    result = pd.Series(final, index=descriptions.index, name="category")
    result.attrs["sources"] = sources
    return result


def categorize_transactions(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Return a copy with `category` (and `category_source`) columns."""
    out = df.copy()
    texts = out["description"].astype(str)
    pipe = model or load_model()
    ml_preds = pipe.predict(texts)

    categories = []
    sources = []
    for text, ml_label in zip(texts, ml_preds):
        rule_label = match_keyword_category(text)
        if rule_label is not None:
            categories.append(rule_label)
            sources.append("rule")
        else:
            categories.append(ml_label)
            sources.append("model")

    out["category"] = categories
    out["category_source"] = sources
    return out
