"""Train a TF-IDF + Logistic Regression classifier for transaction categories.

Training data is labeled real credit-statement rows (see
data/build_real_training.py). We still evaluate on a held-out split first,
then refit on every real row before saving the shipped model.
"""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from src.features import normalize_description
from src.parser import load_transactions
from src.predict import predict_categories

DATA_PATH = "data/real_transactions.csv"
MODEL_PATH = "src/model.pkl"


def build_pipeline() -> Pipeline:
    # Word n-grams catch merchant tokens; character n-grams still work when
    # a PDF glues words together. Location noise is stripped first so the
    # model cannot memorize CITY/STATE as a category.
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                preprocessor=normalize_description,
                                ngram_range=(1, 2),
                                min_df=1,
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                preprocessor=normalize_description,
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=1,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=1.0,
                ),
            ),
        ]
    )


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    df = load_transactions(DATA_PATH)
    X = df["description"]
    y = df["category"]
    print(f"Training on {len(df)} real transactions from {DATA_PATH}")
    print(y.value_counts().to_string())
    print()

    counts = y.value_counts()
    can_stratify = counts.min() >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y if can_stratify else None,
        random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred_ml = pd.Series(pipeline.predict(X_test), index=X_test.index)
    # Same path as the dashboard: keyword rules win, then the model.
    y_pred = predict_categories(X_test, model=pipeline)

    ml_accuracy = (y_pred_ml == y_test).mean()
    accuracy = (y_pred == y_test).mean()
    print(f"ML-only test accuracy: {ml_accuracy:.4f}")
    print(f"Hybrid test accuracy (rules + model): {accuracy:.4f}\n")
    print("Classification report (hybrid):")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = sorted(y.unique())
    print("Confusion matrix (rows=true, cols=predicted, hybrid):")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))
    print()

    misclassified = X_test[y_test != y_pred].to_frame()
    misclassified["true_category"] = y_test[y_test != y_pred]
    misclassified["predicted_category"] = y_pred[y_test != y_pred]
    print("Misclassified rows (hybrid):")
    if misclassified.empty:
        print("(none)")
    else:
        print(misclassified.to_string())

    ml_misses = X_test[y_test != y_pred_ml].to_frame()
    ml_misses["true_category"] = y_test[y_test != y_pred_ml]
    ml_misses["ml_predicted"] = y_pred_ml[y_test != y_pred_ml]
    print("\nML-only misses (corrected by keyword rules when listed as hybrid none):")
    if ml_misses.empty:
        print("(none)")
    else:
        print(ml_misses.to_string())

    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)
    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"\nSaved final pipeline (trained on all real data) to {MODEL_PATH}")


if __name__ == "__main__":
    main()
