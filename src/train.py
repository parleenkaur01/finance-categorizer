"""Train a TF-IDF + Logistic Regression classifier for transaction categories.

Why evaluate BEFORE refitting on all the data:
The whole point of a train/test split is to get an honest estimate of how the
model performs on transactions it has never seen. If we evaluated on data the
model was trained on, it could just be memorizing, and accuracy would look
artificially high (this is exactly what happened with the rule-based baseline).
So we fit on the training split, score on the held-out test split, and report
THAT number as the trustworthy one.

Only after that honest evaluation is done do we throw away the split and
refit the pipeline on every row we have. This second fit isn't evaluated —
it's just meant to squeeze the most signal possible into the model we
actually ship, since more training data generally makes a model better and
we no longer need to hold anything back once we already know how well the
approach performs.
"""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from src.parser import load_transactions

DATA_PATH = "data/sample_transactions.csv"
MODEL_PATH = "src/model.pkl"


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    df = load_transactions(DATA_PATH)
    X = df["description"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = (y_pred == y_test).mean()
    print(f"Test accuracy: {accuracy:.4f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    labels = sorted(y.unique())
    print("Confusion matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)
    print()

    misclassified = X_test[y_test != y_pred].to_frame()
    misclassified["true_category"] = y_test[y_test != y_pred]
    misclassified["predicted_category"] = pd.Series(y_pred, index=y_test.index)[y_test != y_pred]

    print("Misclassified rows:")
    if misclassified.empty:
        print("(none)")
    else:
        print(misclassified.to_string())

    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)
    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"\nSaved final pipeline (trained on all data) to {MODEL_PATH}")


if __name__ == "__main__":
    main()
