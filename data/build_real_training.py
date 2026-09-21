"""Build a labeled training CSV from real statement PDFs.

Labels come from KEYWORD_RULES first, then data/merchant_labels.csv.
Rows that still have no category are skipped."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.categorizer import match_keyword_category
from src.pdf_parser import load_transactions_from_pdf

STATEMENTS_DIR = Path(__file__).parent / "statements"
LABELS_PATH = Path(__file__).parent / "merchant_labels.csv"
OUTPUT_PATH = Path(__file__).parent / "real_transactions.csv"


def _norm(text: str) -> str:
    return str(text).strip().casefold()


def load_merchant_labels(path: Path = LABELS_PATH) -> dict[str, str]:
    """Exact description → category from merchant_labels.csv (case-insensitive)."""
    if not path.exists():
        return {}
    labels = pd.read_csv(path)
    if labels.empty or "description" not in labels.columns or "category" not in labels.columns:
        return {}
    labels = labels.dropna(subset=["description", "category"])
    return {
        _norm(row["description"]): str(row["category"]).strip()
        for _, row in labels.iterrows()
        if str(row["category"]).strip()
    }


def main() -> None:
    pdfs = sorted(STATEMENTS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {STATEMENTS_DIR}")

    frames = []
    for pdf in pdfs:
        frame = load_transactions_from_pdf(pdf)
        frame["source_file"] = pdf.name
        frames.append(frame)
        print(f"Loaded {len(frame):3d} rows from {pdf.name}")

    df = pd.concat(frames, ignore_index=True)
    df["category"] = df["description"].map(match_keyword_category)

    merchant_labels = load_merchant_labels()
    missing = df["category"].isna()
    if merchant_labels and missing.any():
        from_file = df.loc[missing, "description"].map(lambda d: merchant_labels.get(_norm(d)))
        df.loc[missing, "category"] = from_file
        filled = int(from_file.notna().sum())
        if filled:
            print(f"Labeled {filled} rows from {LABELS_PATH.name}")

    unlabeled = df[df["category"].isna()]
    labeled = df[df["category"].notna()]
    if not unlabeled.empty:
        print("\nSkipping unlabeled merchants (add a keyword or a row in merchant_labels.csv):")
        for desc in sorted(unlabeled["description"].unique()):
            print(f"  {desc}")
        print(f"({len(unlabeled)} of {len(df)} rows skipped)")

    if labeled.empty:
        raise SystemExit(
            "No labeled rows to write. Add a keyword rule or merchant_labels.csv row and re-run."
        )

    out = labeled[["date", "description", "amount", "category"]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"\nWrote {len(out)} labeled rows to {OUTPUT_PATH}")
    print(out["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
