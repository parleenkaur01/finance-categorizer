"""Build a labeled training CSV from real statement PDFs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.categorizer import match_keyword_category
from src.pdf_parser import load_transactions_from_pdf

STATEMENTS_DIR = Path(__file__).parent / "statements"
OUTPUT_PATH = Path(__file__).parent / "real_transactions.csv"


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
    unlabeled = df[df["category"].isna()]
    labeled = df[df["category"].notna()]
    if not unlabeled.empty:
        print("\nSkipping unlabeled merchants (no keyword rule yet):")
        for desc in sorted(unlabeled["description"].unique()):
            print(f"  {desc}")
        print(f"({len(unlabeled)} of {len(df)} rows skipped)")

    if labeled.empty:
        raise SystemExit("No labeled rows to write. Add a keyword rule and re-run.")

    out = labeled[["date", "description", "amount", "category"]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"\nWrote {len(out)} labeled rows to {OUTPUT_PATH}")
    print(out["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
