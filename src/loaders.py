"""Unified loaders for bank CSVs and statement PDFs."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from src.parser import load_transactions
from src.pdf_parser import load_transactions_from_pdf


def load_transactions_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_transactions_from_pdf(path)
    if suffix in {".csv", ".txt"}:
        return load_transactions(str(path))
    raise ValueError(f"Unsupported file type '{suffix}'. Upload a .csv or .pdf.")


def load_transactions_upload(filename: str, raw_bytes: bytes) -> pd.DataFrame:
    """Load from Streamlit/uploader bytes using the original filename extension."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return load_transactions_from_pdf(raw_bytes)
    if suffix in {".csv", ".txt"}:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        try:
            return load_transactions(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    raise ValueError(f"Unsupported file type '{suffix}'. Upload a .csv or .pdf.")
