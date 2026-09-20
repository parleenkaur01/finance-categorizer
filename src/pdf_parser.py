"""Extract transactions from credit-card statement PDFs.

Designed primarily for Robinhood-style statements (Tran Date / Post Date /
Reference / Description / Amount), but also accepts lines without a reference:
MM/DD  MM/DD  DESCRIPTION  AMOUNT.
"""
from __future__ import annotations

import io
import re
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd
import pdfplumber

PathLike = Union[str, Path, bytes, BinaryIO]

# MM/DD  MM/DD  REF  DESCRIPTION  AMOUNT  (amount may end with '-' for credits)
_TXN_WITH_REF = re.compile(
    r"^(?P<tran>\d{1,2}/\d{1,2})\s+"
    r"(?P<post>\d{1,2}/\d{1,2})\s+"
    r"(?P<ref>[A-Z0-9]{10,})\s+"
    r"(?P<desc>.+?)\s+"
    r"(?P<amount>\d{1,3}(?:,\d{3})*\.\d{2}-?)$",
    re.IGNORECASE,
)

# MM/DD  MM/DD  DESCRIPTION  AMOUNT  (interest / fee lines without a ref)
_TXN_NO_REF = re.compile(
    r"^(?P<tran>\d{1,2}/\d{1,2})\s+"
    r"(?P<post>\d{1,2}/\d{1,2})\s+"
    r"(?P<desc>.+?)\s+"
    r"(?P<amount>\d{1,3}(?:,\d{3})*\.\d{2}-?)$",
    re.IGNORECASE,
)

_YEAR_PATTERNS = [
    re.compile(
        r"Statement\s*Closing\s*Date\s+[A-Za-z]+\s*\d{1,2},?\s*(?P<year>20\d{2})",
        re.I,
    ),
    re.compile(
        r"Payment\s*Due\s*Date\s+[A-Za-z]+\s*\d{1,2},?\s*(?P<year>20\d{2})",
        re.I,
    ),
    re.compile(r"\b(?P<year>20\d{2})\b"),
]

_SKIP_DESC = re.compile(
    r"^(TOTAL|ANNUAL|BALANCE SUBJECT|TYPE OF|TRANSACTIONS|PAGE\s+\d|"
    r"INTEREST CHARGE CALCULATION)",
    re.I,
)


def _open_pdf(source: PathLike):
    if isinstance(source, (bytes, bytearray)):
        return pdfplumber.open(io.BytesIO(source))
    if hasattr(source, "read"):
        data = source.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        return pdfplumber.open(io.BytesIO(data))
    return pdfplumber.open(str(source))


def _extract_text(source: PathLike) -> str:
    chunks = []
    with _open_pdf(source) as pdf:
        for page in pdf.pages:
            # Tight x_tolerance keeps merchant spaces (TRADER JOE S, not TRADERJOES).
            text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
            chunks.append(text)
    return "\n".join(chunks)


def _infer_year(text: str) -> int:
    for pattern in _YEAR_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group("year"))
    return datetime.now().year


def _parse_date(mmdd: str, year: int) -> datetime:
    month, day = mmdd.split("/")
    return datetime(year, int(month), int(day))


def _amount_from_match(raw_amt: str) -> float | None:
    """Purchases → negative; trailing '-' credits/payments → positive."""
    raw_amt = raw_amt.replace(",", "").strip()
    if raw_amt.endswith("-"):
        return float(raw_amt[:-1])
    value = float(raw_amt)
    if value == 0:
        return None
    return -value


def parse_statement_text(text: str, year: int | None = None) -> pd.DataFrame:
    if year is None:
        year = _infer_year(text)

    rows = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue

        match = _TXN_WITH_REF.match(line) or _TXN_NO_REF.match(line)
        if not match:
            continue

        desc = match.group("desc").strip()
        if _SKIP_DESC.search(desc):
            continue

        amount = _amount_from_match(match.group("amount"))
        if amount is None:
            continue

        rows.append(
            {
                "date": _parse_date(match.group("tran"), year),
                "description": re.sub(r"\s+", " ", desc).strip().upper(),
                "amount": amount,
            }
        )

    if not rows:
        raise ValueError(
            "No transactions found in PDF. "
            "Supported layouts look like: MM/DD MM/DD REF DESCRIPTION AMOUNT"
        )

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_transactions_from_pdf(source: PathLike) -> pd.DataFrame:
    """Load and normalize transactions from a statement PDF path or bytes."""
    text = _extract_text(source)
    return parse_statement_text(text)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python -m src.pdf_parser <statement.pdf>")
        raise SystemExit(1)
    df = load_transactions_from_pdf(path)
    print(df.to_string(index=False))
    print(f"\n{len(df)} transactions")
