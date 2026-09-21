"""Normalize statement descriptions so the model keys off the merchant.

City, state, store numbers, and phone numbers are common on bank PDFs and
should not become category features (otherwise SEATTLE starts meaning
Subscriptions because a few training rows happened to include it).
"""
from __future__ import annotations

import re

STATES = (
    "AL|AK|AZ|AR|CA|CO|CT|DC|DE|FL|GA|HI|IA|ID|IL|IN|KS|KY|LA|"
    "MA|MD|ME|MI|MN|MO|MS|MT|NC|ND|NE|NH|NJ|NM|NV|NY|OH|OK|OR|"
    "PA|RI|SC|SD|TN|TX|UT|VA|VT|WA|WI|WV|WY"
)

_TRAILING_LOCATION = re.compile(
    rf"(?:\s+[A-Z][A-Z0-9'&.-]{{1,24}}){{1,4}}\s+(?:{STATES})\b"
    r"(?:\s+\d{5}(?:-\d{4})?)?\s*$",
    re.IGNORECASE,
)
_PHONE = re.compile(r"\b\d{3}[\s.-]?\d{3}[\s.-]?\d{4}\b|\b\d{10,}\b")
_STORE_NUM = re.compile(r"#\s*\d+|\b\d{4,}\b")
_PUNCT = re.compile(r"[^A-Z0-9* ]+")
_SPACES = re.compile(r"\s+")
_CREDIT_SUFFIX = re.compile(r"\bCREDIT\s*$", re.IGNORECASE)


def normalize_description(text: str) -> str:
    """Strip location / store-number noise and keep merchant tokens."""
    value = str(text).upper().strip()
    value = _CREDIT_SUFFIX.sub("", value)
    for _ in range(2):
        stripped = _TRAILING_LOCATION.sub("", value).strip()
        if stripped == value:
            break
        value = stripped
    value = _PHONE.sub(" ", value)
    value = _STORE_NUM.sub(" ", value)
    value = _PUNCT.sub(" ", value)
    return _SPACES.sub(" ", value).strip()
