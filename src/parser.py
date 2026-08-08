"""Load and normalize raw bank transaction CSVs into a clean DataFrame."""
import pandas as pd

# Maps our canonical column name -> possible header names a bank might use.
# Matching is case-insensitive, so "Date" and "DATE" both hit "date".
COLUMN_ALIASES = {
    "date": ["date", "transaction date", "posted date", "posting date"],
    "description": ["description", "memo", "merchant", "payee", "details"],
    "amount": ["amount", "transaction amount"],
    "debit": ["debit", "debit amount", "withdrawal"],
    "credit": ["credit", "credit amount", "deposit"],
    "category": ["category"],
}


def _find_column(columns_lower_map, aliases):
    """Return the original column name matching one of `aliases`, or None."""
    for alias in aliases:
        if alias in columns_lower_map:
            return columns_lower_map[alias]
    return None


def load_transactions(filepath: str) -> pd.DataFrame:
    raw = pd.read_csv(filepath)

    # Map lowercased header -> original header, so we can look things up
    # case-insensitively but still reference the real column name.
    columns_lower_map = {col.strip().lower(): col for col in raw.columns}

    date_col = _find_column(columns_lower_map, COLUMN_ALIASES["date"])
    description_col = _find_column(columns_lower_map, COLUMN_ALIASES["description"])
    amount_col = _find_column(columns_lower_map, COLUMN_ALIASES["amount"])
    debit_col = _find_column(columns_lower_map, COLUMN_ALIASES["debit"])
    credit_col = _find_column(columns_lower_map, COLUMN_ALIASES["credit"])
    category_col = _find_column(columns_lower_map, COLUMN_ALIASES["category"])

    missing = []
    if date_col is None:
        missing.append("date")
    if description_col is None:
        missing.append("description")
    if amount_col is None and (debit_col is None or credit_col is None):
        missing.append("amount (or debit/credit pair)")
    if missing:
        raise ValueError(
            f"Could not find required column(s): {', '.join(missing)}. "
            f"Available columns: {list(raw.columns)}"
        )

    out = pd.DataFrame()
    #it converts text into a format pandas understands as a real date

    out["date"] = pd.to_datetime(raw[date_col], format="mixed", dayfirst=False)

    out["description"] = (
        raw[description_col].astype(str).str.strip().str.upper()
    )

    if amount_col is not None:
        out["amount"] = raw[amount_col].astype(float)
    else:
        debit = raw[debit_col].fillna(0).astype(float)
        credit = raw[credit_col].fillna(0).astype(float)
        out["amount"] = credit - debit.abs()

    if category_col is not None:
        out["category"] = raw[category_col].astype(str).str.strip()

    out = out.sort_values("date").reset_index(drop=True)

    return out


if __name__ == "__main__":
    df = load_transactions("data/sample_transactions.csv")
    print(df.head())
    print()
    print(df.dtypes)
