"""Flag first-time merchants and unusually high category spend (median / IQR)."""
from __future__ import annotations

import pandas as pd

from src.parser import load_transactions

DATA_PATH = "data/sample_transactions.csv"

# Number of IQRs above the category median before a purchase is "unusually high."
ANOMALY_THRESHOLD = 2.0

HIGH_SPEND_REASON = "Unusually high for this category"
NEW_MERCHANT_REASON = "First time seeing this merchant"


def flag_amount_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["amount_anomaly_score"] = 0.0
    df["is_amount_anomaly"] = False

    expenses = df[df["amount"] < 0]
    if expenses.empty or "category" not in df.columns:
        return df

    for _category, group in expenses.groupby("category"):
        magnitudes = group["amount"].abs()
        q1 = magnitudes.quantile(0.25)
        q3 = magnitudes.quantile(0.75)
        iqr = q3 - q1
        median = magnitudes.median()

        if iqr == 0:
            continue

        score = (magnitudes - median) / iqr
        df.loc[group.index, "amount_anomaly_score"] = score
        # High spend only — unusually cheap purchases are not the product signal.
        unusual = score > ANOMALY_THRESHOLD
        df.loc[group.index[unusual], "is_amount_anomaly"] = True

    return df


def flag_new_merchants(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_new_merchant"] = False

    chronological_order = df.sort_values("date", kind="mergesort").index
    seen_merchants: set[str] = set()
    for idx in chronological_order:
        if df.loc[idx, "amount"] >= 0:
            continue
        description = str(df.loc[idx, "description"])
        if description not in seen_merchants:
            df.loc[idx, "is_new_merchant"] = True
            seen_merchants.add(description)

    return df


def _reason_text(is_new: bool, is_high: bool) -> str:
    parts = []
    if is_new:
        parts.append(NEW_MERCHANT_REASON)
    if is_high:
        parts.append(HIGH_SPEND_REASON)
    return "; ".join(parts)


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = flag_amount_anomalies(df)
    df = flag_new_merchants(df)
    df["anomaly_reason"] = [
        _reason_text(is_new, is_high)
        for is_new, is_high in zip(df["is_new_merchant"], df["is_amount_anomaly"])
    ]
    df["is_anomaly"] = df["is_new_merchant"] | df["is_amount_anomaly"]
    return df


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    df = load_transactions(DATA_PATH)
    result = detect_anomalies(df)
    flagged = result[result["is_anomaly"]].sort_values("date")

    print(f"Flagged purchases ({len(flagged)}):")
    if flagged.empty:
        print("(none)")
    else:
        print(
            flagged[
                ["date", "description", "category", "amount", "anomaly_reason"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
