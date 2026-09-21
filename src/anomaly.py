"""Flag unusual spending: any purchase over $200."""
from __future__ import annotations

import pandas as pd

from src.parser import load_transactions

DATA_PATH = "data/sample_transactions.csv"
SPEND_LIMIT = 200.0


def flag_amount_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_amount_anomaly"] = False
    df["anomaly_reason"] = ""

    expenses = df["amount"] < 0
    unusual = expenses & (df["amount"].abs() > SPEND_LIMIT)

    df.loc[unusual, "is_amount_anomaly"] = True
    df.loc[unusual, "anomaly_reason"] = f"Over ${SPEND_LIMIT:,.0f} limit"
    return df


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    return flag_amount_anomalies(df)


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    df = load_transactions(DATA_PATH)
    result = detect_anomalies(df)
    unusual = result[result["is_amount_anomaly"]].sort_values("amount")

    print(f"Unusual purchases ({len(unusual)}):")
    if unusual.empty:
        print("(none)")
    else:
        print(
            unusual[
                ["date", "description", "category", "amount", "anomaly_reason"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
