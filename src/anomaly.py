"""Statistical anomaly detection for transaction amounts and merchants."""
import pandas as pd

from src.parser import load_transactions

DATA_PATH = "data/sample_transactions.csv"

# Number of IQRs a transaction's amount must sit from its category's median
# before we call it anomalous. Roughly matches the classic Tukey "1.5x IQR"
# boxplot fence once you account for the gap between the median and a
# quartile, while staying a bit more conservative for these small samples.
ANOMALY_THRESHOLD = 2.0


def flag_amount_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["amount_anomaly_score"] = 0.0
    df["is_amount_anomaly"] = False

    # Only expenses get flagged — income swings for reasons (bonus, refund,
    # new job) that aren't "unusual spending", so they're out of scope here.
    expenses = df[df["amount"] < 0]

    for category, group in expenses.groupby("category"):
        magnitudes = group["amount"].abs()

        # Median/IQR instead of mean/std-dev z-scores: with only ~40-50
        # rows per category, even one or two genuine outliers can drag the
        # mean and inflate the std-dev enough to hide themselves (the
        # "masking effect"). The median and IQR are computed from the
        # middle of the distribution, so a handful of extreme values can't
        # skew the very yardstick used to detect them.
        q1 = magnitudes.quantile(0.25)
        q3 = magnitudes.quantile(0.75)
        iqr = q3 - q1
        median = magnitudes.median()

        if iqr == 0:
            # No spread to measure against (e.g. every amount identical) —
            # nothing can be called "unusual" relative to a category with
            # zero variance, so leave this category's scores at 0.
            continue

        score = (magnitudes - median) / iqr
        df.loc[group.index, "amount_anomaly_score"] = score
        df.loc[group.index, "is_amount_anomaly"] = score.abs() > ANOMALY_THRESHOLD

    return df


def flag_new_merchants(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    chronological_order = df.sort_values("date", kind="mergesort").index

    seen_merchants = set()
    is_new = pd.Series(False, index=df.index)
    for idx in chronological_order:
        description = df.loc[idx, "description"]
        if description not in seen_merchants:
            is_new[idx] = True
            seen_merchants.add(description)

    df["is_new_merchant"] = is_new
    return df


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = flag_amount_anomalies(df)
    df = flag_new_merchants(df)
    return df


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    df = load_transactions(DATA_PATH)
    result = detect_anomalies(df)

    amount_anomalies = result[result["is_amount_anomaly"]].sort_values(
        "amount_anomaly_score", ascending=False
    )
    print(f"Amount anomalies ({len(amount_anomalies)}):")
    if amount_anomalies.empty:
        print("(none)")
    else:
        print(
            amount_anomalies[
                ["date", "description", "category", "amount", "amount_anomaly_score"]
            ].to_string(index=False)
        )
    print()

    new_merchants = result[result["is_new_merchant"]]
    print(f"New merchants ({len(new_merchants)}):")
    print(
        new_merchants[["date", "description", "category"]].to_string(index=False)
    )
    print()

    print("Summary:")
    print(f"  {len(amount_anomalies)} amount anomalies out of {len(df)} rows")
    print(f"  {len(new_merchants)} first-time merchants out of {len(df)} rows")


if __name__ == "__main__":
    main()
