"""Rule-based (keyword matching) baseline transaction categorizer."""
from src.parser import load_transactions

# Keywords are real substrings pulled from data/sample_transactions.csv.
# Order matters: categorize_rule_based returns the FIRST category whose
# keyword list matches, so more specific/unambiguous keywords should come
# before categories that could accidentally overlap (e.g. Dining's
# "UBER *EATS" before Transport's more generic "UBER").
CATEGORY_KEYWORDS = {
    "Rent": [
        "BLUEROCK PROPERTIES",
        "GREYSTAR",
        "AVALON BAY",
        "IRVINE COMPANY",
    ],
    "Subscriptions": [
        "NETFLIX",
        "AMAZON PRIME",
        "ADOBE",
        "APPLE.COM/BILL",
        "HULU",
        "SPOTIFY",
        "NYTIMES",
    ],
    "Dining": [
        "PANERA BREAD",
        "TST*",
        "POKE BOWL",
        "UBER *EATS",
        "CHIPOTLE",
        "DOORDASH",
        "STARBUCKS",
    ],
    "Groceries": [
        "COSTCO",
        "WHOLEFDS",
        "KROGER",
        "TRADER JOE",
        "SAFEWAY",
        "PUBLIX",
        "ALDI",
    ],
    "Transport": [
        "METRO TRANSIT",
        "GAS STATION",
        "LYFT",
        "BART CLIPPER",
        "SHELL OIL",
        "CHEVRON",
        "HERTZ",
        "AMTRAK",
        "UBER *TRIP",
    ],
    "Entertainment": [
        "TICKETMASTER",
        "AMC",
        "REGAL CINEMAS",
        "DAVE & BUSTERS",
        "STEAMGAMES",
        "ESCAPE ROOM",
    ],
    "Utilities": [
        "WASTE MGMT",
        "AT&T",
        "COMCAST",
        "PGE ELECTRIC",
        "VERIZON WIRELESS",
    ],
    "Shopping": [
        "TARGET",
        "TJ MAXX",
        "HOME DEPOT",
        "BEST BUY",
        "NIKE.COM",
        "SEPHORA",
        "REI CO-OP",
        "AMZN MKTP",
        "ETSY.COM",
    ],
    "Income": [
        "DIRECT DEP",
        "PAYPAL TRANSFER",
        "INTUIT QB PAYROLL",
        "VENMO CASHOUT",
        "ACME CORP PAYROLL",
        "STATE OF CA REFUND",
    ],
    "Other": [
        "ATM WITHDRAWAL",
        "OVERDRAFT FEE",
        "CASH APP",
        "WIRE TRANSFER FEE",
        "MISC DEBIT ADJUSTMENT",
        "CHECK #",
        "DMV RENEWAL",
    ],
}


def categorize_rule_based(description: str) -> str:
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in description for keyword in keywords):
            return category
    return "Other"


if __name__ == "__main__":
    df = load_transactions("data/sample_transactions.csv")
    df["predicted_category"] = df["description"].apply(categorize_rule_based)
    correct = df["predicted_category"] == df["category"]

    overall_accuracy = correct.mean()
    print(f"Overall accuracy: {overall_accuracy:.1%} ({correct.sum()}/{len(df)})")

    print("\nAccuracy by category:")
    per_category = (
        df.assign(correct=correct)
        .groupby("category")["correct"]
        .agg(["mean", "sum", "count"])
        .sort_values("mean")
    )
    for category, row in per_category.iterrows():
        print(f"  {category:<15} {row['mean']:6.1%}  ({int(row['sum'])}/{int(row['count'])})")

    wrong = df[~correct]
    print(f"\nMisclassified rows ({len(wrong)}):")
    for _, row in wrong.iterrows():
        print(
            f"  [{row['category']} -> {row['predicted_category']}] {row['description']}"
        )
