"""Keyword overrides for merchants the ML model has not seen.

Matching is space-insensitive: PDF extractors often glue words together
(ALASKAAIR, AMAZONRETA), so we compare a compact form of both the
description and each keyword.

More specific rules are listed first (AMAZON PRIME before AMAZON).
"""
from __future__ import annotations

import re

from src.parser import load_transactions

# (keyword, category) — first match wins. Keep specific merchants above
# generic words like AMAZON or CAFE.
KEYWORD_RULES: list[tuple[str, str]] = [
    # Dining
    ("BROKEN YOLK", "Dining"),
    ("STARBUCKS", "Dining"),
    ("PANERA BREAD", "Dining"),
    ("CHIPOTLE", "Dining"),
    # DoorDash: merchant-specific before generic food delivery
    ("DOORDASH TARGET", "Shopping"),
    ("DOORDASH BEVMO", "Shopping"),
    ("DOORDASHDASHPASS", "Subscriptions"),
    ("DASHPASS", "Subscriptions"),
    ("DOORDASH", "Dining"),
    ("GRUBHUB", "Dining"),
    ("UBER EATS", "Dining"),
    ("UBER *EATS", "Dining"),
    ("POKE BOWL", "Dining"),
    ("NEUMOS", "Entertainment"),
    ("TST*", "Dining"),
    ("MCDONALD", "Dining"),
    ("IN-N-OUT", "Dining"),
    ("FIVE GUYS", "Dining"),
    ("SUBWAY", "Dining"),
    ("OLIVE GARDEN", "Dining"),
    ("CAFE", "Dining"),
    ("COFFEE", "Dining"),
    ("NESPRESSO", "Dining"),
    ("RESTAURANT", "Dining"),
    ("DINER", "Dining"),
    ("BOMBAY", "Dining"),
    ("CHAAT HOUSE", "Dining"),
    ("CHAAT", "Dining"),
    ("QDOBA", "Dining"),
    ("CHILI'S", "Dining"),
    ("CHILIS", "Dining"),
    ("SEATTLE BEER", "Dining"),
    ("TACOS EL GORDO", "Dining"),
    ("EL GORDO", "Dining"),
    ("TACOS", "Dining"),
    ("TACO", "Dining"),
    ("PHO SHIZZLE", "Dining"),
    ("PHO SIZZLE", "Dining"),
    ("PHO ", "Dining"),
    ("CAVA", "Dining"),
    ("DOMINO", "Dining"),
    ("BURGER KING", "Dining"),
    ("WINGS N THINGS", "Dining"),
    ("NORTH ITALIA", "Dining"),
    ("TANDOORI", "Dining"),
    ("POTBELLY", "Dining"),
    ("EINSTEIN BROS", "Dining"),
    ("GYRO", "Dining"),
    ("YUMBIT", "Dining"),
    ("IMPECKABLE", "Dining"),
    ("DJUNG", "Dining"),
    ("F&B", "Dining"),
    # Groceries
    ("TRADER JOE", "Groceries"),
    ("WHOLE FOODS", "Groceries"),
    ("WHOLEFDS", "Groceries"),
    ("SAFEWAY", "Groceries"),
    ("KROGER", "Groceries"),
    ("COSTCO", "Groceries"),
    ("PUBLIX", "Groceries"),
    ("ALDI", "Groceries"),
    ("VONS", "Groceries"),
    ("RALPHS", "Groceries"),
    ("SPROUTS", "Groceries"),
    ("H&M", "Shopping"),
    ("PACSUN", "Shopping"),
    ("H MART", "Groceries"),
    ("QFC", "Groceries"),
    ("WALMART", "Groceries"),
    ("WAL-MART", "Groceries"),
    ("GROCERY", "Groceries"),
    ("INDIAN SPICES", "Groceries"),
    ("AZTEC RECREATION", "Entertainment"),
    ("AZTEC MARKET", "Groceries"),
    ("COAST MARKET", "Groceries"),
    ("NEWS EXPRESS", "Groceries"),
    ("INSTACART", "Groceries"),
    # Transport / travel
    ("ALASKA AIR", "Transport"),
    ("ALASKA AIRLINES", "Transport"),
    ("DELTA AIR", "Transport"),
    ("DELTA.COM", "Transport"),
    ("SOUTHWEST", "Transport"),
    ("UNITED AIR", "Transport"),
    ("UNITED.COM", "Transport"),
    ("UBER TRIP", "Transport"),
    ("UBER *TRIP", "Transport"),
    ("LYFT", "Transport"),
    ("LIME", "Transport"),
    ("AMTRAK", "Transport"),
    ("HERTZ", "Transport"),
    ("ARCO", "Transport"),
    ("SHELL OIL", "Transport"),
    ("CHEVRON", "Transport"),
    ("EXXON", "Transport"),
    ("MTS PRONTO", "Transport"),
    ("SEATTLE MONORAIL", "Transport"),
    ("METRO TRANSIT", "Transport"),
    ("BART CLIPPER", "Transport"),
    ("GAS STATION", "Transport"),
    ("ZIPCAR", "Transport"),
    ("PARKING GARAGE", "Transport"),
    ("GARAGE SAN DIEGO", "Transport"),
    ("PARKING", "Transport"),
    # Shopping (Amazon marketplace before generic Amazon)
    ("AMAZON PRIME", "Subscriptions"),
    ("AMZN MKTP", "Shopping"),
    ("AMAZON RETA", "Shopping"),
    ("WWW.AMAZON", "Shopping"),
    ("AMAZON.COM", "Shopping"),
    ("AMAZON", "Shopping"),
    ("GARAGE", "Shopping"),
    ("CVS", "Shopping"),
    ("PHARMACY", "Shopping"),
    ("TARGET", "Shopping"),
    ("BEST BUY", "Shopping"),
    ("HOME DEPOT", "Shopping"),
    ("TJ MAXX", "Shopping"),
    ("NIKE.COM", "Shopping"),
    ("SEPHORA", "Shopping"),
    ("ULTA", "Shopping"),
    ("ETSY.COM", "Shopping"),
    ("REI CO-OP", "Shopping"),
    ("SHOPCIDER", "Shopping"),
    ("CIDER.COM", "Shopping"),
    ("UNIQLO", "Shopping"),
    ("ZARA", "Shopping"),
    ("PACSUN", "Shopping"),
    ("PAC SUN", "Shopping"),
    ("H&M", "Shopping"),
    ("H AND M", "Shopping"),
    # Subscriptions
    ("NETFLIX", "Subscriptions"),
    ("ADOBE", "Subscriptions"),
    ("APPLE.COM/BILL", "Subscriptions"),
    ("HULU", "Subscriptions"),
    ("SPOTIFY", "Subscriptions"),
    ("NYTIMES", "Subscriptions"),
    ("PEACOCK", "Subscriptions"),
    ("DISNEY PLUS", "Subscriptions"),
    ("YOUTUBE PREMIUM", "Subscriptions"),
    ("CURSOR.COM", "Subscriptions"),
    ("CURSOR", "Subscriptions"),
    ("ANTHROPIC", "Subscriptions"),
    ("CLAUDE SUB", "Subscriptions"),
    ("PERPLEXIT", "Subscriptions"),
    ("RAILWAY", "Subscriptions"),
    ("AUDIBLE", "Subscriptions"),
    ("DROPBOX", "Subscriptions"),
    ("PATREON", "Subscriptions"),
    ("PLANET FITNESS", "Subscriptions"),
    # Rent
    ("BLUEROCK PROPERTIES", "Rent"),
    ("GREYSTAR", "Rent"),
    ("AVALON BAY", "Rent"),
    ("IRVINE COMPANY", "Rent"),
    ("EQUITY RESIDENTIAL", "Rent"),
    ("RENT PAYMENT", "Rent"),
    ("LANDLORD", "Rent"),
    # Entertainment
    ("TICKETMASTER", "Entertainment"),
    ("AMC", "Entertainment"),
    ("REGAL CINEMAS", "Entertainment"),
    ("DAVE & BUSTERS", "Entertainment"),
    ("STEAMGAMES", "Entertainment"),
    ("ESCAPE ROOM", "Entertainment"),
    ("TOP GOLF", "Entertainment"),
    ("SIX FLAGS", "Entertainment"),
    ("WATERFRONT", "Entertainment"),
    ("BELMONT PARK", "Entertainment"),
    ("AZTEC RECREATION", "Entertainment"),
    # Utilities — keep AT&T as a literal (compact "ATT" matches SEATTLE).
    ("WASTE MGMT", "Utilities"),
    ("AT&T", "Utilities"),
    ("COMCAST", "Utilities"),
    ("PGE ELECTRIC", "Utilities"),
    ("VERIZON WIRELESS", "Utilities"),
    ("T-MOBILE", "Utilities"),
    ("SPECTRUM", "Utilities"),
    ("XCEL ENERGY", "Utilities"),
    # Income / payments
    ("DIRECT DEP", "Income"),
    ("PAYPAL TRANSFER", "Income"),
    ("INTUIT QB PAYROLL", "Income"),
    ("VENMO CASHOUT", "Income"),
    ("ACME CORP PAYROLL", "Income"),
    ("STATE OF CA REFUND", "Income"),
    ("PAYMENT - THANK YOU", "Income"),
    ("PAYMENT THANK YOU", "Income"),
    ("AUTOPAY PAYMENT", "Income"),
    # Other
    ("ATM WITHDRAWAL", "Other"),
    ("OVERDRAFT FEE", "Other"),
    ("CASH APP", "Other"),
    ("WIRE TRANSFER FEE", "Other"),
    ("MISC DEBIT ADJUSTMENT", "Other"),
    ("CHECK #", "Other"),
    ("DMV RENEWAL", "Other"),
    ("LATE FEE", "Other"),
    ("INTEREST CHARGE", "Other"),
    ("FOREIGN TXN FEE", "Other"),
    ("BANK MAINTENANCE", "Other"),
    ("PDF.NET", "Other"),
    ("UPS STORE", "Other"),
    ("ROBINHOOD", "Other"),
    ("KALSHI", "Other"),
    ("HOTEL", "Other"),
]


def _compact(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(text).upper())


def match_keyword_category(description: str) -> str | None:
    """Return a category if any keyword matches, else None."""
    text = str(description).upper()
    compact_text = _compact(text)
    for keyword, category in KEYWORD_RULES:
        needle = keyword.upper()
        compact_kw = _compact(keyword)
        if needle in text:
            return category
        # Compact match is for glued PDF tokens (ALASKAAIR). Short
        # fragments like ATT (from AT&T) otherwise hit SEATTLE.
        if len(compact_kw) >= 5 and compact_kw in compact_text:
            return category
    return None


def categorize_rule_based(description: str) -> str:
    return match_keyword_category(description) or "Other"


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
        print(
            f"  {category:<15} {row['mean']:6.1%}  "
            f"({int(row['sum'])}/{int(row['count'])})"
        )

    wrong = df[~correct]
    print(f"\nMisclassified rows ({len(wrong)}):")
    for _, row in wrong.iterrows():
        print(
            f"  [{row['category']} -> {row['predicted_category']}] {row['description']}"
        )
