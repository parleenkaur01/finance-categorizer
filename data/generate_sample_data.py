# messy data-handler

"""Generate a synthetic bank-transaction dataset for the finance categorizer."""
import csv
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

OUTPUT_PATH = Path(__file__).parent / "sample_transactions.csv"

START_DATE = date(2026, 5, 1)
END_DATE = date(2026, 7, 31)

# (description template pool, amount range, sign) per category.
# Templates use realistic bank-statement noise: store numbers, city/state,
# POS/ACH prefixes, card processor tags, inconsistent casing.
CATEGORY_TEMPLATES = {
    "Groceries": {
        "amount_range": (-120, -15),
        "descriptions": [
            "TRADER JOE'S #421 SAN DIEGO CA",
            "POS DEBIT WHOLEFDS MKT 10345 AUSTIN TX",
            "SAFEWAY STORE 00456 SEATTLE WA",
            "KROGER #0287 COLUMBUS OH",
            "COSTCO WHSE #0512 SAN JOSE CA",
            "ALDI 71029 CHICAGO IL",
            "PUBLIX #1187 ORLANDO FL",
            "VONS #2214 LOS ANGELES CA",
        ],
    },
    "Dining": {
        "amount_range": (-45, -8),
        "descriptions": [
            "SQ *COFFEE SHOP SAN DIEGO CA",
            "CHIPOTLE 2841 DENVER CO",
            "MCDONALD'S F32910 HOUSTON TX",
            "TST* THE CORNER BISTRO NYC NY",
            "STARBUCKS STORE 08812 PORTLAND OR",
            "DOORDASH*WINGSTOP SAN FRANCISCO CA",
            "UBER *EATS HELP.UBER.COM",
            "PANERA BREAD #601934 DALLAS TX",
        ],
    },
    "Rent": {
        "amount_range": (-2400, -1750),
        "descriptions": [
            "ACH DEBIT GREYSTAR MGMT RENT",
            "ONLINE PMT BLUEROCK PROPERTIES LLC",
            "RENT PAYMENT AVALON BAY COMMUNITIES",
            "ACH DEBIT IRVINE COMPANY APTS",
        ],
    },
    "Subscriptions": {
        "amount_range": (-25, -6),
        "descriptions": [
            "NETFLIX.COM LOS GATOS CA",
            "SPOTIFY USA NEW YORK NY",
            "APPLE.COM/BILL 866-712-7753 CA",
            "AMAZON PRIME*MI79K2 SEATTLE WA",
            "HULU 877-830-4858 CA",
            "NYTIMES*SUBSCRIPTION 800-698-4637",
            "ADOBE  *CREATIVE CLD SAN JOSE CA",
        ],
    },
    "Transport": {
        "amount_range": (-65, -6),
        "descriptions": [
            "UBER *TRIP HELP.UBER.COM",
            "LYFT *RIDE MON 8PM SAN FRANCISCO CA",
            "SHELL OIL 57443921 PHOENIX AZ",
            "CHEVRON 0091827 LOS ANGELES CA",
            "BART CLIPPER RELOAD SAN FRANCISCO CA",
            "METRO TRANSIT FARE MINNEAPOLIS MN",
            "76 - GAS STATION #3221",
        ],
    },
    "Entertainment": {
        "amount_range": (-90, -10),
        "descriptions": [
            "AMC 24 THEATRES #1187 BURBANK CA",
            "TICKETMASTER 800-653-8000 CA",
            "STEAMGAMES.COM 425-889-9642 WA",
            "REGAL CINEMAS 0421 NEW YORK NY",
            "DAVE & BUSTERS #0087 DALLAS TX",
            "TOP GOLF #0412 AUSTIN TX",
        ],
    },
    "Utilities": {
        "amount_range": (-180, -35),
        "descriptions": [
            "ACH DEBIT PGE ELECTRIC BILLPAY",
            "COMCAST CABLE COMM 800-934-6489",
            "AT&T *PAYMENT 800-288-2020 TX",
            "ACH DEBIT CITY WATER UTILITY",
            "VERIZON WIRELESS PAYMENTS",
            "WASTE MGMT #4471 AUTOPAY",
        ],
    },
    "Shopping": {
        "amount_range": (-150, -12),
        "descriptions": [
            "AMZN MKTP US*2K4RT9 AMZN.COM/BILL",
            "TARGET #1145 SAN DIEGO CA",
            "WALMART SUPERCENTER #2291",
            "BEST BUY 00003221 SEATTLE WA",
            "NIKE.COM 800-806-6453 OR",
            "HOME DEPOT #4409 PHOENIX AZ",
            "TJ MAXX #0872 DENVER CO",
            "ETSY.COM - MERCH BROOKLYN NY",
        ],
    },
    "Income": {
        "amount_range": (900, 3200),
        "descriptions": [
            "ACH CREDIT ACME CORP PAYROLL",
            "DIRECT DEP EMPLOYER DISBURSEMENT",
            "REC'D PAYPAL TRANSFER",
            "ACH CREDIT INTUIT QB PAYROLL SVC",
            "VENMO CASHOUT DEPOSIT",
            "ACH CREDIT STATE OF CA REFUND",
        ],
    },
    "Other": {
        "amount_range": (-200, -10),
        "descriptions": [
            "CHECK #1042",
            "ATM WITHDRAWAL #88213 MAIN ST",
            "ZELLE PMT TO J SMITH",
            "OVERDRAFT FEE",
            "WIRE TRANSFER FEE",
            "MISC DEBIT ADJUSTMENT",
            "CASH APP*TRANSFER",
        ],
    },
}

# One-off / rare merchants that appear exactly once each, sprinkled into
# categories to simulate "never seen before" merchants.
RARE_MERCHANTS = [
    ("Shopping", "REI CO-OP #0033 BOULDER CO", -134.20),
    ("Dining", "SQ *POKE BOWL TRUCK SD", -14.75),
    ("Entertainment", "ESCAPE ROOM SD DOWNTOWN", -42.00),
    ("Other", "DMV RENEWAL FEE SACRAMENTO CA", -58.00),
    ("Transport", "AMTRAK .COM 800-872-7245", -89.50),
    ("Shopping", "SEPHORA #1029 SAN DIEGO CA", -47.30),
]

# Intentional outliers: (category, description, amount) — statistically
# unusual within their category, for testing anomaly detection later.
OUTLIERS = [
    ("Dining", "TST* CHEF'S TASTING ROOM LA JOLLA CA", -280.00),
    ("Groceries", "WHOLEFDS MKT 10345 -- CATERING ORDER", -410.00),
    ("Transport", "HERTZ RENT-A-CAR #4471 SAN DIEGO CA", -610.00),
    ("Subscriptions", "APPLE.COM/BILL 866-712-7753 CA", -899.00),
]

TARGET_PER_CATEGORY = 18
CATEGORIES = list(CATEGORY_TEMPLATES.keys())


def random_date():
    span = (END_DATE - START_DATE).days
    return START_DATE + timedelta(days=random.randint(0, span))


def random_amount(low, high):
    return round(random.uniform(low, high), 2)


def build_rows():
    rows = []

    # Reserve slots for rare merchants and outliers, fill the rest normally.
    rare_by_cat = {}
    for cat, desc, amt in RARE_MERCHANTS:
        rare_by_cat.setdefault(cat, []).append((desc, amt))

    outlier_by_cat = {}
    for cat, desc, amt in OUTLIERS:
        outlier_by_cat.setdefault(cat, []).append((desc, amt))

    for cat in CATEGORIES:
        n_rare = len(rare_by_cat.get(cat, []))
        n_outlier = len(outlier_by_cat.get(cat, []))
        n_normal = TARGET_PER_CATEGORY - n_rare - n_outlier

        templates = CATEGORY_TEMPLATES[cat]["descriptions"]
        low, high = CATEGORY_TEMPLATES[cat]["amount_range"]

        for _ in range(n_normal):
            desc = random.choice(templates)
            amt = random_amount(low, high)
            rows.append((random_date(), desc, amt, cat))

        for desc, amt in rare_by_cat.get(cat, []):
            rows.append((random_date(), desc, amt, cat))

        for desc, amt in outlier_by_cat.get(cat, []):
            rows.append((random_date(), desc, amt, cat))

    rows.sort(key=lambda r: r[0])
    return rows


def main():
    rows = build_rows()
    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "description", "amount", "category"])
        for d, desc, amt, cat in rows:
            writer.writerow([d.isoformat(), desc, f"{amt:.2f}", cat])

    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
