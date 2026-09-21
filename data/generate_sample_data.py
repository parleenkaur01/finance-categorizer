# messy data-handler

"""Generate a synthetic bank-transaction dataset for the finance categorizer.

v2: v1's model learned to associate the surface pattern "#NNNN CITY STATE"
with Groceries, because Groceries examples disproportionately used that
format. Real Shopping (Target, Sephora) and Transport (Hertz) transactions
share that same formatting and got misclassified as a result.

v2 fixes this two ways:
  1. Every category now mixes several description formats (with/without
     store numbers, with/without city+state, processor-prefixed, etc.)
     instead of leaning on one dominant shape.
  2. A handful of merchants across DIFFERENT categories deliberately share
     the exact same "#NNNN CITY STATE" (or "#NNNN") suffix, so the model
     is forced to key off the merchant name rather than the surrounding
     number/city noise.
"""
import csv
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

OUTPUT_PATH = Path(__file__).parent / "sample_transactions.csv"

START_DATE = date(2026, 5, 1)
END_DATE = date(2026, 8, 31)

# (merchant pool, amount range) per category. Pools deliberately mix
# formats: some "#NNNN CITY STATE", some bare names, some processor-prefixed.
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
            "WHOLE FOODS MARKET",
            "SPROUTS FARMERS MKT",
            "H MART GROCERY",
            "GROCERY OUTLET BARGAIN MKT",
            "STOP & SHOP #3391 BOSTON MA",
            "FOOD LION #1145 SAN DIEGO CA",
            "QFC #5807 SEATTLE WA",
            "FRED MEYER #1234 PORTLAND OR",
            "WALMART SUPERCENTER #2291",
            "WAL-MART #2479 SAN DIEGO CA",
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
            "FIVE GUYS #1145 SAN DIEGO CA",
            "GRUBHUB*ORDER CHICAGO IL",
            "OLIVE GARDEN #4409 PHOENIX AZ",
            "IN-N-OUT BURGER #221",
            "SUBWAY #8821 AUSTIN TX",
            "NESPRESSO SEATTLE SEATTLE WA",
            "NESPRESSO BOUTIQUE SAN DIEGO CA",
            "PEET'S COFFEE #1187 PORTLAND OR",
            "BLUE BOTTLE COFFEE SAN FRANCISCO CA",
            "DUNKIN #2291 BOSTON MA",
            "DUTCH BROS COFFEE PHOENIX AZ",
            "SQ *BOMBAY BY CHAAT HOUSE SEATTLE WA",
            "QDOBA 1775 SEATTLE WA",
            "CAVA MISSION VALLEY SAN DIEGO CA",
            "TACOS EL GORDO - F STREET SAN DIEGO CA",
            "TACO BELL #1145 AUSTIN TX",
            "PHO SHIZZLE LLC SEATTLE WA",
        ],
    },
    "Rent": {
        "amount_range": (-2400, -1750),
        "descriptions": [
            "ACH DEBIT GREYSTAR MGMT RENT",
            "ONLINE PMT BLUEROCK PROPERTIES LLC",
            "RENT PAYMENT AVALON BAY COMMUNITIES",
            "ACH DEBIT IRVINE COMPANY APTS",
            "ZELLE PMT TO LANDLORD J RIVERA",
            "ACH DEBIT EQUITY RESIDENTIAL",
            "RENT PAYMENT CAMDEN PROPERTY TRUST",
            "ONLINE PMT MAA APARTMENTS",
            "ACH DEBIT UDR COMMUNITIES",
            "RENT PAYMENT ESSEX PROPERTY TRUST",
            "ACH DEBIT MID-AMERICA APT MGMT",
            "ONLINE PMT AIMCO PROPERTIES",
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
            "DISNEY PLUS 888-905-7888 CA",
            "YOUTUBE PREMIUM GOOGLE.COM CA",
            "PATREON* MEMBERSHIP SF CA",
            "PELOTON MEMBERSHIP NYC NY",
            "DROPBOX*SUBSCRIPTION SF CA",
            "PLANET FITNESS #0921 DALLAS TX",
            "AUDIBLE.COM SEATTLE WA",
            "SIRIUSXM RADIO SVC",
            "MASTERCLASS ANNUAL PLAN",
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
            "ENTERPRISE RENT-A-CAR",
            "SPOTHERO PARKING SAN FRANCISCO CA",
            "ARCO AMPM #5521 OAKLAND CA",
            "EXXON 88213 MAIN ST",
            "CITY PARKING GARAGE #12",
            "SOUTHWEST.COM RESERVATIONS TX",
            "DELTA.COM 800-221-1212 GA",
            "SEATTLE MONORAIL SERVICE 2069052600 WA",
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
            "BOWLERO #2291 CHICAGO IL",
            "SPOTIFY LIVE EVENTS SF CA",
            "PLAYSTATION NETWORK SONY CA",
            "XBOX LIVE MICROSOFT WA",
            "SIX FLAGS MAGIC MTN VALENCIA CA",
            "COMEDY CLUB DOWNTOWN LA",
            "MUSEUM OF MODERN ART NYC NY",
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
            "T-MOBILE AUTOPAY 800-937-8997",
            "SPECTRUM PAYMENT CHARTER COMM",
            "XCEL ENERGY BILLPAY",
            "ACH DEBIT SEWER UTILITY DIST",
            "CONSOLIDATED EDISON NYC NY",
            "NATIONAL GRID UTILITY BILLPAY",
            "CRICKET WIRELESS #3221",
            "ACH DEBIT GAS UTILITY CO",
        ],
    },
    "Shopping": {
        "amount_range": (-150, -12),
        "descriptions": [
            "AMZN MKTP US*2K4RT9 AMZN.COM/BILL",
            "TARGET #1145 SAN DIEGO CA",
            "BEST BUY 00003221 SEATTLE WA",
            "NIKE.COM 800-806-6453 OR",
            "HOME DEPOT #4409 PHOENIX AZ",
            "TJ MAXX #0872 DENVER CO",
            "ETSY.COM - MERCH BROOKLYN NY",
            "MACY'S #0231 CHICAGO IL",
            "ULTA BEAUTY #4471 SAN DIEGO CA",
            "ROSS DRESS FOR LESS",
            "GAMESTOP #0093 AUSTIN TX",
            "PETCO ANIMAL SUPPLIES",
            "ZAPPOS.COM 800-927-7671 NV",
            "WAYFAIR.COM ONLINE ORDER",
            "ZARA.COM NEW YORK NY",
            "SEPHORA FASHION VALLEY SAN DIEGO CA",
            "H&M #1187 SEATTLE WA",
            "PACSUN #1235 SAN DIEGO CA",
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
            "ZELLE DEPOSIT FROM CLIENT",
            "ACH CREDIT FREELANCE PAYMENT",
            "DIRECT DEP SOCIAL SECURITY ADMIN",
            "ACH CREDIT DIVIDEND PAYMENT",
            "INTEREST PAYMENT SAVINGS",
            "ACH CREDIT TAX REFUND IRS",
            "CASH APP*DEPOSIT",
            "CASH APP*DIRECT DEPOSIT",
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
            "NOTARY SERVICE FEE",
            "BANK MAINTENANCE FEE",
            "FOREIGN TXN FEE",
            "MONEYGRAM TRANSFER",
            "WESTERN UNION TRANSFER",
            "LATE PAYMENT FEE",
        ],
    },
}

# One-off / rare merchants that appear exactly once each, sprinkled into
# categories to simulate "never seen before" merchants. Kept out of the
# main description pools above so they can't be repeated by random.choice.
RARE_MERCHANTS = [
    ("Shopping", "IKEA", -212.40),
    ("Dining", "SQ *POKE BOWL TRUCK SD", -14.75),
    ("Dining", "NESPRESSO SEATTLE SEATTLE WA", -18.40),
    ("Dining", "NESPRESSO BOUTIQUE", -22.10),
    ("Dining", "TACOS EL GORDO - F STREET619-9558220 CA", -16.80),
    ("Dining", "PHO SHIZZLE LLC SEATTLE WA", -18.25),
    ("Shopping", "PACSUN #1235 SAN DIEGO CA", -42.10),
    ("Shopping", "H&M 0162SAN DIEGO SAN DIEGO CA", -38.90),
    ("Groceries", "WAL-MART #2479 SAN DIEGO CA", -64.20),
    ("Entertainment", "ESCAPE ROOM SD DOWNTOWN", -42.00),
    ("Other", "DMV RENEWAL FEE SACRAMENTO CA", -58.00),
    ("Transport", "ZIPCAR MEMBERSHIP FEE", -9.00),
    ("Groceries", "WINN-DIXIE", -54.20),
    ("Utilities", "SIMPLE MOBILE AUTOPAY", -45.00),
    ("Income", "STRIPE PAYOUT TRANSFER", 612.40),
]

# Intentional outliers: (category, description, amount) — statistically
# unusual within their category, for testing anomaly detection later.
# HERTZ (Transport) vs ULTA BEAUTY (Shopping) below share the exact same
# "#4471 SAN DIEGO CA" suffix on purpose.
OUTLIERS = [
    ("Dining", "TST* CHEF'S TASTING ROOM LA JOLLA CA", -280.00),
    ("Groceries", "WHOLEFDS MKT 10345 -- CATERING ORDER", -410.00),
    ("Transport", "HERTZ RENT-A-CAR #4471 SAN DIEGO CA", -610.00),
    ("Shopping", "ULTA BEAUTY #4471 SAN DIEGO CA", -540.00),
    ("Subscriptions", "APPLE.COM/BILL 866-712-7753 CA", -899.00),
    ("Rent", "ACH DEBIT GREYSTAR MGMT RENT", -3200.00),
    ("Income", "DIRECT DEP EMPLOYER DISBURSEMENT", 150.00),
    ("Utilities", "COMCAST CABLE COMM 800-934-6489", -410.00),
]

TARGET_PER_CATEGORY = 50
CATEGORIES = list(CATEGORY_TEMPLATES.keys())


def random_date():
    span = (END_DATE - START_DATE).days
    return START_DATE + timedelta(days=random.randint(0, span))


def random_amount(low, high):
    return round(random.uniform(low, high), 2)


def build_rows():
    rows = []

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
