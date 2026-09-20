# AI-Powered Personal Finance Tracker

Upload a bank **CSV** or credit-statement **PDF**, auto-categorize transactions with TF-IDF + Logistic Regression, and flag unusual spending with median/IQR anomaly detection.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the model

```bash
python -m src.train
# or: python train.py
```

Prints held-out test accuracy, then saves `src/model.pkl`.

## Run the dashboard

```bash
streamlit run app.py
```

In the sidebar, upload a `.csv` or `.pdf`, or click **Use sample CSV**.

### CSV format

Columns (names are flexible — see `src/parser.py`):

- `date`
- `description` (or memo / merchant)
- `amount` (or debit + credit)

Optional: `category` (if present, the app reports accuracy vs those labels).

### PDF support

`src/pdf_parser.py` extracts Robinhood-style statement rows:

`TranDate  PostDate  Reference  Description  Amount`

Purchases become negative amounts; payments/credits (trailing `-` on the statement) stay positive.

Test a PDF from the CLI:

```bash
python -m src.pdf_parser path/to/statement.pdf
```

## Project layout

| Path | Role |
|------|------|
| `app.py` | Streamlit UI (CSV + PDF upload) |
| `src/parser.py` | CSV loader |
| `src/pdf_parser.py` | PDF statement extractor |
| `src/loaders.py` | Unified CSV/PDF load |
| `src/train.py` | TF-IDF + Logistic Regression training |
| `src/predict.py` | Load `model.pkl` and predict |
| `src/anomaly.py` | Median/IQR + new-merchant flags |
| `src/categorizer.py` | Keyword baseline |
| `data/` | Sample generator + CSVs |
