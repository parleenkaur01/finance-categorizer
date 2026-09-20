"""Streamlit dashboard for the AI-Powered Personal Finance Tracker."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.anomaly import detect_anomalies
from src.loaders import load_transactions_file, load_transactions_upload
from src.predict import categorize_transactions, load_model

SAMPLE_CSV = Path("data/sample_transactions.csv")
SAMPLE_PDF_HINT = "Upload a bank/credit statement (.csv or .pdf)"


st.set_page_config(
    page_title="Personal Finance Tracker",
    page_icon="💸",
    layout="wide",
)

st.title("AI-Powered Personal Finance Tracker")
st.caption(
    "Upload a bank CSV or credit-statement PDF → categorize with keyword rules + "
    "TF-IDF/Logistic Regression → flag unusual spending."
)


@st.cache_resource
def get_model():
    return load_model()


def spending_by_category(df: pd.DataFrame) -> pd.DataFrame:
    expenses = df[df["amount"] < 0].copy()
    expenses["spend"] = expenses["amount"].abs()
    return (
        expenses.groupby("category", as_index=False)["spend"]
        .sum()
        .sort_values("spend", ascending=False)
    )


def monthly_spend(df: pd.DataFrame) -> pd.DataFrame:
    expenses = df[df["amount"] < 0].copy()
    expenses["month"] = expenses["date"].dt.to_period("M").astype(str)
    expenses["spend"] = expenses["amount"].abs()
    return expenses.groupby("month", as_index=False)["spend"].sum()


with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader(
        SAMPLE_PDF_HINT,
        type=["csv", "pdf"],
        help="CSV needs date / description / amount columns. "
        "PDFs: Robinhood-style statements with Tran/Post date rows.",
    )
    use_sample = st.button("Use sample CSV", use_container_width=True)

    st.divider()
    st.markdown(
        "**Model:** TF-IDF (1–2 grams) + Logistic Regression  \n"
        "**Anomalies:** median / IQR per category (threshold = 2.0)"
    )

df = None
source_label = None
error = None

try:
    if uploaded is not None:
        df = load_transactions_upload(uploaded.name, uploaded.getvalue())
        source_label = uploaded.name
    elif use_sample or SAMPLE_CSV.exists():
        if use_sample or uploaded is None:
            # Default to sample on first load so the dashboard isn't empty.
            if use_sample or "bootstrapped" not in st.session_state:
                df = load_transactions_file(SAMPLE_CSV)
                source_label = str(SAMPLE_CSV)
                st.session_state["bootstrapped"] = True
except Exception as exc:  # noqa: BLE001 — surface parse errors in UI
    error = str(exc)

if error:
    st.error(f"Could not load file: {error}")
    st.stop()

if df is None or df.empty:
    st.info("Upload a `.csv` or `.pdf` statement, or click **Use sample CSV**.")
    st.stop()

model = get_model()
# If the file already has labels (sample data), keep them for comparison,
# but still run the model into predicted_category.
labeled = "category" in df.columns
if labeled:
    truth = df["category"].copy()
    df = categorize_transactions(df.drop(columns=["category"]), model=model)
    df = df.rename(columns={"category": "predicted_category"})
    df["category"] = truth
    df["correct"] = df["predicted_category"] == df["category"]
else:
    df = categorize_transactions(df, model=model)
    df["predicted_category"] = df["category"]

# Anomaly detection uses `category` (true labels if present, else predictions).
df = detect_anomalies(df)

st.success(f"Loaded **{len(df)}** transactions from `{source_label}`.")

# KPI row
expenses = df.loc[df["amount"] < 0, "amount"].abs().sum()
income = df.loc[df["amount"] > 0, "amount"].sum()
n_anom = int(df["is_amount_anomaly"].sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{len(df)}")
c2.metric("Total spending", f"${expenses:,.2f}")
c3.metric("Total credits / income", f"${income:,.2f}")
c4.metric("Amount anomalies", f"{n_anom}")

if labeled:
    acc = df["correct"].mean()
    st.metric("Model accuracy vs file labels", f"{acc:.1%}")

tab_overview, tab_txns, tab_anom = st.tabs(
    ["Overview", "Transactions", "Anomalies"]
)

with tab_overview:
    left, right = st.columns(2)
    with left:
        st.subheader("Spending by category")
        by_cat = spending_by_category(df)
        st.bar_chart(by_cat.set_index("category"))
    with right:
        st.subheader("Spending over time")
        by_month = monthly_spend(df)
        if not by_month.empty:
            st.bar_chart(by_month.set_index("month"))
        else:
            st.write("No expense rows to chart.")

with tab_txns:
    st.subheader("Categorized transactions")
    show_cols = ["date", "description", "amount", "predicted_category", "category_source"]
    if labeled:
        show_cols.extend(["category", "correct"])
    show_cols.extend(["is_amount_anomaly", "is_new_merchant"])
    st.dataframe(
        df[show_cols].sort_values("date"),
        use_container_width=True,
        hide_index=True,
    )

with tab_anom:
    st.subheader("Flagged amount outliers")
    anom = df[df["is_amount_anomaly"]].sort_values(
        "amount_anomaly_score", ascending=False
    )
    if anom.empty:
        st.write("No amount anomalies flagged.")
    else:
        st.dataframe(
            anom[
                [
                    "date",
                    "description",
                    "category",
                    "amount",
                    "amount_anomaly_score",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("First-time merchants")
    new_m = df[df["is_new_merchant"]][
        ["date", "description", "predicted_category", "amount"]
    ]
    st.dataframe(new_m, use_container_width=True, hide_index=True)
