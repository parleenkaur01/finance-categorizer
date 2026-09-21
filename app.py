"""Streamlit dashboard for the AI-Powered Personal Finance Tracker."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.anomaly import detect_anomalies
from src.loaders import load_transactions_uploads
from src.predict import MODEL_PATH, categorize_transactions, load_model

SAMPLE_PDF_HINT = "Upload one or more bank/credit statements (.csv or .pdf)"
ALL_MONTHS_LABEL = "Full year"


st.set_page_config(
    page_title="Personal Finance Tracker",
    page_icon="💸",
    layout="wide",
)

st.title("AI-Powered Personal Finance Tracker")


@st.cache_resource
def get_model(mtime: float):
    return load_model()


def month_choices(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return (label, period-key) options, with full-year first."""
    periods = sorted(pd.to_datetime(df["date"]).dt.to_period("M").unique())
    choices = [(ALL_MONTHS_LABEL, ALL_MONTHS_LABEL)]
    for period in periods:
        choices.append((period.strftime("%B %Y"), str(period)))
    return choices


def filter_by_month(df: pd.DataFrame, period_key: str) -> pd.DataFrame:
    if period_key == ALL_MONTHS_LABEL:
        return df
    months = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    return df.loc[months == period_key].copy()


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


def expenses_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["amount"] < 0].copy()


def money(value: float) -> str:
    return f"${value:,.2f}"


_CARD_PAYMENT = (
    "PAYMENT - THANK YOU",
    "PAYMENT THANK YOU",
    "AUTOPAY PAYMENT",
)


def is_card_payment(description: str) -> bool:
    text = str(description).upper()
    return any(token in text for token in _CARD_PAYMENT)


def refunds_only(df: pd.DataFrame) -> pd.DataFrame:
    credits = df[df["amount"] > 0].copy()
    if credits.empty:
        return credits
    return credits[~credits["description"].map(is_card_payment)]


def category_from_chart(event) -> str | None:
    if event is None:
        return None
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    if selection is None:
        return None

    candidates = []
    if isinstance(selection, dict):
        candidates.extend(selection.values())
    for attr in ("pick", "point"):
        value = getattr(selection, attr, None)
        if value is not None:
            candidates.append(value)

    for points in candidates:
        if isinstance(points, list) and points:
            first = points[0]
            if isinstance(first, dict) and first.get("category"):
                return first.get("category")
        if isinstance(points, dict) and points.get("category"):
            return points.get("category")
    return None


def render_purchases(rows: pd.DataFrame) -> None:
    show = rows.sort_values("date", ascending=False)[
        ["date", "description", "amount"]
    ].copy()
    show["date"] = pd.to_datetime(show["date"]).dt.strftime("%b %d, %Y")
    show["amount"] = show["amount"].abs().map(money)
    show = show.rename(
        columns={
            "date": "Date purchased",
            "description": "What you bought",
            "amount": "Amount",
        }
    )
    st.dataframe(show, use_container_width=True, hide_index=True)


def render_credits(rows: pd.DataFrame) -> None:
    show = rows.sort_values("date", ascending=False)[
        ["date", "description", "amount"]
    ].copy()
    show["date"] = pd.to_datetime(show["date"]).dt.strftime("%b %d, %Y")
    show["amount"] = show["amount"].map(money)
    show = show.rename(
        columns={
            "date": "Date",
            "description": "Description",
            "amount": "Amount",
        }
    )
    st.dataframe(show, use_container_width=True, hide_index=True)


with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader(
        SAMPLE_PDF_HINT,
        type=["csv", "pdf"],
        accept_multiple_files=True,
        help="CSV needs date / description / amount columns. "
        "PDFs: Robinhood-style statements with Tran/Post date rows. "
        "You can select several files at once.",
    )

df = None
source_label = None
error = None
load_warnings: list[str] = []

try:
    if uploaded:
        files = [(item.name, item.getvalue()) for item in uploaded]
        df = load_transactions_uploads(files)
        names = [item.name for item in uploaded]
        if len(names) == 1:
            source_label = names[0]
        else:
            source_label = f"{len(names)} files ({', '.join(names)})"
        load_warnings = list(df.attrs.get("load_errors") or [])
except Exception as exc:  # noqa: BLE001 — surface parse errors in UI
    error = str(exc)

if error:
    st.error(f"Could not load file: {error}")
    st.stop()

if df is None or df.empty:
    st.info("Upload one or more `.csv` or `.pdf` statements to get started.")
    st.stop()

model = get_model(MODEL_PATH.stat().st_mtime)
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
if load_warnings:
    st.warning("Some files could not be loaded:\n\n" + "\n".join(f"- {w}" for w in load_warnings))

choices = month_choices(df)
labels = [label for label, _key in choices]
selected_label = st.selectbox(
    "Month",
    labels,
    index=0,
    help="Full year shows all uploaded statements. Pick a month to update totals and categories.",
)
selected_key = dict(choices)[selected_label]
if st.session_state.get("category_month") != selected_key:
    st.session_state["category_month"] = selected_key
    st.session_state.pop("picked_category", None)

cat_df = filter_by_month(df, selected_key)
period_caption = (
    "Full year"
    if selected_key == ALL_MONTHS_LABEL
    else selected_label
)

# KPI row — follows the selected month
expenses = cat_df.loc[cat_df["amount"] < 0, "amount"].abs().sum()
refunds = refunds_only(cat_df)
refunds_total = float(refunds["amount"].sum()) if not refunds.empty else 0.0
n_anom = int(cat_df["is_amount_anomaly"].sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{len(cat_df)}")
c2.metric("Total spending", f"${expenses:,.2f}")
c3.metric("Refunds", money(refunds_total))
c4.metric("Unusual purchases", f"{n_anom}")
st.caption(f"Totals for {period_caption}.")

with st.expander(
    f"Refunds  ·  {money(refunds_total)}  ·  "
    f"{len(refunds)} item{'s' if len(refunds) != 1 else ''}"
):
    if refunds.empty:
        st.caption("No refunds in this period.")
    else:
        render_credits(refunds)

if labeled:
    acc = cat_df["correct"].mean() if not cat_df.empty else 0.0
    st.metric("Model accuracy vs file labels", f"{acc:.1%}")

st.subheader("Spending by category")
if selected_key == ALL_MONTHS_LABEL:
    st.caption("Full year — click a category to see every purchase.")
else:
    st.caption(f"{selected_label} — click a category to see purchases from this month.")

by_cat = spending_by_category(cat_df)
if by_cat.empty:
    st.write("No purchases to chart for this period.")
else:
    pick = alt.selection_point(fields=["category"], name="pick")
    chart = (
        alt.Chart(by_cat)
        .mark_bar()
        .encode(
            x=alt.X("category:N", sort="-y", title=None),
            y=alt.Y("spend:Q", title="Spent", axis=alt.Axis(format="$,.0f")),
            color=alt.condition(pick, alt.value("#176b52"), alt.value("#8eb9a8")),
            tooltip=[
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("spend:Q", title="Spent", format="$,.2f"),
            ],
        )
        .add_params(pick)
        .properties(height=320)
    )
    event = st.altair_chart(chart, on_select="rerun", use_container_width=True)
    picked = category_from_chart(event)
    if picked:
        st.session_state["picked_category"] = picked
    selected = st.session_state.get("picked_category")

    for _, row in by_cat.iterrows():
        category = row["category"]
        spent = float(row["spend"])
        purchases = expenses_only(cat_df)
        purchases = purchases[purchases["category"] == category]
        label = f"{category}  ·  {money(spent)}  ·  {len(purchases)} purchase{'s' if len(purchases) != 1 else ''}"
        with st.expander(label, expanded=(selected == category)):
            if purchases.empty:
                st.caption("No purchases in this category.")
            else:
                render_purchases(purchases)

st.subheader("Spending over time")
by_month = monthly_spend(df)
if not by_month.empty:
    st.bar_chart(by_month.set_index("month"))
else:
    st.write("No expense rows to chart.")

with st.expander("Anomalies"):
    st.subheader("Unusual purchases")
    anom = cat_df[cat_df["is_amount_anomaly"]].sort_values("amount")
    if anom.empty:
        st.write("No purchases over the category limits.")
    else:
        anom_show = anom[
            ["date", "description", "category", "amount", "anomaly_reason"]
        ].copy()
        anom_show["date"] = pd.to_datetime(anom_show["date"]).dt.strftime("%b %d, %Y")
        anom_show["amount"] = anom_show["amount"].abs().map(money)
        anom_show = anom_show.rename(
            columns={
                "date": "Date purchased",
                "description": "What you bought",
                "category": "Category",
                "amount": "Amount",
                "anomaly_reason": "Why",
            }
        )
        st.dataframe(anom_show, use_container_width=True, hide_index=True)
