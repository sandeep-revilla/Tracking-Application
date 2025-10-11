# charts.py
"""
All charts in a single module.
Each chart exposes a render(df, container=None, **kwargs) behavior through the dispatcher:
    from charts import render_chart, AVAILABLE_CHARTS
    render_chart("Monthly Trend", df, container)

Charts are defensive: if required columns are missing they render a placeholder.
Uses plotly.express for interactive charts (works well in Streamlit).
"""

from typing import Optional, Any, Dict, List
import pandas as pd
import plotly.express as px

AVAILABLE_CHARTS = [
    "Monthly Trend",
    "Spending by Type",
    "Spending by Bank",
    "Top Receivers",
    "Daily Spending",
    "Suspicious Overview",
    "Credit vs Debit",
    "Hourly Pattern",
]

# ---------- Helpers ----------
def _safe_to_datetime(series: pd.Series):
    return pd.to_datetime(series, errors="coerce")

def _empty_fig(title: str = "Missing data"):
    fig = px.scatter(title=title)
    fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False})
    return fig

# ---------- Chart functions ----------
def monthly_trend(df: pd.DataFrame, container=None, date_col="DateTime", amount_col="Amount", kind="line"):
    if date_col not in df.columns or amount_col not in df.columns:
        fig = _empty_fig("Monthly Spending Trend — missing columns")
        if container: container.plotly_chart(fig, use_container_width=True)
        return fig

    tmp = df.dropna(subset=[date_col, amount_col]).copy()
    tmp[date_col] = _safe_to_datetime(tmp[date_col])
    tmp = tmp[tmp[date_col].notna()]
    tmp["Month"] = tmp[date_col].dt.to_period("M").astype(str)
    monthly = tmp.groupby("Month")[amount_col].sum().reset_index()
    if monthly.empty:
        fig = _empty_fig("Monthly Spending Trend — no data")
    else:
        fig = px.line(monthly, x="Month", y=amount_col, markers=True, title="Monthly Spending Trend") if kind != "area" else px.area(monthly, x="Month", y=amount_col, title="Monthly Spending Trend", markers=True)
        fig.update_layout(xaxis_title="Month", yaxis_title="Total Amount")
    if container: container.plotly_chart(fig, use_container_width=True)
    return fig

def spending_by_type(df: pd.DataFrame, container=None, type_col="Type", amount_col="Amount", top_n=20):
    if type_col not in df.columns or amount_col not in df.columns:
        fig = _empty_fig("Spending by Type — missing columns")
        if container: container.plotly_chart(fig, use_container_width=True)
        return fig
    tmp = df.dropna(subset=[type_col, amount_col]).copy()
    grouped = tmp.groupby(type_col)[amount_col].sum().reset_index().sort_values(amount_col, ascending=False).head(top_n)
    if grouped.empty:
        fig = _empty_fig("Spending by Type — no data")
    else:
        fig = px.bar(grouped, x=amount_col, y=type_col, orientation="h", title="Spending by Type")
        fig.update_layout(xaxis_title="Total Amount", yaxis_title="Type")
    if container: container.plotly_chart(fig, use_container_width=True)
    return fig

def spending_by_bank(df: pd.DataFrame, container=None, bank_col="Bank", amount_col="Amount", top_n=20):
    if bank_col not in df.columns or amount_col not in df.columns:
        fig = _empty_fig("Spending by Bank — missing columns")
        if container: container.plotly_chart(fig, use_container_width=True)
        return fig
    tmp = df.dropna(subset=[bank_col, amount_col]).copy()
    grouped = tmp.groupby(bank_col)[amount_col].sum().reset_index().sort_values(amount_col, ascending=True).tail(top_n)
    if grouped.empty:
        fig = _empty_fig("Spending by Bank — no data")
    else:
        fig = px.bar(grouped, x=amount_col, y=bank_col, orientation="h", title="Spending by Bank")
        fig.update_layout(xaxis_title="Total Amount", yaxis_title="Bank")
    if container: container.plotly_chart(fig, use_container_width=True)
    return fig

def top_receivers(df: pd.DataFrame, container=None, sender_col="Sender", amount_col="Amount", top_n=15, by="amount"):
    if sender_col not in df.columns:
        fig = _empty_fig("Top Receivers — missing columns")
        if container: container.plotly_chart(fig, use_container_width=True)
        return fig
    tmp = df.dropna(subset=[sender_col]).copy()
    if by == "count":
        agg = tmp[sender_col].value_counts().reset_index()
        agg.columns = [sender_col, "Count"]
        agg = agg.head(top_n)
        if agg.empty:
            fig = _empty_fig("Top Receivers by Count — no data")
        else:
            fig = px.bar(agg, x="Count", y=sender_col, orientation="h", title="Top Receivers by Count")
    else:
        if amount_col not in df.columns:
            fig = _empty_fig("Top Receivers by Amount — amount column missing")
            if container: container.plotly_chart(fig, use_container_width=True)
            return fig
        agg = tmp.groupby(sender_col)[amount_col].sum().reset_index().sort_values(amount_col, ascending=False).head(top_n)
        if agg.empty:
            fig = _empty_fig("Top Receivers by Amount — no data")
        else:
            fig = px.bar(agg, x=amount_col, y=sender_col, orientation="h", title="Top Receivers by Amount")
    if container: container.plotly_chart(fig, use_container_width=True)
    return fig

def daily_spending(df: pd.DataFrame, container=None, date_col="DateTime", amount_col="Amount"):
    if date_col not in df.columns or amount_col not in df.columns:
        fig = _empty_fig("Daily Spending — missing columns")
        if container: container.plotly_chart(fig, use_container_width=True)
        return fig
    tmp = df.dropna(subset=[date_col, amount_col]).copy()
    tmp[date_col] = _safe_to_datetime(tmp[date_col])
    tmp = tmp[tmp[date_col].notna()]
    tmp["Date"] = tmp[date_col].dt.date
    daily = tmp.groupby("Date")[amount_col].sum().reset_index()
    if daily.empty:
        fig = _empty_fig("Daily Spending — no data")
    else:
        fig = px.bar(daily, x="Date", y=amount_col, title="Daily Spending")
        fig.update_layout(xaxis_title="Date", yaxis_title="Total Amount")
    if container: container.plotly_chart(fig, use_container_width=True)
    return fig

def suspicious_overview(df: pd.DataFrame, container=None, suspicious_col="Suspicious", date_col="DateTime"):
    if suspicious_col not in df.columns:
        fig = _empty_fig("Suspicious Overview — missing column")
        if container: container.plotly_chart(fig, use_container_width=True)
        return fig
    tmp = df.copy()
    if date_col in tmp.columns:
        tmp[date_col] = _safe_to_datetime(tmp[date_col])
    else:
        tmp[date_col] = pd.NaT
    tmp = tmp[tmp[suspicious_col] == True]
    if tmp.empty:
        fig = _empty_fig("Suspicious Overview — no suspicious transactions")
    else:
        tmp["Month"] = tmp[date_col].dt.to_period("M").astype(str)
        counts = tmp.groupby("Month").size().reset_index(name="Count")
        fig = px.bar(counts, x="Month", y="Count", title="Monthly Suspicious Transactions")
        fig.update_layout(xaxis_title="Month", yaxis_title="Count")
    if container: container.plotly_chart(fig, use_container_width=True)
    return fig

def credit_vs_debit(df: pd.DataFrame, container=None, type_col="Type", amount_col="Amount", date_col="DateTime"):
    if type_col not in df.columns or amount_col not in df.columns:
        fig = _empty_fig("Credit vs Debit — missing columns")
        if container: container.plotly_chart(fig, use_container_width=True)
        return fig
    tmp = df.dropna(subset=[date_col, type_col, amount_col]).copy()
    tmp[date_col] = _safe_to_datetime(tmp[date_col])
    tmp = tmp[tmp[date_col].notna()]
    tmp["Month"] = tmp[date_col].dt.to_period("M").astype(str)
    pivot = tmp.groupby(["Month", type_col])[amount_col].sum().reset_index()
    if pivot.empty:
        fig = _empty_fig("Credit vs Debit — no data")
    else:
        fig = px.bar(pivot, x="Month", y=amount_col, color=type_col, title="Credit vs Debit by Month")
        fig.update_layout(xaxis_title="Month", yaxis_title="Amount", barmode="relative")
    if container: container.plotly_chart(fig, use_container_width=True)
    return fig

def hourly_pattern(df: pd.DataFrame, container=None, date_col="DateTime", amount_col="Amount"):
    if date_col not in df.columns or amount_col not in df.columns:
        fig = _empty_fig("Hourly Pattern — missing columns")
        if container: container.plotly_chart(fig, use_container_width=True)
        return fig
    tmp = df.dropna(subset=[date_col, amount_col]).copy()
    tmp[date_col] = _safe_to_datetime(tmp[date_col])
    tmp = tmp[tmp[date_col].notna()]
    tmp["Hour"] = tmp[date_col].dt.hour
    hourly = tmp.groupby("Hour")[amount_col].sum().reset_index()
    if hourly.empty:
        fig = _empty_fig("Hourly Pattern — no data")
    else:
        fig = px.line(hourly, x="Hour", y=amount_col, markers=True, title="Hourly Spending Pattern")
        fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Total Amount")
    if container: container.plotly_chart(fig, use_container_width=True)
    return fig

# ---------- Dispatcher ----------
_DISPATCH = {
    "Monthly Trend": monthly_trend,
    "Spending by Type": spending_by_type,
    "Spending by Bank": spending_by_bank,
    "Top Receivers": top_receivers,
    "Daily Spending": daily_spending,
    "Suspicious Overview": suspicious_overview,
    "Credit vs Debit": credit_vs_debit,
    "Hourly Pattern": hourly_pattern,
}

def render_chart(name: str, df: pd.DataFrame, container=None, **kwargs):
    """
    Render chart by name. Returns the plotly figure.
    Example: render_chart("Monthly Trend", cleaned_df, container)
    """
    fn = _DISPATCH.get(name)
    if not fn:
        fig = _empty_fig(f"Unknown chart: {name}")
        if container: container.plotly_chart(fig, use_container_width=True)
        return fig
    return fn(df, container=container, **kwargs)
