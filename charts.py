# charts.py
"""
Chart utilities for spending dashboard.

Provides:
- monthly_trend_line(...): monthly line chart (Debit vs Credit) with cleaning and year filter
- AVAILABLE_CHARTS: list of available chart names
"""

from typing import Optional, List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

AVAILABLE_CHARTS: List[str] = ["Monthly Spending (Line)"]


def _clean_amount_series(s: pd.Series) -> pd.Series:
    """
    Clean a series representing money amounts: remove currency symbols, commas, spaces,
    keep digits, dot, minus. Returns a string series suitable for pd.to_numeric.
    """
    return (
        s.astype(str)
         .str.replace(r"[^\d\.\-]", "", regex=True)  # remove everything except digits, dot, minus
         .replace({"": None})  # convert empty strings to None
    )


def monthly_trend_line(
    df: pd.DataFrame,
    container=None,
    date_col: str = "DateTime",
    amount_col: str = "Amount",
    type_col: str = "Type",
    year: Optional[int] = None,
    currency_symbol: str = "₹"
) -> go.Figure:
    """
    Monthly Spending Line Chart (Debit & Credit).
    - df: cleaned/raw dataframe (we apply safe cleaning here)
    - container: optional Streamlit container to render the chart into
    - date_col: name of the datetime column
    - amount_col: name of the amount column (string or numeric)
    - type_col: name of the type column (expects strings containing 'debit'/'credit')
    - year: optional year filter (int) or None for all years
    - currency_symbol: prefix to show on the y-axis and hover (set to "" to disable)
    Returns a Plotly go.Figure (and renders it into container if provided).
    """

    # Basic column checks
    if date_col not in df.columns or amount_col not in df.columns:
        fig = px.line(title="Monthly Spending — missing columns (DateTime/Amount required)")
        if container is not None:
            container.plotly_chart(fig, use_container_width=True)
        return fig

    tmp = df.copy()

    # 1) parse dates
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp[tmp[date_col].notna()].copy()
    if tmp.empty:
        fig = px.line(title="Monthly Spending — no valid dates")
        if container is not None:
            container.plotly_chart(fig, use_container_width=True)
        return fig

    # 2) optional year filter
    if year is not None:
        tmp = tmp[tmp[date_col].dt.year == int(year)]
        if tmp.empty:
            fig = px.line(title=f"Monthly Spending — no data for {year}")
            if container is not None:
                container.plotly_chart(fig, use_container_width=True)
            return fig

    # 3) clean amount strings then coerce to numeric
    tmp[amount_col] = _clean_amount_series(tmp[amount_col])
    tmp[amount_col] = pd.to_numeric(tmp[amount_col], errors="coerce")
    tmp = tmp[tmp[amount_col].notna()].copy()
    if tmp.empty:
        fig = px.line(title="Monthly Spending — no numeric amounts")
        if container is not None:
            container.plotly_chart(fig, use_container_width=True)
        return fig

    # 4) month start for grouping
    tmp["MonthStart"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()

    # 5) aggregate debit & credit explicitly (case-insensitive)
    if type_col in tmp.columns:
        debit_series = (
            tmp[tmp[type_col].astype(str).str.contains("debit", case=False, na=False)]
            .groupby("MonthStart")[amount_col].sum()
        )
        credit_series = (
            tmp[tmp[type_col].astype(str).str.contains("credit", case=False, na=False)]
            .groupby("MonthStart")[amount_col].sum()
        )
    else:
        # fallback: treat everything as Debit
        debit_series = tmp.groupby("MonthStart")[amount_col].sum()
        credit_series = pd.Series(dtype=float)

    # 6) determine safe start/end month from available data
    all_dates = []
    if not debit_series.empty:
        all_dates.append(debit_series.index.min())
        all_dates.append(debit_series.index.max())
    if not credit_series.empty:
        all_dates.append(credit_series.index.min())
        all_dates.append(credit_series.index.max())
    if tmp["MonthStart"].notna().any():
        all_dates.append(tmp["MonthStart"].min())
        all_dates.append(tmp["MonthStart"].max())
    all_dates = [d for d in all_dates if pd.notna(d)]
    if not all_dates:
        fig = px.line(title="Monthly Spending — no month data")
        if container is not None:
            container.plotly_chart(fig, use_container_width=True)
        return fig
    start, end = min(all_dates), max(all_dates)

    # 7) create continuous monthly index and reindex series
    full_index = pd.date_range(start=start, end=end, freq="MS")
    debit = debit_series.reindex(full_index, fill_value=0).astype(float)
    credit = credit_series.reindex(full_index, fill_value=0).astype(float)

    # 8) pivot df for plotting
    pivot_df = pd.DataFrame({
        "MonthStart": full_index,
        "Debit": debit.values,
        "Credit": credit.values
    })
    pivot_df["Total"] = pivot_df["Debit"] + pivot_df["Credit"]

    # 9) build line traces
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pivot_df["MonthStart"],
        y=pivot_df["Debit"],
        mode="lines+markers",
        name="Debit",
        line=dict(width=2, color="#1f77b4"),
        hovertemplate=(
            "<b>%{x|%b %Y}</b><br>"
            "Type: Debit<br>"
            f"Amount: {currency_symbol}%{{y:,.2f}}<br>"
            f"Total month: {currency_symbol}%{{customdata[0]:,.2f}}<extra></extra>"
        ),
        customdata=np.stack([pivot_df["Total"].values], axis=1)
    ))

    fig.add_trace(go.Scatter(
        x=pivot_df["MonthStart"],
        y=pivot_df["Credit"],
        mode="lines+markers",
        name="Credit",
        line=dict(width=2, color="#2ca02c", dash="dash"),
        hovertemplate=(
            "<b>%{x|%b %Y}</b><br>"
            "Type: Credit<br>"
            f"Amount: {currency_symbol}%{{y:,.2f}}<br>"
            f"Total month: {currency_symbol}%{{customdata[0]:,.2f}}<extra></extra>"
        ),
        customdata=np.stack([pivot_df["Total"].values], axis=1)
    ))

    # 10) layout: ticks and formatting
    tick_step = max(1, int(len(pivot_df) / 8)) if len(pivot_df) > 0 else 1
    tickvals = pivot_df["MonthStart"].iloc[::tick_step].tolist() if len(pivot_df) > 0 else None

    fig.update_layout(
        title="Monthly Spending Trend (Debit vs Credit)",
        xaxis=dict(
            title="Month",
            type="date",
            tickformat="%b %Y",
            tickmode="array" if tickvals is not None else "auto",
            tickvals=tickvals
        ),
        yaxis=dict(
            title="Total Amount",
            tickprefix=f"{currency_symbol} ",
            tickformat=",.2f"
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=60)
    )

    # render in Streamlit container if provided
    if container is not None:
        container.plotly_chart(fig, use_container_width=True)

    return fig
