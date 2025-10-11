# charts.py
"""
Chart utilities for spending dashboard.

Exports:
- monthly_trend_line(df, container, date_col, amount_col, type_col, year)
- AVAILABLE_CHARTS
"""

from typing import Optional, List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

AVAILABLE_CHARTS: List[str] = ["Monthly Spending (Line)"]

def monthly_trend_line(
    df: pd.DataFrame,
    container=None,
    date_col: str = "DateTime",
    amount_col: str = "Amount",
    type_col: str = "Type",
    year: Optional[int] = None,
) -> go.Figure:
    """
    Monthly Spending Line Chart (Debit & Credit).
    Shows monthly total debit and credit over time.
    - year: optional filter (int)
    Returns Plotly Figure and renders to Streamlit container if provided.
    """

    # Defensive checks
    if date_col not in df.columns or amount_col not in df.columns:
        fig = px.line(title="Monthly Spending — missing columns (DateTime/Amount required)")
        if container:
            container.plotly_chart(fig, use_container_width=True)
        return fig

    tmp = df.copy()

    # Ensure valid datetime
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp[tmp[date_col].notna()].copy()
    if tmp.empty:
        fig = px.line(title="Monthly Spending — no valid dates")
        if container:
            container.plotly_chart(fig, use_container_width=True)
        return fig

    # Filter by year if selected
    if year is not None:
        tmp = tmp[tmp[date_col].dt.year == int(year)]
        if tmp.empty:
            fig = px.line(title=f"Monthly Spending — no data for {year}")
            if container:
                container.plotly_chart(fig, use_container_width=True)
            return fig

    # Convert Amount to numeric
    tmp[amount_col] = pd.to_numeric(tmp[amount_col], errors="coerce")
    tmp = tmp[tmp[amount_col].notna()].copy()
    if tmp.empty:
        fig = px.line(title="Monthly Spending — no numeric amounts")
        if container:
            container.plotly_chart(fig, use_container_width=True)
        return fig

    # MonthStart (first day of month)
    tmp["MonthStart"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()

    # Aggregate monthly Debit and Credit totals
    if type_col in tmp.columns:
        debit = (
            tmp[tmp[type_col].astype(str).str.contains("debit", case=False, na=False)]
            .groupby("MonthStart")[amount_col].sum()
        )
        credit = (
            tmp[tmp[type_col].astype(str).str.contains("credit", case=False, na=False)]
            .groupby("MonthStart")[amount_col].sum()
        )
    else:
        debit = tmp.groupby("MonthStart")[amount_col].sum()
        credit = pd.Series(dtype=float)

    # Determine start and end months safely
    all_dates = []
    if not debit.empty:
        all_dates.append(debit.index.min())
        all_dates.append(debit.index.max())
    if not credit.empty:
        all_dates.append(credit.index.min())
        all_dates.append(credit.index.max())
    if tmp["MonthStart"].notna().any():
        all_dates.append(tmp["MonthStart"].min())
        all_dates.append(tmp["MonthStart"].max())
    all_dates = [d for d in all_dates if pd.notna(d)]
    if not all_dates:
        fig = px.line(title="Monthly Spending — no month data")
        if container:
            container.plotly_chart(fig, use_container_width=True)
        return fig
    start, end = min(all_dates), max(all_dates)

    # Create continuous monthly range (fill missing months with 0)
    full_index = pd.date_range(start=start, end=end, freq="MS")
    debit = debit.reindex(full_index, fill_value=0)
    credit = credit.reindex(full_index, fill_value=0)

    # Build dataframe
    pivot_df = pd.DataFrame({
        "MonthStart": full_index,
        "Debit": debit.values,
        "Credit": credit.values,
    })
    pivot_df["Total"] = pivot_df["Debit"] + pivot_df["Credit"]

    # Build line chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pivot_df["MonthStart"],
        y=pivot_df["Debit"],
        mode="lines+markers",
        name="Debit",
        line=dict(width=2),
        hovertemplate="<b>%{x|%b %Y}</b><br>Type: Debit<br>Amount: %{y:,.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=pivot_df["MonthStart"],
        y=pivot_df["Credit"],
        mode="lines+markers",
        name="Credit",
        line=dict(width=2, dash="dot"),
        hovertemplate="<b>%{x|%b %Y}</b><br>Type: Credit<br>Amount: %{y:,.2f}<extra></extra>"
    ))

    # Format layout
    tick_step = max(1, int(len(pivot_df) / 8))
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
        yaxis=dict(title="Total Amount", tickformat=",.2f"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=60)
    )

    if container:
        container.plotly_chart(fig, use_container_width=True)

    return fig
