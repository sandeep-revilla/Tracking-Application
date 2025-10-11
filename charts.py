# charts.py
"""
Chart utilities for spending dashboard.

Exported:
- monthly_trend_bar(df, container, date_col, amount_col, type_col, year, stacked)
- AVAILABLE_CHARTS (list of names)
"""

from typing import Optional, List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

AVAILABLE_CHARTS: List[str] = ["Monthly Spending (Bar)"]

def monthly_trend_bar(
    df: pd.DataFrame,
    container=None,
    date_col: str = "DateTime",
    amount_col: str = "Amount",
    type_col: str = "Type",
    year: Optional[int] = None,
    stacked: bool = True
) -> go.Figure:
    """
    Monthly Spending Bar Chart (Debit & Credit).
    - df: cleaned dataframe (DateTime, Amount, Type recommended)
    - container: Streamlit container (optional) to render into
    - date_col: name of datetime column
    - amount_col: name of numeric amount column
    - type_col: name of transaction type column (expects values containing 'debit' or 'credit')
    - year: if provided (int), filters data to that calendar year
    - stacked: True => stacked bars; False => grouped bars
    Returns Plotly Figure and renders to container if provided.
    """
    # Defensive checks
    if date_col not in df.columns or amount_col not in df.columns:
        fig = px.bar(title="Monthly Spending — missing columns (DateTime/Amount required)")
        if container is not None:
            container.plotly_chart(fig, use_container_width=True)
        return fig

    tmp = df.copy()

    # Parse and validate dates
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp[tmp[date_col].notna()].copy()
    if tmp.empty:
        fig = px.bar(title="Monthly Spending — no valid dates")
        if container is not None:
            container.plotly_chart(fig, use_container_width=True)
        return fig

    # Apply year filter if requested
    if year is not None:
        tmp = tmp[tmp[date_col].dt.year == int(year)]
        if tmp.empty:
            fig = px.bar(title=f"Monthly Spending — no data for {year}")
            if container is not None:
                container.plotly_chart(fig, use_container_width=True)
            return fig

    # Ensure amounts are numeric
    tmp[amount_col] = pd.to_numeric(tmp[amount_col], errors="coerce")
    tmp = tmp[tmp[amount_col].notna()].copy()
    if tmp.empty:
        fig = px.bar(title="Monthly Spending — no numeric amounts")
        if container is not None:
            container.plotly_chart(fig, use_container_width=True)
        return fig

    # Create MonthStart for grouping (first day of month)
    tmp["MonthStart"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()

    # Aggregate Debit & Credit explicitly (case-insensitive)
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
        # fallback if type_col not present — treat everything as Debit
        debit_series = tmp.groupby("MonthStart")[amount_col].sum()
        credit_series = pd.Series(dtype=float)

    # Build safe start/end from actual available months (avoid Timestamp.max/min bug)
    candidates = []
    if not debit_series.empty:
        candidates.append(debit_series.index.min())
    if not credit_series.empty:
        candidates.append(credit_series.index.min())
    if tmp["MonthStart"].notna().any():
        candidates.append(tmp["MonthStart"].min())
    candidates = [c for c in candidates if pd.notna(c)]
    if not candidates:
        fig = px.bar(title="Monthly Spending — no month data")
        if container is not None:
            container.plotly_chart(fig, use_container_width=True)
        return fig
    start = min(candidates)

    candidates_end = []
    if not debit_series.empty:
        candidates_end.append(debit_series.index.max())
    if not credit_series.empty:
        candidates_end.append(credit_series.index.max())
    if tmp["MonthStart"].notna().any():
        candidates_end.append(tmp["MonthStart"].max())
    candidates_end = [c for c in candidates_end if pd.notna(c)]
    end = max(candidates_end) if candidates_end else start

    # Create full monthly index and reindex series (fill missing months with zeros)
    full_index = pd.date_range(start=start, end=end, freq="MS")
    debit = debit_series.reindex(full_index, fill_value=0).astype(float)
    credit = credit_series.reindex(full_index, fill_value=0).astype(float)

    # Build pivot dataframe used for plotting
    pivot_df = pd.DataFrame({
        "MonthStart": full_index,
        "Debit": debit.values,
        "Credit": credit.values
    })
    pivot_df["TotalAmount"] = pivot_df[["Debit", "Credit"]].sum(axis=1)

    # Build Plotly figure with two bar traces
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pivot_df["MonthStart"],
        y=pivot_df["Debit"],
        name="Debit",
        customdata=np.stack([pivot_df["TotalAmount"].values], axis=1),
        hovertemplate="<b>%{x|%b %Y}</b><br>Type: Debit<br>Amount: %{y:,.2f}<br>Total month: %{customdata[0]:,.2f}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=pivot_df["MonthStart"],
        y=pivot_df["Credit"],
        name="Credit",
        customdata=np.stack([pivot_df["TotalAmount"].values], axis=1),
        hovertemplate="<b>%{x|%b %Y}</b><br>Type: Credit<br>Amount: %{y:,.2f}<br>Total month: %{customdata[0]:,.2f}<extra></extra>"
    ))

    # Layout and formatting
    # Reduce tick density based on number of months (max ~8 visible ticks)
    tick_step = max(1, int(len(pivot_df) / 8)) if len(pivot_df) > 0 else 1
    tickvals = pivot_df["MonthStart"].iloc[::tick_step].tolist() if len(pivot_df) > 0 else None

    fig.update_layout(
        barmode="stack" if stacked else "group",
        title="Monthly Spending (Debit vs Credit)",
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

    if container is not None:
        container.plotly_chart(fig, use_container_width=True)

    return fig
