# charts.py — monthly_trend_bar (bar chart version)
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional

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
    - stacked: if True uses stacked bars, else grouped bars.
    - year: filter by year (int) or None for all data.
    Returns Plotly Figure and renders to 'container' if provided.
    """
    # defensive checks
    if date_col not in df.columns or amount_col not in df.columns:
        fig = px.bar(title="Monthly Spending — missing columns")
        if container: container.plotly_chart(fig, use_container_width=True)
        return fig

    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp[tmp[date_col].notna()].copy()
    if tmp.empty:
        fig = px.bar(title="Monthly Spending — no valid dates")
        if container: container.plotly_chart(fig, use_container_width=True)
        return fig

    # apply year filter if requested
    if year is not None:
        tmp = tmp[tmp[date_col].dt.year == int(year)]
        if tmp.empty:
            fig = px.bar(title=f"Monthly Spending — no data for {year}")
            if container: container.plotly_chart(fig, use_container_width=True)
            return fig

    # ensure Amount numeric
    tmp[amount_col] = pd.to_numeric(tmp[amount_col], errors="coerce")
    tmp = tmp[tmp[amount_col].notna()].copy()
    if tmp.empty:
        fig = px.bar(title="Monthly Spending — no numeric amounts")
        if container: container.plotly_chart(fig, use_container_width=True)
        return fig

    # build MonthStart for proper ordering (first day of month)
    tmp["MonthStart"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()

    # aggregate explicit debit and credit sums (case-insensitive)
    if type_col in tmp.columns:
        debit = (tmp[tmp[type_col].astype(str).str.contains("debit", case=False, na=False)]
                 .groupby("MonthStart")[amount_col].sum())
        credit = (tmp[tmp[type_col].astype(str).str.contains("credit", case=False, na=False)]
                  .groupby("MonthStart")[amount_col].sum())
    else:
        # fallback: treat all as Debit
        debit = tmp.groupby("MonthStart")[amount_col].sum()
        credit = pd.Series(dtype=float)

    # create full month index
    start = min(debit.index.min() if not debit.empty else pd.Timestamp.max,
                credit.index.min() if not credit.empty else pd.Timestamp.max)
    end = max(debit.index.max() if not debit.empty else pd.Timestamp.min,
              credit.index.max() if not credit.empty else pd.Timestamp.min)
    if start is pd.Timestamp.max or end is pd.Timestamp.min:
        start = tmp["MonthStart"].min()
        end = tmp["MonthStart"].max()
    full_index = pd.date_range(start=start, end=end, freq="MS")

    debit = debit.reindex(full_index, fill_value=0).astype(float)
    credit = credit.reindex(full_index, fill_value=0).astype(float)

    pivot_df = pd.DataFrame({
        "MonthStart": full_index,
        "Debit": debit.values,
        "Credit": credit.values
    })
    pivot_df["TotalAmount"] = pivot_df[["Debit", "Credit"]].sum(axis=1)

    # build stacked/grouped bar figure
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=pivot_df["MonthStart"],
        y=pivot_df["Debit"],
        name="Debit",
        marker=dict(),
        customdata=np.stack([pivot_df["TotalAmount"].values], axis=1),
        hovertemplate="<b>%{x|%b %Y}</b><br>Type: Debit<br>Amount: %{y:,.2f}<br>Total month: %{customdata[0]:,.2f}<extra></extra>"
    ))

    fig.add_trace(go.Bar(
        x=pivot_df["MonthStart"],
        y=pivot_df["Credit"],
        name="Credit",
        marker=dict(),
        customdata=np.stack([pivot_df["TotalAmount"].values], axis=1),
        hovertemplate="<b>%{x|%b %Y}</b><br>Type: Credit<br>Amount: %{y:,.2f}<br>Total month: %{customdata[0]:,.2f}<extra></extra>"
    ))

    fig.update_layout(
        barmode="stack" if stacked else "group",
        title="Monthly Spending (Debit vs Credit)",
        xaxis=dict(title="Month", tickformat="%b %Y"),
        yaxis=dict(title="Total Amount", tickformat=",.2f"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=60)
    )

    if container is not None:
        container.plotly_chart(fig, use_container_width=True)
    return fig
