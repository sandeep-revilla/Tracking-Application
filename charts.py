# charts.py — improved monthly_trend
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional

def monthly_trend(df: pd.DataFrame,
                  container=None,
                  date_col: str = "DateTime",
                  amount_col: str = "Amount",
                  type_col: str = "Type",
                  year: Optional[int] = None,
                  show_debit_credit: bool = True) -> go.Figure:
    """
    Robust Monthly Spending Trend — shows Debit & Credit lines, year filter, correct hover.
    """

    # defensive checks
    if date_col not in df.columns or amount_col not in df.columns:
        fig = px.line(title="Monthly Spending Trend — missing columns")
        if container: container.plotly_chart(fig, use_container_width=True)
        return fig

    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp[tmp[date_col].notna()].copy()
    if tmp.empty:
        fig = px.line(title="Monthly Spending Trend — no valid dates")
        if container: container.plotly_chart(fig, use_container_width=True)
        return fig

    if year is not None:
        tmp = tmp[tmp[date_col].dt.year == int(year)]
        if tmp.empty:
            fig = px.line(title=f"Monthly Spending Trend — no data for {year}")
            if container: container.plotly_chart(fig, use_container_width=True)
            return fig

    # ensure Amount numeric
    tmp[amount_col] = pd.to_numeric(tmp[amount_col], errors="coerce")
    tmp = tmp[tmp[amount_col].notna()].copy()
    if tmp.empty:
        fig = px.line(title="Monthly Spending Trend — no numeric amounts")
        if container: container.plotly_chart(fig, use_container_width=True)
        return fig

    # month-start timestamps for proper sorting
    tmp["MonthStart"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()

    # aggregate explicitly for Debit and Credit (case-insensitive)
    if type_col in tmp.columns:
        debit_series = (tmp[tmp[type_col].astype(str).str.contains("debit", case=False, na=False)]
                        .groupby("MonthStart")[amount_col].sum())
        credit_series = (tmp[tmp[type_col].astype(str).str.contains("credit", case=False, na=False)]
                         .groupby("MonthStart")[amount_col].sum())
    else:
        # if no type column, treat everything as Debit
        debit_series = tmp.groupby("MonthStart")[amount_col].sum()
        credit_series = pd.Series(dtype=float)

    # create full month index from min to max month
    start = min(debit_series.index.min() if not debit_series.empty else pd.Timestamp.max,
                credit_series.index.min() if not credit_series.empty else pd.Timestamp.max)
    end = max(debit_series.index.max() if not debit_series.empty else pd.Timestamp.min,
              credit_series.index.max() if not credit_series.empty else pd.Timestamp.min)
    if start is pd.Timestamp.max or end is pd.Timestamp.min:
        # fallback: use tmp
        start = tmp["MonthStart"].min()
        end = tmp["MonthStart"].max()
    full_index = pd.date_range(start=start, end=end, freq="MS")

    # reindex and fill zeros
    debit = debit_series.reindex(full_index, fill_value=0).astype(float)
    credit = credit_series.reindex(full_index, fill_value=0).astype(float)
    pivot_df = pd.DataFrame({
        "MonthStart": full_index,
        "Debit": debit.values,
        "Credit": credit.values
    })
    pivot_df["TotalAmount"] = pivot_df[["Debit", "Credit"]].sum(axis=1)

    # build figure with reliable hover (use text for total month so hovertemplate renders)
    fig = go.Figure()
    x = pivot_df["MonthStart"]

    if show_debit_credit:
        fig.add_trace(go.Scatter(
            x=x, y=pivot_df["Debit"],
            mode="lines+markers", name="Debit",
            text=pivot_df["TotalAmount"],
            hovertemplate="<b>%{x|%b %Y}</b><br>Type: Debit<br>Amount: %{y:,.2f}<br>Total month: %{text:,.2f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=x, y=pivot_df["Credit"],
            mode="lines+markers", name="Credit",
            text=pivot_df["TotalAmount"],
            hovertemplate="<b>%{x|%b %Y}</b><br>Type: Credit<br>Amount: %{y:,.2f}<br>Total month: %{text:,.2f}<extra></extra>"
        ))
    else:
        fig.add_trace(go.Scatter(
            x=x, y=pivot_df["TotalAmount"],
            mode="lines+markers", name="Total",
            hovertemplate="<b>%{x|%b %Y}</b><br>Total: %{y:,.2f}<extra></extra>"
        ))

    fig.update_layout(
        title="Monthly Spending Trend",
        xaxis=dict(title="Month", tickformat="%b %Y"),
        yaxis=dict(title="Total Amount", tickformat=",.2f"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    if container is not None:
        container.plotly_chart(fig, use_container_width=True)
    return fig
