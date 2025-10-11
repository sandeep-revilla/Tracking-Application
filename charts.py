# monthly_trend (replace existing one in charts.py)
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional
import plotly.express as px

def monthly_trend(df: pd.DataFrame,
                  container = None,
                  date_col: str = "DateTime",
                  amount_col: str = "Amount",
                  type_col: str = "Type",
                  year: Optional[int] = None,
                  show_debit_credit: bool = True,
                  kind: str = "line") -> go.Figure:
    """
    Monthly spending trend with Debit vs Credit lines, year filter and rich hover info.
    - df: cleaned dataframe (must contain date_col, amount_col, type_col ideally)
    - year: integer (e.g. 2025) to filter; if None, uses full range
    - show_debit_credit: if True, draws separate lines for Debit and Credit
    Returns Plotly Figure and renders to container if given.
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

    # optional year filter
    if year is not None:
        tmp = tmp[tmp[date_col].dt.year == int(year)]
        if tmp.empty:
            fig = px.line(title=f"Monthly Spending Trend — no data for {year}")
            if container: container.plotly_chart(fig, use_container_width=True)
            return fig

    # ensure amounts numeric
    tmp[amount_col] = pd.to_numeric(tmp[amount_col], errors="coerce")
    tmp = tmp[tmp[amount_col].notna()].copy()
    if tmp.empty:
        fig = px.line(title="Monthly Spending Trend — no numeric amounts")
        if container: container.plotly_chart(fig, use_container_width=True)
        return fig

    # create month start timestamps for x-axis (MS = month start)
    tmp["MonthStart"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()

    # group by month & type
    if type_col in tmp.columns:
        grouped = tmp.groupby(["MonthStart", type_col], as_index=False)[amount_col].sum()
        pivot = grouped.pivot(index="MonthStart", columns=type_col, values=amount_col).fillna(0)
    else:
        # if no type column, treat all as Debit by default
        pivot = tmp.groupby("MonthStart")[amount_col].sum().to_frame(name="Amount")

    # ensure consistent month index (fill missing months with zeros)
    start = pivot.index.min()
    end = pivot.index.max()
    full_index = pd.date_range(start=start, end=end, freq="MS")
    pivot = pivot.reindex(full_index, fill_value=0)
    pivot.index.name = "MonthStart"

    # create DataFrame with totals and ensure Debit & Credit columns exist
    pivot_df = pivot.reset_index()
    # normalize expected columns
    debit_col = None
    credit_col = None
    cols_lower = {str(c).lower(): c for c in pivot_df.columns}
    for k, orig in cols_lower.items():
        if "debit" in k:
            debit_col = orig
        if "credit" in k:
            credit_col = orig

    # If named columns not found, try heuristics
    if debit_col is None and "amount" in cols_lower:
        debit_col = cols_lower["amount"]
    if credit_col is None and "amount" in cols_lower and debit_col is not None:
        # only one column exists; keep credit as 0
        credit_col = None

    # compute TotalAmount across all numeric columns (treat missing as 0)
    numeric_cols = [c for c in pivot_df.columns if c != "MonthStart" and np.issubdtype(pivot_df[c].dtype, np.number)]
    pivot_df["TotalAmount"] = pivot_df[numeric_cols].sum(axis=1) if numeric_cols else 0.0

    # Build figure
    fig = go.Figure()
    x = pivot_df["MonthStart"]

    # add debit line
    if show_debit_credit:
        if debit_col is not None:
            y_debit = pivot_df[debit_col].values
        else:
            y_debit = np.zeros(len(x))
        fig.add_trace(go.Scatter(
            x = x,
            y = y_debit,
            mode = "lines+markers",
            name = "Debit",
            customdata = np.stack([pivot_df["TotalAmount"].values], axis=1),
            hovertemplate = "<b>%{x|%b %Y}</b><br>Type: Debit<br>Amount: %{y:,.2f}<br>Total month: %{customdata[0]:,.2f}<extra></extra>"
        ))

        # add credit line
        if credit_col is not None:
            y_credit = pivot_df[credit_col].values
        else:
            y_credit = np.zeros(len(x))
        fig.add_trace(go.Scatter(
            x = x,
            y = y_credit,
            mode = "lines+markers",
            name = "Credit",
            customdata = np.stack([pivot_df["TotalAmount"].values], axis=1),
            hovertemplate = "<b>%{x|%b %Y}</b><br>Type: Credit<br>Amount: %{y:,.2f}<br>Total month: %{customdata[0]:,.2f}<extra></extra>"
        ))
    else:
        # single total line
        fig.add_trace(go.Scatter(
            x = x,
            y = pivot_df["TotalAmount"].values,
            mode = "lines+markers",
            name = "Total",
            hovertemplate = "<b>%{x|%b %Y}</b><br>Total: %{y:,.2f}<extra></extra>"
        ))

    # layout: proper date x-axis and formatted y-axis
    fig.update_layout(
        title = "Monthly Spending Trend",
        xaxis = dict(title="Month", tickformat="%b %Y", dtick="M1"),
        yaxis = dict(title="Total Amount", tickformat=",.2f"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # render in container if provided
    if container is not None:
        container.plotly_chart(fig, use_container_width=True)

    return fig
