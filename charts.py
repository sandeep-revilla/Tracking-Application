# charts.py — final monthly_trend_line (force y-axis range & clean)
from typing import Optional, List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def monthly_trend_line(
    df: pd.DataFrame,
    container=None,
    date_col: str = "DateTime",
    amount_col: str = "Amount",
    type_col: str = "Type",
    year: Optional[int] = None,
    currency_symbol: str = "₹",
) -> go.Figure:
    """
    Monthly Spending Line Chart (Debit & Credit).
    Ensures amounts are cleaned, aggregates monthly, picks sensible tickformat and
    forces y-axis range so labels reflect real amounts.
    """

    # 1. Basic validation
    if date_col not in df.columns or amount_col not in df.columns:
        fig = px.line(title="Missing columns (DateTime and Amount required)")
        if container:
            container.plotly_chart(fig, use_container_width=True)
        return fig

    tmp = df.copy()

    # 2. Ensure datetime
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp[tmp[date_col].notna()].copy()
    if tmp.empty:
        fig = px.line(title="No valid dates")
        if container:
            container.plotly_chart(fig, use_container_width=True)
        return fig

    # 3. Optional year filter
    if year is not None:
        tmp = tmp[tmp[date_col].dt.year == int(year)]
        if tmp.empty:
            fig = px.line(title=f"No data for year {year}")
            if container:
                container.plotly_chart(fig, use_container_width=True)
            return fig

    # 4. Clean amount strings and coerce to numeric
    tmp[amount_col] = (
        tmp[amount_col].astype(str)
        .str.replace(r"[^\d\.\-]", "", regex=True)
        .replace({"": None})
    )
    tmp[amount_col] = pd.to_numeric(tmp[amount_col], errors="coerce")
    tmp = tmp[tmp[amount_col].notna()].copy()
    if tmp.empty:
        fig = px.line(title="No numeric amounts")
        if container:
            container.plotly_chart(fig, use_container_width=True)
        return fig

    # 5. MonthStart for grouping
    tmp["MonthStart"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()

    # 6. Aggregate by month & type
    if type_col in tmp.columns:
        debit_s = (
            tmp[tmp[type_col].astype(str).str.contains("debit", case=False, na=False)]
            .groupby("MonthStart")[amount_col].sum()
        )
        credit_s = (
            tmp[tmp[type_col].astype(str).str.contains("credit", case=False, na=False)]
            .groupby("MonthStart")[amount_col].sum()
        )
    else:
        debit_s = tmp.groupby("MonthStart")[amount_col].sum()
        credit_s = pd.Series(dtype=float)

    # 7. Compute safe start/end based on available series
    candidates = []
    if not debit_s.empty:
        candidates += [debit_s.index.min(), debit_s.index.max()]
    if not credit_s.empty:
        candidates += [credit_s.index.min(), credit_s.index.max()]
    if tmp["MonthStart"].notna().any():
        candidates += [tmp["MonthStart"].min(), tmp["MonthStart"].max()]

    candidates = [c for c in candidates if pd.notna(c)]
    if not candidates:
        fig = px.line(title="No monthly data")
        if container:
            container.plotly_chart(fig, use_container_width=True)
        return fig

    start, end = min(candidates), max(candidates)
    full_index = pd.date_range(start=start, end=end, freq="MS")

    debit = debit_s.reindex(full_index, fill_value=0).astype(float)
    credit = credit_s.reindex(full_index, fill_value=0).astype(float)

    pivot_df = pd.DataFrame({"MonthStart": full_index, "Debit": debit.values, "Credit": credit.values})
    pivot_df["Total"] = pivot_df["Debit"] + pivot_df["Credit"]

    # 8. Choose y-axis formatting and explicit range
    max_total = float(pivot_df["Total"].abs().max()) if not pivot_df["Total"].empty else 0.0
    if max_total >= 1_000_000:
        y_tickformat = ",.0f"
    elif max_total >= 1_000:
        y_tickformat = ",.0f"
    else:
        y_tickformat = ",.2f"

    # set range from 0 to 1.05*max_total (if max_total is 0, leave auto)
    if max_total > 0:
        y_range = [0, max_total * 1.05]
    else:
        y_range = None

    # 9. Build figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pivot_df["MonthStart"],
        y=pivot_df["Debit"],
        mode="lines+markers",
        name="Debit",
        line=dict(width=2),
        hovertemplate=(
            "<b>%{x|%b %Y}</b><br>Type: Debit<br>"
            + f"Amount: {currency_symbol}%{{y:{y_tickformat}}}<br>"
            + f"Total month: {currency_symbol}%{{customdata[0]:{y_tickformat}}}<extra></extra>"
        ),
        customdata=np.stack([pivot_df["Total"].values], axis=1)
    ))

    fig.add_trace(go.Scatter(
        x=pivot_df["MonthStart"],
        y=pivot_df["Credit"],
        mode="lines+markers",
        name="Credit",
        line=dict(width=2, dash="dash"),
        hovertemplate=(
            "<b>%{x|%b %Y}</b><br>Type: Credit<br>"
            + f"Amount: {currency_symbol}%{{y:{y_tickformat}}}<br>"
            + f"Total month: {currency_symbol}%{{customdata[0]:{y_tickformat}}}<extra></extra>"
        ),
        customdata=np.stack([pivot_df["Total"].values], axis=1)
    ))

    # 10. Layout with forced y-range if available
    tick_step = max(1, int(len(pivot_df) / 8)) if len(pivot_df) > 0 else 1
    tickvals = pivot_df["MonthStart"].iloc[::tick_step].tolist() if len(pivot_df) > 0 else None

    layout_update = dict(
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
            tickformat=y_tickformat
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=60)
    )

    if y_range is not None:
        layout_update["yaxis"]["range"] = y_range

    fig.update_layout(**layout_update)

    if container:
        container.plotly_chart(fig, use_container_width=True)

    return fig
