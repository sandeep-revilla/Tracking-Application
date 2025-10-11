# charts.py — debug-friendly monthly_trend_line (replace existing)
from typing import Optional
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
    Debug-friendly monthly line chart. Forces numeric types, shows pivot_df, uses large markers/lines,
    and forces y-axis range from numeric max.
    """
    # Basic validation
    if date_col not in df.columns or amount_col not in df.columns:
        fig = px.line(title="Missing columns (DateTime/Amount required)")
        if container:
            container.plotly_chart(fig, use_container_width=True)
        return fig

    tmp = df.copy()

    # parse dates
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp[tmp[date_col].notna()].copy()
    if tmp.empty:
        fig = px.line(title="No valid dates")
        if container:
            container.plotly_chart(fig, use_container_width=True)
        return fig

    # year filter
    if year is not None:
        tmp = tmp[tmp[date_col].dt.year == int(year)]
        if tmp.empty:
            fig = px.line(title=f"No data for year {year}")
            if container:
                container.plotly_chart(fig, use_container_width=True)
            return fig

    # clean amounts (strip non-digits) and coerce
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

    # month start
    tmp["MonthStart"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()

    # aggregate debit & credit explicitly
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

    # safe min/max calculation
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

    # continuous month index
    full_index = pd.date_range(start=start, end=end, freq="MS")
    debit = debit_s.reindex(full_index, fill_value=0).astype(float)
    credit = credit_s.reindex(full_index, fill_value=0).astype(float)

    pivot_df = pd.DataFrame({
        "MonthStart": full_index,
        "Debit": debit.values,
        "Credit": credit.values
    })
    pivot_df["Total"] = pivot_df["Debit"] + pivot_df["Credit"]

    # --- DEBUG OUTPUT: show pivot_df and numeric summary in Streamlit if available ---
    if container is not None:
        try:
            import streamlit as st
            with container.expander("Debug: aggregated monthly values (pivot_df)"):
                st.write(pivot_df.reset_index(drop=True))
                st.write("dtypes:", pivot_df.dtypes.to_dict())
                st.write("min/max totals:", float(pivot_df["Total"].min()), float(pivot_df["Total"].max()))
        except Exception:
            # ignore printing errors
            pass

    # compute numeric maxima for axis
    max_total = float(pivot_df["Total"].abs().max()) if not pivot_df["Total"].empty else 0.0
    # force y-axis range so ticks reflect true magnitudes
    if max_total > 0:
        y_range = [0, max_total * 1.05]
    else:
        y_range = None

    # build visible traces: larger markers + thicker lines
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pivot_df["MonthStart"],
        y=pivot_df["Debit"],
        mode="lines+markers",
        name="Debit",
        line=dict(width=3),
        marker=dict(size=8),
        hovertemplate=(
            "<b>%{x|%b %Y}</b><br>Type: Debit<br>"
            + f"Amount: {currency_symbol}%{{y:,.2f}}<br>"
            + f"Total month: {currency_symbol}%{{customdata[0]:,.2f}}<extra></extra>"
        ),
        customdata=np.stack([pivot_df["Total"].values], axis=1)
    ))
    fig.add_trace(go.Scatter(
        x=pivot_df["MonthStart"],
        y=pivot_df["Credit"],
        mode="lines+markers",
        name="Credit",
        line=dict(width=3, dash="dash"),
        marker=dict(size=8),
        hovertemplate=(
            "<b>%{x|%b %Y}</b><br>Type: Credit<br>"
            + f"Amount: {currency_symbol}%{{y:,.2f}}<br>"
            + f"Total month: {currency_symbol}%{{customdata[0]:,.2f}}<extra></extra>"
        ),
        customdata=np.stack([pivot_df["Total"].values], axis=1)
    ))

    # tick density
    tick_step = max(1, int(len(pivot_df) / 8)) if len(pivot_df) > 0 else 1
    tickvals = pivot_df["MonthStart"].iloc[::tick_step].tolist() if len(pivot_df) > 0 else None

    layout = dict(
        title="Monthly Spending Trend (Debit vs Credit)",
        xaxis=dict(title="Month", type="date", tickformat="%b %Y",
                   tickmode="array" if tickvals is not None else "auto", tickvals=tickvals),
        yaxis=dict(title="Total Amount", tickprefix=f"{currency_symbol} ", tickformat=",.0f"),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    if y_range is not None:
        layout["yaxis"]["range"] = y_range

    fig.update_layout(**layout)

    if container:
        container.plotly_chart(fig, use_container_width=True)
    return fig
