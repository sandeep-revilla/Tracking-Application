# Replace monthly_trend_line in charts.py with this debug-friendly version
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
    show_debug_table: bool = True
) -> go.Figure:
    """
    Robust monthly line chart with debug output.
    - show_debug_table: if True, shows pivot_df in an expander for inspection.
    """

    # Basic checks
    if date_col not in df.columns or amount_col not in df.columns:
        fig = px.line(title="Monthly Spending — missing columns (DateTime/Amount required)")
        if container: container.plotly_chart(fig, use_container_width=True)
        return fig

    tmp = df.copy()

    # 1) parse dates
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp[tmp[date_col].notna()].copy()
    if tmp.empty:
        fig = px.line(title="Monthly Spending — no valid dates")
        if container: container.plotly_chart(fig, use_container_width=True)
        return fig

    # 2) optional year filter
    if year is not None:
        tmp = tmp[tmp[date_col].dt.year == int(year)]
        if tmp.empty:
            fig = px.line(title=f"Monthly Spending — no data for {year}")
            if container: container.plotly_chart(fig, use_container_width=True)
            return fig

    # 3) try to clean amount strings (safe) and coerce
    try:
        tmp[amount_col] = (
            tmp[amount_col]
            .astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .replace({"": None})
        )
        tmp[amount_col] = pd.to_numeric(tmp[amount_col], errors="coerce")
    except Exception:
        tmp[amount_col] = pd.to_numeric(tmp[amount_col], errors="coerce")

    tmp = tmp[tmp[amount_col].notna()].copy()
    if tmp.empty:
        fig = px.line(title="Monthly Spending — no numeric amounts")
        if container: container.plotly_chart(fig, use_container_width=True)
        return fig

    # 4) MonthStart for grouping
    tmp["MonthStart"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()

    # 5) Aggregate debit & credit
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
        debit_series = tmp.groupby("MonthStart")[amount_col].sum()
        credit_series = pd.Series(dtype=float)

    # 6) build safe full_index range from actual series & tmp
    candidates = []
    if not debit_series.empty:
        candidates.append(debit_series.index.min()); candidates.append(debit_series.index.max())
    if not credit_series.empty:
        candidates.append(credit_series.index.min()); candidates.append(credit_series.index.max())
    if tmp["MonthStart"].notna().any():
        candidates.append(tmp["MonthStart"].min()); candidates.append(tmp["MonthStart"].max())
    candidates = [c for c in candidates if pd.notna(c)]
    if not candidates:
        fig = px.line(title="Monthly Spending — no month data")
        if container: container.plotly_chart(fig, use_container_width=True)
        return fig
    start, end = min(candidates), max(candidates)

    full_index = pd.date_range(start=start, end=end, freq="MS")
    debit = debit_series.reindex(full_index, fill_value=0).astype(float)
    credit = credit_series.reindex(full_index, fill_value=0).astype(float)

    pivot_df = pd.DataFrame({
        "MonthStart": full_index,
        "Debit": debit.values,
        "Credit": credit.values
    })
    pivot_df["Total"] = pivot_df["Debit"] + pivot_df["Credit"]

    # --- DEBUG: show pivot_df to verify the numbers being plotted ---
    if show_debug_table and container is not None:
        try:
            with container.expander("Debug: aggregated monthly values (pivot_df)"):
                # show small table and dtypes so you can confirm plotted values
                st = None
                try:
                    # import streamlit lazily to avoid import when not using Streamlit
                    import streamlit as _st
                    st = _st
                except Exception:
                    st = None
                # show head and summary
                if st is not None:
                    st.write(pivot_df.head(20))
                    st.write("dtypes:", pivot_df.dtypes.to_dict())
                    st.write("min/max totals:", float(pivot_df["Total"].min()), float(pivot_df["Total"].max()))
                else:
                    print(pivot_df.head(20))
        except Exception:
            # don't fail the chart if debug printing fails
            pass

    # 7) Decide tickformat: if values are large (>1000) use no decimals, else show 2 decimals
    max_val = float(pivot_df["Total"].abs().max()) if not pivot_df["Total"].empty else 0.0
    if max_val >= 1000:
        y_tickformat = ",.0f"
    else:
        y_tickformat = ",.2f"

    # 8) Build figure
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

    # 9) Layout: ensure date type, tick density and y tickformat chosen above
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
        yaxis=dict(title="Total Amount", tickprefix=f"{currency_symbol} ", tickformat=y_tickformat),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=60)
    )

    if container is not None:
        container.plotly_chart(fig, use_container_width=True)
    return fig
