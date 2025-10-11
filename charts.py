# charts.py (add or replace monthly_trend function with the code below)
import pandas as pd
import plotly.express as px
from typing import Optional
import streamlit as st  # only used if show_debug with container

def monthly_trend(df: pd.DataFrame,
                  container: Optional[st.delta_generator] = None,
                  date_col: str = "DateTime",
                  amount_col: str = "Amount",
                  kind: str = "line",
                  show_debug: bool = False) -> px.line:
    """
    Robust Monthly Spending Trend chart.
    - df: cleaned DataFrame
    - date_col: datetime column name
    - amount_col: numeric amount column name
    - kind: 'line' or 'area'
    - show_debug: if True and container provided, shows aggregated table in an expander
    Returns: Plotly figure (and renders it to container if given).
    """
    # Defensive checks
    if date_col not in df.columns or amount_col not in df.columns:
        fig = px.line(title="Monthly Spending Trend — missing columns")
        if container: container.plotly_chart(fig, use_container_width=True)
        return fig

    tmp = df.copy()

    # 1) ensure date col is datetime
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp[tmp[date_col].notna()]
    if tmp.empty:
        fig = px.line(title="Monthly Spending Trend — no valid dates")
        if container: container.plotly_chart(fig, use_container_width=True)
        return fig

    # 2) ensure Amount is numeric
    tmp[amount_col] = pd.to_numeric(tmp[amount_col], errors="coerce")
    tmp = tmp[tmp[amount_col].notna()]
    if tmp.empty:
        fig = px.line(title="Monthly Spending Trend — no numeric amounts")
        if container: container.plotly_chart(fig, use_container_width=True)
        return fig

    # 3) month-start timestamp for proper date axis and sorting
    tmp["MonthStart"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()

    # 4) aggregate and sort
    monthly = tmp.groupby("MonthStart", as_index=False)[amount_col].sum().sort_values("MonthStart")
    monthly.columns = ["MonthStart", "TotalAmount"]

    # Optional debug view
    if show_debug and container is not None:
        with container.expander("Monthly aggregation (debug)"):
            container.write(monthly)

    # 5) plot
    if monthly.empty:
        fig = px.line(title="Monthly Spending Trend — no data")
    else:
        if kind == "area":
            fig = px.area(monthly, x="MonthStart", y="TotalAmount", title="Monthly Spending Trend", markers=True)
        else:
            fig = px.line(monthly, x="MonthStart", y="TotalAmount", title="Monthly Spending Trend", markers=True)

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Total Amount",
            xaxis=dict(tickformat="%b %Y"),
            yaxis=dict(tickformat=",.2f"),
            margin=dict(l=60, r=20, t=60, b=60)
        )
        fig.update_traces(hovertemplate="%{x|%b %Y}<br>Total: %{y:,.2f}")

    if container:
        container.plotly_chart(fig, use_container_width=True)
    return fig
