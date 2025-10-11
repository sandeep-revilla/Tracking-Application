# charts.py
import plotly.express as px
import pandas as pd
import streamlit as st

def daily_trend_line(df: pd.DataFrame, container=None, currency_symbol: str = "₹"):
    """Plot daily Debit vs Credit trend using Plotly."""
    if df.empty or "DateTime" not in df.columns or "Amount" not in df.columns:
        st.warning("Insufficient data for daily trend line chart.")
        return

    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime"])

    # Group by date and type
    daily_summary = (
        df.groupby([df["DateTime"].dt.date, "Type"])["Amount"]
        .sum()
        .reset_index()
        .rename(columns={"DateTime": "Date"})
    )

    if daily_summary.empty:
        st.warning("No debit/credit data found.")
        return

    fig = px.line(
        daily_summary,
        x="Date",
        y="Amount",
        color="Type",
        markers=True,
        line_shape="spline",
        title=f"💸 Daily Debit vs Credit Over Time ({currency_symbol})"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=f"Total Amount ({currency_symbol})",
        template="plotly_white",
        legend_title_text="Transaction Type",
        hovermode="x unified"
    )

    if container is None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        with container:
            st.plotly_chart(fig, use_container_width=True)
