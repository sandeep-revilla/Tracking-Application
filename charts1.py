# charts.py
import plotly.express as px
import streamlit as st


def daily_spend_line_chart(df, debit_col='Total_Spent', credit_col=None):
    """
    Display a line chart of daily debit (and optionally credit) spending.

    Args:
        df (pd.DataFrame): Must contain columns ['Date', debit_col]
                           and optionally credit_col.
        debit_col (str): Column name for debit/expenses.
        credit_col (str): Column name for credit/income.
    """

    # --- Validate ---
    if 'Date' not in df.columns:
        st.error("❌ 'Date' column missing from DataFrame.")
        return
    if debit_col not in df.columns:
        st.error(f"❌ '{debit_col}' column missing from DataFrame.")
        return

    # --- Plot single or double line ---
    if credit_col and credit_col in df.columns:
        y_cols = [debit_col, credit_col]
        title = "💸 Daily Debit and Credit Trend"
    else:
        y_cols = [debit_col]
        title = "💸 Daily Spending Trend"

    # Ensure Date is datetime-like or convertible for sensible plotting
    try:
        # If pandas present, conversion will succeed; if not, Plotly will try its best.
        import pandas as _pd
        if not _pd.api.types.is_datetime64_any_dtype(df['Date']):
            df = df.copy()
            df['Date'] = _pd.to_datetime(df['Date'])
    except Exception:
        pass

    fig = px.line(
        df,
        x='Date',
        y=y_cols,
        title=title,
        markers=True,
        line_shape='spline'
    )

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Amount (₹)',
        template='plotly_white',
        legend_title='Type'
    )

    st.plotly_chart(fig, use_container_width=True)
