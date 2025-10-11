import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Debit Spending Tracker", layout="wide")

st.title("💳 Debit Spending Analysis")

# Example: If you already loaded your df_debit before this
# (Make sure 'DateTime' is datetime type and 'Amount' is numeric)
# df_debit = <your cleaned dataframe>

# Convert DateTime just to be safe
df_debit['DateTime'] = pd.to_datetime(df_debit['DateTime'], errors='coerce')
df_debit = df_debit.dropna(subset=['DateTime', 'Amount'])

# Sidebar toggle
mode = st.sidebar.radio(
    "Select View:",
    ["Daily Spend", "Cumulative Spend"],
    index=0
)

# Group by date
daily_spend = (
    df_debit
    .groupby(df_debit['DateTime'].dt.date)['Amount']
    .sum()
    .reset_index()
    .sort_values('DateTime')
)
daily_spend.columns = ['Date', 'Total_Spent']

# Apply cumulative sum if selected
if mode == "Cumulative Spend":
    daily_spend['Total_Spent'] = daily_spend['Total_Spent'].cumsum()

# Create interactive line chart
fig = px.line(
    daily_spend,
    x='Date',
    y='Total_Spent',
    title=f"📊 {mode} Over Time",
    markers=True,
    line_shape='spline'
)

fig.update_traces(line=dict(width=2))
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Total Spent (₹)",
    template="plotly_white",
    title_x=0.5,
    hovermode="x unified"
)

# Display chart
st.plotly_chart(fig, use_container_width=True)

# Optional: Show cleaned data
with st.expander("🔍 View Cleaned Data"):
    st.dataframe(daily_spend, use_container_width=True)
