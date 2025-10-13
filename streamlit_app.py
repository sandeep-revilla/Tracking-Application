# streamlit_app.py - main Streamlit entry (uses transform.py)
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
import importlib

st.set_page_config(page_title="Daily Spend (with transform)", layout="wide")
st.title("💳 Daily Spending — now using transform.py")

# ---------------------- PLACEHOLDERS / FUTURE MODULES ----------------------
# Later we will replace these comments with real imports:
#   from io import read_google_sheet         # -> io.py (data loading from Google Sheets)
#   from charts import render_chart          # -> charts.py (all plotting functions)
# For now we import transform (cleaning + aggregation) which we implemented below.
try:
    transform = importlib.import_module("transform")
except Exception as e:
    st.error("Could not import transform.py — ensure transform.py exists in the same folder.")
    st.exception(e)
    st.stop()

# ---------------- Sidebar: data source (upload or sample) ----------------
with st.sidebar:
    st.header("Data input")
    data_source = st.radio("Load data from", ["Upload CSV/XLSX", "Use sample data"], index=0)
    uploaded = None
    if data_source == "Upload CSV/XLSX":
        uploaded = st.file_uploader("Upload CSV or XLSX (HDFC / Indian Bank / IFTTT sheet)", type=["csv", "xlsx"])
    st.markdown("---")
    st.write("Series to include")
    show_debit = st.checkbox("Debit (Total_Spent)", value=True)
    show_credit = st.checkbox("Credit (Total_Credit)", value=True)
    st.markdown("---")
    st.write("Chart type (placeholder)")
    chart_type = st.selectbox("Chart type", [
        "Daily line", "Stacked area", "Monthly bars", "Rolling average",
        "Cumulative sum", "Calendar heatmap", "Histogram of amounts", "Treemap by category"
    ])
    st.markdown("---")
    if st.button("Refresh"):
        st.experimental_rerun()

# ---------------- Load raw data ----------------
def load_from_upload(uploaded_file):
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        return df
    except Exception as e:
        st.error(f"Failed to parse upload: {e}")
        return pd.DataFrame()

def sample_data():
    today = datetime.utcnow().date()
    rows = []
    for i in range(30):
        d = today - timedelta(days=29-i)
        amt = (i % 5 + 1) * 100
        # Make some entries negative to simulate credits
        if i % 7 == 0:
            amt = -amt
            t = "credit"
        else:
            t = "debit"
        rows.append({"timestamp": pd.to_datetime(d), "description": f"Sample txn {i+1}", "Amount": amt, "Type": t})
    return pd.DataFrame(rows)

if data_source == "Upload CSV/XLSX":
    df_raw = load_from_upload(uploaded)
    if df_raw is None or df_raw.empty:
        st.info("No valid upload detected — using sample data. Upload a CSV/XLSX to replace it.")
        df_raw = sample_data()
else:
    df_raw = sample_data()

# ---------------- Transform (CALL INTO transform.py) ----------------
with st.spinner("Cleaning and deriving columns via transform.convert_columns_and_derives..."):
    converted_df = transform.convert_columns_and_derives(df_raw)

with st.spinner("Computing daily totals via transform.compute_daily_totals..."):
    merged = transform.compute_daily_totals(converted_df)

# ---------------- Sidebar: simple date filters ----------------
with st.sidebar:
    st.header("Filters")
    if not merged.empty:
        merged['Date'] = pd.to_datetime(merged['Date']).dt.normalize()
        min_date = merged['Date'].min().date()
        max_date = merged['Date'].max().date()
        sel_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    else:
        sel_range = (None, None)

# Apply date-range filter to the plot dataframe
plot_df = merged.copy()
if sel_range and sel_range[0] and sel_range[1]:
    plot_df = plot_df[(plot_df['Date'].dt.date >= sel_range[0]) & (plot_df['Date'].dt.date <= sel_range[1])]

plot_df = plot_df.sort_values("Date").reset_index(drop=True)
plot_df['Total_Spent'] = pd.to_numeric(plot_df.get('Total_Spent', 0), errors='coerce').fillna(0.0).astype('float64')
plot_df['Total_Credit'] = pd.to_numeric(plot_df.get('Total_Credit', 0), errors='coerce').fillna(0.0).astype('float64')

# ---------------- Series selection ----------------
series_selected = []
if show_debit:
    series_selected.append('Total_Spent')
if show_credit:
    series_selected.append('Total_Credit')

# ---------------- Chart: simple Altair line using the aggregated data ----------------
st.subheader("Daily Spend and Credit (using transform.py)")

if plot_df.empty:
    st.info("No data to display for the selected range.")
else:
    vars_to_plot = [c for c in ['Total_Spent', 'Total_Credit'] if c in series_selected]
    plot_df_long = plot_df.melt(id_vars='Date', value_vars=vars_to_plot, var_name='Type', value_name='Amount').sort_values('Date')
    if plot_df_long.empty:
        st.info("No series selected or no data for chosen range.")
    else:
        chart = alt.Chart(plot_df_long).mark_line(point=True).encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Amount:Q', title='Amount', axis=alt.Axis(format=",.0f")),
            color=alt.Color('Type:N', title='Type'),
            tooltip=[alt.Tooltip('Date:T', title='Date', format='%Y-%m-%d'),
                     alt.Tooltip('Type:N', title='Type'),
                     alt.Tooltip('Amount:Q', title='Amount', format=',')]
        ).interactive()
        st.altair_chart(chart.properties(height=420), use_container_width=True)

# ---------------- Rows and CSV download (filtered) ----------------
st.subheader("Rows (matching selection)")
rows_df = converted_df.copy()

# ensure timestamp present
if 'timestamp' in rows_df.columns:
    rows_df['timestamp'] = pd.to_datetime(rows_df['timestamp'], errors='coerce')
else:
    if 'date' in rows_df.columns:
        rows_df['timestamp'] = pd.to_datetime(rows_df['date'], errors='coerce')
    else:
        rows_df['timestamp'] = pd.NaT

# apply date filter to rows
if sel_range and sel_range[0] and sel_range[1]:
    rows_df = rows_df[(pd.to_datetime(rows_df['timestamp']).dt.date >= sel_range[0]) &
                      (pd.to_datetime(rows_df['timestamp']).dt.date <= sel_range[1])]

if rows_df.empty:
    st.write("No rows to show for the current selection.")
else:
    st.dataframe(rows_df.reset_index(drop=True))
    csv_bytes = rows_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download rows (CSV)", csv_bytes, file_name="transactions_rows.csv", mime="text/csv")

# ---------------- Footer / notes ----------------
st.markdown("""
---
**Notes:**  
- This app now uses `transform.py` to clean and aggregate data.  
- Placeholders for `io.py` (Google Sheets loading) and `charts.py` (advanced plots) are commented near the top — we'll implement them next.  
""")
