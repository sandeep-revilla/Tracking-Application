# streamlit_app.py - main Streamlit entrypoint (uses charts.py for selectable charts)
import streamlit as st
import pandas as pd
import importlib
from datetime import datetime, timedelta

st.set_page_config(page_title="Daily Spend (selectable charts)", layout="wide")
st.title("💳 Daily Spending — choose a chart from the sidebar")

# ------------------ Imports / placeholders ------------------
# transform.py must exist and provide convert_columns_and_derives + compute_daily_totals
try:
    transform = importlib.import_module("transform")
except Exception as e:
    st.error("transform.py missing or failing to import. Add transform.py to the same directory.")
    st.exception(e)
    st.stop()

# optional I/O helper (for Google Sheets)
try:
    import io_helpers as io_mod
except Exception:
    io_mod = None

# charts module (we just created)
try:
    charts_mod = importlib.import_module("charts")
except Exception as e:
    charts_mod = None
    st.error("charts.py missing or failed to import. Add charts.py to the project.")
    st.exception(e)
    st.stop()

# ------------------ Sidebar: data source & chart selector ------------------
with st.sidebar:
    st.header("Data input & options")
    data_source = st.radio("Load data from", ["Upload CSV/XLSX", "Google Sheet (optional)", "Use sample data"], index=0)
    SHEET_ID = st.text_input("Google Sheet ID (between /d/ and /edit)", value="")
    RANGE = st.text_input("Range or Sheet Name", value="History Transactions")
    CREDS_FILE = st.text_input("Service Account JSON File (optional)", value="creds/service_account.json")
    st.markdown("---")

    st.write("Pick chart")
    chart_type = st.selectbox("Chart type", ["Daily line", "Monthly bars", "Top categories (Top-N)"], index=0)

    st.markdown("---")
    st.write("Series to include (applies to Daily/Monthly charts)")
    show_debit = st.checkbox("Debit (Total_Spent)", value=True)
    show_credit = st.checkbox("Credit (Total_Credit)", value=True)

    # top-n control only relevant when Top categories selected (we show it always in sidebar; charts will use it)
    top_n = st.slider("Top N (for categories)", min_value=3, max_value=20, value=5, step=1)

    st.markdown("---")
    if st.button("Refresh"):
        st.experimental_rerun()

# ------------------ Data loaders (safe wrappers) ------------------
def load_from_upload(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Failed to parse upload: {e}")
        return pd.DataFrame()


def load_from_sheet_safe(sheet_id: str, range_name: str, creds_file: str) -> pd.DataFrame:
    if io_mod is None:
        st.error("io_helpers.py not available. Add io_helpers.py to enable Google Sheets loading.")
        return pd.DataFrame()
    try:
        secrets = st.secrets if hasattr(st, "secrets") else None
        return io_mod.read_google_sheet(sheet_id, range_name, creds_info=None, creds_file=creds_file, secrets=secrets)
    except Exception as e:
        st.error(f"Failed to read Google Sheet: {e}")
        return pd.DataFrame()

# ------------------ Sample data fallback ------------------
def sample_data():
    today = datetime.utcnow().date()
    rows = []
    for i in range(30):
        d = today - timedelta(days=29 - i)
        amt = (i % 5 + 1) * 100
        if i % 7 == 0:
            amt = -amt
            t = "credit"
        else:
            t = "debit"
        rows.append({"timestamp": pd.to_datetime(d), "description": f"Sample txn {i+1}", "Amount": amt, "Type": t})
    return pd.DataFrame(rows)

# ------------------ Load raw data ------------------
uploaded = None
if data_source == "Upload CSV/XLSX":
    uploaded = st.file_uploader("Upload CSV or XLSX (HDFC / Indian Bank / IFTTT sheet)", type=["csv", "xlsx"])

if data_source == "Google Sheet (optional)":
    if not SHEET_ID:
        st.sidebar.info("Enter Google Sheet ID to enable sheet loading.")
        df_raw = pd.DataFrame()
    else:
        with st.spinner("Fetching Google Sheet..."):
            df_raw = load_from_sheet_safe(SHEET_ID, RANGE, CREDS_FILE)
elif data_source == "Upload CSV/XLSX":
    df_raw = load_from_upload(uploaded)
    if df_raw is None or df_raw.empty:
        st.info("No upload provided or file empty — using sample data.")
        df_raw = sample_data()
else:
    df_raw = sample_data()

if df_raw is None or df_raw.empty:
    st.warning("No data loaded — upload a file, provide a Google Sheet ID, or use sample data.")
    st.stop()

# ------------------ Transform data ------------------
with st.spinner("Cleaning and deriving columns (transform.convert_columns_and_derives)..."):
    converted_df = transform.convert_columns_and_derives(df_raw)

with st.spinner("Computing daily totals (transform.compute_daily_totals)..."):
    merged = transform.compute_daily_totals(converted_df)

# ------------------ Sidebar: Date filters ------------------
with st.sidebar:
    st.header("Filters")
    if not merged.empty:
        merged['Date'] = pd.to_datetime(merged['Date']).dt.normalize()
        min_date = merged['Date'].min().date()
        max_date = merged['Date'].max().date()
        sel_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    else:
        sel_range = (None, None)

# ------------------ Apply filters to aggregated plot_df ------------------
plot_df = merged.copy()
if sel_range and sel_range[0] and sel_range[1]:
    plot_df = plot_df[(plot_df['Date'].dt.date >= sel_range[0]) & (plot_df['Date'].dt.date <= sel_range[1])]

plot_df = plot_df.sort_values('Date').reset_index(drop=True)
plot_df['Total_Spent'] = pd.to_numeric(plot_df.get('Total_Spent', 0), errors='coerce').fillna(0.0).astype('float64')
plot_df['Total_Credit'] = pd.to_numeric(plot_df.get('Total_Credit', 0), errors='coerce').fillna(0.0).astype('float64')

# ------------------ Series selection ------------------
series_selected = []
if show_debit:
    series_selected.append('Total_Spent')
if show_credit:
    series_selected.append('Total_Credit')

# ------------------ Render selected chart ------------------
st.subheader(f"Chart: {chart_type}")
selected_date_from_chart = None

try:
    selected_date_from_chart = charts_mod.render_chart(
        plot_df=plot_df,
        converted_df=converted_df,
        chart_type=chart_type,
        series_selected=series_selected,
        top_n=top_n,
        height=420
    )
except Exception as e:
    st.error("charts.render_chart raised an exception.")
    st.exception(e)

# ------------------ Rows view & download ------------------
st.subheader("Rows (matching selection)")
rows_df = converted_df.copy()

# ensure timestamp exists
if 'timestamp' in rows_df.columns:
    rows_df['timestamp'] = pd.to_datetime(rows_df['timestamp'], errors='coerce')
else:
    if 'date' in rows_df.columns:
        rows_df['timestamp'] = pd.to_datetime(rows_df['date'], errors='coerce')
    else:
        rows_df['timestamp'] = pd.NaT

# apply date-range filter to rows
if sel_range and sel_range[0] and sel_range[1]:
    rows_df = rows_df[(pd.to_datetime(rows_df['timestamp']).dt.date >= sel_range[0]) &
                      (pd.to_datetime(rows_df['timestamp']).dt.date <= sel_range[1])]

# apply chart-selected date if any (placeholder; charts return None for now)
if selected_date_from_chart is not None:
    try:
        rows_df = rows_df[rows_df['timestamp'].dt.date == pd.to_datetime(selected_date_from_chart).date()]
    except Exception:
        pass

if rows_df.empty:
    st.write("No rows to show for the current selection.")
else:
    st.dataframe(rows_df.reset_index(drop=True))
    csv_bytes = rows_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download rows (CSV)", csv_bytes, file_name="transactions_rows.csv", mime="text/csv")

# ------------------ Footer / notes ------------------
st.markdown("""
---
**Notes:**  
- Charts available: Daily line, Monthly bars, Top categories (Top-N).  
- Top-N uses the first available of `Category` / `Merchant` / `description` in your transaction data; rename your column to `Category` for best results.  
- Later we will add more chart types, click-to-select, and Plotly integration.
""")
