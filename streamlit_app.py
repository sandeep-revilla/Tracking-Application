# streamlit_app.py - main Streamlit entrypoint (imports transform.py, io_helpers.py, charts.py)
import streamlit as st
import pandas as pd
import importlib
from datetime import datetime, timedelta

st.set_page_config(page_title="Daily Spend (with charts module)", layout="wide")
st.title("💳 Daily Spending — modular (charts.py: line chart)")

# ------------------ Imports / placeholders ------------------
# transform.py should already exist (pure data cleaning + aggregation)
try:
    transform = importlib.import_module("transform")
except Exception as e:
    st.error("transform.py missing or failing to import. Add transform.py to the same directory.")
    st.exception(e)
    st.stop()

# local IO helpers (optional - if present we can read Google Sheets)
try:
    import io_helpers as io_mod
except Exception:
    io_mod = None

# charts module (we implement this now). If missing the app will show a helpful message.
try:
    charts_mod = importlib.import_module("charts")
except Exception as e:
    charts_mod = None
    st.warning("charts.py not found or failed to import. Add charts.py to render charts.")
    st.exception(e)

# Future placeholders:
# - Later we will add `plaid_integration` module for bank APIs
# - Later we will add `utils`, `db`, and `auth` modules as needed

# ------------------ Sidebar: data source & options ------------------
with st.sidebar:
    st.header("Data input & options")
    data_source = st.radio("Load data from", ["Upload CSV/XLSX", "Google Sheet (optional)", "Use sample data"], index=0)

    SHEET_ID = st.text_input("Google Sheet ID (between /d/ and /edit)", value="")
    RANGE = st.text_input("Range or Sheet Name", value="History Transactions")
    CREDS_FILE = st.text_input("Service Account JSON File (optional)", value="creds/service_account.json")

    st.markdown("---")
    st.write("Series to include")
    show_debit = st.checkbox("Debit (Total_Spent)", value=True)
    show_credit = st.checkbox("Credit (Total_Credit)", value=True)

    st.markdown("---")
    # Chart type selector included for UX continuity; only "Daily line" is implemented now.
    chart_type = st.selectbox("Chart type", ["Daily line"], index=0)

    st.markdown("---")
    if st.button("Refresh"):
        st.experimental_rerun()

# ------------------ Data loaders (wrappers) ------------------
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
    """
    Safe wrapper around io_helpers.read_google_sheet.
    If io_helpers is not present or reading fails, returns empty DataFrame.
    """
    if io_mod is None:
        st.error("io_helpers.py not found. Add io_helpers.py to enable Google Sheets loading.")
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

# ------------------ Transform data (clean + aggregate) ------------------
with st.spinner("Cleaning and deriving columns (transform.convert_columns_and_derives)..."):
    converted_df = transform.convert_columns_and_derives(df_raw)

with st.spinner("Computing daily totals (transform.compute_daily_totals)..."):
    merged = transform.compute_daily_totals(converted_df)

# ------------------ Sidebar filters (date/year/month) ------------------
with st.sidebar:
    st.header("Filters")
    if not merged.empty:
        merged['Date'] = pd.to_datetime(merged['Date']).dt.normalize()
        years = sorted(merged['Date'].dt.year.unique().tolist())
        years_opts = ['All'] + [str(y) for y in years]
        sel_year = st.selectbox("Year", years_opts, index=0)

        if sel_year == 'All':
            month_frame = merged.copy()
        else:
            month_frame = merged[merged['Date'].dt.year == int(sel_year)]
        month_nums = sorted(month_frame['Date'].dt.month.unique().tolist())
        month_map = {i: pd.Timestamp(1900, i, 1).strftime('%B') for i in range(1, 13)}
        month_choices = [month_map[m] for m in month_nums]
        sel_months = st.multiselect("Month(s)", options=month_choices, default=month_choices)

        # date-range slider
        try:
            date_min = merged['Date'].min().date()
            date_max = merged['Date'].max().date()
            dr = st.slider("Select date range", min_value=date_min, max_value=date_max,
                           value=(date_min, date_max), format="YYYY-MM-DD")
            sel_date_range = (pd.to_datetime(dr[0]).date(), pd.to_datetime(dr[1]).date())
        except Exception:
            sel_date_range = (None, None)
    else:
        sel_year = 'All'
        sel_months = []
        sel_date_range = (None, None)

# ------------------ Apply filters to aggregated plot_df ------------------
plot_df = merged.copy()
if sel_year != 'All':
    plot_df = plot_df[plot_df['Date'].dt.year == int(sel_year)]

if sel_months:
    month_map = {i: pd.Timestamp(1900, i, 1).strftime('%B') for i in range(1, 13)}
    inv_map = {v: k for k, v in month_map.items()}
    selected_month_nums = [inv_map[m] for m in sel_months if m in inv_map]
    if selected_month_nums:
        plot_df = plot_df[plot_df['Date'].dt.month.isin(selected_month_nums)]

if sel_date_range and sel_date_range[0] and sel_date_range[1]:
    plot_df = plot_df[(plot_df['Date'].dt.date >= sel_date_range[0]) & (plot_df['Date'].dt.date <= sel_date_range[1])]

plot_df = plot_df.sort_values('Date').reset_index(drop=True)
plot_df['Total_Spent'] = pd.to_numeric(plot_df.get('Total_Spent', 0), errors='coerce').fillna(0.0).astype('float64')
plot_df['Total_Credit'] = pd.to_numeric(plot_df.get('Total_Credit', 0), errors='coerce').fillna(0.0).astype('float64')

# ------------------ Series selection ------------------
series_selected = []
if show_debit:
    series_selected.append('Total_Spent')
if show_credit:
    series_selected.append('Total_Credit')

# ------------------ Render chart (delegated to charts.py) ------------------
st.subheader("Daily Spend and Credit (line chart)")
selected_date_from_chart = None

if charts_mod is None:
    st.error("charts.py is not available — add charts.py to render charts (see placeholders).")
else:
    # charts_mod.render_chart is expected to render and optionally return a selected date (click)
    try:
        selected_date_from_chart = charts_mod.render_chart(
            plot_df=plot_df,
            converted_df=converted_df,
            chart_type=chart_type,
            series_selected=series_selected,
            enable_plotly_click=False  # placeholder; click-mode will be added later
        )
    except Exception as e:
        st.error("charts.render_chart raised an exception.")
        st.exception(e)

# ------------------ Rows view & download ------------------
st.subheader("Rows (matching selection)")
rows_df = converted_df.copy()

# ensure timestamp exists and filter rows according to sidebar filters + chart selection if any
if 'timestamp' in rows_df.columns:
    rows_df['timestamp'] = pd.to_datetime(rows_df['timestamp'], errors='coerce')
else:
    if 'date' in rows_df.columns:
        rows_df['timestamp'] = pd.to_datetime(rows_df['date'], errors='coerce')
    else:
        rows_df['timestamp'] = pd.NaT

# apply year/month/date-range filters to rows
if sel_year != 'All':
    try:
        rows_df = rows_df[rows_df['timestamp'].dt.year == int(sel_year)]
    except Exception:
        pass

if sel_months:
    month_map = {i: pd.Timestamp(1900, i, 1).strftime('%B') for i in range(1, 13)}
    inv_map = {v: k for k, v in month_map.items()}
    selected_month_nums = [inv_map[m] for m in sel_months if m in inv_map]
    if selected_month_nums:
        rows_df = rows_df[rows_df['timestamp'].dt.month.isin(selected_month_nums)]

if sel_date_range and sel_date_range[0] and sel_date_range[1]:
    rows_df = rows_df[(rows_df['timestamp'].dt.date >= sel_date_range[0]) & (rows_df['timestamp'].dt.date <= sel_date_range[1])]

# apply chart-selected date if charts.py returns one in future (placeholder)
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
- `transform.py` must be present (handles cleaning/aggregation).  
- `io_helpers.py` is optional (Google Sheets).  
- `charts.py` currently implements a line chart via `render_chart()` — we will extend it later with more chart types and click-to-select functionality.
""")
