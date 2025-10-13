# streamlit_app.py - main Streamlit entrypoint (with safe Google Sheet handling + bank filter)
import streamlit as st
import pandas as pd
import importlib
from datetime import datetime, timedelta

st.set_page_config(page_title="Daily Spend (safe sheets)", layout="wide")
st.title("💳 Daily Spending — safe Google Sheets handling")

# ------------------ Imports / placeholders ------------------
# transform.py must exist and provide convert_columns_and_derives + compute_daily_totals
try:
    transform = importlib.import_module("transform")
except Exception as e:
    st.error("transform.py missing or failing to import. Add transform.py to the same directory.")
    st.exception(e)
    st.stop()

# optional I/O helper
try:
    import io_helpers as io_mod
except Exception:
    io_mod = None

# charts module
try:
    charts_mod = importlib.import_module("charts")
except Exception:
    charts_mod = None

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
    st.write("Chart type (placeholder)")
    chart_type = st.selectbox("Chart type", ["Daily line", "Monthly bars", "Top categories (Top-N)"], index=0)

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
    """
    Safe wrapper that:
      - reads st.secrets["gcp_service_account"] (if present),
      - parses it into a plain dict (using io_helpers.parse_service_account_secret),
      - passes that plain dict to io_helpers.read_google_sheet (a cached function).
    This avoids passing the Streamlit runtime 'st.secrets' object into a cached function.
    """
    if io_mod is None:
        st.error("io_helpers.py not available. Add io_helpers.py to the project to use Google Sheets.")
        return pd.DataFrame()

    creds_info = None
    # Parse st.secrets into a plain dict if present
    try:
        if hasattr(st, "secrets") and st.secrets and "gcp_service_account" in st.secrets:
            raw = st.secrets["gcp_service_account"]
            # parse_service_account_secret returns a plain dict
            creds_info = io_mod.parse_service_account_secret(raw)
    except Exception as e:
        st.warning(f"Failed to parse st.secrets['gcp_service_account']: {e}")
        creds_info = None

    # If creds_info is None, but a local creds_file exists, io_helpers will use that.
    try:
        return io_mod.read_google_sheet(sheet_id, range_name, creds_info=creds_info, creds_file=creds_file)
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


# ------------------ Helper: bank detection & filtering ------------------
def add_bank_column(df: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    """
    Ensure df has a 'Bank' column. If missing (or overwrite=True), try to detect bank name
    from common text fields: 'bank', 'account', 'description', 'message', 'narration', 'beneficiary', 'merchant'.
    Returns a new DataFrame with 'Bank' column (string), unknowns marked 'Unknown'.
    """
    df = df.copy()
    if 'Bank' in df.columns and not overwrite:
        # normalize existing Bank column values to strings and fillna
        df['Bank'] = df['Bank'].astype(str).where(df['Bank'].notna(), None)
        df['Bank'] = df['Bank'].fillna('Unknown')
        return df

    # candidate text columns to scan
    cand_cols = ['bank', 'account', 'account_name', 'description', 'message', 'narration', 'merchant', 'beneficiary', 'note']
    # build a combined lowercased text for each row
    def _row_text(row):
        parts = []
        for c in cand_cols:
            if c in row.index and pd.notna(row[c]):
                parts.append(str(row[c]))
        return " ".join(parts).lower()

    # mapping patterns -> normalized bank name (extend as needed)
    bank_map = {
        'hdfc': 'HDFC Bank',
        'hdfc bank': 'HDFC Bank',
        'hdfcbank': 'HDFC Bank',
        'hdfc card': 'HDFC Bank',
        'hdfccredit': 'HDFC Bank',
        'indian bank': 'Indian Bank',
        'indianbank': 'Indian Bank',
        'indian bank ltd': 'Indian Bank',
        # add more heuristics if needed
    }

    # compute combined text column (vectorized using apply)
    try:
        combined = df.apply(_row_text, axis=1)
    except Exception:
        # fallback: if apply fails (weird dtypes), create empty series
        combined = pd.Series([''] * len(df), index=df.index)

    detected = []
    for text in combined:
        found = None
        for patt, name in bank_map.items():
            if patt in text:
                found = name
                break
        detected.append(found if found is not None else None)

    df['Bank'] = detected
    df['Bank'] = df['Bank'].fillna('Unknown')
    return df


# ------------------ Load raw data according to selection ----------------
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
else:  # sample data
    df_raw = sample_data()

if df_raw is None or df_raw.empty:
    st.warning("No data loaded — upload a file or provide a Google Sheet ID, or use sample data.")
    st.stop()

# ------------------ Transform using transform.py ----------------
with st.spinner("Cleaning and deriving columns..."):
    converted_df = transform.convert_columns_and_derives(df_raw)

# ensure bank column exists (detect heuristically if needed)
converted_df = add_bank_column(converted_df, overwrite=False)

# ------------------ Sidebar: Bank filter ------------------
with st.sidebar:
    st.markdown("---")
    st.write("Filter by Bank")
    # detect unique banks and sort; prefer to show HDFC/Indian selected if present
    banks_detected = sorted([b for b in converted_df['Bank'].unique() if pd.notna(b)])
    # default selection logic: if HDFC/Indian present select them, else select all
    defaults = []
    if 'HDFC Bank' in banks_detected:
        defaults.append('HDFC Bank')
    if 'Indian Bank' in banks_detected:
        defaults.append('Indian Bank')
    if not defaults:
        defaults = banks_detected  # select all by default if HDFC/Indian not found
    sel_banks = st.multiselect("Banks", options=banks_detected, default=defaults)

# filter converted_df according to selection
if sel_banks:
    converted_df_filtered = converted_df[converted_df['Bank'].isin(sel_banks)].copy()
else:
    converted_df_filtered = converted_df.copy()

# ------------------ Compute daily totals (from filtered transactions) ----------------
with st.spinner("Computing daily totals..."):
    merged = transform.compute_daily_totals(converted_df_filtered)

# ------------------ Sidebar: Date filters ------------------
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

plot_df = plot_df.sort_values('Date').reset_index(drop=True)
plot_df['Total_Spent'] = pd.to_numeric(plot_df.get('Total_Spent', 0), errors='coerce').fillna(0.0).astype('float64')
plot_df['Total_Credit'] = pd.to_numeric(plot_df.get('Total_Credit', 0), errors='coerce').fillna(0.0).astype('float64')

# ------------------ Chart & rendering ------------------
st.subheader("Daily Spend and Credit")
if plot_df.empty:
    st.info("No data for the selected filters.")
else:
    # delegate to charts.py if present
    if charts_mod is not None:
        series_selected = []
        if show_debit: series_selected.append('Total_Spent')
        if show_credit: series_selected.append('Total_Credit')
        charts_mod.render_chart(plot_df=plot_df, converted_df=converted_df_filtered, chart_type=chart_type, series_selected=series_selected, top_n=5)
    else:
        st.info("charts.py not available; install or add charts.py for visualizations.")

# ------------------ Rows view & download ----------------
st.subheader("Rows (matching selection)")
rows_df = converted_df_filtered.copy()  # use the filtered transactions so rows match chart

# ensure timestamp exists
if 'timestamp' in rows_df.columns:
    rows_df['timestamp'] = pd.to_datetime(rows_df['timestamp'], errors='coerce')
else:
    if 'date' in rows_df.columns:
        rows_df['timestamp'] = pd.to_datetime(rows_df['date'], errors='coerce')
    else:
        rows_df['timestamp'] = pd.NaT

# apply date-range filter to rows
if sel_date_range and sel_date_range[0] and sel_date_range[1]:
    rows_df = rows_df[(rows_df['timestamp'].dt.date >= sel_date_range[0]) & (rows_df['timestamp'].dt.date <= sel_date_range[1])]

if rows_df.empty:
    st.write("No rows match the current filters/selection.")
else:
    st.dataframe(rows_df.reset_index(drop=True))
    csv_bytes = rows_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download rows (CSV)", csv_bytes, file_name="transactions_rows.csv", mime="text/csv")

# ------------------ Footer / notes ----------------
st.markdown("""
---
