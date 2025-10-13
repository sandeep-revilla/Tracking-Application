# streamlit_app.py - main Streamlit entrypoint (with caching + session state)
import streamlit as st
import pandas as pd
import importlib
from datetime import datetime, timedelta
import json

st.set_page_config(page_title="Daily Spend (with caching + session state)", layout="wide")
st.title("💳 Daily Spending — caching + session state")

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

# charts module (we implemented earlier)
try:
    charts_mod = importlib.import_module("charts")
except Exception as e:
    charts_mod = None
    st.error("charts.py missing or failed to import. Add charts.py to the project.")
    st.exception(e)
    st.stop()

# ------------------ Session state helpers ------------------
def _init_session_state():
    ss = st.session_state
    if 'chart_type' not in ss:
        ss.chart_type = "Daily line"
    if 'show_debit' not in ss:
        ss.show_debit = True
    if 'show_credit' not in ss:
        ss.show_credit = True
    if 'sheet_id' not in ss:
        ss.sheet_id = ""
    if 'range' not in ss:
        ss.range = "History Transactions"
    if 'creds_file' not in ss:
        ss.creds_file = "creds/service_account.json"
    if 'last_refreshed' not in ss:
        ss.last_refreshed = None
    if 'sel_range' not in ss:
        ss.sel_range = (None, None)
    if 'uploaded_bytes' not in ss:
        ss.uploaded_bytes = None
    if 'uploaded_name' not in ss:
        ss.uploaded_name = None

_init_session_state()

# ------------------ Sidebar: data source & chart selector ------------------
with st.sidebar:
    st.header("Data input & options")
    data_source = st.radio("Load data from", ["Upload CSV/XLSX", "Google Sheet (optional)", "Use sample data"], index=0)

    # Bind session to sheet inputs
    st.text_input("Google Sheet ID (between /d/ and /edit)", value=st.session_state.sheet_id, key="sheet_id", on_change=lambda: None)
    st.text_input("Range or Sheet Name", value=st.session_state.range, key="range", on_change=lambda: None)
    st.text_input("Service Account JSON File (optional)", value=st.session_state.creds_file, key="creds_file", on_change=lambda: None)

    st.markdown("---")
    st.write("Pick chart")
    st.selectbox("Chart type", ["Daily line", "Monthly bars", "Top categories (Top-N)"], index=["Daily line","Monthly bars","Top categories (Top-N)"].index(st.session_state.chart_type), key="chart_type")

    st.markdown("---")
    st.write("Series to include (applies to Daily/Monthly charts)")
    st.checkbox("Debit (Total_Spent)", value=st.session_state.show_debit, key="show_debit")
    st.checkbox("Credit (Total_Credit)", value=st.session_state.show_credit, key="show_credit")

    st.markdown("---")
    st.write("Cache / refresh controls")
    # Display last refreshed
    last = st.session_state.last_refreshed
    if last is None:
        st.info("Data not loaded yet (no refresh).")
    else:
        st.write(f"Last refreshed: {last}")

    if st.button("Force refresh (clear cache)"):
        # Clear Streamlit data caches so the next load will re-fetch/recompute
        try:
            st.cache_data.clear()
            st.session_state.last_refreshed = None
            st.success("Cache cleared. Next load will re-fetch data.")
        except Exception as e:
            st.error(f"Failed to clear cache: {e}")

    st.markdown("---")
    top_n = st.slider("Top N (for categories)", min_value=3, max_value=20, value=5, step=1)

    st.markdown("---")
    if st.button("Refresh UI only"):
        st.experimental_rerun()

# ------------------ Data loaders (safe wrappers) ------------------
def load_from_upload_obj(uploaded_file) -> pd.DataFrame:
    """Load uploaded file and store bytes in session_state to persist across reruns."""
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        b = uploaded_file.read()
        st.session_state.uploaded_bytes = b
        st.session_state.uploaded_name = uploaded_file.name
        if uploaded_file.name.lower().endswith(".csv"):
            return pd.read_csv(pd.io.common.BytesIO(b))
        else:
            return pd.read_excel(pd.io.common.BytesIO(b), engine="openpyxl")
    except Exception as e:
        st.error(f"Failed to parse upload: {e}")
        return pd.DataFrame()

def load_uploaded_from_session() -> pd.DataFrame:
    """If uploaded file bytes exist in session_state, load them into a DataFrame (persist across reruns)."""
    b = st.session_state.uploaded_bytes
    name = st.session_state.uploaded_name
    if not b or not name:
        return pd.DataFrame()
    try:
        if name.lower().endswith(".csv"):
            return pd.read_csv(pd.io.common.BytesIO(b))
        else:
            return pd.read_excel(pd.io.common.BytesIO(b), engine="openpyxl")
    except Exception:
        return pd.DataFrame()

def load_from_sheet_safe(sheet_id: str, range_name: str, creds_file: str) -> pd.DataFrame:
    """
    Wrapper that calls io_helpers.read_google_sheet if available.
    This function is cached inside io_helpers; this wrapper simply calls it.
    """
    if io_mod is None:
        st.error("io_helpers.py not available. Add io_helpers.py to the project to use Google Sheets.")
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

# ------------------ Load raw data (using session_state to persist uploads) ------------------
uploaded = None
if data_source == "Upload CSV/XLSX":
    uploaded = st.file_uploader("Upload CSV or XLSX (HDFC / Indian Bank / IFTTT sheet)", type=["csv", "xlsx"])

# logic: if user uploaded now -> load and persist; else if session has uploaded bytes -> use that
if data_source == "Upload CSV/XLSX":
    if uploaded:
        df_raw = load_from_upload_obj(uploaded)
    else:
        df_raw = load_uploaded_from_session()
        if df_raw is None or df_raw.empty:
            st.info("No upload provided — you can upload a CSV/XLSX or use sample data.")
            df_raw = pd.DataFrame()
elif data_source == "Google Sheet (optional)":
    # prefer session_state values (persisted) for sheet id / range
    sheet_id = st.session_state.sheet_id
    range_name = st.session_state.range
    creds_file = st.session_state.creds_file
    if not sheet_id:
        st.sidebar.info("Enter Google Sheet ID to enable sheet loading.")
        df_raw = pd.DataFrame()
    else:
        with st.spinner("Fetching Google Sheet..."):
            df_raw = load_from_sheet_safe(sheet_id, range_name, creds_file)
            # record last refreshed timestamp when sheet successfully loads
            if df_raw is not None and not df_raw.empty:
                st.session_state.last_refreshed = datetime.utcnow().isoformat()
else:
    df_raw = sample_data()

if df_raw is None or df_raw.empty:
    st.warning("No data loaded — upload a file, provide a Google Sheet ID, or use sample data.")
    st.stop()

# ------------------ Cached transform wrappers ------------------
# We cache by passing JSON-serialized DataFrame (orient='split') as the cache key.
@st.cache_data(ttl=300)
def _cached_convert(df_json: str):
    df = pd.read_json(df_json, orient='split')
    return transform.convert_columns_and_derives(df)

@st.cache_data(ttl=300)
def _cached_daily_totals(converted_json: str):
    df = pd.read_json(converted_json, orient='split')
    return transform.compute_daily_totals(df)

# serialize raw DataFrame to JSON for caching key
raw_json = df_raw.to_json(date_format='iso', orient='split')
with st.spinner("Cleaning and deriving columns (cached)..."):
    converted_df = _cached_convert(raw_json)

# serialize converted df for caching daily totals
conv_json = converted_df.to_json(date_format='iso', orient='split')
with st.spinner("Computing daily totals (cached)..."):
    merged = _cached_daily_totals(conv_json)

# ------------------ Sidebar: Date filters ------------------
with st.sidebar:
    st.header("Filters")
    if not merged.empty:
        merged['Date'] = pd.to_datetime(merged['Date']).dt.normalize()
        min_date = merged['Date'].min().date()
        max_date = merged['Date'].max().date()

        # use session_state to persist date input selection
        cur_range = st.session_state.sel_range
        if cur_range is None or cur_range[0] is None:
            cur_range = (min_date, max_date)
            st.session_state.sel_range = cur_range

        dr = st.date_input("Date range", value=cur_range, min_value=min_date, max_value=max_date, key="sel_range")
        st.session_state.sel_range = dr
    else:
        st.session_state.sel_range = (None, None)

# ------------------ Apply filters to aggregated plot_df ------------------
plot_df = merged.copy()
sel_range = st.session_state.sel_range
if sel_range and sel_range[0] and sel_range[1]:
    plot_df = plot_df[(plot_df['Date'].dt.date >= sel_range[0]) & (plot_df['Date'].dt.date <= sel_range[1])]

plot_df = plot_df.sort_values('Date').reset_index(drop=True)
plot_df['Total_Spent'] = pd.to_numeric(plot_df.get('Total_Spent', 0), errors='coerce').fillna(0.0).astype('float64')
plot_df['Total_Credit'] = pd.to_numeric(plot_df.get('Total_Credit', 0), errors='coerce').fillna(0.0).astype('float64')

# ------------------ Series selection ------------------
series_selected = []
if st.session_state.show_debit:
    series_selected.append('Total_Spent')
if st.session_state.show_credit:
    series_selected.append('Total_Credit')

# ------------------ Render selected chart ------------------
st.subheader(f"Chart: {st.session_state.chart_type}")
selected_date_from_chart = None

try:
    selected_date_from_chart = charts_mod.render_chart(
        plot_df=plot_df,
        converted_df=converted_df,
        chart_type=st.session_state.chart_type,
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
- Caching: Google Sheets reads are cached (default TTL 10 minutes). Transform steps (convert + daily totals) are cached (TTL 5 minutes).  
- Use **Force refresh (clear cache)** in the sidebar to clear caches and force a full reload from sources.  
- Session state preserves your chart selection, series toggles, uploaded file (in-memory), and date range across reruns.
""")
