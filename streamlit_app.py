# streamlit_app.py - main Streamlit entrypoint (orchestrates io.py, transform.py, charts.py)
import streamlit as st
import pandas as pd
import importlib

st.set_page_config(page_title="Sheet → Daily Spend (modular)", layout="wide")
st.title("💳 Daily Spending — Modular Version (streamlit_app.py)")

# Try to import helper modules that should live in the same directory
try:
    io_mod = importlib.import_module("io")
    tf = importlib.import_module("transform")
    charts_mod = importlib.import_module("charts")
except Exception as e:
    st.error("Required helper modules not found. Make sure io.py, transform.py and charts.py "
             "exist in the same directory as streamlit_app.py.")
    st.exception(e)
    st.stop()

# ---------------- Sidebar: data source & options ----------------
with st.sidebar:
    st.header("Data source & filters")

    data_source = st.radio("Data source", ["Upload CSV/XLSX", "Google Sheet"], index=0)

    # Google Sheet inputs (used only if user selects that source)
    SHEET_ID = st.text_input("Google Sheet ID (between /d/ and /edit)", value="")
    RANGE = st.text_input("Range or Sheet Name", value="History Transactions")
    CREDS_FILE = st.text_input("Service Account JSON File (optional)", value="creds/service_account.json")

    st.markdown("---")
    st.subheader("Chart options")
    enable_plotly_click = st.checkbox("Enable click-to-select (Plotly)", value=False)
    st.markdown("---")

    st.write("Series to include")
    show_debit = st.checkbox("Debit (Total_Spent)", value=True)
    show_credit = st.checkbox("Credit (Total_Credit)", value=True)

    st.markdown("---")
    st.write("Chart type")
    chart_type = st.selectbox("Chart type", [
        "Daily line", "Stacked area", "Monthly bars", "Rolling average",
        "Cumulative sum", "Calendar heatmap", "Histogram of amounts", "Treemap by category"
    ])

    st.markdown("---")
    st.write("Data input (if Upload selected)")
    uploaded = None
    if data_source == "Upload CSV/XLSX":
        uploaded = st.file_uploader("Upload CSV or XLSX (HDFC / Indian Bank)", type=["csv", "xlsx"],
                                    help="File processed in-memory; not stored.")

    st.markdown("---")
    if st.button("Refresh data"):
        st.experimental_rerun()

# ---------------- Data loading helpers (cached) ----------------
@st.cache_data(ttl=300)
def load_data_from_sheet(sheet_id: str, range_name: str, creds_file: str, secrets: dict):
    try:
        df = io_mod.read_google_sheet(sheet_id, range_name, creds_info=None, creds_file=creds_file) \
            if (secrets is None or "gcp_service_account" not in secrets) else io_mod.read_google_sheet(sheet_id, range_name, creds_info=secrets.get("gcp_service_account"), creds_file=None)
        return df
    except Exception as e:
        st.error(f"Failed to read Google Sheet: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_data_from_upload(uploaded_file):
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        return df
    except Exception as e:
        st.error(f"Failed to parse uploaded file: {e}")
        return pd.DataFrame()

# ---------------- Load data (either uploaded or from sheet) ----------------
if data_source == "Google Sheet":
    if not SHEET_ID:
        st.sidebar.error("Enter Google Sheet ID in the sidebar.")
        st.stop()
    with st.spinner("Fetching Google Sheet..."):
        secrets = st.secrets if hasattr(st, "secrets") else None
        df_raw = load_data_from_sheet(SHEET_ID, RANGE, CREDS_FILE, secrets)
else:
    df_raw = load_data_from_upload(uploaded)

if df_raw is None or df_raw.empty:
    st.warning("No data loaded — upload a CSV/XLSX or provide a valid Google Sheet ID.")
    st.stop()

# ---------------- Transform data ----------------
with st.spinner("Cleaning and deriving columns..."):
    converted_df = tf.convert_columns_and_derives(df_raw)
    merged = tf.compute_daily_totals(converted_df)

# ---------------- Sidebar: Year/Month filters ----------------
with st.sidebar:
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

        # optional date-range slider (derive min/max)
        try:
            date_min = merged['Date'].min().date()
            date_max = merged['Date'].max().date()
            st.write("Date range")
            dr = st.slider("Select date range", min_value=date_min, max_value=date_max,
                           value=(date_min, date_max), format="YYYY-MM-DD")
            sel_date_range = (pd.to_datetime(dr[0]).date(), pd.to_datetime(dr[1]).date())
        except Exception:
            sel_date_range = (None, None)
    else:
        sel_year = 'All'
        sel_months = []
        sel_date_range = (None, None)

# ---------------- Apply filters to merged and prepare plot_df ----------------
plot_df = merged.copy()
if sel_year != 'All':
    plot_df = plot_df[plot_df['Date'].dt.year == int(sel_year)]

if sel_months:
    inv_map = {v: k for k, v in {i: pd.Timestamp(1900, i, 1).strftime('%B'): i for i in range(1,13)}.items()}
    selected_month_nums = [inv_map[m] for m in sel_months if m in inv_map]
    if selected_month_nums:
        plot_df = plot_df[plot_df['Date'].dt.month.isin(selected_month_nums)]

# apply date-range slider
if sel_date_range and sel_date_range[0] and sel_date_range[1]:
    plot_df = plot_df[(plot_df['Date'].dt.date >= sel_date_range[0]) & (plot_df['Date'].dt.date <= sel_date_range[1])]

plot_df = plot_df.sort_values('Date').reset_index(drop=True)
plot_df['Total_Spent'] = pd.to_numeric(plot_df.get('Total_Spent', 0), errors='coerce').fillna(0.0).astype('float64')
plot_df['Total_Credit'] = pd.to_numeric(plot_df.get('Total_Credit', 0), errors='coerce').fillna(0.0).astype('float64')

# ---------------- Series selection (from sidebar checkboxes) ----------------
series_selected = []
if show_debit:
    series_selected.append('Total_Spent')
if show_credit:
    series_selected.append('Total_Credit')

# ---------------- Render chart ----------------
st.subheader("Daily Spend and Credit")
charts_mod.render_chart(plot_df, converted_df, chart_type, series_selected, enable_plotly_click)

# ---------------- Show filtered rows and allow download ----------------
st.subheader("Rows (matching selection)")

rows_df = converted_df.copy()

# ensure timestamp exists on rows_df
if 'timestamp' in rows_df.columns:
    rows_df['timestamp'] = pd.to_datetime(rows_df['timestamp'], errors='coerce')
else:
    if 'date' in rows_df.columns:
        rows_df['timestamp'] = pd.to_datetime(rows_df['date'], errors='coerce')
    else:
        rows_df['timestamp'] = pd.NaT

# apply year/month filters to rows
if sel_year != 'All':
    try:
        rows_df = rows_df[rows_df['timestamp'].dt.year == int(sel_year)]
    except Exception:
        pass

if sel_months:
    inv_map = {v: k for k, v in {i: pd.Timestamp(1900, i, 1).strftime('%B'): i for i in range(1,13)}.items()}
    selected_month_nums = [inv_map[m] for m in sel_months if m in inv_map]
    if selected_month_nums:
        rows_df = rows_df[rows_df['timestamp'].dt.month.isin(selected_month_nums)]

# apply date range filter
if sel_date_range and sel_date_range[0] and sel_date_range[1]:
    rows_df = rows_df[(rows_df['timestamp'].dt.date >= sel_date_range[0]) & (rows_df['timestamp'].dt.date <= sel_date_range[1])]

if rows_df.empty:
    st.write("No rows match the current filters/selection.")
else:
    st.dataframe(rows_df.reset_index(drop=True))
    # CSV download
    csv_bytes = rows_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download matching rows (CSV)", csv_bytes, file_name="transactions_filtered.csv", mime="text/csv")
