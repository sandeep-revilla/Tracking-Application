# streamlit_app.py - main Streamlit entrypoint (imports transform.py and io.py)
import streamlit as st
import pandas as pd
import altair as alt
import importlib
from datetime import datetime, timedelta

st.set_page_config(page_title="Daily Spend (with io + transform)", layout="wide")
st.title("💳 Daily Spending — io + transform integrated (starter)")

# ---------------- PLACEHOLDERS / FUTURE IMPORTS ----------------
# We already have transform.py implemented (pure transform functions).
# We'll import io.py (data loading helpers) now.
# Placeholder for charts.py (visualization helpers) remains here for future:
#   charts_mod = importlib.import_module("charts")    # <- implement later
try:
    io_mod = importlib.import_module("io")
except Exception as e:
    io_mod = None  # io functions will only be used if available; app can still use upload/sample fallback

try:
    transform = importlib.import_module("transform")
except Exception as e:
    st.error("transform.py missing or failing to import. Add transform.py to the same directory.")
    st.exception(e)
    st.stop()

# ---------------- Sidebar: data source & options ----------------
with st.sidebar:
    st.header("Data input & options")
    data_source = st.radio("Load data from", ["Upload CSV/XLSX", "Google Sheet (optional)", "Use sample data"], index=0)

    # Google Sheet inputs (only used if Google Sheet chosen)
    SHEET_ID = st.text_input("Google Sheet ID (between /d/ and /edit)", value="")
    RANGE = st.text_input("Range or Sheet Name", value="History Transactions")
    CREDS_FILE = st.text_input("Service Account JSON File (optional)", value="creds/service_account.json")

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

# ---------------- Data loaders (safe wrappers) ----------------
def load_from_upload(uploaded_file) -> pd.DataFrame:
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

def load_from_sheet_safe(sheet_id: str, range_name: str, creds_file: str) -> pd.DataFrame:
    """
    Wrapper that calls io_mod.read_google_sheet if io_mod is present.
    It does not use st.secrets inside io_mod directly; pass st.secrets here.
    """
    if io_mod is None:
        st.error("io.py not available. Install or add io.py to the project to use Google Sheets.")
        return pd.DataFrame()
    try:
        secrets = st.secrets if hasattr(st, "secrets") else None
        return io_mod.read_google_sheet(sheet_id, range_name, creds_info=None, creds_file=creds_file, secrets=secrets)
    except Exception as e:
        st.error(f"Failed to read Google Sheet: {e}")
        return pd.DataFrame()

# ---------------- Sample data (fallback) ----------------
def sample_data():
    today = datetime.utcnow().date()
    rows = []
    for i in range(30):
        d = today - timedelta(days=29-i)
        amt = (i % 5 + 1) * 100
        if i % 7 == 0:
            amt = -amt
            t = "credit"
        else:
            t = "debit"
        rows.append({"timestamp": pd.to_datetime(d), "description": f"Sample txn {i+1}", "Amount": amt, "Type": t})
    return pd.DataFrame(rows)

# ---------------- Load raw data according to selection ----------------
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

# ---------------- Transform using transform.py ----------------
with st.spinner("Cleaning and deriving columns..."):
    converted_df = transform.convert_columns_and_derives(df_raw)

with st.spinner("Computing daily totals..."):
    merged = transform.compute_daily_totals(converted_df)

# ---------------- Sidebar: Date filters ----------------
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

# ---------------- Apply filters to aggregated plot_df ----------------
plot_df = merged.copy()
if sel_year != 'All':
    plot_df = plot_df[plot_df['Date'].dt.year == int(sel_year)]

if sel_months:
    month_map = {i: pd.Timestamp(1900, i, 1).strftime('%B') for i in range(1, 13)}
    inv_map = {v: k for k, v in month_map.items()}
    selected_month_nums = [inv_map[m] for m in sel_months if m in inv_map]
    if selected_month_nums:
        plot_df = plot_df[plot_df['Date'].dt.month.isin(selected_month_nums)]

# apply date-range slider
if sel_date_range and sel_date_range[0] and sel_date_range[1]:
    plot_df = plot_df[(plot_df['Date'].dt.date >= sel_date_range[0]) & (plot_df['Date'].dt.date <= sel_date_range[1])]

plot_df = plot_df.sort_values('Date').reset_index(drop=True)
plot_df['Total_Spent'] = pd.to_numeric(plot_df.get('Total_Spent', 0), errors='coerce').fillna(0.0).astype('float64')
plot_df['Total_Credit'] = pd.to_numeric(plot_df.get('Total_Credit', 0), errors='coerce').fillna(0.0).astype('float64')

# ---------------- Series selection ----------------
series_selected = []
if show_debit:
    series_selected.append('Total_Spent')
if show_credit:
    series_selected.append('Total_Credit')

# ---------------- Chart (simple, inline for now; later calls to charts.py) ----------------
st.subheader("Daily Spend and Credit (aggregated)")
if plot_df.empty:
    st.info("No data to display for the selected filters.")
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

# ---------------- Rows view & download ----------------
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

# apply filters to rows
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
- `io.py` provides Google Sheets read functionality; if not present the app still works using upload/sample data.  
- `transform.py` is used to clean and aggregate; keep it in the repo.  
- `charts.py` is a planned module (placeholder) — we'll move plotting into it in the next step.
""")
