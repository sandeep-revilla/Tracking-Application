# streamlit_app.py (enhanced)
import streamlit as st
import pandas as pd
import json
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from typing import Optional

st.set_page_config(page_title="Spending Dashboard — Sheet Loader", layout="wide")
st.title("🔐 Private Google Sheet — Spending Dashboard (enhanced)")

# ------------------ Sidebar: source & controls ------------------
source = st.sidebar.selectbox("Data Source", ("Google Sheet (service account)", "Local Excel"))
SHEET_ID = st.sidebar.text_input("Google Sheet ID (between /d/ and /edit)", value="1KZq_GLXdMBfQUhtp-NA8Jg-flxOppw7kFuIN6y_nOXk")
RANGE = st.sidebar.text_input("Range (optional, e.g. 'History Transactions' or 'History Transactions!A1:Z1000')", value="History Transactions")
LOCAL_PATH = st.sidebar.text_input("Local Excel path (used if Local Excel selected)", value="/mnt/data/SMS received (2).xlsx")
if st.sidebar.button("Refresh now"):
    st.experimental_rerun()

# ------------------ Helper: parse service account JSON from st.secrets ------------------
def load_service_account_secret():
    if "gcp_service_account" not in st.secrets:
        raise KeyError("gcp_service_account not found in st.secrets.")
    raw = st.secrets["gcp_service_account"]
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    if (s.startswith('"""') and s.endswith('"""')) or (s.startswith("'''") and s.endswith("'''")):
        s = s[3:-3].strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(s.replace('\\n','\n'))
        except Exception:
            return json.loads(s.replace('\n','\\n'))

# ------------------ Sheets client builder (cached) ------------------
@st.cache_data(ttl=300)
def build_sheets_service():
    creds_json = load_service_account_secret()
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = service_account.Credentials.from_service_account_info(creds_json, scopes=scopes)
    service = build("sheets", "v4", credentials=creds, cache_discovery=False)
    return service

# ------------------ Read functions ------------------
@st.cache_data(ttl=300)
def read_google_sheet(spreadsheet_id: str, range_name: str) -> pd.DataFrame:
    service = build_sheets_service()
    sheet = service.spreadsheets()
    res = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    values = res.get("values", [])
    if not values:
        return pd.DataFrame()
    header = values[0]
    rows = values[1:]
    return pd.DataFrame(rows, columns=header)

@st.cache_data(ttl=300)
def read_local_excel(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Local file not found: {path}")
    if sheet_name:
        try:
            return pd.read_excel(path, sheet_name=sheet_name)
        except Exception:
            # fallback: try reading first sheet
            return pd.read_excel(path)
    return pd.read_excel(path)

# ------------------ Import cleaning & chart modules (safe) ------------------
# Cleaning
try:
    from src.cleaning import clean_history_transactions
except Exception as e:
    clean_history_transactions = None
    st.sidebar.warning(f"cleaning module not found: {e}")

# Charts: try to import all chart modules; if missing, create placeholders
chart_modules = {}
chart_names = {
    "Monthly Trend": ("src.charts.monthly_trend", "render"),
    "Spending by Type": ("src.charts.spending_by_type", "render"),
    "Spending by Bank": ("src.charts.spending_by_bank", "render"),
    "Top Receivers": ("src.charts.top_senders", "render"),
    "Daily Spending": ("src.charts.daily_pattern", "render"),
    "Suspicious Overview": ("src.charts.suspicious_overview", "render"),
    "Credit vs Debit": ("src.charts.credit_vs_debit", "render"),
    "Hourly Pattern": ("src.charts.hourly_pattern", "render"),
}
import importlib
for friendly, (module_path, fn_name) in chart_names.items():
    try:
        m = importlib.import_module(module_path)
        fn = getattr(m, fn_name)
        chart_modules[friendly] = fn
    except Exception as e:
        # fallback stub that shows message
        def _stub(df, container=None, _msg=f"Chart module {module_path} missing: {e}"):
            if container:
                container.error(_msg)
            return None
        chart_modules[friendly] = _stub
        st.sidebar.info(f"Chart module load: {module_path} -> {e}")

# ------------------ Load data ------------------
df_raw = pd.DataFrame()
try:
    if source.startswith("Google"):
        if not SHEET_ID:
            st.info("Enter the Google Sheet ID in the sidebar.")
            st.stop()
        # Range can be sheet name or A1 range; allow user-friendly input
        df_raw = read_google_sheet(SHEET_ID, RANGE)
    else:
        # if user supplied "SheetName!A1:Z" style, split
        sheet_name = None
        if "!" in RANGE:
            sheet_name = RANGE.split("!")[0]
        df_raw = read_local_excel(LOCAL_PATH, sheet_name=sheet_name)
except FileNotFoundError as e:
    st.error(str(e)); st.stop()
except KeyError as e:
    st.error(str(e)); st.stop()
except Exception as e:
    st.error(f"Data load error: {e}"); st.stop()

if df_raw.empty:
    st.info("No data found. Check range/sheet and permissions.")
    st.stop()

st.subheader("Loaded data (preview)")
st.dataframe(df_raw.head(50), use_container_width=True)
st.caption(f"Rows loaded: {len(df_raw)}")

# ------------------ Clean data (cached) ------------------
if clean_history_transactions is None:
    st.error("Cleaning module not available. Please add src/cleaning.py and restart.")
    st.stop()

@st.cache_data(ttl=300)
def _clean(df_in: pd.DataFrame) -> pd.DataFrame:
    return clean_history_transactions(df_in)

try:
    df = _clean(df_raw)
except Exception as e:
    st.error(f"Cleaning failed: {e}")
    st.stop()

# Basic KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", len(df))
col2.metric("Total Spent", f"{df.loc[df['Type'].str.lower().str.contains('debit', na=False), 'Amount'].sum():,.2f}")
col3.metric("Total Received", f"{df.loc[df['Type'].str.lower().str.contains('credit', na=False), 'Amount'].sum():,.2f}")
col4.metric("Suspicious", int(df['Suspicious'].sum()))

# ------------------ Filters ------------------
st.sidebar.markdown("### Filters")
banks = sorted(df['Bank'].cat.categories.tolist()) if 'Bank' in df.columns and pd.api.types.is_categorical_dtype(df['Bank']) else sorted(df['Bank'].dropna().unique().tolist())
bank_sel = st.sidebar.multiselect("Bank", options=banks, default=None)
date_min = df['DateTime'].min() if 'DateTime' in df.columns else None
date_max = df['DateTime'].max() if 'DateTime' in df.columns else None
if pd.notna(date_min) and pd.notna(date_max):
    date_range = st.sidebar.date_input("Date range", value=(date_min.date(), date_max.date()), min_value=date_min.date(), max_value=date_max.date())
else:
    date_range = None
susp_only = st.sidebar.checkbox("Only suspicious", value=False)

filtered = df.copy()
if bank_sel:
    filtered = filtered[filtered['Bank'].isin(bank_sel)]
if date_range:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    filtered = filtered[(filtered['DateTime'] >= start) & (filtered['DateTime'] <= end)]
if susp_only:
    filtered = filtered[filtered['Suspicious'] == True]

# ------------------ Chart selector & render ------------------
st.sidebar.markdown("### Charts")
chart_choice = st.sidebar.selectbox("Choose chart", list(chart_modules.keys()))
container = st.container()

# render selected chart
render_fn = chart_modules.get(chart_choice)
if render_fn is None:
    container.error("Selected chart is unavailable.")
else:
    try:
        # many chart renderers accept (df, container)
        out = render_fn(filtered, container)
        # if a function returns a Plotly fig, display it
        if out is not None and hasattr(out, "to_html") == False:
            # assume chart already rendered in container
            pass
    except Exception as e:
        container.error(f"Chart rendering failed: {e}")

# ------------------ Data table & download ------------------
if st.sidebar.checkbox("Show data table"):
    st.write(filtered.head(500))

csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=csv, file_name="history_transactions_filtered.csv", mime="text/csv")
