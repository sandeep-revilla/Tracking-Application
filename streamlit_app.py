# streamlit_app.py
"""
Streamlit helper: connect to a private Google Sheet via service account (st.secrets or JSON file),
safely convert sheet values to a pandas DataFrame (handles uneven rows and missing header),
show preview and provide raw CSV download.

Purpose: CONNECTION ONLY (no cleaning, no charts).
"""

import streamlit as st
import pandas as pd
import json
import os
from typing import List, Tuple, Optional, Any, Dict
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from cleaning import clean_history_transactions
from charts import monthly_trend

st.set_page_config(page_title="Google Sheet Connector", layout="wide")
st.title("🔐 Google Sheet Connector — (Connection Only)")

# ---------------- Sidebar Inputs ----------------
SHEET_ID = st.sidebar.text_input("Google Sheet ID (between /d/ and /edit)", value="1KZq_GLXdMBfQUhtp-NA8Jg-flxOppw7kFuIN6y_nOXk")
RANGE = st.sidebar.text_input("Range or Sheet Name", value="History Transactions")
st.sidebar.caption("Provide your Service Account JSON via st.secrets['gcp_service_account'] or as a local file below.")
CREDS_FILE = st.sidebar.text_input("Service Account JSON File (optional)", value="creds/service_account.json")

if st.sidebar.button("Refresh Now"):
    st.experimental_rerun()

# ---------------- Helper: Parse Service Account JSON ----------------
def parse_service_account_secret(raw: Any) -> Dict:
    """Accepts dict or JSON string (even with escaped newlines), returns parsed dict."""
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    if (s.startswith('"""') and s.endswith('"""')) or (s.startswith("'''") and s.endswith("'''")):
        s = s[3:-3].strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(s.replace('\\n', '\n'))
        except Exception:
            return json.loads(s.replace('\n', '\\n'))

# ---------------- Sheets Client Builders ----------------
@st.cache_data(ttl=300)
def build_sheets_service_from_info(creds_info: Dict):
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)

@st.cache_data(ttl=300)
def build_sheets_service_from_file(creds_file: str):
    if not os.path.exists(creds_file):
        raise FileNotFoundError(f"Credentials file not found: {creds_file}")
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = service_account.Credentials.from_service_account_file(creds_file, scopes=scopes)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)

# ---------------- Safe Data Conversion Helpers ----------------
def _normalize_rows(values: List[List[str]]) -> Tuple[List[str], List[List]]:
    """Ensure header exists, pad/trim rows to header length."""
    if not values:
        return [], []

    header_row = [str(x).strip() for x in values[0]]
    if all((h == "" or h.lower().startswith(("unnamed", "column", "nan"))) for h in header_row):
        max_cols = max(len(r) for r in values)
        header = [f"col_{i}" for i in range(max_cols)]
        data_rows = values
    else:
        header = header_row
        data_rows = values[1:]

    col_count = len(header)
    normalized = []
    for r in data_rows:
        if len(r) < col_count:
            r = r + [None] * (col_count - len(r))
        elif len(r) > col_count:
            r = r[:col_count]
        normalized.append(r)
    return header, normalized

@st.cache_data(ttl=300)
def values_to_dataframe(values: List[List[str]]) -> pd.DataFrame:
    """Convert sheet values to pandas DataFrame safely."""
    if not values:
        return pd.DataFrame()
    header, rows = _normalize_rows(values)
    try:
        return pd.DataFrame(rows, columns=header)
    except Exception:
        df = pd.DataFrame(rows)
        if header and df.shape[1] == len(header):
            df.columns = header
        return df

# ---------------- Google Sheet Reader ----------------
@st.cache_data(ttl=300)
def read_google_sheet(spreadsheet_id: str, range_name: str, creds_info: Optional[Dict] = None, creds_file: Optional[str] = None) -> pd.DataFrame:
    """Reads the given Google Sheet range and returns a DataFrame."""
    if creds_info is None and (creds_file is None or not os.path.exists(creds_file)):
        if "gcp_service_account" not in st.secrets:
            raise ValueError("No credentials found. Add service account JSON to st.secrets['gcp_service_account'].")
        creds_info = parse_service_account_secret(st.secrets["gcp_service_account"])

    service = (
        build_sheets_service_from_info(creds_info)
        if creds_info
        else build_sheets_service_from_file(creds_file)
    )

    try:
        sheet = service.spreadsheets()
        res = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
        values = res.get("values", [])
    except HttpError as e:
        raise RuntimeError(f"Google Sheets API error: {e}")
    return values_to_dataframe(values)

# ---------------- Main Execution ----------------
if not SHEET_ID:
    st.info("Enter your Google Sheet ID to load data (the long ID between /d/ and /edit in the URL).")
    st.stop()

with st.spinner("🔄 Fetching data from Google Sheets..."):
    try:
        creds_info = None
        if "gcp_service_account" in st.secrets:
            creds_info = parse_service_account_secret(st.secrets["gcp_service_account"])
        df = read_google_sheet(SHEET_ID, RANGE, creds_info=creds_info, creds_file=CREDS_FILE)
    except Exception as e:
        st.error(f"❌ Failed to read Google Sheet: {e}")
        st.stop()
if df.empty:
    st.warning("⚠️ No data returned. Check the sheet name/range and ensure the service account has viewer access.")
    st.stop()

# ---------- Clean the raw DataFrame and show cleaned result ----------
# cached wrapper for cleaning
@st.cache_data(ttl=300)
def _clean_cached(df_raw):
    # assumes clean_history_transactions is already imported from cleaning.py
    return clean_history_transactions(df_raw)

try:
    with st.spinner("Cleaning data..."):
        cleaned_df = _clean_cached(df)
except Exception as e:
    st.error(f"Cleaning failed: {e}")
    st.stop()
# ---------- Show counts, KPIs, download and charts (no tables) ----------

# show how many rows were read from the sheet (raw)
rows_read = len(df)
st.success(f"✅ Successfully loaded data from Google Sheet — {rows_read:,} rows read.")

# aggregated KPIs (displayed prominently)


# safe computations (guard against missing columns)
def safe_sum_by_type(df_in, match_str):
    try:
        return df_in.loc[df_in["Type"].str.lower().str.contains(match_str, na=False), "Amount"].sum()
    except Exception:
        return 0.0

total_debit = safe_sum_by_type(cleaned_df, "debit")
total_credit = safe_sum_by_type(cleaned_df, "credit")

try:
    suspicious_count = int(cleaned_df["Suspicious"].sum())
except Exception:
    suspicious_count = 0

# show KPIs in a single row
col1, col2, col3, col4 = st.columns([1,1,1,1])
col1.metric("Total Debit", f"{total_debit:,.2f}")
col2.metric("Total Credit", f"{total_credit:,.2f}")
col3.metric("Suspicious", f"{suspicious_count:,}")

# small secondary info row: show rows read from sheet (raw) and column list
colA, colB = st.columns([1,3])
# Download cleaned CSV (keep)
clean_csv = cleaned_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Cleaned CSV", data=clean_csv, file_name="history_transactions_cleaned.csv", mime="text/csv")
# Year selector (add after you have cleaned_df)
# build year options safely
if "DateTime" in cleaned_df.columns and not cleaned_df["DateTime"].dropna().empty:
    years = cleaned_df["DateTime"].dropna().dt.year.astype(int).sort_values().unique().tolist()
else:
    years = []

years_opts = ["All"] + [int(y) for y in years]
default_index = len(years_opts) - 1 if years_opts else 0
selected_year = st.sidebar.selectbox("Year", options=years_opts, index=default_index)

year_filter = None if selected_year == "All" else int(selected_year)

# render chart (monthly trend) — use a single container and single call
chart_container = st.container()
monthly_trend(cleaned_df, container=chart_container, year=selected_year if selected_year!="All" else None, show_debit_credit=True)
