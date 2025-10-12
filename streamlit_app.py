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
    # strip triple-quoted wrappers if present
    if (s.startswith('"""') and s.endswith('"""')) or (s.startswith("'''") and s.endswith("'''")):
        s = s[3:-3].strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            # sometimes secrets store newlines as escaped \n
            return json.loads(s.replace('\\n', '\n'))
        except Exception:
            # fallback try the opposite replacement
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
        # no meaningful header row — build synthetic header
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
    # If no credentials provided, try st.secrets
    if creds_info is None and (creds_file is None or not os.path.exists(creds_file)):
        if "gcp_service_account" not in st.secrets:
            raise ValueError("No credentials found. Add service account JSON to st.secrets['gcp_service_account'] or supply a local file.")
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

# guard early if user hasn't provided sheet id
if not SHEET_ID:
    st.info("Enter your Google Sheet ID to load data (the long ID between /d/ and /edit in the URL).")
    st.stop()

with st.spinner("🔄 Fetching data from Google Sheets..."):
    try:
        creds_info = None  # <<-- initialize cleanly (fixes the SyntaxError you saw)
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
rows_read = len(df)
st.success(f"✅ Successfully loaded data from Google Sheet — {rows_read:,} rows read.")

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
colA.write(f"Rows read: {rows_read:,}")
colB.write(f"Columns: {', '.join(cleaned_df.columns.astype(str))}")

# Download cleaned CSV
clean_csv = cleaned_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Cleaned CSV", data=clean_csv, file_name="history_transactions_cleaned.csv", mime="text/csv")

from charts import daily_spend_line_chart

# Ensure DateTime present and convert if necessary (cleaning function should handle this, but safe-guard here)
if 'DateTime' in cleaned_df.columns and not pd.api.types.is_datetime64_any_dtype(cleaned_df['DateTime']):
    try:
        cleaned_df['DateTime'] = pd.to_datetime(cleaned_df['DateTime'])
    except Exception:
        # if conversion fails, leave as-is and chart function will try its best
        pass

# Create a daily spend summary (example)
try:
    df_debit = cleaned_df[cleaned_df['Type'].str.lower() == 'debit']
    daily_spend = (
        df_debit.groupby(cleaned_df['DateTime'].dt.date)['Amount']
        .sum()
        .reset_index()
    )
    daily_spend.columns = ['Date', 'Total_Spent']
except Exception:
    daily_spend = pd.DataFrame(columns=['Date', 'Total_Spent'])

# Optionally, also calculate daily credit
try:
    df_credit = cleaned_df[cleaned_df['Type'].str.lower() == 'credit']
    daily_credit = (
        df_credit.groupby(cleaned_df['DateTime'].dt.date)['Amount']
        .sum()
        .reset_index()
    )
    daily_credit.columns = ['Date', 'Total_Credit']
except Exception:
    daily_credit = pd.DataFrame(columns=['Date', 'Total_Credit'])

# Merge both for plotting (outer join to preserve either side)
if not daily_spend.empty or not daily_credit.empty:
    merged = pd.merge(daily_spend, daily_credit, on='Date', how='outer').fillna(0)
    # Ensure Date column is a datetime-like object
    try:
        merged['Date'] = pd.to_datetime(merged['Date'])
    except Exception:
        pass
else:
    merged = pd.DataFrame(columns=['Date', 'Total_Spent', 'Total_Credit'])

# Call your chart function from charts.py
st.subheader("📊 Daily Spending and Credit Trend")
daily_spend_line_chart(merged, debit_col='Total_Spent', credit_col='Total_Credit')
