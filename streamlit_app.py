# streamlit_app.py
"""
Streamlit helper: connect to a private Google Sheet via service account,
convert to pandas DataFrame, and show a Plotly line chart (no cleaning logic here).
"""

import streamlit as st
import pandas as pd
import json
import os
from typing import List, Tuple, Optional, Any, Dict
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import plotly.express as px

# ---------------- Streamlit Page Setup ----------------
st.set_page_config(page_title="Google Sheet Connector", layout="wide")
st.title("🔐 Google Sheet Connector — (Connection + Chart)")

# ---------------- Sidebar Inputs ----------------
SHEET_ID = st.sidebar.text_input(
    "Google Sheet ID (between /d/ and /edit)",
    value="1KZq_GLXdMBfQUhtp-NA8Jg-flxOppw7kFuIN6y_nOXk",
)
RANGE = st.sidebar.text_input("Range or Sheet Name", value="History Transactions")
st.sidebar.caption(
    "Provide your Service Account JSON via st.secrets['gcp_service_account'] or as a local file below."
)
CREDS_FILE = st.sidebar.text_input(
    "Service Account JSON File (optional)", value="creds/service_account.json"
)

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
            return json.loads(s.replace("\\n", "\n"))
        except Exception:
            return json.loads(s.replace("\n", "\\n"))

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

st.success(f"✅ Successfully loaded data from Google Sheet — {len(df):,} rows read.")

import pandas as pd
import plotly.express as px
import streamlit as st

df['DateTime'] = pd.to_datetime(df['DateTime'])

# Filter only debit transactions
df_debit = df[df['Type'].str.lower() == 'debit']

# (Optional) Aggregate by day if you have many entries per day
daily_spend = df_debit.groupby(df_debit['DateTime'].dt.date)['Amount'].sum().reset_index()
daily_spend.columns = ['Date', 'Total_Spent']
fig = px.line(
    daily_spend,
    x='Date',
    y='Total_Spent',
    title='💸 Daily Spending Over Time',
    markers=True,
    line_shape='spline'
)

fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Total Spent (₹)',
    template='plotly_white'
)

st.plotly_chart(fig, use_container_width=True)
