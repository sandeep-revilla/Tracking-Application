# streamlit_app.py
"""
Streamlit App: Read transactions from Google Sheets and show an interactive
Plotly line chart of daily debit totals (true zigzag, non-cumulative).
"""

import streamlit as st
import pandas as pd
import json
import os
import re
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import plotly.express as px


st.set_page_config(page_title="💸 Debit Trend Dashboard", layout="wide")
st.title("💸 Daily Debit Spending Trend (from Google Sheet)")

# ---------------- Sidebar inputs ----------------
SHEET_ID = st.sidebar.text_input(
    "🔗 Google Sheet ID (between /d/ and /edit)",
    value="1KZq_GLXdMBfQUhtp-NA8Jg-flxOppw7kFuIN6y_nOXk",
)
RANGE = st.sidebar.text_input("📄 Sheet Name or Range", value="History Transactions")
CREDS_FILE = st.sidebar.text_input(
    "🔐 Service Account JSON File (optional)", value="creds/service_account.json"
)

if st.sidebar.button("🔄 Refresh Data"):
    st.experimental_rerun()

# ---------------- Helpers ----------------
def parse_service_account_secret(raw):
    """Accept dict or JSON string stored in st.secrets['gcp_service_account']."""
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    if (s.startswith('"""') and s.endswith('"""')) or (s.startswith("'''") and s.endswith("'''")):
        s = s[3:-3].strip()
    try:
        return json.loads(s)
    except Exception:
        # Try to fix escaped newlines
        try:
            return json.loads(s.replace("\\n", "\n"))
        except Exception:
            return json.loads(s.replace("\n", "\\n"))

@st.cache_data(ttl=300)
def build_sheets_service(creds_info=None, creds_file=None):
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    if creds_info:
        creds = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
    else:
        if not (creds_file and os.path.exists(creds_file)):
            raise FileNotFoundError("No credentials provided. Put service account JSON path in sidebar or st.secrets['gcp_service_account'].")
        creds = service_account.Credentials.from_service_account_file(creds_file, scopes=scopes)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)

@st.cache_data(ttl=300)
def read_google_sheet(spreadsheet_id, range_name, creds_info=None, creds_file=None):
    """Return DataFrame from Google Sheet range using the Sheets API (values.get)."""
    service = build_sheets_service(creds_info, creds_file)
    sheet = service.spreadsheets()
    try:
        res = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    except HttpError as e:
        raise RuntimeError(f"Google Sheets API error: {e}")
    values = res.get("values", [])
    if not values:
        return pd.DataFrame()
    # Normalize header: strip spaces
    header = [str(h).strip() for h in values[0]]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=header)
    return df

def normalize_and_map_columns(df):
    """Make header matching robust and map to standard names."""
    cols = {c.lower().strip(): c for c in df.columns}
    mapping = {}
    # find best matches for required columns
    for key in cols:
        if key in ("datetime", "date time", "date_time", "date"):
            mapping[cols[key]] = "DateTime"
        if "type" in key and ("type" == key or key.startswith("type") or "txn type" in key):
            mapping[cols[key]] = "Type"
        if "amount" in key or "amt" in key:
            mapping[cols[key]] = "Amount"
    # apply rename
    if mapping:
        df = df.rename(columns=mapping)
    return df

def parse_amount_column(series):
    """Safely parse amounts: remove non-numeric characters except - and ."""
    s = series.astype(str).fillna("")
    # Remove commas and common currency symbols/text, keep digits, dot, minus
    cleaned = s.str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace(r"[^\d\.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")

# ---------------- Load sheet ----------------
with st.spinner("📥 Loading data from Google Sheets..."):
    try:
        creds_info = None
        if "gcp_service_account" in st.secrets:
            creds_info = parse_service_account_secret(st.secrets["gcp_service_account"])
        df = read_google_sheet(SHEET_ID, RANGE, creds_info=creds_info, creds_file=CREDS_FILE)
    except Exception as e:
        st.error(f"❌ Error loading Google Sheet: {e}")
        st.stop()

if df.empty:
    st.warning("⚠️ No data found. Check sheet name/range and credentials.")
    st.stop()
st.success(f"✅ Loaded {len(df):,} rows from Google Sheet.")

# ---------------- Clean / standardize columns ----------------
# Ensure dates are sorted for proper line plotting
daily_spend = df.sort_values('Date')

# Create the Plotly line chart (straight line segments)
fig = px.line(
    daily_spend, 
    x='Date', 
    y='Total_Spent', 
    title='Daily Debit Spending', 
    line_shape='linear',     # linear segments between points
    render_mode='svg'        # SVG for crisp lines
)
# Add markers to verify individual values
fig.update_traces(mode='lines+markers', line_color='blue')

st.plotly_chart(fig, use_container_width=True)
