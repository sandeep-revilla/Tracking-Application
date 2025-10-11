# streamlit_app.py
"""
Streamlit App: Connect to Google Sheet (via Service Account) and show a Debit-Only Line Chart
"""

import streamlit as st
import pandas as pd
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import plotly.express as px

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="💰 Google Sheet Debit Chart", layout="wide")
st.title("💰 Google Sheet Connector — Debit Spending Chart")

# ---------------- SIDEBAR INPUTS ----------------
SHEET_ID = st.sidebar.text_input(
    "Google Sheet ID (between /d/ and /edit)",
    value="1KZq_GLXdMBfQUhtp-NA8Jg-flxOppw7kFuIN6y_nOXk",  # example
)
RANGE = st.sidebar.text_input("Range or Sheet Name", value="History Transactions")

st.sidebar.caption("Provide your Service Account JSON file below or via st.secrets.")

CREDS_FILE = st.sidebar.text_input(
    "Service Account JSON File (optional)", value="creds/service_account.json"
)

if st.sidebar.button("🔄 Refresh Now"):
    st.experimental_rerun()


# ---------------- HELPER: Parse Secrets ----------------
def parse_service_account_secret(raw):
    """Parse JSON string or dict from st.secrets or file"""
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(s.replace("\\n", "\n"))
        except Exception:
            return json.loads(s.replace("\n", "\\n"))


# ---------------- GOOGLE SHEET HELPERS ----------------
@st.cache_data(ttl=300)
def build_sheets_service(creds_info=None, creds_file=None):
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    if creds_info:
        creds = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
    else:
        creds = service_account.Credentials.from_service_account_file(creds_file, scopes=scopes)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


@st.cache_data(ttl=300)
def read_google_sheet(spreadsheet_id, range_name, creds_info=None, creds_file=None):
    """Read Google Sheet range and return as DataFrame"""
    service = build_sheets_service(creds_info, creds_file)
    sheet = service.spreadsheets()
    try:
        result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
        values = result.get("values", [])
    except HttpError as e:
        raise RuntimeError(f"Google Sheets API error: {e}")

    if not values:
        return pd.DataFrame()

    header = [h.strip() for h in values[0]]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=header)
    return df


# ---------------- MAIN EXECUTION ----------------
if not SHEET_ID:
    st.info("ℹ️ Enter your Google Sheet ID to load data.")
    st.stop()

with st.spinner("🔄 Loading data from Google Sheets..."):
    try:
        creds_info = None
        if "gcp_service_account" in st.secrets:
            creds_info = parse_service_account_secret(st.secrets["gcp_service_account"])
        df = read_google_sheet(SHEET_ID, RANGE, creds_info=creds_info, creds_file=CREDS_FILE)
    except Exception as e:
        st.error(f"❌ Failed to read Google Sheet: {e}")
        st.stop()

if df.empty:
    st.warning("⚠️ No data found. Check the sheet name/range or access permissions.")
    st.stop()

st.success(f"✅ Loaded {len(df):,} rows from Google Sheet.")

# ---------------- CLEAN DATA ----------------
# Normalize column names
df.columns = df.columns.str.strip().str.title()

# Check required columns (case-insensitive)
required_cols = {"Datetime", "Type", "Amount"}
if not required_cols.issubset(set(df.columns)):
    st.error(f"⚠️ Missing required columns: {required_cols - set(df.columns)}")
    st.stop()

# Convert Amount safely (handles commas and text)
df["Amount"] = (
    df["Amount"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.extract(r"(\d+\.?\d*)")[0]
    .astype(float)
)
# 
# Parse DateTime
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

# Drop invalid rows
df = df.dropna(subset=["Datetime", "Amount"])

# Filter only debit transactions
df_debit = df[df["Type"].str.lower() == "debit"].copy()

if df_debit.empty:
    st.warning("⚠️ No debit transactions found.")
    st.stop()

# Group by date and sum
daily_spend = (
    df_debit.groupby(df_debit["Datetime"].dt.date)["Amount"]
    .sum()
    .reset_index()
    .rename(columns={"Datetime": "Date", "Amount": "Total_Spent"})
    .sort_values("Date")
)

# ---------------- PLOT ----------------
fig = px.line(
    daily_spend,
    x="Date",
    y="Total_Spent",
    title="💸 Daily Debit Spending Over Time",
    markers=True,
    line_shape="linear"
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Total Spent (₹)",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- SHOW RAW DATA ----------------
with st.expander("🔍 View Raw Debit Data"):
    st.dataframe(df_debit.reset_index(drop=True))
