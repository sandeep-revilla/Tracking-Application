# streamlit_app.py
"""
Streamlit App: Connect to Google Sheet and show interactive Plotly Debit Chart (True Zigzag)
"""

import streamlit as st
import pandas as pd
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
import plotly.express as px

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="💸 Debit Trend Dashboard", layout="wide")
st.title("💸 Daily Debit Spending Trend (from Google Sheet)")

# ---------------- SIDEBAR ----------------
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

# ---------------- HELPER FUNCTIONS ----------------
def parse_service_account_secret(raw):
    """Parse JSON secret from st.secrets or string"""
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    try:
        return json.loads(s)
    except Exception:
        return json.loads(s.replace("\\n", "\n"))


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
    """Read a Google Sheet range into a DataFrame"""
    service = build_sheets_service(creds_info, creds_file)
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    values = result.get("values", [])
    if not values:
        return pd.DataFrame()
    df = pd.DataFrame(values[1:], columns=values[0])
    return df


# ---------------- LOAD DATA ----------------
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
    st.warning("⚠️ No data found. Check your range or credentials.")
    st.stop()

st.success(f"✅ Loaded {len(df):,} rows from Google Sheet.")

# ---------------- CLEAN DATA ----------------
required_cols = {"DateTime", "Type", "Amount"}
if not required_cols.issubset(df.columns):
    st.error(f"Missing required columns: {required_cols - set(df.columns)}")
    st.stop()

df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
df = df.dropna(subset=["DateTime", "Amount", "Type"])

# ---------------- FILTER + AGGREGATE ----------------
df_debit = df[df["Type"].str.lower().str.strip() == "debit"]
df_debit["Amount"] = df_debit["Amount"].abs()
df_debit["Date"] = df_debit["DateTime"].dt.date

# ✅ True daily total (not cumulative)
daily_spend = (
    df_debit.groupby("Date", as_index=False)["Amount"]
    .sum()
    .rename(columns={"Amount": "Total_Spent"})
    .sort_values("Date")
)

# ---------------- INTERACTIVE FILTER ----------------
min_date, max_date = daily_spend["Date"].min(), daily_spend["Date"].max()
date_range = st.sidebar.date_input("📅 Select Date Range", [min_date, max_date])

if len(date_range) == 2:
    start, end = date_range
    daily_spend = daily_spend[
        (daily_spend["Date"] >= start) & (daily_spend["Date"] <= end)
    ]

# ---------------- PLOTLY CHART (ZIGZAG) ----------------
fig = px.line(
    daily_spend,
    x="Date",
    y="Total_Spent",
    title="📈 Daily Debit Spending Over Time (Zigzag, True Daily)",
    markers=True,
    line_shape="linear",  # keep zigzag — no smoothing
)

fig.update_traces(line=dict(width=2, color="#0074D9"))
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Total Spent (₹)",
    hovermode="x unified",
    template="plotly_white",
    height=600,
    title_x=0.5,
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- DATA PREVIEW ----------------
with st.expander("🔍 View Cleaned Data"):
    st.dataframe(daily_spend, use_container_width=True)
