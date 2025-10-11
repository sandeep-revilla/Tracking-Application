# streamlit_app.py
"""
Streamlit App: Plot debit spending from Google Sheet using Matplotlib
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="💰 Debit Chart (Matplotlib + Google Sheets)", layout="wide")
st.title("💰 Debit Spending Tracker — Matplotlib + Google Sheets")

# ---------- SIDEBAR INPUT ----------
SHEET_ID = st.sidebar.text_input(
    "Google Sheet ID (between /d/ and /edit)",
    value="1KZq_GLXdMBfQUhtp-NA8Jg-flxOppw7kFuIN6y_nOXk",
)
RANGE = st.sidebar.text_input("Sheet Name or Range", value="History Transactions")
CREDS_FILE = st.sidebar.text_input(
    "Service Account JSON File (optional)", value="creds/service_account.json"
)

if st.sidebar.button("🔄 Refresh"):
    st.experimental_rerun()

# ---------- GOOGLE SHEET HELPERS ----------
def parse_service_account_secret(raw):
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

# ---------- LOAD DATA ----------
with st.spinner("🔄 Fetching data from Google Sheets..."):
    try:
        creds_info = None
        if "gcp_service_account" in st.secrets:
            creds_info = parse_service_account_secret(st.secrets["gcp_service_account"])
        df = read_google_sheet(SHEET_ID, RANGE, creds_info=creds_info, creds_file=CREDS_FILE)
    except Exception as e:
        st.error(f"❌ Error reading Google Sheet: {e}")
        st.stop()

if df.empty:
    st.warning("⚠️ No data found. Check sheet name or permissions.")
    st.stop()

st.success(f"✅ Loaded {len(df)} rows from Google Sheet")

# ---------- CLEAN DATA ----------
df.columns = df.columns.str.strip().str.title()
required_cols = {"Datetime", "Type", "Amount"}
if not required_cols.issubset(df.columns):
    st.error(f"⚠️ Missing required columns: {required_cols - set(df.columns)}")
    st.stop()

df["Amount"] = (
    df["Amount"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.extract(r"(\d+\.?\d*)")[0]
    .astype(float)
)
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
df = df.dropna(subset=["Datetime", "Amount"])

# ---------- FILTER DEBIT ----------
df_debit = df[df["Type"].str.lower() == "debit"]

if df_debit.empty:
    st.warning("⚠️ No debit transactions found.")
    st.stop()

# ---------- AGGREGATE DAILY ----------
daily_spend = (
    df_debit.groupby(df_debit["Datetime"].dt.date)["Amount"]
    .sum()
    .reset_index()
    .rename(columns={"Datetime": "Date", "Amount": "Total_Spent"})
)

# ---------- PLOT WITH MATPLOTLIB ----------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(daily_spend["Date"], daily_spend["Total_Spent"], marker="o", color="tab:blue")
ax.set_title("💸 Daily Debit Spending Over Time", fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Total Spent (₹)")
ax.grid(True, linestyle="--", alpha=0.5)
plt.xticks(rotation=45)
st.pyplot(fig)

# ---------- SHOW RAW DATA ----------
with st.expander("🔍 View Raw Debit Data"):
    st.dataframe(df_debit)
