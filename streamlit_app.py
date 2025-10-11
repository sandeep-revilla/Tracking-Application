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
df = normalize_and_map_columns(df)

required = {"DateTime", "Type", "Amount"}
if not required.issubset(set(df.columns)):
    st.error(f"Missing required columns: {required - set(df.columns)}. Sheet columns: {list(df.columns)}")
    with st.expander("Raw sheet header"):
        st.write(list(df.columns))
    st.stop()

# Parse types
df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
df["Amount"] = parse_amount_column(df["Amount"])
# drop rows lacking essential values
df = df.dropna(subset=["DateTime", "Amount", "Type"])

# ---------------- Filter debits and aggregate by day ----------------
# filter 'debit' robustly
df["Type_clean"] = df["Type"].astype(str).str.lower().str.strip()
df_debit = df[df["Type_clean"].str.contains("debit", na=False)].copy()
if df_debit.empty:
    st.warning("No debit transactions found in the sheet after filtering.")
    st.stop()

# if amounts are stored negative for debits, take absolute to show magnitude
df_debit["Amount"] = df_debit["Amount"].abs()

# Extract date (as datetime for Plotly)
df_debit["Date"] = pd.to_datetime(df_debit["DateTime"].dt.date)

# Group by date (non-cumulative daily totals)
daily_spend = (
    df_debit.groupby("Date", as_index=False)["Amount"]
    .sum()
    .rename(columns={"Amount": "Total_Spent"})
    .sort_values("Date")
)

if daily_spend.empty:
    st.warning("No daily totals available after aggregation.")
    st.stop()

# ---------------- Date range selector ----------------
min_date, max_date = daily_spend["Date"].min().date(), daily_spend["Date"].max().date()
date_range = st.sidebar.date_input("📅 Select Date Range", [min_date, max_date])

if isinstance(date_range, tuple) or isinstance(date_range, list):
    if len(date_range) == 2:
        start, end = date_range
        start = pd.to_datetime(start).normalize()
        end = pd.to_datetime(end).normalize()
        daily_spend = daily_spend[(daily_spend["Date"] >= start) & (daily_spend["Date"] <= end)]
else:
    # single date selected -> show that day only
    selected = pd.to_datetime(date_range).normalize()
    daily_spend = daily_spend[daily_spend["Date"] == selected]

if daily_spend.empty:
    st.warning("No data in the selected date range.")
    st.stop()

# ---------------- Plotly line (true zigzag) ----------------
fig = px.line(
    daily_spend,
    x="Date",
    y="Total_Spent",
    title="📈 Daily Debit Spending (True Daily Totals — Zigzag)",
    labels={"Total_Spent": "Daily Spent (₹)", "Date": "Date"},
    markers=True,
    line_shape="linear",  # linear preserves actual ups/downs
)

# small UI tweaks: show markers clearly and format y axis
fig.update_traces(mode="lines+markers", marker=dict(size=6))
fig.update_layout(
    template="plotly_white",
    hovermode="x unified",
    height=600,
    xaxis=dict(title="Date", tickformat="%b %d\n%Y"),
    yaxis=dict(title="Daily Spent (₹)", tickformat=","),
    margin=dict(l=60, r=30, t=70, b=80),
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- Debug / data preview ----------------
with st.expander("🔍 View Raw & Aggregated Data"):
    st.subheader("Sample raw rows (first 50)")
    st.dataframe(df_debit.head(50))
    st.subheader("Daily aggregated totals")
    st.dataframe(daily_spend.head(50))
    st.write("dtypes:")
    st.write(df_debit.dtypes)
