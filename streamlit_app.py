# streamlit_app.py
"""
Streamlit App: Read transactions from Google Sheets and show a Matplotlib
line chart of daily debit totals (true zigzag, non-cumulative).
"""

import streamlit as st
import pandas as pd
import json
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# matplotlib imports
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

st.set_page_config(page_title="💸 Debit Trend (Matplotlib)", layout="wide")
st.title("💸 Daily Debit Spending Trend — Matplotlib (True Daily Totals)")

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
            raise FileNotFoundError(
                "No credentials provided. Put service account JSON path in sidebar or st.secrets['gcp_service_account']." )
        creds = service_account.Credentials.from_service_account_file(creds_file, scopes=scopes)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)

@st.cache_data(ttl=300)
def read_google_sheet(spreadsheet_id, range_name, creds_info=None, creds_file=None):
    """Read a Google Sheet range using values.get and return a DataFrame."""
    service = build_sheets_service(creds_info, creds_file)
    sheet = service.spreadsheets()
    try:
        res = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    except HttpError as e:
        raise RuntimeError(f"Google Sheets API error: {e}")
    values = res.get("values", [])
    if not values:
        return pd.DataFrame()
    header = [str(h).strip() for h in values[0]]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=header)
    return df

def normalize_and_map_columns(df):
    """Map common header names to DateTime, Type, Amount."""
    cols = {c.lower().strip(): c for c in df.columns}
    mapping = {}
    for key in cols:
        if key in ("datetime", "date time", "date_time", "date"):
            mapping[cols[key]] = "DateTime"
        if "type" in key and ("type" == key or key.startswith("type") or "txn type" in key):
            mapping[cols[key]] = "Type"
        if "amount" in key or "amt" in key:
            mapping[cols[key]] = "Amount"
    if mapping:
        df = df.rename(columns=mapping)
    return df

def parse_amount_column(series):
    """Parse amounts: remove commas & non-digit chars (keep dot and minus)."""
    s = series.astype(str).fillna("")
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

# ---------------- Clean & standardize ----------------
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
df = df.dropna(subset=["DateTime", "Amount", "Type"])

# ---------------- Filter debits and aggregate daily totals ----------------
df["Type_clean"] = df["Type"].astype(str).str.lower().str.strip()
df_debit = df[df["Type_clean"].str.contains("debit", na=False)].copy()
if df_debit.empty:
    st.warning("No debit transactions found after filtering.")
    st.stop()

# Ensure positive amounts for plotting (use magnitude)
df_debit["Amount"] = df_debit["Amount"].abs()

# Create Date column (midnight) for grouping
df_debit["Date"] = pd.to_datetime(df_debit["DateTime"].dt.date)

daily_spend = (
    df_debit.groupby("Date", as_index=False)["Amount"]
    .sum()
    .rename(columns={"Amount": "Total_Spent"})
    .sort_values("Date")
)

if daily_spend.empty:
    st.warning("No aggregated daily totals available.")
    st.stop()

# ---------------- Sidebar plot options ----------------
st.sidebar.write("📈 Plot display options (Matplotlib)")
mode = st.sidebar.radio("Display mode", ["Raw (full amounts)", "Rescale to ₹k (thousands)", "Clip to 95th percentile"])

use_custom = st.sidebar.checkbox("Use custom Y range", value=False)
custom_min = None
custom_max = None
if use_custom:
    custom_min = st.sidebar.number_input("Y-axis min", value=0.0, step=1.0)
    custom_max = st.sidebar.number_input("Y-axis max", value=float(daily_spend["Total_Spent"].max()), step=1.0)

plot_df = daily_spend.copy().sort_values("Date").reset_index(drop=True)
plot_df["Plot_Value"] = plot_df["Total_Spent"]
y_label = "Daily Spent (₹)"

if mode == "Rescale to ₹k (thousands)":
    plot_df["Plot_Value"] = plot_df["Total_Spent"] / 1000.0
    y_label = "Daily Spent (₹k)"
elif mode == "Clip to 95th percentile":
    p95 = plot_df["Total_Spent"].quantile(0.95)
    outliers = (plot_df["Total_Spent"] > p95).sum()
    st.sidebar.write(f"Clipping to 95th percentile = {p95:,.0f} (hiding {outliers} outlier(s))")
    plot_df["Plot_Value"] = plot_df["Total_Spent"].clip(upper=p95)
    y_label = f"Daily Spent (₹) — clipped @95p ({p95:,.0f})"

# Y range
if use_custom:
    y_min, y_max = custom_min, custom_max
else:
    y_min = max(0, float(plot_df["Plot_Value"].min() * 0.0))
    y_max = float(plot_df["Plot_Value"].max() * 1.03)

# ---------------- Diagnostics ----------------
st.subheader("Diagnostics: daily_spend")
col1, col2 = st.columns([1, 2])
with col1:
    st.write("Rows:", len(plot_df))
    st.write(f"Min: {plot_df['Plot_Value'].min():,.2f}")
    st.write(f"Max: {plot_df['Plot_Value'].max():,.2f}")
with col2:
    st.dataframe(plot_df.head(20))
plot_df["diff"] = plot_df["Total_Spent"].diff()
st.write("Day-to-day diffs (first 20 rows):")
st.dataframe(plot_df[["Date", "Total_Spent", "diff"]].head(20))

# ---------------- Matplotlib plotting ----------------
st.subheader("Matplotlib — Line (linear) + markers (true zigzag)")

fig, ax = plt.subplots(figsize=(12, 5.5))

# Plot line + markers
ax.plot(plot_df["Date"], plot_df["Plot_Value"], marker='o', linestyle='-', color='#0074D9', linewidth=1.8, markersize=6)

# Format x-axis for dates
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
plt.xticks(rotation=20)

# Y-axis formatting: comma separator, optionally show 'k' label if rescaled
def yfmt(x, pos):
    if mode == "Rescale to ₹k (thousands)":
        # display with one decimal if needed
        if abs(x) >= 100:
            return f"{x:,.0f}"
        return f"{x:,.1f}"
    else:
        return f"{int(x):,}"
ax.yaxis.set_major_formatter(FuncFormatter(yfmt))

ax.set_ylabel(y_label)
ax.set_xlabel("Date")
ax.set_title("Daily Debit Spending (Matplotlib — linear connections, no smoothing)")

# Set y-limits
ax.set_ylim(y_min, y_max)

# Grid & layout
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()

# Show in Streamlit
st.pyplot(fig)

# ---------------- Optional: show raw aggregated table ----------------
with st.expander("🔍 Daily aggregated totals (raw)"):
    st.dataframe(daily_spend.reset_index(drop=True), use_container_width=True)

with st.expander("🔍 Raw debit transactions sample"):
    st.dataframe(df_debit.head(100), use_container_width=True)
