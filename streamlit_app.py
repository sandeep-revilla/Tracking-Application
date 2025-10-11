# streamlit_app.py
"""
Streamlit App: Read transactions from Google Sheets and show an interactive
Plotly line chart of daily debit totals (true zigzag, non-cumulative).

Paste this file as streamlit_app.py and run:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import json
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import plotly.express as px

# ---------------- Page ----------------
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
        try:
            return json.loads(s.replace("\\n", "\n"))
        except Exception:
            return json.loads(s.replace("\n", "\\n"))

@st.cache_data(ttl=300)
def build_sheets_service(creds_info=None, creds_file=None):
    """Return Google Sheets service client using either creds_info or a file path."""
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    if creds_info:
        creds = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
    else:
        if not (creds_file and os.path.exists(creds_file)):
            raise FileNotFoundError("No credentials provided. Put service account JSON path in sidebar or in st.secrets['gcp_service_account'].")
        creds = service_account.Credentials.from_service_account_file(creds_file, scopes=scopes)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)

@st.cache_data(ttl=300)
def read_google_sheet(spreadsheet_id, range_name, creds_info=None, creds_file=None):
    """Read a Google Sheet range (values.get) and return a pandas DataFrame."""
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
    """Robustly map common header names to DateTime, Type, Amount."""
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
    """Parse amounts safely: strip commas & non-numeric chars but keep - and ."""
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

# ---------------- Clean / standardize ----------------
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

# ---------------- Filter debits and aggregate by day ----------------
df["Type_clean"] = df["Type"].astype(str).str.lower().str.strip()
df_debit = df[df["Type_clean"].str.contains("debit", na=False)].copy()
if df_debit.empty:
    st.warning("No debit transactions found in the sheet after filtering.")
    st.stop()

# Ensure amounts are positive magnitudes for plotting
df_debit["Amount"] = df_debit["Amount"].abs()

# Create a date column (datetime at midnight) for grouping & plotting
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

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start, end = date_range
    start = pd.to_datetime(start).normalize()
    end = pd.to_datetime(end).normalize()
    daily_spend = daily_spend[(daily_spend["Date"] >= start) & (daily_spend["Date"] <= end)]
else:
    # handle single selection fallback
    selected = pd.to_datetime(date_range).normalize()
    daily_spend = daily_spend[daily_spend["Date"] == selected]

if daily_spend.empty:
    st.warning("No data in the selected date range.")
    st.stop()

# ---------------- Plot options to reveal zigzag ----------------
daily_spend = daily_spend.sort_values("Date").reset_index(drop=True)

st.sidebar.write("📈 Plot display options")
mode = st.sidebar.radio(
    "Display mode",
    options=["Raw (full amounts)", "Rescale to ₹k (thousands)", "Clip to 95th percentile"]
)

use_custom = st.sidebar.checkbox("Use custom Y range", value=False)
custom_min = None
custom_max = None
if use_custom:
    custom_min = st.sidebar.number_input("Y-axis min", value=0.0, step=1.0)
    custom_max = st.sidebar.number_input("Y-axis max", value=float(daily_spend["Total_Spent"].max()), step=1.0)

plot_df = daily_spend.copy()
plot_df["Plot_Value"] = plot_df["Total_Spent"]  # default
y_label = "Daily Spent (₹)"

if mode == "Rescale to ₹k (thousands)":
    plot_df["Plot_Value"] = plot_df["Total_Spent"] / 1000.0
    y_label = "Daily Spent (₹k)"
elif mode == "Clip to 95th percentile":
    p95 = plot_df["Total_Spent"].quantile(0.95)
    outliers_count = (plot_df["Total_Spent"] > p95).sum()
    st.sidebar.write(f"Clipping to 95th percentile = {p95:,.0f}. Outliers hidden: {outliers_count}")
    plot_df["Plot_Value"] = plot_df["Total_Spent"].clip(upper=p95)
    y_label = f"Daily Spent (₹) — clipped @95p ({p95:,.0f})"

# Y range
if use_custom:
    y_range = [custom_min, custom_max]
else:
    ymin = float(plot_df["Plot_Value"].min() * 0.0)  # usually zero
    ymax = float(plot_df["Plot_Value"].max() * 1.03)  # slight headroom
    y_range = [ymin, ymax]

# ---------- Diagnostics (optional) ----------
st.subheader("Diagnostics: daily_spend")
col1, col2 = st.columns([1, 2])
with col1:
    st.write("Rows:", len(daily_spend))
    st.write(f"Min: {daily_spend['Total_Spent'].min():,.2f}")
    st.write(f"Max: {daily_spend['Total_Spent'].max():,.2f}")
with col2:
    st.dataframe(daily_spend.head(15))

daily_spend["diff"] = daily_spend["Total_Spent"].diff()
st.write("Day-to-day diffs (first 15 rows):")
st.dataframe(daily_spend[["Date", "Total_Spent", "diff"]].head(15))

# ---------- Scatter (verify points) ----------
st.subheader("Scatter of raw daily totals (verifies point positions)")
fig_scatter = px.scatter(
    plot_df,
    x="Date",
    y="Plot_Value",
    title="Raw daily points (no connecting lines)",
    labels={"Plot_Value": y_label, "Date": "Date"},
    render_mode="svg",
)
fig_scatter.update_traces(marker=dict(size=7, color="darkblue"))
fig_scatter.update_layout(yaxis=dict(tickformat=",.0f", range=y_range), template="plotly_white", height=420)
st.plotly_chart(fig_scatter, use_container_width=True)

# ---------- Line + markers (true zigzag) ----------
st.subheader("Line + markers (linear connections — true zigzag)")
fig = px.line(
    plot_df,
    x="Date",
    y="Plot_Value",
    title="Daily Debit Spending (linear — lines+markers)",
    labels={"Plot_Value": y_label, "Date": "Date"},
    line_shape="linear",
    render_mode="svg",
)
fig.update_traces(mode="lines+markers", line_color="#0074D9", marker=dict(size=6))
fig.update_layout(
    template="plotly_white",
    hovermode="x unified",
    height=560,
    xaxis=dict(tickformat="%b %d\n%Y"),
    yaxis=dict(title=y_label, tickformat=",.0f", range=y_range),
    margin=dict(l=60, r=30, t=70, b=80),
)
st.plotly_chart(fig, use_container_width=True)

# ---------- Debug expanders ----------
with st.expander("🔍 Raw Total_Spent list (for debugging)"):
    st.write(daily_spend[["Date", "Total_Spent"]].to_string(index=False))

with st.expander("🔍 Full aggregated table"):
    st.dataframe(daily_spend.reset_index(drop=True), use_container_width=True)
