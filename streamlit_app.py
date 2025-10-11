# streamlit_app.py
"""
Streamlit App: Read transactions from Google Sheets and show an interactive
Plotly line chart of daily debit totals (true zigzag, non-cumulative).
"""

import streamlit as st
import pandas as pd
import json
import os
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

# ---------- Plot with zoom/rescale options to show zigzag ----------
# we assume 'daily_spend' exists and is sorted by Date
daily_spend = daily_spend.sort_values("Date").reset_index(drop=True)

# Sidebar display options
st.sidebar.write("📈 Plot display options")
mode = st.sidebar.radio(
    "Display mode",
    options=["Raw (full amounts)", "Rescale to ₹k (thousands)", "Clip to 95th percentile"]
)

# Optional custom y-range (for manual zoom)
use_custom = st.sidebar.checkbox("Use custom Y range", value=False)
custom_min = None
custom_max = None
if use_custom:
    custom_min = st.sidebar.number_input("Y-axis min", value=0.0, step=1.0)
    custom_max = st.sidebar.number_input("Y-axis max", value=float(daily_spend["Total_Spent"].max()), step=1.0)

# Prepare plotting series depending on selected mode
plot_df = daily_spend.copy()
y_label = "Daily Spent (₹)"
y_col = "Total_Spent"

if mode == "Rescale to ₹k (thousands)":
    plot_df["Plot_Value"] = plot_df["Total_Spent"] / 1000.0
    y_label = "Daily Spent (₹k)"
    y_col = "Plot_Value"
elif mode == "Clip to 95th percentile":
    p95 = plot_df["Total_Spent"].quantile(0.95)
    # Mark outliers for information
    outliers_count = (plot_df["Total_Spent"] > p95).sum()
    st.sidebar.write(f"Clipping to 95th percentile = {p95:,.0f}. Outliers hidden: {outliers_count}")
    # create a clipped column for plotting (so axis is capped)
    plot_df["Plot_Value"] = plot_df["Total_Spent"].clip(upper=p95)
    y_label = f"Daily Spent (₹) — clipped @95p ({p95:,.0f})"
    y_col = "Plot_Value"
else:
    # Raw
    plot_df["Plot_Value"] = plot_df["Total_Spent"]
    y_col = "Plot_Value"

# Determine y-axis range
if use_custom:
    y_range = [custom_min, custom_max]
else:
    # automatic but prevent +/- tiny padding
    ymin = float(plot_df[y_col].min() * 0.0)  # typically 0
    ymax = float(plot_df[y_col].max() * 1.03)  # small headroom
    y_range = [ymin, ymax]

# Diagnostics quick readout (optional)
st.write(f"Data points: {len(plot_df)} — min {plot_df['Total_Spent'].min():,.0f}, max {plot_df['Total_Spent'].max():,.0f}")

# Plot: line + markers (linear, no smoothing)
fig = px.line(
    plot_df,
    x="Date",
    y=y_col,
    title="Daily Debit Spending (linear — lines+markers)",
    labels={y_col: y_label, "Date": "Date"},
    line_shape="linear",
    render_mode="svg",
)

fig.update_traces(mode="lines+markers", line_color="#0074D9", marker=dict(size=6))
fig.update_layout(
    template="plotly_white",
    hovermode="x unified",
    height=600,
    xaxis=dict(tickformat="%b %d\n%Y"),
    yaxis=dict(title=y_label, tickformat=",.0f", range=y_range),
)

st.plotly_chart(fig, use_container_width=True)

# Expanders: show raw values and diffs for debugging
with st.expander("🔍 Raw Total_Spent list (first 50)"):
    st.write(daily_spend[["Date", "Total_Spent"]].head(50).to_string(index=False))

with st.expander("🔍 Day-to-day diffs (first 30)"):
    diffs = daily_spend.copy()
    diffs["diff"] = diffs["Total_Spent"].diff()
    st.write(diffs[["Date", "Total_Spent", "diff"]].head(30).to_string(index=False))
