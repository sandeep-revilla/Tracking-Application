# streamlit_app.py
"""
Streamlit + Plotly: Read transactions from Google Sheets and show an interactive
Plotly line chart of daily debit totals (non-cumulative). Includes zoom/rescale
modes and a zoomed inset (two-row) view so small zigzags remain visible.
"""

import streamlit as st
import pandas as pd
import json
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="💸 Debit Trend (Plotly)", layout="wide")
st.title("💸 Daily Debit Spending Trend (Plotly — zoom/rescale options)")

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
        if not (creds_file and os.path.exists(creds_file)):
            raise FileNotFoundError("No credentials provided. Put service account JSON path in sidebar or st.secrets['gcp_service_account'].")
        creds = service_account.Credentials.from_service_account_file(creds_file, scopes=scopes)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)

@st.cache_data(ttl=300)
def read_google_sheet(spreadsheet_id, range_name, creds_info=None, creds_file=None):
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
    return pd.DataFrame(rows, columns=header)

def normalize_and_map_columns(df):
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
    s = series.astype(str).fillna("")
    cleaned = s.str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace(r"[^\d\.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")

# ---------------- Load data ----------------
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

# ---------------- Clean & aggregate ----------------
df = normalize_and_map_columns(df)

required = {"DateTime", "Type", "Amount"}
if not required.issubset(set(df.columns)):
    st.error(f"Missing required columns: {required - set(df.columns)}. Sheet columns: {list(df.columns)}")
    with st.expander("Raw sheet header"):
        st.write(list(df.columns))
    st.stop()

df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
df["Amount"] = parse_amount_column(df["Amount"])
df = df.dropna(subset=["DateTime", "Amount", "Type"])
df["Type_clean"] = df["Type"].astype(str).str.lower().str.strip()

df_debit = df[df["Type_clean"].str.contains("debit", na=False)].copy()
if df_debit.empty:
    st.warning("No debit transactions found after filtering.")
    st.stop()

df_debit["Amount"] = df_debit["Amount"].abs()
df_debit["Date"] = pd.to_datetime(df_debit["DateTime"].dt.date)

daily_spend = (
    df_debit.groupby("Date", as_index=False)["Amount"]
    .sum()
    .rename(columns={"Amount": "Total_Spent"})
    .sort_values("Date")
)
if daily_spend.empty:
    st.warning("No daily totals available after aggregation.")
    st.stop()

# ---------------- UI: plot modes ----------------
st.sidebar.write("📈 Display options")
mode = st.sidebar.radio("Mode", ["Raw (full amounts)", "Rescale to ₹k", "Clip to 95th percentile", "Zoomed twin-chart"])
use_custom = st.sidebar.checkbox("Use custom Y range", value=False)
custom_min = None; custom_max = None
if use_custom:
    custom_min = st.sidebar.number_input("Y min", value=0.0)
    custom_max = st.sidebar.number_input("Y max", value=float(daily_spend["Total_Spent"].max()))

# Prepare plot_df
plot_df = daily_spend.copy().reset_index(drop=True)
plot_df["Plot_Value"] = plot_df["Total_Spent"]  # default
y_label = "Daily Spent (₹)"

if mode == "Rescale to ₹k":
    plot_df["Plot_Value"] = plot_df["Total_Spent"] / 1000.0
    y_label = "Daily Spent (₹k)"
elif mode == "Clip to 95th percentile":
    p95 = plot_df["Total_Spent"].quantile(0.95)
    st.sidebar.write(f"Clipping at 95th percentile: {p95:,.0f}")
    plot_df["Plot_Value"] = plot_df["Total_Spent"].clip(upper=p95)
    y_label = f"Daily Spent (₹) — clipped @95p ({p95:,.0f})"

# compute ranges
if use_custom:
    y_range = [custom_min, custom_max]
else:
    ymin = float(plot_df["Plot_Value"].min() * 0.0)
    ymax = float(plot_df["Plot_Value"].max() * 1.03)
    y_range = [ymin, ymax]

# ---------------- Diagnostics (optional) ----------------
st.subheader("Diagnostics")
col1, col2 = st.columns([1,2])
with col1:
    st.write("Rows:", len(plot_df))
    st.write(f"Min: {plot_df['Total_Spent'].min():,.0f}")
    st.write(f"Max: {plot_df['Total_Spent'].max():,.0f}")
with col2:
    st.dataframe(plot_df.head(20))

plot_df["diff"] = plot_df["Total_Spent"].diff()
st.write("First diffs:")
st.dataframe(plot_df[["Date","Total_Spent","diff"]].head(20))

# ---------------- Plotting ----------------
st.subheader("Plot")

if mode != "Zoomed twin-chart":
    # Single plot (raw/rescaled/clipped)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["Date"], y=plot_df["Plot_Value"],
        mode="lines+markers", name="Daily", line=dict(color="#0074D9"), marker=dict(size=6)
    ))
    fig.update_layout(
        template="plotly_white",
        height=560,
        xaxis=dict(title="Date", tickformat="%b %d\n%Y"),
        yaxis=dict(title=y_label, tickformat=",.0f", range=y_range),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    # Zoomed twin-chart: top small chart shows clipped view (95th percentile)
    p95 = plot_df["Total_Spent"].quantile(0.95)
    top_df = plot_df.copy()
    top_df["ZoomVal"] = top_df["Total_Spent"].clip(upper=p95)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.32, 0.68], specs=[[{}],[{}]]
    )

    # Top zoomed (clipped)
    fig.add_trace(go.Scatter(
        x=top_df["Date"], y=top_df["ZoomVal"],
        mode="lines+markers", name=f"Clipped @95% ({p95:,.0f})", line=dict(color="orange"), marker=dict(size=6)
    ), row=1, col=1)

    # Bottom full range
    fig.add_trace(go.Scatter(
        x=plot_df["Date"], y=plot_df["Total_Spent"],
        mode="lines+markers", name="Full (raw)", line=dict(color="#0074D9"), marker=dict(size=6)
    ), row=2, col=1)

    # layout
    fig.update_layout(template="plotly_white", height=760, hovermode="x unified")
    fig.update_xaxes(tickformat="%b %d\n%Y", row=2, col=1)
    fig.update_yaxes(title_text=f"Daily Spent (₹) — clipped view (top)", row=1, col=1)
    fig.update_yaxes(title_text="Daily Spent (₹) — full (bottom)", row=2, col=1)

    # Set top y-range to [0, p95*1.03] to emphasize small values
    fig.update_yaxes(range=[0, p95 * 1.03], row=1, col=1)
    # bottom keeps full auto-range (but we can set small headroom)
    fig.update_yaxes(range=[0, plot_df["Total_Spent"].max() * 1.03], row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

# ---------------- Expanders (debug) ----------------
with st.expander("Raw daily aggregated totals"):
    st.dataframe(daily_spend.reset_index(drop=True), use_container_width=True)

with st.expander("Raw debit rows (sample 100)"):
    st.dataframe(df_debit.head(100), use_container_width=True)
