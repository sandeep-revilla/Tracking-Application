# app.py
import streamlit as st
import pandas as pd
import json
from google.oauth2.service_account import Credentials
import gspread
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Live Expense Tracker", layout="wide")
st.title("💸 Live Expense Tracker (Google Sheets → Streamlit)")

# --- Sidebar: inputs ---
st.sidebar.header("Google Sheet connection")
SHEET_ID = st.sidebar.text_input("Google Sheet ID (between /d/ and /edit)", "")
# Worksheet selection will be filled after connecting
sheet_name_override = st.sidebar.text_input("Worksheet name (optional)", "")
refresh = st.sidebar.button("Refresh now")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Instructions: Put your Google service account JSON in Streamlit Secrets under the key `gcp_service_account`.\n"
    "Share the sheet with the service account email (client_email) as Viewer."
)

# --- Helper: parse service account stored in st.secrets ---
def load_service_account_from_secrets():
    """
    Accepts either:
      - st.secrets["gcp_service_account"] being a dict (already parsed)
      - st.secrets["gcp_service_account"] being a JSON string
      - TOML style triple-quoted JSON string (common when pasting into secrets)
    Returns: dict suitable for Credentials.from_service_account_info
    """
    if "gcp_service_account" not in st.secrets:
        raise KeyError("gcp_service_account not found in Streamlit secrets.")
    raw = st.secrets["gcp_service_account"]

    # If it's already a mapping (Streamlit may parse it), return directly
    if isinstance(raw, dict):
        return raw

    # If it's a string, try to load JSON directly
    if isinstance(raw, str):
        s = raw.strip()
        # If it's TOML triple-quoted style, remove surrounding triple quotes
        if s.startswith('"""') and s.endswith('"""'):
            s = s[3:-3].strip()
        if s.startswith("'''") and s.endswith("'''"):
            s = s[3:-3].strip()
        try:
            parsed = json.loads(s)
            return parsed
        except Exception as e:
            # As a last resort, try to fix newline escapes for private_key
            # Many people paste the JSON where the private_key has literal "\n" that need to be preserved
            try:
                s2 = s.replace('\\n', '\n')
                parsed = json.loads(s2)
                return parsed
            except Exception:
                raise ValueError("Could not parse gcp_service_account secret as JSON. Ensure you pasted the service account JSON correctly.") from e

    raise ValueError("Unsupported type for gcp_service_account in secrets.")


@st.cache_data(ttl=300)
def get_gspread_client():
    try:
        creds_info = load_service_account_from_secrets()
    except KeyError:
        st.error("Service account JSON not found in Streamlit Secrets under key `gcp_service_account`.")
        raise st.StopException
    except Exception as e:
        st.error(f"Error parsing service account JSON from secrets: {e}")
        raise st.StopException

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = Credentials.from_service_account_info(creds_info, scopes=scopes)
    gc = gspread.authorize(credentials)
    return gc


@st.cache_data(ttl=60)
def get_sheet_titles(sheet_id: str):
    """Return list of worksheet titles in the spreadsheet."""
    if not sheet_id:
        return []
    gc = get_gspread_client()
    sh = gc.open_by_key(sheet_id)
    return [ws.title for ws in sh.worksheets()]


@st.cache_data(ttl=60)
def load_sheet_as_df(sheet_id: str, worksheet_name: str | None):
    """Load worksheet into a pandas DataFrame."""
    if not sheet_id:
        return pd.DataFrame()
    gc = get_gspread_client()
    try:
        sh = gc.open_by_key(sheet_id)
    except Exception as e:
        st.error(f"Unable to open sheet by id: {e}")
        return pd.DataFrame()
    try:
        if worksheet_name:
            ws = sh.worksheet(worksheet_name)
        else:
            ws = sh.get_worksheet(0)
    except Exception:
        # fallback to first worksheet
        ws = sh.get_worksheet(0)
    records = ws.get_all_records(empty2zero=False, head=1)
    df = pd.DataFrame.from_records(records)
    return df


# If refresh requested, rerun
if refresh:
    st.experimental_rerun()

# If sheet id provided, try to list worksheets and allow user to pick
worksheet_titles = []
if SHEET_ID:
    try:
        worksheet_titles = get_sheet_titles(SHEET_ID)
    except Exception:
        worksheet_titles = []

selected_sheet = None
if worksheet_titles:
    selected_sheet = st.sidebar.selectbox("Choose worksheet", options=worksheet_titles, index=0)
# If user manually entered worksheet name, prefer that
if sheet_name_override.strip():
    selected_sheet = sheet_name_override.strip()

# Load dataframe
df = load_sheet_as_df(SHEET_ID, selected_sheet)

if df.empty:
    st.info("No data loaded yet. Provide Sheet ID and ensure worksheet has rows and headers.")
    st.stop()

# --- Data cleaning & preview ---
st.subheader("Raw data preview")
st.dataframe(df.head(10), use_container_width=True)

# Column detection heuristics
col_candidates = {c.lower(): c for c in df.columns}
date_col = col_candidates.get("datetime") or col_candidates.get("date") or None
amount_col = col_candidates.get("amount") or col_candidates.get("amt") or None
type_col = col_candidates.get("type") or None
bank_col = col_candidates.get("bank") or None
message_col = col_candidates.get("message") or col_candidates.get("msg") or None
suspicious_col = col_candidates.get("suspicious") or None

work = df.copy()

# Amount parsing
if amount_col and amount_col in work.columns:
    work["Amount"] = pd.to_numeric(work[amount_col], errors="coerce")
else:
    # fallback: take first numeric column if present
    numeric_cols = work.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        work["Amount"] = pd.to_numeric(work[numeric_cols[0]], errors="coerce")
    else:
        # try to extract digits from any column
        def extract_num(x):
            try:
                s = str(x)
                import re
                m = re.search(r"[-]?\d+[\d,]*(\.\d+)?", s.replace(",", ""))
                if m:
                    return float(m.group(0))
            except Exception:
                return None
            return None
        work["Amount"] = work.apply(lambda row: extract_num(row.astype(str).str.cat(sep=" ")), axis=1)

# Date parsing
if date_col and date_col in work.columns:
    work["DateTime"] = pd.to_datetime(work[date_col], errors="coerce")
else:
    # try to detect datetime-like columns
    found = False
    for c in work.columns:
        try:
            tmp = pd.to_datetime(work[c], errors="coerce")
            if tmp.notna().sum() > 0:
                work["DateTime"] = tmp
                found = True
                break
        except Exception:
            continue
    if not found:
        work["DateTime"] = pd.NaT

work["Date"] = pd.to_datetime(work["DateTime"]).dt.date
work["Month"] = pd.to_datetime(work["DateTime"]).dt.to_period("M").astype(str)
work["Weekday"] = pd.to_datetime(work["DateTime"]).dt.day_name()

# Type normalization
if type_col and type_col in work.columns:
    work["Type"] = work[type_col].astype(str).str.lower().str.strip()
else:
    # Infer from message or Amount sign
    work["Type"] = "unknown"
    work.loc[work["Amount"] < 0, "Type"] = "debit"
    work.loc[work["Amount"] > 0, "Type"] = "credit"
    if message_col and message_col in work.columns:
        work["Type"] = work[message_col].astype(str).str.lower().apply(
            lambda x: "debit" if "deb" in x or "withdraw" in x or "paid" in x else ("credit" if "cred" in x or "credited" in x else "unknown")
        ).combine_first(work["Type"])

# Ensure numeric amount exists
work["Amount"] = pd.to_numeric(work["Amount"], errors="coerce")

# --- Summary metrics ---
st.markdown("---")
st.header("Summary metrics")
col1, col2, col3, col4 = st.columns(4)
total_debit = work.loc[work["Type"] == "debit", "Amount"].sum()
total_credit = work.loc[work["Type"] == "credit", "Amount"].sum()
txn_count = len(work)
last_update = work["DateTime"].max()

col1.metric("Total Spent (Debit)", f"{total_debit:,.2f}")
col2.metric("Total Credit", f"{total_credit:,.2f}")
col3.metric("Transactions", txn_count)
col4.metric("Latest txn", str(last_update) if pd.notna(last_update) else "N/A")

# --- Charts ---
st.markdown("---")
st.subheader("Interactive charts")

# Daily spending trend
daily = work[work["Type"] == "debit"].groupby("Date")["Amount"].sum().reset_index()
if not daily.empty:
    fig1 = px.line(daily, x="Date", y="Amount", title="Daily Spending (debits)", markers=True)
    st.plotly_chart(fig1, use_container_width=True)

# Monthly stacked bar
monthly = work.groupby(["Month", "Type"])["Amount"].sum().reset_index()
if not monthly.empty:
    monthly_pivot = monthly.pivot(index="Month", columns="Type", values="Amount").fillna(0).reset_index()
    # Sort months chronologically if Month looks like YYYY-MM
    try:
        monthly_pivot = monthly_pivot.sort_values("Month")
    except Exception:
        pass
    types = [c for c in monthly_pivot.columns if c != "Month"]
    fig2 = px.bar(monthly_pivot, x="Month", y=types, title="Monthly Debit vs Credit (stacked)", barmode="stack")
    st.plotly_chart(fig2, use_container_width=True)

# Top merchants heuristic
if message_col and message_col in work.columns:
    # simple heuristic: look for "To <merchant>" or "at <merchant>"
    merchants = work[message_col].astype(str).str.extract(r"(?:To|to|at|@)\s+([A-Za-z0-9 &\.\-\/]{3,60})", expand=False)
    work["merchant"] = merchants.fillna("").str.strip()
    top_merchants = work[work["merchant"] != ""].groupby("merchant")["Amount"].sum().sort_values(ascending=False).head(10).reset_index()
    if not top_merchants.empty:
        fig3 = px.bar(top_merchants, x="merchant", y="Amount", title="Top merchants by spend (heuristic)")
        st.plotly_chart(fig3, use_container_width=True)

# Average spend by weekday
weekday = work[work["Type"] == "debit"].groupby("Weekday")["Amount"].mean().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
).reset_index()
if not weekday["Amount"].isna().all():
    fig4 = px.bar(weekday, x="Weekday", y="Amount", title="Average spend by weekday")
    st.plotly_chart(fig4, use_container_width=True)

# Bank-wise
if bank_col and bank_col in work.columns:
    bank = work.groupby([bank_col, "Type"])["Amount"].sum().reset_index()
    if not bank.empty:
        fig5 = px.bar(bank, x=bank_col, y="Amount", color="Type", barmode="group", title="Bank-wise Debit vs Credit")
        st.plotly_chart(fig5, use_container_width=True)

# Transaction size distribution
hist = work[work["Type"] == "debit"]["Amount"].dropna()
if not hist.empty:
    fig6 = px.histogram(hist, x=hist, nbins=30, title="Distribution of debit transaction amounts")
    st.plotly_chart(fig6, use_container_width=True)

# Cleaned table with download
st.markdown("---")
st.subheader("Cleaned transactions (preview)")
st.dataframe(work.head(200), use_container_width=True)

csv = work.to_csv(index=False).encode("utf-8")
st.download_button("Download cleaned CSV", data=csv, file_name="transactions_cleaned.csv", mime="text/csv")

st.markdown("---")
st.caption("App reads your Google Sheet live whenever the page is loaded or refreshed. Use the Refresh button to re-pull data immediately.")
