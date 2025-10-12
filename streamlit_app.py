# streamlit_app.py
"""
Single-file Streamlit app:
- Connects to Google Sheets via service account (st.secrets or file)
- Converts sheet values to a safe pandas DataFrame
- Performs basic cleaning (convert amounts, parse datetimes, infer Type)
- Shows KPIs + download button
- Aggregates daily debit/credit totals and plots them with Matplotlib (in-file)
- Includes diagnostic output to verify what's being plotted
"""

import streamlit as st
import pandas as pd
import json
import os
from typing import List, Tuple, Optional, Any, Dict
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# ---------------- Page config ----------------
st.set_page_config(page_title="Google Sheet Connector (Single File)", layout="wide")
st.title("🔐 Google Sheet Connector — Single-file version (Matplotlib)")

# ---------------- Sidebar Inputs ----------------
SHEET_ID = st.sidebar.text_input(
    "Google Sheet ID (between /d/ and /edit)",
    value="1KZq_GLXdMBfQUhtp-NA8Jg-flxOppw7kFuIN6y_nOXk"
)
RANGE = st.sidebar.text_input("Range or Sheet Name", value="History Transactions")
st.sidebar.caption("Provide service account JSON via st.secrets['gcp_service_account'] or as a local file below.")
CREDS_FILE = st.sidebar.text_input("Service Account JSON File (optional)", value="creds/service_account.json")

if st.sidebar.button("Refresh Now"):
    st.experimental_rerun()

# ---------------- Helper: parse service account JSON ----------------
def parse_service_account_secret(raw: Any) -> Dict:
    """Accept dict or JSON string (even with escaped newlines) and return dict."""
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    if (s.startswith('"""') and s.endswith('"""')) or (s.startswith("'''") and s.endswith("'''")):
        s = s[3:-3].strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(s.replace('\\n', '\n'))
        except Exception:
            return json.loads(s.replace('\n', '\\n'))

# ---------------- Sheets client builders ----------------
@st.cache_data(ttl=300)
def build_sheets_service_from_info(creds_info: Dict):
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)

@st.cache_data(ttl=300)
def build_sheets_service_from_file(creds_file: str):
    if not os.path.exists(creds_file):
        raise FileNotFoundError(f"Credentials file not found: {creds_file}")
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = service_account.Credentials.from_service_account_file(creds_file, scopes=scopes)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)

# ---------------- Safe conversion helpers ----------------
def _normalize_rows(values: List[List[str]]) -> Tuple[List[str], List[List]]:
    """Ensure header exists, pad/trim rows to header length."""
    if not values:
        return [], []

    header_row = [str(x).strip() for x in values[0]]
    if all((h == "" or h.lower().startswith(("unnamed", "column", "nan"))) for h in header_row):
        max_cols = max(len(r) for r in values)
        header = [f"col_{i}" for i in range(max_cols)]
        data_rows = values
    else:
        header = header_row
        data_rows = values[1:]

    col_count = len(header)
    normalized = []
    for r in data_rows:
        if len(r) < col_count:
            r = r + [None] * (col_count - len(r))
        elif len(r) > col_count:
            r = r[:col_count]
        normalized.append(r)
    return header, normalized

@st.cache_data(ttl=300)
def values_to_dataframe(values: List[List[str]]) -> pd.DataFrame:
    """Convert sheet values to pandas DataFrame safely."""
    if not values:
        return pd.DataFrame()
    header, rows = _normalize_rows(values)
    try:
        return pd.DataFrame(rows, columns=header)
    except Exception:
        df = pd.DataFrame(rows)
        if header and df.shape[1] == len(header):
            df.columns = header
        return df

# ---------------- Google Sheet reader ----------------
@st.cache_data(ttl=300)
def read_google_sheet(spreadsheet_id: str, range_name: str, creds_info: Optional[Dict] = None, creds_file: Optional[str] = None) -> pd.DataFrame:
    """Reads the given Google Sheet range and returns a DataFrame."""
    if creds_info is None and (creds_file is None or not os.path.exists(creds_file)):
        if "gcp_service_account" not in st.secrets:
            raise ValueError("No credentials found. Add service account JSON to st.secrets['gcp_service_account'] or supply a local file.")
        creds_info = parse_service_account_secret(st.secrets["gcp_service_account"])

    service = (build_sheets_service_from_info(creds_info) if creds_info else build_sheets_service_from_file(creds_file))

    try:
        sheet = service.spreadsheets()
        res = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
        values = res.get("values", [])
    except HttpError as e:
        raise RuntimeError(f"Google Sheets API error: {e}")
    return values_to_dataframe(values)

# ---------------- Basic cleaning function ----------------
def clean_history_transactions(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    A defensive, simple cleaning function. It:
    - normalizes column names
    - finds an Amount column and converts to numeric
    - finds a DateTime/Date column and converts to datetime
    - ensures a Type column exists (infers from Amount sign if missing)
    - creates a 'Suspicious' boolean where Amount is unusually large or missing
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()
    # normalize column names: strip for easy matching
    cols_map = {c: c.strip() for c in df.columns}
    df.rename(columns=cols_map, inplace=True)

    # find amount column candidates
    amount_col = None
    for candidate in ["Amount", "amount", "AMOUNT", "Txn Amount", "Value"]:
        if candidate in df.columns:
            amount_col = candidate
            break
    if amount_col is None:
        # try fuzzy: any column that looks numeric in first rows
        for c in df.columns:
            sample = df[c].astype(str).head(10).str.replace(r'[^\d\.\-]', '', regex=True)
            parsed = pd.to_numeric(sample, errors='coerce')
            if parsed.notna().sum() >= 3:
                amount_col = c
                break

    if amount_col:
        df[amount_col] = df[amount_col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
        df["Amount"] = pd.to_numeric(df[amount_col], errors='coerce')
    else:
        df["Amount"] = pd.NA

    # find date column candidates
    date_col = None
    for candidate in ["DateTime", "Datetime", "Timestamp", "Date", "date", "timestamp"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col:
        try:
            df["DateTime"] = pd.to_datetime(df[date_col], errors='coerce')
        except Exception:
            df["DateTime"] = pd.to_datetime(df[date_col].astype(str), errors='coerce')
    else:
        parsed_dt = None
        for c in df.columns:
            if df[c].dtype == object:
                try:
                    tmp = pd.to_datetime(df[c], errors='coerce')
                    if tmp.notna().sum() >= 3:
                        parsed_dt = tmp
                        break
                except Exception:
                    continue
        if parsed_dt is not None:
            df["DateTime"] = parsed_dt
        else:
            df["DateTime"] = pd.NaT

    # ensure Type column exists (debit/credit)
    if "Type" not in df.columns:
        def infer_type(x):
            try:
                if pd.isna(x):
                    return ""
                if float(x) < 0:
                    return "debit"
                return "credit"
            except Exception:
                return ""
        df["Type"] = df["Amount"].apply(infer_type)
    else:
        df["Type"] = df["Type"].astype(str)

    # Suspicious detection
    try:
        threshold = 1_000_000
        df["Suspicious"] = df["Amount"].apply(lambda x: 1 if (pd.isna(x) or abs(x) >= threshold) else 0)
    except Exception:
        df["Suspicious"] = 0

    # drop rows with no datetime and no amount
    df = df.loc[~(df["DateTime"].isna() & df["Amount"].isna())].reset_index(drop=True)

    df["Type"] = df["Type"].str.lower()

    return df

# ---------------- Main execution ----------------
if not SHEET_ID:
    st.info("Enter your Google Sheet ID to load data (the long ID between /d/ and /edit in the URL).")
    st.stop()

with st.spinner("🔄 Fetching data from Google Sheets..."):
    try:
        creds_info = None
        if "gcp_service_account" in st.secrets:
            creds_info = parse_service_account_secret(st.secrets["gcp_service_account"])
        df_raw = read_google_sheet(SHEET_ID, RANGE, creds_info=creds_info, creds_file=CREDS_FILE)
    except Exception as e:
        st.error(f"❌ Failed to read Google Sheet: {e}")
        st.stop()

if df_raw.empty:
    st.warning("⚠️ No data returned. Check the sheet name/range and ensure the service account has viewer access.")
    st.stop()

# ---------- Clean the raw DataFrame ----------
@st.cache_data(ttl=300)
def _clean_cached(df_raw_in):
    return clean_history_transactions(df_raw_in)

try:
    with st.spinner("Cleaning data..."):
        cleaned_df = _clean_cached(df_raw)
except Exception as e:
    st.error(f"Cleaning failed: {e}")
    st.stop()

# ---------- Show basic counts & KPIs ----------
rows_read = len(df_raw)
st.success(f"✅ Successfully loaded data from Google Sheet — {rows_read:,} rows read.")

def safe_sum_by_type(df_in, match_str):
    try:
        return df_in.loc[df_in["Type"].str.lower().str.contains(match_str, na=False), "Amount"].sum()
    except Exception:
        return 0.0

total_debit = safe_sum_by_type(cleaned_df, "debit")
total_credit = safe_sum_by_type(cleaned_df, "credit")

try:
    suspicious_count = int(cleaned_df["Suspicious"].sum())
except Exception:
    suspicious_count = 0

col1, col2, col3 = st.columns([1,1,1])
col1.metric("Total Debit", f"{total_debit:,.2f}")
col2.metric("Total Credit", f"{total_credit:,.2f}")
col3.metric("Suspicious", f"{suspicious_count:,}")

# show small secondary info
colA, colB = st.columns([1,3])
colA.write(f"Rows read: {rows_read:,}")
colB.write(f"Columns: {', '.join(cleaned_df.columns.astype(str))}")

# Download cleaned CSV
clean_csv = cleaned_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Cleaned CSV", data=clean_csv, file_name="history_transactions_cleaned.csv", mime="text/csv")

# ---------------- Daily aggregation & improved diagnostics & fallback ----------------
st.subheader("Debugging & aggregation (auto-diagnostics)")

# 1) Ensure Amount numeric (cleaner might have done this)
if 'Amount' in cleaned_df.columns:
    cleaned_df['Amount'] = pd.to_numeric(cleaned_df['Amount'].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce')
else:
    cleaned_df['Amount'] = pd.NA

st.write("Amount summary:")
try:
    st.write(cleaned_df['Amount'].describe())
except Exception:
    st.write("No Amount column or not numeric")

# 2) Ensure DateTime exists — try common alternate column names if missing
date_candidates = ['DateTime', 'Datetime', 'Timestamp', 'Date', 'Txn Date', 'TxnDate', 'timestamp', 'date']
if 'DateTime' in cleaned_df.columns:
    cleaned_df['DateTime'] = pd.to_datetime(cleaned_df['DateTime'], errors='coerce')
else:
    for cand in date_candidates:
        if cand in cleaned_df.columns:
            cleaned_df['DateTime'] = pd.to_datetime(cleaned_df[cand], errors='coerce', dayfirst=False)
            st.write(f"Parsed DateTime from column `{cand}`")
            break
# show how many parsed
if 'DateTime' in cleaned_df.columns:
    st.write("DateTime non-null count:", int(cleaned_df['DateTime'].notna().sum()), "null count:", int(cleaned_df['DateTime'].isna().sum()))
else:
    st.write("No DateTime column found after attempts.")

# 3) Normalize and map Type values (common bank notations)
if 'Type' in cleaned_df.columns:
    cleaned_df['Type'] = cleaned_df['Type'].astype(str).str.strip().str.lower()
    mapping = {
        'dr':'debit','d':'debit','debit':'debit','withdrawal':'debit','debited':'debit',
        'cr':'credit','c':'credit','credit':'credit','deposit':'credit','credited':'credit'
    }
    cleaned_df['Type'] = cleaned_df['Type'].map(lambda x: mapping.get(x, x))
else:
    cleaned_df['Type'] = ''

st.write("Type value counts (top):")
try:
    st.write(cleaned_df['Type'].value_counts(dropna=False).head(20))
except Exception:
    st.write("No Type counts available")

# 4) If Type is missing or mostly empty, infer from Amount sign
if cleaned_df['Type'].astype(str).str.strip().eq('').sum() >= len(cleaned_df) * 0.5:
    st.write("Inferring Type from Amount sign (many missing/unknown Type values).")
    cleaned_df['Type'] = cleaned_df['Amount'].apply(lambda v: 'debit' if pd.notna(v) and v < 0 else ('credit' if pd.notna(v) and v > 0 else ''))

# 5) Build df_debit and df_credit
df_debit = cleaned_df[cleaned_df['Type'].astype(str).str.lower() == 'debit'].copy()
df_credit = cleaned_df[cleaned_df['Type'].astype(str).str.lower() == 'credit'].copy()
st.write("Rows found — debit:", df_debit.shape[0], "credit:", df_credit.shape[0])

# 6) Group into daily sums using the filtered frames (defensive)
def compute_daily(df_frame, amount_col='Amount', dt_col='DateTime', out_col='Total'):
    if df_frame.empty or dt_col not in df_frame.columns:
        return pd.DataFrame(columns=['Date', out_col])
    try:
        daily = df_frame.groupby(df_frame[dt_col].dt.date)[amount_col].sum().reset_index()
        daily.columns = ['Date', out_col]
        return daily
    except Exception:
        return pd.DataFrame(columns=['Date', out_col])

daily_spend = compute_daily(df_debit, amount_col='Amount', dt_col='DateTime', out_col='Total_Spent')
daily_credit = compute_daily(df_credit, amount_col='Amount', dt_col='DateTime', out_col='Total_Credit')

st.write("Daily spend sample:", daily_spend.head())
st.write("Daily credit sample:", daily_credit.head())

# 7) Merge; if empty, fallback to overall daily sum (ignore Type)
merged = pd.DataFrame()
if not daily_spend.empty or not daily_credit.empty:
    merged = pd.merge(daily_spend, daily_credit, on='Date', how='outer').fillna(0)
    try:
        merged['Date'] = pd.to_datetime(merged['Date'])
    except Exception:
        pass
    merged = merged.sort_values('Date').reset_index(drop=True)

# fallback: if merged is empty or has no rows with nonzero totals, compute overall daily totals (ignoring Type)
need_fallback = merged.empty or (merged[['Total_Spent','Total_Credit']].sum(axis=1).abs().sum() == 0)
if need_fallback:
    st.write("Merged is empty or zero — computing overall daily totals (ignoring Type) as a fallback.")
    try:
        overall = cleaned_df.dropna(subset=['DateTime']).groupby(cleaned_df['DateTime'].dt.date)['Amount'].sum().reset_index()
        overall.columns = ['Date', 'Total_Spent']
        overall['Total_Credit'] = 0.0
        overall['Date'] = pd.to_datetime(overall['Date'])
        overall = overall.sort_values('Date').reset_index(drop=True)
        merged = overall
        st.write("Overall daily totals (fallback) sample:", merged.head())
    except Exception as e:
        st.write("Could not compute overall fallback daily totals:", e)
        merged = pd.DataFrame(columns=['Date', 'Total_Spent', 'Total_Credit'])

# final diagnostics
st.write("Merged (top 15):")
st.write(merged.head(15))
st.write("Merged shape:", merged.shape)
if not merged.empty:
    st.write("Merged describe:", merged[['Total_Spent','Total_Credit']].describe().applymap(lambda x: float(x) if not pd.isna(x) else x))
    st.write("Total_Spent diffs (first 10):", merged['Total_Spent'].diff().fillna(merged['Total_Spent']).head(10))
