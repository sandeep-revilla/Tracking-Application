# streamlit_app.py
"""
Single-file Streamlit app (no charts) — Amounts as integers; ensure single date & timestamp:
- Connects to Google Sheets via service account (st.secrets or file)
- Converts sheet values to a pandas DataFrame
- Auto-converts columns to inferred types (dates / numeric)
- Converts amount-like column(s) to integer (rounded) using pandas nullable Int64
- Ensures exactly two derived columns:
    - `timestamp` : full datetime (pandas datetime64[ns])
    - `date`      : only the date part (python date objects)
  and removes any other columns named 'timestamp' or 'date' (case-insensitive).
- Shows top 10 rows and column data types
- Offers cleaned CSV download
"""

import streamlit as st
import pandas as pd
import json
import os
from typing import List, Tuple, Optional, Any, Dict
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ---------------- Page config ----------------
st.set_page_config(page_title="Google Sheet Connector — Single Date & Timestamp", layout="wide")
st.title("🔐 Google Sheet Connector — Amounts → Integer; single date & timestamp")

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

# ---------------- Column type conversion utility (Amounts -> Int64; single date/timestamp) ----------------
def convert_column_types_to_integer_with_single_date_and_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns:
      - parse columns with names containing date/time keywords to datetime
      - coerce amount-like columns to integer (rounded) using pandas 'Int64' dtype
      - create exactly two standardized columns:
          - `timestamp` (datetime64[ns])
          - `date`      (python.date objects)
      - remove any other original columns that look like date/time (case-insensitive),
        preserving only the canonical 'timestamp' and 'date' columns.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # Normalize column names (trim whitespace)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # keywords used to detect date/time-like columns
    date_keywords = ['date', 'time', 'timestamp', 'datetime', 'txn']
    num_keywords = ['amount', 'amt', 'value', 'total', 'balance', 'credit', 'debit', 'spent']

    # 1) Parse obvious date-like columns (including Unnamed: 0)
    for col in list(df.columns):
        lname = str(col).lower()
        if any(k in lname for k in date_keywords) or str(col).lower().startswith("unnamed"):
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=False)

    # 2) Coerce amount-like columns to nullable Int64
    amount_columns = []
    for col in list(df.columns):
        lname = str(col).lower()
        if any(k in lname for k in num_keywords):
            coerced = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce')
            coerced = coerced.round(0)
            df[col] = coerced.astype('Int64')
            amount_columns.append(col)

    # 3) Try coercing other object columns that look numeric (sample heuristic)
    for col in list(df.columns):
        if pd.api.types.is_object_dtype(df[col]):
            sample = df[col].astype(str).head(20).str.replace(r'[^\d\.\-]', '', regex=True)
            parsed = pd.to_numeric(sample, errors='coerce')
            if parsed.notna().sum() >= 3:
                coerced = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce')
                coerced = coerced.round(0)
                df[col] = coerced.astype('Int64')

    # 4) Standardize preferred amount column name => 'Amount' if possible
    preferred = None
    candidates = ['Amount', 'amount', 'total_spent', 'totalspent', 'total', 'txn amount', 'value', 'spent']
    for candidate in candidates:
        for col in df.columns:
            if str(col).lower() == str(candidate).lower():
                preferred = col
                break
        if preferred:
            break
    if not preferred and amount_columns:
        preferred = amount_columns[0]

    if preferred and preferred != 'Amount':
        if 'Amount' not in df.columns:
            df.rename(columns={preferred: 'Amount'}, inplace=True)
            preferred = 'Amount'

    # ensure Amount is Int64 if present
    if 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').round(0).astype('Int64')

    # ---------------- determine primary datetime column ----------------
    primary_dt_col = None
    # 1) any column already datetime dtype with non-null values
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) and df[col].notna().sum() > 0:
            primary_dt_col = col
            break
    # 2) look for common date-like column names and parse if necessary
    if primary_dt_col is None:
        for col in df.columns:
            lname = str(col).lower()
            if any(k in lname for k in date_keywords) or str(col).lower().startswith("unnamed"):
                parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=False)
                if parsed.notna().sum() > 0:
                    df[col] = parsed
                    primary_dt_col = col
                    break
    # 3) try parsing object columns with many parseable dates
    if primary_dt_col is None:
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=False)
                if parsed.notna().sum() >= 3:
                    df[col] = parsed
                    primary_dt_col = col
                    break

    # ---------------- create canonical timestamp & date ----------------
    if primary_dt_col:
        timestamp_series = pd.to_datetime(df[primary_dt_col], errors='coerce')
    else:
        timestamp_series = pd.Series([pd.NaT] * len(df), index=df.index, dtype='datetime64[ns]')

    # Create canonical columns
    df['timestamp'] = timestamp_series
    # date as python date objects; preserve missing values as pd.NA
    try:
        date_series = timestamp_series.dt.date
        date_series = date_series.where(pd.notna(timestamp_series), pd.NA)
        df['date'] = date_series
    except Exception:
        df['date'] = pd.NA

    # ---------------- remove other original date/time-like columns ----------------
    # Build a list of columns to drop: any column (original name) that looks date/time-like
    # but skip the canonical 'timestamp' and 'date' we just created.
    cols_to_drop = []
    for col in list(df.columns):
        low = str(col).lower()
        # skip canonical columns
        if low in ('timestamp', 'date'):
            continue
        # if column name contains a date keyword, mark for drop
        if any(k in low for k in date_keywords):
            cols_to_drop.append(col)

    # Drop them (conservative: only drop if they exist)
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)

    # Ensure there is exactly one 'timestamp' and one 'date' column (drop case-variants if any)
    for col in list(df.columns):
        if col not in ('timestamp','date') and str(col).lower() in ('timestamp','date'):
            df.drop(columns=[col], inplace=True)

    # Reorder: put canonical timestamp & date first
    cols = list(df.columns)
    final_cols = []
    if 'timestamp' in cols:
        final_cols.append('timestamp')
    if 'date' in cols:
        final_cols.append('date')
    for c in cols:
        if c not in final_cols:
            final_cols.append(c)
    df = df[final_cols]

    return df

# ---------------- Main execution ----------------
if not SHEET_ID:
    st.info("Enter your Google Sheet ID (the long ID between /d/ and /edit in the URL).")
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

st.success(f"✅ Loaded data — {len(df_raw):,} rows, {df_raw.shape[1]} columns.")

# Convert column types (amounts -> Int64), and ensure single date/timestamp columns
converted_df = convert_column_types_to_integer_with_single_date_and_timestamp(df_raw)

# Show top 10 rows and data types
st.subheader("Top 10 rows (after type conversion)")
st.write(converted_df.head(10))

st.subheader("Column data types")
dt_df = pd.DataFrame({
    "column": converted_df.columns.astype(str),
    "dtype": [str(converted_df[c].dtype) for c in converted_df.columns]
})
st.write(dt_df)

# If an 'Amount' column exists, show a small summary and counts of non-null
if 'Amount' in converted_df.columns:
    st.subheader("Amount summary (Integer, nullable)")
    amt = converted_df['Amount']
    st.write({
        "non_null_count": int(amt.notna().sum()),
        "min": int(amt.min()) if amt.notna().any() else None,
        "max": int(amt.max()) if amt.notna().any() else None,
        "mean": float(amt.dropna().astype(float).mean()) if amt.notna().any() else None
    })

# Show quick info about the derived date/timestamp
st.subheader("Derived date/timestamp info")
if 'timestamp' in converted_df.columns:
    st.write("timestamp non-null count:", int(converted_df['timestamp'].notna().sum()))
if 'date' in converted_df.columns:
    st.write("date non-null count:", int(converted_df['date'].notna().sum()))
    st.write("date sample:", converted_df['date'].dropna().head(5).tolist())

# Download cleaned CSV
csv_bytes = converted_df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Converted CSV (Amounts as integer)", data=csv_bytes, file_name="sheet_converted_integer_amounts_with_single_date_timestamp.csv", mime="text/csv")
