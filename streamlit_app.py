# streamlit_app.py
"""
Single-file Streamlit app (no charts):
- Connects to Google Sheets via service account (st.secrets or file)
- Converts sheet values to a pandas DataFrame
- Auto-converts columns to inferred types (dates / numeric)
- Converts amount-like column(s) to numeric and rounds values to 2 decimals
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
st.set_page_config(page_title="Google Sheet Connector — Read & Type Convert", layout="wide")
st.title("🔐 Google Sheet Connector — Read & Convert Types (No Charts)")

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

# ---------------- Column type conversion utility ----------------
def convert_column_types(df: pd.DataFrame, round_decimals: int = 2) -> pd.DataFrame:
    """
    Heuristically convert columns:
      - parse columns with names containing date/time keywords to datetime
      - coerce amount-like columns to numeric and round to `round_decimals`
      - attempt to convert other numeric-looking columns
    Returns a new DataFrame with converted columns.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # Normalize column names (trim whitespace)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    date_keywords = ['date', 'time', 'timestamp', 'datetime', 'txn']
    num_keywords = ['amount', 'amt', 'value', 'total', 'balance', 'credit', 'debit']

    # First pass: convert obvious date columns (including Unnamed: 0)
    for col in df.columns:
        lname = str(col).lower()
        if any(k in lname for k in date_keywords) or str(col).lower().startswith("unnamed"):
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=False)

    # Second pass: numeric/amount columns
    amount_columns = []
    for col in df.columns:
        lname = str(col).lower()
        if any(k in lname for k in num_keywords):
            # strip non-digit characters and coerce
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce')
            df[col] = df[col].round(round_decimals)
            amount_columns.append(col)

    # Third pass: any remaining object columns that look numeric (try coercion)
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            sample = df[col].astype(str).head(20).str.replace(r'[^\d\.\-]', '', regex=True)
            # if sample has at least 3 parseable numbers, coerce whole column
            parsed = pd.to_numeric(sample, errors='coerce')
            if parsed.notna().sum() >= 3:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce').round(round_decimals)

    # If an Amount-like column name exists (prefer exact 'Amount' else choose first detected), normalize to 'Amount'
    preferred = None
    for candidate in ['Amount', 'amount', 'total_spent', 'totalspent', 'total', 'txn amount', 'value']:
        for col in df.columns:
            if str(col).lower() == str(candidate).lower():
                preferred = col
                break
        if preferred:
            break
    if not preferred and amount_columns:
        preferred = amount_columns[0]

    if preferred and preferred != 'Amount':
        # rename preserved original column to standardized 'Amount' (only if 'Amount' not present)
        if 'Amount' not in df.columns:
            df.rename(columns={preferred: 'Amount'}, inplace=True)
            preferred = 'Amount'

    # Final: return df with converted types
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

st.success(f"✅ Loaded data — {len(df_raw):,} rows, {df_raw.shape[1]} columns.")

# Convert column types and round amount-like columns
converted_df = convert_column_types(df_raw, round_decimals=2)

# Show top 10 rows and data types
st.subheader("Top 10 rows (after type conversion)")
st.write(converted_df.head(10))

st.subheader("Column data types")
# present dtypes in a neat table
dt_df = pd.DataFrame({
    "column": converted_df.columns.astype(str),
    "dtype": [str(converted_df[c].dtype) for c in converted_df.columns]
})
st.write(dt_df)

# If an 'Amount' column exists, show a small summary (and round-up was already applied)
if 'Amount' in converted_df.columns:
    st.subheader("Amount summary")
    st.write(converted_df['Amount'].describe().apply(lambda x: float(x) if pd.notna(x) else x))

# Download cleaned CSV (converted types). Numeric/date columns will be represented in CSV accordingly.
csv_bytes = converted_df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Converted CSV", data=csv_bytes, file_name="sheet_converted.csv", mime="text/csv")
