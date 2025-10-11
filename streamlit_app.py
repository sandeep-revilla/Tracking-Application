# streamlit_app.py
"""
Streamlit helper: connect to a private Google Sheet via service account (st.secrets or JSON file),
safely convert sheet values to a pandas DataFrame (handles uneven rows and missing header),
show preview and provide raw CSV download.

Keeps responsibility single-purpose: connection only. Cleaning and charts should be in other files.
"""

import streamlit as st
import pandas as pd
import json
import os
from typing import List, Tuple, Optional, Any, Dict

# Google Sheets imports
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

st.set_page_config(page_title="Sheet Connector (minimal)", layout="wide")
st.title("🔐 Google Sheet Connector — (connection only)")

# ---------------- Sidebar: user inputs ----------------
st.sidebar.header("Data source")
source = st.sidebar.selectbox("Source", ["Google Sheet (service account)", "Local Excel (fallback)"])
SHEET_ID = st.sidebar.text_input("Google Sheet ID (between /d/ and /edit)", value="")
RANGE = st.sidebar.text_input("Range or sheet name (e.g. History Transactions or Sheet1!A1:Z1000)", value="History Transactions")
LOCAL_PATH = st.sidebar.text_input("Local Excel path (used only if fallback chosen)", value="/mnt/data/SMS received (2).xlsx")
st.sidebar.caption("If using Google Sheets, put service account JSON in st.secrets['gcp_service_account'] (or provide creds file path below).")
CREDS_FILE = st.sidebar.text_input("Service account JSON file (optional)", value="creds/service_account.json")
if st.sidebar.button("Refresh now"):
    st.experimental_rerun()

# ---------------- Helpers: parse service account JSON ----------------
def parse_service_account_secret(raw: Any) -> Dict:
    """
    Accepts dict or string (possibly with escaped newlines or surrounding triple quotes).
    Returns parsed dict suitable for service_account.Credentials.from_service_account_info.
    """
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    # strip triple-quote wrapper if user pasted that
    if (s.startswith('"""') and s.endswith('"""')) or (s.startswith("'''") and s.endswith("'''")):
        s = s[3:-3].strip()
    # try parsing several variants
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return json.loads(s.replace('\\n', '\n'))
    except Exception:
        pass
    try:
        return json.loads(s.replace('\n', '\\n'))
    except Exception as e:
        raise ValueError(f"Unable to parse service account JSON: {e}")

# ---------------- Build Sheets client ----------------
@st.cache_data(ttl=300)
def build_sheets_service_from_info(creds_info: Dict, scopes: Optional[List[str]] = None):
    scopes = scopes or ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
    service = build("sheets", "v4", credentials=creds, cache_discovery=False)
    return service

@st.cache_data(ttl=300)
def build_sheets_service_from_file(creds_file: str, scopes: Optional[List[str]] = None):
    if not os.path.exists(creds_file):
        raise FileNotFoundError(f"Credentials file not found: {creds_file}")
    scopes = scopes or ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = service_account.Credentials.from_service_account_file(creds_file, scopes=scopes)
    service = build("sheets", "v4", credentials=creds, cache_discovery=False)
    return service

# ---------------- Safe conversion helpers ----------------
def _normalize_rows(values: List[List[str]]) -> Tuple[List[str], List[List]]:
    """
    Ensure header exists, pad/trim rows to header length.
    Returns (header, data_rows).
    If header row looks empty or all 'Unnamed', treat first row as data and auto-generate headers.
    """
    if not values:
        return [], []

    # header is first row by default
    header_row = [str(x).strip() for x in values[0]]
    # detect 'bad' header (empty / unnamed)
    if all((h == "" or h.lower().startswith("unnamed") or h.lower().startswith("column") or h.lower().startswith("nan")) for h in header_row):
        # no header: generate column names using max row length
        max_cols = max(len(r) for r in values)
        header = [f"col_{i}" for i in range(max_cols)]
        data_rows = values  # everything is data
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
    """
    Convert sheet values (list-of-lists) into a pandas DataFrame safely.
    """
    if not values:
        return pd.DataFrame()
    header, rows = _normalize_rows(values)
    try:
        df = pd.DataFrame(rows, columns=header)
    except Exception:
        # fallback: construct DF without columns then set if sizes match
        df = pd.DataFrame(rows)
        if header and df.shape[1] == len(header):
            df.columns = header
    return df

# ---------------- Read functions ----------------
@st.cache_data(ttl=300)
def read_google_sheet(spreadsheet_id: str, range_name: str, creds_info: Optional[Dict] = None, creds_file: Optional[str] = None) -> pd.DataFrame:
    """
    Read a Google Sheet range and return a safe DataFrame.
    Provide either creds_info (dict) OR creds_file (path) to authenticate.
    """
    if creds_info is None and (creds_file is None or not os.path.exists(creds_file)):
        # try reading from st.secrets if available
        try:
            if "gcp_service_account" in st.secrets:
                creds_info = parse_service_account_secret(st.secrets["gcp_service_account"])
        except Exception as e:
            raise ValueError(f"Credentials not provided and st.secrets parsing failed: {e}")

    if creds_info is not None:
        service = build_sheets_service_from_info(creds_info)
    else:
        service = build_sheets_service_from_file(creds_file)

    try:
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
        values = result.get("values", [])
    except HttpError as e:
        # bubble up API errors with helpful message
        raise RuntimeError(f"Google Sheets API error: {e}")
    return values_to_dataframe(values)

def read_local_excel(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Read a local Excel file. If sheet_name provided, try that, else fallback to first sheet.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Local Excel file not found: {path}")
    try:
        if sheet_name:
            return pd.read_excel(path, sheet_name=sheet_name)
        else:
            return pd.read_excel(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read local Excel: {e}")

# ---------------- Main: load data based on user's choice ----------------
df = pd.DataFrame()  # final DataFrame

if source.startswith("Google"):
    if not SHEET_ID:
        st.info("Enter the Google Sheet ID in the sidebar (the long id between /d/ and /edit).")
        st.stop()

    with st.spinner("Connecting to Google Sheets..."):
        try:
            creds_info = None
            if "gcp_service_account" in st.secrets:
                try:
                    creds_info = parse_service_account_secret(st.secrets["gcp_service_account"])
                except Exception as e:
                    st.warning(f"Could not parse st.secrets['gcp_service_account']: {e}. Will try creds file if provided.")

            # prefer creds_info if available, else creds_file
            if creds_info is not None:
                df = read_google_sheet(SHEET_ID, RANGE, creds_info=creds_info, creds_file=None)
            elif CREDS_FILE and os.path.exists(CREDS_FILE):
                df = read_google_sheet(SHEET_ID, RANGE, creds_info=None, creds_file=CREDS_FILE)
            else:
                # Try to read using st.secrets only; read_google_sheet will raise if missing
                df = read_google_sheet(SHEET_ID, RANGE)
        except Exception as e:
            st.error(f"Failed to read Google Sheet: {e}")
            st.stop()

    if df.empty:
        st.info("No data returned from the sheet. Check the range and that the service account has Viewer access to the sheet.")
        st.stop()

else:
    # Local fallback selected
    with st.spinner("Reading local Excel file..."):
        # if user included sheet name like 'Sheet1!A1:Z', split and pick sheet name
        sheet_name = None
        if "!" in RANGE:
            sheet_name = RANGE.split("!")[0]
        try:
            df = read_local_excel(LOCAL_PATH, sheet_name=sheet_name)
        except Exception as e:
            st.error(f"Failed to read local Excel: {e}")
            st.stop()
    if df.empty:
        st.info("Local file read returned no data.")
        st.stop()

# ---------------- UI: preview and download ----------------
st.subheader("Loaded data (preview)")
st.dataframe(df.head(100), use_container_width=True)
st.caption(f"Rows loaded: {len(df)} — Columns: {', '.join(df.columns.astype(str))}")

# Provide raw CSV download for quick checks
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download raw CSV", data=csv_bytes, file_name="sheet_raw.csv", mime="text/csv")

# Provide a simple export to Excel option
try:
    with st.spinner("Preparing Excel file..."):
        excel_path = "sheet_raw.xlsx"
        df.to_excel(excel_path, index=False)
    st.download_button("Download raw Excel", data=open(excel_path, "rb"), file_name="sheet_raw.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    # remove temp file if desired (commented out to allow debug); os.remove(excel_path)
except Exception:
    # If writing Excel fails (no openpyxl installed), silently skip
    pass

st.info("This connector returns a raw DataFrame. Run your cleaning module (src/cleaning.py) to normalize types and extract fields before plotting.")
