# io_helpers.py -- data I/O helpers (pure functions, no Streamlit runtime objects passed into cached functions)
import json
import os
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
import streamlit as st

# Optional Google Sheets imports (will raise only when used)
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except Exception:
    service_account = None
    build = None
    HttpError = Exception


def parse_service_account_secret(raw: Any) -> Dict:
    """Parse a service account JSON blob stored as dict or string. Returns a plain dict."""
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


def _normalize_rows(values: List[List[str]]) -> Tuple[List[str], List[List]]:
    """
    Convert Google Sheets 'values' (list of rows) into header + normalized rows.
    If first row looks like a header, use it; otherwise synthesize col_{i} headers.
    """
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


def values_to_dataframe(values: List[List[str]]) -> pd.DataFrame:
    """Turn a Google Sheets 'values' payload into a pandas DataFrame."""
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


def build_sheets_service_from_info(creds_info: Dict):
    """Create Google Sheets API service from service-account info dict."""
    if service_account is None or build is None:
        raise RuntimeError("google-auth or google-api-client not installed.")
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def build_sheets_service_from_file(creds_file: str):
    """Create Google Sheets API service from local service-account JSON file path."""
    if service_account is None or build is None:
        raise RuntimeError("google-auth or google-api-client not installed.")
    if not os.path.exists(creds_file):
        raise FileNotFoundError(f"Credentials file not found: {creds_file}")
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = service_account.Credentials.from_service_account_file(creds_file, scopes=scopes)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


@st.cache_data(ttl=600)
def read_google_sheet(spreadsheet_id: str, range_name: str,
                      creds_info: Optional[Dict] = None, creds_file: Optional[str] = None) -> pd.DataFrame:
    """
    Read a Google Sheet and return a pandas DataFrame.
    - creds_info: parsed JSON dict for service account (plain dict) OR None
    - creds_file: path to service account JSON on disk OR None

    IMPORTANT: Do NOT pass Streamlit runtime objects (e.g., st.secrets) to this function.
    Pass a plain dict for creds_info (use parse_service_account_secret to convert).
    This function is cached by Streamlit (ttl default 600 seconds).
    """
    # prefer explicit creds_info / creds_file; else raise
    if creds_info is None and (creds_file is None or not os.path.exists(creds_file)):
        raise ValueError("No credentials found. Provide creds_info (plain dict) or a valid creds_file path.")

    service = None
    if creds_info is not None:
        service = build_sheets_service_from_info(creds_info)
    else:
        service = build_sheets_service_from_file(creds_file)

    try:
        sheet = service.spreadsheets()
        res = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
        values = res.get("values", [])
    except HttpError as e:
        raise RuntimeError(f"Google Sheets API error: {e}")
    return values_to_dataframe(values)
