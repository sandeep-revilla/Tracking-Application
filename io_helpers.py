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
    """Create Google Sheets API service from service-account info dict (read-only scope)."""
    if service_account is None or build is None:
        raise RuntimeError("google-auth or google-api-client not installed.")
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def build_sheets_service_from_file(creds_file: str):
    """Create Google Sheets API service from local service-account JSON file path (read-only scope)."""
    if service_account is None or build is None:
        raise RuntimeError("google-auth or google-api-client not installed.")
    if not os.path.exists(creds_file):
        raise FileNotFoundError(f"Credentials file not found: {creds_file}")
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = service_account.Credentials.from_service_account_file(creds_file, scopes=scopes)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def build_sheets_service_write_from_info(creds_info: Dict):
    """Create Google Sheets API service from service-account info dict (write access)."""
    if service_account is None or build is None:
        raise RuntimeError("google-auth or google-api-client not installed.")
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def build_sheets_service_write_from_file(creds_file: str):
    """Create Google Sheets API service from local service-account JSON file path (write access)."""
    if service_account is None or build is None:
        raise RuntimeError("google-auth or google-api-client not installed.")
    if not os.path.exists(creds_file):
        raise FileNotFoundError(f"Credentials file not found: {creds_file}")
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
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


# ---------------------------------------------------------------------
# New write helpers
# ---------------------------------------------------------------------

def _get_write_service(creds_info: Optional[Dict] = None, creds_file: Optional[str] = None):
    """
    Return a Google Sheets service with write access.
    Prefers creds_info (plain dict). If not provided, uses creds_file path.
    """
    if creds_info is None and (creds_file is None or not os.path.exists(creds_file)):
        raise ValueError("No credentials found for write operation. Provide creds_info or a valid creds_file path.")
    if creds_info is not None:
        return build_sheets_service_write_from_info(creds_info)
    else:
        return build_sheets_service_write_from_file(creds_file)


def mark_rows_deleted(spreadsheet_id: str, range_name: str,
                      creds_info: Optional[Dict] = None, creds_file: Optional[str] = None,
                      row_indices: Optional[List[int]] = None) -> Dict:
    """
    Soft-delete rows by setting / creating an 'is_deleted' column and marking the specified
    rows as TRUE.

    - spreadsheet_id, range_name: passed to Sheets API (range_name commonly a sheet name like 'History Transactions').
    - creds_info: plain dict from parse_service_account_secret OR creds_file: path on disk.
    - row_indices: list of integers (0-based) referring to the data rows (first data row after header is index 0).
      If None or empty -> nothing is changed.

    Returns: dict with status and count of rows updated.
    """
    if not row_indices:
        return {"status": "no-op", "updated": 0, "message": "No row indices provided."}

    service = _get_write_service(creds_info, creds_file)
    sheet = service.spreadsheets()

    try:
        res = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
        values = res.get("values", [])
    except HttpError as e:
        return {"status": "error", "message": f"Failed to read sheet: {e}"}

    # Normalize header + data rows
    header, data_rows = _normalize_rows(values)
    # If there was no header (empty sheet) treat as error
    if not header:
        return {"status": "error", "message": "Sheet appears empty, cannot mark rows deleted."}

    # Ensure 'is_deleted' column exists
    if 'is_deleted' not in [h.lower() for h in header]:
        header.append('is_deleted')
        for i in range(len(data_rows)):
            data_rows[i].append(None)
        is_deleted_idx = len(header) - 1
    else:
        # find index case-insensitively
        is_deleted_idx = next(i for i, h in enumerate(header) if h.lower() == 'is_deleted')

    updated = 0
    for idx in row_indices:
        if 0 <= idx < len(data_rows):
            # mark as TRUE (Google Sheets user-entered boolean)
            data_rows[idx][is_deleted_idx] = 'TRUE'
            updated += 1

    # Build values to write back (header row + data rows)
    out_values = [header] + data_rows

    # Use update to replace the range content
    try:
        sheet.values().update(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption='USER_ENTERED',
            body={'values': out_values}
        ).execute()
    except HttpError as e:
        return {"status": "error", "message": f"Failed to write sheet: {e}"}

    return {"status": "ok", "updated": updated}


def append_new_row(spreadsheet_id: str, range_name: str, new_row_dict: Dict[str, Any],
                   creds_info: Optional[Dict] = None, creds_file: Optional[str] = None) -> Dict:
    """
    Append a new row to the sheet.

    - new_row_dict: mapping of column name -> value for the new row. Keys are matched to existing header names
      case-sensitively. Missing header columns will be left blank. If the sheet has no header, a header will be
      created from new_row_dict.keys() (order is insertion order of dict).
    - Returns a dict with status and the appended row number when available.

    NOTE: This function will not modify non-mentioned columns; they remain blank.
    """
    service = _get_write_service(creds_info, creds_file)
    sheet = service.spreadsheets()

    try:
        res = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
        values = res.get("values", [])
    except HttpError as e:
        return {"status": "error", "message": f"Failed to read sheet: {e}"}

    header, data_rows = _normalize_rows(values)

    # If no header present, create header from new_row_dict keys
    if not header:
        header = list(new_row_dict.keys())
        # create initial empty data_rows (none)
        data_rows = []
        # Write header first (so append works predictably)
        try:
            sheet.values().update(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueInputOption='USER_ENTERED',
                body={'values': [header]}
            ).execute()
        except HttpError as e:
            return {"status": "error", "message": f"Failed to write header to empty sheet: {e}"}

    # Build row in header order
    row_out = []
    for col in header:
        # match keys case-sensitively; if not present try case-insensitive fallback
        if col in new_row_dict:
            v = new_row_dict[col]
        else:
            # case-insensitive fallback
            found = None
            for k in new_row_dict:
                if k.lower() == col.lower():
                    found = new_row_dict[k]
                    break
            v = found if found is not None else None
        # Convert pandas/numpy types or datetimes to strings so Sheets accepts them cleanly
        if v is None:
            row_out.append(None)
        else:
            # if pandas Timestamp / datetime -> convert to ISO
            try:
                import pandas as _pd
                if isinstance(v, (_pd.Timestamp,)):
                    row_out.append(str(v))
                    continue
            except Exception:
                pass
            try:
                from datetime import datetime, date
                if isinstance(v, (datetime, date)):
                    row_out.append(v.isoformat())
                    continue
            except Exception:
                pass
            row_out.append(v)

    # Append using Sheets API append
    try:
        append_res = sheet.values().append(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption='USER_ENTERED',
            insertDataOption='INSERT_ROWS',
            body={'values': [row_out]}
        ).execute()
    except HttpError as e:
        return {"status": "error", "message": f"Failed to append row: {e}"}

    # Attempt to extract updatedRange / tableRange info for the appended row index (best-effort)
    try:
        updated_range = append_res.get('updates', {}).get('updatedRange')
        # updatedRange looks like "Sheet1!A10:F10" -> we can try to parse the row number
        appended_row_number = None
        if updated_range and '!' in updated_range:
            rng = updated_range.split('!')[1]
            if ':' in rng:
                # final cell like A10:F10 -> parse last token
                last = rng.split(':')[-1]
                # strip letters -> get number
                import re
                m = re.search(r'(\d+)', last)
                if m:
                    appended_row_number = int(m.group(1))
    except Exception:
        appended_row_number = None

    return {"status": "ok", "appended_row_number": appended_row_number, "append_response": append_res}
