# notifications.py — Google Sheet-backed persistent notification store
#
# Sheet layout (tab name: "Notifications"):
#   uid | timestamp | bank | amount | message | subtype | threshold | is_read | created_at
#
# uid = stable hash of (transaction timestamp + bank + amount + message)
# is_read = "false" / "true"  (string, for Sheets compatibility)

import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

NOTIF_SHEET = "Notifications"
NOTIF_COLS  = ["uid", "timestamp", "bank", "amount", "message", "subtype", "threshold", "is_read", "created_at"]


# ─────────────────────────────────────────────────────────────
# UID helpers
# ─────────────────────────────────────────────────────────────

def make_uid(row: dict) -> str:
    """Stable hash from timestamp + bank + amount + message."""
    key = f"{row.get('timestamp','')}|{row.get('bank', row.get('Bank',''))}|{row.get('amount', row.get('Amount',''))}|{str(row.get('message', row.get('Message', row.get('description',''))))[:60]}"
    return hashlib.md5(key.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────
# Sheet I/O helpers (no @st.cache_data — caller caches if needed)
# ─────────────────────────────────────────────────────────────

def _get_write_service(creds_info=None, creds_file=None):
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError:
        raise RuntimeError("google-auth / google-api-client not installed.")

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    if creds_info:
        creds = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
    else:
        creds = service_account.Credentials.from_service_account_file(creds_file, scopes=scopes)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def _ensure_notif_sheet(spreadsheet_id: str, service) -> bool:
    """Create the Notifications tab + header row if it doesn't exist. Returns True on success."""
    try:
        meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        existing = [s['properties']['title'] for s in meta.get('sheets', [])]
        if NOTIF_SHEET not in existing:
            service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={"requests": [{"addSheet": {"properties": {"title": NOTIF_SHEET}}}]}
            ).execute()
            # Write header
            service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=f"{NOTIF_SHEET}!A1",
                valueInputOption="USER_ENTERED",
                body={"values": [NOTIF_COLS]}
            ).execute()
        return True
    except Exception as e:
        print(f"[notifications] _ensure_notif_sheet error: {e}")
        return False


def read_notifications(spreadsheet_id: str, creds_info=None, creds_file=None) -> pd.DataFrame:
    """Read all rows from the Notifications sheet. Returns empty DataFrame on failure."""
    try:
        svc = _get_write_service(creds_info, creds_file)
        _ensure_notif_sheet(spreadsheet_id, svc)
        res = svc.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id, range=NOTIF_SHEET
        ).execute()
        values = res.get("values", [])
        if len(values) <= 1:           # header only or empty
            return pd.DataFrame(columns=NOTIF_COLS)
        header = [str(c).strip() for c in values[0]]
        rows   = values[1:]
        # Pad short rows
        rows = [r + [''] * (len(header) - len(r)) for r in rows]
        df = pd.DataFrame(rows, columns=header)
        return df
    except Exception as e:
        print(f"[notifications] read_notifications error: {e}")
        return pd.DataFrame(columns=NOTIF_COLS)


def _write_all_notifications(spreadsheet_id: str, df: pd.DataFrame, svc) -> bool:
    """Overwrite the entire Notifications sheet with df (including header)."""
    try:
        rows = [NOTIF_COLS] + df[NOTIF_COLS].fillna('').values.tolist()
        # Convert every cell to string so Sheets API accepts it
        rows = [[str(c) for c in row] for row in rows]
        svc.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=NOTIF_SHEET,
            valueInputOption="USER_ENTERED",
            body={"values": rows}
        ).execute()
        return True
    except Exception as e:
        print(f"[notifications] _write_all_notifications error: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def sync_large_transactions(
    transactions_df: pd.DataFrame,
    spreadsheet_id: str,
    threshold: float,
    creds_info=None,
    creds_file=None,
) -> Tuple[int, int]:
    """
    Detect debits > threshold in transactions_df.
    For each one not already in the Notifications sheet, add it as is_read=false.
    Returns (new_added, total_unread).
    """
    if transactions_df is None or transactions_df.empty:
        return 0, 0

    try:
        svc = _get_write_service(creds_info, creds_file)
        _ensure_notif_sheet(spreadsheet_id, svc)

        existing_df = read_notifications(spreadsheet_id, creds_info, creds_file)
        existing_uids = set(existing_df['uid'].tolist()) if not existing_df.empty else set()

        # Filter large debits
        df = transactions_df.copy()
        df['Amount'] = pd.to_numeric(df.get('Amount'), errors='coerce')
        type_col = next((c for c in df.columns if c.lower() == 'type'), None)
        if type_col:
            debit_mask = df[type_col].astype(str).str.lower().str.strip() == 'debit'
        else:
            debit_mask = pd.Series(True, index=df.index)
        flagged = df[debit_mask & (df['Amount'] > threshold)]

        new_rows = []
        for _, row in flagged.iterrows():
            uid = make_uid(row.to_dict())
            if uid in existing_uids:
                continue
            ts  = pd.to_datetime(row.get('timestamp', ''), errors='coerce')
            ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(ts) else ''
            bank    = str(row.get('Bank', row.get('bank', 'Unknown')))
            amount  = str(row.get('Amount', ''))
            message = str(row.get('message', row.get('Message', row.get('description', ''))))[:120]
            subtype = str(row.get('Subtype', row.get('subtype', '')))
            new_rows.append({
                'uid':        uid,
                'timestamp':  ts_str,
                'bank':       bank,
                'amount':     amount,
                'message':    message,
                'subtype':    subtype,
                'threshold':  str(threshold),
                'is_read':    'false',
                'created_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            })

        if new_rows:
            new_df     = pd.DataFrame(new_rows, columns=NOTIF_COLS)
            combined   = pd.concat([existing_df, new_df], ignore_index=True)
            # Ensure all NOTIF_COLS present
            for c in NOTIF_COLS:
                if c not in combined.columns:
                    combined[c] = ''
            _write_all_notifications(spreadsheet_id, combined, svc)

        # Count total unread (re-read to include anything just added)
        final_df   = read_notifications(spreadsheet_id, creds_info, creds_file)
        total_unread = int((final_df['is_read'].astype(str).str.lower() == 'false').sum()) if not final_df.empty else 0
        return len(new_rows), total_unread

    except Exception as e:
        print(f"[notifications] sync_large_transactions error: {e}")
        return 0, 0


def mark_notification_read(
    spreadsheet_id: str,
    uid: str,
    creds_info=None,
    creds_file=None,
) -> bool:
    """Mark a single notification as read by uid. Returns True on success."""
    try:
        svc = _get_write_service(creds_info, creds_file)
        df  = read_notifications(spreadsheet_id, creds_info, creds_file)
        if df.empty:
            return False
        df.loc[df['uid'] == uid, 'is_read'] = 'true'
        return _write_all_notifications(spreadsheet_id, df, svc)
    except Exception as e:
        print(f"[notifications] mark_notification_read error: {e}")
        return False


def mark_all_read(
    spreadsheet_id: str,
    creds_info=None,
    creds_file=None,
) -> bool:
    """Mark all notifications as read."""
    try:
        svc = _get_write_service(creds_info, creds_file)
        df  = read_notifications(spreadsheet_id, creds_info, creds_file)
        if df.empty:
            return True
        df['is_read'] = 'true'
        return _write_all_notifications(spreadsheet_id, df, svc)
    except Exception as e:
        print(f"[notifications] mark_all_read error: {e}")
        return False


def get_unread_count(
    spreadsheet_id: str,
    creds_info=None,
    creds_file=None,
) -> int:
    """Return count of unread notifications."""
    try:
        df = read_notifications(spreadsheet_id, creds_info, creds_file)
        if df.empty:
            return 0
        return int((df['is_read'].astype(str).str.lower() == 'false').sum())
    except Exception:
        return 0
