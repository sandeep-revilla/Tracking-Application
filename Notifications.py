# Notifications.py — Google Sheet-backed persistent notification store
#
# Sheet layout (tab name: "Notifications"):
#   uid | timestamp | bank | amount | message | subtype | threshold | is_seen | created_at
#
# uid = stable hash of (transaction timestamp + bank + amount + message)
# is_seen = "false" / "true"  (string, for Sheets compatibility)

import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

NOTIF_SHEET = "Notifications"
NOTIF_COLS  = ["uid", "timestamp", "bank", "amount", "message", "subtype", "threshold", "is_seen", "created_at"]


# ─────────────────────────────────────────────────────────────
# UID helpers
# ─────────────────────────────────────────────────────────────

def make_uid(row: dict) -> str:
    """Stable hash from timestamp + bank + amount + message."""
    key = (
        f"{row.get('timestamp','')}|"
        f"{row.get('bank', row.get('Bank',''))}|"
        f"{row.get('amount', row.get('Amount',''))}|"
        f"{str(row.get('message', row.get('Message', row.get('description',''))))[:60]}"
    )
    return hashlib.md5(key.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────
# Google Sheets service builder
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


# ─────────────────────────────────────────────────────────────
# Ensure Notifications sheet exists with correct header
# ─────────────────────────────────────────────────────────────

def _ensure_notif_sheet(spreadsheet_id: str, service) -> bool:
    """
    - If Notifications tab does not exist → create it + write header
    - If tab exists but is empty → write header
    - If tab exists and has data → do nothing (preserve existing data)
    Returns True on success.
    """
    try:
        meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        existing_tabs = [s['properties']['title'] for s in meta.get('sheets', [])]

        if NOTIF_SHEET not in existing_tabs:
            service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={"requests": [{"addSheet": {"properties": {"title": NOTIF_SHEET}}}]}
            ).execute()
            service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=f"{NOTIF_SHEET}!A1",
                valueInputOption="USER_ENTERED",
                body={"values": [NOTIF_COLS]}
            ).execute()
            print(f"[notifications] Created '{NOTIF_SHEET}' tab and wrote header.")
        else:
            res = service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=f"{NOTIF_SHEET}!A1:Z1"
            ).execute()
            first_row = res.get("values", [])
            if not first_row or not any(str(c).strip() for c in first_row[0]):
                service.spreadsheets().values().update(
                    spreadsheetId=spreadsheet_id,
                    range=f"{NOTIF_SHEET}!A1",
                    valueInputOption="USER_ENTERED",
                    body={"values": [NOTIF_COLS]}
                ).execute()
                print(f"[notifications] '{NOTIF_SHEET}' tab was empty — wrote header.")

        return True

    except Exception as e:
        print(f"[notifications] _ensure_notif_sheet error: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Read all notifications
# ─────────────────────────────────────────────────────────────

def read_notifications(spreadsheet_id: str, creds_info=None, creds_file=None) -> pd.DataFrame:
    """
    Read all rows from the Notifications sheet.
    Returns empty DataFrame (with correct columns) on failure or empty sheet.
    """
    try:
        svc = _get_write_service(creds_info, creds_file)
        _ensure_notif_sheet(spreadsheet_id, svc)

        res    = svc.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=NOTIF_SHEET
        ).execute()
        values = res.get("values", [])

        if len(values) <= 1:
            return pd.DataFrame(columns=NOTIF_COLS)

        header = [str(c).strip() for c in values[0]]
        rows   = values[1:]
        rows   = [r + [''] * (len(header) - len(r)) for r in rows]

        return pd.DataFrame(rows, columns=header)

    except Exception as e:
        print(f"[notifications] read_notifications error: {e}")
        return pd.DataFrame(columns=NOTIF_COLS)


# ─────────────────────────────────────────────────────────────
# Write all notifications (full overwrite)
# ─────────────────────────────────────────────────────────────

def _write_all_notifications(spreadsheet_id: str, df: pd.DataFrame, svc) -> bool:
    """
    Overwrite the entire Notifications sheet with df (header + all rows).
    Ensures all NOTIF_COLS are present before writing.
    """
    try:
        for c in NOTIF_COLS:
            if c not in df.columns:
                df[c] = ''

        rows = [NOTIF_COLS] + df[NOTIF_COLS].fillna('').values.tolist()
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
# CORE: Sync large transactions → Notifications sheet
# ─────────────────────────────────────────────────────────────

def sync_large_transactions(
    transactions_df: pd.DataFrame,
    spreadsheet_id: str,
    threshold: float,
    creds_info=None,
    creds_file=None,
) -> Tuple[int, int]:
    """
    Scan ALL transactions for debits above threshold.
    - First run (empty sheet): loads every qualifying transaction with is_seen = false
    - Subsequent runs: only appends transactions whose UID is not already in the sheet
    - Never creates duplicates (deduplication via UID)
    Returns (new_added, total_unseen).
    """
    if transactions_df is None or transactions_df.empty:
        return 0, 0

    try:
        svc = _get_write_service(creds_info, creds_file)
        _ensure_notif_sheet(spreadsheet_id, svc)

        existing_df   = read_notifications(spreadsheet_id, creds_info, creds_file)
        existing_uids = set(existing_df['uid'].tolist()) if not existing_df.empty else set()

        df = transactions_df.copy()
        df['Amount'] = pd.to_numeric(df.get('Amount'), errors='coerce')

        type_col = next((c for c in df.columns if c.lower() == 'type'), None)
        if type_col:
            debit_mask = df[type_col].astype(str).str.lower().str.strip() == 'debit'
        else:
            debit_mask = pd.Series(True, index=df.index)

        flagged = df[debit_mask & (df['Amount'] > threshold)]

        if flagged.empty:
            total_unseen = int(
                (existing_df['is_seen'].astype(str).str.lower() == 'false').sum()
            ) if not existing_df.empty and 'is_seen' in existing_df.columns else 0
            return 0, total_unseen

        new_rows = []
        for _, row in flagged.iterrows():
            uid = make_uid(row.to_dict())
            if uid in existing_uids:
                continue

            ts     = pd.to_datetime(row.get('timestamp', ''), errors='coerce')
            ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(ts) else ''

            new_rows.append({
                'uid':        uid,
                'timestamp':  ts_str,
                'bank':       str(row.get('Bank',    row.get('bank',    'Unknown'))),
                'amount':     str(row.get('Amount',  '')),
                'message':    str(row.get('message', row.get('Message', row.get('description', ''))))[:120],
                'subtype':    str(row.get('Subtype', row.get('subtype', ''))),
                'threshold':  str(threshold),
                'is_seen':    'false',
                'created_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            })

        if new_rows:
            new_df   = pd.DataFrame(new_rows, columns=NOTIF_COLS)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            for c in NOTIF_COLS:
                if c not in combined.columns:
                    combined[c] = ''
            _write_all_notifications(spreadsheet_id, combined, svc)
            print(f"[notifications] Added {len(new_rows)} new notification(s) to sheet.")

        final_df     = read_notifications(spreadsheet_id, creds_info, creds_file)
        total_unseen = int(
            (final_df['is_seen'].astype(str).str.lower() == 'false').sum()
        ) if not final_df.empty and 'is_seen' in final_df.columns else 0

        return len(new_rows), total_unseen

    except Exception as e:
        print(f"[notifications] sync_large_transactions error: {e}")
        return 0, 0


# ─────────────────────────────────────────────────────────────
# NEW: Hard delete ALL notification rows (keeps header)
# ─────────────────────────────────────────────────────────────

def delete_all_notifications(
    spreadsheet_id: str,
    creds_info=None,
    creds_file=None,
) -> bool:
    """
    Hard delete — wipes every data row from the Notifications sheet.
    The header row is preserved so the sheet structure stays intact.
    Returns True on success.
    """
    try:
        svc = _get_write_service(creds_info, creds_file)
        _ensure_notif_sheet(spreadsheet_id, svc)

        # Read current sheet to find how many rows exist
        res    = svc.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=NOTIF_SHEET
        ).execute()
        values = res.get("values", [])

        if len(values) <= 1:
            # Nothing to delete — only header or completely empty
            print("[notifications] delete_all_notifications: sheet already empty.")
            return True

        # Overwrite with header only — this effectively deletes all data rows
        svc.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=NOTIF_SHEET,
            valueInputOption="USER_ENTERED",
            body={"values": [NOTIF_COLS]}
        ).execute()

        # Clear any remaining content below row 1 that the update may not cover
        # Get the sheet ID first
        meta     = svc.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        sheet_id = next(
            (s['properties']['sheetId'] for s in meta.get('sheets', [])
             if s['properties']['title'] == NOTIF_SHEET),
            None
        )

        if sheet_id is not None and len(values) > 1:
            # Delete all rows after row 1 (0-indexed: startIndex=1)
            svc.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={
                    "requests": [{
                        "deleteDimension": {
                            "range": {
                                "sheetId":    sheet_id,
                                "dimension":  "ROWS",
                                "startIndex": 1,          # row 2 onwards (0-indexed)
                                "endIndex":   len(values), # up to last data row
                            }
                        }
                    }]
                }
            ).execute()

        print(f"[notifications] Hard deleted {len(values) - 1} notification row(s).")
        return True

    except Exception as e:
        print(f"[notifications] delete_all_notifications error: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# NEW: Reload all notifications from scratch based on threshold
# ─────────────────────────────────────────────────────────────

def reload_all_notifications(
    transactions_df: pd.DataFrame,
    spreadsheet_id: str,
    threshold: float,
    creds_info=None,
    creds_file=None,
) -> Tuple[int, int]:
    """
    1. Hard delete ALL existing notification rows
    2. Re-scan ALL transactions for debits above threshold
    3. Write every qualifying transaction as is_seen = false
    This is used when the user wants a fresh load with a new threshold.
    Returns (total_loaded, total_unseen).
    """
    if transactions_df is None or transactions_df.empty:
        return 0, 0

    try:
        # Step 1: Hard delete all existing rows
        deleted_ok = delete_all_notifications(spreadsheet_id, creds_info, creds_file)
        if not deleted_ok:
            print("[notifications] reload_all_notifications: delete step failed.")
            return 0, 0

        # Step 2: Filter qualifying transactions
        df = transactions_df.copy()
        df['Amount'] = pd.to_numeric(df.get('Amount'), errors='coerce')

        type_col = next((c for c in df.columns if c.lower() == 'type'), None)
        if type_col:
            debit_mask = df[type_col].astype(str).str.lower().str.strip() == 'debit'
        else:
            debit_mask = pd.Series(True, index=df.index)

        flagged = df[debit_mask & (df['Amount'] > threshold)]

        if flagged.empty:
            print(f"[notifications] reload: no transactions above ₹{threshold}.")
            return 0, 0

        # Step 3: Build all rows fresh — all is_seen = false
        new_rows = []
        seen_uids = set()   # local dedupe within this batch

        for _, row in flagged.iterrows():
            uid = make_uid(row.to_dict())
            if uid in seen_uids:
                continue
            seen_uids.add(uid)

            ts     = pd.to_datetime(row.get('timestamp', ''), errors='coerce')
            ts_str = ts.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(ts) else ''

            new_rows.append({
                'uid':        uid,
                'timestamp':  ts_str,
                'bank':       str(row.get('Bank',    row.get('bank',    'Unknown'))),
                'amount':     str(row.get('Amount',  '')),
                'message':    str(row.get('message', row.get('Message', row.get('description', ''))))[:120],
                'subtype':    str(row.get('Subtype', row.get('subtype', ''))),
                'threshold':  str(threshold),
                'is_seen':    'false',
                'created_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            })

        if not new_rows:
            return 0, 0

        # Step 4: Write all new rows to sheet
        svc    = _get_write_service(creds_info, creds_file)
        new_df = pd.DataFrame(new_rows, columns=NOTIF_COLS)
        _write_all_notifications(spreadsheet_id, new_df, svc)

        total_loaded = len(new_rows)
        print(f"[notifications] Reloaded {total_loaded} notification(s) at threshold ₹{threshold}.")
        return total_loaded, total_loaded   # all newly loaded rows are unseen

    except Exception as e:
        print(f"[notifications] reload_all_notifications error: {e}")
        return 0, 0


# ─────────────────────────────────────────────────────────────
# Mark a single notification as seen
# ─────────────────────────────────────────────────────────────

def mark_notification_seen(
    spreadsheet_id: str,
    uid: str,
    creds_info=None,
    creds_file=None,
) -> bool:
    """
    Set is_seen = true for the notification matching uid.
    Called automatically when user opens/clicks a notification.
    Returns True on success.
    """
    try:
        svc = _get_write_service(creds_info, creds_file)
        df  = read_notifications(spreadsheet_id, creds_info, creds_file)

        if df.empty:
            return False

        if uid not in df['uid'].values:
            print(f"[notifications] UID not found: {uid}")
            return False

        df.loc[df['uid'] == uid, 'is_seen'] = 'true'
        return _write_all_notifications(spreadsheet_id, df, svc)

    except Exception as e:
        print(f"[notifications] mark_notification_seen error: {e}")
        return False


# Backward-compatible alias
mark_notification_read = mark_notification_seen


# ─────────────────────────────────────────────────────────────
# Mark ALL notifications as seen
# ─────────────────────────────────────────────────────────────

def mark_all_seen(
    spreadsheet_id: str,
    creds_info=None,
    creds_file=None,
) -> bool:
    """Set is_seen = true for every row in the Notifications sheet."""
    try:
        svc = _get_write_service(creds_info, creds_file)
        df  = read_notifications(spreadsheet_id, creds_info, creds_file)

        if df.empty:
            return True

        df['is_seen'] = 'true'
        return _write_all_notifications(spreadsheet_id, df, svc)

    except Exception as e:
        print(f"[notifications] mark_all_seen error: {e}")
        return False


# Backward-compatible alias
mark_all_read = mark_all_seen


# ─────────────────────────────────────────────────────────────
# Get unseen count
# ─────────────────────────────────────────────────────────────

def get_unseen_count(
    spreadsheet_id: str,
    creds_info=None,
    creds_file=None,
) -> int:
    """Return the number of notifications where is_seen = false."""
    try:
        df = read_notifications(spreadsheet_id, creds_info, creds_file)
        if df.empty or 'is_seen' not in df.columns:
            return 0
        return int((df['is_seen'].astype(str).str.lower() == 'false').sum())
    except Exception:
        return 0
