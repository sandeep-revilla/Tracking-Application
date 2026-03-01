# streamlit_app.py
import streamlit as st
import pandas as pd
import importlib
from datetime import datetime, timedelta, date

st.set_page_config(page_title="Daily Spend", layout="wide")

# ─────────────────────────────────────────────
# Module imports
# ─────────────────────────────────────────────
try:
    transform = importlib.import_module("transform")
except Exception as e:
    st.error("transform.py missing or failing to import.")
    st.exception(e)
    st.stop()

try:
    import io_helpers as io_mod
except Exception:
    io_mod = None

try:
    charts_mod = importlib.import_module("charts")
except Exception:
    charts_mod = None

try:
    import notifications as notif_mod
except Exception:
    notif_mod = None

# ─────────────────────────────────────────────
# Secrets
# ─────────────────────────────────────────────
_secrets             = getattr(st, "secrets", {}) or {}
SHEET_ID_SECRET      = _secrets.get("SHEET_ID")
RANGE_SECRET         = _secrets.get("RANGE")
APPEND_RANGE_SECRET  = _secrets.get("APPEND_RANGE")
CREDS_FILE_SECRET    = _secrets.get("CREDS_FILE")
BALANCE_RANGE_SECRET = _secrets.get("BALANCE_RANGE", "Balances")

# ─────────────────────────────────────────────
# Sidebar: Data source
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Data Source & Settings")
    data_source = st.radio(
        "Load data from",
        ["Google Sheet", "Upload CSV/XLSX", "Use sample data"],
        index=0, key="data_source_radio",
    )
    use_google = data_source.lower().startswith("google")

    if use_google:
        SHEET_ID      = SHEET_ID_SECRET or st.text_input("Google Sheet ID (between /d/ and /edit)", value="")
        RANGE         = RANGE_SECRET    or st.text_input("History sheet name or range", value="History Transactions")
        APPEND_RANGE  = APPEND_RANGE_SECRET or st.text_input("Append sheet name or range", value="Append Transactions")
        BALANCE_RANGE = st.text_input("Balance sheet name or range", value=BALANCE_RANGE_SECRET)
        CREDS_FILE    = CREDS_FILE_SECRET or st.text_input("Service Account JSON File (optional)", value="creds/service_account.json")
        if SHEET_ID_SECRET:
            st.caption("Using Sheet ID from secrets.")
    else:
        SHEET_ID = RANGE = APPEND_RANGE = CREDS_FILE = BALANCE_RANGE = None

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")

# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────
def _get_creds_info():
    if io_mod is None:
        return None
    try:
        if hasattr(st, "secrets") and st.secrets and "gcp_service_account" in st.secrets:
            return io_mod.parse_service_account_secret(st.secrets["gcp_service_account"])
    except Exception:
        pass
    return None


@st.cache_data(ttl=600)
def _read_sheet_with_index(spreadsheet_id, range_name, source_name, creds_info, creds_file):
    try:
        df = io_mod.read_google_sheet(spreadsheet_id, range_name, creds_info=creds_info, creds_file=creds_file)
    except Exception as e:
        st.error(f"Failed to read Google Sheet '{range_name}': {e}")
        return pd.DataFrame()
    if df is None:
        return pd.DataFrame()
    df = df.reset_index(drop=True)
    if not df.empty and source_name != 'balance':
        df['_sheet_row_idx'] = df.index.astype(int)
        df['_source_sheet']  = source_name
    return df


def _to_pydate(val):
    if val is None: return None
    if isinstance(val, date) and not isinstance(val, datetime): return val
    if isinstance(val, datetime): return val.date()
    try:
        ts = pd.to_datetime(val, errors="coerce")
        return None if pd.isna(ts) else ts.date()
    except Exception: return None


def _ensure_min_max_order(min_d, max_d):
    min_d = _to_pydate(min_d) or datetime.utcnow().date()
    max_d = _to_pydate(max_d) or datetime.utcnow().date()
    return (max_d, min_d) if min_d > max_d else (min_d, max_d)


def _format_currency(v):
    return f"₹{v:,.2f}" if v is not None else "N/A"


# ─────────────────────────────────────────────
# Bank detection
# ─────────────────────────────────────────────
BANK_MAP = {
    'hdfc':        'HDFC Bank',
    'indian bank': 'Indian Bank',
    'indianbank':  'Indian Bank',
    'sbi':         'SBI',
    'icici':       'ICICI Bank',
    'axis':        'Axis Bank',
}


def add_bank_column(df: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    df = df.copy()
    if 'Bank' in df.columns and not overwrite and not df['Bank'].isna().all():
        df['Bank'] = df['Bank'].astype(str).str.strip().where(df['Bank'].notna(), 'Unknown')
        return df
    cand_cols = [c for c in ['bank', 'account', 'description', 'message', 'narration'] if c in df.columns]
    combined  = df[cand_cols].fillna('').astype(str).agg(' '.join, axis=1).str.lower()

    def _detect(text):
        for patt, name in BANK_MAP.items():
            if patt in text: return name
        return 'Unknown'

    df['Bank'] = combined.map(_detect)
    return df


# ─────────────────────────────────────────────
# Running balance (vectorized)
# ─────────────────────────────────────────────
def calculate_running_balance(transactions_df: pd.DataFrame, balance_df: pd.DataFrame) -> pd.DataFrame:
    transactions_df = transactions_df.reset_index(drop=True).reset_index().rename(columns={'index': '_temp_uid'})
    if balance_df.empty:
        transactions_df['Balance'] = pd.NA
        return transactions_df.drop(columns=['_temp_uid'])

    df = transactions_df.copy()
    df['Bank'] = df['Bank'].astype(str).str.strip()

    subtype_col = next((c for c in df.columns if c.lower() in ['subtype', 'sub_type', 'sub type']), None)
    df['subtype_norm'] = df[subtype_col].astype(str).str.lower().str.strip() if subtype_col else 'n/a'

    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.to_datetime(df.get('date'), errors='coerce')

    df = df.dropna(subset=['timestamp'])
    if df.empty:
        transactions_df['Balance'] = pd.NA
        return transactions_df.drop(columns=['_temp_uid'])

    df['Amount']       = pd.to_numeric(df['Amount'], errors='coerce').fillna(0.0)
    df['Type_norm']    = df['Type'].astype(str).str.lower()
    df['Signed_Amount']= df['Amount'].where(df['Type_norm'] == 'credit', -df['Amount'])

    bal_info = balance_df[['Bank', 'Start_Balance', 'Start_Date']].copy()
    bal_info['Bank']          = bal_info['Bank'].astype(str).str.strip()
    bal_info['Start_Balance'] = pd.to_numeric(bal_info['Start_Balance'], errors='coerce').fillna(0.0)
    bal_info['Start_Date']    = pd.to_datetime(bal_info['Start_Date'], errors='coerce').fillna(pd.Timestamp.min)

    df = df.sort_values(['timestamp', '_sheet_row_idx']).merge(bal_info, on='Bank', how='left')
    df['Start_Balance'] = df['Start_Balance'].fillna(0.0)
    df['Start_Date']    = df['Start_Date'].fillna(pd.Timestamp.min)

    skip_mask = (
        df['subtype_norm'].isin(['card', 'payzapp wallet']) |
        (df['timestamp'].dt.date < df['Start_Date'].dt.date)
    )
    df['Effective_Signed_Amount'] = df['Signed_Amount'].where(~skip_mask, 0.0)
    df['Running_Change']          = df.groupby('Bank')['Effective_Signed_Amount'].cumsum()
    df['Balance']                 = df['Start_Balance'] + df['Running_Change']

    return transactions_df.merge(df[['_temp_uid', 'Balance']], on='_temp_uid', how='left').drop(columns=['_temp_uid'])


# ─────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────
def load_from_upload(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None: return pd.DataFrame()
    try:
        return (pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith(".csv")
                else pd.read_excel(uploaded_file, engine="openpyxl"))
    except Exception as e:
        st.error(f"Failed to parse upload: {e}")
        return pd.DataFrame()


def sample_data():
    today = datetime.utcnow().date()
    rows = [
        {"timestamp": pd.to_datetime(today - timedelta(days=29 - i)),
         "description": f"Sample txn {i + 1}",
         "Amount": (i % 5 + 1) * 100, "Type": "credit" if i % 7 == 0 else "debit",
         "Bank": "HDFC Bank" if i % 2 == 0 else "Indian Bank",
         "Subtype": "card" if i % 5 == 0 else "bank_transfer"}
        for i in range(30)
    ]
    rows += [
        {"timestamp": pd.to_datetime(today - timedelta(days=2)),
         "description": "Accidental large payment", "Amount": 2500,
         "Type": "debit", "Bank": "HDFC Bank", "Subtype": "upi"},
        {"timestamp": pd.to_datetime(today - timedelta(days=5)),
         "description": "Vendor overpayment", "Amount": 1200,
         "Type": "debit", "Bank": "Indian Bank", "Subtype": "bank_transfer"},
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Load raw data
# ─────────────────────────────────────────────
uploaded = None
if data_source == "Upload CSV/XLSX":
    uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"])

sheet_full_df = pd.DataFrame()
df_raw        = pd.DataFrame()
balance_df    = pd.DataFrame()

if use_google:
    if not SHEET_ID:
        st.warning("Enter Google Sheet ID in the sidebar, or add secrets.")
        st.stop()
    if io_mod is None:
        st.error("io_helpers.py needed for Google Sheets.")
        st.stop()
    with st.spinner("Fetching Google Sheets…"):
        creds_info  = _get_creds_info()
        history_df  = _read_sheet_with_index(SHEET_ID, RANGE,         "history", creds_info, CREDS_FILE)
        append_df   = _read_sheet_with_index(SHEET_ID, APPEND_RANGE,  "append",  creds_info, CREDS_FILE)
        balance_df  = _read_sheet_with_index(SHEET_ID, BALANCE_RANGE, "balance", creds_info, CREDS_FILE)

        if history_df.empty and append_df.empty:
            st.error(f"No data found in '{SHEET_ID}' ranges '{RANGE}' or '{APPEND_RANGE}'.")
            st.stop()

        sheet_full_df = pd.concat([history_df, append_df], ignore_index=True, sort=False)
        if '_sheet_row_idx' not in sheet_full_df.columns:
            sheet_full_df['_sheet_row_idx'] = sheet_full_df.index

        if 'is_deleted' in sheet_full_df.columns:
            deleted_mask = sheet_full_df['is_deleted'].astype(str).str.lower().isin(['true', 't', '1', 'yes'])
            df_raw = sheet_full_df[~deleted_mask].copy().reset_index(drop=True)
        else:
            df_raw = sheet_full_df.copy().reset_index(drop=True)

elif data_source == "Upload CSV/XLSX":
    df_raw = load_from_upload(uploaded)
    if df_raw.empty:
        st.info("Upload a file or select another data source.")
        st.stop()
else:
    df_raw = sample_data()
    balance_df = pd.DataFrame([
        {"Bank": "HDFC Bank",   "Start_Balance": 50000, "Start_Date": "2024-01-01"},
        {"Bank": "Indian Bank", "Start_Balance": 25000, "Start_Date": "2024-01-01"},
    ])

if df_raw.empty:
    st.warning("No data loaded.")
    st.stop()

# ─────────────────────────────────────────────
# Transform + balance
# ─────────────────────────────────────────────
with st.spinner("Cleaning data and calculating balances…"):
    converted_df              = transform.convert_columns_and_derives(df_raw.copy())
    converted_df              = add_bank_column(converted_df, overwrite=False)
    converted_df_with_balance = calculate_running_balance(converted_df, balance_df)

if 'timestamp' not in converted_df_with_balance.columns:
    converted_df_with_balance['timestamp'] = pd.to_datetime(
        converted_df_with_balance.get('date'), errors='coerce'
    )

# ─────────────────────────────────────────────
# Sidebar: Alert threshold
# ─────────────────────────────────────────────
with st.sidebar:
    with st.expander("⚠️ Large Transaction Alerts", expanded=True):
        alert_threshold = st.number_input(
            "Alert me for debits above (₹)",
            min_value=0.0, value=500.0, step=100.0, format="%.0f",
            key="alert_threshold",
        )

# ─────────────────────────────────────────────
# ── NOTIFICATION SYNC (Google Sheets) ────────
# Detect large transactions and persist to Notifications sheet.
# For sample / CSV mode we fall back to session state only.
# ─────────────────────────────────────────────
creds_info = _get_creds_info()

@st.cache_data(ttl=60)
def _cached_notif_df(sheet_id, _creds_info, creds_file):
    """Cache notification reads for 60 s to avoid hammering the API."""
    if notif_mod is None or not sheet_id:
        return pd.DataFrame()
    return notif_mod.read_notifications(sheet_id, creds_info=_creds_info, creds_file=creds_file)

# Sync new large transactions into the Notifications sheet (once per data load)
if use_google and notif_mod and SHEET_ID:
    with st.spinner("Syncing notifications…"):
        notif_mod.sync_large_transactions(
            converted_df_with_balance,
            spreadsheet_id=SHEET_ID,
            threshold=alert_threshold,
            creds_info=creds_info,
            creds_file=CREDS_FILE,
        )
    notif_df    = _cached_notif_df(SHEET_ID, creds_info, CREDS_FILE)
else:
    # Sample / CSV fallback: in-memory session state
    if 'sample_notif_df' not in st.session_state:
        flagged = transform.flag_large_transactions(converted_df_with_balance, threshold=alert_threshold)
        rows = []
        for _, row in flagged.iterrows():
            import hashlib
            uid = hashlib.md5(f"{row.get('timestamp')}|{row.get('Bank')}|{row.get('Amount')}".encode()).hexdigest()
            rows.append({
                'uid':       uid,
                'timestamp': str(row.get('timestamp', '')),
                'bank':      str(row.get('Bank', '')),
                'amount':    str(row.get('Amount', '')),
                'message':   str(row.get('message', row.get('Message', row.get('description', ''))))[:120],
                'subtype':   str(row.get('Subtype', '')),
                'threshold': str(alert_threshold),
                'is_read':   'false',
                'created_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            })
        st.session_state['sample_notif_df'] = pd.DataFrame(rows) if rows else pd.DataFrame()
    notif_df = st.session_state.get('sample_notif_df', pd.DataFrame())

# ─────────────────────────────────────────────
# ── BELL ICON HEADER ─────────────────────────
# ─────────────────────────────────────────────

# CHANGED: compute unread_count defensively from raw notif_df (contains all notifications)
unread_count = 0
if not notif_df.empty and 'is_read' in notif_df.columns:
    try:
        unread_count = int((notif_df['is_read'].astype(str).str.lower() == 'false').sum())
    except Exception:
        unread_count = 0
else:
    unread_count = 0

# Title row: app title left, bell right
title_col, bell_col = st.columns([8, 1])
with title_col:
    st.title("💳 Daily Spending")
with bell_col:
    bell_label = f"🔔 {unread_count}" if unread_count > 0 else "🔔"
    bell_style = "background:#dc3545;color:white;" if unread_count > 0 else "background:#e9ecef;color:#333;"
    # Render bell as an HTML badge + toggle button
    if st.button(
        bell_label,
        key="bell_btn",
        help=f"{unread_count} unread notification(s)" if unread_count > 0 else "No unread notifications",
    ):
        st.session_state['show_notif_panel'] = not st.session_state.get('show_notif_panel', False)

st.info(
    "ℹ️ **Disclaimer:** This data is for reference only and may be incomplete. "
    "Please cross-check with your official bank statements if you notice a large deviation."
)

# ─────────────────────────────────────────────
# ── NOTIFICATION PANEL (inline, same page) ───
# ─────────────────────────────────────────────
if st.session_state.get('show_notif_panel', False):
    st.markdown("---")

    panel_title_col, mark_all_col = st.columns([5, 2])
    with panel_title_col:
        st.markdown(
            f"#### 🔔 Notifications &nbsp;"
            f"<span style='background:#dc3545;color:white;padding:2px 10px;"
            f"border-radius:12px;font-size:14px'>{unread_count} unread</span>",
            unsafe_allow_html=True,
        )
    with mark_all_col:
        if unread_count > 0:
            if st.button("✅ Mark all as read", key="mark_all_read_btn"):
                if use_google and notif_mod and SHEET_ID:
                    notif_mod.mark_all_read(SHEET_ID, creds_info=creds_info, creds_file=CREDS_FILE)
                    st.cache_data.clear()
                else:
                    if 'sample_notif_df' in st.session_state:
                        st.session_state['sample_notif_df']['is_read'] = 'true'
                st.rerun()

    # CHANGED: Filter out read notifications permanently so that once marked read they don't re-appear
    if not notif_df.empty and 'is_read' in notif_df.columns:
        notif_df = notif_df[notif_df['is_read'].astype(str).str.lower() == 'false'].copy()

    if notif_df.empty:
        st.info("No new notifications 🎉")
    else:
        # Sort: unread first, then newest first (note: notif_df now only contains unread)
        display_notif = notif_df.copy()
        display_notif['_unread_sort'] = (
            display_notif['is_read'].astype(str).str.lower() == 'false'
        ).astype(int)
        display_notif = display_notif.sort_values(
            ['_unread_sort', 'created_at'], ascending=[False, False]
        ).reset_index(drop=True)

        # Track which notification is expanded (uid stored in session state)
        if '_expanded_notif_uid' not in st.session_state:
            st.session_state['_expanded_notif_uid'] = None

        for _, nrow in display_notif.iterrows():
            n_uid       = nrow.get('uid', '')
            n_ts        = nrow.get('timestamp', '')
            n_bank      = nrow.get('bank', '')
            n_amount    = float(nrow.get('amount', 0) or 0)
            n_msg       = str(nrow.get('message', ''))[:80]
            n_subtype   = nrow.get('subtype', '—')
            n_threshold = float(nrow.get('threshold', alert_threshold) or alert_threshold)
            n_created   = nrow.get('created_at', '')
            n_is_read   = str(nrow.get('is_read', 'false')).lower() == 'true'
            is_expanded = st.session_state['_expanded_notif_uid'] == n_uid

            # ── Row summary card ─────────────────────────────────────────
            bg_color = "#f8f9fa" if n_is_read else "#fff8f8"
            border   = "#dee2e6" if n_is_read else "#f5c2c7"
            dot      = "⚪" if n_is_read else "🔴"
            arrow    = "▲" if is_expanded else "▼"

            row_left, row_right = st.columns([8, 1])
            with row_left:
                st.markdown(
                    f"""<div style="background:{bg_color};border:1px solid {border};
                        border-radius:8px;padding:10px 14px;margin-bottom:2px;cursor:pointer;">
                        {dot} <b>₹{n_amount:,.0f}</b> &nbsp;·&nbsp;
                        <b>{n_bank}</b> &nbsp;·&nbsp; {n_ts[:16]}
                        <br><small style="color:#666">{n_msg}</small>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with row_right:
                # Toggle button — expand or collapse
                if st.button(arrow, key=f"toggle_{n_uid}", help="Click to expand / collapse"):
                    if is_expanded:
                        st.session_state['_expanded_notif_uid'] = None
                    else:
                        st.session_state['_expanded_notif_uid'] = n_uid
                        # Mark as read when opened
                        if not n_is_read:
                            if use_google and notif_mod and SHEET_ID:
                                notif_mod.mark_notification_read(
                                    SHEET_ID, n_uid,
                                    creds_info=creds_info, creds_file=CREDS_FILE,
                                )
                                st.cache_data.clear()
                            else:
                                if 'sample_notif_df' in st.session_state:
                                    st.session_state['sample_notif_df'].loc[
                                        st.session_state['sample_notif_df']['uid'] == n_uid,
                                        'is_read',
                                    ] = 'true'
                    st.rerun()

            # ── Inline detail card (shown only when expanded) ────────────
            if is_expanded:
                over_by      = n_amount - n_threshold
                status_color = "#6c757d" if n_is_read else "#dc3545"
                status_label = "✅ Verified" if n_is_read else "🔴 Unread"

                st.markdown(
                    f"""
                    <div style="background:#ffffff;border:1.5px solid #f5c2c7;border-radius:10px;
                         padding:20px 24px;margin:4px 0 12px 0;
                         box-shadow:0 2px 8px rgba(0,0,0,0.07);">

                      <div style="display:flex;justify-content:space-between;
                                  align-items:center;margin-bottom:14px;">
                        <span style="font-size:22px;font-weight:700;">₹{n_amount:,.0f}</span>
                        <span style="background:{status_color};color:white;padding:4px 12px;
                              border-radius:20px;font-size:13px;">{status_label}</span>
                      </div>

                      <table style="width:100%;border-collapse:collapse;font-size:14px;">
                        <tr>
                          <td style="color:#888;padding:5px 0;width:150px">🏦 Bank</td>
                          <td><b>{n_bank}</b></td>
                        </tr>
                        <tr>
                          <td style="color:#888;padding:5px 0">📅 Date & Time</td>
                          <td><b>{n_ts}</b></td>
                        </tr>
                        <tr>
                          <td style="color:#888;padding:5px 0">💬 Description</td>
                          <td>{n_msg}</td>
                        </tr>
                        <tr>
                          <td style="color:#888;padding:5px 0">🏷️ Subtype</td>
                          <td>{n_subtype}</td>
                        </tr>
                        <tr>
                          <td style="color:#888;padding:5px 0">⚠️ Threshold</td>
                          <td>₹{n_threshold:,.0f}
                            &nbsp;<span style="color:#dc3545;font-size:12px;">
                              (₹{over_by:,.0f} over)
                            </span>
                          </td>
                        </tr>
                        <tr>
                          <td style="color:#888;padding:5px 0">🕐 Flagged At</td>
                          <td>{n_created}</td>
                        </tr>
                      </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Mark as verified button (only if not already read)
                if not n_is_read:
                    btn_col, _ = st.columns([2, 5])
                    with btn_col:
                        if st.button(
                            "✅ Mark as Verified",
                            key=f"verify_inline_{n_uid}",
                            type="primary",
                            use_container_width=True,
                        ):
                            if use_google and notif_mod and SHEET_ID:
                                notif_mod.mark_notification_read(
                                    SHEET_ID, n_uid,
                                    creds_info=creds_info, creds_file=CREDS_FILE,
                                )
                                st.cache_data.clear()
                            else:
                                if 'sample_notif_df' in st.session_state:
                                    st.session_state['sample_notif_df'].loc[
                                        st.session_state['sample_notif_df']['uid'] == n_uid,
                                        'is_read',
                                    ] = 'true'
                            st.session_state['_expanded_notif_uid'] = None
                            st.rerun()

    st.markdown("---")

# ─────────────────────────────────────────────
# Current Balances
# ─────────────────────────────────────────────
st.subheader("🏦 Current Balances")

latest_balances_df = pd.DataFrame()
if not converted_df_with_balance.empty and not converted_df_with_balance['timestamp'].isnull().all():
    latest_balances_df = converted_df_with_balance.loc[
        converted_df_with_balance.groupby('Bank')['timestamp'].idxmax(skipna=True)
    ]

total_balance = 0.0
if not latest_balances_df.empty and 'Balance' in latest_balances_df.columns:
    banks_to_show = sorted(latest_balances_df['Bank'].unique())
    balance_cols  = st.columns(len(banks_to_show) + 1)
    for i, bank_name in enumerate(banks_to_show):
        row = latest_balances_df[latest_balances_df['Bank'] == bank_name].iloc[0]
        current_balance = row['Balance']
        with balance_cols[i]:
            if pd.notna(current_balance):
                total_balance += current_balance
                st.metric(f"{bank_name} Balance", f"₹{current_balance:,.0f}")
            else:
                st.metric(f"{bank_name} Balance", "N/A", "Check 'Balances' sheet")
    with balance_cols[-1]:
        st.metric("Total Balance", f"₹{total_balance:,.0f}", "All Accounts")
else:
    if use_google:
        st.info(f"Add a sheet named '{BALANCE_RANGE}' with 'Bank', 'Start_Balance', 'Start_Date'.")
    else:
        st.info("Balance tracking not available for this data source.")

st.markdown("---")

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# Sidebar: Chart & filter options
# ─────────────────────────────────────────────
with st.sidebar:
    with st.expander("📊 Chart & Metric Options", expanded=False):
        st.write("**Chart Display**")
        show_debit_chart  = st.checkbox("Show Debit on Chart",  value=True,  key="show_debit_chart")
        show_credit_chart = st.checkbox("Show Credit on Chart", value=True,  key="show_credit_chart")
        chart_type_select = st.selectbox(
            "Chart Type",
            [
                "Daily line",
                "Monthly bars",
                "Top categories (Top-N)",
                "Weekly heatmap",
                "Cumulative spend",
                "Debit vs Credit pie",
                "Bank breakdown",
                "Day-of-week pattern",
            ],
            index=0, key="chart_type_select",
        )

        st.write("**Chart Date Filter**")
        try:
            with st.spinner("Calculating overall totals…"):
                merged_all_totals = transform.compute_daily_totals(converted_df_with_balance.copy())
                if not merged_all_totals.empty:
                    merged_all_totals['Date'] = pd.to_datetime(merged_all_totals['Date']).dt.normalize()
                    all_years = sorted(merged_all_totals['Date'].dt.year.unique().tolist())
                else:
                    all_years = [datetime.utcnow().year]
        except Exception:
            all_years = [datetime.utcnow().year]
            merged_all_totals = pd.DataFrame()

        years_opts_chart = ['All'] + [str(y) for y in all_years]
        sel_year_chart   = st.selectbox("Chart Year", years_opts_chart, index=0, key="sel_year_chart")

        month_map_chart = {i: pd.Timestamp(1900, i, 1).strftime('%B') for i in range(1, 13)}
        if not merged_all_totals.empty:
            mf = merged_all_totals.copy()
            if sel_year_chart != 'All':
                mf = mf[mf['Date'].dt.year == int(sel_year_chart)]
            month_choices_chart = [month_map_chart[m] for m in sorted(mf['Date'].dt.month.unique().tolist())]
        else:
            month_choices_chart = list(month_map_chart.values())

        sel_months_chart = st.multiselect("Chart Month(s)", options=month_choices_chart, default=month_choices_chart, key="sel_months_chart")

        st.markdown("---")
        st.write("**Top-Right Metric Options**")
        metric_year_opts = [str(y) for y in all_years]
        default_metric_year_idx = len(metric_year_opts) - 1 if metric_year_opts else 0
        metric_year  = st.selectbox("Metric Year",  options=metric_year_opts, index=default_metric_year_idx, key="metric_year")
        metric_month = st.selectbox("Metric Month", options=list(month_map_chart.values()), index=datetime.utcnow().month - 1, key="metric_month")
        replace_outliers_checkbox = st.checkbox("Clean outliers for metric avg", value=True, key="replace_outliers_checkbox")

    with st.expander("🔍 Transaction Filters", expanded=True):
        st.write("**Filter Transactions By**")
        banks_available = sorted([b for b in converted_df_with_balance['Bank'].unique() if pd.notna(b)])
        sel_banks = st.multiselect("Bank(s)", options=banks_available, default=banks_available, key="sel_banks")
        sel_types = st.multiselect("Transaction Type(s)", options=["debit", "credit"], default=["debit", "credit"], key="sel_types")
        min_amount_filter = st.number_input("Amount >= (0 to disable)", min_value=0.0, value=0.0, step=100.0, format="%.2f", key="min_amount_filter")

        st.markdown("---")
        st.write("**Select Date Range for Table & Totals**")
        try:
            valid_dates_all = pd.to_datetime(
                converted_df_with_balance.get('timestamp', converted_df_with_balance.get('date')), errors='coerce'
            ).dropna()
            min_date_overall, max_date_overall = (
                _ensure_min_max_order(valid_dates_all.min(), valid_dates_all.max())
                if not valid_dates_all.empty
                else (datetime.utcnow().date() - timedelta(days=365), datetime.utcnow().date())
            )
        except Exception:
            max_date_overall = datetime.utcnow().date()
            min_date_overall = max_date_overall - timedelta(days=365)

        totals_mode = st.radio("Mode", ["Single date", "Date range"], index=0, key="totals_mode")
        if totals_mode == "Single date":
            today = datetime.utcnow().date()
            default_date  = max(min_date_overall, min(today, max_date_overall))
            selected_date = st.date_input("Pick date", value=default_date, min_value=min_date_overall, max_value=max_date_overall, key="selected_date")
            start_sel, end_sel = selected_date, selected_date
        else:
            dr = st.date_input("Pick start & end", value=(min_date_overall, max_date_overall), min_value=min_date_overall, max_value=max_date_overall, key="date_range_picker")
            s_raw, e_raw = (dr[0], dr[1]) if isinstance(dr, (tuple, list)) and len(dr) == 2 else (dr, dr)
            s = _to_pydate(s_raw) or min_date_overall
            e = _to_pydate(e_raw) or max_date_overall
            start_sel = max(min_date_overall, s)
            end_sel   = min(max_date_overall, e)
            if start_sel > end_sel:
                start_sel, end_sel = end_sel, start_sel

# ─────────────────────────────────────────────
# Apply core filters
# ─────────────────────────────────────────────
converted_df_filtered = converted_df_with_balance.copy()

if sel_banks:
    converted_df_filtered = converted_df_filtered[converted_df_filtered['Bank'].isin(sel_banks)]

type_col_name = next((c for c in converted_df_filtered.columns if c.lower() == 'type'), None)
if sel_types and type_col_name:
    converted_df_filtered = converted_df_filtered[
        converted_df_filtered[type_col_name].astype(str).str.lower().isin(sel_types)
    ]

if min_amount_filter > 0.0:
    amount_col_name = next((c for c in converted_df_filtered.columns if c.lower() == 'amount'), None)
    if amount_col_name:
        try:
            converted_df_filtered[amount_col_name] = pd.to_numeric(converted_df_filtered[amount_col_name], errors='coerce')
            converted_df_filtered = converted_df_filtered[converted_df_filtered[amount_col_name] >= min_amount_filter].copy()
        except Exception as e:
            st.warning(f"Could not apply amount filter: {e}")

# ─────────────────────────────────────────────
# Daily totals
# ─────────────────────────────────────────────
with st.spinner("Computing daily totals…"):
    merged = transform.compute_daily_totals(converted_df_filtered.copy())

# ─────────────────────────────────────────────
# Chart data prep
# ─────────────────────────────────────────────
plot_df = merged.copy() if merged is not None else pd.DataFrame()
if not plot_df.empty:
    plot_df['Date'] = pd.to_datetime(plot_df['Date']).dt.normalize()
    if sel_year_chart != 'All':
        plot_df = plot_df[plot_df['Date'].dt.year == int(sel_year_chart)]
    if sel_months_chart:
        inv_map = {v: k for k, v in month_map_chart.items()}
        sel_month_nums = [inv_map[m] for m in sel_months_chart if m in inv_map]
        if sel_month_nums:
            plot_df = plot_df[plot_df['Date'].dt.month.isin(sel_month_nums)]
    plot_df = plot_df.sort_values('Date').reset_index(drop=True)
    for col in ['Total_Spent', 'Total_Credit']:
        plot_df[col] = pd.to_numeric(plot_df.get(col, 0), errors='coerce').fillna(0.0)
else:
    plot_df = pd.DataFrame(columns=['Date', 'Total_Spent', 'Total_Credit'])

# ─────────────────────────────────────────────
# Top-right metric
# ─────────────────────────────────────────────
def _month_year_to_ints(y_str, m_name):
    try: y = int(y_str)
    except Exception: y = datetime.utcnow().year
    try: m = pd.to_datetime(m_name, format='%B').month
    except Exception: m = datetime.utcnow().month
    return y, m


def compute_month_avg_from_merged(mrg_df, yr, mo, replace_outliers=False):
    if mrg_df is None or mrg_df.empty: return None, 0, {}
    df  = mrg_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    dfm  = df[(df['Date'].dt.year == int(yr)) & (df['Date'].dt.month == int(mo))]
    vals = pd.to_numeric(dfm.get('Total_Spent', 0), errors='coerce').fillna(0.0)
    if dfm.empty: return None, 0, {}
    if len(vals) < 3 or not replace_outliers:
        return float(vals.mean()) if not vals.empty else None, len(vals), {"n": len(vals)}
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
    iqr = q3 - q1
    is_outlier = (vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)
    non_out = vals[~is_outlier]
    replacement = float(non_out.median()) if not non_out.empty else float(vals.median())
    vals_r = vals.copy(); vals_r[is_outlier] = replacement
    return float(vals_r.mean()), len(vals), {"outliers_replaced": int(is_outlier.sum()), "n": len(vals)}


try:
    metric_year_int, metric_month_int = _month_year_to_ints(metric_year, metric_month)
    metric_avg, _, _ = compute_month_avg_from_merged(merged_all_totals, metric_year_int, metric_month_int, replace_outliers_checkbox)
    prev_dt  = datetime(metric_year_int, metric_month_int, 1) - pd.DateOffset(months=1)
    prev_avg, _, _ = compute_month_avg_from_merged(merged_all_totals, prev_dt.year, prev_dt.month, replace_outliers_checkbox)
except Exception:
    metric_avg = prev_avg = None

col_a, col_b, col_c = st.columns([6, 2, 2])
with col_c:
    label       = pd.Timestamp(metric_year_int, metric_month_int, 1).strftime("%b-%y")
    metric_text = _format_currency(metric_avg)
    delta_html  = "<span style='font-size:14px;color:gray'>N/A</span>"
    if metric_avg is not None and prev_avg is not None:
        diff = metric_avg - prev_avg
        try: delta_label = f"{(diff / abs(prev_avg) * 100.0):+.1f}%" if abs(prev_avg) > 1e-9 else f"{diff:+.2f}"
        except Exception: delta_label = f"{diff:+.2f}"
        color = "red" if diff > 0 else ("green" if diff < 0 else "gray")
        arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "►")
        delta_html = f"<span style='font-size:14px;color:{color};font-weight:600'>{arrow} {delta_label}</span>"
    st.markdown(
        f"<div style='text-align:right;padding:8px 4px;'>"
        f"<div style='font-size:12px;color:#666;margin-bottom:2px'>{label}</div>"
        f"<div style='font-size:20px;font-weight:700'>{metric_text}</div>"
        f"<div>{delta_html}</div></div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────
st.subheader("📊 Daily Spend and Credit")
if plot_df.empty:
    st.info("No data available for the selected chart filters.")
elif charts_mod is not None:
    series_selected = [s for s, show in [('Total_Spent', show_debit_chart), ('Total_Credit', show_credit_chart)] if show]
    try:
        charts_mod.render_chart(
            plot_df=plot_df, converted_df=converted_df_filtered,
            chart_type=chart_type_select, series_selected=series_selected, top_n=5,
        )
    except Exception as chart_err:
        st.error(f"Failed to render chart: {chart_err}")
        st.exception(chart_err)
else:
    st.info("charts.py not available.")

# ─────────────────────────────────────────────
# Rows view
# ─────────────────────────────────────────────
st.subheader("📝 Rows (matching selection)")

can_delete = use_google and io_mod is not None and not sheet_full_df.empty

if can_delete:
    st.caption("💡 Tick the checkbox on any row(s) below, then click **Delete selected**.")

rows_df = converted_df_filtered.copy()
if 'timestamp' in rows_df.columns:
    rows_df['timestamp'] = pd.to_datetime(rows_df['timestamp'], errors='coerce')
elif 'date' in rows_df.columns:
    rows_df['timestamp'] = pd.to_datetime(rows_df['date'], errors='coerce')
else:
    rows_df['timestamp'] = pd.NaT

if start_sel and end_sel and not rows_df['timestamp'].isnull().all():
    rows_df = rows_df[
        (rows_df['timestamp'].dt.date >= start_sel) &
        (rows_df['timestamp'].dt.date <= end_sel)
    ]

# ── Build display_df ──────────────────────────────────────────────────────
_desired     = ['timestamp', 'bank', 'type', 'amount', 'Balance', 'subtype', 'message']
col_map      = {c.lower(): c for c in rows_df.columns}
display_cols = [col_map[d] for d in _desired if d in col_map]
if not any(c.lower() == 'timestamp' for c in display_cols) and 'date' in col_map:
    display_cols.insert(0, col_map['date'])

display_df = rows_df[display_cols].copy() if display_cols else rows_df.copy()

ts_col  = next((c for c in display_df.columns if c.lower() in ['timestamp', 'date']), None)
amt_col = next((c for c in display_df.columns if c.lower() == 'amount'), None)
bal_col = next((c for c in display_df.columns if c.lower() == 'balance'), None)

if ts_col:  display_df[ts_col] = pd.to_datetime(display_df[ts_col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M').fillna('')
if amt_col: display_df[amt_col] = pd.to_numeric(display_df[amt_col], errors='coerce')
if bal_col: display_df[bal_col] = pd.to_numeric(display_df[bal_col], errors='coerce')

pretty_rename = {
    'timestamp': 'Timestamp', 'date': 'Timestamp', 'bank': 'Bank', 'type': 'Type',
    'amount': 'Amount', 'balance': 'Balance', 'message': 'Message', 'subtype': 'Subtype',
}
display_df = display_df.rename(columns={
    c: pretty_rename[c.lower()] for c in display_df.columns if c.lower() in pretty_rename
})
final_order = [c for c in ['Timestamp', 'Bank', 'Type', 'Amount', 'Balance', 'Subtype', 'Message']
               if c in display_df.columns]
display_df = display_df[final_order].reset_index(drop=True)
if 'Timestamp' in display_df.columns:
    display_df = display_df.sort_values(by='Timestamp', ascending=False).reset_index(drop=True)

# Also keep rows_df sorted the same way so positional indices stay aligned
if 'timestamp' in rows_df.columns:
    rows_df = rows_df.sort_values(by='timestamp', ascending=False).reset_index(drop=True)

col_config = {
    "Amount":  st.column_config.NumberColumn(format="₹%.2f"),
    "Balance": st.column_config.NumberColumn(format="₹%.0f", help="Running balance after this transaction"),
}

# ── Checkbox-based row selection (works on ALL Streamlit versions) ─────────
if can_delete:
    # Inject a "Select" checkbox column into the editable data_editor
    # data_editor with disabled=all-data-cols + one bool column = row selector
    select_df = display_df.copy()
    select_df.insert(0, "🗑️ Select", False)

    # Disable all columns except the checkbox
    disabled_cols = [c for c in select_df.columns if c != "🗑️ Select"]

    edited = st.data_editor(
        select_df,
        use_container_width=True,
        height=420,
        column_config={
            "🗑️ Select": st.column_config.CheckboxColumn("🗑️", help="Tick to mark for deletion", default=False),
            "Amount":    st.column_config.NumberColumn(format="₹%.2f"),
            "Balance":   st.column_config.NumberColumn(format="₹%.0f"),
        },
        disabled=disabled_cols,
        hide_index=True,
        key="txn_editor",
    )

    selected_row_indices = edited.index[edited["🗑️ Select"] == True].tolist()  # noqa: E712

else:
    st.dataframe(
        display_df,
        use_container_width=True,
        height=420,
        column_config=col_config,
        hide_index=True,
    )
    selected_row_indices = []

csv_bytes = display_df.to_csv(index=False).encode("utf-8")

# ── Action bar: delete button + CSV download ─────────────────────────────
if can_delete:
    del_col, dl_col = st.columns([3, 2])

    with dl_col:
        st.download_button(
            "📥 Download Rows (CSV)", csv_bytes,
            file_name="transactions_rows.csv", mime="text/csv",
        )

    with del_col:
        if selected_row_indices:
            n_sel = len(selected_row_indices)
            if st.button(
                f"🗑️ Delete {n_sel} selected row{'s' if n_sel > 1 else ''}",
                type="primary",
                key="open_delete_popup_btn",
            ):
                st.session_state["_show_delete_popup"] = True

    # ── Confirmation popup (container, works on all Streamlit versions) ────
    if st.session_state.get("_show_delete_popup") and selected_row_indices:
        n = len(selected_row_indices)

        popup = st.container()
        popup.markdown("---")
        popup.markdown(
            f"<div style='background:#fff3cd;border:1.5px solid #ffc107;border-radius:10px;"
            f"padding:18px 22px;'>"
            f"<b style='font-size:17px'>⚠️ Confirm Deletion</b><br><br>"
            f"You are about to <b>soft-delete {n} transaction{'s' if n > 1 else ''}</b> "
            f"from the Google Sheet. "
            f"This sets <code>is_deleted = TRUE</code> — they will no longer appear in the app."
            f"</div>",
            unsafe_allow_html=True,
        )

        # Preview the rows about to be deleted
        popup.markdown("**Rows selected for deletion:**")
        preview_cols = [c for c in ['Timestamp', 'Bank', 'Type', 'Amount', 'Message']
                        if c in display_df.columns]
        popup.dataframe(
            display_df.iloc[selected_row_indices][preview_cols].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

        yes_col, no_col, _ = popup.columns([2, 2, 4])

        with yes_col:
            if st.button("✅ Yes, delete", type="primary", use_container_width=True, key="del_confirm_yes"):
                groups    = {}
                any_error = False
                creds     = _get_creds_info()

                for pos in selected_row_indices:
                    if pos >= len(rows_df):
                        continue
                    r   = rows_df.iloc[pos]
                    src = r.get('_source_sheet', 'history') if '_source_sheet' in rows_df.columns else 'history'
                    try:
                        idx = int(r['_sheet_row_idx']) if '_sheet_row_idx' in rows_df.columns else -1
                    except Exception:
                        continue
                    if idx < 0:
                        continue
                    rng = APPEND_RANGE if src == 'append' else RANGE
                    groups.setdefault(rng, []).append(idx)

                total_updated = 0
                for rng, indices in groups.items():
                    try:
                        res = io_mod.mark_rows_deleted(SHEET_ID, rng, creds, CREDS_FILE, indices)
                        if res.get('status') == 'ok':
                            total_updated += res.get('updated', 0)
                        else:
                            st.error(f"Failed ({rng}): {res.get('message')}")
                            any_error = True
                    except Exception as e:
                        st.error(f"Error ({rng}): {e}")
                        any_error = True

                st.session_state["_show_delete_popup"] = False
                if not any_error:
                    st.success(f"✅ Deleted {total_updated} row(s) successfully.")
                    st.cache_data.clear()
                    st.rerun()

        with no_col:
            if st.button("✖ Cancel", use_container_width=True, key="del_confirm_no"):
                st.session_state["_show_delete_popup"] = False
                st.rerun()

        popup.markdown("---")

else:
    st.download_button(
        "📥 Download Rows (CSV)", csv_bytes,
        file_name="transactions_rows.csv", mime="text/csv",
    )

# ─────────────────────────────────────────────
# Add New Row (Google Sheets only)
# ─────────────────────────────────────────────
if use_google and io_mod is not None:
    st.markdown("---")
    st.write("➕ Add New Row (to Append sheet)")
    with st.expander("Open add row form"):
        with st.form("add_row_form", clear_on_submit=True):
            new_date    = st.date_input("Date", value=start_sel, min_value=min_date_overall, max_value=max_date_overall)
            banks_for_add = sorted(set(banks_available + ['Other (enter below)']))
            bank_choice  = st.selectbox("Bank", options=banks_for_add)
            bank_other   = st.text_input("Bank (custom)") if bank_choice == "Other (enter below)" else ""
            txn_type     = st.selectbox("Type", options=["debit", "credit"])
            subtype      = st.selectbox("Subtype", options=["bank_transfer", "card", "upi", "other"])
            amount       = st.number_input("Amount (₹)", value=0.0, step=1.0, format="%.2f")
            message      = st.text_input("Message / Description", value="")
            submit_add   = st.form_submit_button("Save New Row")

            if submit_add:
                chosen_bank = bank_other if bank_other else (bank_choice if bank_choice != "Other (enter below)" else "Unknown")
                dt_combined = datetime.combine(new_date, datetime.utcnow().time())
                new_row = {
                    'DateTime': dt_combined.strftime("%Y-%m-%d %H:%M:%S"), 'timestamp': dt_combined,
                    'date': dt_combined.date(), 'Bank': chosen_bank, 'Type': txn_type,
                    'Amount': amount, 'Message': message, 'is_deleted': 'false', 'Subtype': subtype,
                }
                creds_info = _get_creds_info()
                try:
                    res = io_mod.append_new_row(SHEET_ID, APPEND_RANGE, new_row, creds_info, CREDS_FILE, RANGE)
                    if res.get('status') == 'ok':
                        st.success("Appended row.")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(f"Failed: {res.get('message')}")
                except Exception as e:
                    st.error(f"Error: {e}")

# ─────────────────────────────────────────────
# Totals summary
# ─────────────────────────────────────────────
totals_heading = f"Totals — {start_sel}" if start_sel == end_sel else f"Totals — {start_sel} → {end_sel}"
st.markdown(f"### {totals_heading}")

try:
    sel_df = rows_df.copy()
    col_map_lower = {c.lower(): c for c in sel_df.columns}
    amount_col    = col_map_lower.get('amount')
    type_col      = col_map_lower.get('type')
    credit_sum = debit_sum = 0.0
    credit_count = debit_count = 0

    if sel_df.empty:
        st.info("No transactions match all filters for the selected date range.")
    elif amount_col:
        sel_df[amount_col] = pd.to_numeric(sel_df[amount_col], errors='coerce').fillna(0.0)
        if type_col:
            sel_df['type_norm'] = sel_df[type_col].astype(str).str.lower().str.strip()
            credit_mask = sel_df['type_norm'] == 'credit'
            debit_mask  = sel_df['type_norm'] == 'debit'
            credit_sum  = float(sel_df.loc[credit_mask, amount_col].sum())
            debit_sum   = float(sel_df.loc[debit_mask,  amount_col].sum())
            credit_count = int(credit_mask.sum())
            debit_count  = int(debit_mask.sum())
        else:
            for _, r in sel_df.iterrows():
                amt = r[amount_col]
                if amt < 0: credit_sum += abs(amt); credit_count += 1
                else:        debit_sum  += amt;      debit_count  += 1
    else:
        st.warning("Cannot calculate totals: 'Amount' column not found.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Credits",  f"₹{credit_sum:,.0f}",             f"{credit_count} txns")
    col2.metric("Debits",   f"₹{debit_sum:,.0f}",              f"{debit_count} txns")
    col3.metric("Net Flow", f"₹{(credit_sum - debit_sum):,.0f}")

except Exception as e:
    st.error(f"Failed to compute totals: {e}")
