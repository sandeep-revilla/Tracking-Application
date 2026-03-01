# pages/📋_Transaction_Detail.py
# Shown when user clicks a notification. Receives uid via st.query_params.

import streamlit as st
import pandas as pd
import importlib
import sys, os

st.set_page_config(page_title="Transaction Detail", layout="centered")

# ── resolve imports from parent directory ──────────────────────────────────
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent not in sys.path:
    sys.path.insert(0, parent)

try:
    import notifications as notif_mod
except ImportError:
    notif_mod = None

try:
    import io_helpers as io_mod
except ImportError:
    io_mod = None

# ── read query param ───────────────────────────────────────────────────────
params = st.query_params
uid    = params.get("uid", "")

if not uid:
    st.warning("No notification selected. Go back to the main page and click a notification.")
    if st.button("← Back to Dashboard"):
        st.switch_page("streamlit_app.py")
    st.stop()

# ── credentials ────────────────────────────────────────────────────────────
_secrets      = getattr(st, "secrets", {}) or {}
SHEET_ID      = _secrets.get("SHEET_ID", "")
CREDS_FILE    = _secrets.get("CREDS_FILE", "creds/service_account.json")

def _get_creds_info():
    if io_mod is None:
        return None
    try:
        if hasattr(st, "secrets") and st.secrets and "gcp_service_account" in st.secrets:
            return io_mod.parse_service_account_secret(st.secrets["gcp_service_account"])
    except Exception:
        pass
    return None

creds_info = _get_creds_info()

# ── load notifications ─────────────────────────────────────────────────────
if notif_mod is None or not SHEET_ID:
    st.error("Notifications module or Sheet ID not available.")
    st.stop()

with st.spinner("Loading notification…"):
    notif_df = notif_mod.read_notifications(SHEET_ID, creds_info=creds_info, creds_file=CREDS_FILE)

if notif_df.empty or uid not in notif_df['uid'].values:
    st.error("Notification not found. It may have been deleted.")
    if st.button("← Back to Dashboard"):
        st.switch_page("streamlit_app.py")
    st.stop()

row = notif_df[notif_df['uid'] == uid].iloc[0]

# ── Mark as read ───────────────────────────────────────────────────────────
if str(row.get('is_read', 'false')).lower() != 'true':
    notif_mod.mark_notification_read(SHEET_ID, uid, creds_info=creds_info, creds_file=CREDS_FILE)

# ── UI ─────────────────────────────────────────────────────────────────────
st.markdown("### 📋 Transaction Detail")
st.markdown("---")

amount    = float(row.get('amount', 0) or 0)
threshold = float(row.get('threshold', 500) or 500)
ts        = row.get('timestamp', '')
bank      = row.get('bank', 'Unknown')
message   = row.get('message', '—')
subtype   = row.get('subtype', '—')
created   = row.get('created_at', '')
is_read   = str(row.get('is_read', 'false')).lower() == 'true'

# Status badge
status_color = "#6c757d" if is_read else "#dc3545"
status_label = "✅ Verified" if is_read else "🔴 Unread"

st.markdown(
    f"""
    <div style="background:#fff8f8;border:1.5px solid #f5c2c7;border-radius:10px;padding:24px 28px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
            <span style="font-size:22px;font-weight:700;">₹{amount:,.0f}</span>
            <span style="background:{status_color};color:white;padding:4px 12px;border-radius:20px;font-size:13px;">{status_label}</span>
        </div>
        <table style="width:100%;border-collapse:collapse;font-size:15px;">
            <tr><td style="color:#888;padding:6px 0;width:140px">🏦 Bank</td><td><b>{bank}</b></td></tr>
            <tr><td style="color:#888;padding:6px 0">📅 Date & Time</td><td><b>{ts}</b></td></tr>
            <tr><td style="color:#888;padding:6px 0">💬 Description</td><td>{message}</td></tr>
            <tr><td style="color:#888;padding:6px 0">🏷️ Subtype</td><td>{subtype}</td></tr>
            <tr><td style="color:#888;padding:6px 0">⚠️ Alert Threshold</td><td>₹{threshold:,.0f}</td></tr>
            <tr><td style="color:#888;padding:6px 0">🕐 Flagged At</td><td>{created}</td></tr>
        </table>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Over-by indicator
over_by = amount - threshold
st.markdown(
    f"<div style='background:#fff3cd;border:1px solid #ffc107;border-radius:8px;padding:12px 16px;'>"
    f"⚠️ This transaction is <b>₹{over_by:,.0f} over</b> your ₹{threshold:,.0f} alert threshold.</div>",
    unsafe_allow_html=True,
)

st.markdown("")

col1, col2 = st.columns(2)
with col1:
    if st.button("← Back to Dashboard", use_container_width=True):
        st.switch_page("streamlit_app.py")
with col2:
    if not is_read:
        if st.button("✅ Mark as Verified", type="primary", use_container_width=True):
            notif_mod.mark_notification_read(SHEET_ID, uid, creds_info=creds_info, creds_file=CREDS_FILE)
            st.success("Marked as verified!")
            st.rerun()
    else:
        st.success("Already verified ✓")
