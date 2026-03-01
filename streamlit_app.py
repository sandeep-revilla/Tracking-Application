# ─────────────────────────────────────────────
# ── BELL ICON HEADER & Notifications-only view
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
    # Render a simple toggle button. Clicking flips session state.
    if st.button(
        bell_label,
        key="bell_btn",
        help=f"{unread_count} unread notification(s)" if unread_count > 0 else "No unread notifications",
    ):
        st.session_state['show_notif_panel'] = not st.session_state.get('show_notif_panel', False)

# Check whether notifications-only view is requested
show_notif = st.session_state.get('show_notif_panel', False)

# If notifications view is requested: render ONLY the notifications panel and stop the app here.
if show_notif:
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
                st.experimental_rerun()

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
                    st.experimental_rerun()

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
                            st.experimental_rerun()

    st.markdown("---")
    # Important: stop rendering the rest of the page (so only notifications show)
    st.stop()

# If we reach here, notifications panel is NOT open — render the main app as usual.
st.info(
    "ℹ️ **Disclaimer:** This data is for reference only and may be incomplete. "
    "Please cross-check with your official bank statements if you notice a large deviation."
)
