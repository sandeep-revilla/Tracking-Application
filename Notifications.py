with mark_all_col:
        btn1, btn2, btn3 = st.columns(3)

        # ── Mark all as seen ─────────────────────────────────────────────
        with btn1:
            if unseen_count > 0:
                if st.button("✅ Mark all seen", key="mark_all_seen_btn", use_container_width=True):
                    if use_google and notif_mod and SHEET_ID:
                        notif_mod.mark_all_seen(SHEET_ID, creds_info=creds_info, creds_file=CREDS_FILE)
                        st.cache_data.clear()
                    else:
                        if 'sample_notif_df' in st.session_state:
                            st.session_state['sample_notif_df']['is_seen'] = 'true'
                    st.rerun()

        # ── Hard delete all ───────────────────────────────────────────────
        with btn2:
            if use_google and notif_mod and SHEET_ID:
                if st.button("🗑️ Delete All", key="delete_all_notif_btn",
                             use_container_width=True, type="primary"):
                    st.session_state['_show_delete_all_notif_popup'] = True

        # ── Reload from threshold ─────────────────────────────────────────
        with btn3:
            if use_google and notif_mod and SHEET_ID:
                if st.button(f"🔄 Reload ₹{int(alert_threshold):,}", key="reload_notif_btn",
                             use_container_width=True):
                    st.session_state['_show_reload_notif_popup'] = True

    # ── Delete All confirmation popup ─────────────────────────────────────
    if st.session_state.get('_show_delete_all_notif_popup'):
        st.markdown("---")
        st.markdown(
            "<div style='background:#fff3cd;border:1.5px solid #ffc107;"
            "border-radius:10px;padding:16px 20px;'>"
            "<b style='font-size:16px'>⚠️ Confirm Hard Delete</b><br><br>"
            "This will <b>permanently delete ALL notification rows</b> from the sheet. "
            "The header will be preserved but all data will be gone. "
            "This cannot be undone."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        c1, c2, _ = st.columns([2, 2, 4])
        with c1:
            if st.button("✅ Yes, delete all", type="primary",
                         use_container_width=True, key="del_all_notif_confirm"):
                with st.spinner("Deleting all notifications…"):
                    ok = notif_mod.delete_all_notifications(
                        SHEET_ID, creds_info=creds_info, creds_file=CREDS_FILE
                    )
                st.session_state['_show_delete_all_notif_popup'] = False
                st.cache_data.clear()
                if ok:
                    st.success("✅ All notifications deleted.")
                else:
                    st.error("Failed to delete notifications. Check logs.")
                st.rerun()
        with c2:
            if st.button("✖ Cancel", use_container_width=True, key="del_all_notif_cancel"):
                st.session_state['_show_delete_all_notif_popup'] = False
                st.rerun()
        st.markdown("---")

    # ── Reload confirmation popup ─────────────────────────────────────────
    if st.session_state.get('_show_reload_notif_popup'):
        st.markdown("---")
        st.markdown(
            f"<div style='background:#cfe2ff;border:1.5px solid #084298;"
            f"border-radius:10px;padding:16px 20px;'>"
            f"<b style='font-size:16px'>🔄 Confirm Reload</b><br><br>"
            f"This will <b>delete all existing notifications</b> and re-scan "
            f"<b>all historical transactions</b> for debits above "
            f"<b>₹{int(alert_threshold):,}</b>. "
            f"All reloaded notifications will be marked as <b>unseen</b>."
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        c1, c2, _ = st.columns([2, 2, 4])
        with c1:
            if st.button("✅ Yes, reload", type="primary",
                         use_container_width=True, key="reload_notif_confirm"):
                with st.spinner(f"Reloading notifications at ₹{int(alert_threshold):,}…"):
                    total_loaded, _ = notif_mod.reload_all_notifications(
                        converted_df_with_balance,
                        spreadsheet_id=SHEET_ID,
                        threshold=alert_threshold,
                        creds_info=creds_info,
                        creds_file=CREDS_FILE,
                    )
                st.session_state['_show_reload_notif_popup'] = False
                st.cache_data.clear()
                if total_loaded > 0:
                    st.success(f"✅ Loaded {total_loaded} notification(s) at ₹{int(alert_threshold):,}.")
                else:
                    st.info(f"No transactions found above ₹{int(alert_threshold):,}.")
                st.rerun()
        with c2:
            if st.button("✖ Cancel", use_container_width=True, key="reload_notif_cancel"):
                st.session_state['_show_reload_notif_popup'] = False
                st.rerun()
        st.markdown("---")
