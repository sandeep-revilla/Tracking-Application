# ─────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────
st.subheader("📊 Daily Spend and Credit")
if plot_df.empty and chart_type_select not in (
    "Weekly heatmap", "Debit vs Credit pie", "Bank breakdown", "Day-of-week pattern"
):
    st.info("No data available for the selected chart filters.")
elif charts_mod is not None:
    series_selected = [
        s for s, show in [
            ('Total_Spent',  show_debit_chart),
            ('Total_Credit', show_credit_chart),
        ] if show
    ]
    try:
        selected_date = charts_mod.render_chart(
            plot_df=plot_df,
            converted_df=converted_df_filtered,
            chart_type=chart_type_select,
            series_selected=series_selected,
            top_n=5,
        )
    except Exception as chart_err:
        st.error(f"Failed to render chart: {chart_err}")
        st.exception(chart_err)
        selected_date = None

    # ── Heatmap drilldown — show transactions for clicked day ─────────────
    if chart_type_select == "Weekly heatmap":

        # Persist the clicked date in session state so it survives reruns
        if selected_date:
            st.session_state['heatmap_selected_date'] = selected_date

        drill_date = st.session_state.get('heatmap_selected_date')

        if drill_date:
            # ── Clear button ──────────────────────────────────────────────
            clear_col, _ = st.columns([2, 6])
            with clear_col:
                if st.button("✖ Clear selection", key="heatmap_clear_btn"):
                    st.session_state.pop('heatmap_selected_date', None)
                    st.rerun()

            # ── Filter transactions for that exact date ───────────────────
            drill_df = converted_df_filtered.copy()

            if 'timestamp' in drill_df.columns:
                drill_df['_drill_date'] = pd.to_datetime(
                    drill_df['timestamp'], errors='coerce'
                ).dt.strftime('%Y-%m-%d')
            elif 'date' in drill_df.columns:
                drill_df['_drill_date'] = pd.to_datetime(
                    drill_df['date'], errors='coerce'
                ).dt.strftime('%Y-%m-%d')
            else:
                drill_df['_drill_date'] = None

            day_txns = drill_df[drill_df['_drill_date'] == drill_date].copy()

            # ── Format display date nicely ────────────────────────────────
            try:
                display_date = pd.to_datetime(drill_date).strftime('%d %b %Y')
            except Exception:
                display_date = drill_date

            st.markdown(f"---")
            st.markdown(
                f"#### 📋 Transactions for **{display_date}** "
                f"— {len(day_txns)} transaction{'s' if len(day_txns) != 1 else ''}",
            )

            if day_txns.empty:
                st.info("No transactions found for this date.")
            else:
                # ── Build clean display table ─────────────────────────────
                _desired_drill  = ['timestamp', 'bank', 'type', 'amount', 'Balance', 'subtype', 'message']
                _col_map_drill  = {c.lower(): c for c in day_txns.columns}
                _display_cols   = [_col_map_drill[d] for d in _desired_drill if d in _col_map_drill]

                drill_display = day_txns[_display_cols].copy() if _display_cols else day_txns.copy()

                # Format timestamp
                _ts_col = next((c for c in drill_display.columns if c.lower() in ['timestamp', 'date']), None)
                if _ts_col:
                    drill_display[_ts_col] = pd.to_datetime(
                        drill_display[_ts_col], errors='coerce'
                    ).dt.strftime('%H:%M')   # show only time since date is already in heading

                # Format amount + balance
                _amt_col = next((c for c in drill_display.columns if c.lower() == 'amount'), None)
                _bal_col = next((c for c in drill_display.columns if c.lower() == 'balance'), None)
                if _amt_col:
                    drill_display[_amt_col] = pd.to_numeric(drill_display[_amt_col], errors='coerce')
                if _bal_col:
                    drill_display[_bal_col] = pd.to_numeric(drill_display[_bal_col], errors='coerce')

                # Rename columns
                _pretty = {
                    'timestamp': 'Time', 'date': 'Time', 'bank': 'Bank',
                    'type': 'Type', 'amount': 'Amount', 'balance': 'Balance',
                    'message': 'Message', 'subtype': 'Subtype',
                }
                drill_display = drill_display.rename(columns={
                    c: _pretty[c.lower()] for c in drill_display.columns if c.lower() in _pretty
                })

                # Order columns
                _final_order = [
                    c for c in ['Time', 'Bank', 'Type', 'Amount', 'Balance', 'Subtype', 'Message']
                    if c in drill_display.columns
                ]
                drill_display = drill_display[_final_order].reset_index(drop=True)

                # Sort by time
                if 'Time' in drill_display.columns:
                    drill_display = drill_display.sort_values('Time').reset_index(drop=True)

                st.dataframe(
                    drill_display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Amount":  st.column_config.NumberColumn(format="₹%.2f"),
                        "Balance": st.column_config.NumberColumn(format="₹%.0f"),
                    },
                )

                # ── Day totals ────────────────────────────────────────────
                if 'Amount' in drill_display.columns and 'Type' in drill_display.columns:
                    day_debit  = drill_display.loc[
                        drill_display['Type'].str.lower() == 'debit', 'Amount'
                    ].sum()
                    day_credit = drill_display.loc[
                        drill_display['Type'].str.lower() == 'credit', 'Amount'
                    ].sum()
                    d1, d2, d3 = st.columns(3)
                    d1.metric("Day Debits",  f"₹{day_debit:,.0f}")
                    d2.metric("Day Credits", f"₹{day_credit:,.0f}")
                    d3.metric("Net",         f"₹{day_credit - day_debit:,.0f}")

else:
    st.info("charts.py not available.")
