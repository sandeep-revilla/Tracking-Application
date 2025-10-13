# === FILE: app.py (main Streamlit app) ===


# Apply filters to merged
plot_df = merged.copy()
if sel_year != 'All':
plot_df = plot_df[plot_df['Date'].dt.year == int(sel_year)]
if sel_months:
inv_map = {v:k for k,v in {i: pd.Timestamp(1900, i, 1).strftime('%B') for i in range(1,13)}.items()}
selected_month_nums = [inv_map[m] for m in sel_months if m in inv_map]
if selected_month_nums:
plot_df = plot_df[plot_df['Date'].dt.month.isin(selected_month_nums)]


plot_df = plot_df.sort_values('Date').reset_index(drop=True)


# Chart selector UI
chart_type = st.sidebar.selectbox("Chart type", [
"Daily line", "Stacked area", "Monthly bars", "Rolling average",
"Cumulative sum", "Calendar heatmap", "Histogram of amounts", "Treemap by category"
])


# Series selection
series_selected = []
if show_debit: series_selected.append('Total_Spent')
if show_credit: series_selected.append('Total_Credit')


# Render chart
charts_mod.render_chart(plot_df, converted_df, chart_type, series_selected, enable_plotly_click)


# Show rows for selected date range (simple approach: show all filtered rows)
st.subheader("Rows (matching selection)")
rows_df = converted_df.copy()
# apply year/month filters
if sel_year != 'All':
try:
rows_df = rows_df[rows_df['timestamp'].dt.year == int(sel_year)]
except Exception:
pass
if sel_months:
inv_map = {v: k for k, v in {i: pd.Timestamp(1900, i, 1).strftime('%B') for i in range(1,13)}.items()}
selected_month_nums = [inv_map[m] for m in sel_months if m in inv_map]
if selected_month_nums:
rows_df = rows_df[rows_df['timestamp'].dt.month.isin(selected_month_nums)]


if rows_df.empty:
st.write("No rows match the current filters/selection.")
else:
st.dataframe(rows_df.reset_index(drop=True))


# End of modularized files
