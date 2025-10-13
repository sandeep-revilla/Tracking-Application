"""


elif chart_type == 'Calendar heatmap':
cals = plot_df[['Date','Total_Spent']].copy()
cals['dow'] = cals['Date'].dt.weekday
cals['week'] = cals['Date'].dt.isocalendar().week
cals['amount'] = cals['Total_Spent']
heat = alt.Chart(cals).mark_rect().encode(
x=alt.X('week:O', title='Week of year'),
y=alt.Y('dow:O', title='Day of week'),
color=alt.Color('amount:Q', title='Total Spent', scale=alt.Scale(scheme='reds')),
tooltip=[alt.Tooltip('Date:T'), alt.Tooltip('amount:Q', format=',')]
).properties(height=180)
st.altair_chart(heat, use_container_width=True)


elif chart_type == 'Histogram of amounts':
tx = converted_df.copy()
tx['Amount_numeric'] = pd.to_numeric(tx.get('Amount',0), errors='coerce').fillna(0)
hist = alt.Chart(tx).mark_bar().encode(
alt.X('Amount_numeric:Q', bin=alt.Bin(maxbins=40), title='Transaction amount'),
y='count()',
tooltip=[alt.Tooltip('count()', title='Count')]
).interactive()
st.altair_chart(hist, use_container_width=True)


elif chart_type == 'Treemap by category':
try:
import plotly.express as px
tx = converted_df.copy()
if 'Category' not in tx.columns:
tx['Category'] = tx.get('category', 'Unknown')
cat = tx.groupby('Category')['Amount'].sum().reset_index()
cat['Size'] = cat['Amount'].abs()
fig = px.treemap(cat, path=['Category'], values='Size', title='Spend by Category')
st.plotly_chart(fig, use_container_width=True)
except Exception as e:
st.error(f"Treemap requires plotly: {e}")


else:
st.info("Unknown chart type selected.")
