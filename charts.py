# charts.py - visualization utilities
import streamlit as st
import pandas as pd
import altair as alt
from typing import Optional

def render_chart(plot_df: pd.DataFrame,
                 converted_df: pd.DataFrame,
                 chart_type: str,
                 series_selected: list,
                 enable_plotly_click: bool = False,
                 height: int = 420) -> Optional[pd.Timestamp]:
    """
    Render a chart for aggregated data.
    Currently supports only a simple Altair line chart ("Daily line").
    - plot_df: aggregated DataFrame with Date, Total_Spent, Total_Credit
    - converted_df: cleaned transactions DataFrame (not used for line chart but passed for future features)
    - chart_type: string selecting chart (only "Daily line" supported now)
    - series_selected: list of series to include (e.g., ['Total_Spent'])
    - enable_plotly_click: placeholder (not used yet)
    Returns:
      - None (no clicked date supported yet). Later this can return the date selected by clicking a point.
    """
    if plot_df is None or plot_df.empty:
        st.info("No data available for charting.")
        return None

    # prepare long form for plotting
    vars_to_plot = [c for c in ['Total_Spent', 'Total_Credit'] if c in series_selected and c in plot_df.columns]
    if not vars_to_plot:
        st.info("No series selected for plotting.")
        return None

    # melt to long form
    plot_df_long = plot_df.melt(id_vars='Date', value_vars=vars_to_plot, var_name='Type', value_name='Amount').sort_values('Date')
    plot_df_long['Date'] = pd.to_datetime(plot_df_long['Date'])
    plot_df_long['Amount'] = pd.to_numeric(plot_df_long['Amount'], errors='coerce').fillna(0.0)

    # Build Altair line chart
    color_scale = alt.Scale(domain=['Total_Spent', 'Total_Credit'], range=['#d62728', '#2ca02c'])
    chart = alt.Chart(plot_df_long).mark_line(point=True).encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Amount:Q', title='Amount', axis=alt.Axis(format=",.0f")),
        color=alt.Color('Type:N', title='Type', scale=color_scale),
        tooltip=[alt.Tooltip('Date:T', title='Date', format='%Y-%m-%d'),
                 alt.Tooltip('Type:N', title='Type'),
                 alt.Tooltip('Amount:Q', title='Amount', format=',')]
    ).interactive()

    st.altair_chart(chart.properties(height=height), use_container_width=True)

    # No click-to-select currently — return None
    return None
