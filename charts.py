# charts.py - visualization utilities
# Charts available:
#   1. Daily line          — daily spend/credit with click-to-drilldown
#   2. Monthly bars        — month-by-month grouped bars
#   3. Top categories      — horizontal bar by category/merchant
#   4. Weekly heatmap      — spend intensity by day-of-week × week
#   5. Cumulative spend    — running total across the month
#   6. Debit vs Credit pie — proportion of spend vs income
#   7. Bank breakdown      — stacked bar showing spend per bank per month
#   8. Day-of-week pattern — average spend by Mon–Sun

import streamlit as st
import pandas as pd
import altair as alt
from typing import Optional, List

alt.data_transformers.disable_max_rows()


# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

def _ensure_date_col(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    if col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _is_deleted_mask(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    isdel_col = next((c for c in df.columns if str(c).lower() == 'is_deleted'), None)
    if isdel_col is None:
        return None
    s = df[isdel_col]
    try:
        if pd.api.types.is_bool_dtype(s):
            return s.fillna(False).astype(bool)
        if pd.api.types.is_numeric_dtype(s):
            return s.fillna(0).astype(int) == 1
        return s.astype(str).str.strip().str.lower().fillna('').isin(['true', 't', '1', 'yes', 'y'])
    except Exception:
        try:
            return s.astype(str).str.strip().str.lower().fillna('').isin(['true', 't', '1', 'yes', 'y'])
        except Exception:
            return None


def _filter_out_deleted(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    mask = _is_deleted_mask(df)
    if mask is None:
        return df.copy()
    try:
        return df.loc[~mask].copy().reset_index(drop=True)
    except Exception:
        return df.copy()


COLOR_SPENT  = '#d62728'
COLOR_CREDIT = '#2ca02c'
COLOR_SCALE  = alt.Scale(domain=['Total_Spent', 'Total_Credit'], range=[COLOR_SPENT, COLOR_CREDIT])
TYPE_SCALE   = alt.Scale(domain=['debit', 'credit'], range=[COLOR_SPENT, COLOR_CREDIT])


# ─────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────

def render_chart(
    plot_df: pd.DataFrame,
    converted_df: pd.DataFrame,
    chart_type: str,
    series_selected: List[str],
    top_n: int = 5,
    height: int = 420,
) -> None:
    plot_df      = _filter_out_deleted(plot_df)      if plot_df      is not None else pd.DataFrame()
    converted_df = _filter_out_deleted(converted_df) if converted_df is not None else pd.DataFrame()

    if plot_df.empty and chart_type not in (
        "Weekly heatmap", "Debit vs Credit pie", "Bank breakdown", "Day-of-week pattern"
    ):
        st.info("No aggregated data available for charting.")
        return

    chart_type = (chart_type or "").strip()

    dispatch = {
        "Daily line":             lambda: _render_daily_line(plot_df, converted_df, series_selected, height),
        "Monthly bars":           lambda: _render_monthly_bars(plot_df, series_selected, height),
        "Top categories (Top-N)": lambda: _render_top_categories(converted_df, top_n, height),
        "Weekly heatmap":         lambda: _render_weekly_heatmap(converted_df, height),
        "Cumulative spend":       lambda: _render_cumulative_spend(plot_df, series_selected, height),
        "Debit vs Credit pie":    lambda: _render_debit_credit_pie(converted_df, height),
        "Bank breakdown":         lambda: _render_bank_breakdown(converted_df, height),
        "Day-of-week pattern":    lambda: _render_dow_pattern(converted_df, height),
    }

    fn = dispatch.get(chart_type)
    if fn:
        fn()
    else:
        st.error(f"Unknown chart type: {chart_type}")


# ─────────────────────────────────────────────────────────────
# 1. Daily line
# ─────────────────────────────────────────────────────────────

def _render_daily_line(plot_df, converted_df, series_selected, height):
    df = _ensure_date_col(plot_df, "Date")
    vars_to_plot = [c for c in ['Total_Spent', 'Total_Credit'] if c in series_selected and c in df.columns]
    if not vars_to_plot:
        st.info("No series selected for plotting.")
        return

    long = df.melt(id_vars='Date', value_vars=vars_to_plot, var_name='Type', value_name='Amount').sort_values('Date')
    long['Amount'] = pd.to_numeric(long['Amount'], errors='coerce').fillna(0.0)

    date_sel = alt.selection_point(fields=['Date'], nearest=True, on='click', empty='none', clear='dblclick')

    base = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Amount:Q', title='Amount', axis=alt.Axis(format=",.0f")),
        color=alt.Color('Type:N', title='Type', scale=COLOR_SCALE),
        tooltip=[
            alt.Tooltip('Date:T', title='Date', format='%Y-%m-%d'),
            alt.Tooltip('Type:N', title='Type'),
            alt.Tooltip('Amount:Q', title='Amount', format=','),
        ],
        opacity=alt.condition(date_sel, alt.value(1.0), alt.value(0.8)),
    ).add_params(date_sel).interactive()

    if converted_df is None or converted_df.empty:
        st.altair_chart(base.properties(height=height), use_container_width=True)
        return

    tx = converted_df.copy()
    if 'timestamp' in tx.columns:
        tx['timestamp'] = pd.to_datetime(tx['timestamp'], errors='coerce')
    elif 'date' in tx.columns:
        tx['timestamp'] = pd.to_datetime(tx['date'], errors='coerce')
    else:
        tx['timestamp'] = pd.NaT
    tx['Date']           = pd.to_datetime(tx['timestamp'], errors='coerce').dt.normalize()
    tx['Date_str']       = tx['Date'].dt.strftime('%Y-%m-%d')
    tx['Amount_numeric'] = pd.to_numeric(tx.get('Amount', 0), errors='coerce').fillna(0.0)
    tx = tx.reset_index().rename(columns={'index': 'row_index'})
    try:
        tx['rank'] = tx.groupby('Date')['Amount_numeric'].rank(method='first', ascending=False).fillna(999999).astype(int)
    except Exception:
        tx['rank'] = range(len(tx))

    detail = alt.Chart(tx).transform_filter(date_sel).mark_bar().encode(
        x=alt.X('Amount_numeric:Q', title='Amount', axis=alt.Axis(format=",.0f")),
        y=alt.Y('rank:O', title=None, axis=None),
        color=alt.Color('Type:N', scale=TYPE_SCALE, legend=None),
        tooltip=[
            alt.Tooltip('Date_str:N', title='Date'),
            alt.Tooltip('Bank:N',     title='Bank'),
            alt.Tooltip('Type:N',     title='Type'),
            alt.Tooltip('Amount_numeric:Q', title='Amount', format=','),
            alt.Tooltip('Message:N',  title='Message'),
        ],
    ).properties(height=max(120, int(height * 0.35)))

    st.altair_chart(
        alt.vconcat(
            base.properties(height=max(200, int(height * 0.6))),
            detail,
        ).resolve_scale(color='independent'),
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────
# 2. Monthly bars
# ─────────────────────────────────────────────────────────────

def _render_monthly_bars(plot_df, series_selected, height):
    df = _ensure_date_col(plot_df, "Date").copy()
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)

    vars_to_plot = [c for c in ['Total_Spent', 'Total_Credit'] if c in series_selected and c in df.columns]
    if not vars_to_plot:
        st.info("No series selected for plotting.")
        return

    agg  = df.groupby('YearMonth')[vars_to_plot].sum().reset_index()
    long = agg.melt(id_vars='YearMonth', value_vars=vars_to_plot, var_name='Type', value_name='Amount')
    long['Amount'] = pd.to_numeric(long['Amount'], errors='coerce').fillna(0.0)
    order = sorted(long['YearMonth'].unique(), key=lambda x: pd.to_datetime(x + "-01"))

    chart = alt.Chart(long).mark_bar().encode(
        x=alt.X('YearMonth:N', sort=order, title='Month'),
        y=alt.Y('Amount:Q',    title='Amount', axis=alt.Axis(format=",.0f")),
        color=alt.Color('Type:N', title='Type', scale=COLOR_SCALE),
        tooltip=[
            alt.Tooltip('YearMonth:N', title='Month'),
            alt.Tooltip('Type:N',      title='Type'),
            alt.Tooltip('Amount:Q',    title='Amount', format=','),
        ],
    ).interactive()

    st.altair_chart(chart.properties(height=height), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# 3. Top categories
# ─────────────────────────────────────────────────────────────

def _render_top_categories(converted_df, top_n, height):
    if converted_df is None or converted_df.empty:
        st.info("No transaction data available for Top-N categories.")
        return

    df = converted_df.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['Amount_numeric'] = pd.to_numeric(df.get('Amount', 0), errors='coerce').fillna(0.0)

    cat_col = next((c for c in ['Category', 'Merchant', 'merchant', 'category', 'description'] if c in df.columns), None)
    if cat_col is None:
        st.info("Top-N chart needs a Category or Merchant column.")
        return

    if 'Type' in df.columns:
        spend_mask = df['Type'].astype(str).str.lower().str.strip() == 'debit'
    else:
        spend_mask = df['Amount_numeric'] > 0

    spend_df = df[spend_mask]
    if spend_df.empty:
        st.info("No debit transactions found for Top-N.")
        return

    agg = (spend_df.groupby(cat_col)['Amount_numeric'].sum()
           .reset_index().rename(columns={cat_col: 'Category', 'Amount_numeric': 'Total'})
           .sort_values('Total', ascending=False).head(top_n))

    chart = alt.Chart(agg).mark_bar(color=COLOR_SPENT).encode(
        x=alt.X('Total:Q',    title='Total spend', axis=alt.Axis(format=",.0f")),
        y=alt.Y('Category:N', sort='-x', title='Category'),
        tooltip=[
            alt.Tooltip('Category:N', title='Category'),
            alt.Tooltip('Total:Q',    title='Total', format=','),
        ],
    ).properties(height=max(200, min(600, 40 * len(agg))))

    st.altair_chart(chart, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# 4. Weekly heatmap  ← UPDATED
#    - Month filter selectbox sits RIGHT ABOVE the chart
#    - Green = low spend, Red = high spend
# ─────────────────────────────────────────────────────────────

def _render_weekly_heatmap(converted_df, height):
    if converted_df is None or converted_df.empty:
        st.info("No transaction data available for heatmap.")
        return

    df = converted_df.copy()
    ts = 'timestamp' if 'timestamp' in df.columns else ('date' if 'date' in df.columns else None)
    if ts is None:
        st.info("No date column found for heatmap.")
        return

    df['_ts'] = pd.to_datetime(df[ts], errors='coerce')
    df = df.dropna(subset=['_ts'])
    df['Amount_numeric'] = pd.to_numeric(df.get('Amount', 0), errors='coerce').fillna(0.0)

    if 'Type' in df.columns:
        df = df[df['Type'].astype(str).str.lower().str.strip() == 'debit']

    if df.empty:
        st.info("No debit transactions to display.")
        return

    # ── Month filter — sits directly above the heatmap ───────────────────
    df['_ym'] = df['_ts'].dt.to_period('M').astype(str)
    available_months = sorted(df['_ym'].unique(), reverse=True)

    filter_col, _ = st.columns([2, 4])
    with filter_col:
        chosen_month = st.selectbox(
            "📅 Select Month",
            options=available_months,
            index=0,                      # default = most recent month
            key="heatmap_month_filter",
        )

    df = df[df['_ym'] == chosen_month]

    if df.empty:
        st.info(f"No debit transactions for {chosen_month}.")
        return

    # ── Build heatmap data ────────────────────────────────────────────────
    df['DayOfWeek'] = df['_ts'].dt.day_name()
    df['WeekStart'] = df['_ts'].dt.to_period('W').apply(
        lambda p: p.start_time.strftime('%Y-%m-%d')
    )

    daily = df.groupby(['WeekStart', 'DayOfWeek'])['Amount_numeric'].sum().reset_index()
    daily.columns = ['Week', 'Day', 'Spend']

    day_order  = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    week_order = sorted(daily['Week'].unique())

    # ── Chart: green (low) → red (high) ──────────────────────────────────
    chart = alt.Chart(daily).mark_rect(
        cornerRadius=4,
        stroke='white',
        strokeWidth=2,
    ).encode(
        x=alt.X(
            'Day:O',
            sort=day_order,
            title='Day of Week',
            axis=alt.Axis(labelAngle=0, labelFontSize=12),
        ),
        y=alt.Y(
            'Week:O',
            sort=week_order,
            title='Week Starting',
            axis=alt.Axis(labelFontSize=11),
        ),
        color=alt.Color(
            'Spend:Q',
            title='₹ Spent',
            scale=alt.Scale(
                scheme='redyellowgreen',  # green = low, yellow = mid, red = high
                reverse=True,            # flip so HIGH spend = RED
            ),
            legend=alt.Legend(
                title='₹ Spent',
                gradientLength=120,
                labelFontSize=11,
            ),
        ),
        tooltip=[
            alt.Tooltip('Week:O',  title='Week of'),
            alt.Tooltip('Day:O',   title='Day'),
            alt.Tooltip('Spend:Q', title='₹ Spent', format=',.0f'),
        ],
    ).properties(
        height=max(160, min(400, 70 * len(week_order))),
        title=alt.TitleParams(
            text=f"Weekly Spend Heatmap — {chosen_month}",
            fontSize=14,
            anchor='start',
        ),
    )

    # ── Day-total text labels inside each cell ────────────────────────────
    text = alt.Chart(daily).mark_text(
        fontSize=11,
        fontWeight='bold',
        color='black',
    ).encode(
        x=alt.X('Day:O',  sort=day_order),
        y=alt.Y('Week:O', sort=week_order),
        text=alt.Text('Spend:Q', format=',.0f'),
        opacity=alt.condition(
            alt.datum.Spend > 0,
            alt.value(0.75),
            alt.value(0),              # hide zero-spend cells
        ),
    )

    st.altair_chart((chart + text), use_container_width=True)

    # ── Summary row below the chart ───────────────────────────────────────
    total_spend  = daily['Spend'].sum()
    peak_row     = daily.loc[daily['Spend'].idxmax()]
    peak_day     = peak_row['Day']
    peak_week    = peak_row['Week']
    peak_amount  = peak_row['Spend']

    s1, s2, s3 = st.columns(3)
    s1.metric("Total Debit",   f"₹{total_spend:,.0f}")
    s2.metric("Peak Day",      peak_day)
    s3.metric("Peak Amount",   f"₹{peak_amount:,.0f}", f"Week of {peak_week}")


# ─────────────────────────────────────────────────────────────
# 5. Cumulative spend
# ─────────────────────────────────────────────────────────────

def _render_cumulative_spend(plot_df, series_selected, height):
    st.markdown("##### 📈 Cumulative Spend — running total within each month")

    df = _ensure_date_col(plot_df, "Date").copy()
    if df.empty or 'Total_Spent' not in df.columns:
        st.info("No spend data available for cumulative chart.")
        return

    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
    months_available = sorted(df['YearMonth'].unique(), reverse=True)

    selected_months = st.multiselect(
        "Months to compare",
        options=months_available,
        default=months_available[:3],
        key="cumul_month_select",
    )
    if not selected_months:
        st.info("Select at least one month.")
        return

    df = df[df['YearMonth'].isin(selected_months)].copy()
    df['Day'] = df['Date'].dt.day

    rows = []
    for ym, grp in df.groupby('YearMonth'):
        grp = grp.sort_values('Day')
        grp['Cumulative'] = grp['Total_Spent'].cumsum()
        rows.append(grp[['Day', 'Cumulative', 'YearMonth']])
    cum_df = pd.concat(rows, ignore_index=True)

    chart = alt.Chart(cum_df).mark_line(point=True).encode(
        x=alt.X('Day:Q',        title='Day of month', scale=alt.Scale(domain=[1, 31])),
        y=alt.Y('Cumulative:Q', title='Cumulative Spend (₹)', axis=alt.Axis(format=",.0f")),
        color=alt.Color('YearMonth:N', title='Month'),
        tooltip=[
            alt.Tooltip('YearMonth:N', title='Month'),
            alt.Tooltip('Day:Q',        title='Day'),
            alt.Tooltip('Cumulative:Q', title='₹ Total so far', format=',.0f'),
        ],
    ).interactive()

    st.altair_chart(chart.properties(height=height), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# 6. Debit vs Credit pie
# ─────────────────────────────────────────────────────────────

def _render_debit_credit_pie(converted_df, height):
    st.markdown("##### 🥧 Debit vs Credit — overall proportion")

    if converted_df is None or converted_df.empty:
        st.info("No transaction data available.")
        return

    df = converted_df.copy()
    df['Amount_numeric'] = pd.to_numeric(df.get('Amount', 0), errors='coerce').fillna(0.0)

    if 'Type' not in df.columns:
        st.info("No Type column found.")
        return

    summary = (df.groupby(df['Type'].astype(str).str.lower().str.strip())['Amount_numeric']
               .sum().reset_index())
    summary.columns = ['Type', 'Total']
    summary = summary[summary['Total'] > 0]

    if summary.empty:
        st.info("No data to display.")
        return

    chart = alt.Chart(summary).mark_arc(outerRadius=130).encode(
        theta=alt.Theta('Total:Q'),
        color=alt.Color('Type:N', scale=TYPE_SCALE, title='Type'),
        tooltip=[
            alt.Tooltip('Type:N',  title='Type'),
            alt.Tooltip('Total:Q', title='₹ Total', format=',.0f'),
        ],
    ).properties(height=min(height, 320))

    text = alt.Chart(summary).mark_text(radius=160, size=13).encode(
        theta=alt.Theta('Total:Q', stack=True),
        text=alt.Text('Total:Q', format=',.0f'),
        color=alt.Color('Type:N', scale=TYPE_SCALE, legend=None),
    )

    st.altair_chart(chart + text, use_container_width=True)

    total = summary['Total'].sum()
    for _, row in summary.iterrows():
        pct = row['Total'] / total * 100 if total else 0
        st.markdown(f"- **{row['Type'].title()}**: ₹{row['Total']:,.0f} &nbsp;({pct:.1f}%)")


# ─────────────────────────────────────────────────────────────
# 7. Bank breakdown
# ─────────────────────────────────────────────────────────────

def _render_bank_breakdown(converted_df, height):
    st.markdown("##### 🏦 Bank Breakdown — monthly debit per bank")

    if converted_df is None or converted_df.empty:
        st.info("No transaction data available.")
        return

    df = converted_df.copy()
    ts = 'timestamp' if 'timestamp' in df.columns else ('date' if 'date' in df.columns else None)
    if ts is None or 'Bank' not in df.columns:
        st.info("Bank or date column not found.")
        return

    df['_ts'] = pd.to_datetime(df[ts], errors='coerce')
    df = df.dropna(subset=['_ts'])
    df['Amount_numeric'] = pd.to_numeric(df.get('Amount', 0), errors='coerce').fillna(0.0)
    df['YearMonth'] = df['_ts'].dt.to_period('M').astype(str)

    if 'Type' in df.columns:
        df = df[df['Type'].astype(str).str.lower().str.strip() == 'debit']

    if df.empty:
        st.info("No debit transactions found.")
        return

    agg = df.groupby(['YearMonth', 'Bank'])['Amount_numeric'].sum().reset_index()
    agg.columns = ['Month', 'Bank', 'Amount']
    month_order = sorted(agg['Month'].unique(), key=lambda x: pd.to_datetime(x + "-01"))

    chart = alt.Chart(agg).mark_bar().encode(
        x=alt.X('Month:N',  sort=month_order, title='Month'),
        y=alt.Y('Amount:Q', title='₹ Debited',  axis=alt.Axis(format=",.0f")),
        color=alt.Color('Bank:N', title='Bank'),
        tooltip=[
            alt.Tooltip('Month:N',  title='Month'),
            alt.Tooltip('Bank:N',   title='Bank'),
            alt.Tooltip('Amount:Q', title='₹ Debited', format=',.0f'),
        ],
    ).interactive()

    st.altair_chart(chart.properties(height=height), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# 8. Day-of-week pattern
# ─────────────────────────────────────────────────────────────

def _render_dow_pattern(converted_df, height):
    st.markdown("##### 📅 Day-of-Week Spend Pattern — where do you spend most?")

    if converted_df is None or converted_df.empty:
        st.info("No transaction data available.")
        return

    df = converted_df.copy()
    ts = 'timestamp' if 'timestamp' in df.columns else ('date' if 'date' in df.columns else None)
    if ts is None:
        st.info("No date column found.")
        return

    df['_ts'] = pd.to_datetime(df[ts], errors='coerce')
    df = df.dropna(subset=['_ts'])
    df['Amount_numeric'] = pd.to_numeric(df.get('Amount', 0), errors='coerce').fillna(0.0)

    if 'Type' in df.columns:
        df = df[df['Type'].astype(str).str.lower().str.strip() == 'debit']

    if df.empty:
        st.info("No debit transactions found.")
        return

    df['DayOfWeek'] = df['_ts'].dt.day_name()
    df['Date']      = df['_ts'].dt.normalize()

    daily      = df.groupby(['Date', 'DayOfWeek'])['Amount_numeric'].sum().reset_index()
    avg_by_dow = daily.groupby('DayOfWeek')['Amount_numeric'].mean().reset_index()
    avg_by_dow.columns = ['Day', 'Avg_Spend']

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    bars = alt.Chart(avg_by_dow).mark_bar().encode(
        x=alt.X('Day:N',       sort=day_order, title='Day of Week', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Avg_Spend:Q', title='Avg ₹ Spent',  axis=alt.Axis(format=",.0f")),
        color=alt.condition(
            alt.datum.Day == 'Saturday',
            alt.value('#e6550d'),
            alt.condition(
                alt.datum.Day == 'Sunday',
                alt.value('#e6550d'),
                alt.value('#4c78a8'),
            ),
        ),
        tooltip=[
            alt.Tooltip('Day:N',       title='Day'),
            alt.Tooltip('Avg_Spend:Q', title='Avg ₹ Spent', format=',.0f'),
        ],
    ).properties(height=height)

    avg_line = alt.Chart(
        pd.DataFrame({'avg': [avg_by_dow['Avg_Spend'].mean()]})
    ).mark_rule(color='red', strokeDash=[4, 4], strokeWidth=2).encode(
        y='avg:Q',
        tooltip=[alt.Tooltip('avg:Q', title='Overall avg', format=',.0f')],
    )

    st.altair_chart((bars + avg_line).interactive(), use_container_width=True)
    st.caption("🟧 Weekends highlighted in orange · Red dashed line = overall average")
