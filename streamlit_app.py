# streamlit_app.py  -- minimal starter app (placeholders for modularization)
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

# ---------------- Page config ----------------
st.set_page_config(page_title="Daily Spend (starter)", layout="wide")
st.title("💳 Daily Spending — Starter (minimal, runs now)")

# =========================
# PLACEHOLDERS / FUTURE IMPORTS
# =========================
# Later we'll replace these comments with real imports:
#   from io import read_google_sheet          # -> io.py
#   from transform import convert_columns_and_derives, compute_daily_totals   # -> transform.py
#   from charts import render_chart           # -> charts.py
#
# For now this starter app performs minimal cleaning itself so it starts reliably.

# ---------------- Sidebar: data source (upload) ----------------
with st.sidebar:
    st.header("Data input")
    data_source = st.radio("Load data from", ["Upload CSV/XLSX", "Use sample data"], index=0)
    uploaded = None
    if data_source == "Upload CSV/XLSX":
        uploaded = st.file_uploader("Upload CSV or XLSX (HDFC / Indian Bank / IFTTT sheet)", type=["csv", "xlsx"])
    st.markdown("---")
    st.write("Series to include")
    show_debit = st.checkbox("Debit (Total_Spent)", value=True)
    show_credit = st.checkbox("Credit (Total_Credit)", value=True)
    st.markdown("---")
    if st.button("Refresh"):
        st.experimental_rerun()

# ---------------- Load / parse raw data ----------------
def load_from_upload(uploaded_file):
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        return df
    except Exception as e:
        st.error(f"Failed to parse upload: {e}")
        return pd.DataFrame()

def sample_data():
    # small sample for starter app: 30 days of random transactions
    today = datetime.utcnow().date()
    rows = []
    for i in range(30):
        d = today - timedelta(days=29-i)
        # alternate debit/credit, random-ish amounts
        amt = (i % 5 + 1) * 100
        if i % 7 == 0:
            # credit occasionally
            amt = -amt
            t = "credit"
        else:
            t = "debit"
        rows.append({"timestamp": pd.to_datetime(d), "description": f"Sample txn {i+1}", "Amount": amt, "Type": t})
    return pd.DataFrame(rows)

if data_source == "Upload CSV/XLSX":
    df_raw = load_from_upload(uploaded)
    if df_raw.empty:
        st.info("No valid upload detected — using sample data. Upload a CSV/XLSX to replace it.")
        df_raw = sample_data()
else:
    df_raw = sample_data()

# ---------------- Minimal cleaning (temporary; will be replaced by transform.py) ----------------
def minimal_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()

    # If there is no obvious timestamp column, try 'date' or fallback to index / sample
    ts_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower() or "timestamp" in c.lower()]
    if ts_cols:
        df['timestamp'] = pd.to_datetime(df[ts_cols[0]], errors='coerce')
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    else:
        # if Amount and description exist but no timestamp, generate synthetic based on sample order
        if 'Amount' not in df.columns:
            # try to find any numeric-like column
            for c in df.columns:
                try:
                    pd.to_numeric(df[c].astype(str).str.replace(r'[^\d\.\-]', '', regex=True))
                    df.rename(columns={c: 'Amount'}, inplace=True)
                    break
                except Exception:
                    pass
        if 'timestamp' not in df.columns or df['timestamp'].isna().all():
            df['timestamp'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df)).to_series().reset_index(drop=True)

    # normalize Amount
    if 'Amount' not in df.columns:
        # guess amount-like column
        for c in df.columns:
            if c.lower() in ('amt', 'amount', 'value', 'txn amount', 'debit', 'credit'):
                df.rename(columns={c: 'Amount'}, inplace=True)
                break
    # coerce numeric
    df['Amount'] = pd.to_numeric(df.get('Amount', 0), errors='coerce').fillna(0)

    # Type inference: if Type column present use it; otherwise map sign
    if 'Type' not in df.columns:
        df['Type'] = df['Amount'].apply(lambda x: 'debit' if x > 0 else ('credit' if x < 0 else None))

    # canonical date
    df['date'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.date

    return df

converted_df = minimal_cleaning(df_raw)

# ---------------- Compute daily totals (temporary; will be replaced by transform.py) ----------------
def compute_daily_totals_minimal(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=['Date', 'Total_Spent', 'Total_Credit'])
    w = df.copy()
    w['Amount_numeric'] = pd.to_numeric(w.get('Amount', 0), errors='coerce').fillna(0.0).astype('float64')
    # treat positive amounts as spend (debit), negatives as credit (sample)
    debit = w[w['Amount_numeric'] > 0].groupby(pd.to_datetime(w['date']).dt.normalize())['Amount_numeric'].sum().reset_index().rename(columns={'date':'Date','Amount_numeric':'Total_Spent'})
    credit = w[w['Amount_numeric'] < 0].groupby(pd.to_datetime(w['date']).dt.normalize())['Amount_numeric'].sum().reset_index().rename(columns={'date':'Date','Amount_numeric':'Total_Credit'})
    # debit sums are positive; credit sums are negative — convert credit to positive numbers for charting
    if 'Total_Credit' in credit.columns:
        credit['Total_Credit'] = credit['Total_Credit'].abs()
    merged = pd.merge(debit, credit, on='Date', how='outer').fillna(0)
    merged['Date'] = pd.to_datetime(merged['Date']).dt.normalize()
    if 'Total_Spent' not in merged.columns:
        merged['Total_Spent'] = 0.0
    if 'Total_Credit' not in merged.columns:
        merged['Total_Credit'] = 0.0
    merged = merged.sort_values('Date').reset_index(drop=True)
    return merged

merged = compute_daily_totals_minimal(converted_df)

# ---------------- Sidebar filters (year/month) ----------------
with st.sidebar:
    st.header("Filters")
    if not merged.empty:
        min_date = merged['Date'].min().date()
        max_date = merged['Date'].max().date()
        sel_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    else:
        sel_range = (None, None)

# apply date filter to merged
plot_df = merged.copy()
if sel_range and sel_range[0] and sel_range[1]:
    plot_df = plot_df[(plot_df['Date'].dt.date >= sel_range[0]) & (plot_df['Date'].dt.date <= sel_range[1])]

# ---------------- Series selection ----------------
series_selected = []
if show_debit:
    series_selected.append('Total_Spent')
if show_credit:
    series_selected.append('Total_Credit')

# ---------------- Chart (Altair) ----------------
st.subheader("Daily Spend and Credit (starter chart)")
if plot_df.empty:
    st.info("No data to display.")
else:
    # create long form for plotting
    plot_df_long = plot_df.melt(id_vars='Date', value_vars=[c for c in ['Total_Spent','Total_Credit'] if c in series_selected],
                                var_name='Type', value_name='Amount').sort_values('Date')
    if plot_df_long.empty:
        st.info("No series selected or no data for chosen range.")
    else:
        chart = alt.Chart(plot_df_long).mark_line(point=True).encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Amount:Q', title='Amount', axis=alt.Axis(format=",.0f")),
            color=alt.Color('Type:N', title='Type'),
            tooltip=[alt.Tooltip('Date:T', title='Date', format='%Y-%m-%d'),
                     alt.Tooltip('Type:N', title='Type'),
                     alt.Tooltip('Amount:Q', title='Amount', format=',')]
        ).interactive()
        st.altair_chart(chart.properties(height=420), use_container_width=True)

# ---------------- Show rows and download ----------------
st.subheader("Rows (matching selection)")
rows_df = converted_df.copy()
# apply date filter to rows
if sel_range and sel_range[0] and sel_range[1]:
    rows_df = rows_df[(pd.to_datetime(rows_df['timestamp']).dt.date >= sel_range[0]) &
                      (pd.to_datetime(rows_df['timestamp']).dt.date <= sel_range[1])]

if rows_df.empty:
    st.write("No rows to show for the current selection.")
else:
    st.dataframe(rows_df.reset_index(drop=True))
    csv_bytes = rows_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download rows (CSV)", csv_bytes, file_name="transactions_rows.csv", mime="text/csv")

# ---------------- Footer note ----------------
st.markdown(
    """
    ---
    **Notes:**  
    - This is a *starter* app that runs immediately.  
    - Later we'll replace the inline cleaning & aggregation with `transform.py`, fetch-sheet logic with `io.py`, and richer charts with `charts.py`.  
    - Places marked in comments are the spots to plug those modules in.
    """
)
