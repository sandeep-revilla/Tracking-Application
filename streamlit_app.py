# streamlit_app.py - main Streamlit entrypoint (with safe Google Sheet handling + bank filter)
import streamlit as st
import pandas as pd
import importlib
from datetime import datetime, timedelta, date

st.set_page_config(page_title="Daily Spend", layout="wide")
st.title("💳 Daily Spending")

# ------------------ Imports / placeholders ------------------
# transform.py must exist and provide convert_columns_and_derives + compute_daily_totals
try:
    transform = importlib.import_module("transform")
except Exception as e:
    st.error("transform.py missing or failing to import. Add transform.py to the same directory.")
    st.exception(e)
    st.stop()

# optional I/O helper
try:
    import io_helpers as io_mod
except Exception:
    io_mod = None

# charts module
try:
    charts_mod = importlib.import_module("charts")
except Exception:
    charts_mod = None

# ------------------ Sidebar: data source & options ------------------
with st.sidebar:
    st.header("Data input & options")
    data_source = st.radio(
        "Load data from",
        ["Upload CSV/XLSX", "Google Sheet (optional)", "Use sample data"],
        index=0
    )

    SHEET_ID = st.text_input("Google Sheet ID (between /d/ and /edit)", value="1KZq_GLXdMBfQUhtp-NA8Jg-flxOppw7kFuIN6y_nOXk")
    RANGE = st.text_input("Range or Sheet Name", value="History Transactions")
    CREDS_FILE = st.text_input("Service Account JSON File (optional)", value="creds/service_account.json")

    st.markdown("---")
    st.write("Series to include")
    show_debit = st.checkbox("Debit (Total_Spent)", value=True)
    show_credit = st.checkbox("Credit (Total_Credit)", value=True)

    st.markdown("---")
    st.write("Chart type (placeholder)")
    chart_type = st.selectbox("Chart type", ["Daily line", "Monthly bars", "Top categories (Top-N)"], index=0)

    st.markdown("---")
    if st.button("Refresh"):
        st.experimental_rerun()

# ------------------ Data loaders (safe wrappers) ------------------
def load_from_upload(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Failed to parse upload: {e}")
        return pd.DataFrame()


def load_from_sheet_safe(sheet_id: str, range_name: str, creds_file: str) -> pd.DataFrame:
    """
    Safe wrapper that:
      - reads st.secrets["gcp_service_account"] (if present),
      - parses it into a plain dict (using io_helpers.parse_service_account_secret),
      - passes that plain dict to io_helpers.read_google_sheet (a cached function).
    This avoids passing the Streamlit runtime 'st.secrets' object into a cached function.
    """
    if io_mod is None:
        st.error("io_helpers.py not available. Add io_helpers.py to the project to use Google Sheets.")
        return pd.DataFrame()

    creds_info = None
    # Parse st.secrets into a plain dict if present
    try:
        if hasattr(st, "secrets") and st.secrets and "gcp_service_account" in st.secrets:
            raw = st.secrets["gcp_service_account"]
            # parse_service_account_secret returns a plain dict
            creds_info = io_mod.parse_service_account_secret(raw)
    except Exception as e:
        st.warning(f"Failed to parse st.secrets['gcp_service_account']: {e}")
        creds_info = None

    # If creds_info is None, but a local creds_file exists, io_helpers will use that.
    try:
        return io_mod.read_google_sheet(sheet_id, range_name, creds_info=creds_info, creds_file=creds_file)
    except Exception as e:
        st.error(f"Failed to read Google Sheet: {e}")
        return pd.DataFrame()

# ------------------ Sample data fallback ------------------
def sample_data():
    today = datetime.utcnow().date()
    rows = []
    for i in range(30):
        d = today - timedelta(days=29 - i)
        amt = (i % 5 + 1) * 100
        if i % 7 == 0:
            amt = -amt
            t = "credit"
        else:
            t = "debit"
        rows.append({"timestamp": pd.to_datetime(d), "description": f"Sample txn {i+1}", "Amount": amt, "Type": t})
    return pd.DataFrame(rows)


# ------------------ Helper: bank detection & filtering ------------------
def add_bank_column(df: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    """
    Ensure df has a 'Bank' column. If missing (or overwrite=True), try to detect bank name
    from common text fields: 'bank', 'account', 'description', 'message', 'narration', 'beneficiary', 'merchant'.
    Returns a new DataFrame with 'Bank' column (string), unknowns marked 'Unknown'.
    """
    df = df.copy()
    if 'Bank' in df.columns and not overwrite:
        # normalize existing Bank column values to strings and fillna
        df['Bank'] = df['Bank'].astype(str).where(df['Bank'].notna(), None)
        df['Bank'] = df['Bank'].fillna('Unknown')
        return df

    # candidate text columns to scan
    cand_cols = ['bank', 'account', 'account_name', 'description', 'message', 'narration', 'merchant', 'beneficiary', 'note']
    # build a combined lowercased text for each row
    def _row_text(row):
        parts = []
        for c in cand_cols:
            if c in row.index and pd.notna(row[c]):
                parts.append(str(row[c]))
        return " ".join(parts).lower()

    # mapping patterns -> normalized bank name (extend as needed)
    bank_map = {
        'hdfc': 'HDFC Bank',
        'hdfc bank': 'HDFC Bank',
        'hdfcbank': 'HDFC Bank',
        'hdfc card': 'HDFC Bank',
        'hdfccredit': 'HDFC Bank',
        'indian bank': 'Indian Bank',
        'indianbank': 'Indian Bank',
        'indian bank ltd': 'Indian Bank',
        # add more heuristics if needed
    }

    # compute combined text column (vectorized using apply)
    try:
        combined = df.apply(_row_text, axis=1)
    except Exception:
        # fallback: if apply fails (weird dtypes), create empty series
        combined = pd.Series([''] * len(df), index=df.index)

    detected = []
    for text in combined:
        found = None
        for patt, name in bank_map.items():
            if patt in text:
                found = name
                break
        detected.append(found if found is not None else None)

    df['Bank'] = detected
    df['Bank'] = df['Bank'].fillna('Unknown')
    return df


# ------------------ Load raw data according to selection ----------------
uploaded = None
if data_source == "Upload CSV/XLSX":
    uploaded = st.file_uploader("Upload CSV or XLSX (HDFC / Indian Bank / IFTTT sheet)", type=["csv", "xlsx"])

if data_source == "Google Sheet (optional)":
    if not SHEET_ID:
        st.sidebar.info("Enter Google Sheet ID to enable sheet loading.")
        df_raw = pd.DataFrame()
    else:
        with st.spinner("Fetching Google Sheet..."):
            df_raw = load_from_sheet_safe(SHEET_ID, RANGE, CREDS_FILE)
elif data_source == "Upload CSV/XLSX":
    df_raw = load_from_upload(uploaded)
    if df_raw is None or df_raw.empty:
        st.info("No upload provided or file empty — using sample data.")
        df_raw = sample_data()
else:  # sample data
    df_raw = sample_data()

if df_raw is None or df_raw.empty:
    st.warning("No data loaded — upload a file or provide a Google Sheet ID, or use sample data.")
    st.stop()

# ------------------ Transform using transform.py ----------------
with st.spinner("Cleaning and deriving columns..."):
    converted_df = transform.convert_columns_and_derives(df_raw)

# ensure bank column exists (detect heuristically if needed)
converted_df = add_bank_column(converted_df, overwrite=False)

# ------------------ Sidebar: Bank filter ------------------
with st.sidebar:
    st.markdown("---")
    st.write("Filter by Bank")
    # detect unique banks and sort; prefer to show HDFC/Indian selected if present
    banks_detected = sorted([b for b in converted_df['Bank'].unique() if pd.notna(b)])
    # default selection logic: if HDFC/Indian present select them, else select all
    defaults = []
    if 'HDFC Bank' in banks_detected:
        defaults.append('HDFC Bank')
    if 'Indian Bank' in banks_detected:
        defaults.append('Indian Bank')
    if not defaults:
        defaults = banks_detected  # select all by default if HDFC/Indian not found
    sel_banks = st.multiselect("Banks", options=banks_detected, default=defaults)

# filter converted_df according to selection
if sel_banks:
    converted_df_filtered = converted_df[converted_df['Bank'].isin(sel_banks)].copy()
else:
    converted_df_filtered = converted_df.copy()

# ------------------ Compute daily totals (from filtered transactions) ----------------
with st.spinner("Computing daily totals..."):
    merged = transform.compute_daily_totals(converted_df_filtered)

# ------------------ Sidebar: Date filters (moved above the table so table can obey selection) ----------------
with st.sidebar:
    st.header("Filters")
    if not merged.empty:
        merged['Date'] = pd.to_datetime(merged['Date']).dt.normalize()
        years = sorted(merged['Date'].dt.year.unique().tolist())
        years_opts = ['All'] + [str(y) for y in years]
        sel_year = st.selectbox("Year", years_opts, index=0)

        if sel_year == 'All':
            month_frame = merged.copy()
        else:
            month_frame = merged[merged['Date'].dt.year == int(sel_year)]
        month_nums = sorted(month_frame['Date'].dt.month.unique().tolist())
        month_map = {i: pd.Timestamp(1900, i, 1).strftime('%B') for i in range(1, 13)}
        month_choices = [month_map[m] for m in month_nums]
        sel_months = st.multiselect("Month(s)", options=month_choices, default=month_choices)
    else:
        sel_year = 'All'
        sel_months = []

# Build safe min/max from available filtered rows (fallback to last 365 days)
try:
    tmp = converted_df_filtered.copy()
    if 'timestamp' in tmp.columns:
        tmp['timestamp'] = pd.to_datetime(tmp['timestamp'], errors='coerce')
    elif 'date' in tmp.columns:
        tmp['timestamp'] = pd.to_datetime(tmp['date'], errors='coerce')
    else:
        tmp['timestamp'] = pd.NaT
    valid_dates = tmp['timestamp'].dropna()
    if not valid_dates.empty:
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()
    else:
        max_date = datetime.utcnow().date()
        min_date = max_date - timedelta(days=365)
except Exception:
    max_date = datetime.utcnow().date()
    min_date = max_date - timedelta(days=365)

with st.sidebar:
    st.markdown("---")
    st.write("Select a date (or range) for the totals & table below")
    totals_mode = st.radio("Totals mode", ["Single date", "Date range"], index=0)
    if totals_mode == "Single date":
        selected_date = st.date_input("Pick date", value=datetime.utcnow().date(), min_value=min_date, max_value=max_date)
        selected_date_range_for_totals = (selected_date, selected_date)
    else:
        # date_input with tuple returns (start, end)
        default_range = (min_date, max_date)
        dr = st.date_input("Pick start & end", value=default_range, min_value=min_date, max_value=max_date)
        # dr might be a single date if user selects only one; ensure a tuple
        if isinstance(dr, (tuple, list)):
            selected_date_range_for_totals = (dr[0], dr[1])
        else:
            selected_date_range_for_totals = (dr, dr)

# ------------------ Apply year/month filters to aggregated plot_df ------------------
plot_df = merged.copy()
if sel_year != 'All':
    plot_df = plot_df[plot_df['Date'].dt.year == int(sel_year)]
if sel_months:
    month_map = {i: pd.Timestamp(1900, i, 1).strftime('%B') for i in range(1, 13)}
    inv_map = {v: k for k, v in month_map.items()}
    selected_month_nums = [inv_map[m] for m in sel_months if m in inv_map]
    if selected_month_nums:
        plot_df = plot_df[plot_df['Date'].dt.month.isin(selected_month_nums)]

plot_df = plot_df.sort_values('Date').reset_index(drop=True)
plot_df['Total_Spent'] = pd.to_numeric(plot_df.get('Total_Spent', 0), errors='coerce').fillna(0.0).astype('float64')
plot_df['Total_Credit'] = pd.to_numeric(plot_df.get('Total_Credit', 0), errors='coerce').fillna(0.0).astype('float64')

# ------------------ Chart & rendering ------------------
st.subheader("Daily Spend and Credit")
if plot_df.empty:
    st.info("No data for the selected filters.")
else:
    # delegate to charts.py if present
    if charts_mod is not None:
        series_selected = []
        if show_debit: series_selected.append('Total_Spent')
        if show_credit: series_selected.append('Total_Credit')
        charts_mod.render_chart(
            plot_df=plot_df,
            converted_df=converted_df_filtered,
            chart_type=chart_type,
            series_selected=series_selected,
            top_n=5
        )
    else:
        st.info("charts.py not available; install or add charts.py for visualizations.")

# ------------------ Rows view & download (show only selected columns) ----------------
st.subheader("Rows (matching selection)")

# start from filtered transactions so rows match the chart & bank selection
rows_df = converted_df_filtered.copy()

# ensure timestamp exists (original logic)
if 'timestamp' in rows_df.columns:
    rows_df['timestamp'] = pd.to_datetime(rows_df['timestamp'], errors='coerce')
else:
    if 'date' in rows_df.columns:
        rows_df['timestamp'] = pd.to_datetime(rows_df['date'], errors='coerce')
    else:
        rows_df['timestamp'] = pd.NaT

# apply selected date-range filter to rows (inclusive)
start_sel, end_sel = selected_date_range_for_totals
if isinstance(start_sel, datetime):
    start_sel = start_sel.date()
if isinstance(end_sel, datetime):
    end_sel = end_sel.date()

if start_sel and end_sel:
    rows_df = rows_df[
        (rows_df['timestamp'].dt.date >= start_sel) &
        (rows_df['timestamp'].dt.date <= end_sel)
    ]

# Desired columns (case-insensitive)
_desired = ['timestamp', 'bank', 'type', 'amount', 'suspicious', 'message']

# Map actual columns in the dataframe (preserve original casing)
col_map = {c.lower(): c for c in rows_df.columns}

display_cols = []
for d in _desired:
    if d in col_map:
        display_cols.append(col_map[d])

# If timestamp not found but 'date' exists, include it
if not any(c.lower() == 'timestamp' for c in display_cols) and 'date' in col_map:
    display_cols.insert(0, col_map['date'])

# If we couldn't find any of the desired columns, show the full table as a fallback
if not display_cols:
    st.warning("None of the preferred columns (timestamp, Bank, Type, Amount, Suspicious) were found — showing full table.")
    st.dataframe(rows_df.reset_index(drop=True), use_container_width=True, height=400)
    csv_bytes = rows_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download rows (CSV)", csv_bytes, file_name="transactions_rows.csv", mime="text/csv")
else:
    # Build the display dataframe with the columns we found
    display_df = rows_df[display_cols].copy()

    # Format timestamp-like column (if present)
    for c in display_df.columns:
        if c.lower() == 'timestamp' or c.lower() == 'date' or c.lower().startswith('date'):
            display_df[c] = pd.to_datetime(display_df[c], errors='coerce')
            # display nicely as ISO strings
            display_df[c] = display_df[c].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')

    # Coerce amount to numeric if present
    for c in display_df.columns:
        if c.lower() == 'amount':
            display_df[c] = pd.to_numeric(display_df[c], errors='coerce')

    # Pretty rename columns
    pretty_rename = {}
    for c in display_df.columns:
        lc = c.lower()
        if lc == 'timestamp' or lc == 'date' or lc.startswith('date'):
            pretty_rename[c] = 'Timestamp'
        elif lc == 'bank':
            pretty_rename[c] = 'Bank'
        elif lc == 'type':
            pretty_rename[c] = 'Type'
        elif lc == 'amount':
            pretty_rename[c] = 'Amount'
        elif lc == 'suspicious':
            pretty_rename[c] = 'Suspicious'
    if pretty_rename:
        display_df = display_df.rename(columns=pretty_rename)

    # Ensure order: Timestamp, Bank, Type, Amount, Suspicious (include whichever exist)
    final_order = [c for c in ['Timestamp', 'Bank', 'Type', 'Amount', 'Suspicious'] if c in display_df.columns]
    display_df = display_df[final_order]

    # Show table and download only these columns (use full width + fixed height to avoid excessive dragging)
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True, height=420)
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download rows (CSV)", csv_bytes, file_name="transactions_rows.csv", mime="text/csv")

# ------------------ Totals for selected date / range ------------------
# Build a human-friendly title (date + weekday for single day, or start → end for range)
if start_sel == end_sel:
    try:
        title_date = start_sel.strftime("%Y-%m-%d (%A)")
    except Exception:
        title_date = str(start_sel)
    totals_heading = f"Totals — {title_date}"
else:
    totals_heading = f"Totals — {start_sel} → {end_sel}"

st.markdown(f"### {totals_heading}")

try:
    # Work on a copy of filtered rows that already had timestamp normalized
    tmp_rows = converted_df_filtered.copy()
    if 'timestamp' in tmp_rows.columns:
        tmp_rows['timestamp'] = pd.to_datetime(tmp_rows['timestamp'], errors='coerce')
    else:
        if 'date' in tmp_rows.columns:
            tmp_rows['timestamp'] = pd.to_datetime(tmp_rows['date'], errors='coerce')
        else:
            tmp_rows['timestamp'] = pd.NaT

    # mask rows inside selection (inclusive)
    mask_sel = tmp_rows['timestamp'].dt.date.between(start_sel, end_sel)
    sel_df = tmp_rows[mask_sel].copy()

    # find case-insensitive column names
    col_map_lower = {c.lower(): c for c in sel_df.columns}
    amount_col = col_map_lower.get('amount')
    type_col = col_map_lower.get('type')

    if sel_df.empty:
        st.info(f"No transactions for selected date/range ({start_sel} to {end_sel}).")
        credit_sum = 0.0
        debit_sum = 0.0
        credit_count = 0
        debit_count = 0
    else:
        if amount_col is None:
            credit_sum = 0.0
            debit_sum = 0.0
            credit_count = 0
            debit_count = 0
        else:
            if type_col is not None:
                sel_df['type_norm'] = sel_df[type_col].astype(str).str.lower().str.strip()
                credit_mask = sel_df['type_norm'] == 'credit'
                debit_mask = sel_df['type_norm'] == 'debit'
                credit_sum = pd.to_numeric(sel_df.loc[credit_mask, amount_col], errors='coerce').fillna(0.0).sum()
                debit_sum = pd.to_numeric(sel_df.loc[debit_mask, amount_col], errors='coerce').fillna(0.0).sum()
                credit_count = int(credit_mask.sum())
                debit_count = int(debit_mask.sum())
            else:
                # no Type column - fallback heuristic
                credit_sum = 0.0
                debit_sum = 0.0
                credit_count = 0
                debit_count = 0
                text_cols = [c for c in sel_df.columns if sel_df[c].dtype == object]
                for _, r in sel_df.iterrows():
                    amt = pd.to_numeric(r.get(amount_col, 0), errors='coerce')
                    if pd.isna(amt):
                        amt = 0.0
                    txt = " ".join(str(r[c]) for c in text_cols if pd.notna(r[c])).lower()
                    if 'credit' in txt:
                        credit_sum += amt
                        credit_count += 1
                    else:
                        debit_sum += amt
                        debit_count += 1

    # Show as three metrics horizontally
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Credits ({start_sel} → {end_sel})", f"₹{credit_sum:,.0f}", f"{credit_count} txns")
    col2.metric(f"Debits ({start_sel} → {end_sel})", f"₹{debit_sum:,.0f}", f"{debit_count} txns")
    col3.metric("Net (Credits − Debits)", f"₹{(credit_sum - debit_sum):,.0f}")

except Exception as e:
    st.error(f"Failed to compute totals for selected date/range: {e}")
