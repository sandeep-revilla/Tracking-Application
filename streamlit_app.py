# streamlit_app.py - main Streamlit entrypoint (with safe Google Sheet handling + bank filter + grouped filters + type filter)
import streamlit as st
import pandas as pd
import importlib
from datetime import datetime, timedelta, date, time as dt_time
import math

st.set_page_config(page_title="Daily Spend", layout="wide")
st.title("💳 Daily Spending")

# ------------------ Imports / placeholders ------------------
try:
    transform = importlib.import_module("transform")
except Exception as e:
    st.error("transform.py missing or failing to import.")
    st.exception(e)
    st.stop()

try:
    import io_helpers as io_mod
except Exception:
    io_mod = None

try:
    charts_mod = importlib.import_module("charts")
except Exception:
    charts_mod = None

# ------------------ Read secrets ------------------
_secrets = getattr(st, "secrets", {}) or {}
SHEET_ID_SECRET = _secrets.get("SHEET_ID")
RANGE_SECRET = _secrets.get("RANGE")
APPEND_RANGE_SECRET = _secrets.get("APPEND_RANGE")
CREDS_FILE_SECRET = _secrets.get("CREDS_FILE")

# ------------------ Sidebar: data source & options (Now Collapsible) ------------------
with st.sidebar:
    with st.expander("⚙️ Data Source & Settings", expanded=False):
        data_source = st.radio(
            "Load data from",
            ["Google Sheet", "Upload CSV/XLSX", "Use sample data"],
            index=0,
            key="data_source_radio"
        )
        use_google = isinstance(data_source, str) and data_source.lower().startswith("google")

        if use_google:
            SHEET_ID = SHEET_ID_SECRET if SHEET_ID_SECRET else st.text_input("Google Sheet ID", value="")
            RANGE = RANGE_SECRET if RANGE_SECRET else st.text_input("History sheet name", value="History Transactions")
            APPEND_RANGE = APPEND_RANGE_SECRET if APPEND_RANGE_SECRET else st.text_input("Append sheet name", value="Append Transactions")
            CREDS_FILE = CREDS_FILE_SECRET if CREDS_FILE_SECRET else st.text_input("Service Account JSON (optional)", value="creds/service_account.json")
            if SHEET_ID_SECRET: st.caption("Using Sheet ID from secrets.") # Show confirmation if secret is used
        else:
            SHEET_ID, RANGE, APPEND_RANGE, CREDS_FILE = None, None, None, None

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear(); st.experimental_rerun()

    st.markdown("---")

# --- Rest of Helpers ---
def _get_creds_info():
    if io_mod is None: return None
    try:
        if hasattr(st, "secrets") and "gcp_service_account" in st.secrets:
            return io_mod.parse_service_account_secret(st.secrets["gcp_service_account"])
    except Exception: return None
    return None

def _read_sheet_with_index(spreadsheet_id: str, range_name: str, source_name: str, creds_info, creds_file):
    try: df = io_mod.read_google_sheet(spreadsheet_id, range_name, creds_info=creds_info, creds_file=creds_file)
    except Exception: return pd.DataFrame()
    if df is None: return pd.DataFrame()
    df = df.reset_index(drop=True); df['_sheet_row_idx'] = df.index.astype(int); df['_source_sheet'] = source_name
    return df

def _to_pydate(val):
    if val is None: return None
    if isinstance(val, date) and not isinstance(val, datetime): return val
    if isinstance(val, datetime): return val.date()
    try: ts = pd.to_datetime(val, errors="coerce"); return None if pd.isna(ts) else ts.date()
    except Exception: return None

def _ensure_min_max_order(min_d, max_d):
    min_d = _to_pydate(min_d) or datetime.utcnow().date(); max_d = _to_pydate(max_d) or datetime.utcnow().date()
    if min_d > max_d: min_d, max_d = max_d, min_d
    return min_d, max_d

# --- Data Loaders ---
def load_from_upload(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None: return pd.DataFrame()
    try: return pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith(".csv") else pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e: st.error(f"Failed to parse upload: {e}"); return pd.DataFrame()

# --- Sample Data ---
def sample_data():
    today = datetime.utcnow().date(); rows = []
    for i in range(30):
        d = today - timedelta(days=29 - i); amt = (i % 5 + 1) * 100
        t = "credit" if i % 7 == 0 else "debit"; amt = abs(amt)
        rows.append({"timestamp": pd.to_datetime(d), "description": f"Sample txn {i+1}", "Amount": amt, "Type": t})
    return pd.DataFrame(rows)

# --- Bank Detection ---
def add_bank_column(df: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    df = df.copy()
    if 'Bank' in df.columns and not overwrite: df['Bank'] = df['Bank'].astype(str).where(df['Bank'].notna(), 'Unknown'); return df
    cand_cols = ['bank', 'account', 'description', 'message', 'narration']
    def _row_text(row): return " ".join([str(row[c]) for c in cand_cols if c in row.index and pd.notna(row[c])]).lower()
    bank_map = {'hdfc': 'HDFC Bank', 'indian bank': 'Indian Bank', 'indianbank': 'Indian Bank'}
    combined = df.apply(_row_text, axis=1); detected = ['Unknown'] * len(df)
    for i, text in enumerate(combined):
        for patt, name in bank_map.items():
            if patt in text: detected[i] = name; break
    df['Bank'] = detected
    return df

# ------------------ Load raw data ------------------
uploaded = None
if data_source == "Upload CSV/XLSX":
    uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"])

sheet_full_df = pd.DataFrame(); df_raw = pd.DataFrame()

if use_google:
    if not SHEET_ID: st.warning("Enter Google Sheet ID in sidebar settings."); st.stop()
    if io_mod is None: st.error("io_helpers.py needed for Google Sheets."); st.stop()
    with st.spinner("Fetching Google Sheets..."):
        creds_info = _get_creds_info()
        history_df = _read_sheet_with_index(SHEET_ID, RANGE, "history", creds_info, CREDS_FILE)
        append_df = _read_sheet_with_index(SHEET_ID, APPEND_RANGE, "append", creds_info, CREDS_FILE)
        if history_df.empty and append_df.empty: st.error(f"No data in Google Sheet '{SHEET_ID}'."); st.stop()
        sheet_full_df = pd.concat([history_df, append_df], ignore_index=True, sort=False)
        if '_sheet_row_idx' not in sheet_full_df.columns: sheet_full_df['_sheet_row_idx'] = pd.NA
        if 'is_deleted' in sheet_full_df.columns:
            deleted_mask = sheet_full_df['is_deleted'].astype(str).str.lower().isin(['true', 't', '1', 'yes'])
            df_raw = sheet_full_df.loc[~deleted_mask].copy().reset_index(drop=True)
        else: df_raw = sheet_full_df.copy().reset_index(drop=True)
elif data_source == "Upload CSV/XLSX":
    df_raw = load_from_upload(uploaded)
    if df_raw.empty: st.info("Upload file or select another source."); st.stop()
else: df_raw = sample_data()

if df_raw.empty: st.warning("No data loaded."); st.stop()

# ------------------ Transform data ------------------
with st.spinner("Cleaning data..."):
    converted_df = transform.convert_columns_and_derives(df_raw.copy())
    converted_df = add_bank_column(converted_df, overwrite=False)
    amt_col_raw = next((c for c in converted_df.columns if c.lower() == 'amount'), None)
    if amt_col_raw: converted_df[amt_col_raw] = pd.to_numeric(converted_df[amt_col_raw], errors='coerce')
    type_col_raw = next((c for c in converted_df.columns if c.lower() == 'type'), None)
    if type_col_raw: converted_df['type_normalized'] = converted_df[type_col_raw].astype(str).str.lower().str.strip()
    else:
         if amt_col_raw: # Infer type if missing
             converted_df['type_normalized'] = converted_df[amt_col_raw].apply(lambda x: 'credit' if pd.notna(x) and x < 0 else 'debit')
             converted_df[amt_col_raw] = converted_df[amt_col_raw].abs()
         else: converted_df['type_normalized'] = 'unknown'

# --- Define Filters ---
sel_banks, min_amount_filter, sel_types = [], 0.0, ['debit', 'credit']
start_sel, end_sel = None, None
sel_year_chart, sel_months_chart = 'All', []
metric_year, metric_month = None, None
replace_outliers_checkbox = False
show_debit_chart, show_credit_chart = True, True
chart_type_select = "Daily line"
min_date_overall, max_date_overall = datetime.utcnow().date() - timedelta(days=365), datetime.utcnow().date() # Default range

# Calculate overall totals & date range *once* before sidebar definition
try:
    with st.spinner("Analyzing data range..."): merged_all_totals = transform.compute_daily_totals(converted_df.copy())
    if not merged_all_totals.empty: merged_all_totals['Date'] = pd.to_datetime(merged_all_totals['Date']).dt.normalize(); all_years = sorted(merged_all_totals['Date'].dt.year.unique().tolist())
    else: all_years = [datetime.utcnow().year]; merged_all_totals = pd.DataFrame()
    valid_dates_all = pd.to_datetime(converted_df.get('timestamp', converted_df.get('date')), errors='coerce').dropna()
    min_date_overall, max_date_overall = _ensure_min_max_order(valid_dates_all.min(), valid_dates_all.max()) if not valid_dates_all.empty else (min_date_overall, max_date_overall)
except Exception as e:
     st.warning(f"Could not determine full date range: {e}")
     all_years = [datetime.utcnow().year]; merged_all_totals = pd.DataFrame()


with st.sidebar:
    with st.expander("📊 Chart & Metric Options", expanded=False):
        st.write("**Chart Display**"); show_debit_chart = st.checkbox("Show Debit", value=True, key="show_debit_chart"); show_credit_chart = st.checkbox("Show Credit", value=True, key="show_credit_chart"); chart_type_select = st.selectbox("Chart Type", ["Daily line", "Monthly bars", "Top categories (Top-N)"], index=0, key="chart_type_select")
        st.write("**Chart Date Filter**"); years_opts_chart = ['All'] + [str(y) for y in all_years]; sel_year_chart = st.selectbox("Chart Year", years_opts_chart, index=0, key="sel_year_chart"); month_map_chart = {i: pd.Timestamp(1900,i,1).strftime('%B') for i in range(1,13)}; month_choices_chart=list(month_map_chart.values()); sel_months_chart=st.multiselect("Chart Month(s)", options=month_choices_chart, default=month_choices_chart, key="sel_months_chart")
        st.markdown("---"); st.write("**Top-Right Metric**"); metric_year_opts=[str(y) for y in all_years]; default_metric_year_idx=len(metric_year_opts)-1 if metric_year_opts else 0; metric_year=st.selectbox("Metric Year",options=metric_year_opts,index=default_metric_year_idx,key="metric_year"); metric_month_choices=list(month_map_chart.values()); default_metric_month_idx=datetime.utcnow().month-1; metric_month=st.selectbox("Metric Month",options=metric_month_choices,index=default_metric_month_idx,key="metric_month"); replace_outliers_checkbox=st.checkbox("Clean outliers for metric",value=False,key="replace_outliers_checkbox"); st.caption("Uses IQR. Affects only top-right avg.")

    with st.expander("🔍 Transaction Filters", expanded=True):
        st.write("**Filter Transactions By**"); banks_available = sorted([b for b in converted_df['Bank'].unique() if pd.notna(b) and b != 'Unknown'])+['Unknown']; sel_banks=st.multiselect("Bank(s)",options=banks_available, default=banks_available, key="sel_banks"); min_amount_filter=st.number_input("Amount >= (0 to disable)", min_value=0.0, value=0.0, step=100.0, format="%.2f", key="min_amount_filter"); transaction_types=['debit','credit']; sel_types=st.multiselect("Type(s)",options=transaction_types, default=transaction_types, key="sel_types")
        st.markdown("---"); st.write("**Select Date Range (Table & Totals)**"); totals_mode = st.radio("Mode", ["Single date", "Date range"], index=1, key="totals_mode") # Default to range
        if totals_mode == "Single date": today = datetime.utcnow().date(); default_date = max(min_date_overall, min(today, max_date_overall)); selected_date=st.date_input("Pick date",value=default_date, min_value=min_date_overall, max_value=max_date_overall, key="selected_date"); start_sel, end_sel = selected_date, selected_date
        else: dr = st.date_input("Pick start & end", value=(min_date_overall, max_date_overall), min_value=min_date_overall, max_value=max_date_overall, key="date_range_picker"); s_raw, e_raw = dr if isinstance(dr,(tuple,list)) and len(dr)==2 else (dr,dr); s,e = _to_pydate(s_raw), _to_pydate(e_raw); start_sel=max(min_date_overall, s) if s else min_date_overall; end_sel=min(max_date_overall, e) if e else max_date_overall; if start_sel > end_sel: start_sel, end_sel = end_sel, start_sel

# --- Apply Core Filters ---
converted_df_filtered = converted_df.copy()
if sel_banks: converted_df_filtered = converted_df_filtered[converted_df_filtered['Bank'].isin(sel_banks)]
if min_amount_filter > 0.0:
    amt_col = next((c for c in converted_df_filtered.columns if c.lower()=='amount'), None)
    if amt_col: try: converted_df_filtered = converted_df_filtered[converted_df_filtered[amt_col] >= min_amount_filter].copy() except: pass
if sel_types and 'type_normalized' in converted_df_filtered.columns:
    valid_sel_types = [t.lower().strip() for t in sel_types if isinstance(t, str)]
    if valid_sel_types: converted_df_filtered = converted_df_filtered[converted_df_filtered['type_normalized'].isin(valid_sel_types)].copy()

# --- Compute daily totals ---
with st.spinner("Computing daily totals..."): merged = transform.compute_daily_totals(converted_df_filtered.copy())

# --- Prepare Chart Data ---
plot_df = merged.copy() if merged is not None else pd.DataFrame()
if not plot_df.empty:
     plot_df['Date'] = pd.to_datetime(plot_df['Date']).dt.normalize()
     if sel_year_chart != 'All': plot_df = plot_df[plot_df['Date'].dt.year == int(sel_year_chart)]
     if sel_months_chart:
         inv_map_chart = {v: k for k, v in month_map_chart.items()}
         selected_month_nums_chart = [inv_map_chart[m] for m in sel_months_chart if m in inv_map_chart]
         if selected_month_nums_chart: plot_df = plot_df[plot_df['Date'].dt.month.isin(selected_month_nums_chart)]
     plot_df = plot_df.sort_values('Date').reset_index(drop=True)
     for col in ['Total_Spent', 'Total_Credit']: plot_df[col] = pd.to_numeric(plot_df.get(col, 0), errors='coerce').fillna(0.0)
else: plot_df = pd.DataFrame(columns=['Date', 'Total_Spent', 'Total_Credit'])


# --- Top-Right Metric ---
# --- CORRECTED _month_year_to_date FUNCTION ---
def _month_year_to_date(year_str, month_name):
    """Return a (year:int, month:int) tuple from inputs."""
    try:
        y = int(year_str)
    except Exception:
        y = datetime.utcnow().year # Fallback year
    try:
        m = pd.to_datetime(month_name, format='%B').month # Try parsing full month name
    except Exception:
        # Fallback month
        m = datetime.utcnow().month
    return y, m
# --- END CORRECTION ---

def _safe_mean(s): s=pd.to_numeric(s, errors='coerce').dropna(); return float(s.mean()) if not s.empty else None
def _format_currency(v): return f"₹{v:,.2f}" if v is not None else "N/A"
def compute_month_avg_from_merged(mrg_df, yr, mo, replace_outliers=False):
    if mrg_df is None or mrg_df.empty: return None, 0, {}
    df = mrg_df.copy(); df['Date'] = pd.to_datetime(df['Date'])
    mask = (df['Date'].dt.year == int(yr)) & (df['Date'].dt.month == int(mo))
    dfm = df.loc[mask]; vals = pd.to_numeric(dfm.get('Total_Spent', 0), errors='coerce').fillna(0.0)
    if dfm.empty: return None, 0, {}
    if len(vals) < 3 or not replace_outliers: return _safe_mean(vals), len(vals), {"n": len(vals)}
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75); iqr = q3 - q1; lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    is_outlier = (vals < lower) | (vals > upper); non_outliers = vals[~is_outlier]
    replacement = float(non_outliers.median()) if not non_outliers.empty else float(vals.median())
    vals_replaced = vals.copy(); vals_replaced[is_outlier] = replacement
    return _safe_mean(vals_replaced), len(vals), {"outliers_replaced": int(is_outlier.sum()), "n": len(vals)}

try:
    metric_year_int, metric_month_int = _month_year_to_date(metric_year, metric_month)
    metric_avg, _, _ = compute_month_avg_from_merged(merged_all_totals, metric_year_int, metric_month_int, replace_outliers_checkbox)
    prev_dt = datetime(metric_year_int, metric_month_int, 1) - pd.DateOffset(months=1)
    prev_avg, _, _ = compute_month_avg_from_merged(merged_all_totals, prev_dt.year, prev_dt.month, replace_outliers_checkbox)
except: metric_avg, prev_avg = None, None
col_a, col_b, col_c = st.columns([6, 2, 2])
with col_c: # Metric display
    label = pd.Timestamp(metric_year_int, metric_month_int, 1).strftime("%b-%y"); metric_text = _format_currency(metric_avg)
    delta_html = "<span style='font-size:14px;color:gray'>N/A</span>"
    if metric_avg is not None and prev_avg is not None: diff = metric_avg - prev_avg; try: delta_label = f"{(diff / abs(prev_avg) * 100.0):+.1f}%" if abs(prev_avg)>1e-9 else f"{diff:+.2f}" except: delta_label=f"{diff:+.2f}"; color="red" if diff>0 else ("green" if diff<0 else "gray"); arrow="▲" if diff>0 else ("▼" if diff<0 else "►"); delta_html=f"<span style='font-size:14px;color:{color}; font-weight:600'>{arrow} {delta_label}</span>"
    st.markdown(f"<div style='text-align:right; padding:8px 4px;'><div style='font-size:12px;color:#666;margin-bottom:2px'>{label}</div><div style='font-size:20px;font-weight:700'>{metric_text}</div><div>{delta_html}</div></div>", unsafe_allow_html=True)


# --- Chart Rendering ---
st.subheader("📊 Daily Spend and Credit")
if plot_df.empty: st.info("No data for selected chart filters.")
elif charts_mod:
    series_selected_chart = []
    if show_debit_chart: series_selected_chart.append('Total_Spent')
    if show_credit_chart: series_selected_chart.append('Total_Credit')
    try: charts_mod.render_chart(plot_df, converted_df_filtered, chart_type_select, series_selected_chart, 5)
    except Exception as chart_err: st.error(f"Chart render error: {chart_err}"); st.exception(chart_err)
else: st.info("charts.py not available.")


# --- Rows view & download ---
st.subheader("📝 Rows (matching selection)")
rows_df = converted_df_filtered.copy() # Start from globally filtered
if 'timestamp' in rows_df.columns: rows_df['timestamp'] = pd.to_datetime(rows_df['timestamp'], errors='coerce')
elif 'date' in rows_df.columns: rows_df['timestamp'] = pd.to_datetime(rows_df['date'], errors='coerce')
else: rows_df['timestamp'] = pd.NaT
if start_sel and end_sel and 'timestamp' in rows_df.columns and not rows_df['timestamp'].isnull().all(): # Apply date range
    rows_df = rows_df[ (rows_df['timestamp'].dt.date >= start_sel) & (rows_df['timestamp'].dt.date <= end_sel) ]

_desired = ['timestamp', 'bank', 'type', 'amount', 'message']
col_map = {c.lower(): c for c in rows_df.columns}; display_cols = [col_map[d] for d in _desired if d in col_map]
if not any(c.lower() == 'timestamp' for c in display_cols) and 'date' in col_map: display_cols.insert(0, col_map['date'])

if rows_df.empty: st.info("No rows match all filters for this date range."); display_df = pd.DataFrame(columns=['Timestamp', 'Bank', 'Type', 'Amount', 'Message'])
elif not display_cols: st.warning("Preferred columns not found."); display_df = rows_df
else:
    display_df = rows_df[display_cols].copy()
    ts_col = next((c for c in display_df.columns if c.lower() in ['timestamp','date']), None)
    if ts_col: display_df[ts_col] = pd.to_datetime(display_df[ts_col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
    amt_col = next((c for c in display_df.columns if c.lower() == 'amount'), None) # Amount already numeric
    type_col_disp = next((c for c in display_df.columns if c.lower() == 'type'), None)
    if type_col_disp: display_df = display_df.rename(columns={type_col_disp: 'Type'})
    elif 'type_normalized' in display_df.columns: display_df = display_df.rename(columns={'type_normalized': 'Type'})
    pretty_rename = {'timestamp':'Timestamp','date':'Timestamp','bank':'Bank','amount':'Amount','message':'Message'}
    display_df = display_df.rename(columns={c:pretty_rename[c.lower()] for c in display_df.columns if c.lower() in pretty_rename})
    final_order = [c for c in ['Timestamp', 'Bank', 'Type', 'Amount', 'Message'] if c in display_df.columns]
    display_df = display_df[final_order]

st.dataframe(display_df.reset_index(drop=True), use_container_width=True, height=420)
csv_bytes = display_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Rows (CSV)", csv_bytes, file_name="transactions_rows.csv", mime="text/csv")


# --- Build selectable mapping for Delete UI ---
selectable = False; selectable_labels = []; selectable_label_to_target = {}
if use_google and io_mod and not sheet_full_df.empty:
    map_df = rows_df.copy() # Use final rows_df
    if '_sheet_row_idx' in map_df.columns and '_source_sheet' in map_df.columns:
        for i, r in map_df.iterrows():
             try: idx = int(r['_sheet_row_idx'])
             except: continue
             ts = pd.to_datetime(r.get('timestamp','')).strftime('%Y-%m-%d %H:%M') if pd.notna(r.get('timestamp')) else ''; amt = r.get('Amount',''); msg = str(r.get('Message',r.get('message','')))[:60]; src = r.get('_source_sheet','history')
             label = f"[{src}:{idx+2}] {ts} | {amt} | {msg}"; tgt_range = APPEND_RANGE if src=='append' else RANGE
             selectable_labels.append(label); selectable_label_to_target[label] = (tgt_range, idx)
        if selectable_labels: selectable = True


# --- Delete UI ---
if use_google and io_mod and not sheet_full_df.empty:
    st.markdown("---"); st.write("🗑️ Bulk Actions (Google Sheet only)")
    col_a, col_b = st.columns([3, 1])
    with col_a: selected_labels = st.multiselect("Select rows to remove", options=selectable_labels, key="delete_multi") if selectable else []
    with col_b: remove_btn = st.button("Remove selected", key="remove_rows_btn", disabled=not selectable or not selected_labels)
    if not selectable and use_google: st.info("Row selection unavailable (cannot map rows).")

    if remove_btn and selected_labels:
        groups = {}; any_error = False; total_updated = 0; creds_info = _get_creds_info()
        for lbl in selected_labels: tgt = selectable_label_to_target.get(lbl); rng,idx=tgt if tgt else (None,None); groups.setdefault(rng,[]).append(idx) if rng else None
        for rng, indices in groups.items():
             if not indices: continue
             try: res = io_mod.mark_rows_deleted(SHEET_ID, rng, creds_info, CREDS_FILE, indices)
             except Exception as e: st.error(f"Error ({rng}): {e}"); any_error=True; continue
             if res.get('status')=='ok': total_updated += res.get('updated',0)
             else: st.error(f"Failed ({rng}): {res.get('message')}"); any_error = True
        if not any_error: st.success(f"Marked {total_updated} rows deleted."); st.cache_data.clear(); st.experimental_rerun()


# --- Add New Row UI ---
if use_google and io_mod:
    st.markdown("---"); st.write("➕ Add New Row (to Append sheet)")
    with st.expander("Open add row form"):
        with st.form("add_row_form", clear_on_submit=True):
             new_date = st.date_input("Date", value=start_sel or datetime.utcnow().date(), min_value=min_date_overall, max_value=max_date_overall)
             banks_for_add = sorted(list(set(banks_available + ['Other (enter below)', 'Unknown'])))
             bank_choice = st.selectbox("Bank", options=banks_for_add, index=banks_for_add.index('Unknown') if 'Unknown' in banks_for_add else 0)
             bank_other = st.text_input("Bank (custom)") if bank_choice=="Other (enter below)" else ""
             txn_type = st.selectbox("Type", options=["debit", "credit"])
             amount = st.number_input("Amount (₹)", value=0.0, step=1.0, format="%.2f")
             message = st.text_input("Message / Description", value="")
             submit_add = st.form_submit_button("Save New Row")
             if submit_add:
                 chosen_bank = bank_other if bank_other else (bank_choice if bank_choice != "Other (enter below)" else "Unknown")
                 dt_combined = datetime.combine(new_date, datetime.utcnow().time())
                 new_row = {'DateTime': dt_combined.strftime("%Y-%m-%d %H:%M:%S"), 'timestamp': dt_combined, 'date': dt_combined.date(), 'Bank': chosen_bank, 'Type': txn_type, 'Amount': amount, 'Message': message, 'is_deleted': 'false'}
                 creds_info = _get_creds_info()
                 try:
                     res = io_mod.append_new_row(SHEET_ID, APPEND_RANGE, new_row, creds_info, CREDS_FILE, RANGE)
                     if res.get('status')=='ok': st.success("Appended row."); st.cache_data.clear(); st.experimental_rerun()
                     else: st.error(f"Failed: {res.get('message')}")
                 except Exception as e: st.error(f"Error: {e}")


# --- Totals Display ---
totals_heading = f"Totals — {start_sel}" if start_sel == end_sel else f"Totals — {start_sel} → {end_sel}"
st.markdown(f"### {totals_heading}")
try:
    sel_df = rows_df.copy() # Use final rows_df
    amt_col = next((c for c in sel_df.columns if c.lower()=='amount'), None)
    type_col_norm = 'type_normalized' if 'type_normalized' in sel_df.columns else None
    credit_sum, debit_sum, credit_count, debit_count = 0.0, 0.0, 0, 0
    if sel_df.empty: st.info(f"No transactions match all filters for this date range.")
    elif amt_col and type_col_norm: # Calculate only if essential columns exist
         credit_mask = sel_df[type_col_norm] == 'credit'; debit_mask = sel_df[type_col_norm] == 'debit'
         credit_sum = sel_df.loc[credit_mask, amt_col].sum(); debit_sum = sel_df.loc[debit_mask, amt_col].sum()
         credit_count = int(credit_mask.sum()); debit_count = int(debit_mask.sum())
    else: st.warning("Cannot calculate totals: 'Amount' or 'Type' column missing/invalid.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Credits", f"₹{credit_sum:,.0f}", f"{credit_count} txns")
    col2.metric("Debits", f"₹{debit_sum:,.0f}", f"{debit_count} txns")
    col3.metric("Net", f"₹{(credit_sum - debit_sum):,.0f}")
except Exception as e: st.error(f"Failed to compute totals: {e}")
