# streamlit_app.py - main Streamlit entrypoint (with safe Google Sheet handling + bank filter + grouped filters)
import streamlit as st
import pandas as pd
import importlib
from datetime import datetime, timedelta, date, time as dt_time
import math

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

# optional I/O helper (must be present for Google Sheets features)
try:
    import io_helpers as io_mod
except Exception:
    io_mod = None

# charts module
try:
    charts_mod = importlib.import_module("charts")
except Exception:
    charts_mod = None

# ------------------ Read secrets (preferred) ------------------
_secrets = getattr(st, "secrets", {}) or {}
SHEET_ID_SECRET = _secrets.get("SHEET_ID")
RANGE_SECRET = _secrets.get("RANGE")
APPEND_RANGE_SECRET = _secrets.get("APPEND_RANGE")
CREDS_FILE_SECRET = _secrets.get("CREDS_FILE")  # optional

# ------------------ Sidebar: data source & options ------------------
with st.sidebar:
    st.header("⚙️ Data Source & Settings")
    data_source = st.radio(
        "Load data from",
        ["Google Sheet", "Upload CSV/XLSX", "Use sample data"],
        index=0,
        key="data_source_radio" # Added key for stability
    )

    use_google = isinstance(data_source, str) and data_source.lower().startswith("google")

    # Show Google Sheet settings only if selected
    if use_google:
        if SHEET_ID_SECRET:
            SHEET_ID = SHEET_ID_SECRET
            st.caption(f"Using Sheet ID from secrets.")
        else:
            SHEET_ID = st.text_input("Google Sheet ID (between /d/ and /edit)", value="")

        if RANGE_SECRET:
            RANGE = RANGE_SECRET
        else:
            RANGE = st.text_input("History sheet name or range", value="History Transactions")

        if APPEND_RANGE_SECRET:
            APPEND_RANGE = APPEND_RANGE_SECRET
        else:
            APPEND_RANGE = st.text_input("Append sheet name or range", value="Append Transactions")

        if CREDS_FILE_SECRET:
            CREDS_FILE = CREDS_FILE_SECRET
        else:
            CREDS_FILE = st.text_input("Service Account JSON File (optional)", value="creds/service_account.json")
    else:
        # Define placeholders if not using Google Sheets to avoid errors later
        SHEET_ID, RANGE, APPEND_RANGE, CREDS_FILE = None, None, None, None


    if st.button("🔄 Refresh Data"):
        # Clear cache if needed, or just rerun
        st.cache_data.clear() # Example: clear cache on manual refresh
        st.experimental_rerun()

    st.markdown("---") # Separator

# ------------------ Helpers (rest of helpers remain the same) ------------------
def _get_creds_info():
    """Return plain creds dict or None (safe to pass into io_helpers functions)."""
    if io_mod is None:
        return None
    try:
        if hasattr(st, "secrets") and st.secrets and "gcp_service_account" in st.secrets:
            raw = st.secrets["gcp_service_account"]
            return io_mod.parse_service_account_secret(raw)
    except Exception:
        return None
    return None

def _read_sheet_with_index(spreadsheet_id: str, range_name: str, source_name: str, creds_info, creds_file):
    """ Read sheet, add index/source cols. """
    try:
        df = io_mod.read_google_sheet(spreadsheet_id, range_name, creds_info=creds_info, creds_file=creds_file)
    except Exception:
        return pd.DataFrame()
    if df is None:
        return pd.DataFrame()
    df = df.reset_index(drop=True)
    if not df.empty:
        df['_sheet_row_idx'] = df.index.astype(int)
    df['_source_sheet'] = source_name
    return df

def _to_pydate(val):
    """Coerce to python date or None."""
    if val is None: return None
    if isinstance(val, date) and not isinstance(val, datetime): return val
    if isinstance(val, datetime): return val.date()
    try:
        ts = pd.to_datetime(val, errors="coerce")
        return None if pd.isna(ts) else ts.date()
    except Exception: return None

def _ensure_min_max_order(min_d, max_d):
    """Ensure dates are ordered min -> max."""
    min_d = _to_pydate(min_d) or datetime.utcnow().date()
    max_d = _to_pydate(max_d) or datetime.utcnow().date()
    if min_d > max_d:
        min_d, max_d = max_d, min_d
    return min_d, max_d

# ------------------ Data loaders (safe wrappers - remain the same) ------------------
def load_from_upload(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None: return pd.DataFrame()
    try:
        return pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith(".csv") else pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Failed to parse upload: {e}"); return pd.DataFrame()

def load_from_sheet_safe(sheet_id: str, range_name: str, creds_file: str) -> pd.DataFrame:
    if io_mod is None: st.error("io_helpers.py needed for Google Sheets."); return pd.DataFrame()
    creds_info = _get_creds_info()
    try:
        return io_mod.read_google_sheet(sheet_id, range_name, creds_info=creds_info, creds_file=creds_file)
    except Exception as e:
        st.error(f"Failed to read Google Sheet: {e}"); return pd.DataFrame()

# ------------------ Sample data fallback (remains the same) ------------------
def sample_data():
    today = datetime.utcnow().date()
    rows = []
    for i in range(30):
        d = today - timedelta(days=29 - i)
        amt = (i % 5 + 1) * 100
        t = "credit" if i % 7 == 0 else "debit"
        if t == "credit": amt = -amt # Sample credits are negative
        rows.append({"timestamp": pd.to_datetime(d), "description": f"Sample txn {i+1}", "Amount": amt, "Type": t})
    return pd.DataFrame(rows)

# ------------------ Helper: bank detection (remains the same) ------------------
def add_bank_column(df: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    df = df.copy()
    if 'Bank' in df.columns and not overwrite:
        df['Bank'] = df['Bank'].astype(str).where(df['Bank'].notna(), 'Unknown')
        return df

    cand_cols = ['bank', 'account', 'description', 'message', 'narration']
    def _row_text(row):
        return " ".join([str(row[c]) for c in cand_cols if c in row.index and pd.notna(row[c])]).lower()

    bank_map = {'hdfc': 'HDFC Bank', 'indian bank': 'Indian Bank', 'indianbank': 'Indian Bank'}
    combined = df.apply(_row_text, axis=1)
    detected = ['Unknown'] * len(df)
    for i, text in enumerate(combined):
        for patt, name in bank_map.items():
            if patt in text:
                detected[i] = name; break
    df['Bank'] = detected
    return df

# ------------------ Load raw data according to selection ----------------
uploaded = None
if data_source == "Upload CSV/XLSX":
    # Moved uploader here to avoid showing it when Google Sheet is selected
    uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"])

sheet_full_df = pd.DataFrame()
df_raw = pd.DataFrame()

if use_google:
    if not SHEET_ID:
        st.warning("Enter Google Sheet ID in the sidebar, or add secrets.")
        st.stop()
    if io_mod is None: st.error("io_helpers.py needed for Google Sheets."); st.stop()

    with st.spinner("Fetching Google Sheets..."):
        creds_info = _get_creds_info()
        history_df = _read_sheet_with_index(SHEET_ID, RANGE, "history", creds_info, CREDS_FILE)
        append_df = _read_sheet_with_index(SHEET_ID, APPEND_RANGE, "append", creds_info, CREDS_FILE)
        if history_df.empty and append_df.empty:
             st.error(f"No data found in Google Sheet '{SHEET_ID}' ranges '{RANGE}' or '{APPEND_RANGE}'. Check Sheet ID and range names."); st.stop()
        sheet_full_df = pd.concat([history_df, append_df], ignore_index=True, sort=False)
        if '_sheet_row_idx' not in sheet_full_df.columns: sheet_full_df['_sheet_row_idx'] = pd.NA

        if 'is_deleted' in sheet_full_df.columns:
            deleted_mask = sheet_full_df['is_deleted'].astype(str).str.lower().isin(['true', 't', '1', 'yes'])
            df_raw = sheet_full_df.loc[~deleted_mask].copy().reset_index(drop=True)
        else:
            df_raw = sheet_full_df.copy().reset_index(drop=True)

elif data_source == "Upload CSV/XLSX":
    df_raw = load_from_upload(uploaded)
    if df_raw.empty: st.info("Upload a file or select another data source."); st.stop()
else:  # sample data
    df_raw = sample_data()

if df_raw.empty:
    st.warning("No data loaded."); st.stop()

# ------------------ Transform using transform.py ------------------
with st.spinner("Cleaning data..."):
    converted_df = transform.convert_columns_and_derives(df_raw.copy()) # Pass copy
    converted_df = add_bank_column(converted_df, overwrite=False)

# --- Define Filters in Sidebar Expanders ---
with st.sidebar:
    # --- Expander 1: Chart & Metric Options ---
    with st.expander("📊 Chart & Metric Options", expanded=False):
        st.write("**Chart Display**")
        show_debit_chart = st.checkbox("Show Debit on Chart", value=True, key="show_debit_chart")
        show_credit_chart = st.checkbox("Show Credit on Chart", value=True, key="show_credit_chart")
        chart_type_select = st.selectbox("Chart Type", ["Daily line", "Monthly bars", "Top categories (Top-N)"], index=0, key="chart_type_select")

        st.write("**Chart Date Filter**")
        # Base calculation on 'merged_all_totals' which reflects *all* data before amount filtering
        try:
             # Calculate totals once here for filter options
             with st.spinner("Calculating overall totals for filters..."):
                 merged_all_totals = transform.compute_daily_totals(converted_df.copy()) # Use unfiltered data
             if not merged_all_totals.empty:
                 merged_all_totals['Date'] = pd.to_datetime(merged_all_totals['Date']).dt.normalize()
                 all_years = sorted(merged_all_totals['Date'].dt.year.unique().tolist())
             else: all_years = [datetime.utcnow().year]
        except Exception:
             all_years = [datetime.utcnow().year]
             merged_all_totals = pd.DataFrame()

        years_opts_chart = ['All'] + [str(y) for y in all_years]
        sel_year_chart = st.selectbox("Chart Year", years_opts_chart, index=0, key="sel_year_chart")

        month_map_chart = {i: pd.Timestamp(1900, i, 1).strftime('%B') for i in range(1, 13)}
        if not merged_all_totals.empty:
            month_frame_chart = merged_all_totals.copy()
            if sel_year_chart != 'All':
                month_frame_chart = month_frame_chart[month_frame_chart['Date'].dt.year == int(sel_year_chart)]
            month_nums_chart = sorted(month_frame_chart['Date'].dt.month.unique().tolist())
            month_choices_chart = [month_map_chart[m] for m in month_nums_chart]
        else:
            month_choices_chart = list(month_map_chart.values())

        sel_months_chart = st.multiselect("Chart Month(s)", options=month_choices_chart, default=month_choices_chart, key="sel_months_chart")

        st.markdown("---")
        st.write("**Top-Right Metric Options**")
        metric_year_opts = [str(y) for y in all_years]
        default_metric_year_idx = len(metric_year_opts) - 1 if metric_year_opts else 0
        metric_year = st.selectbox("Metric Year", options=metric_year_opts, index=default_metric_year_idx, key="metric_year")

        metric_month_choices = list(month_map_chart.values()) # Use full month list
        default_metric_month_idx = datetime.utcnow().month - 1
        metric_month = st.selectbox("Metric Month", options=metric_month_choices, index=default_metric_month_idx, key="metric_month")

        replace_outliers_checkbox = st.checkbox("Clean outliers for metric avg", value=False, key="replace_outliers_checkbox")
        st.caption("Uses IQR rule. Affects only the top-right metric.")

    # --- Expander 2: Transaction Filters ---
    with st.expander("🔍 Transaction Filters", expanded=True): # Expand by default
        st.write("**Filter Transactions By**")
        banks_available = sorted([b for b in converted_df['Bank'].unique() if pd.notna(b)])
        sel_banks = st.multiselect("Bank(s)", options=banks_available, default=banks_available, key="sel_banks")

        min_amount_filter = st.number_input(
            "Amount >= (0 to disable)",
            min_value=0.0, value=0.0, step=100.0, format="%.2f", key="min_amount_filter"
        )

        st.markdown("---")
        st.write("**Select Date Range for Table & Totals**")
        # Build safe min/max from available *unfiltered* rows for date pickers
        try:
            valid_dates_all = pd.to_datetime(converted_df.get('timestamp', converted_df.get('date')), errors='coerce').dropna()
            min_date_overall, max_date_overall = _ensure_min_max_order(valid_dates_all.min(), valid_dates_all.max()) if not valid_dates_all.empty else (datetime.utcnow().date() - timedelta(days=365), datetime.utcnow().date())
        except Exception:
            max_date_overall = datetime.utcnow().date(); min_date_overall = max_date_overall - timedelta(days=365)


        totals_mode = st.radio("Mode", ["Single date", "Date range"], index=0, key="totals_mode")

        if totals_mode == "Single date":
            today = datetime.utcnow().date()
            default_date = max(min_date_overall, min(today, max_date_overall))
            selected_date = st.date_input("Pick date", value=default_date, min_value=min_date_overall, max_value=max_date_overall, key="selected_date")
            start_sel, end_sel = selected_date, selected_date
        else: # Date range
            dr = st.date_input("Pick start & end", value=(min_date_overall, max_date_overall), min_value=min_date_overall, max_value=max_date_overall, key="date_range_picker")
            if isinstance(dr, (tuple, list)) and len(dr) == 2:
                s_raw, e_raw = dr
            else: s_raw, e_raw = dr, dr # Handle single date selection in range mode
            s, e = _to_pydate(s_raw), _to_pydate(e_raw)
            # Clamp and order
            start_sel = max(min_date_overall, s) if s else min_date_overall
            end_sel = min(max_date_overall, e) if e else max_date_overall
            if start_sel > end_sel: start_sel, end_sel = end_sel, start_sel

# --- Apply Core Filters (Bank, Amount) ---
# Start with the cleaned data
converted_df_filtered = converted_df.copy()

# Apply Bank Filter
if sel_banks:
    converted_df_filtered = converted_df_filtered[converted_df_filtered['Bank'].isin(sel_banks)]

# Apply Amount Filter (Globally)
if min_amount_filter > 0.0:
    amount_col_name = next((col for col in converted_df_filtered.columns if col.lower() == 'amount'), None)
    if amount_col_name:
        try:
            converted_df_filtered[amount_col_name] = pd.to_numeric(converted_df_filtered[amount_col_name], errors='coerce')
            # Use abs() if you want to filter credits > threshold too, otherwise just filter debits
            # converted_df_filtered = converted_df_filtered[converted_df_filtered[amount_col_name].abs() >= min_amount_filter].copy()
            converted_df_filtered = converted_df_filtered[converted_df_filtered[amount_col_name] >= min_amount_filter].copy()

        except Exception as e: st.warning(f"Could not apply amount filter: {e}")
    else: st.warning("Amount filter needs 'Amount' column.")


# --- Compute daily totals (NOW uses filtered data) ---
with st.spinner("Computing daily totals..."):
    merged = transform.compute_daily_totals(converted_df_filtered.copy()) # Pass copy

# --- Prepare Chart Data (Apply Chart Date Filters) ---
plot_df = merged.copy() if merged is not None else pd.DataFrame()
if not plot_df.empty:
     plot_df['Date'] = pd.to_datetime(plot_df['Date']).dt.normalize() # Ensure Date is datetime
     if sel_year_chart != 'All':
         plot_df = plot_df[plot_df['Date'].dt.year == int(sel_year_chart)]
     if sel_months_chart:
         inv_map_chart = {v: k for k, v in month_map_chart.items()}
         selected_month_nums_chart = [inv_map_chart[m] for m in sel_months_chart if m in inv_map_chart]
         if selected_month_nums_chart:
             plot_df = plot_df[plot_df['Date'].dt.month.isin(selected_month_nums_chart)]

     plot_df = plot_df.sort_values('Date').reset_index(drop=True)
     # Ensure numeric columns exist for chart
     for col in ['Total_Spent', 'Total_Credit']:
         if col in plot_df.columns:
             plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce').fillna(0.0)
         else:
             plot_df[col] = 0.0
else:
     plot_df = pd.DataFrame(columns=['Date', 'Total_Spent', 'Total_Credit']) # Empty structure


# ------------------ Top-Right Metric Calculation & Rendering (remains mostly the same) ------------------
# Uses 'merged_all_totals' calculated earlier for a consistent monthly average view
def _safe_mean(s): s=pd.to_numeric(s, errors='coerce').dropna(); return float(s.mean()) if not s.empty else None
def _format_currency(v): return f"₹{v:,.2f}" if v is not None else "N/A"
def _month_year_to_date(y_str, m_name):
    try: y = int(y_str)
    except: y = datetime.utcnow().year
    try: m = pd.to_datetime(m_name, format='%B').month
    except: m=datetime.utcnow().month
    return y, m

def compute_month_avg_from_merged(mrg_df, yr, mo, replace_outliers=False):
    if mrg_df is None or mrg_df.empty: return None, 0, {}
    df = mrg_df.copy(); df['Date'] = pd.to_datetime(df['Date'])
    mask = (df['Date'].dt.year == int(yr)) & (df['Date'].dt.month == int(mo))
    dfm = df.loc[mask]; vals = pd.to_numeric(dfm.get('Total_Spent', 0), errors='coerce').fillna(0.0)
    if dfm.empty: return None, 0, {}
    if len(vals) < 3 or not replace_outliers: return _safe_mean(vals), len(vals), {"n": len(vals)}
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75); iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    is_outlier = (vals < lower) | (vals > upper); non_outliers = vals[~is_outlier]
    replacement = float(non_outliers.median()) if not non_outliers.empty else float(vals.median())
    vals_replaced = vals.copy(); vals_replaced[is_outlier] = replacement
    return _safe_mean(vals_replaced), len(vals), {"outliers_replaced": int(is_outlier.sum()), "n": len(vals)}

try:
    metric_year_int, metric_month_int = _month_year_to_date(metric_year, metric_month)
    # Use merged_all_totals for consistent metric view
    metric_avg, _, _ = compute_month_avg_from_merged(merged_all_totals, metric_year_int, metric_month_int, replace_outliers_checkbox)
    prev_dt = datetime(metric_year_int, metric_month_int, 1) - pd.DateOffset(months=1)
    prev_avg, _, _ = compute_month_avg_from_merged(merged_all_totals, prev_dt.year, prev_dt.month, replace_outliers_checkbox)
except Exception: metric_avg, prev_avg = None, None

col_a, col_b, col_c = st.columns([6, 2, 2])
with col_c: # Metric display
    label = pd.Timestamp(metric_year_int, metric_month_int, 1).strftime("%b-%y")
    metric_text = _format_currency(metric_avg)
    delta_html = "<span style='font-size:14px;color:gray'>N/A</span>"
    if metric_avg is not None and prev_avg is not None:
        diff = metric_avg - prev_avg
        try: delta_label = f"{(diff / abs(prev_avg) * 100.0):+.1f}%" if abs(prev_avg) > 1e-9 else f"{diff:+.2f}"
        except: delta_label = f"{diff:+.2f}"
        color = "red" if diff > 0 else ("green" if diff < 0 else "gray")
        arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "►")
        delta_html = f"<span style='font-size:14px;color:{color}; font-weight:600'>{arrow} {delta_label}</span>"
    st.markdown(f"<div style='text-align:right; padding:8px 4px;'><div style='font-size:12px;color:#666;margin-bottom:2px'>{label}</div><div style='font-size:20px;font-weight:700'>{metric_text}</div><div>{delta_html}</div></div>", unsafe_allow_html=True)


# ------------------ Chart Rendering ------------------
st.subheader("📊 Daily Spend and Credit")
if plot_df.empty:
    st.info("No data available for the selected chart filters.")
else:
    if charts_mod is not None:
        series_selected_chart = []
        if show_debit_chart: series_selected_chart.append('Total_Spent')
        if show_credit_chart: series_selected_chart.append('Total_Credit')
        try:
             charts_mod.render_chart(
                 plot_df=plot_df, # Use chart-filtered data
                 converted_df=converted_df_filtered, # Pass globally filtered for Top-N
                 chart_type=chart_type_select,
                 series_selected=series_selected_chart,
                 top_n=5 # Example, maybe make this configurable later
             )
        except Exception as chart_err:
             st.error(f"Failed to render chart: {chart_err}")
             st.exception(chart_err)
    else:
        st.info("charts.py not available.")


# ------------------ Rows view & download ------------------
st.subheader("📝 Rows (matching selection)")

# Start from globally filtered (bank + amount)
rows_df = converted_df_filtered.copy()

# Ensure timestamp exists for date filtering
if 'timestamp' in rows_df.columns: rows_df['timestamp'] = pd.to_datetime(rows_df['timestamp'], errors='coerce')
elif 'date' in rows_df.columns: rows_df['timestamp'] = pd.to_datetime(rows_df['date'], errors='coerce')
else: rows_df['timestamp'] = pd.NaT

# Apply Table/Totals Date Range filter (start_sel, end_sel)
if start_sel and end_sel and 'timestamp' in rows_df.columns and not rows_df['timestamp'].isnull().all():
    rows_df = rows_df[ (rows_df['timestamp'].dt.date >= start_sel) & (rows_df['timestamp'].dt.date <= end_sel) ]

# --- Display Table ---
_desired = ['timestamp', 'bank', 'type', 'amount', 'suspicious', 'message']
col_map = {c.lower(): c for c in rows_df.columns}
display_cols = [col_map[d] for d in _desired if d in col_map]
if not any(c.lower() == 'timestamp' for c in display_cols) and 'date' in col_map: display_cols.insert(0, col_map['date'])

if not display_cols: # Fallback
    st.warning("Could not find preferred columns - showing raw data."); display_df = rows_df
else:
    display_df = rows_df[display_cols].copy()
    # Formatting... (Timestamp, Amount, Renaming)
    ts_col = next((c for c in display_df.columns if c.lower() in ['timestamp', 'date']), None)
    if ts_col: display_df[ts_col] = pd.to_datetime(display_df[ts_col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
    amt_col = next((c for c in display_df.columns if c.lower() == 'amount'), None)
    if amt_col: display_df[amt_col] = pd.to_numeric(display_df[amt_col], errors='coerce')
    pretty_rename = {'timestamp':'Timestamp','date':'Timestamp','bank':'Bank','type':'Type','amount':'Amount','suspicious':'Suspicious','message':'Message'}
    display_df = display_df.rename(columns={c:pretty_rename[c.lower()] for c in display_df.columns if c.lower() in pretty_rename})
    final_order = [c for c in ['Timestamp', 'Bank', 'Type', 'Amount', 'Suspicious', 'Message'] if c in display_df.columns]
    display_df = display_df[final_order]

st.dataframe(display_df.reset_index(drop=True), use_container_width=True, height=420)
csv_bytes = display_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Rows (CSV)", csv_bytes, file_name="transactions_rows.csv", mime="text/csv")


# ------------------ Build selectable mapping for Delete UI ------------------
selectable = False; selectable_labels = []; selectable_label_to_target = {}
if use_google and io_mod is not None and not sheet_full_df.empty:
    # Use rows_df which is already filtered by bank, amount, AND date range
    map_df = rows_df.copy()
    if '_sheet_row_idx' in map_df.columns and '_source_sheet' in map_df.columns:
        for i, r in map_df.iterrows():
             try: idx = int(r['_sheet_row_idx'])
             except: continue # Skip if index is invalid
             ts = pd.to_datetime(r.get('timestamp', '')).strftime('%Y-%m-%d %H:%M') if pd.notna(r.get('timestamp')) else ''
             amt = r.get('Amount', ''); msg = str(r.get('Message', r.get('message', '')) )[:60]; src = r.get('_source_sheet', 'history')
             label = f"[{src}:{idx+2}] {ts} | {amt} | {msg}" # Show GSheet row index (idx+2 assumes 1 header row)
             tgt_range = APPEND_RANGE if src == 'append' else RANGE
             selectable_labels.append(label); selectable_label_to_target[label] = (tgt_range, idx)
        if selectable_labels: selectable = True


# ------------------ Delete UI (remains the same logic, uses filtered selectable_labels) ------------------
if use_google and io_mod is not None and not sheet_full_df.empty:
    st.markdown("---"); st.write("🗑️ Bulk Actions (Google Sheet only)")
    col_a, col_b = st.columns([3, 1])
    with col_a:
        selected_labels = st.multiselect("Select rows to remove (soft-delete)", options=selectable_labels, key="delete_multi") if selectable else []
        if not selectable: st.info("Row selection unavailable (cannot map rows to sheet).")
    with col_b: remove_btn = st.button("Remove selected", key="remove_rows_btn", disabled=not selectable)

    if remove_btn and selected_labels:
        groups = {}; any_error = False; total_updated = 0; creds_info = _get_creds_info()
        for lbl in selected_labels:
             tgt = selectable_label_to_target.get(lbl); rng, idx = tgt if tgt else (None, None)
             if rng and idx is not None: groups.setdefault(rng, []).append(idx)
        for rng, indices in groups.items():
             try:
                 res = io_mod.mark_rows_deleted(SHEET_ID, rng, creds_info, CREDS_FILE, indices)
                 if res.get('status') == 'ok': total_updated += res.get('updated', 0)
                 else: st.error(f"Failed ({rng}): {res.get('message')}"); any_error = True
             except Exception as e: st.error(f"Error ({rng}): {e}"); any_error = True
        if not any_error: st.success(f"Marked {total_updated} rows deleted."); st.cache_data.clear(); st.experimental_rerun()


# ------------------ Add New Row UI (remains the same logic) ------------------
if use_google and io_mod is not None:
    st.markdown("---"); st.write("➕ Add New Row (to Append sheet)")
    with st.expander("Open add row form"):
        with st.form("add_row_form", clear_on_submit=True):
             new_date = st.date_input("Date", value=start_sel, min_value=min_date_overall, max_value=max_date_overall)
             banks_for_add = sorted(list(set(banks_available + ['Other (enter below)'])))
             bank_choice = st.selectbox("Bank", options=banks_for_add)
             bank_other = st.text_input("Bank (custom)") if bank_choice == "Other (enter below)" else ""
             txn_type = st.selectbox("Type", options=["debit", "credit"])
             amount = st.number_input("Amount (₹)", value=0.0, step=1.0, format="%.2f")
             message = st.text_input("Message / Description", value="")
             submit_add = st.form_submit_button("Save New Row")

             if submit_add:
                 chosen_bank = bank_other if bank_other else (bank_choice if bank_choice != "Other (enter below)" else "Unknown")
                 dt_combined = datetime.combine(new_date, datetime.utcnow().time()) # Combine date + current time
                 new_row = {'DateTime': dt_combined.strftime("%Y-%m-%d %H:%M:%S"), 'timestamp': dt_combined, 'date': dt_combined.date(),
                            'Bank': chosen_bank, 'Type': txn_type, 'Amount': amount, 'Message': message, 'is_deleted': 'false'}
                 creds_info = _get_creds_info()
                 try:
                     res = io_mod.append_new_row(SHEET_ID, APPEND_RANGE, new_row, creds_info, CREDS_FILE, RANGE)
                     if res.get('status') == 'ok': st.success("Appended row."); st.cache_data.clear(); st.experimental_rerun()
                     else: st.error(f"Failed: {res.get('message')}")
                 except Exception as e: st.error(f"Error: {e}")


# ------------------ Totals for selected date / range (Uses globally filtered + date filtered data) ------------------
totals_heading = f"Totals — {start_sel}" if start_sel == end_sel else f"Totals — {start_sel} → {end_sel}"
st.markdown(f"### {totals_heading}")

try:
    # Use rows_df which is already filtered by Bank, Amount, AND Date Range
    sel_df = rows_df.copy()
    col_map_lower = {c.lower(): c for c in sel_df.columns}
    amount_col = col_map_lower.get('amount'); type_col = col_map_lower.get('type')
    credit_sum, debit_sum, credit_count, debit_count = 0.0, 0.0, 0, 0

    if sel_df.empty:
        st.info(f"No transactions match all filters for the selected date range.")
    elif amount_col:
        sel_df[amount_col] = pd.to_numeric(sel_df[amount_col], errors='coerce').fillna(0.0)
        if type_col:
            sel_df['type_norm'] = sel_df[type_col].astype(str).str.lower().str.strip()
            credit_mask = sel_df['type_norm'] == 'credit'; debit_mask = sel_df['type_norm'] == 'debit'
            credit_sum = sel_df.loc[credit_mask, amount_col].sum(); debit_sum = sel_df.loc[debit_mask, amount_col].sum()
            credit_count = int(credit_mask.sum()); debit_count = int(debit_mask.sum())
        else: # Heuristic if no Type column
             for _, r in sel_df.iterrows():
                 amt = r[amount_col]
                 # Basic guess: negative amounts are credits (adjust if needed)
                 if amt < 0: credit_sum += abs(amt); credit_count += 1
                 else: debit_sum += amt; debit_count += 1
    else: st.warning("Cannot calculate totals: 'Amount' column not found.")


    col1, col2, col3 = st.columns(3)
    col1.metric("Credits", f"₹{credit_sum:,.0f}", f"{credit_count} txns")
    col2.metric("Debits", f"₹{debit_sum:,.0f}", f"{debit_count} txns")
    col3.metric("Net", f"₹{(credit_sum - debit_sum):,.0f}") # Simple Net

except Exception as e:
    st.error(f"Failed to compute totals: {e}")
