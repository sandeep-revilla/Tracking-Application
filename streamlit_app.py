# streamlit_app.py - main Streamlit entrypoint (with safe Google Sheet handling + bank filter)
import streamlit as st
import pandas as pd
import importlib
from datetime import datetime, timedelta, date, time as dt_time

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
    st.header("Data input & options")
    data_source = st.radio(
        "Load data from",
        ["Upload CSV/XLSX", "Google Sheet (optional)", "Use sample data"],
        index=0
    )

    st.markdown("**Google Sheet configuration (preferred via Streamlit secrets)**")
    st.caption("Add the following keys to your Streamlit secrets.toml / deployment: SHEET_ID, RANGE, APPEND_RANGE, CREDS_FILE (optional)")

    # If secrets provided use them, otherwise show text inputs for local override
    if SHEET_ID_SECRET:
        st.write("SHEET_ID: (loaded from secrets)")
        SHEET_ID = SHEET_ID_SECRET
    else:
        SHEET_ID = st.text_input("Google Sheet ID (between /d/ and /edit)", value="")

    if RANGE_SECRET:
        st.write("RANGE (history sheet): (loaded from secrets)")
        RANGE = RANGE_SECRET
    else:
        RANGE = st.text_input("History sheet name or range", value="History Transactions")

    if APPEND_RANGE_SECRET:
        st.write("APPEND_RANGE (append sheet): (loaded from secrets)")
        APPEND_RANGE = APPEND_RANGE_SECRET
    else:
        APPEND_RANGE = st.text_input("Append sheet name or range", value="Append Transactions")

    if CREDS_FILE_SECRET:
        st.write("CREDS_FILE: (loaded from secrets)")
        CREDS_FILE = CREDS_FILE_SECRET
    else:
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

# ------------------ Helpers ------------------
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
    """
    Read a sheet, return DataFrame with added columns:
      - _sheet_row_idx (0-based index of data rows after header)
      - _source_sheet (source_name string)
    If read fails or empty, returns empty DataFrame.
    """
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
    Backward-compatible wrapper to read single sheet (keeps prior behavior).
    """
    if io_mod is None:
        st.error("io_helpers.py not available. Add io_helpers.py to the project to use Google Sheets.")
        return pd.DataFrame()

    creds_info = _get_creds_info()
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
    df = df.copy()
    if 'Bank' in df.columns and not overwrite:
        df['Bank'] = df['Bank'].astype(str).where(df['Bank'].notna(), None)
        df['Bank'] = df['Bank'].fillna('Unknown')
        return df

    cand_cols = ['bank', 'account', 'account_name', 'description', 'message', 'narration', 'merchant', 'beneficiary', 'note']
    def _row_text(row):
        parts = []
        for c in cand_cols:
            if c in row.index and pd.notna(row[c]):
                parts.append(str(row[c]))
        return " ".join(parts).lower()

    bank_map = {
        'hdfc': 'HDFC Bank',
        'hdfc bank': 'HDFC Bank',
        'hdfcbank': 'HDFC Bank',
        'hdfc card': 'HDFC Bank',
        'hdfccredit': 'HDFC Bank',
        'indian bank': 'Indian Bank',
        'indianbank': 'Indian Bank',
        'indian bank ltd': 'Indian Bank',
    }

    try:
        combined = df.apply(_row_text, axis=1)
    except Exception:
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

sheet_full_df = pd.DataFrame()
df_raw = pd.DataFrame()

if data_source == "Google Sheet (optional)":
    if not SHEET_ID:
        st.sidebar.info("Enter Google Sheet ID to enable sheet loading (or add to Streamlit secrets).")
        df_raw = pd.DataFrame()
    else:
        with st.spinner("Fetching Google Sheets (History + Append)..."):
            if io_mod is None:
                st.error("io_helpers.py is required to read Google Sheets. Add io_helpers.py to the project.")
                st.stop()
            creds_info = _get_creds_info()

            # Read history sheet
            history_df = _read_sheet_with_index(SHEET_ID, RANGE, "history", creds_info, CREDS_FILE)
            # Read append sheet (may be missing / empty)
            append_df = _read_sheet_with_index(SHEET_ID, APPEND_RANGE, "append", creds_info, CREDS_FILE)

            # Keep combined sheet_full_df for mapping back to sheet row indices
            if history_df is None:
                history_df = pd.DataFrame()
            if append_df is None:
                append_df = pd.DataFrame()
            # ensure _sheet_row_idx exists numeric when empty
            if history_df.empty and append_df.empty:
                sheet_full_df = pd.DataFrame()
            else:
                sheet_full_df = pd.concat([history_df, append_df], ignore_index=True, sort=False)
                if '_sheet_row_idx' not in sheet_full_df.columns:
                    sheet_full_df['_sheet_row_idx'] = pd.NA

            # Now create df_raw by removing soft-deleted rows (so UI/transform operate on visible rows only)
            if not sheet_full_df.empty and 'is_deleted' in sheet_full_df.columns:
                try:
                    deleted_mask = sheet_full_df['is_deleted'].astype(str).str.lower().isin(['true', 't', '1', 'yes'])
                except Exception:
                    deleted_mask = sheet_full_df['is_deleted'].astype(str).str.lower().isin(['true', 't', '1', 'yes'])
                df_raw = sheet_full_df.loc[~deleted_mask].copy().reset_index(drop=True)
            else:
                df_raw = sheet_full_df.copy().reset_index(drop=True)

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

# ------------------ Transform using transform.py ------------------
with st.spinner("Cleaning and deriving columns..."):
    # transform is expected to preserve extra cols like '_sheet_row_idx' and '_source_sheet'
    converted_df = transform.convert_columns_and_derives(df_raw)

# ensure bank column exists (detect heuristically if needed)
converted_df = add_bank_column(converted_df, overwrite=False)

# ------------------ Sidebar: Bank filter ------------------
with st.sidebar:
    st.markdown("---")
    st.write("Filter by Bank")
    banks_detected = sorted([b for b in converted_df['Bank'].unique() if pd.notna(b)])
    defaults = []
    if 'HDFC Bank' in banks_detected:
        defaults.append('HDFC Bank')
    if 'Indian Bank' in banks_detected:
        defaults.append('Indian Bank')
    if not defaults:
        defaults = banks_detected
    sel_banks = st.multiselect("Banks", options=banks_detected, default=defaults)

# filter converted_df according to selection
if sel_banks:
    converted_df_filtered = converted_df[converted_df['Bank'].isin(sel_banks)].copy()
else:
    converted_df_filtered = converted_df.copy()

# ------------------ Compute daily totals (from filtered transactions) ------------------
with st.spinner("Computing daily totals..."):
    merged = transform.compute_daily_totals(converted_df_filtered)

# ------------------ Sidebar: Date filters (moved above the table so table can obey selection) ------------------
with st.sidebar:
    st.header("Filters")
    if merged is not None and not merged.empty:
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
        default_range = (min_date, max_date)
        dr = st.date_input("Pick start & end", value=default_range, min_value=min_date, max_value=max_date)
        if isinstance(dr, (tuple, list)):
            selected_date_range_for_totals = (dr[0], dr[1])
        else:
            selected_date_range_for_totals = (dr, dr)

# ------------------ Normalize start_sel / end_sel right away ------------------
# This ensures start_sel/end_sel always exist and are datetime.date
try:
    start_sel, end_sel = selected_date_range_for_totals
except Exception:
    # fallback to last 30 days if something unexpected happened
    end_sel = datetime.utcnow().date()
    start_sel = end_sel - timedelta(days=30)

# if values are datetime -> convert to date
if isinstance(start_sel, datetime):
    start_sel = start_sel.date()
if isinstance(end_sel, datetime):
    end_sel = end_sel.date()

# Ensure ordering and clamp to min/max
if start_sel is None:
    start_sel = min_date
if end_sel is None:
    end_sel = max_date
if start_sel < min_date:
    start_sel = min_date
if end_sel > max_date:
    end_sel = max_date
if start_sel > end_sel:
    start_sel, end_sel = end_sel, start_sel

# ------------------ Apply year/month filters to aggregated plot_df ------------------
plot_df = merged.copy() if merged is not None else pd.DataFrame()
if sel_year != 'All' and not plot_df.empty:
    plot_df = plot_df[plot_df['Date'].dt.year == int(sel_year)]
if sel_months and not plot_df.empty:
    month_map = {i: pd.Timestamp(1900, i, 1).strftime('%B') for i in range(1, 13)}
    inv_map = {v: k for k, v in month_map.items()}
    selected_month_nums = [inv_map[m] for m in sel_months if m in inv_map]
    if selected_month_nums:
        plot_df = plot_df[plot_df['Date'].dt.month.isin(selected_month_nums)]

plot_df = plot_df.sort_values('Date').reset_index(drop=True) if not plot_df.empty else pd.DataFrame()
plot_df['Total_Spent'] = pd.to_numeric(plot_df.get('Total_Spent', 0), errors='coerce').fillna(0.0).astype('float64') if not plot_df.empty else pd.Series(dtype='float64')
plot_df['Total_Credit'] = pd.to_numeric(plot_df.get('Total_Credit', 0), errors='coerce').fillna(0.0).astype('float64') if not plot_df.empty else pd.Series(dtype='float64')

# ------------------ Chart & rendering ------------------
st.subheader("Daily Spend and Credit")
if plot_df.empty:
    st.info("No data for the selected filters.")
else:
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

# ------------------ Rows view & download (show only selected columns) ------------------
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
# Use the normalized start_sel / end_sel defined earlier
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
    st.warning("None of the preferred columns (timestamp, Bank, Type, Amount, Suspicious, message) were found — showing full table.")
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
        elif lc == 'message':
            pretty_rename[c] = 'Message'
    if pretty_rename:
        display_df = display_df.rename(columns=pretty_rename)

    # Ensure order: Timestamp, Bank, Type, Amount, Suspicious (include whichever exist)
    final_order = [c for c in ['Timestamp', 'Bank', 'Type', 'Amount', 'Suspicious', 'Message'] if c in display_df.columns]
    display_df = display_df[final_order]

    # Show table and download only these columns (use full width + fixed height to avoid excessive dragging)
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True, height=420)
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download rows (CSV)", csv_bytes, file_name="transactions_rows.csv", mime="text/csv")

    # ------------------ Build selectable mapping (label -> (sheet_range, sheet_row_idx)) ------------------
    selectable = False
    selectable_labels = []
    selectable_label_to_target = {}  # label -> (range_name, idx)
    if data_source == "Google Sheet (optional)" and io_mod is not None and not sheet_full_df.empty:
        # Use converted_df_filtered (which should preserve _sheet_row_idx and _source_sheet)
        map_df = converted_df_filtered.copy()
        # Ensure mapping columns exist
        if '_sheet_row_idx' in map_df.columns and '_source_sheet' in map_df.columns:
            # filter same date-range as display to avoid mismatches
            try:
                map_df['timestamp'] = pd.to_datetime(map_df['timestamp'], errors='coerce')
            except Exception:
                pass
            # Keep only rows in the displayed date window
            if start_sel and end_sel:
                map_df = map_df[
                    (map_df['timestamp'].dt.date >= start_sel) &
                    (map_df['timestamp'].dt.date <= end_sel)
                ]
            # Build labels
            for i, r in map_df.iterrows():
                ts = ''
                if 'timestamp' in r and pd.notna(r['timestamp']):
                    try:
                        ts = pd.to_datetime(r['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:
                        ts = str(r['timestamp'])
                amt = r.get('Amount', '')
                msg = r.get('Message', '') if 'Message' in r else (r.get('message', '') if 'message' in r else '')
                src = r.get('_source_sheet', 'history')
                idx = r.get('_sheet_row_idx')
                label = f"{i+1} | {src} | {ts} | {amt} | {str(msg)[:60]}"
                # map to corresponding range name used for write (history->RANGE, append->APPEND_RANGE)
                if src == 'append':
                    tgt_range = APPEND_RANGE
                else:
                    tgt_range = RANGE
                # store mapping only if idx is not null
                try:
                    if pd.isna(idx):
                        continue
                except Exception:
                    pass
                selectable_labels.append(label)
                selectable_label_to_target[label] = (tgt_range, int(idx))
            if selectable_labels:
                selectable = True

    # ------------------ NEW: Remove selected rows (soft-delete) UI ------------------
    if data_source == "Google Sheet (optional)" and io_mod is not None and not sheet_full_df.empty:
        st.markdown("---")
        st.write("Bulk actions (Google Sheet only)")
        col_a, col_b = st.columns([3, 1])
        with col_a:
            if selectable:
                selected_labels = st.multiselect(
                    "Select rows to remove (soft-delete)",
                    options=selectable_labels
                )
            else:
                st.info("Row selection for deletion is not available (cannot map visible rows back to sheet rows).")
                selected_labels = []
        with col_b:
            remove_btn = st.button("Remove selected rows", key="remove_rows_btn")

        if remove_btn:
            if not io_mod:
                st.error("io_helpers not available; cannot perform delete.")
            elif not SHEET_ID:
                st.error("No Sheet ID provided.")
            elif not selected_labels:
                st.warning("No rows selected.")
            else:
                # Group selected labels by target range
                groups = {}
                for lbl in selected_labels:
                    tgt = selectable_label_to_target.get(lbl)
                    if not tgt:
                        continue
                    rng, idx = tgt
                    groups.setdefault(rng, []).append(idx)

                creds_info = _get_creds_info()
                any_error = False
                total_updated = 0
                for rng, indices in groups.items():
                    try:
                        res = io_mod.mark_rows_deleted(
                            spreadsheet_id=SHEET_ID,
                            range_name=rng,
                            creds_info=creds_info,
                            creds_file=CREDS_FILE,
                            row_indices=indices
                        )
                        if res.get('status') == 'ok':
                            total_updated += int(res.get('updated', 0))
                        else:
                            st.error(f"Failed to mark rows deleted in {rng}: {res.get('message')}")
                            any_error = True
                    except Exception as e:
                        st.error(f"Error while marking rows deleted in {rng}: {e}")
                        any_error = True

                if not any_error:
                    st.success(f"Marked {total_updated} rows as deleted.")
                    st.experimental_rerun()

    # ------------------ NEW: Add new row form (writes to Append sheet only) ------------------
    if data_source == "Google Sheet (optional)" and io_mod is not None:
        st.markdown("---")
        st.write("Add a new row to the Append sheet")
        with st.expander("Open add row form"):
            with st.form("add_row_form", clear_on_submit=True):
                default_dt = start_sel if 'start_sel' in locals() else datetime.utcnow().date()
                new_date = st.date_input("Date", value=default_dt, min_value=min_date, max_value=max_date)
                bank_choice = st.selectbox("Bank", options=(banks_detected + ["Other (enter below)"]) if banks_detected else ["Other (enter below)"])
                bank_other = ""
                if bank_choice == "Other (enter below)":
                    bank_other = st.text_input("Bank (custom)")
                txn_type = st.selectbox("Type", options=["debit", "credit"])
                amount = st.number_input("Amount (₹)", value=0.0, step=1.0, format="%f")
                message = st.text_input("Message / Description", value="")
                submit_add = st.form_submit_button("Save new row")

                if submit_add:
                    chosen_bank = bank_other if bank_choice == "Other (enter below)" and bank_other else (bank_choice if bank_choice != "Other (enter below)" else "")
                    # combine selected date with current time to build DateTime
                    now = datetime.utcnow()
                    # use current UTC time-of-day from now
                    try:
                        dt_combined = datetime.combine(new_date, now.time())
                    except Exception:
                        dt_combined = now
                    # prefer writing a 'DateTime' column (sheet expects DateTime); also include timestamp/date
                    new_row = {
                        'DateTime': dt_combined.strftime("%Y-%m-%d %H:%M:%S"),
                        'timestamp': dt_combined,
                        'date': dt_combined.date(),
                        'Bank': chosen_bank,
                        'Type': txn_type,
                        'Amount': amount,
                        'Message': message,
                        'is_deleted': 'false'
                    }

                    creds_info = _get_creds_info()
                    try:
                        res = io_mod.append_new_row(
                            spreadsheet_id=SHEET_ID,
                            range_name=APPEND_RANGE,
                            new_row_dict=new_row,
                            creds_info=creds_info,
                            creds_file=CREDS_FILE,
                            history_range=RANGE  # ensures headers are synced
                        )
                        if res.get('status') == 'ok':
                            st.success("Appended new row to Append sheet.")
                            st.experimental_rerun()
                        else:
                            st.error(f"Failed to append row: {res.get('message')}")
                    except Exception as e:
                        st.error(f"Error while appending new row: {e}")

# ------------------ Totals for selected date / range ------------------
# compute title / heading
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
    tmp_rows = converted_df_filtered.copy()
    if 'timestamp' in tmp_rows.columns:
        tmp_rows['timestamp'] = pd.to_datetime(tmp_rows['timestamp'], errors='coerce')
    else:
        if 'date' in tmp_rows.columns:
            tmp_rows['timestamp'] = pd.to_datetime(tmp_rows['date'], errors='coerce')
        else:
            tmp_rows['timestamp'] = pd.NaT

    mask_sel = tmp_rows['timestamp'].dt.date.between(start_sel, end_sel)
    sel_df = tmp_rows[mask_sel].copy()

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

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Credits ({start_sel} → {end_sel})", f"₹{credit_sum:,.0f}", f"{credit_count} txns")
    col2.metric(f"Debits ({start_sel} → {end_sel})", f"₹{debit_sum:,.0f}", f"{debit_count} txns")
    col3.metric("Net (Credits − Debits)", f"₹{(credit_sum - debit_sum):,.0f}")

except Exception as e:
    st.error(f"Failed to compute totals for selected date/range: {e}")
