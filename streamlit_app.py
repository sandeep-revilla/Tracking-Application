# streamlit_app.py
"""
Streamlit app — sidebar filters + click-to-show-rows

Features:
- Read Google Sheet (service account via st.secrets or local JSON)
- Normalize columns: Amount -> rounded Int64; detect timestamp + date
- Compute daily totals (Total_Spent / Total_Credit)
- Sidebar filters: Year, Month(s), Series (debit/credit)
- Sidebar date selector (shows corresponding table rows below chart)
- Optional Plotly "click-to-select" mode (requires `streamlit-plotly-events` package)
- Altair chart used by default (fast, interactive); Plotly used only when click-to-select toggled
"""

import streamlit as st
import pandas as pd
import json
import os
from typing import List, Tuple, Optional, Any, Dict

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import altair as alt

# Try to import the optional helper for Plotly click capture
_plotly_events_available = False
try:
    from streamlit_plotly_events import plotly_events
    _plotly_events_available = True
except Exception:
    _plotly_events_available = False

# ---------------- Page config ----------------
st.set_page_config(page_title="Sheet → Daily Spend (interactive)", layout="wide")
st.title("💳 Daily Spending — Filters in sidebar + click-to-show-rows")

# ---------------- Sidebar: Google sheet inputs + filters ----------------
with st.sidebar:
    st.header("Data source & filters")

    SHEET_ID = st.text_input(
        "Google Sheet ID (between /d/ and /edit)",
        value="1KZq_GLXdMBfQUhtp-NA8Jg-flxOppw7kFuIN6y_nOXk"
    )
    RANGE = st.text_input("Range or Sheet Name", value="History Transactions")
    st.caption("Provide service account JSON via st.secrets['gcp_service_account'] or a local file path below.")
    CREDS_FILE = st.text_input("Service Account JSON File (optional)", value="creds/service_account.json")

    st.markdown("---")
    st.subheader("Chart options")
    enable_plotly_click = st.checkbox(
        "Enable click-to-select (Plotly) — requires `streamlit-plotly-events`",
        value=False
    )
    if enable_plotly_click and not _plotly_events_available:
        st.warning("`streamlit-plotly-events` not installed. Toggle off or install it (`pip install streamlit-plotly-events`).")

    st.markdown("---")
    st.write("Series to include")
    show_debit = st.checkbox("Debit (Total_Spent)", value=True)
    show_credit = st.checkbox("Credit (Total_Credit)", value=True)

    st.markdown("---")
    st.write("Date selection (click-to-select will override this when used)")
    # placeholder for date selector; will be populated later after data loads
    selected_date_sidebar = st.empty()

    st.markdown("---")
    if st.button("Refresh data"):
        st.experimental_rerun()

# ---------------- Helper functions (same conversion + aggregation used earlier) ----------------
def parse_service_account_secret(raw: Any) -> Dict:
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    if (s.startswith('"""') and s.endswith('"""')) or (s.startswith("'''") and s.endswith("'''")):
        s = s[3:-3].strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(s.replace('\\n', '\n'))
        except Exception:
            return json.loads(s.replace('\n', '\\n'))

@st.cache_data(ttl=300)
def build_sheets_service_from_info(creds_info: Dict):
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)

@st.cache_data(ttl=300)
def build_sheets_service_from_file(creds_file: str):
    if not os.path.exists(creds_file):
        raise FileNotFoundError(f"Credentials file not found: {creds_file}")
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = service_account.Credentials.from_service_account_file(creds_file, scopes=scopes)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)

def _normalize_rows(values: List[List[str]]) -> Tuple[List[str], List[List]]:
    if not values:
        return [], []
    header_row = [str(x).strip() for x in values[0]]
    if all((h == "" or h.lower().startswith(("unnamed", "column", "nan"))) for h in header_row):
        max_cols = max(len(r) for r in values)
        header = [f"col_{i}" for i in range(max_cols)]
        data_rows = values
    else:
        header = header_row
        data_rows = values[1:]
    col_count = len(header)
    normalized = []
    for r in data_rows:
        if len(r) < col_count:
            r = r + [None] * (col_count - len(r))
        elif len(r) > col_count:
            r = r[:col_count]
        normalized.append(r)
    return header, normalized

@st.cache_data(ttl=300)
def values_to_dataframe(values: List[List[str]]) -> pd.DataFrame:
    if not values:
        return pd.DataFrame()
    header, rows = _normalize_rows(values)
    try:
        return pd.DataFrame(rows, columns=header)
    except Exception:
        df = pd.DataFrame(rows)
        if header and df.shape[1] == len(header):
            df.columns = header
        return df

@st.cache_data(ttl=300)
def read_google_sheet(spreadsheet_id: str, range_name: str,
                      creds_info: Optional[Dict] = None, creds_file: Optional[str] = None) -> pd.DataFrame:
    if creds_info is None and (creds_file is None or not os.path.exists(creds_file)):
        if "gcp_service_account" not in st.secrets:
            raise ValueError("No credentials found. Add service account JSON to st.secrets['gcp_service_account'] or supply a local file.")
        creds_info = parse_service_account_secret(st.secrets["gcp_service_account"])
    service = (build_sheets_service_from_info(creds_info) if creds_info else build_sheets_service_from_file(creds_file))
    try:
        sheet = service.spreadsheets()
        res = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
        values = res.get("values", [])
    except HttpError as e:
        raise RuntimeError(f"Google Sheets API error: {e}")
    return values_to_dataframe(values)

def convert_columns_and_derives(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    date_keywords = ['date', 'time', 'timestamp', 'datetime', 'txn']
    num_keywords = ['amount', 'amt', 'value', 'total', 'balance', 'credit', 'debit', 'spent']

    # Parse date-like columns
    for col in list(df.columns):
        lname = str(col).lower()
        if any(k in lname for k in date_keywords) or lname.startswith("unnamed"):
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=False)

    # Coerce amounts to Int64
    amount_cols = []
    for col in list(df.columns):
        lname = str(col).lower()
        if any(k in lname for k in num_keywords):
            coerced = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce')
            coerced = coerced.round(0)
            df[col] = coerced.astype('Int64')
            amount_cols.append(col)

    # Heuristic: convert object columns that look numeric
    for col in list(df.columns):
        if pd.api.types.is_object_dtype(df[col]):
            sample = df[col].astype(str).head(20).str.replace(r'[^\d\.\-]', '', regex=True)
            parsed = pd.to_numeric(sample, errors='coerce')
            if parsed.notna().sum() >= 3:
                coerced = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce').round(0)
                df[col] = coerced.astype('Int64')

    # Canonical Amount column
    preferred = None
    for candidate in ['amount','total_spent','totalspent','total','txn amount','value','spent']:
        for col in df.columns:
            if str(col).lower() == candidate:
                preferred = col
                break
        if preferred:
            break
    if not preferred and amount_cols:
        preferred = amount_cols[0]
    if preferred and preferred != 'Amount':
        if 'Amount' not in df.columns:
            df.rename(columns={preferred: 'Amount'}, inplace=True)

    if 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').round(0).astype('Int64')
    else:
        df['Amount'] = pd.NA

    # Pick primary datetime column
    primary_dt_col = None
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) and df[col].notna().sum() > 0:
            primary_dt_col = col
            break
    if primary_dt_col is None:
        for col in df.columns:
            lname = str(col).lower()
            if any(k in lname for k in date_keywords) or lname.startswith("unnamed"):
                parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=False)
                if parsed.notna().sum() > 0:
                    df[col] = parsed
                    primary_dt_col = col
                    break
    if primary_dt_col is None:
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=False)
                if parsed.notna().sum() >= 3:
                    df[col] = parsed
                    primary_dt_col = col
                    break

    # canonical timestamp and date
    if primary_dt_col:
        df['timestamp'] = pd.to_datetime(df[primary_dt_col], errors='coerce')
    else:
        df['timestamp'] = pd.NaT
    try:
        date_series = df['timestamp'].dt.date
        date_series = date_series.where(pd.notna(df['timestamp']), pd.NA)
        df['date'] = date_series
    except Exception:
        df['date'] = pd.NA

    # drop original date-like columns except canonical ones
    cols_to_drop = []
    for col in list(df.columns):
        low = str(col).lower()
        if low in ('timestamp', 'date'):
            continue
        if any(k in low for k in date_keywords):
            cols_to_drop.append(col)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # reorder
    cols = list(df.columns)
    final_cols = []
    if 'timestamp' in cols:
        final_cols.append('timestamp')
    if 'date' in cols:
        final_cols.append('date')
    for c in cols:
        if c not in final_cols:
            final_cols.append(c)
    df = df[final_cols]
    return df

def compute_daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=['Date','Total_Spent','Total_Credit'])
    w = df.copy()
    if 'date' in w.columns and w['date'].notna().any():
        grp = pd.to_datetime(w['date']).dt.normalize()
    elif 'timestamp' in w.columns and w['timestamp'].notna().any():
        grp = pd.to_datetime(w['timestamp']).dt.normalize()
    else:
        found = None
        for c in w.columns:
            if pd.api.types.is_datetime64_any_dtype(w[c]) and w[c].notna().any():
                found = c; break
        if found:
            grp = pd.to_datetime(w[found]).dt.normalize()
        else:
            return pd.DataFrame(columns=['Date','Total_Spent','Total_Credit'])
    w['_group_date'] = grp
    w['Amount_numeric'] = pd.to_numeric(w.get('Amount', 0), errors='coerce').fillna(0.0).astype('float64')

    if 'Type' in w.columns and w['Type'].astype(str).str.strip().any():
        w['Type_norm'] = w['Type'].astype(str).str.lower().str.strip()
        debit_df = w[w['Type_norm'] == 'debit']
        credit_df = w[w['Type_norm'] == 'credit']
        daily_spend = debit_df.groupby(debit_df['_group_date'])['Amount_numeric'].sum().reset_index().rename(columns={'_group_date':'Date','Amount_numeric':'Total_Spent'})
        daily_credit = credit_df.groupby(credit_df['_group_date'])['Amount_numeric'].sum().reset_index().rename(columns={'_group_date':'Date','Amount_numeric':'Total_Credit'})
    else:
        daily_spend = w.groupby(w['_group_date'])['Amount_numeric'].sum().reset_index().rename(columns={'_group_date':'Date','Amount_numeric':'Total_Spent'})
        daily_credit = pd.DataFrame(columns=['Date','Total_Credit'])

    merged = pd.merge(daily_spend, daily_credit, on='Date', how='outer').fillna(0)
    merged['Date'] = pd.to_datetime(merged['Date']).dt.normalize()
    merged['Total_Spent'] = merged['Total_Spent'].astype('float64')
    merged['Total_Credit'] = merged.get('Total_Credit', 0).astype('float64') if 'Total_Credit' in merged else 0.0
    merged = merged.sort_values('Date').reset_index(drop=True)
    return merged

# ---------------- Load data ----------------
if not SHEET_ID:
    st.sidebar.error("Enter Google Sheet ID in the sidebar.")
    st.stop()

with st.spinner("Fetching Google Sheet..."):
    try:
        creds_info = None
        if "gcp_service_account" in st.secrets:
            creds_info = parse_service_account_secret(st.secrets["gcp_service_account"])
        df_raw = read_google_sheet(SHEET_ID, RANGE, creds_info=creds_info, creds_file=CREDS_FILE)
    except Exception as e:
        st.error(f"Failed to read Google Sheet: {e}")
        st.stop()

if df_raw.empty:
    st.warning("No data returned from sheet. Check sheet id / range / permissions.")
    st.stop()

# convert + derive
converted_df = convert_columns_and_derives(df_raw)

# compute merged daily totals
merged = compute_daily_totals(converted_df)

# ---------------- Sidebar: Year/Month filters & date selector population ----------------
with st.sidebar:
    # year choices
    if not merged.empty:
        merged['Date'] = pd.to_datetime(merged['Date']).dt.normalize()
        years = sorted(merged['Date'].dt.year.unique().tolist())
        years_opts = ['All'] + [str(y) for y in years]
        sel_year = st.selectbox("Year", years_opts, index=0)
        # months in selected year
        if sel_year == 'All':
            month_frame = merged.copy()
        else:
            month_frame = merged[merged['Date'].dt.year == int(sel_year)]
        month_nums = sorted(month_frame['Date'].dt.month.unique().tolist())
        month_map = {i: pd.Timestamp(1900, i, 1).strftime('%B') for i in range(1,13)}
        month_choices = [month_map[m] for m in month_nums]
        sel_months = st.multiselect("Month(s)", options=month_choices, default=month_choices)
        # populate date selector: show dates matching current year/month filter
        # map back names to numbers for selection
        inv_map = {v:k for k,v in month_map.items()}
        selected_month_nums = [inv_map[m] for m in sel_months] if sel_months else []
        filter_dates = merged.copy()
        if sel_year != 'All':
            filter_dates = filter_dates[filter_dates['Date'].dt.year == int(sel_year)]
        if selected_month_nums:
            filter_dates = filter_dates[filter_dates['Date'].dt.month.isin(selected_month_nums)]
        date_options = ['All'] + [d.date().isoformat() for d in filter_dates['Date'].dt.to_pydatetime()]
        # unique + sorted
        date_options = sorted(list(dict.fromkeys(date_options)))
        sel_date = st.selectbox("Select date to show rows", options=date_options, index=0)
    else:
        sel_year = 'All'
        sel_months = []
        sel_date = 'All'

# ---------------- Apply year/month filters to merged and prepare plot_df ----------------
plot_df = merged.copy()
if sel_year != 'All':
    plot_df = plot_df[plot_df['Date'].dt.year == int(sel_year)]
# convert selected month names back to nums
if sel_months:
    inv_map = {v:k for k,v in {i: pd.Timestamp(1900, i, 1).strftime('%B') for i in range(1,13)}.items()}
    selected_month_nums = [inv_map[m] for m in sel_months if m in inv_map]
    if selected_month_nums:
        plot_df = plot_df[plot_df['Date'].dt.month.isin(selected_month_nums)]

plot_df = plot_df.sort_values('Date').reset_index(drop=True)
plot_df['Total_Spent'] = pd.to_numeric(plot_df.get('Total_Spent', 0), errors='coerce').fillna(0.0).astype('float64')
plot_df['Total_Credit'] = pd.to_numeric(plot_df.get('Total_Credit', 0), errors='coerce').fillna(0.0).astype('float64')

# ---------------- Chart & click-to-select logic ----------------
st.subheader("Daily Spend and Credit")

# prepare long form
if plot_df.empty:
    st.info("No data for the selected filters.")
else:
    plot_df_long = plot_df.melt(id_vars='Date', value_vars=['Total_Spent', 'Total_Credit'],
                                var_name='Type', value_name='Amount').sort_values('Date')
    plot_df_long['Amount'] = pd.to_numeric(plot_df_long['Amount'], errors='coerce').fillna(0.0).astype('float64')
    plot_df_long['Date'] = pd.to_datetime(plot_df_long['Date'])

    # determine which series to plot from checkboxes (sidebar)
    series_selected = []
    if show_debit: series_selected.append('Total_Spent')
    if show_credit: series_selected.append('Total_Credit')

    if not series_selected:
        st.info("Enable at least one series (debit or credit) in the sidebar.")
    else:
        plot_df_long = plot_df_long[plot_df_long['Type'].isin(series_selected)].copy()

        # If plotly click mode requested and available -> use Plotly and capture click
        clicked_date = None
        plotly_used = False
        if enable_plotly_click and _plotly_events_available:
            plotly_used = True
            import plotly.express as px
            fig = px.line(plot_df_long, x='Date', y='Amount', color='Type', markers=True,
                          title="(Plotly) Click a point to select its date")
            fig.update_layout(hovermode='x unified', legend_title='Type')
            # display via plotly_events and capture click
            res = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="plotly_click_1")
            # res is a list of dicts describing clicked points, if any
            if res:
                # try to extract x value robustly
                first = res[0]
                xval = first.get('x') or first.get('x_val') or first.get('bbox', {}).get('x')
                # fallback: 'pointNumber' or 'pointIndex' not helpful; xval may be ISO string
                try:
                    if xval is not None:
                        clicked_date = pd.to_datetime(xval).date()
                except Exception:
                    clicked_date = None
        else:
            # show Altair chart (default)
            color_scale = alt.Scale(domain=['Total_Spent', 'Total_Credit'], range=['#d62728', '#2ca02c'])
            legend_sel = alt.selection_multi(fields=['Type'], bind='legend')
            base = alt.Chart(plot_df_long).mark_line(point=True).encode(
                x=alt.X('Date:T', title='Date'),
                y=alt.Y('Amount:Q', title='Amount', axis=alt.Axis(format=",.0f")),
                color=alt.Color('Type:N', title='Type', scale=color_scale),
                tooltip=[alt.Tooltip('Date:T', title='Date', format='%Y-%m-%d'),
                         alt.Tooltip('Type:N', title='Type'),
                         alt.Tooltip('Amount:Q', title='Amount', format=',')],
                opacity=alt.condition(legend_sel, alt.value(1.0), alt.value(0.25))
            ).add_selection(legend_sel).interactive()
            st.altair_chart(base.properties(height=450), use_container_width=True)

        # Determine selected date: priority order
        # 1) if Plotly click produced a date -> use it
        # 2) else if user selected a date in sidebar -> use that
        # 3) else 'All'
        selected_date_value = None
        if clicked_date is not None:
            selected_date_value = clicked_date
        else:
            # use sel_date from sidebar
            if sel_date != 'All':
                try:
                    selected_date_value = pd.to_datetime(sel_date).date()
                except Exception:
                    selected_date_value = None
            else:
                selected_date_value = None

        # Show note to user when click-mode is not available
        if enable_plotly_click and not _plotly_events_available:
            st.info("Plotly click-mode was requested but `streamlit-plotly-events` is not installed. Install it with `pip install streamlit-plotly-events` to enable click-to-select.")

        # Show selection info
        if selected_date_value is not None:
            st.markdown(f"**Showing rows for date:** {selected_date_value.isoformat()}")
        else:
            st.markdown("**Showing rows for:** All dates (filtered by Year/Month)")

        # ---------------- Show table of underlying rows filtered by the selected date and sidebar filters ----------------
        # Filter converted_df by the same Year/Month filters, then by selected date if provided
        rows_df = converted_df.copy()

        # convert timestamp to datetime, ensure date col exists
        if 'timestamp' in rows_df.columns:
            rows_df['timestamp'] = pd.to_datetime(rows_df['timestamp'], errors='coerce')
        else:
            # try to create from 'date' if absent
            if 'date' in rows_df.columns:
                rows_df['timestamp'] = pd.to_datetime(rows_df['date'], errors='coerce')
            else:
                rows_df['timestamp'] = pd.NaT

        # apply year/month sidebar filters to rows_df
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

        # apply selected date (if any)
        if selected_date_value is not None:
            rows_df = rows_df[rows_df['timestamp'].dt.date == selected_date_value]

        # show a compact table below the chart
        st.subheader("Rows (matching selection)")
        if rows_df.empty:
            st.write("No rows match the current filters/selection.")
        else:
            # show all columns but limit rows to a reasonable number first; provide an expand option
            st.dataframe(rows_df.reset_index(drop=True))

# End of file
