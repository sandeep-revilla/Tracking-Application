# streamlit_app.py
"""
Single-file Streamlit app:
- Reads Google Sheet (service account via st.secrets or local JSON file)
- Converts columns: amounts -> rounded nullable Int64; detects timestamp & date
- Produces a single canonical `timestamp` (datetime64[ns]) and `date` (python.date)
- Aggregates daily totals (Total_Spent / Total_Credit)
- Simple Altair chart (default axis formatting — scientific notation allowed)
- Diagnostics: top rows + merged.describe()
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

# ---------------- Page config ----------------
st.set_page_config(page_title="Sheet → Daily Spend (Altair simple)", layout="wide")
st.title("💳 Daily Spending — Altair (simple axis formatting)")

# ---------------- Sidebar inputs ----------------
SHEET_ID = st.sidebar.text_input(
    "Google Sheet ID (between /d/ and /edit)",
    value="1KZq_GLXdMBfQUhtp-NA8Jg-flxOppw7kFuIN6y_nOXk"
)
RANGE = st.sidebar.text_input("Range or Sheet Name", value="History Transactions")
st.sidebar.caption("Provide service account JSON via st.secrets['gcp_service_account'] or as a local file below.")
CREDS_FILE = st.sidebar.text_input("Service Account JSON File (optional)", value="creds/service_account.json")
if st.sidebar.button("Refresh Now"):
    st.experimental_rerun()

# ---------------- Helpers: parse secret & sheets client ----------------
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

# ---------------- Safe conversion from sheet values -> DataFrame ----------------
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

# ---------------- Read Google Sheet ----------------
@st.cache_data(ttl=300)
def read_google_sheet(spreadsheet_id: str, range_name: str, creds_info: Optional[Dict] = None, creds_file: Optional[str] = None) -> pd.DataFrame:
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

# ---------------- Column conversion utility ----------------
def convert_columns_and_derives(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    date_keywords = ['date', 'time', 'timestamp', 'datetime', 'txn']
    num_keywords = ['amount', 'amt', 'value', 'total', 'balance', 'credit', 'debit', 'spent']

    # Parse obvious date-like columns
    for col in list(df.columns):
        lname = str(col).lower()
        if any(k in lname for k in date_keywords) or lname.startswith("unnamed"):
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=False)

    # Coerce amount-like columns to Int64
    amount_cols = []
    for col in list(df.columns):
        lname = str(col).lower()
        if any(k in lname for k in num_keywords):
            coerced = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce')
            coerced = coerced.round(0)
            df[col] = coerced.astype('Int64')
            amount_cols.append(col)

    # Heuristic: convert other object columns that look numeric
    for col in list(df.columns):
        if pd.api.types.is_object_dtype(df[col]):
            sample = df[col].astype(str).head(20).str.replace(r'[^\d\.\-]', '', regex=True)
            parsed = pd.to_numeric(sample, errors='coerce')
            if parsed.notna().sum() >= 3:
                coerced = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce').round(0)
                df[col] = coerced.astype('Int64')

    # Prefer a canonical Amount column name
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

    # Ensure Amount exists as Int64 if possible
    if 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').round(0).astype('Int64')
    else:
        df['Amount'] = pd.NA

    # Determine primary datetime column
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

    # Create canonical timestamp & date
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

    # Remove other original date-like columns (preserve canonical ones)
    cols_to_drop = []
    for col in list(df.columns):
        low = str(col).lower()
        if low in ('timestamp', 'date'):
            continue
        if any(k in low for k in date_keywords):
            cols_to_drop.append(col)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Reorder: timestamp, date first
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

# ---------------- Daily aggregation ----------------
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

# ---------------- Main flow ----------------
if not SHEET_ID:
    st.info("Enter your Google Sheet ID to load data.")
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

st.success(f"Loaded {len(df_raw):,} rows, {df_raw.shape[1]} columns.")

# Convert & derive
converted_df = convert_columns_and_derives(df_raw)

# Diagnostics: show top rows and dtypes
st.subheader("Top 10 rows (after conversion)")
st.write(converted_df.head(10))

st.subheader("Column data types")
dt_df = pd.DataFrame({
    "column": converted_df.columns.astype(str),
    "dtype": [str(converted_df[c].dtype) for c in converted_df.columns]
})
st.write(dt_df)

if 'Amount' in converted_df.columns:
    st.subheader("Amount summary")
    amt = converted_df['Amount']
    st.write({
        "non_null_count": int(amt.notna().sum()),
        "min": int(amt.min()) if amt.notna().any() else None,
        "max": int(amt.max()) if amt.notna().any() else None,
        "mean": float(amt.dropna().astype(float).mean()) if amt.notna().any() else None
    })

st.subheader("Derived date/timestamp info")
st.write("timestamp non-null:", int(converted_df['timestamp'].notna().sum()))
st.write("date non-null:", int(converted_df['date'].notna().sum()))

st.download_button("⬇️ Download converted CSV", data=converted_df.to_csv(index=False).encode('utf-8'),
                   file_name="converted_sheet.csv", mime="text/csv")

# ---------------- Aggregation & plotting ----------------
merged = compute_daily_totals(converted_df)
st.subheader("Daily totals (merged) — top rows")
st.write(merged.head(10))

st.subheader("Daily totals description")
try:
    st.write(merged[['Total_Spent','Total_Credit']].describe().applymap(lambda x: float(x) if pd.notna(x) else x))
except Exception:
    st.write("No daily totals to describe.")

# interactive checkboxes to select series
st.markdown("### Select series to display")
show_debit = st.checkbox("Show debit (Total_Spent)", value=True)
show_credit = st.checkbox("Show credit (Total_Credit)", value=True)

# prepare plot DataFrame
merged['Date'] = pd.to_datetime(merged['Date']).dt.normalize()
merged = merged.sort_values('Date').reset_index(drop=True)
merged['Total_Spent'] = pd.to_numeric(merged.get('Total_Spent', 0), errors='coerce').fillna(0.0).astype('float64')
merged['Total_Credit'] = pd.to_numeric(merged.get('Total_Credit', 0), errors='coerce').fillna(0.0).astype('float64')

plot_df = merged.copy()
if plot_df.empty:
    st.info("No daily totals available to plot.")
else:
    plot_df_long = plot_df.melt(id_vars='Date', value_vars=['Total_Spent', 'Total_Credit'],
                                var_name='Type', value_name='Amount').sort_values('Date')

    selected = []
    if show_debit: selected.append('Total_Spent')
    if show_credit: selected.append('Total_Credit')

    if not selected:
        st.info("Select at least one series to display.")
    else:
        plot_df_long = plot_df_long[plot_df_long['Type'].isin(selected)].copy()
        plot_df_long['Amount'] = pd.to_numeric(plot_df_long['Amount'], errors='coerce').fillna(0.0).astype('float64')
        plot_df_long['Date'] = pd.to_datetime(plot_df_long['Date'])

        # simple Altair chart — default axis formatting (scientific notation will appear if relevant)
        selection = alt.selection_multi(fields=['Type'], bind='legend')
        chart = (
            alt.Chart(plot_df_long)
            .mark_line(point=True)
            .encode(
                x=alt.X('Date:T', title='Date'),
                y=alt.Y('Amount:Q', title='Amount'),  # default formatting (allows scientific notation)
                color=alt.Color('Type:N', title='Type'),
                tooltip=[alt.Tooltip('Date:T', title='Date', format='%Y-%m-%d'),
                         alt.Tooltip('Type:N', title='Type'),
                         alt.Tooltip('Amount:Q', title='Amount')]
            )
            .add_selection(selection)
            .transform_filter(selection)  # legend click filters series
            .properties(title="Daily Spend and Credit — Altair (simple axes)", height=450)
            .interactive()
        )

        st.altair_chart(chart, use_container_width=True)
