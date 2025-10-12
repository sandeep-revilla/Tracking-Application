# streamlit_app.py
"""
Single-file Streamlit app (no charts) — Amounts as integers; ensure single date & timestamp:
- Connects to Google Sheets via service account (st.secrets or file)
- Converts sheet values to a pandas DataFrame
- Auto-converts columns to inferred types (dates / numeric)
- Converts amount-like column(s) to integer (rounded) using pandas nullable Int64
- Ensures exactly two derived columns:
    - `timestamp` : full datetime (pandas datetime64[ns])
    - `date`      : only the date part (python date objects)
  and removes any other columns named 'timestamp' or 'date' (case-insensitive).
- Shows top 10 rows and column data types
- Offers cleaned CSV download
"""

import streamlit as st
import pandas as pd
import json
import os
from typing import List, Tuple, Optional, Any, Dict
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ---------------- Page config ----------------
st.set_page_config(page_title="Google Sheet Connector — Single Date & Timestamp", layout="wide")
st.title("🔐 Google Sheet Connector — Amounts → Integer; single date & timestamp")

# ---------------- Sidebar Inputs ----------------
SHEET_ID = st.sidebar.text_input(
    "Google Sheet ID (between /d/ and /edit)",
    value="1KZq_GLXdMBfQUhtp-NA8Jg-flxOppw7kFuIN6y_nOXk"
)
RANGE = st.sidebar.text_input("Range or Sheet Name", value="History Transactions")
st.sidebar.caption("Provide service account JSON via st.secrets['gcp_service_account'] or as a local file below.")
CREDS_FILE = st.sidebar.text_input("Service Account JSON File (optional)", value="creds/service_account.json")

if st.sidebar.button("Refresh Now"):
    st.experimental_rerun()

# ---------------- Helper: parse service account JSON ----------------
def parse_service_account_secret(raw: Any) -> Dict:
    """Accept dict or JSON string (even with escaped newlines) and return dict."""
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

# ---------------- Sheets client builders ----------------
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

# ---------------- Safe conversion helpers ----------------
def _normalize_rows(values: List[List[str]]) -> Tuple[List[str], List[List]]:
    """Ensure header exists, pad/trim rows to header length."""
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
    """Convert sheet values to pandas DataFrame safely."""
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

# ---------------- Google Sheet reader ----------------
@st.cache_data(ttl=300)
def read_google_sheet(spreadsheet_id: str, range_name: str, creds_info: Optional[Dict] = None, creds_file: Optional[str] = None) -> pd.DataFrame:
    """Reads the given Google Sheet range and returns a DataFrame."""
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

# ---------------- Column type conversion utility (Amounts -> Int64; single date/timestamp) ----------------
def convert_column_types_to_integer_with_single_date_and_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns:
      - parse columns with names containing date/time keywords to datetime
      - coerce amount-like columns to integer (rounded) using pandas 'Int64' dtype
      - create exactly two standardized columns:
          - `timestamp` (datetime64[ns])
          - `date`      (python.date objects)
      - remove any other original columns that look like date/time (case-insensitive),
        preserving only the canonical 'timestamp' and 'date' columns.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # Normalize column names (trim whitespace)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # keywords used to detect date/time-like columns
    date_keywords = ['date', 'time', 'timestamp', 'datetime', 'txn']
    num_keywords = ['amount', 'amt', 'value', 'total', 'balance', 'credit', 'debit', 'spent']

    # 1) Parse obvious date-like columns (including Unnamed: 0)
    for col in list(df.columns):
        lname = str(col).lower()
        if any(k in lname for k in date_keywords) or str(col).lower().startswith("unnamed"):
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=False)

    # 2) Coerce amount-like columns to nullable Int64
    amount_columns = []
    for col in list(df.columns):
        lname = str(col).lower()
        if any(k in lname for k in num_keywords):
            coerced = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce')
            coerced = coerced.round(0)
            df[col] = coerced.astype('Int64')
            amount_columns.append(col)

    # 3) Try coercing other object columns that look numeric (sample heuristic)
    for col in list(df.columns):
        if pd.api.types.is_object_dtype(df[col]):
            sample = df[col].astype(str).head(20).str.replace(r'[^\d\.\-]', '', regex=True)
            parsed = pd.to_numeric(sample, errors='coerce')
            if parsed.notna().sum() >= 3:
                coerced = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce')
                coerced = coerced.round(0)
                df[col] = coerced.astype('Int64')

    # 4) Standardize preferred amount column name => 'Amount' if possible
    preferred = None
    candidates = ['Amount', 'amount', 'total_spent', 'totalspent', 'total', 'txn amount', 'value', 'spent']
    for candidate in candidates:
        for col in df.columns:
            if str(col).lower() == str(candidate).lower():
                preferred = col
                break
        if preferred:
            break
    if not preferred and amount_columns:
        preferred = amount_columns[0]

    if preferred and preferred != 'Amount':
        if 'Amount' not in df.columns:
            df.rename(columns={preferred: 'Amount'}, inplace=True)
            preferred = 'Amount'

    # ensure Amount is Int64 if present
    if 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').round(0).astype('Int64')

    # ---------------- determine primary datetime column ----------------
    primary_dt_col = None
    # 1) any column already datetime dtype with non-null values
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) and df[col].notna().sum() > 0:
            primary_dt_col = col
            break
    # 2) look for common date-like column names and parse if necessary
    if primary_dt_col is None:
        for col in df.columns:
            lname = str(col).lower()
            if any(k in lname for k in date_keywords) or str(col).lower().startswith("unnamed"):
                parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=False)
                if parsed.notna().sum() > 0:
                    df[col] = parsed
                    primary_dt_col = col
                    break
    # 3) try parsing object columns with many parseable dates
    if primary_dt_col is None:
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=False)
                if parsed.notna().sum() >= 3:
                    df[col] = parsed
                    primary_dt_col = col
                    break

    # ---------------- create canonical timestamp & date ----------------
    if primary_dt_col:
        timestamp_series = pd.to_datetime(df[primary_dt_col], errors='coerce')
    else:
        timestamp_series = pd.Series([pd.NaT] * len(df), index=df.index, dtype='datetime64[ns]')

    # Create canonical columns
    df['timestamp'] = timestamp_series
    # date as python date objects; preserve missing values as pd.NA
    try:
        date_series = timestamp_series.dt.date
        date_series = date_series.where(pd.notna(timestamp_series), pd.NA)
        df['date'] = date_series
    except Exception:
        df['date'] = pd.NA

    # ---------------- remove other original date/time-like columns ----------------
    # Build a list of columns to drop: any column (original name) that looks date/time-like
    # but skip the canonical 'timestamp' and 'date' we just created.
    cols_to_drop = []
    for col in list(df.columns):
        low = str(col).lower()
        # skip canonical columns
        if low in ('timestamp', 'date'):
            continue
        # if column name contains a date keyword, mark for drop
        if any(k in low for k in date_keywords):
            cols_to_drop.append(col)

    # Drop them (conservative: only drop if they exist)
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)

    # Ensure there is exactly one 'timestamp' and one 'date' column (drop case-variants if any)
    for col in list(df.columns):
        if col not in ('timestamp','date') and str(col).lower() in ('timestamp','date'):
            df.drop(columns=[col], inplace=True)

    # Reorder: put canonical timestamp & date first
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

# ---------------- Main execution ----------------
if not SHEET_ID:
    st.info("Enter your Google Sheet ID (the long ID between /d/ and /edit in the URL).")
    st.stop()

with st.spinner("🔄 Fetching data from Google Sheets..."):
    try:
        creds_info = None
        if "gcp_service_account" in st.secrets:
            creds_info = parse_service_account_secret(st.secrets["gcp_service_account"])
        df_raw = read_google_sheet(SHEET_ID, RANGE, creds_info=creds_info, creds_file=CREDS_FILE)
    except Exception as e:
        st.error(f"❌ Failed to read Google Sheet: {e}")
        st.stop()

if df_raw.empty:
    st.warning("⚠️ No data returned. Check the sheet name/range and ensure the service account has viewer access.")
    st.stop()

st.success(f"✅ Loaded data — {len(df_raw):,} rows, {df_raw.shape[1]} columns.")

# Convert column types (amounts -> Int64), and ensure single date/timestamp columns
converted_df = convert_column_types_to_integer_with_single_date_and_timestamp(df_raw)

# Show top 10 rows and data types
st.subheader("Top 10 rows (after type conversion)")
st.write(converted_df.head(10))

st.subheader("Column data types")
dt_df = pd.DataFrame({
    "column": converted_df.columns.astype(str),
    "dtype": [str(converted_df[c].dtype) for c in converted_df.columns]
})
st.write(dt_df)

# If an 'Amount' column exists, show a small summary and counts of non-null
if 'Amount' in converted_df.columns:
    st.subheader("Amount summary (Integer, nullable)")
    amt = converted_df['Amount']
    st.write({
        "non_null_count": int(amt.notna().sum()),
        "min": int(amt.min()) if amt.notna().any() else None,
        "max": int(amt.max()) if amt.notna().any() else None,
        "mean": float(amt.dropna().astype(float).mean()) if amt.notna().any() else None
    })

# Show quick info about the derived date/timestamp
st.subheader("Derived date/timestamp info")
if 'timestamp' in converted_df.columns:
    st.write("timestamp non-null count:", int(converted_df['timestamp'].notna().sum()))
if 'date' in converted_df.columns:
    st.write("date non-null count:", int(converted_df['date'].notna().sum()))
    st.write("date sample:", converted_df['date'].dropna().head(5).tolist())

# Download cleaned CSV
csv_bytes = converted_df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Converted CSV (Amounts as integer)", data=csv_bytes, file_name="sheet_converted_integer_amounts_with_single_date_timestamp.csv", mime="text/csv")

# --- Daily totals computation + Plotly and Matplotlib plotters ---

import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

def compute_daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return merged daily totals DataFrame with columns:
      - Date (datetime64[ns])
      - Total_Spent (sum of Amount for Type == 'debit' or overall if no Type)
      - Total_Credit (sum of Amount for Type == 'credit', 0 if none)
    Expects `df` to contain:
      - 'Amount' (numeric / Int64)
      - either 'date' (python.date) or 'timestamp' (datetime)
      - optional 'Type' column with 'debit' / 'credit'
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['Date', 'Total_Spent', 'Total_Credit'])

    # use the 'date' column if present, else derive from 'timestamp'
    if 'date' in df.columns and df['date'].notna().any():
        date_series = pd.to_datetime(df['date'])
    elif 'timestamp' in df.columns and df['timestamp'].notna().any():
        date_series = pd.to_datetime(df['timestamp']).dt.date
        date_series = pd.to_datetime(date_series)
    else:
        # try any datetime-like
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) and df[col].notna().any():
                date_series = pd.to_datetime(df[col]).dt.date
                date_series = pd.to_datetime(date_series)
                break
        else:
            # no date available
            return pd.DataFrame(columns=['Date', 'Total_Spent', 'Total_Credit'])

    working = df.copy()
    working['_plot_date'] = date_series.dt.date  # group by pure date (no time)

    # Ensure Amount numeric (coerce missing to 0 for aggregation)
    if 'Amount' not in working.columns:
        working['Amount'] = 0
    working['Amount_numeric'] = pd.to_numeric(working['Amount'], errors='coerce').fillna(0.0)

    # If Type exists, separate debit/credit; otherwise use overall as Total_Spent
    if 'Type' in working.columns and working['Type'].astype(str).str.strip().any():
        debit = working[working['Type'].astype(str).str.lower() == 'debit']
        credit = working[working['Type'].astype(str).str.lower() == 'credit']
        daily_spend = (debit.groupby(debit['_plot_date'])['Amount_numeric'].sum().reset_index().rename(columns={'_plot_date':'Date','Amount_numeric':'Total_Spent'}))
        daily_credit = (credit.groupby(credit['_plot_date'])['Amount_numeric'].sum().reset_index().rename(columns={'_plot_date':'Date','Amount_numeric':'Total_Credit'}))
    else:
        overall = (working.groupby(working['_plot_date'])['Amount_numeric'].sum().reset_index().rename(columns={'_plot_date':'Date','Amount_numeric':'Total_Spent'}))
        daily_spend = overall
        daily_credit = pd.DataFrame(columns=['Date','Total_Credit'])

    # Merge debit + credit
    merged = pd.merge(daily_spend, daily_credit, on='Date', how='outer').fillna(0)
    merged['Date'] = pd.to_datetime(merged['Date'])
    merged = merged.sort_values('Date').reset_index(drop=True)
    return merged

def plotly_daily_spend(merged_df: pd.DataFrame, title: str = "Daily Spend and Credit (Plotly)"):
    """Interactive Plotly line chart for merged daily totals."""
    if merged_df is None or merged_df.empty:
        st.info("No daily data to plot (Plotly).")
        return

    # Choose which columns to plot
    y_cols = []
    if 'Total_Spent' in merged_df.columns:
        y_cols.append('Total_Spent')
    if 'Total_Credit' in merged_df.columns and merged_df['Total_Credit'].sum() != 0:
        y_cols.append('Total_Credit')

    if not y_cols:
        st.info("No Total_Spent/Total_Credit columns found for Plotly.")
        return

    # Melt for a clean legend/hover
    plot_df = merged_df[['Date'] + y_cols].melt(id_vars='Date', var_name='Type', value_name='Amount')

    fig = px.line(plot_df, x='Date', y='Amount', color='Type', markers=True, title=title)
    fig.update_layout(template='plotly_white', xaxis_title='Date', yaxis_title='Amount',
                      legend_title='Type', hovermode='x unified')
    fig.update_traces(hovertemplate='%{x|%Y-%m-%d}: %{y:.0f}')
    st.plotly_chart(fig, use_container_width=True)

def matplotlib_daily_spend(merged_df: pd.DataFrame, title: str = "Daily Spend and Credit (Matplotlib)"):
    """Matplotlib static chart for merged daily totals (suitable for st.pyplot)."""
    if merged_df is None or merged_df.empty:
        st.info("No daily data to plot (Matplotlib).")
        return

    # Prepare x and y
    x = pd.to_datetime(merged_df['Date'])
    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)

    if 'Total_Spent' in merged_df.columns:
        ax.plot(x, merged_df['Total_Spent'], marker='o', linestyle='-', linewidth=2, label='Total_Spent')
    if 'Total_Credit' in merged_df.columns and merged_df['Total_Credit'].sum() != 0:
        ax.plot(x, merged_df['Total_Credit'], marker='o', linestyle='--', linewidth=2, label='Total_Credit')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Amount')
    ax.grid(axis='y', alpha=0.3)

    # Format x-axis dates nicely
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # Y-axis integer formatting
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{int(val):,}"))

    ax.legend(loc='upper left')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# --- Usage example (paste after converted_df exists) ---
if 'converted_df' in globals():
    merged = compute_daily_totals(converted_df)
    st.subheader("Daily totals (top rows)")
    st.write(merged.head(10))

    # show both charts side-by-side
    c1, c2 = st.columns(2)
    with c1:
        plotly_daily_spend(merged, title="Daily Spend and Credit — Plotly")
    with c2:
        matplotlib_daily_spend(merged, title="Daily Spend and Credit — Matplotlib")
else:
    st.warning("converted_df not found in the current namespace. Run conversion step first.")

# ---- Replace prior daily-aggregation & charting code with this unified, corrected block ----
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

def compute_daily_totals_consistent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministically compute daily totals from converted_df.
    Returns DataFrame with:
      - Date (datetime64[ns], midnight)
      - Total_Spent (float)
      - Total_Credit (float)
    Logic:
      - group by 'date' (preferred) or 'timestamp' (fallback) or any datetime-like column
      - ensure Amount is numeric (coerce)
      - if 'Type' exists, sum debit and credit separately; otherwise treat all as Total_Spent
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['Date','Total_Spent','Total_Credit'])

    working = df.copy()

    # pick grouping date series (datetime64[ns] at midnight)
    if 'date' in working.columns and working['date'].notna().any():
        grp = pd.to_datetime(working['date']).dt.normalize()
    elif 'timestamp' in working.columns and working['timestamp'].notna().any():
        grp = pd.to_datetime(working['timestamp']).dt.normalize()
    else:
        # try any datetime-like column
        found = None
        for c in working.columns:
            if pd.api.types.is_datetime64_any_dtype(working[c]) and working[c].notna().any():
                found = c
                break
        if found:
            grp = pd.to_datetime(working[found]).dt.normalize()
        else:
            # nothing to group by
            return pd.DataFrame(columns=['Date','Total_Spent','Total_Credit'])

    working['_group_date'] = grp

    # ensure Amount numeric
    if 'Amount' in working.columns:
        working['Amount_numeric'] = pd.to_numeric(working['Amount'], errors='coerce').fillna(0.0)
    else:
        # try find other numeric columns
        numeric_cols = [c for c in working.columns if pd.api.types.is_integer_dtype(working[c]) or pd.api.types.is_float_dtype(working[c])]
        if numeric_cols:
            working['Amount_numeric'] = pd.to_numeric(working[numeric_cols[0]], errors='coerce').fillna(0.0)
        else:
            working['Amount_numeric'] = 0.0

    # compute daily totals
    if 'Type' in working.columns and working['Type'].astype(str).str.strip().any():
        # treat 'debit' and 'credit' text-insensitively
        w = working.copy()
        w['Type_norm'] = w['Type'].astype(str).str.lower().str.strip()
        debit_df = w[w['Type_norm'] == 'debit']
        credit_df = w[w['Type_norm'] == 'credit']

        daily_spend = debit_df.groupby(debit_df['_group_date'])['Amount_numeric'].sum().reset_index().rename(columns={'_group_date':'Date','Amount_numeric':'Total_Spent'})
        daily_credit = credit_df.groupby(credit_df['_group_date'])['Amount_numeric'].sum().reset_index().rename(columns={'_group_date':'Date','Amount_numeric':'Total_Credit'})
    else:
        # no Type: all amounts treated as Total_Spent
        daily_spend = working.groupby(working['_group_date'])['Amount_numeric'].sum().reset_index().rename(columns={'_group_date':'Date','Amount_numeric':'Total_Spent'})
        daily_credit = pd.DataFrame(columns=['Date','Total_Credit'])

    # Merge both, fill zeros, sort
    merged = pd.merge(daily_spend, daily_credit, on='Date', how='outer').fillna(0)
    merged['Date'] = pd.to_datetime(merged['Date']).dt.normalize()
    merged = merged.sort_values('Date').reset_index(drop=True)

    # ensure numeric dtype
    merged['Total_Spent'] = pd.to_numeric(merged['Total_Spent'], errors='coerce').fillna(0.0)
    if 'Total_Credit' in merged.columns:
        merged['Total_Credit'] = pd.to_numeric(merged['Total_Credit'], errors='coerce').fillna(0.0)
    else:
        merged['Total_Credit'] = 0.0

    return merged

def plotly_daily_spend_consistent(merged_df: pd.DataFrame, y_max: float = None):
    if merged_df is None or merged_df.empty:
        st.info("No daily data to plot (Plotly).")
        return
    # melt
    plot_df = merged_df.melt(id_vars='Date', value_vars=['Total_Spent','Total_Credit'], var_name='Type', value_name='Amount')
    # remove all-zero series for legend cleanliness
    plot_df = plot_df[~((plot_df['Type']=='Total_Credit') & (plot_df['Amount']==0))]
    fig = px.line(plot_df, x='Date', y='Amount', color='Type', markers=True, title="Daily Spend and Credit — Plotly")
    fig.update_layout(template='plotly_white', xaxis_title='Date', yaxis_title='Amount', legend_title='Type', hovermode='x unified')
    # set same y-range if provided
    if y_max is not None:
        fig.update_yaxes(range=[0, float(y_max)*1.05])
    st.plotly_chart(fig, use_container_width=True)

def matplotlib_daily_spend_consistent(merged_df: pd.DataFrame, y_max: float = None):
    if merged_df is None or merged_df.empty:
        st.info("No daily data to plot (Matplotlib).")
        return
    x = pd.to_datetime(merged_df['Date'])
    fig, ax = plt.subplots(figsize=(9,3.5), dpi=100)
    ax.plot(x, merged_df['Total_Spent'], marker='o', linestyle='-', linewidth=2, label='Total_Spent')
    # plot credit only if non-zero
    if merged_df['Total_Credit'].sum() != 0:
        ax.plot(x, merged_df['Total_Credit'], marker='o', linestyle='--', linewidth=2, label='Total_Credit')
    ax.set_title("Daily Spend and Credit — Matplotlib")
    ax.set_xlabel('Date')
    ax.set_ylabel('Amount')
    ax.grid(axis='y', alpha=0.25)
    # x-axis format
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    # y-limit unify if provided
    if y_max is not None:
        ax.set_ylim(0, float(y_max)*1.05)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{int(val):,}"))
    ax.legend(loc='upper left')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# --- compute merged and print diagnostics ---
merged = compute_daily_totals_consistent(converted_df)

st.subheader("Daily totals (merged) — top rows")
st.write(merged.head(10))

st.subheader("Daily totals description")
try:
    st.write(merged[['Total_Spent','Total_Credit']].describe().applymap(lambda x: float(x) if pd.notna(x) else x))
except Exception:
    st.write("Describe unavailable (empty)")

# determine y_max for unified axis (use max of both series)
if not merged.empty:
    y_max = max(merged['Total_Spent'].max(skipna=True) if 'Total_Spent' in merged.columns else 0,
                merged['Total_Credit'].max(skipna=True) if 'Total_Credit' in merged.columns else 0)
else:
    y_max = None

# show both charts side-by-side using same merged & same y-range
c1, c2 = st.columns([1,1])
with c1:
    plotly_daily_spend_consistent(merged, y_max=y_max)
with c2:
    matplotlib_daily_spend_consistent(merged, y_max=y_max)

# ---- Unified, dtype-safe daily aggregation + both plotters (Plotly + Matplotlib) ----
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

def compute_daily_totals_consistent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily totals and return a DataFrame with:
      - Date (datetime64[ns], midnight)
      - Total_Spent (float64)
      - Total_Credit (float64)
    Ensures numeric dtypes are native numpy types (no pandas extension Int64).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['Date','Total_Spent','Total_Credit'])

    w = df.copy()

    # choose grouping date
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

    # ensure Amount_numeric as float64
    if 'Amount' in w.columns:
        w['Amount_numeric'] = pd.to_numeric(w['Amount'], errors='coerce').fillna(0.0).astype('float64')
    else:
        # fallback: pick any numeric-like column
        numeric_cols = [c for c in w.columns if pd.api.types.is_integer_dtype(w[c]) or pd.api.types.is_float_dtype(w[c])]
        if numeric_cols:
            w['Amount_numeric'] = pd.to_numeric(w[numeric_cols[0]], errors='coerce').fillna(0.0).astype('float64')
        else:
            w['Amount_numeric'] = 0.0

    # compute debit/credit or overall
    if 'Type' in w.columns and w['Type'].astype(str).str.strip().any():
        w['Type_norm'] = w['Type'].astype(str).str.lower().str.strip()
        debit_df = w[w['Type_norm'] == 'debit']
        credit_df = w[w['Type_norm'] == 'credit']

        daily_spend = (debit_df.groupby(debit_df['_group_date'])['Amount_numeric']
                       .sum().reset_index().rename(columns={'_group_date':'Date','Amount_numeric':'Total_Spent'}))
        daily_credit = (credit_df.groupby(credit_df['_group_date'])['Amount_numeric']
                        .sum().reset_index().rename(columns={'_group_date':'Date','Amount_numeric':'Total_Credit'}))
    else:
        daily_spend = (w.groupby(w['_group_date'])['Amount_numeric']
                       .sum().reset_index().rename(columns={'_group_date':'Date','Amount_numeric':'Total_Spent'}))
        daily_credit = pd.DataFrame(columns=['Date','Total_Credit'])

    merged = pd.merge(daily_spend, daily_credit, on='Date', how='outer').fillna(0)
    merged['Date'] = pd.to_datetime(merged['Date']).dt.normalize()

    # convert to native numeric dtypes (float64) to avoid Plotly issues with pandas extension types
    merged['Total_Spent'] = merged['Total_Spent'].astype('float64')
    merged['Total_Credit'] = merged.get('Total_Credit', 0).astype('float64') if 'Total_Credit' in merged else np.array([0.0]*len(merged), dtype='float64')

    merged = merged.sort_values('Date').reset_index(drop=True)
    return merged

def plotly_daily_spend_consistent(merged_df: pd.DataFrame, y_max: float = None):
    if merged_df is None or merged_df.empty:
        st.info("No daily data to plot (Plotly).")
        return

    # prepare melted dataframe and ensure Amount is float
    plot_df = merged_df.melt(id_vars='Date', value_vars=['Total_Spent','Total_Credit'], var_name='Type', value_name='Amount')
    plot_df['Amount'] = pd.to_numeric(plot_df['Amount'], errors='coerce').fillna(0.0).astype('float64')
    # drop zero-credit series if all zero
    if plot_df[(plot_df['Type']=='Total_Credit')]['Amount'].sum() == 0:
        plot_df = plot_df[plot_df['Type']!='Total_Credit']

    # create fig
    fig = px.line(plot_df, x='Date', y='Amount', color='Type', markers=True, title="Daily Spend and Credit — Plotly")
    fig.update_traces(mode='lines+markers', marker={'size':6})
    fig.update_layout(template='plotly_white', xaxis_title='Date', yaxis_title='Amount', legend_title='Type', hovermode='x unified')

    # enforce same y-range if passed
    if y_max is not None:
        fig.update_yaxes(range=[0, float(y_max)*1.05])

    # format hover and y tick formatting
    fig.update_traces(hovertemplate='%{x|%Y-%m-%d}: %{y:.0f}')
    fig.update_yaxes(tickformat=",")  # thousand separators
    st.plotly_chart(fig, use_container_width=True)

def matplotlib_daily_spend_consistent(merged_df: pd.DataFrame, y_max: float = None):
    if merged_df is None or merged_df.empty:
        st.info("No daily data to plot (Matplotlib).")
        return

    x = pd.to_datetime(merged_df['Date'])
    fig, ax = plt.subplots(figsize=(9,3.5), dpi=100)

    ax.plot(x, merged_df['Total_Spent'], marker='o', linestyle='-', linewidth=2, label='Total_Spent')
    if merged_df['Total_Credit'].sum() != 0:
        ax.plot(x, merged_df['Total_Credit'], marker='o', linestyle='--', linewidth=2, label='Total_Credit')

    ax.set_title("Daily Spend and Credit — Matplotlib")
    ax.set_xlabel('Date'); ax.set_ylabel('Amount')
    ax.grid(axis='y', alpha=0.25)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    if y_max is not None:
        ax.set_ylim(0, float(y_max)*1.05)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{int(val):,}"))
    ax.legend(loc='upper left')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# --- compute merged and show diagnostics and charts ---
merged = compute_daily_totals_consistent(converted_df)

st.subheader("Daily totals (merged) — top rows")
st.write(merged.head(10))

st.subheader("Daily totals description")
try:
    st.write(merged[['Total_Spent','Total_Credit']].describe().applymap(lambda x: float(x) if pd.notna(x) else x))
except Exception:
    st.write("Describe unavailable (empty)")

# unified y_max
if not merged.empty:
    y_max = max(float(merged['Total_Spent'].max(skipna=True)), float(merged['Total_Credit'].max(skipna=True)))
else:
    y_max = None

c1, c2 = st.columns([1,1])
with c1:
    plotly_daily_spend_consistent(merged, y_max=y_max)
with c2:
    matplotlib_daily_spend_consistent(merged, y_max=y_max)

show_debit = st.checkbox("Show debit", value=True)
show_credit = st.checkbox("Show credit", value=True)
series_to_plot = []
if show_debit: series_to_plot.append('Total_Spent')
if show_credit: series_to_plot.append('Total_Credit')


