# app.py
import streamlit as st
import pandas as pd
import json
import gspread
import plotly.express as px

st.set_page_config(page_title="Live Expense Tracker", layout="wide")
st.title("💸 Live Expense Tracker (Google Sheets → Streamlit)")

# -------------------------
# Sidebar: connection inputs
# -------------------------
st.sidebar.header("Google Sheet connection")
SHEET_ID = st.sidebar.text_input("Google Sheet ID (between /d/ and /edit)", "")
sheet_name_override = st.sidebar.text_input("Worksheet name (optional)", "")
refresh = st.sidebar.button("Refresh now")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Instructions: Put your Google service account JSON in Streamlit Secrets under the key "
    "`gcp_service_account` (Cloud: add in Settings → Secrets; Local: .streamlit/secrets.toml). "
    "Share the sheet with the service account email (client_email) as Viewer."
)

# -------------------------
# Robust secret loader
# -------------------------
def load_service_account_secret() -> dict:
    """
    Returns a parsed dict for the service account JSON stored in st.secrets["gcp_service_account"].
    Accepts:
      - st.secrets['gcp_service_account'] as a dict (already parsed)
      - a string containing raw JSON
      - a triple-quoted TOML string ("""...""") from .streamlit/secrets.toml
    Raises ValueError if parsing fails.
    """
    if "gcp_service_account" not in st.secrets:
        raise KeyError("gcp_service_account not found in Streamlit secrets.")
    raw = st.secrets["gcp_service_account"]

    # already dict-like
    if isinstance(raw, dict):
        return raw

    s = str(raw).strip()

    # If user pasted TOML triple-quoted JSON, remove surrounding quotes
    if (s.startswith('"""') and s.endswith('"""')) or (s.startswith("'''") and s.endswith("'''")):
        s = s[3:-3].strip()

    # Try parsing JSON directly; handle common private_key newline variants
    try:
        parsed = json.loads(s)
        return parsed
    except Exception:
        # Replace literal newlines in the private key so JSON becomes valid if needed,
        # also try replacing escaped newlines with actual newlines if appropriate.
        try:
            parsed = json.loads(s.replace('\\n', '\n'))
            return parsed
        except Exception:
            try:
                parsed = json.loads(s.replace('\n', '\\n'))
                return parsed
            except Exception as e:
                raise ValueError("Could not parse gcp_service_account secret as JSON. "
                                 "Ensure you pasted the service account JSON correctly.") from e


# -------------------------
# gspread client factory (robust)
# -------------------------
@st.cache_data(ttl=300)
def get_gspread_client():
    try:
        creds_info = load_service_account_secret()
    except KeyError as e:
        st.error("Service account JSON not found in secrets. Add `gcp_service_account` to Secrets.")
        raise st.StopException
    except Exception as e:
        st.error(f"Error parsing service account JSON from secrets: {e}")
        raise st.StopException

    # Use gspread helper which avoids internal auth attribute mismatch.
    try:
        gc = gspread.service_account_from_dict(creds_info)
        return gc
    except Exception as e:
        st.error(f"Failed to create gspread client: {e}")
        raise st.StopException


# -------------------------
# Sheet operations
# -------------------------
@st.cache_data(ttl=60)
def get_sheet_titles(sheet_id: str):
    if not sheet_id:
        return []
    gc = get_gspread_client()
    try:
        sh = gc.open_by_key(sheet_id)
    except Exception as e:
        raise RuntimeError(f"Unable to open sheet by id: {e}") from e
    return [ws.title for ws in sh.worksheets()]


@st.cache_data(ttl=60)
def load_sheet_as_df(sheet_id: str, worksheet_name: str | None) -> pd.DataFrame:
    if not sheet_id:
        return pd.DataFrame()
    gc = get_gspread_client()
    try:
        sh = gc.open_by_key(sheet_id)
    except Exception as e:
        st.error(f"Unable to open sheet by id: {e}")
        return pd.DataFrame()
    try:
        ws = sh.worksheet(worksheet_name) if worksheet_name else sh.get_worksheet(0)
    except Exception:
        # fallback to first worksheet
        try:
            ws = sh.get_worksheet(0)
        except Exception as e:
            st.error(f"Unable to access worksheet: {e}")
            return pd.DataFrame()
    try:
        records = ws.get_all_records(empty2zero=False, head=1)
        df = pd.DataFrame.from_records(records)
    except Exception as e:
        st.error(f"Failed to read data from worksheet: {e}")
        return pd.DataFrame()
    return df


# Refresh handling
if refresh:
    st.experimental_rerun()

# Populate worksheet dropdown (if sheet id provided)
worksheet_titles = []
if SHEET_ID:
    try:
        worksheet_titles = get_sheet_titles(SHEET_ID)
    except Exception as e:
        # show non-blocking message; get_sheet_titles will surface the error when called directly
        st.warning(f"Could not list worksheets: {e}")
        worksheet_titles = []

selected_sheet = None
if worksheet_titles:
    selected_sheet = st.sidebar.selectbox("Choose worksheet", options=worksheet_titles, index=0)
if sheet_name_override.strip():
    selected_sheet = sheet_name_override.strip()

# Load dataframe
df = load_sheet_as_df(SHEET_ID, selected_sheet)

if df.empty:
    st.info("No data loaded yet. Provide Sheet ID and ensure worksheet has rows and headers.")
    st.stop()

# ---------- Data cleaning ----------
st.subheader("Raw data preview")
st.dataframe(df.head(10), use_container_width=True)

# heuristics for important columns
col_map = {c.lower(): c for c in df.columns}
date_col = col_map.get("datetime") or col_map.get("date")
amount_col = col_map.get("amount") or col_map.get("amt")
type_col = col_map.get("type")
message_col = col_map.get("message") or col_map.get("msg")
bank_col = col_map.get("bank")

work = df.copy()

# Amount
if amount_col and amount_col in work.columns:
    work["Amount"] = pd.to_numeric(work[amount_col], errors="coerce")
else:
    numeric_cols = work.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        work["Amount"] = pd.to_numeric(work[numeric_cols[0]], errors="coerce")
    else:
        # attempt to extract numbers from text columns (simple fallback)
        def extract_num_from_row(row):
            import re
            s = " ".join([str(x) for x in row])
            m = re.search(r"[-]?\d+[\d,]*(\.\d+)?", s.replace(",", ""))
            return float(m.group(0)) if m else None
        work["Amount"] = work.apply(extract_num_from_row, axis=1)

# Date
if date_col and date_col in work.columns:
    work["DateTime"] = pd.to_datetime(work[date_col], errors="coerce")
else:
    # try converting any column to datetime
    dt_assigned = False
    for c in work.columns:
        try:
            tmp = pd.to_datetime(work[c], errors="coerce")
            if tmp.notna().sum() > 0:
                work["DateTime"] = tmp
                dt_assigned = True
                break
        except Exception:
            continue
    if not dt_assigned:
        work["DateTime"] = pd.NaT

work["Date"] = pd.to_datetime(work["DateTime"]).dt.date
work["Month"] = pd.to_datetime(work["DateTime"]).dt.to_period("M").astype(str)
work["Weekday"] = pd.to_datetime(work["DateTime"]).dt.day_name()

# Type normalization
if type_col and type_col in work.columns:
    work["Type"] = work[type_col].astype(str).str.lower().str.strip()
else:
    work["Type"] = "unknown"
    work.loc[work["Amount"] < 0, "Type"] = "debit"
    work.loc[work["Amount"] > 0, "Type"] = "credit"
    if message_col and message_col in work.columns:
        work["Type"] = work[message_col].astype(str).str.lower().apply(
            lambda x: "debit" if any(tok in x for tok in ["deb", "withdraw", "paid", "purchase", "spent"]) else
                      ("credit" if any(tok in x for tok in ["cred", "credited", "refund"]) else "unknown")
        ).combine_first(work["Type"])

work["Amount"] = pd.to_numeric(work["Amount"], errors="coerce")

# ---------- Summary metrics ----------
st.markdown("---")
st.header("Summary metrics")
col1, col2, col3, col4 = st.columns(4)
total_debit = work.loc[work["Type"] == "debit", "Amount"].sum(min_count=1)
total_credit = work.loc[work["Type"] == "credit", "Amount"].sum(min_count=1)
txn_count = len(work)
last_update = work["DateTime"].max()

col1.metric("Total Spent (Debit)", f"{(total_debit or 0):,.2f}")
col2.metric("Total Credit", f"{(total_credit or 0):,.2f}")
col3.metric("Transactions", txn_count)
col4.metric("Latest txn", str(last_update) if pd.notna(last_update) else "N/A")

# ---------- Charts ----------
st.markdown("---")
st.subheader("Interactive charts")

# Daily spending
daily = work[work["Type"] == "debit"].groupby("Date")["Amount"].sum().reset_index()
if not daily.empty:
    fig1 = px.line(daily, x="Date", y="Amount", title="Daily Spending (debits)", markers=True)
    st.plotly_chart(fig1, use_container_width=True)

# Monthly stacked bar
monthly = work.groupby(["Month", "Type"])["Amount"].sum().reset_index()
if not monthly.empty:
    monthly_pivot = monthly.pivot(index="Month", columns="Type", values="Amount").fillna(0).reset_index()
    types = [c for c in monthly_pivot.columns if c != "Month"]
    fig2 = px.bar(monthly_pivot, x="Month", y=types, title="Monthly Debit vs Credit (stacked)", barmode="stack")
    st.plotly_chart(fig2, use_container_width=True)

# Top merchants (simple heuristic)
if message_col and message_col in work.columns:
    merchants = work[message_col].astype(str).str.extract(r"(?:to|at|@)\s+([A-Za-z0-9 &\.\-\/]{3,60})", expand=False)
    work["merchant"] = merchants.fillna("").str.strip()
    top_merchants = work[work["merchant"] != ""].groupby("merchant")["Amount"].sum().sort_values(ascending=False).head(10).reset_index()
    if not top_merchants.empty:
        fig3 = px.bar(top_merchants, x="merchant", y="Amount", title="Top merchants by spend (heuristic)")
        st.plotly_chart(fig3, use_container_width=True)

# Weekday average
weekday = work[work["Type"] == "debit"].groupby("Weekday")["Amount"].mean().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
).reset_index()
if not weekday["Amount"].isna().all():
    fig4 = px.bar(weekday, x="Weekday", y="Amount", title="Average spend by weekday")
    st.plotly_chart(fig4, use_container_width=True)

# Preview cleaned table + download
st.markdown("---")
st.subheader("Cleaned transactions (preview)")
st.dataframe(work.head(200), use_container_width=True)

csv = work.to_csv(index=False).encode("utf-8")
st.download_button("Download cleaned CSV", data=csv, file_name="transactions_cleaned.csv", mime="text/csv")

st.markdown("---")
st.caption("App reads your Google Sheet live whenever the page is loaded or refreshed. Use the Refresh button to re-pull data immediately.")
