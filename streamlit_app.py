# streamlit_app_min.py
import streamlit as st
import pandas as pd
import json, os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import plotly.express as px

st.title("Debug: Daily debit totals (true daily, zigzag)")

# ---------- simple google sheets reader (same idea) ----------
SHEET_ID = st.sidebar.text_input("Sheet ID", value="1KZq_GLXdMBfQUhtp-NA8Jg-flxOppw7kFuIN6y_nOXk")
RANGE = st.sidebar.text_input("Sheet name / range", value="History Transactions")
CREDS_FILE = st.sidebar.text_input("Service account JSON (optional)", value="creds/service_account.json")

def build_sheets_service(creds_file=None):
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    if cre ds := None: pass
    if os.path.exists(creds_file):
        creds = service_account.Credentials.from_service_account_file(creds_file, scopes=scopes)
        return build("sheets", "v4", credentials=creds, cache_discovery=False)
    else:
        st.error("No local creds file provided. Put credentials or use st.secrets for cloud.")
        st.stop()

def read_sheet(spreadsheet_id, range_name, creds_file):
    svc = build_sheets_service(creds_file)
    sheet = svc.spreadsheets()
    res = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    vals = res.get("values", [])
    if not vals:
        return pd.DataFrame()
    header = [h.strip() for h in vals[0]]
    rows = vals[1:]
    return pd.DataFrame(rows, columns=header)

# --------- load
df = read_sheet(SHEET_ID, RANGE, CREDS_FILE)
st.write("Raw sheet rows:", len(df))
st.dataframe(df.head(10))

# --------- Normalize and parse
# tolerant column name matching
cols_lower = {c.lower().strip(): c for c in df.columns}
if "datetime" in cols_lower:
    dt_col = cols_lower["datetime"]
elif "date" in cols_lower:
    dt_col = cols_lower["date"]
else:
    st.error("No DateTime/Date column found. Columns: " + ", ".join(df.columns))
    st.stop()

amt_col = None
for k in cols_lower:
    if "amount" in k or "amt" in k:
        amt_col = cols_lower[k]
        break
if not amt_col:
    st.error("No Amount column found. Columns: " + ", ".join(df.columns))
    st.stop()

type_col = None
for k in cols_lower:
    if k == "type" or "type" in k:
        type_col = cols_lower[k]
        break
if not type_col:
    st.error("No Type column found.")
    st.stop()

# parse
df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
df[amt_col] = (df[amt_col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(r"[^\d\.\-]", "", regex=True))
df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce")
df = df.dropna(subset=[dt_col, amt_col, type_col])
df[type_col] = df[type_col].astype(str).str.lower().str.strip()

# filter only debit rows (robust contains)
df_debit = df[df[type_col].str.contains("debit", na=False)].copy()
st.write("Debit rows:", len(df_debit))
st.dataframe(df_debit.head(10))

# ensure positive magnitudes
df_debit[amt_col] = df_debit[amt_col].abs()

# group by DATE (not cumulative)
df_debit["DateOnly"] = pd.to_datetime(df_debit[dt_col].dt.date)  # keep as datetime (midnight)
daily_spend = df_debit.groupby("DateOnly", as_index=False)[amt_col].sum().rename(columns={amt_col: "Total_Spent"})
daily_spend = daily_spend.sort_values("DateOnly").reset_index(drop=True)

st.subheader("Diagnostics: daily_spend")
st.dataframe(daily_spend.head(30))

daily_spend["diff"] = daily_spend["Total_Spent"].diff()
st.subheader("First diffs (positive = up, negative = down)")
st.dataframe(daily_spend[["DateOnly", "Total_Spent", "diff"]].head(30))

# Plot — straight lines (zigzag)
fig = px.line(daily_spend, x="DateOnly", y="Total_Spent",
              title="Daily Debit (true daily totals — zigzag)", markers=True,
              line_shape="linear")
fig.update_traces(mode="lines+markers")
fig.update_layout(xaxis_title="Date", yaxis_title="Total Spent (₹)", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)
