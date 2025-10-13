# === FILE: app.py (main Streamlit app) ===
"""
Main Streamlit app that wires the sidebar UI, loads data (via io.py), transforms it (transform.py)
and renders charts (charts.py). Keep this file minimal: it orchestrates the pieces.
"""


import streamlit as st
import pandas as pd
from datetime import date


# local module imports (make sure files are in the same directory)
import io as io_mod
import transform as tf
import charts as charts_mod


st.set_page_config(page_title="Sheet → Daily Spend (modular)", layout="wide")
st.title("💳 Daily Spending — Modular Version")


# Sidebar: data source
with st.sidebar:
st.header("Data source & filters")
SHEET_ID = st.text_input("Google Sheet ID (between /d/ and /edit)", value="")
RANGE = st.text_input("Range or Sheet Name", value="History Transactions")
CREDS_FILE = st.text_input("Service Account JSON File (optional)", value="creds/service_account.json")
st.markdown("---")
st.subheader("Chart options")
enable_plotly_click = st.checkbox("Enable click-to-select (Plotly)", value=False)
st.markdown("---")
st.write("Series to include")
show_debit = st.checkbox("Debit (Total_Spent)", value=True)
show_credit = st.checkbox("Credit (Total_Credit)", value=True)
st.markdown("---")
st.write("Data input: choose CSV upload (recommended for now) or Google Sheet")
data_source = st.radio("Data source", ["Upload CSV/XLSX", "Google Sheet"], index=0)


# Load data (either uploaded file or Google Sheet)
uploaded = None
if data_source == "Upload CSV/XLSX":
uploaded = st.file_uploader("Upload CSV or XLSX (HDFC / Indian Bank)", type=["csv","xlsx"], help="File processed in-memory; not stored.")


# End of modularized files
