# streamlit_app.py
import streamlit as st
import pandas as pd
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

st.set_page_config(page_title="Private Sheet Loader", layout="wide")
st.title("🔐 Private Google Sheet — connector (minimal)")

# Sidebar inputs
SHEET_ID = st.sidebar.text_input("Google Sheet ID (between /d/ and /edit)", "1KZq_GLXdM8FQUhtp-NA8Jq-fIxOppw7kFuIN6y_nQXk")
RANGE = st.sidebar.text_input("Range (optional, e.g. Sheet1!A1:Z1000)", "A1:Z1000")
if st.sidebar.button("Refresh now"):
    st.experimental_rerun()

# Helper: parse service account JSON from st.secrets
def load_service_account_secret():
    if "gcp_service_account" not in st.secrets:
        raise KeyError("gcp_service_account not found in st.secrets.")
    raw = st.secrets["gcp_service_account"]
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    # strip surrounding triple quotes if present
    if (s.startswith('"""') and s.endswith('"""')) or (s.startswith("'''") and s.endswith("'''")):
        s = s[3:-3].strip()
    # try a few parse variants
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(s.replace('\\n','\n'))
        except Exception:
            return json.loads(s.replace('\n','\\n'))

# Build Sheets API client (cached)
@st.cache_data(ttl=300)
def build_sheets_service():
    creds_json = load_service_account_secret()
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = service_account.Credentials.from_service_account_info(creds_json, scopes=scopes)
    service = build("sheets", "v4", credentials=creds, cache_discovery=False)
    return service

# Run
if not SHEET_ID:
    st.info("Enter the private Google Sheet ID in the sidebar to load data.")
    st.stop()

try:
    service = build_sheets_service()
except KeyError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Failed to build Sheets client: {e}")
    st.stop()

try:
    sheet = service.spreadsheets()
    res = sheet.values().get(spreadsheetId=SHEET_ID, range=RANGE).execute()
    values = res.get("values", [])
except Exception as e:
    st.error(f"API read error: {e}")
    st.stop()

if not values:
    st.info("No data returned. Check the range and that the service account has Viewer access to the sheet.")
    st.stop()

header = values[0]
rows = values[1:]
df = pd.DataFrame(rows, columns=header)

st.subheader("Loaded data (preview)")
st.dataframe(df.head(50), use_container_width=True)
st.caption(f"Rows loaded: {len(df)}")
