"""
col_count = len(header)
normalized = []
for r in data_rows:
if len(r) < col_count:
r = r + [None] * (col_count - len(r))
elif len(r) > col_count:
r = r[:col_count]
normalized.append(r)
return header, normalized




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




def build_sheets_service_from_info(creds_info: Dict):
scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
creds = service_account.Credentials.from_service_account_info(creds_info, scopes=scopes)
return build("sheets", "v4", credentials=creds, cache_discovery=False)




def build_sheets_service_from_file(creds_file: str):
if not os.path.exists(creds_file):
raise FileNotFoundError(f"Credentials file not found: {creds_file}")
scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
creds = service_account.Credentials.from_service_account_file(creds_file, scopes=scopes)
return build("sheets", "v4", credentials=creds, cache_discovery=False)




def read_google_sheet(spreadsheet_id: str, range_name: str,
creds_info: Optional[Dict] = None, creds_file: Optional[str] = None,
secrets: Optional[Dict] = None) -> pd.DataFrame:
"""
Read a Google Sheet and return a DataFrame.
If creds_info is None and creds_file is not provided, the function will try to use
the provided `secrets` dict (typically st.secrets) containing key 'gcp_service_account'.
"""
if creds_info is None and (creds_file is None or not os.path.exists(creds_file)):
if not secrets or 'gcp_service_account' not in secrets:
raise ValueError("No credentials found. Add service account JSON to secrets['gcp_service_account'] or supply a local file.")
creds_info = parse_service_account_secret(secrets['gcp_service_account'])
service = (build_sheets_service_from_info(creds_info) if creds_info else build_sheets_service_from_file(creds_file))
try:
sheet = service.spreadsheets()
res = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
values = res.get("values", [])
except HttpError as e:
raise RuntimeError(f"Google Sheets API error: {e}")
return values_to_dataframe(values)
