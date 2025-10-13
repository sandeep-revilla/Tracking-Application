"""
except Exception:
pass


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
