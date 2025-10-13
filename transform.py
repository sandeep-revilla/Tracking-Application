# transform.py  -- pure data transformation utilities
import pandas as pd

def _is_date_like_column(col_name: str) -> bool:
    lname = str(col_name).lower()
    date_keywords = ['date', 'time', 'timestamp', 'datetime', 'txn']
    return any(k in lname for k in date_keywords) or lname.startswith("unnamed")

def _is_amount_like_column(col_name: str) -> bool:
    lname = str(col_name).lower()
    num_keywords = ['amount', 'amt', 'value', 'total', 'balance', 'credit', 'debit', 'spent']
    return any(k in lname for k in num_keywords)

def convert_columns_and_derives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns:
      - detect and coerce date/time columns -> timestamp
      - detect and coerce numeric columns -> Amount (float)
      - create 'date' column (date part of timestamp)
      - infer Type ('debit'/'credit') if missing based on sign of Amount
    Returns a cleaned DataFrame (does not mutate input).
    Rows where Type is unknown are dropped.
    """
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    # strip column names
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # 1) Detect & coerce date-like columns to datetime (prefer the first reasonable column)
    primary_dt_col = None
    for col in df.columns:
        if _is_date_like_column(col):
            parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=False)
            if parsed.notna().sum() > 0:
                df[col] = parsed
                primary_dt_col = col
                break
    # fallback: try to find any column that can be parsed as datetime
    if primary_dt_col is None:
        for col in df.columns:
            if df[col].dtype == object:
                parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=False)
                if parsed.notna().sum() >= 3:
                    df[col] = parsed
                    primary_dt_col = col
                    break

    # 2) Detect amount-like columns and coerce to numeric (choose preferred)
    amount_cols = []
    for col in df.columns:
        if _is_amount_like_column(col):
            coerced = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce')
            df[col] = coerced
            amount_cols.append(col)

    # Heuristic: if no obvious amount column found, try to coerce any object column that looks numeric
    if not amount_cols:
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].astype(str).head(20).str.replace(r'[^\d\.\-]', '', regex=True)
                parsed = pd.to_numeric(sample, errors='coerce')
                if parsed.notna().sum() >= 3:
                    coerced = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce')
                    df[col] = coerced
                    amount_cols.append(col)
                    break

    # Choose preferred amount column name
    preferred = None
    for candidate in ['amount','total_spent','totalspent','total','txn amount','value','spent','amt']:
        for col in df.columns:
            if str(col).lower() == candidate:
                preferred = col
                break
        if preferred:
            break
    if not preferred and amount_cols:
        preferred = amount_cols[0]

    if preferred:
        if preferred != 'Amount':
            if 'Amount' not in df.columns:
                df.rename(columns={preferred: 'Amount'}, inplace=True)
    # Ensure Amount column exists
    if 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').astype('float64')
    else:
        # create Amount with NA if nothing found
        df['Amount'] = pd.NA

    # 3) Create canonical timestamp and date columns
    if primary_dt_col:
        df['timestamp'] = pd.to_datetime(df[primary_dt_col], errors='coerce')
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        # Last resort: no date-like column found, leave NaT
        df['timestamp'] = pd.NaT

    # date (date part)
    try:
        df['date'] = df['timestamp'].dt.date
    except Exception:
        df['date'] = pd.NA

    # 4) Infer Type if missing: positive Amount -> 'debit' (spent), negative -> 'credit'
    if 'Type' not in df.columns:
        try:
            df['Type'] = pd.NA
            mask_pos = df['Amount'].notna() & (df['Amount'] > 0)
            mask_neg = df['Amount'].notna() & (df['Amount'] < 0)
            df.loc[mask_pos, 'Type'] = 'debit'
            df.loc[mask_neg, 'Type'] = 'credit'
        except Exception:
            # leave Type as NA if something goes wrong
            df['Type'] = pd.NA
    else:
        # normalize existing Type values
        try:
            df['Type'] = df['Type'].astype(str).str.lower().str.strip()
        except Exception:
            pass

    # Normalize Type column to canonical lower-case strings and mark empties as 'unknown'
    try:
        df['Type'] = df['Type'].astype(str).str.lower().str.strip()
        df['Type'] = df['Type'].replace({'nan': 'unknown', 'none': 'unknown', '': 'unknown', 'na': 'unknown', 'null': 'unknown'})
    except Exception:
        # fallback: if normalization fails, set everything to unknown to be safe
        df['Type'] = 'unknown'

    # DROP rows where Type is unknown (user requested)
    try:
        df = df[df['Type'] != 'unknown'].copy()
    except Exception:
        # if something unexpected happens, keep dataframe as-is
        pass

    # final reorder: put timestamp and date at front
    cols = list(df.columns)
    final = []
    if 'timestamp' in cols:
        final.append('timestamp')
    if 'date' in cols:
        final.append('date')
    for c in cols:
        if c not in final:
            final.append(c)
    df = df[final]

    return df


def compute_daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily totals DataFrame with columns:
      - Date (normalized datetime)
      - Total_Spent (float)  <- sum of debits / spend
      - Total_Credit (float) <- sum of credits
    Logic:
      - If 'Type' column exists and contains debit/credit: use it.
      - Else: treat positive Amount as spend (debit), negative as credit.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['Date', 'Total_Spent', 'Total_Credit'])

    w = df.copy()

    # determine grouping date
    if 'date' in w.columns and w['date'].notna().any():
        grp = pd.to_datetime(w['date']).dt.normalize()
    elif 'timestamp' in w.columns and w['timestamp'].notna().any():
        grp = pd.to_datetime(w['timestamp']).dt.normalize()
    else:
        # try to find any datetime-like column
        found = None
        for c in w.columns:
            if pd.api.types.is_datetime64_any_dtype(w[c]) and w[c].notna().any():
                found = c
                break
        if found:
            grp = pd.to_datetime(w[found]).dt.normalize()
        else:
            return pd.DataFrame(columns=['Date', 'Total_Spent', 'Total_Credit'])

    w['_group_date'] = grp
    w['Amount_numeric'] = pd.to_numeric(w.get('Amount', 0), errors='coerce').fillna(0.0).astype('float64')

    # If Type present and has meaningful values, use it
    if 'Type' in w.columns and w['Type'].astype(str).str.strip().any():
        w['Type_norm'] = w['Type'].astype(str).str.lower().str.strip()
        debit_df = w[w['Type_norm'] == 'debit']
        credit_df = w[w['Type_norm'] == 'credit']
        daily_spend = debit_df.groupby(debit_df['_group_date'])['Amount_numeric'].sum().reset_index().rename(columns={'_group_date': 'Date', 'Amount_numeric': 'Total_Spent'})
        daily_credit = credit_df.groupby(credit_df['_group_date'])['Amount_numeric'].sum().reset_index().rename(columns={'_group_date': 'Date', 'Amount_numeric': 'Total_Credit'})
    else:
        # fallback: positive = spend, negative = credit
        debit_df = w[w['Amount_numeric'] > 0]
        credit_df = w[w['Amount_numeric'] < 0]
        daily_spend = debit_df.groupby(debit_df['_group_date'])['Amount_numeric'].sum().reset_index().rename(columns={'_group_date': 'Date', 'Amount_numeric': 'Total_Spent'})
        daily_credit = credit_df.groupby(credit_df['_group_date'])['Amount_numeric'].sum().reset_index().rename(columns={'_group_date': 'Date', 'Amount_numeric': 'Total_Credit'})
        # credit sums are negative values; make them positive for reporting
        if 'Total_Credit' in daily_credit.columns:
            daily_credit['Total_Credit'] = daily_credit['Total_Credit'].abs()

    merged = pd.merge(daily_spend, daily_credit, on='Date', how='outer').fillna(0)
    merged['Date'] = pd.to_datetime(merged['Date']).dt.normalize()
    merged['Total_Spent'] = merged.get('Total_Spent', 0).astype('float64')
    merged['Total_Credit'] = merged.get('Total_Credit', 0).astype('float64')
    merged = merged.sort_values('Date').reset_index(drop=True)
    return merged
