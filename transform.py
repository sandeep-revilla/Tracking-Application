# transform.py  -- pure data transformation utilities (extended with outlier handling)
import pandas as pd
from typing import Optional, Tuple, Dict
import numpy as np

def _is_date_like_column(col_name: str) -> bool:
    lname = str(col_name).lower()
    date_keywords = ['date', 'time', 'timestamp', 'datetime', 'txn']
    return any(k in lname for k in date_keywords) or lname.startswith("unnamed")

def _is_amount_like_column(col_name: str) -> bool:
    lname = str(col_name).lower()
    num_keywords = ['amount', 'amt', 'value', 'total', 'balance', 'credit', 'debit', 'spent']
    return any(k in lname for k in num_keywords)

def _detect_is_deleted_mask(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    If an 'is_deleted' column (case-insensitive) exists in df, return a boolean mask
    where True indicates the row is deleted. Accepts boolean, numeric, and common string
    values ('true','t','1','yes'). Returns None if no such column exists.
    """
    isdel_col = next((c for c in df.columns if str(c).lower() == 'is_deleted'), None)
    if isdel_col is None:
        return None

    s = df[isdel_col]
    try:
        # boolean dtype
        if pd.api.types.is_bool_dtype(s):
            return s.fillna(False).astype(bool)
        # numeric (1/0)
        if pd.api.types.is_numeric_dtype(s):
            return s.fillna(0).astype(int) == 1
        # strings / mixed - normalize and check common true tokens
        lowered = s.astype(str).str.strip().str.lower().fillna('')
        return lowered.isin(['true', 't', '1', 'yes', 'y'])
    except Exception:
        try:
            lowered = s.astype(str).str.strip().str.lower().fillna('')
            return lowered.isin(['true', 't', '1', 'yes', 'y'])
        except Exception:
            return None

def _prefer_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """
    Prefer an explicit 'DateTime' (case-insensitive) or any column containing 'datetime'
    if present. Otherwise return None (caller will fallback to heuristic detection).
    """
    for c in df.columns:
        if str(c).lower() == 'datetime' or str(c).lower() == 'date_time':
            return c
    for c in df.columns:
        if 'datetime' in str(c).lower():
            return c
    return None

def convert_columns_and_derives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns:
      - remove rows marked deleted via an 'is_deleted' column (case-insensitive)
      - detect and coerce date/time columns -> timestamp (preferring DateTime if present)
      - detect and coerce numeric columns -> Amount (float)
      - create 'date' column (date part of timestamp)
      - infer Type ('debit'/'credit') if missing based on sign of Amount

    Returns a cleaned DataFrame (does not mutate input). Preserves any extra columns
    (like _sheet_row_idx, _source_sheet) so they survive transformation.
    Rows where Type is unknown are dropped (same behavior as before).
    """
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    # Work on a copy to avoid mutating caller's DataFrame
    df = df.copy()

    # Preserve original column names (strip only whitespace)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # 0) Filter out soft-deleted rows early (if present)
    try:
        isdel_mask = _detect_is_deleted_mask(df)
        if isdel_mask is not None:
            df = df.loc[~isdel_mask].copy().reset_index(drop=True)
    except Exception:
        # if detection fails, continue without filtering
        pass

    # 1) Prefer explicit DateTime-like column if available
    primary_dt_col = _prefer_datetime_column(df)

    # 2) If not found, fall back to the original heuristic detection
    if primary_dt_col is None:
        for col in df.columns:
            try:
                if _is_date_like_column(col):
                    parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=False)
                    if parsed.notna().sum() > 0:
                        primary_dt_col = col
                        df[col] = parsed
                        break
            except Exception:
                continue

    # fallback: try to find any object column that can be parsed as datetime
    if primary_dt_col is None:
        for col in df.columns:
            try:
                if df[col].dtype == object:
                    parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=False)
                    if parsed.notna().sum() >= 3:
                        primary_dt_col = col
                        df[col] = parsed
                        break
            except Exception:
                continue

    # 3) Detect amount-like columns and coerce to numeric (choose preferred)
    amount_cols = []
    for col in df.columns:
        try:
            if _is_amount_like_column(col):
                coerced = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce')
                df[col] = coerced
                amount_cols.append(col)
        except Exception:
            continue

    # Heuristic: if no obvious amount column found, try to coerce any object column that looks numeric
    if not amount_cols:
        for col in df.columns:
            try:
                if df[col].dtype == object:
                    sample = df[col].astype(str).head(20).str.replace(r'[^\d\.\-]', '', regex=True)
                    parsed = pd.to_numeric(sample, errors='coerce')
                    if parsed.notna().sum() >= 3:
                        coerced = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce')
                        df[col] = coerced
                        amount_cols.append(col)
                        break
            except Exception:
                continue

    # Choose preferred amount column name (case-insensitive matching)
    preferred = None
    candidates = ['amount','total_spent','totalspent','total','txn amount','value','spent','amt']
    for candidate in candidates:
        for col in df.columns:
            if str(col).lower() == candidate:
                preferred = col
                break
        if preferred:
            break
    if not preferred and amount_cols:
        preferred = amount_cols[0]

    if preferred:
        # create canonical 'Amount' column (unless it already exists and is numeric)
        if preferred != 'Amount':
            if 'Amount' not in df.columns:
                try:
                    df.rename(columns={preferred: 'Amount'}, inplace=True)
                except Exception:
                    df['Amount'] = pd.to_numeric(df[preferred], errors='coerce')
        else:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    else:
        if 'Amount' not in df.columns:
            df['Amount'] = pd.NA
        else:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

    # 4) Create canonical timestamp and date columns (prefer primary_dt_col)
    try:
        if primary_dt_col:
            # parse preserving time component
            df['timestamp'] = pd.to_datetime(df[primary_dt_col], errors='coerce')
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            df['timestamp'] = pd.NaT
    except Exception:
        df['timestamp'] = pd.NaT

    # date (date part) - keep as date objects (not datetime) to ease grouping
    try:
        df['date'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.date
    except Exception:
        df['date'] = pd.NA

    # 5) Infer or normalize Type (case-insensitive detection of existing 'type' column)
    existing_type_col = next((c for c in df.columns if str(c).lower() == 'type'), None)

    if existing_type_col is None:
        try:
            df['Type'] = pd.NA
            mask_pos = df['Amount'].notna() & (df['Amount'] > 0)
            mask_neg = df['Amount'].notna() & (df['Amount'] < 0)
            df.loc[mask_pos, 'Type'] = 'debit'
            df.loc[mask_neg, 'Type'] = 'credit'
        except Exception:
            df['Type'] = pd.NA
    else:
        try:
            df['Type'] = df[existing_type_col].astype(str).str.lower().str.strip()
        except Exception:
            df['Type'] = df[existing_type_col].astype(str)

    # Normalize Type column to canonical lower-case strings and mark empties as 'unknown'
    try:
        df['Type'] = df['Type'].astype(str).str.lower().str.strip()
        df['Type'] = df['Type'].replace({'nan': 'unknown', 'none': 'unknown', '': 'unknown', 'na': 'unknown', 'null': 'unknown'})
    except Exception:
        df['Type'] = 'unknown'

    # DROP rows where Type is unknown (preserve other columns)
    try:
        df = df[df['Type'] != 'unknown'].copy().reset_index(drop=True)
    except Exception:
        pass

    # final reorder: put timestamp and date at front but preserve other columns (including _sheet_row_idx, is_deleted etc.)
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
            try:
                if pd.api.types.is_datetime64_any_dtype(w[c]) and w[c].notna().any():
                    found = c
                    break
            except Exception:
                continue
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

# ---------------------------
# New: Outlier detection & replacement helpers
# ---------------------------

def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers in a numeric pandas Series using the IQR rule.
    Returns a boolean Series (True where value is outlier).
    Non-numeric or NA values are treated as non-outliers (False).
    """
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty:
        # return same-index False series
        return pd.Series(False, index=series.index)
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    # For initial mask, consider original index; NA values are non-outliers
    mask = pd.Series(False, index=series.index)
    numeric = pd.to_numeric(series, errors='coerce')
    mask |= (numeric < lower) & numeric.notna()
    mask |= (numeric > upper) & numeric.notna()
    return mask

def detect_outliers_mad(series: pd.Series, thresh: float = 3.5) -> pd.Series:
    """
    Detect outliers using Median Absolute Deviation (MAD).
    Returns boolean Series where True indicates an outlier.
    Works robustly with skewed distributions.
    """
    numeric = pd.to_numeric(series, errors='coerce')
    med = numeric.median()
    mad = (numeric - med).abs().median()
    if pd.isna(mad) or mad == 0:
        # fallback: no MAD information -> no outliers
        return pd.Series(False, index=series.index)
    normalized = ((numeric - med).abs()) / mad
    mask = normalized > thresh
    mask = mask.fillna(False)
    return mask.astype(bool)

def replace_outliers(
    series: pd.Series,
    detect: str = 'iqr',
    iqr_multiplier: float = 1.5,
    mad_thresh: float = 3.5,
    method: str = 'median',
    min_count: int = 3
) -> Tuple[pd.Series, Dict]:
    """
    Return a copy of the series where detected outliers are replaced according to `method`.

    Parameters:
      - series: numeric-ish pandas Series (index preserved)
      - detect: 'iqr' or 'mad' (outlier detection method)
      - iqr_multiplier: multiplier used for IQR detection
      - mad_thresh: threshold for MAD detection
      - method: replacement statistic: 'median' (default), 'mean', 'trimmed_mean'
      - min_count: minimum non-NA count required to attempt replacement; if less, returns original series

    Returns:
      - replaced_series: pandas Series (float dtype) with outliers replaced
      - info: dict with keys: n (count), outliers_detected, replacement_value, method, detect, iqr, lower, upper, mad, notes
    """
    numeric = pd.to_numeric(series, errors='coerce')
    s = numeric.dropna()
    info: Dict = {"n": int(len(s)), "outliers_detected": 0, "replacement_value": None, "method": method, "detect": detect, "iqr": None, "lower": None, "upper": None, "mad": None, "notes": ""}

    # If not enough points, do not replace
    if len(s) < min_count:
        info["notes"] = "insufficient_data_for_replacement"
        return numeric.astype('float64'), info

    # detect outliers
    if detect == 'mad':
        mask = detect_outliers_mad(numeric, thresh=mad_thresh)
        info["mad"] = float(((numeric - numeric.median()).abs().median())) if not pd.isna(numeric.median()) else None
    else:
        mask = detect_outliers_iqr(numeric, multiplier=iqr_multiplier)
        # record iqr bounds if possible
        s_nonnull = numeric.dropna()
        q1 = s_nonnull.quantile(0.25)
        q3 = s_nonnull.quantile(0.75)
        iqr = q3 - q1
        info["iqr"] = float(iqr) if not pd.isna(iqr) else None
        info["lower"] = float(q1 - iqr * iqr_multiplier) if not pd.isna(q1) and not pd.isna(iqr) else None
        info["upper"] = float(q3 + iqr * iqr_multiplier) if not pd.isna(q3) and not pd.isna(iqr) else None

    outlier_count = int(mask.sum())
    info["outliers_detected"] = outlier_count

    if outlier_count == 0:
        info["notes"] = "no_outliers_found"
        return numeric.astype('float64'), info

    # compute replacement value from non-outlier values
    non_outliers = numeric[~mask].dropna()
    if non_outliers.empty:
        # fallback to median of all values
        replacement = float(numeric.median())
        info["notes"] = "all_values_flagged_as_outliers_fallback_to_median"
    else:
        if method == 'median':
            replacement = float(non_outliers.median())
        elif method == 'mean':
            replacement = float(non_outliers.mean())
        elif method == 'trimmed_mean':
            # trimmed mean (remove top & bottom 10% of non_outliers)
            n = len(non_outliers)
            trim = max(0, int(n * 0.1))
            arr = np.sort(non_outliers.to_numpy())
            if n - 2 * trim <= 0:
                replacement = float(non_outliers.median())
            else:
                trimmed = arr[trim: n - trim]
                replacement = float(trimmed.mean())
        else:
            # default fallback
            replacement = float(non_outliers.median())
            info["notes"] = f"unknown_method_{method}_fallback_median"

    info["replacement_value"] = float(replacement)

    # build replaced series
    replaced = numeric.copy().astype('float64')
    replaced[mask] = replacement

    return replaced, info

# ---------------------------
# New: Monthly average helpers that operate on 'merged' output from compute_daily_totals
# ---------------------------

def compute_monthly_average(
    merged_df: pd.DataFrame,
    year: int,
    month: int,
    replace_outliers: bool = False,
    detect: str = 'iqr',
    iqr_multiplier: float = 1.5,
    mad_thresh: float = 3.5,
    replacement_method: str = 'median',
    min_count_for_replacement: int = 3
) -> Tuple[Optional[float], int, Dict]:
    """
    Compute the average daily Total_Spent for a specific year and month from the 'merged' DataFrame
    (output of compute_daily_totals).

    Parameters:
      - merged_df: DataFrame with 'Date' column and 'Total_Spent' column (daily totals)
      - year, month: target month
      - replace_outliers: if True, replace detected outlier *days* before computing average
      - detect, iqr_multiplier, mad_thresh, replacement_method: forwarded to detect/replace helpers
      - min_count_for_replacement: minimum number of days required to attempt replacement

    Returns:
      - avg_value (float) or None if no data
      - count_days (int)
      - info dict with details (outliers_replaced, n, detect, replacement_value, etc.)
    """
    info: Dict = {"n": 0, "outliers_replaced": 0, "detect": detect, "iqr_multiplier": iqr_multiplier, "replacement_method": replacement_method}
    if merged_df is None or merged_df.empty:
        info["reason"] = "no_data"
        return None, 0, info

    df = merged_df.copy()
    if 'Date' not in df.columns:
        info["reason"] = "no_date_col"
        return None, 0, info

    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    mask = (df['Date'].dt.year == int(year)) & (df['Date'].dt.month == int(month))
    dfm = df.loc[mask].copy()
    if dfm.empty:
        info["reason"] = "no_rows_for_month"
        return None, 0, info

    vals = pd.to_numeric(dfm.get('Total_Spent', 0), errors='coerce').fillna(0.0)
    info["n"] = int(len(vals))

    if len(vals) == 0:
        info["reason"] = "no_numeric_values"
        return None, 0, info

    if not replace_outliers or len(vals) < min_count_for_replacement:
        # simple mean
        avg = float(vals.mean()) if len(vals) > 0 else None
        info["outliers_replaced"] = 0
        info["avg_raw"] = float(vals.mean()) if len(vals) > 0 else None
        return (avg, int(len(vals),), info) if False else (avg, int(len(vals)), info)

    # replace outliers
    replaced_series, replace_info = replace_outliers(
        vals,
        detect=detect,
        iqr_multiplier=iqr_multiplier,
        mad_thresh=mad_thresh,
        method=replacement_method,
        min_count=min_count_for_replacement
    )
    info.update(replace_info)
    info["outliers_replaced"] = int(replace_info.get("outliers_detected", 0))
    avg_replaced = float(pd.to_numeric(replaced_series, errors='coerce').dropna().mean()) if not replaced_series.dropna().empty else None
    info["avg_replaced"] = avg_replaced
    return avg_replaced, int(len(replaced_series.dropna())), info

def compute_monthly_average_with_prev(
    merged_df: pd.DataFrame,
    year: int,
    month: int,
    replace_outliers: bool = False,
    detect: str = 'iqr',
    iqr_multiplier: float = 1.5,
    mad_thresh: float = 3.5,
    replacement_method: str = 'median',
    min_count_for_replacement: int = 3
) -> Dict:
    """
    Convenience function: compute the average for (year, month) and for previous month
    Returns a dict with keys:
      - 'year','month'
      - 'avg','count','info' for current month (under 'current')
      - 'avg','count','info' for previous month (under 'previous')
    """
    # compute previous month
    try:
        dt = pd.Timestamp(year=int(year), month=int(month), day=1)
        prev_dt = dt - pd.DateOffset(months=1)
        prev_year = int(prev_dt.year)
        prev_month = int(prev_dt.month)
    except Exception:
        # fallback to simple arithmetic
        prev_month = month - 1
        prev_year = year
        if prev_month == 0:
            prev_month = 12
            prev_year = year - 1

    cur_avg, cur_count, cur_info = compute_monthly_average(
        merged_df, year, month,
        replace_outliers=replace_outliers,
        detect=detect,
        iqr_multiplier=iqr_multiplier,
        mad_thresh=mad_thresh,
        replacement_method=replacement_method,
        min_count_for_replacement=min_count_for_replacement
    )

    prev_avg, prev_count, prev_info = compute_monthly_average(
        merged_df, prev_year, prev_month,
        replace_outliers=replace_outliers,
        detect=detect,
        iqr_multiplier=iqr_multiplier,
        mad_thresh=mad_thresh,
        replacement_method=replacement_method,
        min_count_for_replacement=min_count_for_replacement
    )

    return {
        "year": int(year),
        "month": int(month),
        "current": {"avg": cur_avg, "count": cur_count, "info": cur_info},
        "previous": {"avg": prev_avg, "count": prev_count, "info": prev_info}
    }

# End of file
