# transform.py  -- pure data transformation utilities
import pandas as pd
from typing import Optional, Tuple, Dict
import numpy as np


# ---------------------------------------------------------------------------
# Column-type detection helpers
# ---------------------------------------------------------------------------

def _is_date_like_column(col_name: str) -> bool:
    lname = str(col_name).lower()
    return any(k in lname for k in ['date', 'time', 'timestamp', 'datetime', 'txn']) or lname.startswith("unnamed")


def _is_amount_like_column(col_name: str) -> bool:
    lname = str(col_name).lower()
    return any(k in lname for k in ['amount', 'amt', 'value', 'total', 'balance', 'credit', 'debit', 'spent'])


# ---------------------------------------------------------------------------
# Soft-delete helpers
# ---------------------------------------------------------------------------

def _detect_is_deleted_mask(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return boolean mask (True = deleted) if is_deleted column exists, else None."""
    isdel_col = next((c for c in df.columns if str(c).lower() == 'is_deleted'), None)
    if isdel_col is None:
        return None
    s = df[isdel_col]
    try:
        if pd.api.types.is_bool_dtype(s):
            return s.fillna(False).astype(bool)
        if pd.api.types.is_numeric_dtype(s):
            return s.fillna(0).astype(int) == 1
        return s.astype(str).str.strip().str.lower().fillna('').isin(['true', 't', '1', 'yes', 'y'])
    except Exception:
        try:
            return s.astype(str).str.strip().str.lower().fillna('').isin(['true', 't', '1', 'yes', 'y'])
        except Exception:
            return None


def _prefer_datetime_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if str(c).lower() in ('datetime', 'date_time'):
            return c
    for c in df.columns:
        if 'datetime' in str(c).lower():
            return c
    return None


# ---------------------------------------------------------------------------
# Core transform
# ---------------------------------------------------------------------------

def convert_columns_and_derives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns:
      - remove soft-deleted rows
      - detect & coerce date/time -> timestamp
      - detect & coerce numeric -> Amount
      - create 'date' (date part of timestamp)
      - infer Type (debit/credit) when missing
    Preserves _sheet_row_idx, _source_sheet, and other extra columns.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # 0) Filter soft-deleted rows
    try:
        isdel_mask = _detect_is_deleted_mask(df)
        if isdel_mask is not None:
            df = df.loc[~isdel_mask].copy().reset_index(drop=True)
    except Exception:
        pass

    # 1) Detect primary datetime column
    primary_dt_col = _prefer_datetime_column(df)
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

    # 2) Detect & coerce amount-like columns
    amount_cols = []
    for col in df.columns:
        try:
            if _is_amount_like_column(col):
                coerced = pd.to_numeric(
                    df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce'
                )
                df[col] = coerced
                amount_cols.append(col)
        except Exception:
            continue

    if not amount_cols:
        for col in df.columns:
            try:
                if df[col].dtype == object:
                    sample = df[col].astype(str).head(20).str.replace(r'[^\d\.\-]', '', regex=True)
                    if pd.to_numeric(sample, errors='coerce').notna().sum() >= 3:
                        coerced = pd.to_numeric(
                            df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce'
                        )
                        df[col] = coerced
                        amount_cols.append(col)
                        break
            except Exception:
                continue

    # Pick preferred Amount column
    preferred = None
    for candidate in ['amount', 'total_spent', 'totalspent', 'total', 'txn amount', 'value', 'spent', 'amt']:
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
            else:
                df['Amount'] = pd.to_numeric(df[preferred], errors='coerce')
        else:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    else:
        if 'Amount' not in df.columns:
            df['Amount'] = pd.NA
        else:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

    # 3) Canonical timestamp + date columns
    try:
        if primary_dt_col:
            df['timestamp'] = pd.to_datetime(df[primary_dt_col], errors='coerce')
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            df['timestamp'] = pd.NaT
    except Exception:
        df['timestamp'] = pd.NaT

    try:
        df['date'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.date
    except Exception:
        df['date'] = pd.NA

    # 4) Normalize / infer Type column
    existing_type_col = next((c for c in df.columns if str(c).lower() == 'type'), None)
    if existing_type_col is None:
        try:
            df['Type'] = pd.NA
            df.loc[df['Amount'].notna() & (df['Amount'] > 0), 'Type'] = 'debit'
            df.loc[df['Amount'].notna() & (df['Amount'] < 0), 'Type'] = 'credit'
        except Exception:
            df['Type'] = pd.NA
    else:
        df['Type'] = df[existing_type_col].astype(str).str.lower().str.strip()

    df['Type'] = df['Type'].astype(str).str.lower().str.strip()
    df['Type'] = df['Type'].replace({'nan': 'unknown', 'none': 'unknown', '': 'unknown', 'na': 'unknown', 'null': 'unknown'})
    df = df[df['Type'] != 'unknown'].copy().reset_index(drop=True)

    # 5) Reorder: timestamp & date first, preserve all other columns
    cols = list(df.columns)
    final = [c for c in ['timestamp', 'date'] if c in cols]
    final += [c for c in cols if c not in final]
    return df[final]


# ---------------------------------------------------------------------------
# Daily totals
# ---------------------------------------------------------------------------

def compute_daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transactions into daily totals.
    Returns DataFrame with: Date, Total_Spent, Total_Credit
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['Date', 'Total_Spent', 'Total_Credit'])

    w = df.copy()

    if 'date' in w.columns and w['date'].notna().any():
        grp = pd.to_datetime(w['date']).dt.normalize()
    elif 'timestamp' in w.columns and w['timestamp'].notna().any():
        grp = pd.to_datetime(w['timestamp']).dt.normalize()
    else:
        found = next(
            (c for c in w.columns if pd.api.types.is_datetime64_any_dtype(w[c]) and w[c].notna().any()), None
        )
        grp = pd.to_datetime(w[found]).dt.normalize() if found else None
        if grp is None:
            return pd.DataFrame(columns=['Date', 'Total_Spent', 'Total_Credit'])

    w['_group_date'] = grp
    w['Amount_numeric'] = pd.to_numeric(w.get('Amount', 0), errors='coerce').fillna(0.0).astype('float64')

    if 'Type' in w.columns and w['Type'].astype(str).str.strip().any():
        w['Type_norm'] = w['Type'].astype(str).str.lower().str.strip()
        debit_df = w[w['Type_norm'] == 'debit']
        credit_df = w[w['Type_norm'] == 'credit']
    else:
        debit_df = w[w['Amount_numeric'] > 0]
        credit_df = w[w['Amount_numeric'] < 0]

    daily_spend = (
        debit_df.groupby('_group_date')['Amount_numeric'].sum()
        .reset_index().rename(columns={'_group_date': 'Date', 'Amount_numeric': 'Total_Spent'})
    )
    daily_credit = (
        credit_df.groupby('_group_date')['Amount_numeric'].sum()
        .reset_index().rename(columns={'_group_date': 'Date', 'Amount_numeric': 'Total_Credit'})
    )
    if 'Total_Credit' in daily_credit.columns and 'Type' not in w.columns:
        daily_credit['Total_Credit'] = daily_credit['Total_Credit'].abs()

    merged = pd.merge(daily_spend, daily_credit, on='Date', how='outer').fillna(0)
    merged['Date'] = pd.to_datetime(merged['Date']).dt.normalize()
    merged['Total_Spent'] = merged['Total_Spent'].astype('float64')
    merged['Total_Credit'] = merged['Total_Credit'].astype('float64')
    return merged.sort_values('Date').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Outlier detection & replacement (FIXED: renamed to avoid self-shadowing)
# ---------------------------------------------------------------------------

def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Detect outliers using IQR rule. Returns boolean Series (True = outlier)."""
    numeric = pd.to_numeric(series, errors='coerce')
    s = numeric.dropna()
    if s.empty:
        return pd.Series(False, index=series.index)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    mask = pd.Series(False, index=series.index)
    mask |= (numeric < lower) & numeric.notna()
    mask |= (numeric > upper) & numeric.notna()
    return mask


def detect_outliers_mad(series: pd.Series, thresh: float = 3.5) -> pd.Series:
    """Detect outliers using Median Absolute Deviation. Returns boolean Series."""
    numeric = pd.to_numeric(series, errors='coerce')
    med = numeric.median()
    mad = (numeric - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return pd.Series(False, index=series.index)
    return (((numeric - med).abs()) / mad > thresh).fillna(False).astype(bool)


def replace_outliers_in_series(
    series: pd.Series,
    detect: str = 'iqr',
    iqr_multiplier: float = 1.5,
    mad_thresh: float = 3.5,
    method: str = 'median',
    min_count: int = 3
) -> Tuple[pd.Series, Dict]:
    """
    Replace detected outliers in a Series with a robust statistic.
    RENAMED from replace_outliers to avoid function-shadowing bug.
    """
    numeric = pd.to_numeric(series, errors='coerce')
    s = numeric.dropna()
    info: Dict = {
        "n": int(len(s)), "outliers_detected": 0, "replacement_value": None,
        "method": method, "detect": detect, "iqr": None, "lower": None,
        "upper": None, "mad": None, "notes": ""
    }

    if len(s) < min_count:
        info["notes"] = "insufficient_data_for_replacement"
        return numeric.astype('float64'), info

    if detect == 'mad':
        mask = detect_outliers_mad(numeric, thresh=mad_thresh)
        info["mad"] = float((numeric - numeric.median()).abs().median()) if not pd.isna(numeric.median()) else None
    else:
        mask = detect_outliers_iqr(numeric, multiplier=iqr_multiplier)
        s_nn = numeric.dropna()
        q1, q3 = s_nn.quantile(0.25), s_nn.quantile(0.75)
        iqr = q3 - q1
        info.update({
            "iqr": float(iqr) if not pd.isna(iqr) else None,
            "lower": float(q1 - iqr * iqr_multiplier) if not pd.isna(q1) else None,
            "upper": float(q3 + iqr * iqr_multiplier) if not pd.isna(q3) else None,
        })

    info["outliers_detected"] = int(mask.sum())
    if info["outliers_detected"] == 0:
        info["notes"] = "no_outliers_found"
        return numeric.astype('float64'), info

    non_outliers = numeric[~mask].dropna()
    if non_outliers.empty:
        replacement = float(numeric.median())
        info["notes"] = "all_values_flagged_fallback_to_median"
    elif method == 'mean':
        replacement = float(non_outliers.mean())
    elif method == 'trimmed_mean':
        n = len(non_outliers)
        trim = max(0, int(n * 0.1))
        arr = np.sort(non_outliers.to_numpy())
        replacement = float(arr[trim: n - trim].mean()) if n - 2 * trim > 0 else float(non_outliers.median())
    else:  # median (default)
        replacement = float(non_outliers.median())

    info["replacement_value"] = float(replacement)
    replaced = numeric.copy().astype('float64')
    replaced[mask] = replacement
    return replaced, info


# Keep the old name as an alias for backward compatibility
replace_outliers = replace_outliers_in_series


# ---------------------------------------------------------------------------
# Monthly average helpers
# ---------------------------------------------------------------------------

def compute_monthly_average(
    merged_df: pd.DataFrame,
    year: int,
    month: int,
    use_outlier_replacement: bool = False,
    detect: str = 'iqr',
    iqr_multiplier: float = 1.5,
    mad_thresh: float = 3.5,
    replacement_method: str = 'median',
    min_count_for_replacement: int = 3
) -> Tuple[Optional[float], int, Dict]:
    """Average daily Total_Spent for a given year/month. Returns (avg, count, info)."""
    info: Dict = {"n": 0, "outliers_replaced": 0}
    if merged_df is None or merged_df.empty or 'Date' not in merged_df.columns:
        info["reason"] = "no_data"
        return None, 0, info

    df = merged_df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    mask = (df['Date'].dt.year == int(year)) & (df['Date'].dt.month == int(month))
    dfm = df.loc[mask]
    if dfm.empty:
        return None, 0, {**info, "reason": "no_rows_for_month"}

    vals = pd.to_numeric(dfm.get('Total_Spent', 0), errors='coerce').fillna(0.0)
    info["n"] = int(len(vals))

    if not use_outlier_replacement or len(vals) < min_count_for_replacement:
        return float(vals.mean()), int(len(vals)), info

    replaced_series, replace_info = replace_outliers_in_series(
        vals, detect=detect, iqr_multiplier=iqr_multiplier, mad_thresh=mad_thresh,
        method=replacement_method, min_count=min_count_for_replacement
    )
    info.update(replace_info)
    info["outliers_replaced"] = int(replace_info.get("outliers_detected", 0))
    avg = float(replaced_series.dropna().mean()) if not replaced_series.dropna().empty else None
    return avg, int(len(replaced_series.dropna())), info


def compute_monthly_average_with_prev(
    merged_df: pd.DataFrame,
    year: int,
    month: int,
    use_outlier_replacement: bool = False,
    **kwargs
) -> Dict:
    """Compute average for (year, month) and the prior month."""
    try:
        dt = pd.Timestamp(year=int(year), month=int(month), day=1)
        prev_dt = dt - pd.DateOffset(months=1)
        prev_year, prev_month = int(prev_dt.year), int(prev_dt.month)
    except Exception:
        prev_month = month - 1 or 12
        prev_year = year if month > 1 else year - 1

    cur = compute_monthly_average(merged_df, year, month, use_outlier_replacement=use_outlier_replacement, **kwargs)
    prev = compute_monthly_average(merged_df, prev_year, prev_month, use_outlier_replacement=use_outlier_replacement, **kwargs)
    return {
        "year": int(year), "month": int(month),
        "current": {"avg": cur[0], "count": cur[1], "info": cur[2]},
        "previous": {"avg": prev[0], "count": prev[1], "info": prev[2]},
    }


# ---------------------------------------------------------------------------
# Large transaction flag helper  (NEW)
# ---------------------------------------------------------------------------

def flag_large_transactions(df: pd.DataFrame, threshold: float = 500.0) -> pd.DataFrame:
    """
    Return debit rows where Amount > threshold.
    Adds '_flag_reason' column for display.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    w = df.copy()
    w['Amount'] = pd.to_numeric(w.get('Amount'), errors='coerce')

    type_col = next((c for c in w.columns if c.lower() == 'type'), None)
    debit_mask = (
        w[type_col].astype(str).str.lower().str.strip() == 'debit'
        if type_col else pd.Series(True, index=w.index)
    )

    flagged = w[debit_mask & (w['Amount'] > threshold)].copy()
    flagged['_flag_reason'] = flagged['Amount'].apply(
        lambda a: f"₹{a:,.0f} exceeds ₹{threshold:,.0f} threshold"
    )
    return flagged
