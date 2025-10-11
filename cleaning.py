# src/cleaning.py
"""
Cleaning utilities for History Transactions sheet.
Expose: clean_history_transactions(df) -> cleaned_df
"""

from __future__ import annotations
import re
from typing import Optional
import pandas as pd
import numpy as np

def parse_datetime(x) -> pd.Timestamp:
    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        try:
            return pd.to_datetime(str(x).strip(), dayfirst=False, errors="coerce")
        except Exception:
            return pd.NaT

def parse_amount(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x)
    # remove common currency words/symbols and spaces
    s = re.sub(r"[₹,]|INR", "", s, flags=re.IGNORECASE).strip()
    s = s.replace("(", "-").replace(")", "")
    m = re.search(r"-?[0-9]+(?:\.[0-9]+)?", s)
    return float(m.group(0)) if m else np.nan

def to_bool(x) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y", "t", "suspicious"}

def normalize_type(x) -> str:
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip().lower()
    if "debit" in s or "dr" in s or "sent" in s or "paid" in s:
        return "Debit"
    if "credit" in s or "cr" in s or "received" in s:
        return "Credit"
    # common UPI/card hints
    if "upi" in s:
        return "Debit"
    if "salary" in s:
        return "Credit"
    return str(x).strip().title()

def normalize_bank(x) -> str:
    if pd.isna(x):
        return "UNKNOWN"
    s = str(x).strip().upper()
    # small mapping examples; expand as needed
    s = s.replace("HDFCBK", "HDFC").replace("HDFC BANK", "HDFC")
    s = s.replace("SBI ", "SBI").strip()
    return s

def clean_history_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize a DataFrame that represents History Transactions.
    Returns a cleaned DataFrame with helper columns: Date, Month, Hour.
    """
    df = df.copy()

    # normalize column names (strip)
    df.columns = [str(c).strip() for c in df.columns]

    # Parse DateTime
    if "DateTime" in df.columns:
        df["DateTime"] = df["DateTime"].apply(parse_datetime)
    else:
        # try common alternatives
        for cand in ("Date", "Timestamp", "date_time"):
            if cand in df.columns:
                df["DateTime"] = df[cand].apply(parse_datetime)
                break
        else:
            df["DateTime"] = pd.NaT

    # Parse Amount
    if "Amount" in df.columns:
        df["Amount"] = df["Amount"].apply(parse_amount)
    else:
        # attempt to extract amount from Message if present
        if "Message" in df.columns:
            df["Amount"] = df["Message"].astype(str).apply(lambda s: parse_amount(re.search(r"([₹]?[0-9\.,() -]+)", s).group(0)) if re.search(r"([₹]?[0-9\.,() -]+)", s) else np.nan)
        else:
            df["Amount"] = np.nan

    # Normalize Type
    if "Type" in df.columns:
        df["Type"] = df["Type"].apply(normalize_type).astype("category")
    else:
        df["Type"] = "Unknown"

    # Normalize Bank
    if "Bank" in df.columns:
        df["Bank"] = df["Bank"].apply(normalize_bank).astype("category")
    else:
        df["Bank"] = "UNKNOWN"

    # Sender
    if "Sender" in df.columns:
        df["Sender"] = df["Sender"].fillna("").astype(str).str.strip()
    else:
        df["Sender"] = ""

    # Suspicious flag
    if "Suspicious" in df.columns:
        df["Suspicious"] = df["Suspicious"].apply(to_bool)
    else:
        df["Suspicious"] = False

    # Remove obvious noise messages (OTP, BALANCE, FAILED, DECLINED)
    if "Message" in df.columns:
        noise_mask = df["Message"].astype(str).str.contains(r"\b(OTP|BALANCE|FAILED|DECLINED|DECLINE)\b", case=False, na=False)
        if noise_mask.any():
            df = df[~noise_mask].reset_index(drop=True)

    # Drop duplicates using MessageHash if available; else DateTime+Amount+Sender
    if "MessageHash" in df.columns:
        df = df.drop_duplicates(subset=["MessageHash"])
    else:
        subset = [c for c in ("DateTime", "Amount", "Sender") if c in df.columns]
        if subset:
            df = df.drop_duplicates(subset=subset, keep="first")

    # Helper columns
    df["Date"] = pd.to_datetime(df["DateTime"]).dt.date
    df["Month"] = pd.to_datetime(df["DateTime"]).dt.to_period("M").astype(str)
    df["Hour"] = pd.to_datetime(df["DateTime"]).dt.hour

    # Force types
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    if not pd.api.types.is_categorical_dtype(df["Bank"]):
        df["Bank"] = df["Bank"].astype("category")
    if not pd.api.types.is_categorical_dtype(df["Type"]):
        df["Type"] = df["Type"].astype("category")

    # Reorder columns to put cleaned helper cols near front
    cols = list(df.columns)
    for c in ("DateTime", "Date", "Month", "Hour", "Amount", "Type", "Bank", "Sender", "Suspicious"):
        if c in cols:
            cols.insert(0, cols.pop(cols.index(c)))
    # keep unique preserving order
    seen = set()
    ordered = [x for x in cols if not (x in seen or seen.add(x))]
    df = df[ordered]

    return df
