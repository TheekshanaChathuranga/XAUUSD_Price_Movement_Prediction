"""
STEP 1: XAU/USD Market Price Data Collection
============================================
Collects OHLC + Volume data for Gold (XAU/USD) from Yahoo Finance.
Output: xauusd_raw_prices.csv
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, date

# ─── CONFIG ──────────────────────────────────────────────────────────────────
START_DATE  = "2015-01-01"
END_DATE    = date.today().strftime("%Y-%m-%d")
OUTPUT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "xauusd_raw_prices.csv")

# Yahoo Finance ticker for Gold Spot USD
TICKER = "GC=F"   # Gold Futures (most liquid proxy for XAU/USD spot)
BACKUP_TICKER = "GLD"  # Gold ETF fallback

# ─── COLLECTION ──────────────────────────────────────────────────────────────
def fetch_xauusd(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance and standardise columns."""
    print(f"[INFO] Downloading {ticker} from {start} to {end} (interval={interval}) ...")
    raw = yf.download(ticker, start=start, end=end, interval=interval,
                      auto_adjust=True, progress=True)

    if raw.empty:
        raise ValueError(f"[ERROR] No data returned for ticker: {ticker}")

    # Flatten MultiIndex columns if present (yfinance ≥ 0.2.x)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Standardise
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index).normalize()          # strip time → YYYY-MM-DD
    df = df[~df.index.duplicated(keep="last")]                # remove duplicate dates
    df = df.sort_index()

    # Rename Volume to Tick_Volume for consistency with MT5 convention
    df.rename(columns={"Volume": "Tick_Volume"}, inplace=True)

    # Fill tiny gaps (weekends, holidays) by forward-fill (max 3 days)
    df = df.asfreq("B").ffill(limit=3)                        # business-day frequency
    df.index = pd.to_datetime(df.index)                       # ensure dtype is datetime64[ns]

    print(f"[OK]   Shape after cleaning: {df.shape}")
    print(f"       Date range : {df.index.min().date()} → {df.index.max().date()}")
    print(f"       Null values:\n{df.isnull().sum()}\n")
    return df

def validate(df: pd.DataFrame) -> None:
    """Basic sanity checks before saving."""
    assert df.index.is_monotonic_increasing, "Index not sorted!"
    assert df["Close"].gt(0).all(),          "Zero/negative Close prices detected!"
    assert pd.api.types.is_datetime64_any_dtype(df.index), "Date index is not datetime type!"
    print("[PASS] All validation checks passed.")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        df = fetch_xauusd(TICKER, START_DATE, END_DATE)
    except Exception as e:
        print(f"[WARN] Primary ticker failed ({e}). Trying backup: {BACKUP_TICKER}")
        df = fetch_xauusd(BACKUP_TICKER, START_DATE, END_DATE)

    validate(df)

    # Reset index so Date becomes a column
    df.reset_index(inplace=True)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")           # enforce YYYY-MM-DD string

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[SAVED] {OUTPUT_FILE}")
    print(df.head(5).to_string(index=False))
    print("  ...")
    print(df.tail(3).to_string(index=False))
    print(f"\nTotal rows: {len(df):,}")
