"""
STEP 2: Macroeconomic Data Collection via FRED API
===================================================
Collects: CPI, Federal Funds Rate, Unemployment Rate, NFP,
          US Dollar Index (DXY proxy), WTI Crude Oil.

HOW TO GET YOUR FREE FRED API KEY
----------------------------------
1. Visit https://fred.stlouisfed.org/docs/api/api_key.html
2. Create a free account → click "Request API Key"
3. Copy the 32-character key into FRED_API_KEY below (or set as env var).

Output: fred_macro_raw.csv
"""

import os
import time
import pandas as pd
import requests
from datetime import date

# ─── CONFIG ──────────────────────────────────────────────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY", "YOUR_FRED_API_KEY_HERE")
START_DATE   = "2015-01-01"
END_DATE     = date.today().strftime("%Y-%m-%d")
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE  = os.path.join(OUTPUT_DIR, "fred_macro_raw.csv")

# Series definitions: {column_name: (fred_series_id, human_description)}
FRED_SERIES = {
    "CPI_US"          : ("CPIAUCSL",  "Consumer Price Index – All Urban Consumers (SA)"),
    "FedFunds_Rate"   : ("FEDFUNDS",  "Effective Federal Funds Rate (%)"),
    "Unemployment_Rate": ("UNRATE",   "US Unemployment Rate (%)"),
    "NFP_Change"      : ("PAYEMS",    "Nonfarm Payrolls Total (Thousands)"),
    "WTI_Crude_Oil"   : ("DCOILWTICO","WTI Crude Oil Price (USD/bbl, daily)"),
    "PCE_Deflator"    : ("PCEPI",     "PCE Price Index (Fed preferred inflation gauge)"),
    "US_10Y_Yield"    : ("DGS10",     "10-Year Treasury Constant Maturity Rate (%)"),
    "Real_GDP_Growth" : ("A191RL1Q225SBEA", "Real GDP Growth Rate QoQ (%)"),
    "M2_Money_Supply" : ("M2SL",      "M2 Money Stock (Billions USD, SA)"),
}

# ─── FRED API HELPER ─────────────────────────────────────────────────────────
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

def fetch_fred_series(series_id: str, name: str, start: str, end: str) -> pd.Series:
    """Fetch a single FRED series and return as a named pd.Series indexed by date."""
    params = {
        "series_id"        : series_id,
        "api_key"          : FRED_API_KEY,
        "file_type"        : "json",
        "observation_start": start,
        "observation_end"  : end,
        "units"            : "lin",      # raw level values
        "sort_order"       : "asc",
    }
    resp = requests.get(FRED_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "observations" not in data:
        raise ValueError(f"FRED returned no observations for {series_id}: {data}")

    rows = [(obs["date"], obs["value"]) for obs in data["observations"]
            if obs["value"] != "."]          # "." = missing in FRED

    s = pd.Series(
        data  = [float(v) for _, v in rows],
        index = pd.to_datetime([d for d, _ in rows]),
        name  = name,
    )
    s.index.name = "Date"
    print(f"  [{series_id}] {name}: {len(s)} observations  "
          f"({s.index.min().date()} → {s.index.max().date()})")
    return s

# ─── DXY FALLBACK (Yahoo Finance) ────────────────────────────────────────────
def fetch_dxy_yahoo(start: str, end: str) -> pd.Series:
    """DXY is not on FRED; pull from Yahoo Finance as fallback."""
    try:
        import yfinance as yf
        raw = yf.download("DX-Y.NYB", start=start, end=end,
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        s = raw["Close"].rename("DXY_Index")
        s.index = pd.to_datetime(s.index).normalize()
        s.index.name = "Date"
        print(f"  [DX-Y.NYB] DXY_Index (Yahoo): {len(s)} observations")
        return s
    except Exception as e:
        print(f"  [WARN] DXY via Yahoo failed: {e}")
        return pd.Series(name="DXY_Index", dtype=float)

# ─── MERGE STRATEGY ──────────────────────────────────────────────────────────
def merge_to_daily(series_dict: dict, start: str, end: str) -> pd.DataFrame:
    """
    Merge all series onto a daily business-day index.
    Monthly/quarterly series are forward-filled across trading days
    (standard practice in financial ML: use the last known reading).
    """
    bday_idx = pd.bdate_range(start=start, end=end, name="Date")
    df = pd.DataFrame(index=bday_idx)

    for name, s in series_dict.items():
        if s.empty:
            print(f"  [SKIP] {name} is empty, skipping merge.")
            continue
        s = s[~s.index.duplicated(keep="last")].sort_index()
        # Reindex to business days, then forward-fill (max 65 days ≈ ~2 months for quarterly)
        df[name] = s.reindex(df.index, method="ffill", limit=65)

    return df

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if FRED_API_KEY == "YOUR_FRED_API_KEY_HERE":
        print("=" * 65)
        print("  ACTION REQUIRED: Set your FRED API key.")
        print("  Get a free key at: https://fred.stlouisfed.org")
        print("  Then either:")
        print("    export FRED_API_KEY=your_key   (Linux/Mac)")
        print("    set FRED_API_KEY=your_key       (Windows CMD)")
        print("  Or paste it into FRED_API_KEY variable in this script.")
        print("=" * 65)
        raise SystemExit(1)

    print(f"\n[INFO] Collecting FRED macroeconomic data: {START_DATE} → {END_DATE}\n")

    all_series = {}

    # FRED series
    for col_name, (series_id, desc) in FRED_SERIES.items():
        try:
            s = fetch_fred_series(series_id, col_name, START_DATE, END_DATE)
            all_series[col_name] = s
        except Exception as e:
            print(f"  [ERROR] {col_name} ({series_id}): {e}")
        time.sleep(0.3)   # polite rate limiting

    # DXY via Yahoo Finance
    all_series["DXY_Index"] = fetch_dxy_yahoo(START_DATE, END_DATE)

    # Merge onto daily business-day grid
    print("\n[INFO] Merging all series to daily business-day index ...")
    df = merge_to_daily(all_series, START_DATE, END_DATE)

    # Sanity check
    print(f"\n[INFO] Final shape: {df.shape}")
    print(f"       Date range : {df.index.min().date()} → {df.index.max().date()}")
    null_pct = (df.isnull().sum() / len(df) * 100).round(2)
    print(f"       Null % per column:\n{null_pct}\n")

    # Save
    df.index = df.index.strftime("%Y-%m-%d")
    df.index.name = "Date"
    df.reset_index(inplace=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"[SAVED] {OUTPUT_FILE}")
    print(df.head(5).to_string(index=False))
    print(f"\nTotal rows: {len(df):,}")
