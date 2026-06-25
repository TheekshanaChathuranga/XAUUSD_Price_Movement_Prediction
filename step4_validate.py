"""
STEP 4: Centralized Data Storage & Inspection
==============================================
Validates all three CSV files produced by Steps 1–3.

Checks:
  • File existence & non-empty
  • Required columns present
  • Date column format (YYYY-MM-DD) & parseable
  • No duplicate dates (price data)
  • Null rate per column
  • Value range sanity (price > 0, rate 0–100, etc.)
  • Overlap between price data dates and macro data dates
  • Basic dataset statistics printed in a report
"""

import os
import sys
import pandas as pd

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "price" : os.path.join(OUTPUT_DIR, "xauusd_raw_prices.csv"),
    "macro" : os.path.join(OUTPUT_DIR, "fred_macro_raw.csv"),
    "news"  : os.path.join(OUTPUT_DIR, "financial_news_raw.csv"),
}

REQUIRED_COLS = {
    "price": ["Date", "Open", "High", "Low", "Close", "Tick_Volume"],
    "macro": ["Date", "CPI_US", "FedFunds_Rate", "Unemployment_Rate",
              "NFP_Change", "WTI_Crude_Oil", "DXY_Index"],
    "news" : ["Date", "Datetime", "Headline", "Source"],
}

PASSED = []
FAILED = []

def section(title: str) -> None:
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")

def check(condition: bool, msg: str) -> None:
    if condition:
        PASSED.append(msg)
        print(f"  ✓  {msg}")
    else:
        FAILED.append(msg)
        print(f"  ✗  {msg}  ← FAILED")

def validate_file(label: str, path: str, req_cols: list) -> pd.DataFrame | None:
    section(f"Validating: {os.path.basename(path)}")

    # 1. Existence
    exists = os.path.isfile(path)
    check(exists, f"File exists at {path}")
    if not exists:
        return None

    # 2. Load
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        check(False, f"File is readable ({e})")
        return None
    check(True, f"File is readable  ({len(df):,} rows × {df.shape[1]} cols)")

    # 3. Non-empty
    check(len(df) > 0, "DataFrame has rows")

    # 4. Required columns
    missing_cols = [c for c in req_cols if c not in df.columns]
    check(len(missing_cols) == 0,
          f"Required columns present  (missing: {missing_cols or 'none'})")

    # 5. Date column format
    if "Date" in df.columns:
        sample = df["Date"].dropna().head(5).tolist()
        date_ok = all(pd.to_datetime(d, errors="coerce") is not pd.NaT
                      and str(d)[:4].isdigit() for d in sample)
        check(date_ok, f"Date column parseable as YYYY-MM-DD  (sample: {sample[:2]})")

        # 6. No duplicate dates (for price data)
        if label == "price":
            n_dupes = df["Date"].duplicated().sum()
            check(n_dupes == 0, f"No duplicate Date entries  (found: {n_dupes})")

    # 7. Null rates
    null_pct = (df.isnull().sum() / len(df) * 100).round(2)
    high_null = null_pct[null_pct > 20]
    check(len(high_null) == 0,
          f"No column with >20% nulls  "
          f"(high-null cols: {high_null.to_dict() if not high_null.empty else 'none'})")

    # 8. Value range sanity
    if label == "price" and "Close" in df.columns:
        close_min = pd.to_numeric(df["Close"], errors="coerce").min()
        check(close_min > 0, f"Close prices all positive  (min={close_min:.2f})")
        high_gt_low = (pd.to_numeric(df["High"], errors="coerce") >=
                       pd.to_numeric(df["Low"],  errors="coerce")).all()
        check(high_gt_low, "High ≥ Low for all rows")

    if label == "macro" and "FedFunds_Rate" in df.columns:
        rate_ok = pd.to_numeric(df["FedFunds_Rate"], errors="coerce").between(0, 25).all()
        check(rate_ok, "Federal Funds Rate values in plausible range (0–25%)")

    if label == "news" and "Headline" in df.columns:
        empty_headlines = df["Headline"].isna().sum() + (df["Headline"] == "").sum()
        check(empty_headlines == 0, f"No empty headlines  (empty count: {empty_headlines})")
        avg_len = df["Headline"].str.len().mean()
        check(avg_len > 20, f"Average headline length > 20 chars  (avg={avg_len:.1f})")

    # 9. Statistics summary
    print(f"\n  ── Summary ─────────────────────────────────────────────")
    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
        print(f"    Date range : {dates.min().date()} → {dates.max().date()}")
        print(f"    Row count  : {len(df):,}")
    if label == "macro":
        print(f"    Columns    : {list(df.columns)}")
    if label == "news" and "Source" in df.columns:
        print(f"    Sources    :\n{df['Source'].value_counts().to_string()}")
    print(f"    Null rates (%):\n{null_pct.to_string()}")

    return df

# ─── CROSS-FILE ALIGNMENT CHECK ───────────────────────────────────────────────
def alignment_check(df_price: pd.DataFrame, df_macro: pd.DataFrame) -> None:
    section("Cross-File Alignment Check (Price ↔ Macro)")
    if df_price is None or df_macro is None:
        print("  [SKIP] One or both files failed to load.")
        return

    price_dates = set(pd.to_datetime(df_price["Date"], errors="coerce").dropna()
                      .dt.strftime("%Y-%m-%d"))
    macro_dates = set(pd.to_datetime(df_macro["Date"], errors="coerce").dropna()
                      .dt.strftime("%Y-%m-%d"))

    overlap = price_dates & macro_dates
    overlap_pct = len(overlap) / max(len(price_dates), 1) * 100
    print(f"  Price dates  : {len(price_dates):,}")
    print(f"  Macro dates  : {len(macro_dates):,}")
    print(f"  Overlapping  : {len(overlap):,}  ({overlap_pct:.1f}% of price rows covered)")
    check(overlap_pct > 70,
          f"Macro dates cover ≥70% of price dates  ({overlap_pct:.1f}%)")

# ─── FINAL REPORT ─────────────────────────────────────────────────────────────
def print_report() -> None:
    section("FINAL VALIDATION REPORT")
    total = len(PASSED) + len(FAILED)
    print(f"\n  PASSED: {len(PASSED)}/{total}")
    print(f"  FAILED: {len(FAILED)}/{total}\n")
    if FAILED:
        print("  ── Failed Checks ────────────────────────────────────────")
        for f in FAILED:
            print(f"    ✗ {f}")
        print()
    if len(FAILED) == 0:
        print("  ✅  ALL CHECKS PASSED — Ready to proceed to Phase 2!")
    else:
        print("  ⚠️   Fix the failed checks before moving to Phase 2.")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nData_Collection Sanity Check — {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df_price = validate_file("price", FILES["price"], REQUIRED_COLS["price"])
    df_macro = validate_file("macro", FILES["macro"], REQUIRED_COLS["macro"])
    df_news  = validate_file("news",  FILES["news"],  REQUIRED_COLS["news"])

    alignment_check(df_price, df_macro)
    print_report()

    sys.exit(0 if not FAILED else 1)
