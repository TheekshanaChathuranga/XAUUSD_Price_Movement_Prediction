#!/usr/bin/env python3
"""
DAILY REFRESH PIPELINE
======================
Run this every morning to get TODAY's fresh forecast.
It re-collects data and regenerates live_inference_data.csv.

Usage:
    python daily_refresh.py

Steps:
    1. Collect fresh XAU/USD prices        (step1)
    2. Collect fresh FRED macro data       (step2)
    3. Collect fresh news (GDELT)          (step3b)
    4. Preprocess & sentiment scoring      (step5)
    5. Align & fuse features               (step6)
    6. Regenerate live inference row       (already done in step6)

Note: step3_collect_news.py (Google/RSS scraper) is optional and slow.
      It's skipped here; GDELT (step3b) covers news well.
"""

import os
import sys
import subprocess
import time
from datetime import datetime, date

if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding="utf-8")
    os.environ["PYTHONIOENCODING"] = "utf-8"

DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("Collect XAU/USD Prices",       "step1_collect_xauusd.py"),
    ("Collect FRED Macro Data",       "step2_collect_macro.py"),
    ("Collect GDELT News",            "step3b_gdelt.py"),
    ("Preprocess & FinBERT Sentiment","step5_preprocess_features.py"),
    ("Align & Fuse Features",         "step6_align_fusion.py"),
]

def run_step(label, script):
    path = os.path.join(DIR, script)
    print(f"\n{'─'*60}")
    print(f"  ▶  {label}")
    print(f"{'─'*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, path])
    elapsed = time.time() - t0
    ok = result.returncode == 0
    print(f"\n  {'OK' if ok else 'FAILED'}  ({elapsed:.1f}s)")
    return ok

def health_check():
    import pandas as pd
    today = date.today()
    print(f"\n{'═'*60}")
    print("  DATA FRESHNESS REPORT")
    print(f"{'═'*60}")
    checks = {
        "live_inference_data.csv": "Date",
        "xauusd_raw_prices.csv":   "Date",
        "gdelt_news_raw.csv":      "Date",
        "master_features.csv":     "Date",
    }
    all_fresh = True
    for fname, col in checks.items():
        fpath = os.path.join(DIR, fname)
        try:
            df = pd.read_csv(fpath)
            latest = pd.to_datetime(df[col]).max().date()
            age = (today - latest).days
            status = "FRESH" if age <= 1 else f"STALE ({age}d old)"
            freshness = "OK" if age <= 1 else "WARNING"
            print(f"  [{freshness:^7}]  {fname:<35}  Latest: {latest}")
            if age > 1:
                all_fresh = False
        except Exception as e:
            print(f"  [ERROR  ]  {fname:<35}  {e}")
            all_fresh = False
    return all_fresh

if __name__ == "__main__":
    # Set FRED API key
    if not os.getenv("FRED_API_KEY"):
        os.environ["FRED_API_KEY"] = "e29012b6f6978622fbaa6dabd709f6c6"

    print(f"\n{'═'*60}")
    print("  GOLD AI — DAILY REFRESH PIPELINE")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*60}")

    results = {}
    for label, script in STEPS:
        ok = run_step(label, script)
        results[label] = ok
        if not ok:
            print(f"\n[ABORT] {label} failed. Fix the error above and re-run.")
            sys.exit(1)

    print(f"\n{'═'*60}")
    print("  REFRESH COMPLETE — ALL STEPS SUCCEEDED")
    print(f"{'═'*60}")

    all_fresh = health_check()

    print(f"\n{'═'*60}")
    if all_fresh:
        print("  TODAY's forecast is ready.")
        print("  Start or restart the API server:")
        print("    python step11_api_server.py")
    else:
        print("  WARNING: Some data may still be stale.")
        print("  Check the errors above.")
    print(f"{'═'*60}\n")
