#!/usr/bin/env python3
"""
RUN_ALL.py  —  Master Data Collection Pipeline
===============================================
Runs Steps 1 → 4 in order and logs everything.

Usage:
    python run_all.py

Prerequisites:
    pip install yfinance fredapi requests beautifulsoup4 pandas lxml

Required before running:
    export FRED_API_KEY=your_32_char_key_here
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# Fix Unicode encoding on Windows
if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding='utf-8')
    os.environ["PYTHONIOENCODING"] = "utf-8"

LOG_DIR  = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "pipeline_run.log")

STEPS = [
    ("Step 1: XAU/USD Price Data",      "step1_collect_xauusd.py"),
    ("Step 2: FRED Macro Data",          "step2_collect_macro.py"),
    ("Step 3: Financial News",           "step3_collect_news.py"),
    ("Step 4: Validation & Sanity Check","step4_validate.py"),
]

def run_step(label: str, script: str) -> bool:
    script_path = os.path.join(LOG_DIR, script)
    print(f"\n{'━'*65}")
    print(f"  ▶  {label}")
    print(f"{'━'*65}")
    start = time.time()

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,   # stream output to terminal
    )

    elapsed = time.time() - start
    ok = result.returncode == 0
    status = "✓ DONE" if ok else "✗ FAILED"
    print(f"\n  {status}  ({elapsed:.1f}s)\n")
    return ok

def main():
    # Fallback for API Key if environment variable is not set
    if not os.getenv("FRED_API_KEY"):
        os.environ["FRED_API_KEY"] = "e29012b6f6978622fbaa6dabd709f6c6"

    print(f"\n{'═'*65}")
    print(f"  GOLD AI — DATA COLLECTION PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*65}")

    results = {}
    for label, script in STEPS:
        ok = run_step(label, script)
        results[label] = ok
        if not ok and "Validation" not in label:
            print(f"[ABORT] {label} failed — stopping pipeline.")
            break

    print(f"\n{'═'*65}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'═'*65}")
    all_ok = all(results.values())
    for step, ok in results.items():
        icon = "✓" if ok else "✗"
        print(f"  {icon}  {step}")

    print()
    if all_ok:
        print("  🎉  All steps completed successfully!")
        print(f"  📁  Output files in: {LOG_DIR}")
        print("       • xauusd_raw_prices.csv")
        print("       • fred_macro_raw.csv")
        print("       • financial_news_raw.csv")
        print("\n  ➜   Proceed to Phase 2: Preprocessing & NLP")
    else:
        print("  ⚠️   Some steps failed. Check output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
