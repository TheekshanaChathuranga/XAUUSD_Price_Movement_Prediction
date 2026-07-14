#!/usr/bin/env python3
"""
RUN_ALL.py  —  Full End-to-End XAUUSD Prediction Pipeline
==========================================================
Runs Steps 1 → 9 (including Step 5B Volume Profile) in order.

Pipeline:
  Step 1  : XAU/USD price data collection (Yahoo Finance / GC=F)
  Step 2  : FRED macro data collection
  Step 3  : Financial news collection (RSS / scraping)
  Step 3B : GDELT news collection
  Step 4  : Validation & sanity checks
  Step 5  : Feature engineering + FinBERT sentiment scoring
  Step 5B : Gold Futures Volume Profile feature engineering  ← NEW
  Step 6  : Dataset alignment & fusion
  Step 7  : Ensemble model training (CatBoost / XGBoost / LightGBM)
  Step 8  : SHAP interpretability
  Step 9  : Backtest strategy simulation + VP win-rate filters

Usage:
    python run_all.py              # full pipeline
    python run_all.py --from 5    # resume from step 5 onwards

Prerequisites:
    pip install yfinance fredapi requests beautifulsoup4 pandas lxml
    pip install torch transformers catboost xgboost lightgbm shap

Required before running:
    set FRED_API_KEY=your_32_char_key_here   (Windows)
    export FRED_API_KEY=your_32_char_key_here (Linux/Mac)
"""

import os
import sys
import subprocess
import time
import argparse
from datetime import datetime

# Fix Unicode encoding on Windows
if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding='utf-8')
    os.environ["PYTHONIOENCODING"] = "utf-8"

LOG_DIR  = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "pipeline_run.log")

# ── Step definitions (step_id, label, script, abort_on_fail) ─────────────────
STEPS = [
    (1,    "Step 1  : XAU/USD Price Collection",        "step1_collect_xauusd.py",    True),
    (2,    "Step 2  : FRED Macro Data Collection",       "step2_collect_macro.py",     True),
    (3,    "Step 3  : Financial News Collection",        "step3_collect_news.py",      False),
    ("3b", "Step 3B : GDELT News Collection",            "step3b_gdelt.py",            False),
    (4,    "Step 4  : Validation & Sanity Check",        "step4_validate.py",          False),
    (5,    "Step 5  : Feature Engineering + FinBERT",    "step5_preprocess_features.py", True),
    ("5b", "Step 5B : Volume Profile Features",          "step5b_volume_profile.py",   True),
    (6,    "Step 6  : Dataset Alignment & Fusion",       "step6_align_fusion.py",      True),
    (7,    "Step 7  : Ensemble Model Training",          "step7_train_ensemble.py",    True),
    (8,    "Step 8  : SHAP Interpretability",            "step8_shap_interpretability.py", False),
    (9,    "Step 9  : Backtest + VP Win-Rate Filters",   "step9_backtest_strategy.py", False),
]


def run_step(label: str, script: str) -> tuple[bool, float]:
    script_path = os.path.join(LOG_DIR, script)
    if not os.path.exists(script_path):
        print(f"  [SKIP] Script not found: {script}")
        return True, 0.0   # non-fatal skip

    print(f"\n{'━'*65}")
    print(f"  ▶  {label}")
    print(f"{'━'*65}")
    start = time.time()

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
    )

    elapsed = time.time() - start
    ok = result.returncode == 0
    status = "✓ DONE" if ok else "✗ FAILED"
    print(f"\n  {status}  ({elapsed:.1f}s)\n")
    return ok, elapsed


def parse_args():
    parser = argparse.ArgumentParser(description="XAUUSD Full Pipeline Runner")
    parser.add_argument(
        "--from", dest="from_step", default=None,
        help="Resume pipeline from this step number (e.g. 5 or 5b)"
    )
    parser.add_argument(
        "--only", dest="only_step", default=None,
        help="Run only one specific step (e.g. 9)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # API key fallback
    if not os.getenv("FRED_API_KEY"):
        os.environ["FRED_API_KEY"] = "e29012b6f6978622fbaa6dabd709f6c6"

    print(f"\n{'═'*65}")
    print(f"  XAUUSD GOLD AI — FULL END-TO-END PIPELINE")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Steps   : 1 → 9  (including 3B and 5B Volume Profile)")
    print(f"{'═'*65}")

    # Determine which steps to run
    active = False if args.from_step else True
    results   = {}
    timings   = {}

    for step_id, label, script, abort_on_fail in STEPS:
        sid = str(step_id)

        # --from logic: start running when we reach the target step
        if args.from_step and not active:
            if sid == str(args.from_step):
                active = True
            else:
                print(f"  [SKIP] {label}")
                continue

        # --only logic
        if args.only_step and sid != str(args.only_step):
            continue

        ok, elapsed = run_step(label, script)
        results[label] = ok
        timings[label] = elapsed

        if not ok and abort_on_fail:
            print(f"\n[ABORT] {label} failed — stopping pipeline.")
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = sum(timings.values())
    print(f"\n{'═'*65}")
    print(f"  PIPELINE SUMMARY  ({total_time:.0f}s total)")
    print(f"{'═'*65}")

    all_ok = all(results.values())
    for step, ok in results.items():
        icon    = "✓" if ok else "✗"
        elapsed = timings.get(step, 0)
        print(f"  {icon}  {step:<50}  {elapsed:>6.1f}s")

    print()
    if all_ok:
        print("  🎉  All steps completed successfully!")
        print(f"  📁  Outputs in: {LOG_DIR}")
        print()
        print("  Data files:")
        print("    • xauusd_raw_prices.csv")
        print("    • fred_macro_raw.csv")
        print("    • master_features.csv")
        print("    • volume_profile_features.csv  ← VP features (NEW)")
        print()
        print("  Model files:")
        print("    • catboost_prod.cbm  /  lgb_prod.txt  /  xgb_prod.json")
        print("    • meta_learner.pkl   /  scaler.pkl")
        print()
        print("  Backtest outputs:")
        print("    • backtest_trade_log.csv")
        print("    • backtest_performance.png")
        print("    • backtest_config.json")
    else:
        print("  ⚠️   Some steps failed — check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
