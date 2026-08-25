"""
Step 9b: SL/TP ATR-Multiplier Grid Search
==========================================
Sweeps SL and TP ATR multipliers across a 4x4 grid, runs the full intra-bar
SL/TP simulation for each combination, and ranks results by profit factor.

WIN-RATE TRAP GUARD:
    Any config where:
      - Profit Factor < TRAP_MIN_PF  (default 1.30), OR
      - Avg Win / |Avg Loss| < TRAP_MIN_WL_RATIO  (default 0.80)
    is flagged [WR-TRAP]. High win rate with collapsing reward:risk is NOT
    a genuine improvement and is explicitly called out in the output.

Usage:
    python step9b_sl_tp_sweep.py

Outputs:
    sl_tp_sweep_results.csv   -- full grid sorted by profit factor
    Console                   -- top-5 table + written recommendation
    model_threshold.json      -- updated with chosen SL/TP params
    model_threshold.json.bak  -- backup of previous config
"""

import os
import sys
import json
import shutil
import numpy as np
import pandas as pd

if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step9_backtest_strategy as s9

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# SWEEP PARAMETERS  -- edit here to change the grid
# ---------------------------------------------------------------------------
SL_MULTS = [0.50, 0.75, 1.00, 1.25]    # stop-loss ATR multipliers
TP_MULTS = [0.75, 1.00, 1.50, 2.00]    # take-profit ATR multipliers

TRAP_MIN_PF       = 1.30    # flag if profit factor below this
TRAP_MIN_WL_RATIO = 0.80    # flag if avg_win / avg_loss below this


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def trap_flag(pf, wl_ratio):
    """Return 'OK' or a descriptive WR-TRAP warning string."""
    flags = []
    if pf < TRAP_MIN_PF:
        flags.append("PF<" + str(TRAP_MIN_PF))
    if wl_ratio < TRAP_MIN_WL_RATIO:
        flags.append("WL<" + str(TRAP_MIN_WL_RATIO))
    return ("WR-TRAP: " + ", ".join(flags)) if flags else "OK"


def run_single_backtest(df_base, sl_mult, tp_mult):
    """Run one full backtest for (sl_mult, tp_mult). Returns metrics dict."""
    df = df_base.copy()
    df = s9.simulate_trading_with_sl_tp(df, sl_mult=sl_mult, tp_mult=tp_mult)

    df["Prev_Signal"] = df["Signal_BT"].shift(1).fillna(0)
    gross = df["Strategy_Return_Gross"]
    net   = df["Strategy_Return_Net"]

    wins    = gross[gross > 0]
    losses  = gross[gross < 0]
    nonzero = gross[gross != 0]

    win_rate      = len(wins) / len(nonzero)  if len(nonzero) > 0 else 0.0
    avg_win       = float(wins.mean())        if len(wins)    > 0 else 0.0
    avg_loss      = float(losses.mean())      if len(losses)  > 0 else 0.0
    wl_ratio      = avg_win / abs(avg_loss)   if abs(avg_loss) > 1e-10 else float("inf")
    gross_profit  = wins.sum()
    gross_loss    = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-8 else float("inf")

    mean_r = net.mean()
    std_r  = net.std()
    sharpe = (mean_r / (std_r + 1e-8)) * np.sqrt(252)

    cum      = np.exp(net.cumsum())
    roll_max = np.maximum.accumulate(cum)
    max_dd   = float((cum / roll_max - 1.0).min()) * 100

    total_return_pct = float((np.exp(net.sum()) - 1) * 100)
    n_active = int((df["Prev_Signal"] != 0).sum())
    n_trades = int((df["Position_Size"] != 0).sum())

    return {
        "sl_mult":       sl_mult,
        "tp_mult":       tp_mult,
        "rr_ratio":      round(tp_mult / sl_mult, 2),
        "n_active_days": n_active,
        "n_trades":      n_trades,
        "win_rate_pct":  round(win_rate * 100, 2),
        "avg_win_bp":    round(avg_win * 10000, 2),
        "avg_loss_bp":   round(avg_loss * 10000, 2),
        "wl_ratio":      round(wl_ratio, 3),
        "profit_factor": round(profit_factor, 4),
        "sharpe":        round(sharpe, 4),
        "max_dd_pct":    round(max_dd, 2),
        "net_roi_pct":   round(total_return_pct, 2),
        "trap_flag":     trap_flag(profit_factor, wl_ratio),
    }


def print_results_table(results_df, n_show=5):
    top = results_df.head(n_show)
    sep = "=" * 108
    print()
    print(sep)
    print("  TOP " + str(n_show) + " CONFIGURATIONS BY PROFIT FACTOR")
    print(sep)
    cols = ["#", "SL*ATR", "TP*ATR", "R:R", "WR%", "AvgWin(bp)",
            "AvgLoss(bp)", "W/L", "ProfFactor", "Sharpe", "MaxDD%", "NetROI%", "Flag"]
    widths = [3, 8, 8, 6, 8, 12, 13, 7, 12, 9, 9, 10, 10]
    header = "  " + "".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("  " + "-" * 106)
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        flag_s = "OK" if row["trap_flag"] == "OK" else "[!] TRAP"
        vals = [
            str(rank),
            str(row["sl_mult"]),
            str(row["tp_mult"]),
            str(row["rr_ratio"]),
            str(round(row["win_rate_pct"], 1)) + "%",
            str(round(row["avg_win_bp"], 1)),
            str(round(row["avg_loss_bp"], 1)),
            str(round(row["wl_ratio"], 3)),
            str(round(row["profit_factor"], 4)),
            str(round(row["sharpe"], 4)),
            str(round(row["max_dd_pct"], 2)) + "%",
            str(round(row["net_roi_pct"], 2)) + "%",
            flag_s,
        ]
        print("  " + "".join(v.ljust(w) for v, w in zip(vals, widths)))
    print(sep)


def write_recommendation(results_df):
    best_all  = results_df.iloc[0]
    clean     = results_df[results_df["trap_flag"] == "OK"]
    best_clean = clean.iloc[0] if len(clean) > 0 else best_all

    sep = "=" * 80
    print()
    print(sep)
    print("  RECOMMENDATION")
    print(sep)

    if best_all["trap_flag"] != "OK":
        print()
        print("  [!] Best-by-PF config (SL=" + str(best_all["sl_mult"]) + "x, TP="
              + str(best_all["tp_mult"]) + "x) is flagged: " + best_all["trap_flag"])
        print("      Recommending best non-trap config instead.")
        chosen = best_clean
    else:
        chosen = best_all

    min_wr_needed = round(1.0 / (1.0 + chosen["rr_ratio"]) * 100, 1)
    sl_s  = str(chosen["sl_mult"])
    tp_s  = str(chosen["tp_mult"])
    rr_s  = str(chosen["rr_ratio"])

    print()
    print("  CHOSEN CONFIG:")
    print("    SL = " + sl_s + "x ATR    TP = " + tp_s + "x ATR    (R:R = 1:" + rr_s + ")")
    print()
    print("  Performance:")
    print("    Win Rate          : " + str(chosen["win_rate_pct"]) + "%")
    print("    Profit Factor     : " + str(chosen["profit_factor"]))
    print("    Sharpe (Ann.)     : " + str(chosen["sharpe"]))
    print("    Max Drawdown      : " + str(chosen["max_dd_pct"]) + "%")
    print("    Net ROI           : " + str(chosen["net_roi_pct"]) + "%")
    print("    Avg Win / Avg Loss: " + str(chosen["wl_ratio"])
          + ("  [OK]" if chosen["wl_ratio"] >= TRAP_MIN_WL_RATIO else "  [BORDERLINE]"))
    print("    Trap Flag         : " + str(chosen["trap_flag"]))
    print()
    print("  Rationale:")
    print("    SL=" + sl_s + "xATR cuts losing trades before they compound while")
    print("    still allowing typical daily noise to resolve.")
    print("    TP=" + tp_s + "xATR captures realistic intra-day momentum without")
    print("    requiring a rare multi-ATR impulse move.")
    print("    Break-even WR at this R:R is " + str(min_wr_needed)
          + "% -- observed " + str(chosen["win_rate_pct"]) + "% gives positive expectancy.")
    print()
    print("  Next steps:")
    print("    1. model_threshold.json has been updated with these values.")
    print("    2. Re-run step9_backtest_strategy.py (USE_SL_TP=True) for updated charts.")
    print("    3. step10_live_inference.py now reads SL/TP from model_threshold.json.")
    print(sep)

    return chosen


def save_chosen_to_threshold_json(chosen_row):
    threshold_path = os.path.join(OUTPUT_DIR, "model_threshold.json")
    backup_path    = os.path.join(OUTPUT_DIR, "model_threshold.json.bak")

    if os.path.exists(threshold_path):
        shutil.copy2(threshold_path, backup_path)
        print()
        print("  Backed up model_threshold.json  ->  " + backup_path)
        with open(threshold_path) as fh:
            cfg = json.load(fh)
    else:
        cfg = {}

    cfg["sl_atr_mult"]     = float(chosen_row["sl_mult"])
    cfg["tp_atr_mult"]     = float(chosen_row["tp_mult"])
    cfg["atr_period"]      = s9.ATR_PERIOD
    cfg["sl_tp_tie_break"] = s9.SL_TP_TIE_BREAK
    cfg["sweep_best_pf"]   = float(chosen_row["profit_factor"])
    cfg["sweep_best_wr"]   = float(chosen_row["win_rate_pct"])
    cfg["sweep_note"] = (
        "Chosen by step9b_sl_tp_sweep.py  |  "
        "SL=" + str(chosen_row["sl_mult"]) + "xATR  "
        "TP=" + str(chosen_row["tp_mult"]) + "xATR  |  "
        "Old values backed up to model_threshold.json.bak"
    )

    with open(threshold_path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    print("  Updated model_threshold.json with chosen SL/TP parameters.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  STEP 9b: SL/TP ATR MULTIPLIER GRID SEARCH")
    print("  SL grid : " + str(SL_MULTS))
    print("  TP grid : " + str(TP_MULTS))
    print("  Total   : " + str(len(SL_MULTS) * len(TP_MULTS)) + " combinations")
    print("=" * 70)

    # Step 1: Load data once -- shared across all sweep runs
    print()
    print("[1] Loading and preparing data (shared for all runs)...")
    s9.USE_SL_TP = True   # ensure ATR gets computed inside load_and_merge_data
    df_base = s9.load_and_merge_data()
    df_base, long_thresh, short_thresh = s9.generate_adaptive_signals(df_base)
    df_base = s9.apply_position_sizing(df_base)
    print("    Data ready: " + str(len(df_base)) + " rows")

    # Step 2: Grid sweep
    print()
    print("[2] Running parameter sweep (suppressing per-run verbosity)...")
    orig_print = __builtins__.__dict__.get("print", print) if hasattr(__builtins__, "__dict__") else print

    col_header = ("    " + "SL*ATR".ljust(8) + "TP*ATR".ljust(8) +
                  "WR%".ljust(8) + "ProfFactor".ljust(12) +
                  "Sharpe".ljust(9) + "MaxDD%".ljust(9) + "Flag")
    print(col_header)
    print("    " + "-" * 62)

    results = []
    for sl in SL_MULTS:
        for tp in TP_MULTS:
            # Suppress noisy step9 output during sweep
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                row = run_single_backtest(df_base, sl_mult=sl, tp_mult=tp)
            finally:
                sys.stdout = old_stdout
            results.append(row)
            flag_s = "" if row["trap_flag"] == "OK" else "  [!] WR-TRAP"
            line = ("    " + str(sl).ljust(8) + str(tp).ljust(8) +
                    (str(round(row["win_rate_pct"], 1)) + "%").ljust(8) +
                    str(round(row["profit_factor"], 4)).ljust(12) +
                    str(round(row["sharpe"], 4)).ljust(9) +
                    (str(round(row["max_dd_pct"], 2)) + "%").ljust(9) + flag_s)
            print(line)

    results_df = (pd.DataFrame(results)
                  .sort_values("profit_factor", ascending=False)
                  .reset_index(drop=True))

    # Step 3: Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "sl_tp_sweep_results.csv")
    results_df.to_csv(csv_path, index=False, float_format="%.4f")
    print()
    print("[3] Full results saved  ->  " + csv_path)

    # Step 4: Top-5 table
    print_results_table(results_df, n_show=5)

    # Step 5: Trap summary
    traps = results_df[results_df["trap_flag"] != "OK"]
    if len(traps) > 0:
        print()
        print("[!] WR-TRAP FLAGGED CONFIGS (" + str(len(traps)) + " of "
              + str(len(results_df)) + " total):")
        for _, t in traps.iterrows():
            print("    SL=" + str(t["sl_mult"]) + "x  TP=" + str(t["tp_mult"]) +
                  "x  WR=" + str(t["win_rate_pct"]) + "%  PF=" +
                  str(t["profit_factor"]) + "  W/L=" + str(t["wl_ratio"]) +
                  "  ->  " + t["trap_flag"])

    # Step 6: Recommendation + update config
    chosen = write_recommendation(results_df)
    save_chosen_to_threshold_json(chosen)

    print()
    print("Grid search complete. " + str(len(results)) + " configs evaluated.")


if __name__ == "__main__":
    main()
