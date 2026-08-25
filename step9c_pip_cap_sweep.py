"""
Step 9c: SL/TP Pip-Range Cap Grid Search
==========================================
Sweeps SL and TP pip range caps across a grid:
  MIN_PIPS in [50, 100, 150]
  MAX_PIPS in [300, 400, 500]

Uses out-of-fold validation predictions and simulates trading with clamped
pips (1 pip = $0.10). Reports performance across 5 sequential folds and the
full test set. Logs cases where R:R degrades below 1:2.

WIN-RATE TRAP GUARD:
    Flags any config where:
      - Profit Factor < 1.30, OR
      - R:R ratio on capped trades degrades below 1:1.5

Usage:
    python step9c_pip_cap_sweep.py
"""

import os
import sys
import numpy as np
import pandas as pd
import math

if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step9_backtest_strategy as s9
import step3b_position_sizing as s3

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# GRID PARAMETERS
# ---------------------------------------------------------------------------
MIN_PIP_VALS = [50, 100, 150]
MAX_PIP_VALS = [300, 400, 500]

TRAP_MIN_PF = 1.30
TRAP_MIN_RR = 1.50

# ---------------------------------------------------------------------------
# SIMULATION WITH PIP CAP
# ---------------------------------------------------------------------------

def simulate_trading_with_pip_caps(df, sl_mult, tp_mult, min_pips, max_pips):
    """
    Simulates trading with intra-bar SL/TP and hard pip range caps.
    1 pip = $0.10 price move.
    """
    n = len(df)
    gross_returns = np.zeros(n)
    tc_costs      = np.zeros(n)
    
    signal_arr  = df['Signal_BT'].values
    pos_arr     = df['Position_Size'].values
    open_arr    = df['Open'].values   if 'Open'  in df.columns else df['Close'].values
    high_arr    = df['High'].values   if 'High'  in df.columns else df['Close'].values
    low_arr     = df['Low'].values    if 'Low'   in df.columns else df['Close'].values
    close_arr   = df['Close'].values
    atr_arr     = df['ATR'].values
    prev_close  = np.concatenate([[np.nan], close_arr[:-1]])

    # Metrics trackers
    trade_count = 0
    cap_sl_min_count = 0
    cap_sl_max_count = 0
    cap_tp_max_count = 0
    rr_degraded_count = 0
    trade_durations = []
    
    # Store SL/TP details for tracking
    sl_distances_pips = []
    tp_distances_pips = []
    
    # Track trade entry indexes
    entry_bar = -1
    
    for i in range(1, n):
        prev_sig  = signal_arr[i - 1]
        cur_sig   = signal_arr[i]
        pos_size  = pos_arr[i - 1]
        
        is_new_trade = False
        if i == 1:
            if pos_size != 0.0:
                is_new_trade = True
        else:
            prev_pos = pos_arr[i - 2]
            if pos_size != 0.0 and (prev_pos == 0.0 or np.sign(prev_pos) != np.sign(pos_size)):
                is_new_trade = True

        if pos_size == 0.0:
            if prev_sig == 0 and cur_sig != 0:
                tc_costs[i] += s9.REALISTIC_TC
            continue
            
        if entry_bar == -1:
            entry_bar = i
            
        entry_price = prev_close[i]
        atr_entry   = atr_arr[i - 1]
        
        if np.isnan(atr_entry) or atr_entry <= 0:
            gross_returns[i] = pos_size * np.log(close_arr[i] / entry_price) if entry_price > 0 else 0.0
            trade_durations.append(i - entry_bar + 1)
            entry_bar = -1
        else:
            # 1. Base ATR-scaled distances
            raw_sl_dist = sl_mult * atr_entry
            raw_tp_dist = tp_mult * atr_entry
            
            # 2. Convert to $0.10/pip units
            raw_sl_pips = raw_sl_dist / 0.10
            
            # 3. Clamp SL pips first
            clamped_sl_pips = np.clip(raw_sl_pips, min_pips, max_pips)
            if is_new_trade:
                if raw_sl_pips < min_pips:
                    cap_sl_min_count += 1
                elif raw_sl_pips > max_pips:
                    cap_sl_max_count += 1
                
            # 4. Set TP based on the 1:2 R:R ratio
            rr_ratio = tp_mult / sl_mult
            raw_tp_pips = clamped_sl_pips * rr_ratio
            
            # 5. Re-check if TP exceeds MAX_PIPS
            clamped_tp_pips = raw_tp_pips
            rr_degraded_trade = False
            if clamped_tp_pips > max_pips:
                clamped_tp_pips = max_pips
                if is_new_trade:
                    cap_tp_max_count += 1
                # If capped TP makes reward-to-risk ratio fall below 1.5, mark it
                realized_rr = clamped_tp_pips / clamped_sl_pips
                if realized_rr < TRAP_MIN_RR:
                    rr_degraded_trade = True
                    if is_new_trade:
                        rr_degraded_count += 1
                    
            if is_new_trade:
                sl_distances_pips.append(clamped_sl_pips)
                tp_distances_pips.append(clamped_tp_pips)
            
            # Convert back to dollar distances
            clamped_sl_dist = clamped_sl_pips * 0.10
            clamped_tp_dist = clamped_tp_pips * 0.10
            
            if pos_size > 0:  # LONG
                sl_price = entry_price - clamped_sl_dist
                tp_price = entry_price + clamped_tp_dist
                sl_hit   = low_arr[i]  <= sl_price
                tp_hit   = high_arr[i] >= tp_price
            else:             # SHORT
                sl_price = entry_price + clamped_sl_dist
                tp_price = entry_price - clamped_tp_dist
                sl_hit   = high_arr[i] >= sl_price
                tp_hit   = low_arr[i]  <= tp_price
                
            if sl_hit and tp_hit:
                if s9.SL_TP_TIE_BREAK == 'pessimistic':
                    exit_price = sl_price
                elif s9.SL_TP_TIE_BREAK == 'optimistic':
                    exit_price = tp_price
                else:
                    bar_open = open_arr[i]
                    if pos_size > 0:
                        proximity_tp = abs(bar_open - tp_price)
                        proximity_sl = abs(bar_open - sl_price)
                        exit_price = tp_price if proximity_tp <= proximity_sl else sl_price
                    else:
                        proximity_tp = abs(bar_open - tp_price)
                        proximity_sl = abs(bar_open - sl_price)
                        exit_price = tp_price if proximity_tp <= proximity_sl else sl_price
                trade_durations.append(i - entry_bar + 1)
                entry_bar = -1
            elif sl_hit:
                exit_price = sl_price
                trade_durations.append(i - entry_bar + 1)
                entry_bar = -1
            elif tp_hit:
                exit_price = tp_price
                trade_durations.append(i - entry_bar + 1)
                entry_bar = -1
            else:
                exit_price = close_arr[i]  # held through bar
                # If next day signal is neutral, we exit at close
                if i < n - 1 and signal_arr[i] == 0:
                    trade_durations.append(i - entry_bar + 1)
                    entry_bar = -1
                
            gross_returns[i] = pos_size * np.log(exit_price / entry_price) if entry_price > 0 else 0.0

        # Transaction costs
        prev_signal_t = signal_arr[i - 1]
        curr_signal_t = signal_arr[i]
        is_entry   = (signal_arr[i - 2] == 0) & (prev_signal_t != 0) if i >= 2 else False
        is_exit    = (prev_signal_t != 0) & (curr_signal_t == 0)
        is_reversal= (prev_signal_t != 0) & (curr_signal_t != 0) & (prev_signal_t != curr_signal_t)
        tc_costs[i] = (float(is_entry)    * s9.REALISTIC_TC +
                       float(is_exit)     * s9.EXIT_TC +
                       float(is_reversal) * (s9.REALISTIC_TC + s9.EXIT_TC))
                       
        if is_entry or is_reversal:
            trade_count += 1

    df_out = df.copy()
    df_out['Strategy_Return_Gross'] = gross_returns
    df_out['TC_Cost']               = tc_costs
    df_out['Strategy_Return_Net']   = gross_returns - tc_costs
    
    # Vol scaling
    df_out['RealizedVol'] = df_out['Close_Return'].rolling(s9.VOL_LOOKBACK, min_periods=5).std() * np.sqrt(252)
    df_out['VolScale']    = s9.VOL_TARGET / df_out['RealizedVol'].clip(lower=0.01)
    df_out['VolScale']    = df_out['VolScale'].clip(upper=2.0)
    df_out['Strategy_Return_VolAdj'] = df_out['Strategy_Return_Net'] * df_out['VolScale'].shift(1).fillna(1.0)
    df_out = df_out.dropna(subset=['Strategy_Return_Gross']).reset_index(drop=True)
    
    # Stats summary
    stats = {
        "trade_count": trade_count,
        "cap_sl_min_count": cap_sl_min_count,
        "cap_sl_max_count": cap_sl_max_count,
        "cap_tp_max_count": cap_tp_max_count,
        "rr_degraded_count": rr_degraded_count,
        "avg_duration": float(np.mean(trade_durations)) if trade_durations else 0.0,
        "sl_distances_pips": sl_distances_pips,
        "tp_distances_pips": tp_distances_pips
    }
    
    return df_out, stats

# ---------------------------------------------------------------------------
# METRICS EVALUATION
# ---------------------------------------------------------------------------

def evaluate_metrics(df, stats):
    """
    Computes ROI, Profit Factor, Win Rate, Sharpe, Max Drawdown.
    """
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
    max_dd   = float((cum / roll_max - 1.0).min()) * 100 if len(cum) > 0 else 0.0
    total_return_pct = float((np.exp(net.sum()) - 1) * 100) if len(net) > 0 else 0.0

    return {
        "n_trades":      stats["trade_count"],
        "win_rate_pct":  round(win_rate * 100, 2),
        "wl_ratio":      round(wl_ratio, 3),
        "profit_factor": round(profit_factor, 4),
        "sharpe":        round(sharpe, 4),
        "max_dd_pct":    round(max_dd, 2),
        "net_roi_pct":   round(total_return_pct, 2),
        "avg_duration":  round(stats["avg_duration"], 2),
        "min_capped_pct": round(stats["cap_sl_min_count"] / max(stats["trade_count"], 1) * 100, 1),
        "max_capped_pct": round(stats["cap_sl_max_count"] / max(stats["trade_count"], 1) * 100, 1),
        "tp_capped_pct":  round(stats["cap_tp_max_count"] / max(stats["trade_count"], 1) * 100, 1),
        "rr_degraded_pct": round(stats["rr_degraded_count"] / max(stats["trade_count"], 1) * 100, 1),
    }

# ---------------------------------------------------------------------------
# MAIN SWEEP LOOP
# ---------------------------------------------------------------------------

def main():
    print("=" * 75)
    print("  STEP 9c: SL/TP PIP RANGE CAP GRID SEARCH (1 pip = $0.10)")
    print("=" * 75)

    # 1. Load predictions & prices
    print("[1] Loading predictions & prices...")
    s9.USE_SL_TP = True
    df_base = s9.load_and_merge_data()
    df_base, _, _ = s9.generate_adaptive_signals(df_base)
    df_base = s9.apply_position_sizing(df_base)
    print(f"    Loaded {len(df_base)} rows.")

    # 2. Get baseline distribution of uncapped SL/TP
    # We will simulate trading once with a very large cap to get the natural distribution.
    _, base_stats = simulate_trading_with_pip_caps(df_base, s9.SL_ATR_MULT, s9.TP_ATR_MULT, 0.0, 99999.0)
    sl_raw = base_stats["sl_distances_pips"]
    tp_raw = base_stats["tp_distances_pips"]
    
    print("\n[2] Uncapped Pip-Size Distribution (Reference ATR 0.75x/1.0x):")
    print(f"    SL Pips — Min: {np.min(sl_raw):.1f} | P50: {np.median(sl_raw):.1f} | P90: {np.quantile(sl_raw, 0.90):.1f} | Max: {np.max(sl_raw):.1f}")
    print(f"    TP Pips — Min: {np.min(tp_raw):.1f} | P50: {np.median(tp_raw):.1f} | P90: {np.quantile(tp_raw, 0.90):.1f} | Max: {np.max(tp_raw):.1f}")
    
    # 3. Position sizing verification ($50 target risk)
    print("\n[3] Position Sizing Check ($50 target risk):")
    test_sl_pips_list = [500, 1000, 3000, 5000, 10580]
    for pips_display in test_sl_pips_list:
        dollar_dist = pips_display * 0.10
        cb_pips = dollar_dist / 0.01
        
        exact_lots = 50.0 / (cb_pips * 1.0)
        lot_size = math.floor(exact_lots / 0.01) * 0.01
        realized_risk = lot_size * cb_pips * 1.0
        
        print(f"    Capped SL = {pips_display:>5} pips ($0.10 convention) -> Price Dist: ${dollar_dist:>5.2f} -> Codebase Pips: {cb_pips:>6.1f} -> Lots: {lot_size:.2f} -> Risk: ${realized_risk:.2f}")

    # 4. Partition test data into 5 sequential folds (Walk-Forward Folds)
    n_rows = len(df_base)
    fold_size = n_rows // 5
    folds = []
    for k in range(5):
        start = k * fold_size
        end = n_rows if k == 4 else (k + 1) * fold_size
        folds.append((start, end))

    print(f"\n[4] Walk-Forward CV Setup: 5 chronological non-overlapping folds.")
    for k, (s, e) in enumerate(folds):
        print(f"    Fold {k+1}: rows {s} to {e} ({e-s} days) | Dates: {df_base['Date'].iloc[s].strftime('%Y-%m-%d')} to {df_base['Date'].iloc[min(e-1, n_rows-1)].strftime('%Y-%m-%d')}")

    # 5. Grid Search
    print("\n[5] Running Grid Search...")
    grid_results = []
    
    for min_p in MIN_PIP_VALS:
        for max_p in MAX_PIP_VALS:
            # 5a. Evaluate on overall test set
            df_full, stats_full = simulate_trading_with_pip_caps(
                df_base, s9.SL_ATR_MULT, s9.TP_ATR_MULT, min_p, max_p
            )
            metrics_full = evaluate_metrics(df_full, stats_full)
            
            # 5b. Evaluate on folds individually
            fold_pfs = []
            fold_wrs = []
            fold_sharpes = []
            
            for k, (s, e) in enumerate(folds):
                df_fold = df_base.iloc[max(0, s-20):e].copy()
                df_fold_sim, stats_fold = simulate_trading_with_pip_caps(
                    df_fold, s9.SL_ATR_MULT, s9.TP_ATR_MULT, min_p, max_p
                )
                df_fold_sim = df_fold_sim[df_fold_sim['Date'] >= df_base['Date'].iloc[s]].reset_index(drop=True)
                metrics_fold = evaluate_metrics(df_fold_sim, stats_fold)
                
                fold_pfs.append(metrics_fold["profit_factor"])
                fold_wrs.append(metrics_fold["win_rate_pct"])
                fold_sharpes.append(metrics_fold["sharpe"])
                
            avg_fold_pf = np.mean([f for f in fold_pfs if not np.isnan(f) and not np.isinf(f)])
            avg_fold_wr = np.mean(fold_wrs)
            avg_fold_sharpe = np.mean(fold_sharpes)
            
            is_trap = "OK"
            if metrics_full["profit_factor"] < TRAP_MIN_PF:
                is_trap = "WR-TRAP (PF < 1.30)"
            elif metrics_full["rr_degraded_pct"] > 50.0:
                is_trap = "R:R-DEGRADED (>50% trades)"
                
            grid_results.append({
                "min_pips": min_p,
                "max_pips": max_p,
                "win_rate_pct": metrics_full["win_rate_pct"],
                "n_trades": metrics_full["n_trades"],
                "avg_duration": metrics_full["avg_duration"],
                "profit_factor": metrics_full["profit_factor"],
                "wl_ratio": metrics_full["wl_ratio"],
                "sharpe": metrics_full["sharpe"],
                "max_dd_pct": metrics_full["max_dd_pct"],
                "net_roi_pct": metrics_full["net_roi_pct"],
                "min_capped_pct": metrics_full["min_capped_pct"],
                "max_capped_pct": metrics_full["max_capped_pct"],
                "tp_capped_pct": metrics_full["tp_capped_pct"],
                "rr_degraded_pct": metrics_full["rr_degraded_pct"],
                "avg_fold_pf": round(avg_fold_pf, 4),
                "avg_fold_wr": round(avg_fold_wr, 2),
                "avg_fold_sharpe": round(avg_fold_sharpe, 4),
                "flag": is_trap
            })
            
    results_df = pd.DataFrame(grid_results).sort_values("profit_factor", ascending=False).reset_index(drop=True)
    
    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "pip_cap_sweep_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n[6] Full sweep results saved to {csv_path}")
    
    # Print Table
    print("\n" + "=" * 105)
    print("  GRID RESULTS RANKED BY PROFIT FACTOR")
    print("=" * 105)
    cols = ["MinP", "MaxP", "WR%", "Trades", "Dur(d)", "PF", "W/L", "Sharpe", "MaxDD%", "ROI%", "Capped%", "Fold PF", "Flag"]
    widths = [6, 6, 7, 8, 8, 8, 7, 8, 9, 8, 9, 9, 12]
    header = "  " + "".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("  " + "-" * 103)
    for _, row in results_df.iterrows():
        cap_str = f"{row['min_capped_pct']:.0f}/{row['max_capped_pct']:.0f}%"
        vals = [
            str(row["min_pips"]),
            str(row["max_pips"]),
            f"{row['win_rate_pct']:.1f}%",
            str(int(row["n_trades"])),
            str(row["avg_duration"]),
            f"{row['profit_factor']:.4f}",
            f"{row['wl_ratio']:.3f}",
            f"{row['sharpe']:.4f}",
            f"{row['max_dd_pct']:.2f}%",
            f"{row['net_roi_pct']:.2f}%",
            cap_str,
            f"{row['avg_fold_pf']:.4f}",
            str(row["flag"])
        ]
        print("  " + "".join(v.ljust(w) for v, w in zip(vals, widths)))
    print("=" * 105)

    # Recommendations
    best_row = results_df.iloc[0]
    print("\n[7] RECOMMENDATION:")
    print(f"    Best config: MIN_PIPS = {best_row['min_pips']} pips, MAX_PIPS = {best_row['max_pips']} pips")
    print(f"    Overall Profit Factor: {best_row['profit_factor']} (Average Fold PF: {best_row['avg_fold_pf']})")
    print(f"    Overall Win Rate: {best_row['win_rate_pct']}% | Sharpe: {best_row['sharpe']} | ROI: {best_row['net_roi_pct']}%")
    print(f"    Trades min-capped: {best_row['min_capped_pct']}% | max-capped: {best_row['max_capped_pct']}% | R:R degraded: {best_row['rr_degraded_pct']}%")
    print(f"    Status: {best_row['flag']}")

if __name__ == "__main__":
    main()
