"""
Backtest Comparison: Technical vs Fundamental Trades (with Lower Range Theory)
=============================================================================
Runs separate backtests for:
1. Technical Trade (Lower Range Theory: 0.25x ATR SL / 0.50x ATR TP)
2. Technical Trade (Swing Theory: 1.50x ATR SL / 3.00x ATR TP) - for comparison
3. Fundamental Trade (Lower Range Theory: 0.25x ATR SL / 0.50x ATR TP)

Enforces the pip cap limits (MIN_PIPS = 100, MAX_PIPS = 500, 1 pip = $0.10).
Outputs win rate, trade count, profit factor, Sharpe ratio, and drawdowns.
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
import step9c_pip_cap_sweep as s9c

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# LOAD & PREPARE DATA
# ---------------------------------------------------------------------------

def load_data_with_fundamental_signals():
    """
    Loads baseline predictions and merges with Mean_Sentiment from the master dataset
    to generate Fundamental news-based signals.
    """
    s9.USE_SL_TP = True
    df = s9.load_and_merge_data()
    
    # 1. Generate Technical Ensemble signals
    df, _, _ = s9.generate_adaptive_signals(df)
    df = s9.apply_position_sizing(df)
    
    # Keep copy of technical position sizing columns
    df['Tech_Signal'] = df['Signal_BT']
    df['Tech_Position_Size'] = df['Position_Size']
    
    # 2. Merge with master features to get daily aggregate news sentiment
    master_path = os.path.join(OUTPUT_DIR, "multimodal_master_dataset.csv")
    if not os.path.exists(master_path):
        print(f"Error: {master_path} not found!")
        sys.exit(1)
        
    master_df = pd.read_csv(master_path, usecols=['Date', 'Mean_Sentiment'])
    master_df['Date'] = pd.to_datetime(master_df['Date'])
    
    df = df.merge(master_df, on='Date', how='inner')
    
    # 3. Generate Fundamental signals (Mean_Sentiment thresholds at +/- 0.15)
    df['Fund_Signal'] = 0
    df.loc[df['Mean_Sentiment'] > 0.15, 'Fund_Signal'] = 1
    df.loc[df['Mean_Sentiment'] < -0.15, 'Fund_Signal'] = -1
    
    # 4. Generate Fundamental position sizing (confidence-weighted)
    # Edge is normalized |Mean_Sentiment| relative to 0.50
    edge = np.abs(df['Mean_Sentiment']) / 0.50
    edge = np.clip(edge, 0.0, 1.0)
    fund_scale = 0.3 + 0.7 * edge
    
    df['Fund_Position_Size_Base'] = df['Fund_Signal'] * fund_scale
    df['Fund_Position_Size'] = df['Fund_Position_Size_Base'].copy()
    
    # Apply Volume Profile filters to Fundamental Trade to keep it apples-to-apples
    has_vp = s9.USE_VP_FILTERS and ('POC_Distance_60' in df.columns)
    if has_vp:
        poc_dist   = df['POC_Distance_60'].values
        in_any_lvn = df['In_Any_LVN'].values if 'In_Any_LVN' in df.columns else np.zeros(len(df))
        vah_break  = df['VAH_Breakout_Strength'].values if 'VAH_Breakout_Strength' in df.columns else np.zeros(len(df))
        val_break  = df['VAL_Breakdown_Strength'].values if 'VAL_Breakdown_Strength' in df.columns else np.zeros(len(df))
        signal     = df['Fund_Signal'].values
        
        for i in range(len(df)):
            if signal[i] == 0:
                continue
            
            current_size = df.at[df.index[i], 'Fund_Position_Size']
            
            # POC Alignment
            if signal[i] == 1 and poc_dist[i] > s9.VP_POC_TOLERANCE:
                df.at[df.index[i], 'Fund_Position_Size'] = 0.0
                continue
            elif signal[i] == -1 and poc_dist[i] < -s9.VP_POC_TOLERANCE:
                df.at[df.index[i], 'Fund_Position_Size'] = 0.0
                continue
                
            # LVN reduction
            if in_any_lvn[i] == 1:
                df.at[df.index[i], 'Fund_Position_Size'] = current_size * s9.VP_LVN_SIZE_MULT
                current_size = df.at[df.index[i], 'Fund_Position_Size']
                
            # VAH/VAL boost
            if signal[i] == 1 and vah_break[i] == 1:
                df.at[df.index[i], 'Fund_Position_Size'] = min(abs(current_size) * s9.VP_VAH_SIZE_MULT, 1.0) * np.sign(current_size)
            elif signal[i] == -1 and val_break[i] == 1:
                df.at[df.index[i], 'Fund_Position_Size'] = min(abs(current_size) * s9.VP_VAH_SIZE_MULT, 1.0) * np.sign(current_size)
                
    return df

# ---------------------------------------------------------------------------
# RUN INDIVIDUAL BACKTEST
# ---------------------------------------------------------------------------

def run_backtest_for_strategy(df_base, strategy_type, sl_mult, tp_mult, min_pips=100, max_pips=500):
    """
    Runs trading simulation for a given signal type (Tech or Fund) and SL/TP config.
    """
    df = df_base.copy()
    
    # Swap signal and position size to Strategy's signal
    if strategy_type == "Technical":
        df['Signal_BT'] = df['Tech_Signal']
        df['Position_Size'] = df['Tech_Position_Size']
    elif strategy_type == "Fundamental":
        df['Signal_BT'] = df['Fund_Signal']
        df['Position_Size'] = df['Fund_Position_Size']
        
    # Simulate trading with pip capping
    df_sim, stats = s9c.simulate_trading_with_pip_caps(
        df, sl_mult=sl_mult, tp_mult=tp_mult, min_pips=min_pips, max_pips=max_pips
    )
    
    # Calculate metrics
    metrics = s9c.evaluate_metrics(df_sim, stats)
    return metrics, df_sim

# ---------------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("  XAU/USD BACKTEST: TECHNICAL vs FUNDAMENTAL (Lower Range Theory)")
    print("  Lower Range Theory Multipliers: SL = 0.25x ATR | TP = 0.50x ATR")
    print("  Swing Theory Multipliers      : SL = 1.50x ATR | TP = 3.00x ATR")
    print("  Pip Cap Constraints          : MIN = 100 pips | MAX = 500 pips")
    print("=" * 80)

    # 1. Load data
    print("\n[1] Loading data & merging signals...")
    df = load_data_with_fundamental_signals()
    print(f"    Data ready. Row count: {len(df)}")
    print(f"    Technical Signals   — LONG: {(df['Tech_Signal'] == 1).sum()} | SHORT: {(df['Tech_Signal'] == -1).sum()} | NEUTRAL: {(df['Tech_Signal'] == 0).sum()}")
    print(f"    Fundamental Signals — LONG: {(df['Fund_Signal'] == 1).sum()} | SHORT: {(df['Fund_Signal'] == -1).sum()} | NEUTRAL: {(df['Fund_Signal'] == 0).sum()}")

    # 2. Run the three configurations
    print("\n[2] Simulating trade strategies...")
    
    # Config A: Technical Trade (Lower Range Theory)
    tech_lower_metrics, _ = run_backtest_for_strategy(
        df, "Technical", sl_mult=0.25, tp_mult=0.50
    )
    
    # Config B: Technical Trade (Swing Theory)
    tech_swing_metrics, _ = run_backtest_for_strategy(
        df, "Technical", sl_mult=1.50, tp_mult=3.00
    )
    
    # Config C: Fundamental Trade (Lower Range Theory)
    fund_lower_metrics, _ = run_backtest_for_strategy(
        df, "Fundamental", sl_mult=0.25, tp_mult=0.50
    )

    # 3. Print Results Table
    print("\n" + "=" * 105)
    print("  BACKTEST PERFORMANCE COMPARISON TABLE")
    print("=" * 105)
    cols = ["Strategy / Theory", "SL/TP Mult", "Trades", "Win Rate %", "Profit Factor", "W/L Ratio", "Sharpe", "Max DD %", "Net ROI %", "Capped SL%"]
    widths = [32, 13, 8, 12, 15, 12, 9, 10, 10, 11]
    header = "".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * 105)
    
    strategies = [
        ("Technical (Lower Range Theory)", "0.25x / 0.50x", tech_lower_metrics),
        ("Technical (Swing Theory Reference)", "1.50x / 3.00x", tech_swing_metrics),
        ("Fundamental (Lower Range Theory)", "0.25x / 0.50x", fund_lower_metrics)
    ]
    
    for label, mults, m in strategies:
        capped_sl_str = f"{m['min_capped_pct']:.0f}/{m['max_capped_pct']:.0f}%"
        vals = [
            label,
            mults,
            str(int(m["n_trades"])),
            f"{m['win_rate_pct']:.2f}%",
            f"{m['profit_factor']:.4f}",
            f"{m['wl_ratio']:.3f}",
            f"{m['sharpe']:.4f}",
            f"{m['max_dd_pct']:.2f}%",
            f"{m['net_roi_pct']:.2f}%",
            capped_sl_str
        ]
        print("".join(v.ljust(w) for v, w in zip(vals, widths)))
    print("=" * 105)

    # 4. Detailed findings
    print("\n[3] KEY FINDINGS:")
    print(f"    1. Technical Trade under Lower Range Theory achieved a Win Rate of {tech_lower_metrics['win_rate_pct']}% with PF = {tech_lower_metrics['profit_factor']}.")
    print(f"       Compared to Swing Theory, the Lower Range Theory improved Technical win rate by {tech_lower_metrics['win_rate_pct'] - tech_swing_metrics['win_rate_pct']:.2f}%.")
    print(f"    2. Fundamental Trade under Lower Range Theory achieved a Win Rate of {fund_lower_metrics['win_rate_pct']}% with PF = {fund_lower_metrics['profit_factor']}.")
    print(f"    3. Note on Capping: Under 0.25x ATR SL, {tech_lower_metrics['min_capped_pct']}% of trades were min-capped (SL < 100 pips), protecting against micro-stops.")

if __name__ == "__main__":
    main()
