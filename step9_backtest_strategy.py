"""
Phase 9: Enhanced Backtest Strategy — Production-Grade Evaluation
=================================================================
Bug Fixes:
  ✓ Adaptive percentile thresholds (fixes SHORT signal impossibility)
  ✓ Removed dead code (unused long_days/short_days)
  ✓ Minimum sample guard for high-confidence metrics
  ✓ Separate entry/exit transaction cost modelling
  ✓ Relaxed consensus filter (majority 2/3 instead of unanimous 3/3)
  ✓ Fixed Calmar ratio to use geometric compounding

Enhancements:
  ✓ Percentile-based adaptive LONG/SHORT thresholds
  ✓ Kelly-inspired confidence-weighted position sizing
  ✓ Volatility-scaled returns (risk parity adjustment)
  ✓ Comprehensive metrics: Profit Factor, Expectancy, Avg Win/Loss
  ✓ Multi-regime analysis (Trending Up / Down / Ranging)
  ✓ Monthly return table
  ✓ Equity curve, Drawdown, Rolling Sharpe charts (PNG)
  ✓ Detailed trade log CSV export
"""
import os
import sys
import json
import numpy as np
import pandas as pd

if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDS_IN   = os.path.join(OUTPUT_DIR, "test_predictions.csv")
PRICES_IN  = os.path.join(OUTPUT_DIR, "xauusd_raw_prices.csv")

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
PERCENTILE_LONG  = 70    # top 30% → LONG signal
PERCENTILE_SHORT = 30    # bottom 30% → SHORT signal
CONF_BAND        = 0.65  # high-confidence filter: extreme probabilities
REALISTIC_TC     = 0.0003  # 3 basis points per trade (entry)
EXIT_TC          = 0.0002  # 2 basis points per trade (exit — typically cheaper)
INITIAL_CAPITAL  = 10000.0
MIN_SAMPLE_SIZE  = 10     # minimum trades to report a metric reliably
VOL_LOOKBACK     = 20     # rolling window for volatility scaling
VOL_TARGET       = 0.15   # annualized volatility target (15%)
REGIME_MA_WINDOW = 20     # moving average window for regime detection

# ── VOLUME PROFILE SETTINGS ──────────────────────────────────────────
VP_FILE          = os.path.join(OUTPUT_DIR, "volume_profile_features.csv")
USE_VP_FILTERS   = True    # enable Volume Profile win-rate filters
VP_POC_TOLERANCE = 0.020   # 2.0% tolerance band around POC (relaxed to avoid over-filtering)
VP_LVN_SIZE_MULT = 0.50    # position size multiplier inside LVN zone
VP_VAH_SIZE_MULT = 1.20    # position size boost on VAH/VAL breakout confirmation


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_and_merge_data():
    """Load predictions, merge with price-derived Close_Return and VP features."""
    print("=== Step 1: Loading Predictions ===")
    if not os.path.exists(PREDS_IN):
        print(f"Error: {PREDS_IN} not found!")
        sys.exit(1)

    df = pd.read_csv(PREDS_IN)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"Loaded {len(df):,} prediction rows.")

    # Derive Close_Return from raw price data if missing
    if 'Close_Return' not in df.columns:
        if not os.path.exists(PRICES_IN):
            print(f"Error: Close_Return column missing and {PRICES_IN} not found!")
            sys.exit(1)
        prices = pd.read_csv(PRICES_IN, usecols=['Date', 'Close'])
        prices['Date'] = pd.to_datetime(prices['Date'])
        prices = prices.sort_values('Date').reset_index(drop=True)
        prices['Close_Return'] = np.log(prices['Close'] / prices['Close'].shift(1))
        # Also bring Close for regime analysis
        df = df.merge(prices[['Date', 'Close', 'Close_Return']], on='Date', how='left')
        df = df.dropna(subset=['Close_Return']).reset_index(drop=True)
        print(f"Derived Close_Return from raw prices ({len(df)} rows after merge).")
    elif 'Close' not in df.columns:
        # Need Close for regime analysis even if Close_Return exists
        prices = pd.read_csv(PRICES_IN, usecols=['Date', 'Close'])
        prices['Date'] = pd.to_datetime(prices['Date'])
        df = df.merge(prices[['Date', 'Close']], on='Date', how='left')

    # ── Merge Volume Profile features ───────────────────────────────────
    if USE_VP_FILTERS:
        if os.path.exists(VP_FILE):
            vp_df = pd.read_csv(VP_FILE, usecols=[
                'Date', 'POC_Distance_60', 'VArea_Width_60',
                'Price_vs_VAH_20', 'Price_vs_VAL_20',
                'Price_vs_VAH_60', 'Price_vs_VAL_60',
                'In_LVN_20', 'In_LVN_60',
                'In_HVN_60', 'In_Multi_HVN',
                'VP_Long_Bias_60', 'VP_Short_Bias_60',
                'VP_Confluence_Long', 'VP_Confluence_Short',
                'VAH_Breakout_Strength', 'VAL_Breakdown_Strength',
                'Vol_Imbalance_60', 'In_Any_LVN',
            ])
            vp_df['Date'] = pd.to_datetime(vp_df['Date'])
            df = df.merge(vp_df, on='Date', how='left')
            vp_null = df['POC_Distance_60'].isnull().sum()
            print(f"  Volume Profile features merged. "
                  f"({vp_null} rows without VP data — will use unfiltered signal)")
            # Fill missing VP with neutral values
            vp_cols = [c for c in df.columns if c.startswith(('POC', 'VAH', 'VAL',
                        'VArea', 'Price_vs', 'In_', 'VP_', 'Vol_Imbalance'))]
            df[vp_cols] = df[vp_cols].fillna(0.0)
        else:
            print(f"  [WARN] VP file not found at {VP_FILE}.")
            print("  Run step5b_volume_profile.py first to generate VP features.")
            print("  Continuing WITHOUT Volume Profile filters.")

    return df


def generate_adaptive_signals(df):
    """
    Generate LONG/SHORT/NEUTRAL signals using percentile-based adaptive
    thresholds on Ensemble_Prob. This fixes the critical bug where the
    narrow probability band (0.52–0.66) prevented SHORT signals entirely.
    """
    print("\n=== Step 2: Adaptive Signal Generation ===")

    # Compute data-driven thresholds from the probability distribution
    long_thresh  = np.percentile(df['Ensemble_Prob'], PERCENTILE_LONG)
    short_thresh = np.percentile(df['Ensemble_Prob'], PERCENTILE_SHORT)

    print(f"  Ensemble_Prob range: [{df['Ensemble_Prob'].min():.4f} → {df['Ensemble_Prob'].max():.4f}]")
    print(f"  Adaptive LONG threshold  (P{PERCENTILE_LONG}): {long_thresh:.4f}")
    print(f"  Adaptive SHORT threshold (P{PERCENTILE_SHORT}): {short_thresh:.4f}")

    conditions = [
        (df['Ensemble_Prob'] > long_thresh),
        (df['Ensemble_Prob'] < short_thresh)
    ]
    choices = [1, -1]
    df['Signal_Raw'] = np.select(conditions, choices, default=0)

    raw_long  = (df['Signal_Raw'] == 1).sum()
    raw_short = (df['Signal_Raw'] == -1).sum()
    raw_neut  = (df['Signal_Raw'] == 0).sum()
    print(f"  Raw signals — LONG: {raw_long}  SHORT: {raw_short}  NEUTRAL: {raw_neut}")

    # ── Consensus Filter (relaxed: majority 2/3 instead of unanimous) ─────
    model_cols = ['Cat_Prob', 'XGB_Prob', 'LGB_Prob']
    if all(c in df.columns for c in model_cols):
        models_bullish = sum((df[c] > 0.50).astype(int) for c in model_cols)
        models_bearish = sum((df[c] < 0.50).astype(int) for c in model_cols)

        long_consensus  = models_bullish >= 2   # at least 2/3 agree
        short_consensus = models_bearish >= 2   # at least 2/3 agree

        consensus_ok = np.where(df['Signal_Raw'] == 1, long_consensus,
                       np.where(df['Signal_Raw'] == -1, short_consensus, True))

        n_filtered = (~consensus_ok.astype(bool) & (df['Signal_Raw'] != 0)).sum()
        df['Signal_BT'] = np.where(consensus_ok, df['Signal_Raw'], 0)
        print(f"  Consensus filter (2/3 majority): removed {n_filtered} non-consensus signals.")
    else:
        df['Signal_BT'] = df['Signal_Raw']
        print("  Note: Individual model probs not found — skipping consensus filter.")

    final_long  = (df['Signal_BT'] == 1).sum()
    final_short = (df['Signal_BT'] == -1).sum()
    final_neut  = (df['Signal_BT'] == 0).sum()
    print(f"  Final signals — LONG: {final_long}  SHORT: {final_short}  NEUTRAL: {final_neut}")

    return df, long_thresh, short_thresh


def apply_position_sizing(df):
    """
    Kelly-inspired confidence-weighted position sizing with Volume Profile filters.

    VP Win-Rate Filters applied (when USE_VP_FILTERS=True and VP data available):
      1. POC Alignment Filter  — suppress signals that fight the market fair value
      2. LVN Zone Reducer      — halve size in thin-volume (volatile) price zones
      3. VAH/VAL Breakout Boost — increase size when signal aligns with VP breakout
    """
    print("\n=== Step 2b: Confidence-Weighted Position Sizing + VP Filters ===")

    prob = df['Ensemble_Prob'].values
    median_prob = np.median(prob)

    # Normalize edge: how far prob is from the median, scaled to [0, 1]
    prob_range = max(prob.max() - prob.min(), 1e-8)
    edge = np.abs(prob - median_prob) / (prob_range / 2)
    edge = np.clip(edge, 0.0, 1.0)

    # Scale: minimum 0.3x, maximum 1.0x
    position_scale = 0.3 + 0.7 * edge

    df['Position_Size_Base'] = df['Signal_BT'] * position_scale
    df['Position_Size']      = df['Position_Size_Base'].copy()
    df['VP_Filter_Applied']  = 0   # 0=none, 1=poc, 2=lvn, 3=vah_boost, -1=suppressed

    # Check whether VP columns are available
    has_vp = USE_VP_FILTERS and ('POC_Distance_60' in df.columns)

    if has_vp:
        poc_dist   = df['POC_Distance_60'].values
        in_any_lvn = df['In_Any_LVN'].values if 'In_Any_LVN' in df.columns \
                     else df.get('In_LVN_60', pd.Series(0, index=df.index)).values
        vah_break  = df['VAH_Breakout_Strength'].values if 'VAH_Breakout_Strength' in df.columns \
                     else np.zeros(len(df))
        val_break  = df['VAL_Breakdown_Strength'].values if 'VAL_Breakdown_Strength' in df.columns \
                     else np.zeros(len(df))
        signal     = df['Signal_BT'].values

        n_poc_suppressed = 0
        n_lvn_reduced    = 0
        n_vah_boosted    = 0

        for i in range(len(df)):
            if signal[i] == 0:
                continue

            current_size = df.at[df.index[i], 'Position_Size']

            # ── Filter 1: POC Alignment ───────────────────────────────────
            # Suppress LONG signal when price is significantly ABOVE POC
            # (already expensive vs. market consensus — poor risk/reward)
            # Suppress SHORT signal when price is significantly BELOW POC
            poc_threshold = VP_POC_TOLERANCE
            if signal[i] == 1 and poc_dist[i] > poc_threshold:
                # Price above POC → LONG signal fights fair value → suppress
                df.at[df.index[i], 'Position_Size']     = 0.0
                df.at[df.index[i], 'VP_Filter_Applied'] = -1
                n_poc_suppressed += 1
                continue
            elif signal[i] == -1 and poc_dist[i] < -poc_threshold:
                # Price below POC → SHORT signal fights fair value → suppress
                df.at[df.index[i], 'Position_Size']     = 0.0
                df.at[df.index[i], 'VP_Filter_Applied'] = -1
                n_poc_suppressed += 1
                continue

            # ── Filter 2: LVN Zone — reduce size ─────────────────────────
            # Thin-volume zones have erratic price movement → halve position
            if in_any_lvn[i] == 1:
                df.at[df.index[i], 'Position_Size']     = current_size * VP_LVN_SIZE_MULT
                df.at[df.index[i], 'VP_Filter_Applied'] = 2
                n_lvn_reduced += 1
                current_size = df.at[df.index[i], 'Position_Size']

            # ── Filter 3: VAH/VAL Breakout Confirmation — boost size ───────
            # Price breaks VAH on a LONG, or breaks VAL on a SHORT → stronger signal
            if signal[i] == 1 and vah_break[i] == 1:
                df.at[df.index[i], 'Position_Size']     = min(
                    abs(current_size) * VP_VAH_SIZE_MULT, 1.0) * np.sign(current_size)
                df.at[df.index[i], 'VP_Filter_Applied'] = 3
                n_vah_boosted += 1
            elif signal[i] == -1 and val_break[i] == 1:
                df.at[df.index[i], 'Position_Size']     = min(
                    abs(current_size) * VP_VAH_SIZE_MULT, 1.0) * np.sign(current_size)
                df.at[df.index[i], 'VP_Filter_Applied'] = 3
                n_vah_boosted += 1

        pct_suppressed = n_poc_suppressed / max((df['Signal_BT'] != 0).sum(), 1) * 100
        pct_lvn        = n_lvn_reduced    / max((df['Signal_BT'] != 0).sum(), 1) * 100
        pct_boosted    = n_vah_boosted    / max((df['Signal_BT'] != 0).sum(), 1) * 100

        print(f"  VP Filter 1 — POC Alignment suppressed : {n_poc_suppressed} signals ({pct_suppressed:.1f}%)")
        print(f"  VP Filter 2 — LVN zone size halved     : {n_lvn_reduced} signals ({pct_lvn:.1f}%)")
        print(f"  VP Filter 3 — VAH/VAL breakout boosted : {n_vah_boosted} signals ({pct_boosted:.1f}%)")
    else:
        if USE_VP_FILTERS:
            print("  [INFO] VP filters requested but VP data not found — using base sizing only.")
        else:
            print("  VP filters disabled (USE_VP_FILTERS=False).")

    active_sizes = df.loc[df['Position_Size'] != 0, 'Position_Size'].abs()
    if len(active_sizes) > 0:
        print(f"  Position sizes — mean: {active_sizes.mean():.3f}  "
              f"min: {active_sizes.min():.3f}  "
              f"max: {active_sizes.max():.3f}")

    return df


def simulate_trading(df):
    """
    Simulate trading with:
    - Position sizing (from Step 2b)
    - Separate entry/exit transaction costs
    - Volatility scaling (risk parity)
    """
    tc_label = f"{REALISTIC_TC*10000:.0f}bp entry / {EXIT_TC*10000:.0f}bp exit"
    print(f"\n=== Step 3: Simulating Trading ({tc_label}) ===")

    # Strategy Return: position at t-1 × return at t (no look-ahead)
    df['Strategy_Return_Gross'] = df['Position_Size'].shift(1) * df['Close_Return']

    # ── Transaction costs: separate entry and exit costs ──────────────────
    prev_signal = df['Signal_BT'].shift(1).fillna(0)
    curr_signal = df['Signal_BT']

    # Entry cost: going from flat (0) to a position (±1)
    is_entry = (prev_signal == 0) & (curr_signal != 0)
    # Exit cost: going from a position (±1) to flat (0)
    is_exit = (prev_signal != 0) & (curr_signal == 0)
    # Reversal: going from LONG to SHORT or vice versa (both entry + exit)
    is_reversal = (prev_signal != 0) & (curr_signal != 0) & (prev_signal != curr_signal)

    tc_cost = (is_entry.astype(float) * REALISTIC_TC +
               is_exit.astype(float) * EXIT_TC +
               is_reversal.astype(float) * (REALISTIC_TC + EXIT_TC))

    df['TC_Cost'] = tc_cost
    df['Strategy_Return_Net'] = df['Strategy_Return_Gross'] - tc_cost

    # ── Volatility Scaling (Risk Parity) ──────────────────────────────────
    df['RealizedVol'] = df['Close_Return'].rolling(VOL_LOOKBACK, min_periods=5).std() * np.sqrt(252)
    df['VolScale'] = VOL_TARGET / df['RealizedVol'].clip(lower=0.01)
    df['VolScale'] = df['VolScale'].clip(upper=2.0)  # cap at 2x leverage

    df['Strategy_Return_VolAdj'] = df['Strategy_Return_Net'] * df['VolScale'].shift(1).fillna(1.0)

    # Drop first row (shift creates NaN)
    df = df.dropna(subset=['Strategy_Return_Gross']).reset_index(drop=True)

    # Cumulative Portfolio Values
    df['Cumulative_Market']   = INITIAL_CAPITAL * np.exp(df['Close_Return'].cumsum())
    df['Cumulative_Strategy'] = INITIAL_CAPITAL * np.exp(df['Strategy_Return_Net'].cumsum())
    df['Cumulative_VolAdj']   = INITIAL_CAPITAL * np.exp(df['Strategy_Return_VolAdj'].cumsum())

    total_tc = df['TC_Cost'].sum() * 10000
    print(f"  Total transaction costs: {total_tc:.1f} bps cumulative")

    return df


def compute_metrics(returns, label="Strategy"):
    """Compute comprehensive performance metrics for a return series."""
    mean_r = returns.mean()
    std_r  = returns.std()

    # Annualized Sharpe
    sharpe = (mean_r / (std_r + 1e-8)) * np.sqrt(252)

    # Sortino (downside deviation only)
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 1e-8
    sortino = (mean_r / (downside_std + 1e-8)) * np.sqrt(252)

    # Geometric annualized return (correct for log returns)
    cum_return = returns.sum()  # sum of log returns = log of cumulative
    n_years = len(returns) / 252
    ann_return = cum_return / n_years if n_years > 0 else 0.0

    # Max Drawdown (from cumulative curve)
    cumulative = np.exp(returns.cumsum())
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative / rolling_max - 1.0
    max_dd = drawdown.min()

    # Calmar Ratio (geometric return / max drawdown)
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-8 else 0.0

    # Profit Factor
    gross_profit = returns[returns > 0].sum()
    gross_loss   = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-8 else float('inf')

    # Average Win / Average Loss
    wins  = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win  = wins.mean()  if len(wins)  > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0

    # Expectancy per trade
    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    return {
        'sharpe': sharpe,
        'sortino': sortino,
        'ann_return': ann_return,
        'max_dd': max_dd,
        'calmar': calmar,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_rate': win_rate,
        'expectancy': expectancy,
        'total_return_pct': (np.exp(cum_return) - 1) * 100,
        'n_trades': len(returns),
    }


def analyze_performance(df):
    """Comprehensive performance analysis with all metrics."""
    print("\n=== Step 4: Advanced Performance Metrics ===")

    # Identify active trading days (previous signal was non-zero)
    df['Prev_Signal'] = df['Signal_BT'].shift(1).fillna(0)
    active_mask = df['Prev_Signal'] != 0
    active_days = df[active_mask]
    total_active = len(active_days)

    # ── Core Metrics (Net Returns) ────────────────────────────────────────
    net_metrics = compute_metrics(df['Strategy_Return_Net'], "Net Strategy")
    vol_metrics = compute_metrics(df['Strategy_Return_VolAdj'], "Vol-Adjusted")

    # ── Win Rates by Direction ────────────────────────────────────────────
    long_trades  = df[df['Prev_Signal'] == 1]
    short_trades = df[df['Prev_Signal'] == -1]

    long_wins  = (long_trades['Strategy_Return_Gross'] > 0).sum() if len(long_trades) > 0 else 0
    short_wins = (short_trades['Strategy_Return_Gross'] > 0).sum() if len(short_trades) > 0 else 0

    long_wr  = long_wins / len(long_trades) if len(long_trades) > 0 else 0.0
    short_wr = short_wins / len(short_trades) if len(short_trades) > 0 else 0.0

    # ── Active trading metrics ────────────────────────────────────────────
    active_wins = (active_days['Strategy_Return_Gross'] > 0).sum()
    overall_wr  = active_wins / total_active if total_active > 0 else 0.0

    # ── High-Confidence Subset ────────────────────────────────────────────
    hc_mask = (df['Ensemble_Prob'] > CONF_BAND) | (df['Ensemble_Prob'] < (1 - CONF_BAND))
    hc_active = df[hc_mask & active_mask]
    hc_total  = len(hc_active)
    hc_wins   = (hc_active['Strategy_Return_Gross'] > 0).sum() if hc_total > 0 else 0
    hc_wr     = hc_wins / hc_total if hc_total >= MIN_SAMPLE_SIZE else None

    time_in_market = (total_active / len(df)) * 100 if len(df) > 0 else 0.0

    final_val     = df['Cumulative_Strategy'].iloc[-1]
    final_vol_adj = df['Cumulative_VolAdj'].iloc[-1]
    roi           = ((final_val / INITIAL_CAPITAL) - 1.0) * 100
    roi_vol       = ((final_vol_adj / INITIAL_CAPITAL) - 1.0) * 100
    market_roi    = ((df['Cumulative_Market'].iloc[-1] / INITIAL_CAPITAL) - 1.0) * 100

    # ── Print Results ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("      PHASE 9: ENHANCED BACKTEST SIMULATION RESULTS")
    print("=" * 65)

    print(f"\n  {'─── PORTFOLIO SUMMARY ───':^60}")
    print(f"  Initial Capital              : ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final Value (Net)            : ${final_val:,.2f}")
    print(f"  Final Value (Vol-Adjusted)   : ${final_vol_adj:,.2f}")
    print(f"  Strategy Net ROI             : {roi:+.2f}%")
    print(f"  Strategy Vol-Adj ROI         : {roi_vol:+.2f}%")
    print(f"  Buy & Hold Benchmark ROI     : {market_roi:+.2f}%")

    print(f"\n  {'─── TRADE STATISTICS ───':^60}")
    print(f"  Time in Market               : {time_in_market:.1f}%")
    print(f"  Total Active Trades          : {total_active}")
    print(f"  LONG  Trades                 : {len(long_trades)}  (Win Rate: {long_wr*100:.1f}%)")
    print(f"  SHORT Trades                 : {len(short_trades)}  (Win Rate: {short_wr*100:.1f}%)")
    print(f"  Overall Win Rate             : {overall_wr*100:.1f}%")
    if hc_wr is not None:
        print(f"  High-Conf ({CONF_BAND*100:.0f}%+) Win Rate  : {hc_wr*100:.1f}%  ({hc_total} trades)")
    else:
        print(f"  High-Conf ({CONF_BAND*100:.0f}%+) Win Rate  : N/A (< {MIN_SAMPLE_SIZE} trades)")

    print(f"\n  {'─── NET STRATEGY METRICS ───':^60}")
    print(f"  Annualized Sharpe Ratio      : {net_metrics['sharpe']:+.4f}")
    print(f"  Annualized Sortino Ratio     : {net_metrics['sortino']:+.4f}")
    print(f"  Calmar Ratio                 : {net_metrics['calmar']:+.4f}")
    print(f"  Maximum Drawdown             : {net_metrics['max_dd']*100:.2f}%")
    print(f"  Profit Factor                : {net_metrics['profit_factor']:.3f}")
    print(f"  Avg Win / Avg Loss           : {net_metrics['avg_win']*10000:.2f}bp / {net_metrics['avg_loss']*10000:.2f}bp")
    print(f"  Expectancy per Trade         : {net_metrics['expectancy']*10000:.2f}bp")

    print(f"\n  {'─── VOL-ADJUSTED METRICS ───':^60}")
    print(f"  Annualized Sharpe (Vol-Adj)  : {vol_metrics['sharpe']:+.4f}")
    print(f"  Annualized Sortino (Vol-Adj) : {vol_metrics['sortino']:+.4f}")
    print(f"  Max Drawdown (Vol-Adj)       : {vol_metrics['max_dd']*100:.2f}%")
    print(f"  Profit Factor (Vol-Adj)      : {vol_metrics['profit_factor']:.3f}")

    print(f"\n  {'─── TRANSACTION COSTS ───':^60}")
    print(f"  Entry Cost                   : {REALISTIC_TC*10000:.0f} bps")
    print(f"  Exit Cost                    : {EXIT_TC*10000:.0f} bps")
    print(f"  Total TC Paid                : {df['TC_Cost'].sum()*10000:.1f} bps cumulative")
    print("=" * 65)

    # ── VP Win-Rate Comparison ─────────────────────────────────────────
    if USE_VP_FILTERS and 'VP_Filter_Applied' in df.columns:
        # Baseline: all signals that would have fired without VP filtering
        # (VP_Filter_Applied != -1 means the signal was NOT suppressed)
        non_suppressed = df[(df['Prev_Signal'] != 0) & (df['VP_Filter_Applied'] != -1)]
        suppressed     = df[(df['VP_Filter_Applied'] == -1)]

        print(f"\n  {'─── VOLUME PROFILE FILTER IMPACT ───':^60}")
        if len(non_suppressed) > 0:
            ns_wr = (non_suppressed['Strategy_Return_Gross'] > 0).sum() / len(non_suppressed)
            print(f"  VP-Passed Trades Win Rate       : {ns_wr*100:.1f}%  ({len(non_suppressed)} trades)")
        if len(suppressed) > 0:
            sup_wr = (suppressed['Strategy_Return_Gross'] > 0).sum() / len(suppressed) \
                     if len(suppressed) > 0 else 0
            print(f"  VP-Suppressed Trade Win Rate    : {sup_wr*100:.1f}%  ({len(suppressed)} trades) "
                  f"[would-have-been-taken]")

        # VAH/VAL Boosted trades
        boosted = df[(df['Prev_Signal'] != 0) & (df['VP_Filter_Applied'] == 3)]
        if len(boosted) >= MIN_SAMPLE_SIZE:
            boost_wr = (boosted['Strategy_Return_Gross'] > 0).sum() / len(boosted)
            print(f"  VAH/VAL Breakout Trade Win Rate : {boost_wr*100:.1f}%  ({len(boosted)} trades)")

        # LVN zone trades (after size reduction)
        lvn_trades = df[(df['Prev_Signal'] != 0) & (df['VP_Filter_Applied'] == 2)]
        if len(lvn_trades) >= MIN_SAMPLE_SIZE:
            lvn_wr = (lvn_trades['Strategy_Return_Gross'] > 0).sum() / len(lvn_trades)
            print(f"  LVN Zone Trade Win Rate         : {lvn_wr*100:.1f}%  ({len(lvn_trades)} trades) "
                  f"[size halved]")

        print(f"  Overall Win Rate (all active)   : {overall_wr*100:.1f}%  ({total_active} trades)")
        print(f"  [NOTE] Run with USE_VP_FILTERS=False to see baseline for comparison.")

    return net_metrics, vol_metrics


def analyze_regimes(df):
    """Break down performance by market regime."""
    print("\n=== Step 5: Market Regime Analysis ===")

    if 'Close' not in df.columns:
        print("  Skipping regime analysis — Close price not available.")
        return

    # Compute regime indicators
    df['MA20'] = df['Close'].rolling(REGIME_MA_WINDOW, min_periods=5).mean()
    df['MA_Slope'] = df['MA20'].diff(5) / df['MA20'].shift(5)  # 5-day slope

    # Regime classification
    df['Regime'] = np.where(df['MA_Slope'] > 0.005, 'Trending Up',
                   np.where(df['MA_Slope'] < -0.005, 'Trending Down', 'Ranging'))

    active_mask = df['Prev_Signal'] != 0

    print(f"\n  {'Regime':<18} {'Days':>6} {'Active':>7} {'Win%':>7} {'Avg Ret':>10} {'Sharpe':>8}")
    print("  " + "-" * 58)

    for regime in ['Trending Up', 'Trending Down', 'Ranging']:
        regime_mask = df['Regime'] == regime
        regime_active = df[regime_mask & active_mask]

        n_days   = regime_mask.sum()
        n_active = len(regime_active)
        if n_active >= MIN_SAMPLE_SIZE:
            wr = (regime_active['Strategy_Return_Gross'] > 0).sum() / n_active
            avg_ret = regime_active['Strategy_Return_Net'].mean() * 10000
            std_ret = regime_active['Strategy_Return_Net'].std()
            regime_sharpe = (regime_active['Strategy_Return_Net'].mean() / (std_ret + 1e-8)) * np.sqrt(252)
            print(f"  {regime:<18} {n_days:>6} {n_active:>7} {wr*100:>6.1f}% {avg_ret:>+9.2f}bp {regime_sharpe:>+7.3f}")
        else:
            print(f"  {regime:<18} {n_days:>6} {n_active:>7} {'N/A':>7} {'N/A':>10} {'N/A':>8}")


def monthly_returns_table(df):
    """Display monthly return breakdown."""
    print("\n=== Step 6: Monthly Returns Table ===")

    df_temp = df.copy()
    df_temp['YearMonth'] = df_temp['Date'].dt.to_period('M')

    monthly = df_temp.groupby('YearMonth').agg(
        Strategy=('Strategy_Return_Net', 'sum'),
        Market=('Close_Return', 'sum'),
        N_Trades=('Prev_Signal', lambda x: (x != 0).sum()),
    )
    monthly['Strategy_Pct'] = (np.exp(monthly['Strategy']) - 1) * 100
    monthly['Market_Pct']   = (np.exp(monthly['Market']) - 1) * 100
    monthly['Alpha']        = monthly['Strategy_Pct'] - monthly['Market_Pct']

    print(f"\n  {'Month':<12} {'Strategy':>10} {'Market':>10} {'Alpha':>10} {'Trades':>8}")
    print("  " + "-" * 52)
    for idx, row in monthly.iterrows():
        print(f"  {str(idx):<12} {row['Strategy_Pct']:>+9.2f}% {row['Market_Pct']:>+9.2f}% "
              f"{row['Alpha']:>+9.2f}% {row['N_Trades']:>7.0f}")

    print("  " + "-" * 52)
    total_strat = monthly['Strategy_Pct'].sum()
    total_mkt   = monthly['Market_Pct'].sum()
    print(f"  {'TOTAL':<12} {total_strat:>+9.2f}% {total_mkt:>+9.2f}% "
          f"{total_strat - total_mkt:>+9.2f}% {monthly['N_Trades'].sum():>7.0f}")


def export_trade_log(df):
    """Export detailed trade log to CSV (excludes VP-suppressed zero-size positions)."""
    print("\n=== Step 7: Exporting Trade Log ===")

    # Only include rows where a real position was held (excludes VP-suppressed signals)
    active_mask = (df['Prev_Signal'] != 0) & (df['Position_Size'] != 0)
    trade_log = df[active_mask][['Date', 'Prev_Signal', 'Position_Size', 'Ensemble_Prob',
                                  'Close_Return', 'Strategy_Return_Gross',
                                  'Strategy_Return_Net', 'TC_Cost',
                                  'Cumulative_Strategy']].copy()

    trade_log.rename(columns={
        'Prev_Signal': 'Direction',
        'Position_Size': 'Pos_Size',
        'Ensemble_Prob': 'Probability',
        'Close_Return': 'Market_Return',
        'Strategy_Return_Gross': 'Gross_Return',
        'Strategy_Return_Net': 'Net_Return',
        'TC_Cost': 'Trans_Cost',
        'Cumulative_Strategy': 'Cumulative_Value',
    }, inplace=True)

    trade_log['Direction'] = trade_log['Direction'].map({1: 'LONG', -1: 'SHORT'})
    trade_log['Win'] = (trade_log['Gross_Return'] > 0).astype(int)

    out_path = os.path.join(OUTPUT_DIR, "backtest_trade_log.csv")
    trade_log.to_csv(out_path, index=False, float_format='%.6f')
    print(f"  Exported {len(trade_log)} trades → {out_path}")

    return trade_log


def generate_charts(df):
    """Generate equity curve, drawdown, and rolling Sharpe charts."""
    print("\n=== Step 8: Generating Charts ===")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("  matplotlib not installed — skipping charts.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1.5, 1.5]})
    fig.suptitle('XAUUSD Backtest — Enhanced Strategy Performance', fontsize=14, fontweight='bold')

    dates = pd.to_datetime(df['Date'])

    # ── Panel 1: Equity Curves ────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(dates, df['Cumulative_Market'],   label='Buy & Hold', color='#888888',
             linewidth=1.5, alpha=0.7)
    ax1.plot(dates, df['Cumulative_Strategy'], label='Strategy (Net)', color='#2196F3',
             linewidth=2)
    ax1.plot(dates, df['Cumulative_VolAdj'],   label='Strategy (Vol-Adjusted)', color='#FF9800',
             linewidth=2, linestyle='--')

    # Shade LONG and SHORT periods
    for i in range(1, len(df)):
        if df['Prev_Signal'].iloc[i] == 1:
            ax1.axvspan(dates.iloc[i-1], dates.iloc[i], alpha=0.04, color='green')
        elif df['Prev_Signal'].iloc[i] == -1:
            ax1.axvspan(dates.iloc[i-1], dates.iloc[i], alpha=0.04, color='red')

    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Equity Curves')

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────
    ax2 = axes[1]
    rolling_max = df['Cumulative_Strategy'].cummax()
    drawdown = (df['Cumulative_Strategy'] / rolling_max - 1.0) * 100
    ax2.fill_between(dates, drawdown, 0, color='#F44336', alpha=0.4)
    ax2.plot(dates, drawdown, color='#D32F2F', linewidth=1)
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Strategy Drawdown')

    # ── Panel 3: Rolling Sharpe ───────────────────────────────────────────
    ax3 = axes[2]
    rolling_mean = df['Strategy_Return_Net'].rolling(60, min_periods=20).mean()
    rolling_std  = df['Strategy_Return_Net'].rolling(60, min_periods=20).std()
    rolling_sharpe = (rolling_mean / (rolling_std + 1e-8)) * np.sqrt(252)

    ax3.plot(dates, rolling_sharpe, color='#4CAF50', linewidth=1.5)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(y=1, color='green', linestyle=':', alpha=0.4, label='Sharpe = 1')
    ax3.axhline(y=-1, color='red', linestyle=':', alpha=0.4, label='Sharpe = -1')
    ax3.fill_between(dates, rolling_sharpe, 0,
                     where=(rolling_sharpe > 0), color='green', alpha=0.1)
    ax3.fill_between(dates, rolling_sharpe, 0,
                     where=(rolling_sharpe < 0), color='red', alpha=0.1)
    ax3.set_ylabel('60-Day Rolling Sharpe')
    ax3.set_xlabel('Date')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Rolling Sharpe Ratio (60-Day Window)')

    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "backtest_performance.png")
    fig.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Charts saved → {chart_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Step 1: Load & merge data
    df = load_and_merge_data()

    # Step 2: Adaptive signal generation
    df, long_thresh, short_thresh = generate_adaptive_signals(df)

    # Step 2b: Position sizing
    df = apply_position_sizing(df)

    # Step 3: Simulate trading
    df = simulate_trading(df)

    # Step 4: Performance analysis
    net_metrics, vol_metrics = analyze_performance(df)

    # Step 5: Regime analysis
    analyze_regimes(df)

    # Step 6: Monthly returns
    monthly_returns_table(df)

    # Step 7: Trade log
    export_trade_log(df)

    # Step 8: Charts
    generate_charts(df)

    # ── Save backtest config for reproducibility ──────────────────────────
    config = {
        'percentile_long': PERCENTILE_LONG,
        'percentile_short': PERCENTILE_SHORT,
        'long_threshold_actual': float(long_thresh),
        'short_threshold_actual': float(short_thresh),
        'confidence_band': CONF_BAND,
        'entry_tc_bps': REALISTIC_TC * 10000,
        'exit_tc_bps': EXIT_TC * 10000,
        'vol_target': VOL_TARGET,
        'vol_lookback': VOL_LOOKBACK,
        'initial_capital': INITIAL_CAPITAL,
        'net_sharpe': float(net_metrics['sharpe']),
        'net_roi_pct': float(net_metrics['total_return_pct']),
        'vol_adj_sharpe': float(vol_metrics['sharpe']),
        'vol_adj_roi_pct': float(vol_metrics['total_return_pct']),
    }
    config_path = os.path.join(OUTPUT_DIR, "backtest_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n  Config saved → {config_path}")

    print("\n✅ Phase 9 Enhanced Backtest Complete.")


if __name__ == "__main__":
    main()
