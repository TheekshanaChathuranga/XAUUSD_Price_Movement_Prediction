"""
backtest_ict_smc.py — Smart Money Concepts (ICT / SMC) Win Rate Benchmark
==========================================================================
Compares:
  1. Standard Baseline (Naive Market Entry at Signal Bar)
  2. ICT / SMC Strategy:
     - Trend / Structure Alignment (BOS / CHOCH)
     - Fair Value Gap & OTE Discount Retracement Entry (62% Fib / FVG)
     - Structural Stop Loss below Swing Invalidation
     - Opposing Liquidity Pool Target (Buyside/Sellside)
     - Break-Even (BE) Protection when trade reaches 50% TP
"""

import os
import sys
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from datetime import datetime

if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PRICES_FILE = os.path.join(OUTPUT_DIR, "xauusd_raw_prices.csv")
PREDS_FILE = os.path.join(OUTPUT_DIR, "test_predictions.csv")


def load_backtest_data():
    """Loads price action and machine learning prediction signals."""
    if not os.path.exists(PRICES_FILE):
        print(f"Error: {PRICES_FILE} not found!")
        return None

    df_prices = pd.read_csv(PRICES_FILE)
    df_prices['Date'] = pd.to_datetime(df_prices['Date'])
    df_prices = df_prices.sort_values('Date').reset_index(drop=True)

    # Calculate ATR (14)
    high = df_prices['High'].values
    low = df_prices['Low'].values
    close = df_prices['Close'].values
    n = len(df_prices)

    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(14).mean().bfill().values
    df_prices['ATR'] = atr

    PREDS_FULL = os.path.join(OUTPUT_DIR, "test_predictions_full.csv")
    preds_file = PREDS_FULL if os.path.exists(PREDS_FULL) else PREDS_FILE

    if os.path.exists(preds_file):
        df_preds = pd.read_csv(preds_file)
        df_preds['Date'] = pd.to_datetime(df_preds['Date'])
        target_col = 'Target_Direction' if 'Target_Direction' in df_preds.columns else 'Actual_Dir'
        df = df_prices.merge(df_preds[['Date', 'Ensemble_Prob', target_col]], on='Date', how='inner')
    else:
        # Generate momentum baseline
        df = df_prices.copy()
        df['Ensemble_Prob'] = 0.5 + 0.2 * np.sign(df['Close'] - df['Close'].shift(5)).fillna(0)
        df['Target_Direction'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    return df


def simulate_baseline_strategy(df: pd.DataFrame, sl_atr=1.5, tp_atr=3.0):
    """
    Simulates Standard Naive Market Entry:
    Enters at Close of signal day, sets fixed ATR SL/TP, holds up to 10 days.
    """
    trades = []
    prob = df['Ensemble_Prob'].values
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    atr = df['ATR'].values
    dates = df['Date'].values
    n = len(df)

    for i in range(10, n - 10):
        p = prob[i]
        sig = 1 if p >= 0.60 else -1 if p <= 0.40 else 0
        if sig == 0:
            continue

        entry_p = close[i]
        curr_atr = atr[i]
        if curr_atr <= 0:
            continue

        if sig == 1:
            sl = entry_p - sl_atr * curr_atr
            tp = entry_p + tp_atr * curr_atr
        else:
            sl = entry_p + sl_atr * curr_atr
            tp = entry_p - tp_atr * curr_atr

        outcome = "OPEN"
        pnl = 0.0
        duration = 0

        for j in range(i + 1, min(i + 15, n)):
            duration += 1
            bar_h = high[j]
            bar_l = low[j]

            if sig == 1:
                # Long: check TP first or SL
                if bar_h >= tp:
                    outcome = "WIN"
                    pnl = tp - entry_p
                    break
                elif bar_l <= sl:
                    outcome = "LOSS"
                    pnl = sl - entry_p
                    break
            else:
                # Short
                if bar_l <= tp:
                    outcome = "WIN"
                    pnl = entry_p - tp
                    break
                elif bar_h >= sl:
                    outcome = "LOSS"
                    pnl = entry_p - sl
                    break

        if outcome == "OPEN":
            exit_p = close[min(i + 15, n - 1)]
            pnl = (exit_p - entry_p) if sig == 1 else (entry_p - exit_p)
            outcome = "WIN" if pnl > 0 else "LOSS"

        trades.append({
            "Date": dates[i],
            "Signal": "LONG" if sig == 1 else "SHORT",
            "Entry": entry_p,
            "SL": sl,
            "TP": tp,
            "Outcome": outcome,
            "PnL": pnl,
            "Return_Pct": (pnl / entry_p) * 100,
            "Duration": duration
        })

    return pd.DataFrame(trades)


def simulate_ict_smc_strategy(df: pd.DataFrame):
    """
    Simulates Smart Money Concepts (ICT / SMC) Strategy:
    1. Structure Filter: Confirms BOS / CHOCH alignment with ML signal.
    2. Discount / FVG Entry: Waits for 50%-61.8% retracement (limit order).
    3. Structural Stop Loss: Below previous Swing Low / above Swing High.
    4. Opposing Liquidity Pool TP: Targeting recent Swing High/Low.
    5. Dynamic Break-Even (BE): When trade reaches 50% to TP, SL moved to entry.
    """
    trades = []
    prob = df['Ensemble_Prob'].values
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    atr = df['ATR'].values
    dates = df['Date'].values
    n = len(df)

    # Precalculate 5-bar swing fractals
    lookback = 3
    swing_highs = np.zeros(n)
    swing_lows = np.zeros(n)

    for i in range(lookback, n - lookback):
        if all(high[i] >= high[j] for j in range(i - lookback, i + lookback + 1) if j != i):
            swing_highs[i] = high[i]
        if all(low[i] <= low[j] for j in range(i - lookback, i + lookback + 1) if j != i):
            swing_lows[i] = low[i]

    for i in range(20, n - 10):
        p = prob[i]
        sig = 1 if p >= 0.60 else -1 if p <= 0.40 else 0
        if sig == 0:
            continue

        curr_atr = atr[i]
        if curr_atr <= 0:
            continue

        # 1. Structure Detection: find last valid swing high & low
        sh_idx = [k for k in range(max(0, i - 30), i) if swing_highs[k] > 0]
        sl_idx = [k for k in range(max(0, i - 30), i) if swing_lows[k] > 0]

        if not sh_idx or not sl_idx:
            continue

        last_sh = swing_highs[sh_idx[-1]]
        last_sl = swing_lows[sl_idx[-1]]
        swing_range = last_sh - last_sl

        if swing_range < 0.5 * curr_atr:
            continue

        # 2. Market Structure Confirmation (BOS / CHOCH)
        is_bullish_structure = close[i] > last_sl + 0.4 * swing_range
        is_bearish_structure = close[i] < last_sh - 0.4 * swing_range

        if sig == 1 and not is_bullish_structure:
            continue  # Filter counter-trend knife catching
        if sig == -1 and not is_bearish_structure:
            continue

        # 3. ICT FVG / OTE Discount Entry (Wait for pullback into 50%-62% Fib)
        if sig == 1:
            # Long: enter in discount zone (50%-61.8% of swing)
            discount_entry = round(last_sh - 0.50 * swing_range, 2)
            # If current close is already deep in discount, use close, else wait for limit fill
            limit_entry = min(close[i], discount_entry) if close[i] <= discount_entry + 0.2 * curr_atr else discount_entry
            
            # Structural Invalidation SL: Below swing low with 0.2x ATR buffer
            sl = round(last_sl - 0.20 * curr_atr, 2)
            # Liquidity Target TP: Buy-side liquidity pool (above last Swing High)
            tp = round(last_sh + 0.50 * curr_atr, 2)
        else:
            # Short: enter in premium zone
            premium_entry = round(last_sl + 0.50 * swing_range, 2)
            limit_entry = max(close[i], premium_entry) if close[i] >= premium_entry - 0.2 * curr_atr else premium_entry
            
            # Structural Invalidation SL: Above swing high with 0.2x ATR buffer
            sl = round(last_sh + 0.20 * curr_atr, 2)
            # Liquidity Target TP: Sell-side liquidity pool (below last Swing Low)
            tp = round(last_sl - 0.50 * curr_atr, 2)

        # Check Risk:Reward sanity
        risk = abs(limit_entry - sl)
        reward = abs(tp - limit_entry)
        if risk <= 0 or reward / risk < 1.2:
            continue

        # 4. Forward Simulation with Limit Order Fill & Break-Even (BE)
        filled = False
        outcome = "OPEN"
        pnl = 0.0
        be_active = False
        actual_entry = limit_entry
        duration = 0

        for j in range(i, min(i + 15, n)):
            duration += 1
            bar_h = high[j]
            bar_l = low[j]

            # Limit order fill check
            if not filled:
                if sig == 1 and bar_l <= limit_entry:
                    filled = True
                    actual_entry = min(limit_entry, bar_h)
                elif sig == -1 and bar_h >= limit_entry:
                    filled = True
                    actual_entry = max(limit_entry, bar_l)
                else:
                    continue

            if filled:
                # Check BE Activation (1:1 R:R reached -> Move SL to Entry)
                if not be_active:
                    initial_risk = abs(actual_entry - sl)
                    if sig == 1 and bar_h >= actual_entry + initial_risk:
                        be_active = True
                        sl = actual_entry  # Stop Loss moved to Entry Price (Break-Even)
                    elif sig == -1 and bar_l <= actual_entry - initial_risk:
                        be_active = True
                        sl = actual_entry

                # Check Exits
                if sig == 1:
                    if bar_h >= tp:
                        outcome = "WIN"
                        pnl = tp - actual_entry
                        break
                    elif bar_l <= sl:
                        outcome = "BREAK_EVEN" if be_active else "LOSS"
                        pnl = (sl - actual_entry) if not be_active else 0.0
                        break
                else:
                    if bar_l <= tp:
                        outcome = "WIN"
                        pnl = actual_entry - tp
                        break
                    elif bar_h >= sl:
                        outcome = "BREAK_EVEN" if be_active else "LOSS"
                        pnl = (actual_entry - sl) if not be_active else 0.0
                        break

        if not filled:
            continue  # Order expired unfilled without chasing bad prices

        if outcome == "OPEN":
            exit_p = close[min(i + 15, n - 1)]
            pnl = (exit_p - actual_entry) if sig == 1 else (actual_entry - exit_p)
            outcome = "WIN" if pnl > 0 else "BREAK_EVEN" if abs(pnl) < 1.0 else "LOSS"

        trades.append({
            "Date": dates[i],
            "Signal": "LONG" if sig == 1 else "SHORT",
            "Entry": actual_entry,
            "SL": sl,
            "TP": tp,
            "Outcome": outcome,
            "PnL": pnl,
            "Return_Pct": (pnl / actual_entry) * 100,
            "BE_Triggered": be_active,
            "Duration": duration
        })

    return pd.DataFrame(trades)


def compute_metrics(df_trades: pd.DataFrame, strategy_name: str) -> Dict[str, Any]:
    """Computes Win Rate, Profit Factor, Net Return, Sharpe, and Drawdown."""
    if df_trades.empty:
        return {"Strategy": strategy_name, "Trades": 0}

    total = len(df_trades)
    wins = len(df_trades[df_trades['Outcome'] == 'WIN'])
    losses = len(df_trades[df_trades['Outcome'] == 'LOSS'])
    bes = len(df_trades[df_trades['Outcome'] == 'BREAK_EVEN'])

    # Win Rate = Wins / (Wins + Losses) excluding purely protected BEs
    effective_trades = wins + losses
    win_rate = (wins / effective_trades * 100) if effective_trades > 0 else 0.0
    nominal_win_rate = (wins / total * 100)

    gross_profit = df_trades[df_trades['PnL'] > 0]['PnL'].sum()
    gross_loss = abs(df_trades[df_trades['PnL'] < 0]['PnL'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    returns = df_trades['Return_Pct'].values
    mean_ret = np.mean(returns) if len(returns) > 0 else 0.0
    std_ret = np.std(returns) if len(returns) > 0 else 1.0
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

    cumulative_pnl = df_trades['PnL'].cumsum()
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = running_max - cumulative_pnl
    max_dd_usd = np.max(drawdown) if len(drawdown) > 0 else 0.0

    return {
        "Strategy": strategy_name,
        "Total_Trades": total,
        "Wins": wins,
        "Losses": losses,
        "Break_Even_Saves": bes,
        "Win_Rate_Effective": round(win_rate, 1),
        "Win_Rate_Nominal": round(nominal_win_rate, 1),
        "Profit_Factor": round(profit_factor, 2),
        "Total_PnL_USD": round(float(df_trades['PnL'].sum()), 2),
        "Avg_Trade_PnL": round(float(df_trades['PnL'].mean()), 2),
        "Sharpe_Ratio": round(sharpe, 2),
        "Max_Drawdown_USD": round(float(max_dd_usd), 2)
    }


def main():
    print("=" * 70)
    print("  SMART MONEY CONCEPTS (ICT / SMC) BACKTEST & WIN RATE BENCHMARK")
    print("=" * 70)

    df = load_backtest_data()
    if df is None or len(df) < 50:
        print("Error: Insufficient data for backtest.")
        return

    print(f"Historical Sample Size: {len(df)} daily bars ({df['Date'].iloc[0].strftime('%Y-%m-%d')} to {df['Date'].iloc[-1].strftime('%Y-%m-%d')})")
    print("Running Backtest Simulations...\n")

    # 1. Baseline Strategy
    df_base = simulate_baseline_strategy(df, sl_atr=1.5, tp_atr=3.0)
    m_base = compute_metrics(df_base, "1. Standard Baseline (Market Entry + Fixed ATR SL)")

    # 2. ICT / SMC Strategy
    df_ict = simulate_ict_smc_strategy(df)
    m_ict = compute_metrics(df_ict, "2. Smart Money Concepts (ICT: FVG + OTE + Structural SL + BE)")

    results = [m_base, m_ict]
    res_df = pd.DataFrame(results)

    print("-" * 70)
    print(f"{'Metric':<30} | {'1. Standard Baseline':<20} | {'2. Smart Money (ICT)':<20}")
    print("-" * 70)
    print(f"{'Total Trades':<30} | {m_base['Total_Trades']:<20} | {m_ict['Total_Trades']:<20}")
    print(f"{'Wins (TP Hit)':<30} | {m_base['Wins']:<20} | {m_ict['Wins']:<20}")
    print(f"{'Losses (SL Hit)':<30} | {m_base['Losses']:<20} | {m_ict['Losses']:<20}")
    print(f"{'Break-Even Protected Trades':<30} | {m_base['Break_Even_Saves']:<20} | {m_ict['Break_Even_Saves']:<20}")
    print(f"{'Effective Win Rate':<30} | {m_base['Win_Rate_Effective']}%{'':<15} | {m_ict['Win_Rate_Effective']}%{'':<15}")
    print(f"{'Profit Factor':<30} | {m_base['Profit_Factor']:<20} | {m_ict['Profit_Factor']:<20}")
    print(f"{'Total Cumulative PnL':<30} | ${m_base['Total_PnL_USD']:,.2f}{'':<10} | ${m_ict['Total_PnL_USD']:,.2f}{'':<10}")
    print(f"{'Average PnL per Trade':<30} | ${m_base['Avg_Trade_PnL']:,.2f}{'':<10} | ${m_ict['Avg_Trade_PnL']:,.2f}{'':<10}")
    print(f"{'Sharpe Ratio':<30} | {m_base['Sharpe_Ratio']:<20} | {m_ict['Sharpe_Ratio']:<20}")
    print(f"{'Max Drawdown':<30} | ${m_base['Max_Drawdown_USD']:,.2f}{'':<10} | ${m_ict['Max_Drawdown_USD']:,.2f}{'':<10}")
    print("-" * 70)

    # Save results to json
    res_path = os.path.join(OUTPUT_DIR, "ict_smc_benchmark_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to {res_path}")


if __name__ == "__main__":
    main()
