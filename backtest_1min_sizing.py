"""
BACKTEST: Walk-Forward $50-Risk Position Sizing
================================================
Replays backtest_trade_log.csv with the exact $50-risk / 1:2 R:R sizing from
step3b_position_sizing.py and reports dollar-denominated metrics.

This does NOT retrain the model -- it applies the new sizing to the existing
walk-forward trade log to show what the P&L would have been with fixed-risk sizing
instead of the original percentage-based sizing.

Outputs:
  - Console: win rate, profit factor, expectancy ($), max drawdown ($), avg win/loss ($)
  - File: backtest_sizing_results.json
"""

import os, sys, json, math, logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("backtest_sizing")

OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG    = os.path.join(OUTPUT_DIR, "backtest_trade_log.csv")
PRICES_FILE  = os.path.join(OUTPUT_DIR, "xauusd_raw_prices.csv")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "backtest_sizing_results.json")


def run_backtest():
    print("="*65)
    print("  BACKTEST: $50-Risk Fixed Position Sizing")
    print("="*65)

    # ---- Load trade log ----------------------------------------------------
    if not os.path.exists(TRADE_LOG):
        print(f"ERROR: {TRADE_LOG} not found. Run step9_backtest_strategy.py first.")
        sys.exit(1)
    trades = pd.read_csv(TRADE_LOG)
    trades["Date"] = pd.to_datetime(trades["Date"])
    print(f"Loaded {len(trades)} trades from {TRADE_LOG}")

    # ---- Load prices for ATR ------------------------------------------------
    prices = pd.read_csv(PRICES_FILE)
    prices["Date"] = pd.to_datetime(prices["Date"])
    hl  = prices["High"] - prices["Low"]
    hc  = (prices["High"] - prices["Close"].shift()).abs()
    lc  = (prices["Low"]  - prices["Close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    prices["ATR_14"] = tr.rolling(14).mean()
    prices = prices.set_index("Date")

    # ---- Import sizing function --------------------------------------------
    try:
        from step3b_position_sizing import calculate_position, RISK_USD, RR_RATIO, PIP_VALUE_PER_LOT, PIP_SIZE, SPREAD_PIPS, SLIPPAGE_PIPS, MIN_LOT
    except ImportError:
        print("ERROR: step3b_position_sizing.py not found.")
        sys.exit(1)

    # ---- Apply sizing to each trade ----------------------------------------
    results = []
    rejected_count = 0
    for _, row in trades.iterrows():
        date     = row["Date"]
        signal   = row.get("Direction", row.get("Signal", "LONG"))
        win      = int(row.get("Win", 0))
        net_ret  = float(row.get("Net_Return", 0.0))

        # Estimate entry price from prices on that date
        try:
            prow = prices.loc[prices.index.get_indexer([date], method="nearest")[0:1]]
            if prow.empty:
                prow = prices.loc[date:date]
            entry_price = float(prices.loc[prices.index.get_indexer([date], method="nearest")[-1], "Close"])
            atr_val     = float(prices.loc[prices.index.get_indexer([date], method="nearest")[-1], "ATR_14"])
        except Exception:
            # fallback: use average close
            entry_price = float(prices["Close"].mean())
            atr_val     = float(prices["ATR_14"].mean())

        if math.isnan(atr_val) or atr_val <= 0:
            atr_val = 15.0  # sensible default for gold

        # SL distance = 0.75 * ATR (matches step9 default SL_ATR_MULT=0.75)
        sl_atr_mult = 0.75
        sl_dist = atr_val * sl_atr_mult
        if signal == "LONG":
            sl_price = entry_price - sl_dist
        else:
            sl_price = entry_price + sl_dist

        # Apply sizing
        sizing = calculate_position(
            entry_price=entry_price,
            sl_price=sl_price,
            signal=signal,
            spread_pips=SPREAD_PIPS,
            slippage_pips=SLIPPAGE_PIPS,
        )

        if sizing["rejected"]:
            rejected_count += 1
            log.info(f"[{date.date()}] {signal} REJECTED: {sizing['reject_reason']}")
            continue

        lot_size = sizing["lot_size"]
        risk_usd = sizing["risk_usd_realized"]
        tp_usd   = sizing["tp_usd"]

        # Replicate outcome: win -> +tp_usd, loss -> -risk_usd
        # For partial results (net_ret != 1/-1), scale proportionally
        if win == 1:
            pnl = tp_usd
        else:
            pnl = -risk_usd

        # Account for spread cost
        pnl -= sizing["spread_cost_usd"]

        results.append({
            "date":       str(date.date()),
            "signal":     signal,
            "entry":      entry_price,
            "sl":         sl_price,
            "lot_size":   lot_size,
            "risk_usd":   risk_usd,
            "tp_usd":     tp_usd,
            "win":        win,
            "pnl_usd":    round(pnl, 2),
            "spread_cost":sizing["spread_cost_usd"],
        })

    if not results:
        print("ERROR: No valid trades after sizing. Check broker params in step3b.")
        sys.exit(1)

    df = pd.DataFrame(results)
    n        = len(df)
    wins     = df["win"].sum()
    win_rate = wins / n * 100

    pnl        = df["pnl_usd"]
    pos_pnl    = pnl[pnl > 0]
    neg_pnl    = pnl[pnl < 0]

    gross_profit = pos_pnl.sum()
    gross_loss   = abs(neg_pnl.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    expectancy   = pnl.mean()
    avg_win      = pos_pnl.mean() if len(pos_pnl) > 0 else 0.0
    avg_loss     = neg_pnl.mean() if len(neg_pnl) > 0 else 0.0

    # Max drawdown ($)
    cumulative = pnl.cumsum()
    rolling_max = cumulative.cummax()
    drawdown    = cumulative - rolling_max
    max_dd      = abs(drawdown.min())

    # Long vs Short breakdown
    long_t  = df[df["signal"]=="LONG"]
    short_t = df[df["signal"]=="SHORT"]
    long_wr  = long_t["win"].mean()*100 if len(long_t) > 0 else 0.0
    short_wr = short_t["win"].mean()*100 if len(short_t) > 0 else 0.0

    total_return = pnl.sum()

    summary = {
        "total_trades":   n,
        "rejected_trades":rejected_count,
        "win_rate_pct":   round(win_rate, 2),
        "profit_factor":  round(profit_factor, 3),
        "expectancy_usd": round(expectancy, 2),
        "avg_win_usd":    round(avg_win, 2),
        "avg_loss_usd":   round(avg_loss, 2),
        "max_drawdown_usd":round(max_dd, 2),
        "total_return_usd":round(total_return, 2),
        "long_trades":    len(long_t),
        "short_trades":   len(short_t),
        "long_wr_pct":    round(long_wr, 2),
        "short_wr_pct":   round(short_wr, 2),
        "risk_per_trade_usd": RISK_USD,
        "rr_ratio":           RR_RATIO,
        "broker_note": "VERIFY PIP_VALUE_PER_LOT, MIN_LOT, LOT_INCREMENT with actual broker before live trading.",
    }

    # ---- Print results ------------------------------------------------------
    print("\n" + "="*65)
    print("  RESULTS: $50-Risk Fixed Sizing Walk-Forward Backtest")
    print("="*65)
    print(f"  Total trades          : {n:,}  (rejected: {rejected_count})")
    print(f"  Win rate              : {win_rate:.2f}%")
    print(f"  Profit factor         : {profit_factor:.3f}")
    print(f"  Expectancy per trade  : ${expectancy:.2f}")
    print(f"  Average win           : ${avg_win:.2f}")
    print(f"  Average loss          : ${avg_loss:.2f}")
    print(f"  Max drawdown ($)      : ${max_dd:.2f}")
    print(f"  Total return ($)      : ${total_return:.2f}")
    print(f"")
    print(f"  LONG  trades          : {len(long_t)} @ {long_wr:.1f}% WR")
    print(f"  SHORT trades          : {len(short_t)} @ {short_wr:.1f}% WR")
    print("="*65)
    print(f"  *** Risk per trade: ${RISK_USD} | R:R: 1:{RR_RATIO} ***")
    print(f"  *** BROKER FLAGS: Verify pip value / lot sizes before live use ***")
    print("="*65)

    # ---- Save JSON ----------------------------------------------------------
    with open(RESULTS_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    return summary


if __name__ == "__main__":
    run_backtest()
