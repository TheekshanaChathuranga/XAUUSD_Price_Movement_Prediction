import os
import sys
import numpy as np
import pandas as pd

if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDS_IN = os.path.join(OUTPUT_DIR, "test_predictions.csv")

def main():
    print("=== Step 1: Loading Predictions ===")
    if not os.path.exists(PREDS_IN):
        print(f"Error: {PREDS_IN} not found!")
        sys.exit(1)
        
    df = pd.read_csv(PREDS_IN)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print("\n=== Step 2: Simulating Real Trading with No-Trade Zone ===")
    initial_capital = 10000.0
    transaction_cost = 0.0001  # 1 basis point or 0.01%
    
    # Dynamic Thresholding
    long_threshold = 0.55
    short_threshold = 0.45
    
    conditions = [
        (df['Ensemble_Prob'] > long_threshold),
        (df['Ensemble_Prob'] < short_threshold)
    ]
    choices = [1, -1]
    df['Signal'] = np.select(conditions, choices, default=0)
    
    # Strategy Return: if signal_t-1 is 1, return is market return. If -1, -market return. If 0, 0.
    df['Strategy_Return_Gross'] = df['Signal'].shift(1) * df['Close_Return']
    
    # Calculate transaction costs ONLY when the signal changes (rebalancing)
    signal_changes = (df['Signal'] != df['Signal'].shift(1)).astype(int)
    # Exclude the very first shift which is always a change from NaN
    signal_changes.iloc[0] = 0
    
    df['Strategy_Return_Net'] = df['Strategy_Return_Gross'] - (signal_changes * transaction_cost)
    
    # Drop first row because of shift
    df = df.dropna().reset_index(drop=True)
    
    # Cumulative Portfolio Value
    df['Cumulative_Market'] = initial_capital * np.exp(df['Close_Return'].cumsum())
    df['Cumulative_Strategy'] = initial_capital * np.exp(df['Strategy_Return_Net'].cumsum())
    
    print("\n=== Step 3: Advanced Performance Metrics ===")
    mean_strat = df['Strategy_Return_Net'].mean()
    std_strat = df['Strategy_Return_Net'].std()
    
    # Sharpe Ratio
    annualized_sharpe = (mean_strat / (std_strat + 1e-8)) * np.sqrt(252)
    
    # Sortino Ratio (only penalize downside volatility)
    downside_returns = df[df['Strategy_Return_Net'] < 0]['Strategy_Return_Net']
    downside_std = downside_returns.std()
    annualized_sortino = (mean_strat / (downside_std + 1e-8)) * np.sqrt(252)
    
    # Max Drawdown
    rolling_max = df['Cumulative_Strategy'].cummax()
    drawdown = df['Cumulative_Strategy'] / rolling_max - 1.0
    max_drawdown = drawdown.min()
    
    # Calmar Ratio
    annualized_return = df['Strategy_Return_Net'].mean() * 252
    calmar_ratio = annualized_return / abs(max_drawdown + 1e-8)
    
    # Win Rate on ACTIVE trading days
    active_days = df[df['Signal'].shift(1) != 0]
    total_active_trades = len(active_days)
    winning_trades = len(active_days[active_days['Strategy_Return_Gross'] > 0])
    
    if total_active_trades > 0:
        win_rate = winning_trades / total_active_trades
    else:
        win_rate = 0.0
        
    time_in_market = (total_active_trades / len(df)) * 100
    
    final_val = df['Cumulative_Strategy'].iloc[-1]
    roi = ((final_val / initial_capital) - 1.0) * 100
    market_roi = ((df['Cumulative_Market'].iloc[-1] / initial_capital) - 1.0) * 100
    
    print("\n" + "="*55)
    print("      PHASE 6: REAL TRADING SIMULATION RESULTS")
    print("="*55)
    print(f"  Initial Capital          : ${initial_capital:,.2f}")
    print(f"  Final Portfolio Value    : ${final_val:,.2f}")
    print(f"  Strategy Net ROI         : {roi:.2f}%")
    print(f"  Buy & Hold Benchmark ROI : {market_roi:.2f}%")
    print("-" * 55)
    print(f"  Time in Market           : {time_in_market:.2f}% (Days Active)")
    print(f"  Win Rate (Active Trades) : {win_rate * 100:.2f}%")
    print(f"  Annualized Sharpe Ratio  : {annualized_sharpe:.4f}")
    print(f"  Annualized Sortino Ratio : {annualized_sortino:.4f}")
    print(f"  Calmar Ratio             : {calmar_ratio:.4f}")
    print(f"  Maximum Drawdown         : {max_drawdown * 100:.2f}%")
    print("="*55)

if __name__ == "__main__":
    main()
