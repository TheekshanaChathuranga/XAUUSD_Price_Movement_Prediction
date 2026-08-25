import os
import sys
import re
import numpy as np
import pandas as pd

# Fix Unicode encoding on Windows
if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR  = os.path.dirname(os.path.abspath(__file__))
FEATURES_IN = os.path.join(OUTPUT_DIR, "master_features.csv")
DATASET_OUT = os.path.join(OUTPUT_DIR, "multimodal_master_dataset.csv")

# ── WIN-RATE UPGRADE CONSTANTS ────────────────────────────────────────────────
# SD-based dynamic labeling (Dakalbab et al.)
# Threshold multiplier on rolling std — 0.5σ captures significant daily moves
SD_WINDOW     = 20     # rolling window matching volatility gate
SD_MULTIPLIER = 0.5    # 0.5σ threshold — calibrate empirically if desired

# Priority macro features with full 1–5 day lag structure (Chai et al.)
# Crude oil ≈89% of gold forecast error variance; VIX 4-7.5%; DXY only 0.5%
PRIORITY_MACRO  = ['WTI_Crude_Oil', 'VIX_Index']
STANDARD_MACRO  = [
    'CPI_US', 'FedFunds_Rate', 'Unemployment_Rate', 'NFP_Change',
    'PCE_Deflator', 'US_10Y_Yield', 'Real_GDP_Growth',
    'M2_Money_Supply', 'DXY_Index'
]


def find_bb_columns(df_cols):
    """
    Auto-detect Bollinger Band column names regardless of pandas_ta version.
    Returns (bbl, bbm, bbu, bbb, bbp) — any may be None if not found.
    """
    col_list = list(df_cols)
    bbl = bbm = bbu = bbb = bbp = None
    for c in col_list:
        cl = c.upper()
        if re.match(r'BBL', cl):  bbl = c
        elif re.match(r'BBM', cl): bbm = c
        elif re.match(r'BBU', cl): bbu = c
        elif re.match(r'BBB', cl): bbb = c
        elif re.match(r'BBP', cl): bbp = c
    return bbl, bbm, bbu, bbb, bbp


def main():
    print("=== Step 1: Loading Preprocessed Master Features ===")
    if not os.path.exists(FEATURES_IN):
        print(f"Error: Preprocessed features file {FEATURES_IN} not found!")
        print("Please run step5_preprocess_features.py first.")
        sys.exit(1)

    df = pd.read_csv(FEATURES_IN)
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"Loaded master features dataset: {df.shape[0]:,} rows × {df.shape[1]} columns.")

    print("\n=== Step 2: Handling Gaps and Missing Values ===")
    news_cols = [
        'Mean_Sentiment', 'Sentiment_Dispersion', 'News_Volume',
        'Sentiment_Lag_1', 'Sentiment_Lag_3',
        'Sentiment_SMA_5', 'Sentiment_SMA_10', 'Sentiment_SMA_20'
    ]
    existing_news_cols = [col for col in news_cols if col in df.columns]
    df[existing_news_cols] = df[existing_news_cols].fillna(0)
    print(f"Filled missing values in news columns: {existing_news_cols}")

    # Forward-fill remaining missing macro or price data
    # Do NOT apply backward-fill (bfill) here — would introduce look-ahead bias
    df = df.ffill()
    print("Applied forward-fill to resolve any remaining gaps.")

    print("\n=== Step 3: Stationary Transformations (Log Returns & Differencing) ===")
    # A. Convert absolute price levels to log returns
    for col in ['Open', 'High', 'Low', 'Close']:
        df[f'{col}_Return'] = np.log(df[col] / df[col].shift(1))

    # B. Compute ATR (Average True Range) for volatility regime feature — BEFORE dropping price cols
    hl = df['High'] - df['Low']
    hc = np.abs(df['High'] - df['Close'].shift(1))
    lc = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()

    # B2. Volatility Regime: rolling std of Close_Return vs 90th percentile
    df['Return_Vol_20'] = df['Close_Return'].rolling(20).std()
    vol_90pct = df['Return_Vol_20'].quantile(0.90)
    df['High_Vol_Regime'] = (df['Return_Vol_20'] > vol_90pct).astype(int)

    # B3. Convert non-stationary indicators (EMA, BBands levels) to Close price ratios
    df['EMA_50_Ratio'] = df['EMA_50'] / df['Close']

    # Auto-detect Bollinger Band columns (handles different pandas_ta versions)
    bbl, bbm, bbu, bbb, bbp = find_bb_columns(df.columns)
    if bbl: df['BBL_Ratio'] = df[bbl] / df['Close']
    if bbm: df['BBM_Ratio'] = df[bbm] / df['Close']
    if bbu: df['BBU_Ratio'] = df[bbu] / df['Close']

    # BB Width: (upper - lower) / middle — measures band squeeze/expansion
    if bbl and bbu and bbm:
        df['BB_Width'] = (df[bbu] - df[bbl]) / (df[bbm] + 1e-9)

    # Drop original absolute price columns + raw BB level columns
    cols_to_drop = ['Open', 'High', 'Low', 'EMA_50']
    for bb_col in [bbl, bbm, bbu, bbb, bbp]:
        if bb_col and bb_col in df.columns:
            cols_to_drop.append(bb_col)
            
    # B4. Convert absolute Volume Profile & Order Flow levels to Close price ratios
    vp_abs_levels = [c for c in df.columns if c.startswith(('POC_', 'VAH_', 'VAL_', 'VWAP_')) and c.count('_') == 1]
    for col in vp_abs_levels:
        df[f'{col}_Ratio'] = df[col] / df['Close']
        cols_to_drop.append(col)

    # Only drop columns that actually exist
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped raw price/BB level/VP absolute columns: {cols_to_drop}")

    # C. Apply first-order differencing to macroeconomic levels (and their lags)
    # ── Priority macro: WTI_Crude_Oil + VIX_Index get full 1-5 day lag structure ──
    # Chai et al. SVAR: WTI ~89% of gold forecast error variance;
    # VIX contribution grows 4%→7.5% over forecast horizon.
    # Full 5-day lag window matches the empirical macro digestion window.
    for base_col in PRIORITY_MACRO:
        if base_col in df.columns:
            df[f'{base_col}_Diff'] = df[base_col].diff()
            # Return (log-diff) for momentum
            df[f'{base_col}_Return'] = np.log(
                df[base_col].clip(lower=1e-8) / df[base_col].clip(lower=1e-8).shift(1)
            )
            df = df.drop(columns=[base_col])
        # Full 1-5 day lag structure (not just 1 and 3)
        for lag in [1, 2, 3, 4, 5]:
            lag_col = f'{base_col}_Lag_{lag}'
            if lag_col in df.columns:
                df[f'{lag_col}_Diff'] = df[lag_col].diff()
                df = df.drop(columns=[lag_col])

    # ── Oil-Gold spread momentum (dominant driver feature, Chai et al.) ──────────
    # Captures divergence between crude oil momentum and gold momentum.
    if 'WTI_Crude_Oil_Return' in df.columns and 'Close_Return' in df.columns:
        df['Oil_Gold_Spread_Momentum'] = (
            df['WTI_Crude_Oil_Return'].rolling(5).mean()
            - df['Close_Return'].rolling(5).mean()
        )
        print("Added Oil_Gold_Spread_Momentum (Chai et al. dominant driver feature).")

    # ── Standard macro: lags 1 and 3 only (DXY fully absorbed in ~5 days, 0.5% variance) ──
    for base_col in STANDARD_MACRO:
        if base_col in df.columns:
            df[f'{base_col}_Diff'] = df[base_col].diff()
            df = df.drop(columns=[base_col])
        for lag in [1, 3]:
            lag_col = f'{base_col}_Lag_{lag}'
            if lag_col in df.columns:
                df[f'{lag_col}_Diff'] = df[lag_col].diff()
                df = df.drop(columns=[lag_col])

    print("Transformed all prices and macroeconomic indicators to stationary series.")
    print(f"  Priority macro (1-5 day lags): {PRIORITY_MACRO}")
    print(f"  Standard macro (1,3 day lags): {STANDARD_MACRO}")

    # D. Additional momentum features for enhanced win rate
    # Price momentum: close above/below its 20-day rolling mean
    if 'Close_Return' in df.columns:
        df['Return_Momentum_5'] = df['Close_Return'].rolling(5).sum()  # 5-day cumulative return
        df['Return_Skew_20']    = df['Close_Return'].rolling(20).skew().fillna(0)  # return distribution skew

    print("Added momentum features: Return_Momentum_5, Return_Skew_20, ATR_14, BB_Width, High_Vol_Regime.")

    print("\n=== Step 4: Target Variable Generation ===")
    # 1. Shift returns by -1 to align today's features with TOMORROW'S return (target)
    df['Next_Day_Return'] = df['Close_Return'].shift(-1)

    # 2. Extract the absolutely most recent day for live inference BEFORE filtering flat days
    inference_df = df.iloc[[-1]].copy()
    inference_df = inference_df.drop(columns=['Next_Day_Return'], errors='ignore')

    # 3a. Existing binary label (unchanged — kept as primary target)
    rows_before_filter = len(df)
    df = df[df['Next_Day_Return'].abs() >= 0.0008].copy()
    rows_after_filter = len(df)
    print(f"Filtered {rows_before_filter - rows_after_filter} flat days (|return| < 0.08%).")

    df['Target_Direction'] = np.where(df['Next_Day_Return'] > 0, 1, 0)
    print("Generated primary binary classification target: 'Target_Direction'.")

    # 3b. SD-based dynamic label (Dakalbab et al.)
    # Identifies "significant" moves using a regime-adaptive rolling std threshold.
    # This label is nearly uncorrelated with fixed-threshold labels (corr ~0.13-0.42)
    # — meaning it captures genuinely different market behavior.
    rolling_std = df['Close_Return'].rolling(SD_WINDOW).std().shift(1)  # no lookahead
    significant = df['Next_Day_Return'].abs() > (rolling_std.fillna(1e-4) * SD_MULTIPLIER)

    # 3-class version: 1=sig-up, 0=insignificant, -1=sig-down
    df['Target_SD_3class'] = 0
    df.loc[significant & (df['Next_Day_Return'] > 0), 'Target_SD_3class'] = 1
    df.loc[significant & (df['Next_Day_Return'] < 0), 'Target_SD_3class'] = -1

    # Binary version (significant move or not) — used as an alternative classifier target
    df['Target_SD_Binary'] = (df['Target_SD_3class'] != 0).astype(int)

    sd_sig_pct = df['Target_SD_Binary'].mean() * 100
    sd_up_pct  = (df['Target_SD_3class'] == 1).sum() / len(df) * 100
    sd_dn_pct  = (df['Target_SD_3class'] == -1).sum() / len(df) * 100
    print(f"\nSD-Label stats (window={SD_WINDOW}, mult={SD_MULTIPLIER}\u03c3):")
    print(f"  Significant moves    : {sd_sig_pct:.1f}% of days")
    print(f"  Sig-Up (1)           : {sd_up_pct:.1f}%")
    print(f"  Sig-Down (-1)        : {sd_dn_pct:.1f}%")
    print(f"  Insignificant (0)    : {100-sd_sig_pct:.1f}%")
    print("  [NOTE] Run step7 with target='Target_Direction' (primary) and compare")
    print("         vs 'Target_SD_Binary' (adaptive) to test label robustness.")
    print("Generated adaptive SD-based target: 'Target_SD_Binary' + 'Target_SD_3class'.")

    # Drop original Close column (only keep Close_Return)
    df = df.drop(columns=['Close'], errors='ignore')
    inference_df = inference_df.drop(columns=['Close'], errors='ignore')

    print("\n=== Step 5: Row Cleanup and Serialization ===")
    inference_out = os.path.join(OUTPUT_DIR, "live_inference_data.csv")
    inference_df.to_csv(inference_out, index=False)
    print(f"Saved live inference data (today's row) to: {inference_out}")

    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    rows_after = len(df)
    print(f"Dropped {rows_before - rows_after} rows containing NaNs (warm-up and last row).")

    df = df.drop(columns=['Next_Day_Return'])
    print("Dropped temporary return columns.")

    df.to_csv(DATASET_OUT, index=False)
    print(f"Successfully saved final multimodal master dataset to: {DATASET_OUT}")

    print("\n=== Fusion Validation Report ===")
    print(f"Final shape: {df.shape}")
    print(f"Date range : {df['Date'].min().date()} to {df['Date'].max().date()}")

    target_counts = df['Target_Direction'].value_counts()
    target_pct = df['Target_Direction'].value_counts(normalize=True) * 100
    print("\nTarget Class Distribution:")
    for cls in [0, 1]:
        cnt = target_counts.get(cls, 0)
        pct = target_pct.get(cls, 0.0)
        label = "Down/Flat (0)" if cls == 0 else "Up (1)"
        print(f"  {label:<15}: {cnt:,} rows ({pct:.2f}%)")

    nans = df.isnull().sum().sum()
    print(f"\nRemaining NaN values in dataset: {nans}")
    if nans > 0:
        print("Warning: There are still NaN values in the dataset!")
        print(df.isnull().sum()[df.isnull().sum() > 0])

if __name__ == "__main__":
    main()
