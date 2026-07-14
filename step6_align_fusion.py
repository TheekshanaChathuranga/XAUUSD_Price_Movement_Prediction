import os
import sys
import re
import numpy as np
import pandas as pd

# Fix Unicode encoding on Windows
if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_IN = os.path.join(OUTPUT_DIR, "master_features.csv")
DATASET_OUT = os.path.join(OUTPUT_DIR, "multimodal_master_dataset.csv")


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
    # Only drop columns that actually exist
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped raw price/BB level columns: {cols_to_drop}")

    # C. Apply first-order differencing to macroeconomic levels (and their lags)
    macro_cols = [
        'CPI_US', 'FedFunds_Rate', 'Unemployment_Rate', 'NFP_Change',
        'WTI_Crude_Oil', 'PCE_Deflator', 'US_10Y_Yield', 'Real_GDP_Growth',
        'M2_Money_Supply', 'DXY_Index'
    ]
    for base_col in macro_cols:
        if base_col in df.columns:
            df[f'{base_col}_Diff'] = df[base_col].diff()
            df = df.drop(columns=[base_col])
        for lag in [1, 3]:
            lag_col = f'{base_col}_Lag_{lag}'
            if lag_col in df.columns:
                df[f'{lag_col}_Diff'] = df[lag_col].diff()
                df = df.drop(columns=[lag_col])

    print("Transformed all prices and macroeconomic indicators to stationary series.")

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

    # 3. Filter flat days where |return| < 0.08% (0.0008 log return)
    rows_before_filter = len(df)
    df = df[df['Next_Day_Return'].abs() >= 0.0008].copy()
    rows_after_filter = len(df)
    print(f"Filtered {rows_before_filter - rows_after_filter} flat days (|return| < 0.08%).")

    # 4. Create binary target direction (1 if tomorrow goes up, 0 if down)
    df['Target_Direction'] = np.where(df['Next_Day_Return'] > 0, 1, 0)
    print("Generated binary classification target: 'Target_Direction'.")

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
