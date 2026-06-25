import os
import sys
import numpy as np
import pandas as pd

# Fix Unicode encoding on Windows
if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_IN = os.path.join(OUTPUT_DIR, "master_features.csv")
DATASET_OUT = os.path.join(OUTPUT_DIR, "multimodal_master_dataset.csv")

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
    # Fill missing news sentiment columns with 0 (neutral sentiment / no articles)
    news_cols = [
        'Mean_Sentiment', 'Sentiment_Dispersion', 'News_Volume',
        'Sentiment_Lag_1', 'Sentiment_Lag_3',
        'Sentiment_SMA_5', 'Sentiment_SMA_10', 'Sentiment_SMA_20'
    ]
    
    # Check if all news columns exist in df before filling
    existing_news_cols = [col for col in news_cols if col in df.columns]
    df[existing_news_cols] = df[existing_news_cols].fillna(0)
    print(f"Filled missing values in news columns: {existing_news_cols}")

    # Forward-fill remaining missing macro or price data
    # Do NOT apply backward-fill (bfill) here, because backward-filling technical indicators
    # (like EMA_50, RSI_14, MACD, lags) would introduce look-ahead bias at the beginning of the series.
    # Instead, we let those leading NaNs remain so they are properly discarded during dropna() row cleanup.
    df = df.ffill()
    print("Applied forward-fill to resolve any remaining gaps (preserved leading NaNs for cleanup).")

    print("\n=== Step 3: Stationary Transformations (Log Returns & Differencing) ===")
    # A. Convert absolute price levels to log returns
    for col in ['Open', 'High', 'Low', 'Close']:
        df[f'{col}_Return'] = np.log(df[col] / df[col].shift(1))
        
    # B. Convert non-stationary indicators (EMA, BBands levels) to Close price ratios
    df['EMA_50_Ratio'] = df['EMA_50'] / df['Close']
    df['BBL_Ratio'] = df['BBL_20_2.0_2.0'] / df['Close']
    df['BBM_Ratio'] = df['BBM_20_2.0_2.0'] / df['Close']
    df['BBU_Ratio'] = df['BBU_20_2.0_2.0'] / df['Close']
    
    # Drop original absolute price columns
    df = df.drop(columns=['Open', 'High', 'Low', 'EMA_50', 'BBL_20_2.0_2.0', 'BBM_20_2.0_2.0', 'BBU_20_2.0_2.0'])
    
    # C. Apply first-order differencing to macroeconomic levels (and their lags)
    macro_cols = [
        'CPI_US', 'FedFunds_Rate', 'Unemployment_Rate', 'NFP_Change', 
        'WTI_Crude_Oil', 'PCE_Deflator', 'US_10Y_Yield', 'Real_GDP_Growth', 
        'M2_Money_Supply', 'DXY_Index'
    ]
    for base_col in macro_cols:
        # Difference base column
        df[f'{base_col}_Diff'] = df[base_col].diff()
        df = df.drop(columns=[base_col])
        # Difference corresponding lags
        for lag in [1, 3]:
            lag_col = f'{base_col}_Lag_{lag}'
            if lag_col in df.columns:
                df[f'{lag_col}_Diff'] = df[lag_col].diff()
                df = df.drop(columns=[lag_col])
                
    print("Transformed all prices and macroeconomic indicators to stationary series.")

    print("\n=== Step 4: Target Variable Generation ===")
    # 1. Calculate today's log return based on closing prices (uses the newly created Close_Return)
    # 2. Shift returns by -1 to align today's features with TOMORROW'S return (target)
    df['Next_Day_Return'] = df['Close_Return'].shift(-1)
    
    # 3. Filter flat days where |return| < 0.08% (0.0008 log return)
    rows_before_filter = len(df)
    df = df[df['Next_Day_Return'].abs() >= 0.0008].copy()
    rows_after_filter = len(df)
    print(f"Filtered {rows_before_filter - rows_after_filter} flat days (|return| < 0.08%).")
    
    # 4. Create binary target direction (1 if tomorrow goes up, 0 if down)
    df['Target_Direction'] = np.where(df['Next_Day_Return'] > 0, 1, 0)
    print("Generated binary classification target: 'Target_Direction'.")

    # Drop original Close column (only keep Close_Return to prevent leakage/non-stationarity)
    df = df.drop(columns=['Close'])

    print("\n=== Step 5: Row Cleanup and Serialization ===")
    # Extract the absolutely most recent day for live inference BEFORE dropping NaNs
    inference_df = df.iloc[[-1]].copy()
    inference_df = inference_df.drop(columns=['Next_Day_Return', 'Target_Direction'], errors='ignore')
    inference_out = os.path.join(OUTPUT_DIR, "live_inference_data.csv")
    inference_df.to_csv(inference_out, index=False)
    print(f"Saved live inference data (today's row) to: {inference_out}")

    # Drop rows containing NaNs (e.g. warm-up rows for technical indicators at the start,
    # differencing NaNs, and the last row which has no target variable for tomorrow)
    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    rows_after = len(df)
    print(f"Dropped {rows_before - rows_after} rows containing NaNs (warm-up periods and last row target).")

    # Drop temporary columns to prevent data leakage in features
    df = df.drop(columns=['Next_Day_Return'])
    print("Dropped temporary return columns to prevent look-ahead bias leakage.")

    # Save dataset
    df.to_csv(DATASET_OUT, index=False)
    print(f"Successfully saved final multimodal master dataset to: {DATASET_OUT}")

    # Print validation report
    print("\n=== Fusion Validation Report ===")
    print(f"Final shape: {df.shape}")
    print(f"Date range : {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # Target distribution check
    target_counts = df['Target_Direction'].value_counts()
    target_pct = df['Target_Direction'].value_counts(normalize=True) * 100
    print("\nTarget Class Distribution:")
    for cls in [0, 1]:
        cnt = target_counts.get(cls, 0)
        pct = target_pct.get(cls, 0.0)
        label = "Down/Flat (0)" if cls == 0 else "Up (1)"
        print(f"  {label:<15}: {cnt:,} rows ({pct:.2f}%)")

    # Check for NaNs
    nans = df.isnull().sum().sum()
    print(f"\nRemaining NaN values in dataset: {nans}")
    if nans > 0:
        print("Warning: There are still NaN values in the dataset!")

if __name__ == "__main__":
    main()
