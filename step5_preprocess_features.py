import os
import sys
import pandas as pd
import pandas_ta as ta
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Fix Unicode encoding on Windows
if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PRICE_FILE = os.path.join(OUTPUT_DIR, "xauusd_raw_prices.csv")
MACRO_FILE = os.path.join(OUTPUT_DIR, "fred_macro_raw.csv")
GDELT_FILE = os.path.join(OUTPUT_DIR, "gdelt_news_raw.csv")
GOOGLE_NEWS_FILE = os.path.join(OUTPUT_DIR, "financial_news_raw.csv")
MASTER_OUT = os.path.join(OUTPUT_DIR, "master_features.csv")

def main():
    print("=== Step 1: Loading Prices & Engineering Technical Indicators ===")
    if not os.path.exists(PRICE_FILE):
        print(f"Error: Price file {PRICE_FILE} not found!")
        sys.exit(1)
        
    prices = pd.read_csv(PRICE_FILE)
    prices['Date'] = pd.to_datetime(prices['Date'])
    prices = prices.sort_values('Date').reset_index(drop=True)
    print(f"Loaded {len(prices):,} rows of price data.")

    # Engineering Indicators
    prices['RSI_14'] = ta.rsi(prices['Close'], length=14)
    
    # MACD (12, 26, 9)
    macd = ta.macd(prices['Close'], fast=12, slow=26, signal=9)
    prices = pd.concat([prices, macd], axis=1)
    
    # Bollinger Bands (20, std=2)
    bb = ta.bbands(prices['Close'], length=20, std=2)
    prices = pd.concat([prices, bb], axis=1)
    
    # EMA (50)
    prices['EMA_50'] = ta.ema(prices['Close'], length=50)
    print("Technical indicators engineered successfully.")

    print("\n=== Step 2: Combining News & Scoring with FinBERT ===")
    news_dfs = []
    if os.path.exists(GDELT_FILE):
        gdelt = pd.read_csv(GDELT_FILE)
        news_dfs.append(gdelt)
        print(f"Loaded {len(gdelt):,} rows from GDELT.")
    if os.path.exists(GOOGLE_NEWS_FILE):
        gnews = pd.read_csv(GOOGLE_NEWS_FILE)
        news_dfs.append(gnews)
        print(f"Loaded {len(gnews):,} rows from Google News/RSS.")

    if not news_dfs:
        print("Error: No news files found!")
        sys.exit(1)

    all_news = pd.concat(news_dfs, ignore_index=True)
    all_news['Date'] = pd.to_datetime(all_news['Date']).dt.date
    # Drop exact headline duplicates
    all_news = all_news.drop_duplicates(subset=['Headline']).reset_index(drop=True)
    print(f"Fused news dataset contains {len(all_news):,} unique headlines.")

    # Sentiment scoring via FinBERT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for FinBERT: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
    model.eval()

    batch_size = 64
    sentiments = []
    
    # Batch sentiment inference
    for i in tqdm(range(0, len(all_news), batch_size), desc="FinBERT Scoring"):
        batch_headlines = all_news['Headline'].iloc[i:i+batch_size].astype(str).tolist()
        inputs = tokenizer(batch_headlines, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
        
        # FinBERT labels: [positive, negative, neutral]
        # Polarity = positive - negative
        for prob in probs:
            polarity = prob[0] - prob[1]
            sentiments.append(polarity)

    all_news['Polarity_Score'] = sentiments
    
    # Daily aggregation
    daily_sent = all_news.groupby('Date').agg(
        Mean_Sentiment=('Polarity_Score', 'mean'),
        Sentiment_Dispersion=('Polarity_Score', 'std'),
        News_Volume=('Polarity_Score', 'count')
    ).reset_index()
    
    daily_sent['Sentiment_Dispersion'] = daily_sent['Sentiment_Dispersion'].fillna(0.0)
    daily_sent['Date'] = pd.to_datetime(daily_sent['Date'])
    print(f"Aggregated sentiment over {len(daily_sent):,} unique dates.")

    print("\n=== Step 3: Interpolating FRED Macro Data ===")
    if not os.path.exists(MACRO_FILE):
        print(f"Error: Macro file {MACRO_FILE} not found!")
        sys.exit(1)

    macro = pd.read_csv(MACRO_FILE)
    macro['Date'] = pd.to_datetime(macro['Date'])
    
    # Linear interpolation + forward fill/backward fill to guarantee zero missing values
    macro = macro.set_index('Date').resample('D').interpolate(method='linear').ffill().bfill().reset_index()
    print(f"Interpolated macro dataset: {len(macro):,} daily rows.")

    print("\n=== Step 4: Creating Sentiment Memory & Merging Master Dataset ===")
    # Reindex daily sentiment to match the full date range of prices to ensure no gaps
    min_date = prices['Date'].min()
    max_date = prices['Date'].max()
    all_days = pd.date_range(start=min_date, end=max_date, freq='D')
    
    daily_sent = daily_sent.set_index('Date').reindex(all_days).reset_index().rename(columns={'index': 'Date'})
    
    # Fill days with no news with neutral values
    daily_sent['Mean_Sentiment'] = daily_sent['Mean_Sentiment'].fillna(0.0)
    daily_sent['Sentiment_Dispersion'] = daily_sent['Sentiment_Dispersion'].fillna(0.0)
    daily_sent['News_Volume'] = daily_sent['News_Volume'].fillna(0.0)
    
    # Master merge (aligning all datasets to trading day dates first)
    master = pd.merge(prices, macro, on='Date', how='inner')
    master = pd.merge(master, daily_sent, on='Date', how='inner')

    # Calculate Lags on the merged trading day grid to avoid look-ahead bias from calendar/weekend interpolation
    macro_cols = [c for c in macro.columns if c != 'Date']
    for col in macro_cols:
        master[f'{col}_Lag_1'] = master[col].shift(1)
        master[f'{col}_Lag_3'] = master[col].shift(3)
        
    # Sentiment Memory computed on the trading day grid
    master['Sentiment_Lag_1'] = master['Mean_Sentiment'].shift(1)
    master['Sentiment_Lag_3'] = master['Mean_Sentiment'].shift(3)
    master['Sentiment_SMA_5'] = master['Mean_Sentiment'].rolling(window=5).mean()
    master['Sentiment_SMA_10'] = master['Mean_Sentiment'].rolling(window=10).mean()
    master['Sentiment_SMA_20'] = master['Mean_Sentiment'].rolling(window=20).mean()

    # ── FEATURE ENGINEERING 2.0: Regime-Aware Momentum Signals ────────────────
    # Technique 1a: Sentiment vs Price Divergence
    # If news is bullish (positive) but price is below EMA_50, it signals a reversal setup
    price_momentum = (master['Close'] - master['EMA_50']) / (master['EMA_50'] + 1e-9)
    master['Sentiment_Price_Divergence'] = master['Mean_Sentiment'] - price_momentum

    # Technique 1b: RSI Regime (Overbought=1, Neutral=0, Oversold=-1)
    master['RSI_Regime'] = 0
    master.loc[master['RSI_14'] > 70, 'RSI_Regime'] = 1   # Overbought
    master.loc[master['RSI_14'] < 30, 'RSI_Regime'] = -1  # Oversold

    # Technique 1c: Macro Pressure Index (composite macro stress signal)
    # Rising Fed Funds + Rising 10Y Yield = bearish gold pressure
    fedfunds_chg = master['FedFunds_Rate'].diff().fillna(0)
    yield_chg    = master['US_10Y_Yield'].diff().fillna(0)
    master['Macro_Pressure_Index'] = fedfunds_chg + yield_chg

    # Technique 1d: News Surprise Score (abnormal news volume vs 20-day avg)
    news_vol_mean = master['News_Volume'].rolling(window=20).mean().shift(1)
    news_vol_std  = master['News_Volume'].rolling(window=20).std().shift(1) + 1e-9
    master['News_Surprise_Score'] = (master['News_Volume'] - news_vol_mean) / news_vol_std
    master['News_Surprise_Score'] = master['News_Surprise_Score'].fillna(0)

    print("Feature Engineering 2.0: Added Sentiment_Price_Divergence, RSI_Regime, Macro_Pressure_Index, News_Surprise_Score.")

    # Note: the first few rows will contain NaNs for rolling indicators (RSI, Bollinger, SMA, MACD, Lags) due to window sizes.
    
    print(f"Merging complete. Master dataset shape: {master.shape}")
    
    # Save the master features
    master.to_csv(MASTER_OUT, index=False)
    print(f"Successfully saved master features dataset to: {MASTER_OUT}")

    # Output simple validation summary
    print("\n=== Validation Summary ===")
    print(f"Master file: {MASTER_OUT}")
    print(f"Date range : {master['Date'].min().date()} to {master['Date'].max().date()}")
    print(f"Total rows : {len(master)}")
    print(f"Columns    : {list(master.columns)}")
    print("\nMissing values status:")
    print(master.isnull().sum()[master.isnull().sum() > 0])

if __name__ == "__main__":
    main()

