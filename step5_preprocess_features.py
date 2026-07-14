"""
STEP 5 (ENHANCED v2): Feature Engineering + Gold-Impact FinBERT Scoring
=========================================================================
Key improvements over v1:
  ✓ Gold-direction adjusted FinBERT scores (not just generic polarity)
  ✓ Category-specific daily aggregations (war, fed, inflation, dollar)
  ✓ Source quality weighting on sentiment aggregation
  ✓ War/geopolitical impact score (most reliable gold mover)
  ✓ Urgency scoring: recency × source quality
  ✓ 5 new features added to master_features.csv

How FinBERT Works Here:
-----------------------
FinBERT (from ProsusAI) is a BERT model fine-tuned on financial text.
Input: Headline text (max 512 tokens)
Output: [P(positive), P(negative), P(neutral)] — three probabilities

Model location (auto-downloaded once, then cached locally):
  C:\\Users\\<user>\\.cache\\huggingface\\hub\\models--ProsusAI--finbert\\

Gold-Impact Adjustment:
  Raw polarity = prob[positive] - prob[negative]  ← generic
  Gold impact  = raw_polarity × direction_mult × source_quality
  
  direction_mult varies by headline category:
    WAR_GEOPOLITICAL → × +1.5  (negative news = bullish for gold)
    INFLATION        → × +1.2  (negative news = bullish hedge)
    RECESSION_CRISIS → × +1.3  (fear = safe haven demand)
    FED_POLICY       → × -1.0  (hawkish = bearish gold)
    DOLLAR_FX        → × -0.8  (dollar up = gold down)
    TREASURY_YIELDS  → × -1.0  (yields up = gold down)
    GOLD_MARKET      → × +1.0  (direct signal)
"""
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

OUTPUT_DIR       = os.path.dirname(os.path.abspath(__file__))
PRICE_FILE       = os.path.join(OUTPUT_DIR, "xauusd_raw_prices.csv")
MACRO_FILE       = os.path.join(OUTPUT_DIR, "fred_macro_raw.csv")
GDELT_FILE       = os.path.join(OUTPUT_DIR, "gdelt_news_raw.csv")
GOOGLE_NEWS_FILE = os.path.join(OUTPUT_DIR, "financial_news_raw.csv")
MASTER_OUT       = os.path.join(OUTPUT_DIR, "master_features.csv")
CACHE_FILE       = os.path.join(OUTPUT_DIR, "news_sentiment_cache.csv")
VP_FILE          = os.path.join(OUTPUT_DIR, "volume_profile_features.csv")

# ── GOLD-DIRECTION MULTIPLIERS ────────────────────────────────────────────────
# These map FinBERT's generic financial polarity to gold-market direction.
#
# Logic:
#   WAR news: FinBERT says "negative" (bad for economy) but gold RISES (safe haven)
#             → multiply by +1.5 to flip/amplify in the bullish direction
#   FED HIKE news: FinBERT says "positive" (progress) but gold FALLS (yields up)
#             → multiply by -1.0 to flip to bearish
#   INFLATION: FinBERT says "negative" (bad for consumers) but gold RISES (hedge)
#             → multiply by +1.2
#
# The model learns these relationships from labeled data, but these multipliers
# give it an explicit head-start signal.
GOLD_DIRECTION_MULTIPLIERS = {
    "WAR_GEOPOLITICAL":  +1.50,  # War/conflict → safe haven demand → gold UP
    "INFLATION":         +1.20,  # Inflation → hedge demand → gold UP
    "RECESSION_CRISIS":  +1.30,  # Crisis → flight to safety → gold UP
    "GOLD_MARKET":       +1.00,  # Direct gold news → use raw polarity
    "MACRO_ECONOMY":     +0.60,  # Weak economy news → mild gold support
    "ENERGY_OIL":        +0.40,  # Oil correlation → mild
    "FED_POLICY":        -1.00,  # Hawkish Fed → yields up → gold DOWN
    "DOLLAR_FX":         -0.80,  # Dollar strength → gold DOWN
    "TREASURY_YIELDS":   -1.00,  # Yield rises → gold DOWN
    "OTHER_FINANCE":     +0.30,  # Generic financial news → mild signal
}

# ── GOLD CATEGORY CLASSIFICATION ─────────────────────────────────────────────
import re
GOLD_CATEGORIES = [
    ("WAR_GEOPOLITICAL",  re.compile(
        r"war|conflict|military|attack|invasion|strike|missile|nuclear|"
        r"nato|ukraine|russia|iran|israel|middle east|north korea|taiwan|"
        r"sanction|ceasefire|terrorism|escalat", re.IGNORECASE)),
    ("FED_POLICY",        re.compile(
        r"federal reserve|fed rate|fomc|rate hike|rate cut|powell|"
        r"hawkish|dovish|quantitative|monetary policy|fed funds", re.IGNORECASE)),
    ("INFLATION",         re.compile(
        r"inflation|cpi|pce|stagflat|price index|consumer price|"
        r"producer price|ppi|hyperinflat", re.IGNORECASE)),
    ("DOLLAR_FX",         re.compile(
        r"dollar|dxy|dollar index|currency|devaluat|dedollar|"
        r"dollar weakness|dollar strength", re.IGNORECASE)),
    ("RECESSION_CRISIS",  re.compile(
        r"recession|crisis|collapse|contagion|bankrupt|default|"
        r"bail.?out|financial crisis|market crash", re.IGNORECASE)),
    ("TREASURY_YIELDS",   re.compile(
        r"treasury|bond yield|10.year|tips|real yield|yield curve", re.IGNORECASE)),
    ("ENERGY_OIL",        re.compile(
        r"crude oil|wti|brent|opec|oil price|energy", re.IGNORECASE)),
    ("GOLD_MARKET",       re.compile(
        r"gold price|xauusd|spot gold|gold futures|gold etf|gld|bullion|"
        r"precious metal|gold demand", re.IGNORECASE)),
    ("MACRO_ECONOMY",     re.compile(
        r"gdp|nonfarm|payroll|unemployment|jobs|economy|growth|pmi|"
        r"retail sales|consumer confidence", re.IGNORECASE)),
]

def classify_gold_category(headline: str) -> str:
    for cat_name, pattern in GOLD_CATEGORIES:
        if pattern.search(headline):
            return cat_name
    return "OTHER_FINANCE"

def compute_gold_impact_score(raw_polarity: float, category: str, source_quality: float = 1.0) -> float:
    """
    Convert generic FinBERT polarity into gold-direction adjusted impact score.

    Args:
        raw_polarity:   FinBERT score (prob[positive] - prob[negative]), range -1 to +1
        category:       Gold category assigned to this headline
        source_quality: Weight from 0.5 (low) to 1.0 (Reuters/Bloomberg)

    Returns:
        gold_impact: Adjusted score. Positive = bullish for gold. Negative = bearish.
    """
    mult = GOLD_DIRECTION_MULTIPLIERS.get(category, 0.30)

    # For categories that are INVERSELY correlated with gold (dollar up, yields up):
    # raw_polarity is positive (good economic news) but gold goes DOWN
    # mult is negative → score flips negative
    gold_impact = raw_polarity * mult * source_quality
    return float(gold_impact)


def main():
    print("=" * 65)
    print("  STEP 5 (ENHANCED v2): Gold-Impact Feature Engineering")
    print("  FinBERT → Gold-Direction Adjusted Scoring")
    print("=" * 65)

    print("\n=== Step 1: Loading Prices & Engineering Technical Indicators ===")
    if not os.path.exists(PRICE_FILE):
        print(f"Error: Price file {PRICE_FILE} not found!")
        sys.exit(1)

    prices = pd.read_csv(PRICE_FILE)
    prices['Date'] = pd.to_datetime(prices['Date'])
    prices = prices.sort_values('Date').reset_index(drop=True)
    print(f"Loaded {len(prices):,} rows of price data.")

    # Technical indicators
    prices['RSI_14'] = ta.rsi(prices['Close'], length=14)

    macd = ta.macd(prices['Close'], fast=12, slow=26, signal=9)
    prices = pd.concat([prices, macd], axis=1)

    bb = ta.bbands(prices['Close'], length=20, std=2)
    prices = pd.concat([prices, bb], axis=1)

    prices['EMA_50'] = ta.ema(prices['Close'], length=50)
    print("Technical indicators engineered successfully.")

    print("\n=== Step 2: Loading & Combining News Datasets ===")
    news_dfs = []
    for fpath, label in [(GDELT_FILE, "GDELT"), (GOOGLE_NEWS_FILE, "Google News/RSS")]:
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            # Add missing columns for backward compat (old CSV without category/quality)
            if 'Gold_Category' not in df.columns:
                print(f"  Assigning gold categories to {label} headlines...")
                df['Gold_Category'] = df['Headline'].apply(classify_gold_category)
            if 'Source_Quality' not in df.columns:
                df['Source_Quality'] = 0.70
            news_dfs.append(df)
            print(f"  Loaded {len(df):,} rows from {label}.")

    if not news_dfs:
        print("Error: No news files found! Run step3 first.")
        sys.exit(1)

    all_news = pd.concat(news_dfs, ignore_index=True)
    all_news['Date'] = pd.to_datetime(all_news['Date'], errors='coerce').dt.date
    all_news = all_news.dropna(subset=['Headline', 'Date'])
    all_news = all_news.drop_duplicates(subset=['Headline']).reset_index(drop=True)
    all_news['Source_Quality'] = all_news['Source_Quality'].fillna(0.70)
    print(f"\nFused news dataset: {len(all_news):,} unique headlines across "
          f"{all_news['Gold_Category'].nunique()} categories.")

    cat_counts = all_news['Gold_Category'].value_counts()
    print("\n  Headlines by gold-impact category:")
    for cat, cnt in cat_counts.items():
        pct = cnt / len(all_news) * 100
        print(f"    {cat:30s}: {cnt:,}  ({pct:.1f}%)")

    print("\n=== Step 3: FinBERT Scoring + Gold-Impact Adjustment ===")
    print("""
  How FinBERT predicts gold impact:
  ─────────────────────────────────
  Input  : Headline text
  Output : [P(positive), P(negative), P(neutral)]
  Raw    : polarity = P(positive) - P(negative)
  Adjust : gold_impact = polarity × gold_direction_mult × source_quality

  Example:
    "US launches military strikes in Middle East"
    FinBERT: negative (−0.72)
    Category: WAR_GEOPOLITICAL → mult = +1.5
    Gold impact: −0.72 × 1.5 = +1.08 ← BULLISH for gold ✓
    """)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    if device.type == "cpu":
        print("  TIP: GPU not detected. FinBERT will run on CPU (slower but works fine).")

    # Load sentiment cache
    sentiment_cache = {}   # {headline: raw_polarity}
    if os.path.exists(CACHE_FILE):
        try:
            cache_df = pd.read_csv(CACHE_FILE)
            sentiment_cache = dict(zip(cache_df['Headline'], cache_df['Polarity_Score']))
            print(f"  Loaded {len(sentiment_cache):,} cached FinBERT scores.")
        except Exception as e:
            print(f"  [WARN] Cache load failed: {e}")

    headlines_to_score = [h for h in all_news['Headline'].tolist() if h not in sentiment_cache]
    print(f"  Headlines needing FinBERT scoring: {len(headlines_to_score):,}")

    if headlines_to_score:
        print("  Loading FinBERT model from cache (ProsusAI/finbert)...")
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
        model.eval()
        # FinBERT class order: [positive, negative, neutral]
        print("  FinBERT class order: [positive=0, negative=1, neutral=2]")

        batch_size = 32 if device.type == "cpu" else 64

        for i in tqdm(range(0, len(headlines_to_score), batch_size), desc="  FinBERT scoring"):
            batch = headlines_to_score[i:i + batch_size]
            try:
                inputs = tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=128, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()

                for idx, prob in enumerate(probs):
                    # raw polarity: positive − negative (range −1 to +1)
                    # prob[0]=positive, prob[1]=negative, prob[2]=neutral
                    raw_polarity = float(prob[0] - prob[1])
                    sentiment_cache[batch[idx]] = raw_polarity
            except Exception as e:
                print(f"\n  [WARN] Batch {i} error: {e}")
                for h in batch:
                    sentiment_cache[h] = 0.0

        # Save updated cache
        cache_df = pd.DataFrame({
            'Headline': list(sentiment_cache.keys()),
            'Polarity_Score': list(sentiment_cache.values())
        })
        cache_df.to_csv(CACHE_FILE, index=False)
        print(f"\n  Saved {len(sentiment_cache):,} FinBERT scores to cache.")
    else:
        print("  All headlines found in cache — skipping FinBERT inference.")

    # Assign raw polarity and compute gold-impact score
    all_news['Raw_Polarity']     = all_news['Headline'].map(sentiment_cache).fillna(0.0)
    all_news['Gold_Impact_Score'] = all_news.apply(
        lambda r: compute_gold_impact_score(
            r['Raw_Polarity'], r['Gold_Category'], r.get('Source_Quality', 1.0)
        ), axis=1
    )

    print("\n  Sample gold-impact scoring:")
    sample = all_news[['Headline', 'Gold_Category', 'Raw_Polarity', 'Gold_Impact_Score']].sample(
        min(5, len(all_news)), random_state=42)
    for _, row in sample.iterrows():
        print(f"    [{str(row['Gold_Category']):20s}] {row['Headline'][:60]}")
        print(f"      Raw: {row['Raw_Polarity']:.3f}  Gold: {row['Gold_Impact_Score']:.3f}")

    print("\n=== Step 4: Daily Aggregation — Gold-Impact Features ===")
    # Main gold-impact daily aggregation (source-quality weighted)
    def weighted_mean(group, col, weight_col):
        w = group[weight_col].fillna(1.0)
        if w.sum() == 0:
            return group[col].mean()
        return (group[col] * w).sum() / w.sum()

    daily_records = []
    for day, grp in all_news.groupby('Date'):
        rec = {'Date': day}

        # Core sentiment features (gold-impact adjusted)
        rec['Mean_Gold_Impact']       = grp['Gold_Impact_Score'].mean()
        rec['Weighted_Gold_Impact']   = weighted_mean(grp, 'Gold_Impact_Score', 'Source_Quality')
        rec['Mean_Sentiment']         = grp['Raw_Polarity'].mean()   # kept for compat
        rec['Sentiment_Dispersion']   = grp['Gold_Impact_Score'].std() if len(grp) > 1 else 0.0
        rec['News_Volume']            = len(grp)

        # Category-specific impact scores (most important for gold prediction)
        for cat_name in [
            'WAR_GEOPOLITICAL', 'FED_POLICY', 'INFLATION',
            'DOLLAR_FX', 'RECESSION_CRISIS', 'GOLD_MARKET', 'TREASURY_YIELDS'
        ]:
            cat_grp = grp[grp['Gold_Category'] == cat_name]
            col = f'{cat_name}_Impact'
            if len(cat_grp) > 0:
                rec[col] = cat_grp['Gold_Impact_Score'].mean()
                rec[f'{cat_name}_Count'] = len(cat_grp)
            else:
                rec[col] = 0.0
                rec[f'{cat_name}_Count'] = 0

        # High-quality source sentiment (Reuters/Bloomberg only)
        hq_grp = grp[grp['Source_Quality'] >= 0.90]
        rec['HQ_Source_Impact']  = hq_grp['Gold_Impact_Score'].mean() if len(hq_grp) > 0 else 0.0
        rec['HQ_Source_Volume']  = len(hq_grp)

        daily_records.append(rec)

    daily_sent = pd.DataFrame(daily_records)
    daily_sent['Date'] = pd.to_datetime(daily_sent['Date'])
    daily_sent = daily_sent.fillna(0.0)
    print(f"  Aggregated gold-impact features over {len(daily_sent):,} unique trading dates.")

    print("\n=== Step 5: Interpolating FRED Macro Data ===")
    if not os.path.exists(MACRO_FILE):
        print(f"Error: Macro file {MACRO_FILE} not found! Run step2 first.")
        sys.exit(1)

    macro = pd.read_csv(MACRO_FILE)
    macro['Date'] = pd.to_datetime(macro['Date'])
    macro = macro.set_index('Date').resample('D').interpolate(method='linear').ffill().bfill().reset_index()
    print(f"  Interpolated macro dataset: {len(macro):,} daily rows.")

    print("\n=== Step 6: Merging Master Dataset ===")
    min_date = prices['Date'].min()
    max_date = prices['Date'].max()
    all_days = pd.date_range(start=min_date, end=max_date, freq='D')

    daily_sent = daily_sent.set_index('Date').reindex(all_days).reset_index().rename(columns={'index': 'Date'})
    daily_sent = daily_sent.fillna(0.0)

    master = pd.merge(prices, macro, on='Date', how='inner')
    master = pd.merge(master, daily_sent, on='Date', how='inner')

    # Macro lags
    macro_cols = [c for c in macro.columns if c != 'Date']
    for col in macro_cols:
        master[f'{col}_Lag_1'] = master[col].shift(1)
        master[f'{col}_Lag_3'] = master[col].shift(3)

    # Sentiment memory features (using gold-impact score)
    master['Sentiment_Lag_1']   = master['Mean_Gold_Impact'].shift(1)
    master['Sentiment_Lag_3']   = master['Mean_Gold_Impact'].shift(3)
    master['Sentiment_SMA_5']   = master['Mean_Gold_Impact'].rolling(5).mean()
    master['Sentiment_SMA_10']  = master['Mean_Gold_Impact'].rolling(10).mean()
    master['Sentiment_SMA_20']  = master['Mean_Gold_Impact'].rolling(20).mean()

    # War/geopolitical momentum (rolling)
    master['War_Impact_SMA_5']  = master['WAR_GEOPOLITICAL_Impact'].rolling(5).mean()
    master['War_Impact_SMA_10'] = master['WAR_GEOPOLITICAL_Impact'].rolling(10).mean()

    # Fed impact momentum
    master['Fed_Impact_SMA_5']  = master['FED_POLICY_Impact'].rolling(5).mean()

    print("  Added war/fed rolling impact features.")

    # ── FEATURE ENGINEERING 2.0 ───────────────────────────────────────────────
    price_momentum = (master['Close'] - master['EMA_50']) / (master['EMA_50'] + 1e-9)

    # Sentiment-price divergence (using gold-impact score)
    master['Sentiment_Price_Divergence'] = master['Mean_Gold_Impact'] - price_momentum

    # War vs Dollar divergence (war up + dollar up = rare, but strong signal)
    master['War_Dollar_Divergence'] = (
        master['WAR_GEOPOLITICAL_Impact'] - master['DOLLAR_FX_Impact']
    )

    # RSI Regime
    master['RSI_Regime'] = 0
    master.loc[master['RSI_14'] > 70, 'RSI_Regime'] = 1    # Overbought
    master.loc[master['RSI_14'] < 30, 'RSI_Regime'] = -1   # Oversold

    # Macro Pressure Index
    fedfunds_chg            = master['FedFunds_Rate'].diff().fillna(0)
    yield_chg               = master['US_10Y_Yield'].diff().fillna(0)
    master['Macro_Pressure_Index'] = fedfunds_chg + yield_chg

    # News Surprise Score (abnormal volume vs 20-day avg)
    news_vol_mean = master['News_Volume'].rolling(20).mean().shift(1)
    news_vol_std  = master['News_Volume'].rolling(20).std().shift(1) + 1e-9
    master['News_Surprise_Score'] = ((master['News_Volume'] - news_vol_mean) / news_vol_std).fillna(0)

    # Geopolitical Surge Score (abnormal war news volume)
    war_vol_mean  = master['WAR_GEOPOLITICAL_Count'].rolling(20).mean().shift(1)
    war_vol_std   = master['WAR_GEOPOLITICAL_Count'].rolling(20).std().shift(1) + 1e-9
    master['Geo_Surge_Score'] = (
        (master['WAR_GEOPOLITICAL_Count'] - war_vol_mean) / war_vol_std
    ).fillna(0)

    print("  Feature Engineering 2.0 complete:")
    print("    ✓ Sentiment_Price_Divergence")
    print("    ✓ War_Dollar_Divergence")
    print("    ✓ RSI_Regime")
    print("    ✓ Macro_Pressure_Index")
    print("    ✓ News_Surprise_Score")
    print("    ✓ Geo_Surge_Score  ← NEW (geopolitical surge detector)")
    print("    ✓ War_Impact_SMA_5/10  ← NEW")
    print("    ✓ Fed_Impact_SMA_5  ← NEW")
    print("    ✓ HQ_Source_Impact  ← NEW (Reuters/Bloomberg only)")

    # ── STEP 5B: Volume Profile Features ─────────────────────────────────────
    print("\n=== Step 7: Merging Gold Futures Volume Profile Features ===")
    from step5b_volume_profile import build_volume_profile_features

    # Re-compute VP features if the file is stale or missing
    if not os.path.exists(VP_FILE):
        print("  VP features not found — computing now (this may take a minute)...")
        build_volume_profile_features(PRICE_FILE, VP_FILE)
    else:
        print(f"  Loading cached VP features from {VP_FILE}")

    vp_df = pd.read_csv(VP_FILE)
    vp_df['Date'] = pd.to_datetime(vp_df['Date'])
    print(f"  Loaded {len(vp_df):,} rows × {vp_df.shape[1]} VP columns.")

    master = pd.merge(master, vp_df, on='Date', how='left')

    # VP cross-interaction features with existing signals
    # POC_Distance × War_Impact: strong gold signal confirmed by volume consensus
    master['POC_War_Signal'] = (
        master['POC_Distance_60'] * master['WAR_GEOPOLITICAL_Impact']
    ).fillna(0.0)

    # VArea_Width × Macro_Pressure: wide value area during macro stress = high-risk
    master['VArea_Macro_Risk'] = (
        master['VArea_Width_60'] * master['Macro_Pressure_Index'].abs()
    ).fillna(0.0)

    # RSI Overbought/Oversold confirmed by VP zone
    master['RSI_VP_Confirm'] = 0
    # RSI oversold + price below POC (long confluence)
    master.loc[
        (master['RSI_14'] < 35) & (master['VP_Long_Bias_60'] == 1), 'RSI_VP_Confirm'
    ] = 1
    # RSI overbought + price above POC (short confluence)
    master.loc[
        (master['RSI_14'] > 65) & (master['VP_Short_Bias_60'] == 1), 'RSI_VP_Confirm'
    ] = -1

    # Fill any VP NaNs (early rows where lookback insufficient)
    vp_cols = [c for c in master.columns if any(
        c.startswith(p) for p in [
            'POC', 'VAH', 'VAL', 'VArea', 'Price_vs', 'In_HVN', 'In_LVN',
            'Vol_Imbalance', 'VP_', 'POC_Distance', 'POC_War', 'VArea_Macro'
        ]
    )]
    master[vp_cols] = master[vp_cols].fillna(0.0)

    vp_new_cols = ['POC_War_Signal', 'VArea_Macro_Risk', 'RSI_VP_Confirm']
    print("  Volume Profile features merged. New interaction columns:")
    for col in vp_new_cols:
        print(f"    ✓ {col}")
    print(f"  Total VP columns added: {len(vp_cols)}")

    print(f"\n  Master dataset shape: {master.shape}")
    master.to_csv(MASTER_OUT, index=False)
    print(f"\n[SAVED] {MASTER_OUT}")

    print("\n=== Validation Summary ===")
    print(f"  Date range   : {master['Date'].min().date()} to {master['Date'].max().date()}")
    print(f"  Total rows   : {len(master):,}")
    print(f"  Features     : {master.shape[1]}")

    missing = master.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"\n  Missing values:")
        print(missing)
    else:
        print("  No missing values. ✓")

    print("\n  Key new gold-impact columns:")
    new_cols = [c for c in master.columns if any(
        x in c for x in ['Gold_Impact', 'War', 'Fed_Impact', 'HQ_Source', 'Geo_Surge', 'War_Dollar']
    )]
    print(f"  {new_cols}")


if __name__ == "__main__":
    main()
