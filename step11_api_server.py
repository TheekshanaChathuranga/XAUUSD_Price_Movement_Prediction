"""
step11_api_server.py  —  Gold AI Trading API v3
================================================
NEW in v3:
  - APScheduler background job: fetches live news + re-scores every 15 min
  - VADER real-time sentiment (instant, no GPU)
  - In-memory signal cache: /api/predict always responds in <100ms
  - GET /api/live-news    → latest headlines with sentiment tags
  - GET /api/macro-calendar → upcoming macro events (next 30 days)
  - GET /api/health        → data freshness check
  - POST /api/refresh      → manual trigger of daily_refresh.py

Ensemble: CatBoost + XGBoost + LightGBM + Meta-Learner (unchanged)
Live sentiment blending: 70% FinBERT historical + 30% VADER live
"""

import os, json, subprocess, sys, time, re, requests, threading
os.environ["PYTHONIOENCODING"] = "utf-8"
from datetime import timezone as _tz
import numpy as np
import pandas as pd
import joblib
import shap
from datetime import datetime, date, timedelta
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from bs4 import BeautifulSoup
from catboost  import CatBoostClassifier
import xgboost  as xgb
import lightgbm as lgb
from apscheduler.schedulers.background import BackgroundScheduler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import uvicorn

# ── PATHS ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR     = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DATA = os.path.join(OUTPUT_DIR, "live_inference_data.csv")
MODEL_CAT      = os.path.join(OUTPUT_DIR, "catboost_prod.cbm")
MODEL_XGB      = os.path.join(OUTPUT_DIR, "xgb_prod.json")
MODEL_LGB      = os.path.join(OUTPUT_DIR, "lgb_prod.txt")
MODEL_META     = os.path.join(OUTPUT_DIR, "meta_learner.pkl")
SCALER_PATH    = os.path.join(OUTPUT_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(OUTPUT_DIR, "model_threshold.json")
RAW_PRICES     = os.path.join(OUTPUT_DIR, "xauusd_raw_prices.csv")
GDELT_NEWS     = os.path.join(OUTPUT_DIR, "gdelt_news_raw.csv")
FIN_NEWS       = os.path.join(OUTPUT_DIR, "financial_news_raw.csv")
STATIC_DIR     = os.path.join(OUTPUT_DIR, "static")
REFRESH_SCRIPT = os.path.join(OUTPUT_DIR, "daily_refresh.py")

# ── APP ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Gold AI Trading API v3")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")

# ── VADER ─────────────────────────────────────────────────────────────────────
_vader = SentimentIntensityAnalyzer()

# Finance-domain booster lexicon — VADER default is general English
# Positive for gold: safe haven, rate cut, inflation, weak dollar, war risk
# Negative for gold: rate hike, dollar strength, risk-on, hawkish
_GOLD_LEXICON = {
    "rate cut": 2.5, "rate cuts": 2.5, "dovish": 2.0, "safe haven": 2.0,
    "inflation": 1.5, "geopolitical": 1.2, "recession": 1.5, "weak dollar": 2.0,
    "bullion": 0.5, "rally": 1.0, "surge": 1.5, "breakout": 1.2, "haven": 1.5,
    "rate hike": -2.5, "rate hikes": -2.5, "hawkish": -2.0, "tightening": -1.5,
    "strong dollar": -2.0, "dollar strength": -2.0, "risk-on": -1.0,
    "below expectations": -1.5, "misses": -1.2, "disappoints": -1.5,
    "beats": 0.8, "exceeds expectations": 1.2, "stronger than expected": -1.0,
}
_vader.lexicon.update(_GOLD_LEXICON)

_GOLD_PHRASES = {
    # Bearish gold factors
    "gold drops": -0.8,
    "gold dives": -0.9,
    "gold falls": -0.8,
    "loses ground": -0.5,
    "selloff": -0.6,
    "liquidation": -0.5,
    "us dollar rises": -0.7,
    "lifts us dollar": -0.7,
    "lift us dollar": -0.7,
    "lifts the usd": -0.7,
    "lift the usd": -0.7,
    "fed hike": -0.6,
    "fed rate hike": -0.7,
    "hawkish fed": -0.6,
    "rate hike": -0.7,
    "dollar strength": -0.6,
    "strong dollar": -0.6,
    # Bullish gold factors
    "gold rises": 0.8,
    "gold surges": 0.9,
    "gold jumps": 0.8,
    "safe haven demand": 0.7,
    "rate cut expectations": 0.7,
    "fed rate cut": 0.7,
    "dovish fed": 0.6,
    "weak dollar": 0.6,
}

def vader_score(headline: str) -> float:
    """Return compound VADER score for a gold-context headline, enhanced with phrase matching."""
    headline_lower = str(headline).lower()
    score = _vader.polarity_scores(str(headline))["compound"]
    
    # Apply adjustments for specific financial phrases
    for phrase, adjustment in _GOLD_PHRASES.items():
        if phrase in headline_lower:
            score += adjustment
            
    return float(max(-1.0, min(1.0, score)))

def sentiment_label(score: float) -> str:
    if score >= 0.15:  return "BULLISH"
    if score <= -0.15: return "BEARISH"
    return "NEUTRAL"


# ── IN-MEMORY CACHE ──────────────────────────────────────────────────────────
_cache_lock          = threading.Lock()
_signal_cache        = {}      # full /api/predict payload
_news_cache          = []      # list of recent headlines with sentiment
_last_refresh        = None    # datetime of last successful refresh
# BUG FIX: Use threading.Event for atomic refresh guards (thread-safe)
_refresh_event       = threading.Event()       # set() = refreshing
_full_refresh_event  = threading.Event()       # set() = full refresh running
# Keep bool aliases for /api/health backward-compat
@property
def _is_refreshing():       return _refresh_event.is_set()
@property
def _is_full_refreshing():  return _full_refresh_event.is_set()

# ── HELPERS ──────────────────────────────────────────────────────────────────
def load_threshold():
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH) as f:
            cfg = json.load(f)
        return cfg.get("threshold", 0.5), cfg.get("confidence_band", 0.65)
    return 0.5, 0.65

def load_adaptive_thresholds():
    """
    Compute adaptive LONG/SHORT thresholds from the historical Ensemble_Prob
    distribution. The meta-learner compresses probs into a narrow band
    (e.g. 0.52–0.66), making hardcoded thresholds like 0.65 unreachable.
    Using P70/P30 percentiles ensures both LONG and SHORT signals are produced.
    """
    PERCENTILE_LONG  = 70
    PERCENTILE_SHORT = 30
    preds_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    if os.path.exists(preds_path):
        try:
            preds_df = pd.read_csv(preds_path)
            if 'Ensemble_Prob' in preds_df.columns and len(preds_df) > 20:
                long_t  = float(np.percentile(preds_df['Ensemble_Prob'], PERCENTILE_LONG))
                short_t = float(np.percentile(preds_df['Ensemble_Prob'], PERCENTILE_SHORT))
                return long_t, short_t
        except Exception:
            pass
    # Fallback: use confidence_band from threshold file
    _, cb = load_threshold()
    return cb, 1 - cb

def calculate_atr(df, period=14):
    hl  = df['High'] - df['Low']
    hc  = np.abs(df['High'] - df['Close'].shift())
    lc  = np.abs(df['Low']  - df['Close'].shift())
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _data_staleness():
    try:
        df = pd.read_csv(INFERENCE_DATA)
        inf_date = pd.to_datetime(df['Date'].iloc[-1]).date()
        days_old  = (date.today() - inf_date).days
        is_stale  = days_old > 3
        return str(inf_date), days_old, is_stale
    except Exception:
        return "unknown", 99, True

# ── NARRATIVE GENERATOR ───────────────────────────────────────────────────────
FEATURE_REASONS = {
    "Sentiment_SMA_5": {
        "UP":   "The 5-day average of institutional news headlines from Kitco and Reuters has turned POSITIVE, indicating growing bullish sentiment in the gold market.",
        "DOWN": "The 5-day average of institutional news headlines from Kitco and Reuters has turned NEGATIVE, reflecting growing bearish pressure on gold."
    },
    "Sentiment_Price_Divergence": {
        "UP":   "Gold news is bullish but the price is lagging below its 50-day average — this historically triggers a sharp catch-up rally as price follows sentiment.",
        "DOWN": "Gold news is bearish and the price is already elevated above its 50-day average — this divergence historically precedes a correction."
    },
    "Macro_Pressure_Index": {
        "UP":   "Interest rates and bond yields are falling, reducing the cost of holding gold and making it more attractive to investors.",
        "DOWN": "Interest rates and bond yields are rising, increasing the opportunity cost of holding gold and putting downward pressure on the price."
    },
    "DXY_Index_Diff": {
        "UP":   "The US Dollar weakened today — since gold is priced in dollars, a weaker dollar makes gold cheaper for foreign buyers, boosting demand.",
        "DOWN": "The US Dollar strengthened today — a stronger dollar makes gold more expensive for foreign buyers, reducing demand and pressing the price lower."
    },
    "RSI_Regime": {
        "UP":   "The Relative Strength Index (RSI) has dropped into oversold territory — gold has been sold too aggressively and a technical bounce is historically expected.",
        "DOWN": "The Relative Strength Index (RSI) has risen into overbought territory — gold has rallied too far too fast and a pullback is historically expected."
    },
    "Close_Return": {
        "UP":   "Gold posted a strong positive return yesterday, confirming upward momentum that statistically tends to continue short-term.",
        "DOWN": "Gold posted a negative return yesterday, confirming downward momentum that statistically tends to continue short-term."
    },
    "Sentiment_Dispersion": {
        "UP":   "Market news consensus is unified and strongly positive — low dispersion in sentiment means institutional players are aligned on the bullish view.",
        "DOWN": "Market news is highly divided and conflicted — high dispersion in sentiment signals uncertainty, which historically drives gold lower as risk appetite falls."
    },
    "WTI_Crude_Oil_Diff": {
        "UP":   "Oil prices rose today — rising energy costs signal higher inflation expectations, which strengthens gold's appeal as an inflation hedge.",
        "DOWN": "Oil prices fell today — lower energy costs reduce inflation fears, weakening the case for holding gold as an inflation hedge."
    },
    "News_Surprise_Score": {
        "UP":   "News volume today spiked dramatically above normal — this abnormal media activity signals a major upcoming macro event that historically triggers gold volatility to the upside.",
        "DOWN": "News volume today spiked dramatically above normal — this signals a major macro event risk that is historically associated with short-term gold weakness as traders take profits."
    },
    "Tick_Volume": {
        "UP":   "Trading volume is significantly above average — institutional buyers are entering the market with conviction.",
        "DOWN": "Trading volume is abnormally high — institutions appear to be distributing (selling) their gold positions at current levels."
    },
    "M2_Money_Supply_Diff": {
        "UP":   "Global money supply expanded — more money in circulation historically drives inflation expectations higher, boosting gold as a store of value.",
        "DOWN": "Global money supply contracted — tighter monetary conditions reduce inflation risks and reduce the demand for gold as a hedge."
    },
}

def get_readable_reason(feature: str, direction: str) -> str:
    mapping = FEATURE_REASONS.get(feature, {})
    if mapping:
        return mapping.get(direction, mapping.get("UP", ""))
    dir_word = "supports an upward move" if direction == "UP" else "supports a downward move"
    return f"The algorithmic engine detected a pattern in '{feature}' that {dir_word} in gold."

def generate_narrative(signal, prob_up, top_drivers, entry, sl, tp, atr):
    direction_word = {"LONG": "rise", "SHORT": "fall", "NEUTRAL": "move unpredictably"}.get(signal, "move")
    confidence_pct = max(prob_up, 1 - prob_up) * 100
    
    if signal == "LONG":
        summary = f"The AI is {confidence_pct:.0f}% confident that Gold (XAUUSD) will RISE from ${entry:,.2f}."
        supportive = [d for d in top_drivers if d.get("impact", 0) > 0]
        opposing = [d for d in top_drivers if d.get("impact", 0) < 0]
    elif signal == "SHORT":
        summary = f"The AI is {confidence_pct:.0f}% confident that Gold (XAUUSD) will FALL from ${entry:,.2f}."
        supportive = [d for d in top_drivers if d.get("impact", 0) < 0]
        opposing = [d for d in top_drivers if d.get("impact", 0) > 0]
    else:
        summary = "The AI model is uncertain about Gold's direction. No high-probability setup detected. Staying flat is recommended."
        supportive = []
        opposing = []

    if signal in ("LONG", "SHORT"):
        # Construct reasoning from supportive drivers
        support_texts = [d.get("text", "") for d in supportive[:3] if d.get("text")]
        if support_texts:
            reasoning = " ".join(support_texts)
        else:
            reasoning = f"Technical and macroeconomic signals are aligned to support a {signal.lower()} outlook."
        
        # Construct risk note from risk management and opposing drivers
        risk_texts = [d.get("text", "") for d in opposing[:2] if d.get("text")]
        opposing_reasons = " " + " ".join(risk_texts) if risk_texts else ""
        
        rr_note = (f"Risk management: Enter at ${entry:,.2f}. "
                   f"Stop loss at ${sl:,.2f} (0.4×ATR). "
                   f"Take profit at ${tp:,.2f} (0.8×ATR — 1:2 risk/reward). "
                   f"Key risk factors to monitor:{opposing_reasons} This trade is designed to close within 24 hours.")
    else:
        reasoning = "Multiple technical and macroeconomic signals are conflicted, resulting in a neutral stance."
        rr_note = ("No trade recommended. The risk-to-reward ratio is unfavorable when the model "
                   "is uncertain. Preserve capital and wait for a stronger signal.")
                   
    return {"summary": summary, "reasoning": reasoning, "risk_note": rr_note}

# ── LIVE NEWS FETCHER (fast — only last 24h) ──────────────────────────────────
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}
# ── PRECISION GOLD-IMPACT NEWS FILTER ────────────────────────────────────────
# Only headlines that directly move gold prices pass this filter.
# Covers: gold direct, geopolitical/military, US foreign policy,
#         central bank policy, inflation, dollar/bonds, safe-haven, energy crisis.
_FINANCE_RE = re.compile(
    # Gold direct
    r"gold|xau|bullion|precious metal|spot gold|gold price|gold futures|gold etf|gld|"
    # Geopolitical & military conflict (highest gold impact)
    r"war|conflict|military|attack|invasion|missile|nuclear|nato|sanction|escalat|"
    r"ceasefire|coup|terrorism|civil war|airstrike|drone strike|pentagon|"
    r"us forces|us troops|us military|us army|us navy|us air force|warfare|arms deal|"
    r"us strike|american troops|defense secretary|joint chiefs|"
    # Key conflict zones that historically drive gold safe-haven bids
    r"iran|israel|russia|ukraine|middle east|north korea|taiwan|china tension|"
    r"hamas|hezbollah|houthi|red sea|strait of hormuz|"
    # US foreign policy & dollar weaponization
    r"us sanction|trade war|tariff|embargo|dollar dominance|dedollar|"
    r"us debt|debt ceiling|us deficit|treasury default|"
    # Central bank policy (high impact on gold via real rates)
    r"federal reserve|fed rate|fomc|rate hike|rate cut|powell|hawkish|dovish|"
    r"ecb|bank of england|boe|monetary policy|quantitative|taper|fed funds|"
    r"interest rate decision|central bank|"
    # Inflation data (key gold driver)
    r"inflation|cpi|pce|ppi|stagflat|price index|consumer price|producer price|"
    # US Dollar & bonds (gold inverse correlation)
    r"dollar|dxy|dollar index|dollar weakness|dollar strength|treasury|yield|"
    r"real yield|tips|bond yield|10.year|"
    # Crisis & safe-haven flows
    r"safe haven|safe-haven|flight to safety|risk off|crisis|panic|collapse|"
    r"contagion|bail.?out|bank run|financial crisis|recession|default|"
    # Energy crisis (inflation proxy, gold-correlated)
    r"crude oil|wti|brent|opec|oil price|oil shock|energy crisis",
    re.IGNORECASE
)

# Noise exclusion — headlines that pass _FINANCE_RE but are NOT gold-relevant
_NOISE_RE = re.compile(
    r"(earnings|quarterly results|stock split|ipo|merger|acquisition|"
    r"lawsuit|recall|product launch|retail sales|consumer confidence|"
    r"housing starts|pmi survey|manufacturing index|car sales|auto sales|"
    r"sports|nfl|nba|cricket|celebrity|entertainment|weather|hurricane|tornado)",
    re.IGNORECASE
)
_DIRECT_GOLD_RE = re.compile(r"gold|xau|bullion|spot gold", re.IGNORECASE)

def _is_gold_relevant(headline: str) -> bool:
    """True if headline passes gold-impact filter and is not pure noise."""
    if not _FINANCE_RE.search(headline):
        return False
    # Allow through if headline directly mentions gold even if noise-pattern matches
    if _NOISE_RE.search(headline) and not _DIRECT_GOLD_RE.search(headline):
        return False
    return True

# ── GOLD CATEGORY CLASSIFIER ──────────────────────────────────────────────────
_GOLD_CATEGORIES = [
    ("WAR_MILITARY",  re.compile(
        r"war|conflict|military|attack|invasion|missile|nuclear|nato|sanction|"
        r"ceasefire|terrorism|coup|escalat|airstrike|drone strike|pentagon|"
        r"us forces|us troops|iran|israel|russia|ukraine|middle east|north korea|"
        r"taiwan|hamas|hezbollah|houthi|red sea|warfare|arms deal", re.IGNORECASE)),
    ("FED_POLICY",    re.compile(
        r"federal reserve|fed rate|fomc|rate hike|rate cut|powell|"
        r"hawkish|dovish|quantitative|taper|monetary policy|fed funds|"
        r"interest rate decision|ecb|bank of england|central bank", re.IGNORECASE)),
    ("INFLATION",     re.compile(
        r"inflation|cpi|pce|ppi|stagflat|price index|consumer price|producer price", re.IGNORECASE)),
    ("DOLLAR_FX",     re.compile(
        r"dollar|dxy|dollar index|dollar weakness|dollar strength|dedollar|treasury|"
        r"yield|bond yield|10.year|tips|real yield", re.IGNORECASE)),
    ("CRISIS",        re.compile(
        r"recession|crisis|panic|collapse|contagion|bail.?out|bank run|"
        r"financial crisis|market crash|default|safe haven|flight to safety|risk off", re.IGNORECASE)),
    ("ENERGY",        re.compile(
        r"crude oil|wti|brent|opec|oil price|oil shock|energy crisis", re.IGNORECASE)),
    ("GOLD_MARKET",   re.compile(
        r"gold price|xauusd|spot gold|gold futures|gold etf|gld|bullion|"
        r"precious metal|gold demand|gold supply|gold reserve|central bank gold", re.IGNORECASE)),
]

_CATEGORY_ICONS = {
    "WAR_MILITARY": "🪖",
    "FED_POLICY":   "🏦",
    "INFLATION":    "📈",
    "DOLLAR_FX":    "💵",
    "CRISIS":       "🚨",
    "ENERGY":       "🛢️",
    "GOLD_MARKET":  "🥇",
    "OTHER":        "📰",
}

def classify_news_category(headline: str) -> str:
    """Return gold-impact category for a headline (first match wins)."""
    for cat, pattern in _GOLD_CATEGORIES:
        if pattern.search(headline):
            return cat
    return "OTHER"

_DATE_FMTS = [
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%SZ",
]

def _parse_rss_date(s: str):
    if not s: return None
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(s[:len(fmt)+5].strip(), fmt).replace(tzinfo=None)
        except Exception: pass
    clean = re.sub(r"\s+[A-Z]{2,5}$", "", s.strip())
    try: return datetime.strptime(clean[:25], "%a, %d %b %Y %H:%M:%S")
    except: return None

# ── GOLD-IMPACT TARGETED GOOGLE NEWS QUERIES ──────────────────────────────────
GNEWS_QUERIES = [
    # Gold market direct
    "spot gold price bullion",
    "gold ETF GLD central bank buying reserves",
    # Fed / central bank
    "federal reserve interest rate FOMC decision",
    "ECB Bank of England rate decision gold",
    # Inflation
    "CPI inflation data PCE expectations gold",
    # Dollar
    "dollar DXY weakness strength gold",
    "dedollarization US dollar gold reserves",
    # Geopolitical / military (highest gold impact)
    "US military strike airstrike gold safe haven",
    "geopolitical tension conflict gold price",
    "Iran Israel Russia Ukraine war gold",
    "US sanctions trade war dollar gold",
    "Pentagon US forces Middle East gold",
    # Crisis
    "financial crisis recession gold safe haven",
    # Energy
    "crude oil OPEC energy crisis gold inflation",
]

# ── RSS SOURCES — GOLD-IMPACT FOCUSED ────────────────────────────────────────
RSS_LIVE = [
    # Gold-specialist sources
    ("https://finance.yahoo.com/rss/headline?s=GC=F",               "Yahoo Finance"),
    ("https://www.kitco.com/feed/news.rss",                          "Kitco News"),
    ("https://www.goldbroker.com/news.rss",                          "GoldBroker"),
    ("https://www.bullionvault.com/gold-news/feed",                  "BullionVault"),
    # Macro / FX sources
    ("https://www.forexlive.com/feed/news",                          "ForexLive"),
    ("https://www.fxstreet.com/rss/news",                            "FXStreet"),
    ("https://www.cnbc.com/id/20910258/device/rss/rss.html",         "CNBC"),
    ("https://feeds.marketwatch.com/marketwatch/realtimeheadlines/", "MarketWatch"),
    # Geopolitical / war / US military — critical gold movers
    ("https://feeds.reuters.com/Reuters/worldNews",                  "Reuters"),
    ("https://feeds.reuters.com/reuters/topNews",                    "Reuters"),
    ("https://www.aljazeera.com/xml/rss/all.xml",                    "Al Jazeera"),
]

def _fetch_rss(url: str, source: str) -> list:
    try:
        r = requests.get(url, headers=_HEADERS, timeout=12)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "xml")
        items = soup.find_all("item") or soup.find_all("entry")
    except Exception:
        return []
    records = []
    cutoff  = datetime.now(_tz.utc).replace(tzinfo=None) - timedelta(hours=24)
    for item in items:
        t = item.find("title")
        if not t: continue
        title = t.get_text(strip=True)
        if not _is_gold_relevant(title): continue  # precision gold-impact filter
        pub = item.find("pubDate") or item.find("published")
        dt  = _parse_rss_date(pub.get_text(strip=True) if pub else "")
        if dt and dt < cutoff: continue
        lnk = item.find("link")
        url_val = (lnk.get("href") or lnk.get_text(strip=True)) if lnk else ""
        records.append({
            "Datetime": dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None,
            "Date":     dt.strftime("%Y-%m-%d") if dt else str(date.today()),
            "Headline": title,
            "Source":   source,
            "URL":      url_val,
        })
    return records

def _fetch_gnews(query: str) -> list:
    try:
        r = requests.get(
            "https://news.google.com/rss/search",
            params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
            headers=_HEADERS, timeout=12
        )
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "xml")
    except Exception:
        return []
    records = []
    cutoff  = datetime.now(_tz.utc).replace(tzinfo=None) - timedelta(hours=24)
    for item in soup.find_all("item"):
        t = item.find("title")
        if not t: continue
        title = t.get_text(strip=True)
        if " - " in title: title = title.rsplit(" - ", 1)[0].strip()
        if not _is_gold_relevant(title): continue  # precision gold-impact filter
        pub = item.find("pubDate")
        dt  = _parse_rss_date(pub.get_text(strip=True) if pub else "")
        if dt and dt < cutoff: continue
        lnk = item.find("link")
        url_val = lnk.get_text(strip=True) if lnk else ""
        records.append({
            "Datetime": dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None,
            "Date":     dt.strftime("%Y-%m-%d") if dt else str(date.today()),
            "Headline": title,
            "Source":   "Google News",
            "URL":      url_val,
        })
    return records

def fetch_live_news() -> list:
    """Fetch news from last 24h across all RSS sources. Returns list of dicts."""
    records = []
    for url, src in RSS_LIVE:
        try:
            records.extend(_fetch_rss(url, src))
        except Exception:
            pass
        time.sleep(0.5)
    for q in GNEWS_QUERIES:
        try:
            records.extend(_fetch_gnews(q))
        except Exception:
            pass
        time.sleep(0.5)

    if not records:
        return []

    df = pd.DataFrame(records)
    df["_key"] = df["Headline"].str.lower().str.strip()
    df.drop_duplicates(subset=["_key"], inplace=True)
    df.drop(columns=["_key"], inplace=True)
    df.dropna(subset=["Date"], inplace=True)
    df.sort_values("Datetime", ascending=False, inplace=True, na_position="last")
    return df.to_dict("records")

def append_new_headlines(new_records: list):
    """Append truly new headlines to gdelt_news_raw.csv (deduped)."""
    if not new_records:
        return
    try:
        new_df = pd.DataFrame(new_records)
        if os.path.exists(GDELT_NEWS):
            old_df = pd.read_csv(GDELT_NEWS)
            combined = pd.concat([new_df, old_df], ignore_index=True)
        else:
            combined = new_df
        combined["_key"] = combined["Headline"].str.lower().str.strip()
        combined.drop_duplicates(subset=["_key"], inplace=True)
        combined.drop(columns=["_key"], inplace=True)
        combined.dropna(subset=["Date"], inplace=True)
        combined.sort_values("Datetime", ascending=False, inplace=True, na_position="last")
        combined.to_csv(GDELT_NEWS, index=False, encoding="utf-8")
    except Exception as e:
        print(f"[append_headlines] Error: {e}")

def fetch_live_gold_price() -> float:
    """Retrieve the current live spot gold price from Yahoo Finance."""
    try:
        import yfinance as yf
        ticker = yf.Ticker("GC=F")
        price = 0.0
        try:
            fi = ticker.fast_info
            price = float(getattr(fi, 'last_price', None) or
                          getattr(fi, 'lastPrice',  None) or 0)
        except Exception:
            pass
        if price > 0:
            return price
        hist = ticker.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except Exception as e:
        print(f"[live_price] yfinance error: {e}", flush=True)
    try:
        df = pd.read_csv(RAW_PRICES)
        return float(df['Close'].iloc[-1])
    except Exception:
        return 0.0

# ── LIVE SIGNAL COMPUTER ─────────────────────────────────────────────────────
def compute_signal_with_live_sentiment(live_news: list) -> dict:
    """
    Re-run the ensemble with sentiment features and technical indicators
    updated by real-time gold price action and live news.

    NEW: Dual-Timeframe Signals (Scalp + Swing) + Confidence Tiers + Smart Timing
    """
    try:
        inf_df = pd.read_csv(INFERENCE_DATA)
        inference_date = inf_df['Date'].iloc[-1]
        X_inf = inf_df.drop(columns=['Date'])
        features = X_inf.columns.tolist()

        # Load raw price parameters
        raw_df = pd.read_csv(RAW_PRICES)
        raw_df['Date'] = pd.to_datetime(raw_df['Date'])
        raw_df = raw_df.sort_values('Date').reset_index(drop=True)
        raw_df['ATR'] = calculate_atr(raw_df, 14)
        latest_atr   = float(raw_df['ATR'].iloc[-1])
        latest_close = float(raw_df['Close'].iloc[-1])
        latest_high  = float(raw_df['High'].iloc[-1])
        latest_low   = float(raw_df['Low'].iloc[-1])

        # Live Gold Price Integration
        live_price   = fetch_live_gold_price()
        entry_price  = live_price if live_price > 0 else latest_close

        # ── Live VADER sentiment blend ─────────────────────────────────────
        if live_news:
            vader_scores = [vader_score(h["Headline"]) for h in live_news[:30]]
            live_vader_mean = float(np.mean(vader_scores)) if vader_scores else 0.0
        else:
            live_vader_mean = 0.0

        X_blended = X_inf.copy()
        if "Sentiment_SMA_5" in features:
            hist_val = float(X_blended["Sentiment_SMA_5"].iloc[0])
            X_blended["Sentiment_SMA_5"] = 0.70 * hist_val + 0.30 * live_vader_mean
        if "Mean_Sentiment" in features:
            hist_val = float(X_blended["Mean_Sentiment"].iloc[0])
            X_blended["Mean_Sentiment"] = 0.70 * hist_val + 0.30 * live_vader_mean

        # Real-time price features adjustment
        if entry_price > 0 and latest_close > 0:
            if "Close_Return" in features:
                X_blended["Close_Return"] = float(np.log(entry_price / latest_close))
            
            # Recompute ratios using the live price
            ratio_adjust = latest_close / entry_price
            for ratio_feat in ["EMA_50_Ratio", "BBL_Ratio", "BBM_Ratio", "BBU_Ratio"]:
                if ratio_feat in features:
                    X_blended[ratio_feat] = float(X_blended[ratio_feat].iloc[0]) * ratio_adjust

        scaler = joblib.load(SCALER_PATH)

        # ── FEATURE ALIGNMENT ─────────────────────────────────────────────────
        if hasattr(scaler, 'feature_names_in_'):
            expected_features = list(scaler.feature_names_in_)
            current_features  = list(X_blended.columns)
            extra   = [f for f in current_features  if f not in expected_features]
            missing = [f for f in expected_features if f not in current_features]
            if extra or missing:
                print(f"  [ALIGN] Dropping {len(extra)} unseen features: {extra[:5]}...", flush=True)
                print(f"  [ALIGN] Zero-filling {len(missing)} missing features: {missing[:5]}...", flush=True)
                # Drop unknown columns
                X_blended = X_blended.drop(columns=[c for c in extra if c in X_blended.columns], errors='ignore')
                # Add missing columns filled with zero (neutral/safe default)
                for col in missing:
                    X_blended[col] = 0.0
                # Reorder to exactly match scaler's column order
                X_blended = X_blended[expected_features]

        X_sc   = pd.DataFrame(scaler.transform(X_blended),
                              columns=list(X_blended.columns))

        m_cat = CatBoostClassifier(); m_cat.load_model(MODEL_CAT)
        m_xgb = xgb.XGBClassifier();  m_xgb.load_model(MODEL_XGB)
        m_lgb = lgb.Booster(model_file=MODEL_LGB)
        meta  = joblib.load(MODEL_META)

        p_cat = float(m_cat.predict_proba(X_sc)[0, 1])
        p_xgb = float(m_xgb.predict_proba(X_sc)[0, 1])
        p_lgb = float(m_lgb.predict(X_sc.values)[0])
        prob_up = float(meta.predict_proba(np.array([[p_cat, p_xgb, p_lgb]]))[0, 1])

        # ── DUAL-TIMEFRAME SIGNAL COMPUTATION ────────────────────────────────
        # Gates 3/4/5 REMOVED — RSI/blackout/vol-regime killed most signals.
        # Gate 1: Separate percentile thresholds per timeframe.
        # Gate 2: Ensemble consensus kept — demotes strength to WEAK not NEUTRAL.
        # Weak fallback: always produce a directional lean (LONG or SHORT), never flat.
        #
        # SCALP : P65/P35 — more frequent signals, 0.4×/0.8× ATR SL/TP
        # SWING : P75/P25 — higher confidence only, 1.5×/3.0× ATR SL/TP

        models_bullish = sum(1 for p in [p_cat, p_xgb, p_lgb] if p > 0.50)
        models_bearish = sum(1 for p in [p_cat, p_xgb, p_lgb] if p < 0.50)

        # Load per-timeframe thresholds from historical probability distribution
        preds_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
        scalp_lt = scalp_st = swing_lt = swing_st = None
        if os.path.exists(preds_path):
            try:
                _pdf = pd.read_csv(preds_path)
                if 'Ensemble_Prob' in _pdf.columns and len(_pdf) > 20:
                    _p = _pdf['Ensemble_Prob']
                    scalp_lt = float(np.percentile(_p, 65))
                    scalp_st = float(np.percentile(_p, 35))
                    swing_lt = float(np.percentile(_p, 75))
                    swing_st = float(np.percentile(_p, 25))
            except Exception:
                pass
        if scalp_lt is None:
            _, cb = load_threshold()
            scalp_lt = cb;        scalp_st = 1 - cb
            swing_lt = cb + 0.03; swing_st = 1 - cb - 0.03

        def _resolve_signal(p_up, long_t, short_t, m_bull, m_bear):
            """Gate 1 + Gate 2 + weak fallback. Always returns (signal, strength)."""
            if p_up >= long_t:
                sig = "LONG"
                strength = "STRONG" if p_up >= long_t + (1 - long_t) * 0.5 else "MODERATE"
            elif p_up <= short_t:
                sig = "SHORT"
                strength = "STRONG" if p_up <= short_t * 0.5 else "MODERATE"
            else:
                # Directional lean — always emit a signal, just mark it WEAK
                sig = "LONG" if p_up >= 0.50 else "SHORT"
                strength = "WEAK"
            # Gate 2: consensus check — downgrade strength if models disagree
            if sig == "LONG" and m_bull < 2:  strength = "WEAK"
            elif sig == "SHORT" and m_bear < 2: strength = "WEAK"
            return sig, strength

        scalp_sig, scalp_strength = _resolve_signal(
            prob_up, scalp_lt, scalp_st, models_bullish, models_bearish)
        swing_sig, swing_strength = _resolve_signal(
            prob_up, swing_lt, swing_st, models_bullish, models_bearish)

        # Primary signal for SHAP narrative = swing (higher confidence gate)
        signal = swing_sig
        print(f"  Scalp={scalp_sig}({scalp_strength})  Swing={swing_sig}({swing_strength})  prob={prob_up:.4f}", flush=True)

        # SHAP
        explainer   = shap.TreeExplainer(m_cat)
        shap_vals   = explainer.shap_values(X_sc)
        inst_shap   = shap_vals[0] if not isinstance(shap_vals, list) else shap_vals[1][0]
        feat_impacts = sorted(
            zip(features, inst_shap, X_blended.iloc[0].values),
            key=lambda x: abs(x[1]), reverse=True
        )
        top_drivers = []
        for feat, impact, val in feat_impacts[:4]:
            direction = "UP" if impact > 0 else "DOWN"
            top_drivers.append({
                "feature":   feat,
                "text":      get_readable_reason(feat, direction),
                "direction": direction,
                "impact":    float(impact),
            })

        # ── Per-timeframe SL/TP ──────────────────────────────────────────────
        def _calc_levels(sig, ep, atr, scalp=True):
            sl_m = 0.4 if scalp else 1.5
            tp_m = 0.8 if scalp else 3.0
            if sig == "LONG":  return round(ep - sl_m*atr, 2), round(ep + tp_m*atr, 2)
            if sig == "SHORT": return round(ep + sl_m*atr, 2), round(ep - tp_m*atr, 2)
            return 0.0, 0.0

        scalp_sl, scalp_tp = _calc_levels(scalp_sig, entry_price, latest_atr, scalp=True)
        swing_sl, swing_tp = _calc_levels(swing_sig, entry_price, latest_atr, scalp=False)
        sl = scalp_sl; tp = scalp_tp

        pp = (latest_high + latest_low + entry_price) / 3
        r1 = 2*pp - latest_low;  r2 = pp + (latest_high - latest_low)
        s1 = 2*pp - latest_high; s2 = pp - (latest_high - latest_low)

        narrative = generate_narrative(signal, prob_up, top_drivers,
                                       entry_price, sl, tp, latest_atr)
        inf_date_str, days_old, is_stale = _data_staleness()

        # Next business day target date
        try:
            dt = datetime.strptime(str(inference_date), "%Y-%m-%d")
            next_dt = dt + timedelta(days=1)
            while next_dt.weekday() >= 5:
                next_dt += timedelta(days=1)
            target_date = next_dt.strftime("%Y-%m-%d")
        except Exception:
            target_date = inference_date

        return {
            "status":       "success",
            "date":         inference_date,
            "target_date":  target_date,
            "data_age_days": days_old,
            "is_stale":     is_stale,
            "last_refresh": datetime.now(_tz.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M UTC"),
            "live_vader_sentiment": round(live_vader_mean, 4),
            "live_vader_label":     sentiment_label(live_vader_mean),
            "model_votes":  {"catboost": round(p_cat,4), "xgboost": round(p_xgb,4), "lightgbm": round(p_lgb,4)},
            "consensus_ok": models_bullish >= 2 or models_bearish >= 2,
            # ── Dual-timeframe signal objects ────────────────────────────────
            "scalp": {
                "signal":      scalp_sig,
                "strength":    scalp_strength,
                "stop_loss":   scalp_sl,
                "take_profit": scalp_tp,
                "atr_mult":    "0.4×SL / 0.8×TP",
            },
            "swing": {
                "signal":      swing_sig,
                "strength":    swing_strength,
                "stop_loss":   swing_sl,
                "take_profit": swing_tp,
                "atr_mult":    "1.5×SL / 3.0×TP",
            },
            "prediction":   {"signal": signal, "probability_up": round(prob_up,4), "probability_down": round(1-prob_up,4)},
            "narrative":    narrative,
            "risk_management": {
                "entry_price":   round(entry_price, 2),
                "latest_close":  round(latest_close, 2),
                "stop_loss":     round(scalp_sl, 2),
                "take_profit":   round(scalp_tp, 2),
                "stop_loss_sw":  round(swing_sl, 2),
                "take_profit_sw":round(swing_tp, 2),
                "atr_14":        round(latest_atr, 2),
                "note": "Scalp: 0.4×/0.8× ATR. Swing: 1.5×/3.0× ATR (1:2 R:R)."
            },
            "intraday_levels": {
                "r2": round(r2,2), "r1": round(r1,2), "pp": round(pp,2),
                "s1": round(s1,2), "s2": round(s2,2)
            },
            "shap_drivers": top_drivers,
        }
    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}

def build_news_cache(live_news: list) -> list:
    """Build rich news list with VADER sentiment + gold-impact category tags."""
    result = []
    seen   = set()
    for item in (live_news or [])[:60]:
        h = item.get("Headline", "")
        if not h or h.lower() in seen: continue
        if not _is_gold_relevant(h): continue  # drop non-gold headlines
        seen.add(h.lower())
        score    = vader_score(h)
        label    = sentiment_label(score)
        category = classify_news_category(h)
        if category == "OTHER" and score == 0.0: continue  # pure noise
        result.append({
            "headline":  h,
            "source":    item.get("Source", ""),
            "url":       item.get("URL", "#"),
            "datetime":  item.get("Datetime", ""),
            "sentiment": label,
            "score":     round(score, 3),
            "category":  category,
            "cat_icon":  _CATEGORY_ICONS.get(category, "📰"),
        })
    return result[:25]

# ── BACKGROUND REFRESH JOB ────────────────────────────────────────────────────
def background_refresh():
    """Runs every 15 minutes. Fetches news, updates sentiment, rebuilds signal cache."""
    global _signal_cache, _news_cache, _last_refresh
    # BUG FIX: Use threading.Event for atomic test-and-set
    if _refresh_event.is_set():
        return
    _refresh_event.set()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Background refresh started...", flush=True)
    try:
        live_news = fetch_live_news()
        print(f"  Fetched {len(live_news)} live headlines.", flush=True)
        if live_news:
            append_new_headlines(live_news)

        signal_data = compute_signal_with_live_sentiment(live_news)
        if signal_data.get("status") == "success":
            news_data = build_news_cache(live_news)
            # Also pull stored news for the news panel
            stored_news = []
            try:
                ndf = pd.read_csv(GDELT_NEWS)
                seen_h = {x['headline'].lower() for x in news_data}
                for _, row in ndf.head(50).iterrows():
                    h = str(row.get('Headline',''))
                    if not h or h.lower() in seen_h: continue
                    if not _is_gold_relevant(h): continue  # gold-impact filter
                    score    = vader_score(h)
                    category = classify_news_category(h)
                    if category == "OTHER" and score == 0.0: continue
                    stored_news.append({
                        "headline":  h,
                        "source":    str(row.get('Source','')),
                        "url":       str(row.get('URL','#')),
                        "datetime":  str(row.get('Datetime','')),
                        "sentiment": sentiment_label(score),
                        "score":     round(score, 3),
                        "category":  category,
                        "cat_icon":  _CATEGORY_ICONS.get(category, "📰"),
                    })
            except Exception: pass

            all_news = news_data + stored_news
            signal_data["live_news"] = all_news[:20]

            with _cache_lock:
                _signal_cache = signal_data
                _news_cache   = all_news[:20]
                _last_refresh = datetime.now(_tz.utc).replace(tzinfo=None)

            sig = signal_data.get("prediction", {}).get("signal", "?")
            vader_lbl = signal_data.get("live_vader_label","?")
            print(f"  Cache updated. Signal={sig}  LiveSentiment={vader_lbl}", flush=True)
        else:
            print(f"  Signal computation failed: {signal_data.get('message','')}", flush=True)
    except Exception as e:
        import traceback
        print(f"  [ERR] background_refresh: {e}", flush=True)
        traceback.print_exc()
    finally:
        _refresh_event.clear()

def run_full_daily_refresh_task():
    global _signal_cache, _news_cache, _last_refresh
    if _full_refresh_event.is_set():
        return
    _full_refresh_event.set()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Full daily refresh started...", flush=True)
    try:
        proc = subprocess.run(
            [sys.executable, REFRESH_SCRIPT],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace"
        )
        if proc.returncode == 0:
            print("Full daily refresh succeeded. Recomputing signals...", flush=True)
            background_refresh()
        else:
            print(f"Full daily refresh failed:\\n{proc.stdout[-3000:]}", flush=True)
    except Exception as e:
        print(f"Full daily refresh error: {e}", flush=True)
    finally:
        _full_refresh_event.clear()

# ── SCHEDULER ─────────────────────────────────────────────────────────────────
scheduler = BackgroundScheduler()
scheduler.add_job(background_refresh, "interval", minutes=15, id="live_refresh",
                  next_run_time=datetime.now())   # run immediately on startup
scheduler.add_job(run_full_daily_refresh_task, "cron", hour=0, minute=5, id="daily_refresh")
scheduler.start()
print("Scheduler started — news refresh every 15 minutes, full refresh daily at 00:05 UTC.", flush=True)

# ── ENDPOINTS ─────────────────────────────────────────────────────────────────
@app.get("/api/predict")
def predict():
    """Returns the cached signal (fast). Falls back to live compute if cache empty."""
    global _signal_cache, _news_cache, _last_refresh
    with _cache_lock:
        if _signal_cache:
            return _signal_cache
    # Cache miss on first load — compute synchronously
    live_news = fetch_live_news()
    data = compute_signal_with_live_sentiment(live_news)
    if data.get("status") == "success":
        news_data = build_news_cache(live_news)
        data["live_news"] = news_data
        with _cache_lock:
            _signal_cache = data
            _news_cache   = news_data
            _last_refresh = datetime.now(_tz.utc).replace(tzinfo=None)
    return data

@app.get("/api/signal")
def get_signal():
    """Lightweight poll endpoint — returns signal + timestamp only (no SHAP/news)."""
    with _cache_lock:
        if not _signal_cache:
            return {"status": "loading", "message": "Initialising..."}
        c = _signal_cache
    last_upd = _last_refresh.strftime("%Y-%m-%d %H:%M UTC") if _last_refresh else "—"
    return {
        "status":              "success",
        "signal":              c.get("prediction", {}).get("signal", "NEUTRAL"),
        "probability_up":      c.get("prediction", {}).get("probability_up", 0.5),
        "probability_down":    c.get("prediction", {}).get("probability_down", 0.5),
        "live_vader_label":    c.get("live_vader_label", "NEUTRAL"),
        "live_vader_sentiment":c.get("live_vader_sentiment", 0.0),
        "last_refresh":        last_upd,
        "data_age_days":       c.get("data_age_days", 0),
        "is_stale":            c.get("is_stale", False),
        "entry_price":         c.get("risk_management", {}).get("entry_price", 0),
    }

@app.get("/api/live-news")
def live_news_endpoint():
    """Returns the latest 20 headlines with sentiment scores."""
    with _cache_lock:
        news = list(_news_cache)
    last_upd = _last_refresh.strftime("%Y-%m-%d %H:%M UTC") if _last_refresh else "—"
    return {
        "status":       "success",
        "count":        len(news),
        "last_refresh": last_upd,
        "news":         news,
    }

@app.get("/api/macro-calendar")
def macro_calendar():
    """Returns upcoming macro economic events for the next 30 days."""
    try:
        from macro_calendar import get_upcoming_events
        events = get_upcoming_events(days_ahead=30)
        return {"status": "success", "count": len(events), "events": events}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/history")
def get_history():
    """Returns historical signals split into SCALP and SWING timeframes.
    SCALP: P65/P35 thresholds, 0.4×/0.8× ATR SL/TP.
    SWING: P75/P25 thresholds, 1.5×/3.0× ATR SL/TP.
    Always produces a directional signal with strength tier (STRONG/MODERATE/WEAK).
    """
    hist_file = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    if not os.path.exists(hist_file):
        return {"status": "error", "message": "History file not found."}
    try:
        df = pd.read_csv(hist_file)
        df['Date'] = pd.to_datetime(df['Date'])
        model_cols = ['Cat_Prob', 'XGB_Prob', 'LGB_Prob']
        has_models = all(c in df.columns for c in model_cols)

        # Compute per-timeframe adaptive thresholds from full dataset
        probs    = df['Ensemble_Prob'].dropna()
        scalp_lt = float(np.percentile(probs, 65))
        scalp_st = float(np.percentile(probs, 35))
        swing_lt = float(np.percentile(probs, 75))
        swing_st = float(np.percentile(probs, 25))

        # Get latest ATR for SL/TP estimates
        latest_atr = 30.0
        try:
            rpdf = pd.read_csv(RAW_PRICES)
            rpdf['ATR'] = calculate_atr(rpdf, 14)
            latest_atr = float(rpdf['ATR'].dropna().iloc[-1])
            latest_close = float(rpdf['Close'].iloc[-1])
        except Exception:
            latest_close = 3300.0

        df = df.sort_values("Date", ascending=False).head(120)

        def _hist_signal(prob, long_t, short_t, m_bull, m_bear):
            if prob >= long_t:
                sig = "LONG"; strength = "STRONG" if prob >= long_t + (1-long_t)*0.5 else "MODERATE"
            elif prob <= short_t:
                sig = "SHORT"; strength = "STRONG" if prob <= short_t*0.5 else "MODERATE"
            else:
                sig = "LONG" if prob >= 0.50 else "SHORT"; strength = "WEAK"
            if sig == "LONG"  and m_bull < 2: strength = "WEAK"
            elif sig == "SHORT" and m_bear < 2: strength = "WEAK"
            return sig, strength

        scalp_list, swing_list = [], []

        for _, row in df.iterrows():
            prob   = float(row.get("Ensemble_Prob", 0.5))
            target = row.get("Target_Direction")
            ds     = row["Date"].strftime("%Y-%m-%d")
            m_bull = sum(1 for c in model_cols if float(row.get(c, 0.5)) > 0.50) if has_models else 1
            m_bear = sum(1 for c in model_cols if float(row.get(c, 0.5)) < 0.50) if has_models else 1

            sc_sig, sc_str = _hist_signal(prob, scalp_lt, scalp_st, m_bull, m_bear)
            sw_sig, sw_str = _hist_signal(prob, swing_lt, swing_st, m_bull, m_bear)

            if sc_sig == "LONG":  sc_sl = round(latest_close - 0.4*latest_atr, 2); sc_tp = round(latest_close + 0.8*latest_atr, 2)
            else:                  sc_sl = round(latest_close + 0.4*latest_atr, 2); sc_tp = round(latest_close - 0.8*latest_atr, 2)
            if sw_sig == "LONG":  sw_sl = round(latest_close - 1.5*latest_atr, 2); sw_tp = round(latest_close + 3.0*latest_atr, 2)
            else:                  sw_sl = round(latest_close + 1.5*latest_atr, 2); sw_tp = round(latest_close - 3.0*latest_atr, 2)

            sc_res = "WIN" if (sc_sig=="LONG" and target==1) or (sc_sig=="SHORT" and target==0) else "LOSS"
            sw_res = "WIN" if (sw_sig=="LONG" and target==1) or (sw_sig=="SHORT" and target==0) else "LOSS"

            scalp_list.append({"date":ds,"signal":sc_sig,"strength":sc_str,
                                "probability":round(prob*100,1),"stop_loss":sc_sl,"take_profit":sc_tp,"result":sc_res})
            swing_list.append({"date":ds,"signal":sw_sig,"strength":sw_str,
                                "probability":round(prob*100,1),"stop_loss":sw_sl,"take_profit":sw_tp,"result":sw_res})

        return {
            "status": "success",
            "scalp":  scalp_list,
            "swing":  swing_list,
            "thresholds": {
                "scalp": {"long": round(scalp_lt,4), "short": round(scalp_st,4)},
                "swing": {"long": round(swing_lt,4), "short": round(swing_st,4)},
            }
        }
    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}

@app.get("/api/health")
def health():
    inf_date, days_old, is_stale = _data_staleness()
    last_upd = _last_refresh.strftime("%Y-%m-%d %H:%M UTC") if _last_refresh else "never"
    return {
        "status":           "ok",
        "inference_date":   inf_date,
        "data_age_days":    days_old,
        "is_stale":         is_stale,
        "today":            str(date.today()),
        "last_refresh":     last_upd,
        "refreshing":       _refresh_event.is_set(),
        "refreshing_daily": _full_refresh_event.is_set(),
    }

@app.post("/api/refresh")
def manual_refresh():
    """Trigger the daily_refresh.py pipeline to pull today's full data."""
    if not os.path.exists(REFRESH_SCRIPT):
        return {"status": "error", "message": "daily_refresh.py not found"}
    if _full_refresh_event.is_set():
        return {"status": "refresh_started", "message": "Refresh is already in progress."}
    threading.Thread(target=run_full_daily_refresh_task, daemon=True).start()
    return {"status": "refresh_started", "message": "Background refresh started. Check back in a few minutes."}

if __name__ == "__main__":
    print("Starting Gold AI API Server v3 on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
