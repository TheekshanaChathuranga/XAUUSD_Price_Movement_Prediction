"""
STEP 3: Financial News Collection — Gold Price & Macro Sentiment
================================================================
ENHANCED v2: Gold-Impact Focused News Collection

Key improvements:
  ✓ 7 new gold-impact Google News queries (war, geopolitical, Fed shock, etc.)
  ✓ 5 new RSS sources (Reuters World, AP, Kitco, WGC, Investing.com)
  ✓ Gold_Category tagging on every headline
  ✓ Source quality score assigned per outlet
  ✓ Expanded keyword filter with war/conflict/military terms

Data Sources:
  Source A │ Google News RSS     │ 18 targeted queries (was 11)
  Source B │ Yahoo Finance RSS   │ Gold, Silver, Oil, DXY tickers
  Source C │ ForexLive RSS       │ Live FX / central bank / macro
  Source D │ FXStreet RSS        │ Gold, forex, ECB, Fed
  Source E │ CNBC Economy RSS    │ Inflation, Fed, US economy
  Source F │ Seeking Alpha RSS   │ Gold ETF (GLD) commentary
  Source G │ MarketWatch RSS     │ Real-time financial headlines
  Source H │ Financial Times RSS │ Global macro headlines
  Source I │ Reuters World RSS   │ Breaking geopolitical/war news ← NEW
  Source J │ Kitco Gold News     │ Dedicated gold news outlet    ← NEW
  Source K │ Investing.com RSS   │ Gold & commodity news         ← NEW
  Source L │ AP Top News RSS     │ War/conflict/US foreign policy ← NEW

Output: financial_news_raw.csv
Columns: Date, Datetime, Headline, Source, URL, Gold_Category, Source_Quality
"""

import os
import sys
import time
import re
import requests
import pandas as pd
from datetime import datetime, date
from bs4 import BeautifulSoup

# ── Windows UTF-8 encoding fix ────────────────────────────────────────────────
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "financial_news_raw.csv")
END_DATE    = date.today().strftime("%Y-%m-%d")

# ── GOLD-IMPACT KEYWORD FILTER ────────────────────────────────────────────────
# ONLY keywords that directly and materially move gold prices.
# Removed generic terms: economy, growth, pmi, retail sales, consumer confidence,
# commodity, nonfarm payroll alone (too broad — kept with gold context only).
FINANCE_KEYWORDS = [
    # Precious metals & gold market (direct)
    "gold", "xau", "bullion", "precious metal", "silver", "platinum",
    "spot gold", "gold futures", "gold etf", "gld", "gold price",
    "gold demand", "gold supply", "gold reserve", "central bank gold",
    # Macro policy — high gold impact
    "federal reserve", "fed rate", "interest rate", "fomc", "rate hike", "rate cut",
    "inflation", "cpi", "pce", "gdp contraction", "stagflation",
    "quantitative easing", "quantitative tightening", "taper", "monetary policy",
    "central bank", "powell", "lagarde", "ecb", "boe", "bank of england",
    "dovish", "hawkish", "fed funds", "ppi", "producer price",
    # Currency & bonds — gold inverse/correlation
    "dollar", "dxy", "treasury", "yield", "tips", "real yield",
    "bond yield", "10-year yield", "forex", "fx rate", "dedollarization",
    "dollar weakness", "dollar strength", "dollar dominance",
    # Geopolitical & WAR (strongest gold safe-haven driver)
    "war", "conflict", "military", "strike", "attack", "invasion",
    "sanction", "geopolit", "ukraine", "russia", "middle east",
    "iran", "israel", "north korea", "taiwan", "china tension",
    "nato", "nuclear", "missile", "escalat", "ceasefire", "coup",
    "terrorism", "civil war", "airstrike", "drone strike",
    # US Military & defense (key gold movers)
    "pentagon", "us forces", "us troops", "us military", "us army", "us navy",
    "us air force", "defense secretary", "joint chiefs", "warfare", "arms deal",
    "us strike", "american troops", "us sanctions",
    # Conflict hotspots driving gold safe-haven
    "hamas", "hezbollah", "houthi", "red sea", "strait of hormuz",
    "ukraine war", "russia sanctions", "iran nuclear", "israel Gaza",
    # Safe haven & crisis
    "safe haven", "flight to safety", "risk off", "crisis", "panic",
    "collapse", "contagion", "bail", "default", "bankrupt", "bank run",
    "financial crisis", "recession", "market crash",
    # Energy (inflation proxy, gold-correlated)
    "crude oil", "wti", "brent", "opec", "energy", "oil price", "oil shock",
    "energy crisis",
    # Trade & tariffs
    "tariff", "trade war", "embargo", "us debt", "debt ceiling",
    # Gold demand drivers
    "stagflation", "hyperinflation", "devaluation",
    "dollar weakness", "dedollarization",
]
FINANCE_PATTERN = re.compile("|".join(FINANCE_KEYWORDS), re.IGNORECASE)

# ── GOLD CATEGORY CLASSIFICATION ─────────────────────────────────────────────
# Each headline gets a category tag for downstream feature engineering.
# Ordered by priority (first match wins).
GOLD_CATEGORIES = [
    # (category_name, pattern, gold_direction_hint)
    ("WAR_GEOPOLITICAL",   re.compile(
        r"war|conflict|military|attack|invasion|strike|missile|nuclear|"
        r"nato|ukraine|russia|iran|israel|middle east|north korea|taiwan|"
        r"sanction|ceasefire|terrorism|coup|escalat", re.IGNORECASE), "bullish"),
    ("FED_POLICY",         re.compile(
        r"federal reserve|fed rate|fomc|rate hike|rate cut|powell|"
        r"hawkish|dovish|quantitative|taper|monetary policy|fed funds", re.IGNORECASE), "mixed"),
    ("INFLATION",          re.compile(
        r"inflation|cpi|pce|deflat|stagflat|price index|"
        r"consumer price|producer price|ppi", re.IGNORECASE), "bullish"),
    ("DOLLAR_FX",          re.compile(
        r"dollar|dxy|dollar index|dollar strength|dollar weakness|"
        r"dedollar|currency|devaluat|forex|fx rate", re.IGNORECASE), "bearish_usd_up"),
    ("RECESSION_CRISIS",   re.compile(
        r"recession|crisis|panic|collapse|contagion|bankrupt|default|"
        r"bail.?out|financial crisis|market crash|bear market", re.IGNORECASE), "bullish"),
    ("TREASURY_YIELDS",    re.compile(
        r"treasury|bond yield|10.year|tips|real yield|yield curve|"
        r"bond market|us bond", re.IGNORECASE), "bearish"),
    ("ENERGY_OIL",         re.compile(
        r"crude oil|wti|brent|opec|oil price|energy|petroleum", re.IGNORECASE), "mixed"),
    ("GOLD_MARKET",        re.compile(
        r"gold price|xauusd|spot gold|gold futures|gold etf|gld|"
        r"bullion|precious metal|gold demand|gold supply", re.IGNORECASE), "direct"),
    ("MACRO_ECONOMY",      re.compile(
        r"gdp|nonfarm|payroll|unemployment|jobs|economic|economy|"
        r"growth|pmi|retail sales|consumer confidence", re.IGNORECASE), "mixed"),
]

def classify_gold_category(headline: str) -> str:
    """Assign a gold-impact category to a headline."""
    for cat_name, pattern, _ in GOLD_CATEGORIES:
        if pattern.search(headline):
            return cat_name
    return "OTHER_FINANCE"

# ── SOURCE QUALITY SCORES ─────────────────────────────────────────────────────
# 1.0 = highest quality, 0.5 = lower quality
SOURCE_QUALITY = {
    "Reuters":          1.0,
    "AP News":          1.0,
    "Bloomberg":        1.0,
    "Financial Times":  0.95,
    "CNBC":             0.90,
    "Kitco":            0.90,
    "WSJ":              0.90,
    "MarketWatch":      0.85,
    "Investing.com":    0.80,
    "FXStreet":         0.80,
    "ForexLive":        0.80,
    "Yahoo Finance":    0.75,
    "Seeking Alpha":    0.65,
    "Google News":      0.70,
    "GDELT":            0.60,
}

def get_source_quality(source: str) -> float:
    """Return quality weight for a given source name."""
    for key, score in SOURCE_QUALITY.items():
        if key.lower() in source.lower():
            return score
    return 0.65

# ── HTTP HEADERS ──────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

# ── DATE PARSER ───────────────────────────────────────────────────────────────
_DATE_FORMATS = [
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%SZ",
    "%a, %d %b %Y %H:%M:%S +0000",
    "%a, %d %b %Y %H:%M:%S GMT",
]

def _parse_dt(pub_raw: str):
    """Parse an RSS pubDate string into a datetime (or None)."""
    if not pub_raw:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(pub_raw[:len(fmt) + 5].strip(), fmt)
        except Exception:
            pass
    clean = re.sub(r"\s+[A-Z]{2,5}$", "", pub_raw.strip())
    try:
        return datetime.strptime(clean[:25], "%a, %d %b %Y %H:%M:%S")
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE A: GOOGLE NEWS RSS — ENHANCED (18 queries, was 11)
# ─────────────────────────────────────────────────────────────────────────────
GOOGLE_NEWS_QUERIES = [
    # ── Original 11 (kept) ──────────────────────────────────────────────────
    ("Gold market",              "gold bullion commodity market"),
    ("Gold silver precious",     "gold silver precious metal"),
    ("Gold inflation dollar",    "gold inflation hedge dollar"),
    ("Federal Reserve policy",   "federal reserve interest rate"),
    ("FOMC Powell decision",     "FOMC rate decision Powell"),
    ("Inflation CPI data",        "CPI inflation data PCE expectations gold"),
    ("Treasury yield bonds",      "treasury yield bond gold recession"),
    ("Crude oil OPEC energy",     "crude oil OPEC energy crisis gold"),
    ("Dollar DXY forex",          "dollar DXY weakness strength gold"),
    ("Geopolitical safe haven",   "geopolitical risk safe haven gold"),
    ("ECB Bank of England",       "ECB Bank of England central bank rate gold"),

    # ── NEW: War & Conflict (highest gold impact) ────────────────────────────
    ("US Military strike gold",   "US military strike airstrike gold safe haven"),
    ("War conflict gold safe",    "war military conflict attack gold"),
    ("Middle East gold",          "Middle East Israel Iran conflict oil gold"),
    ("Russia Ukraine conflict",   "Russia Ukraine war sanction commodity gold"),
    ("US China tension",          "US China tariff trade war tension dollar gold"),
    ("Pentagon US forces gold",   "Pentagon US forces Middle East gold safe haven"),
    ("Iran nuclear gold",         "Iran nuclear deal sanction dollar gold"),

    # ── NEW: US Foreign Policy & Dollar Weaponization ────────────────────────
    ("US sanctions dollar gold",  "US sanctions trade war dollar gold"),
    ("Dedollarization gold",      "dedollarization US dollar gold reserves"),
    ("Debt ceiling dollar gold",  "US debt ceiling dollar default gold safe"),

    # ── NEW: Extreme macro events ────────────────────────────────────────────
    ("Fed emergency cut",         "Federal Reserve emergency rate cut crisis gold"),
    ("Inflation shock gold",      "inflation shock CPI hot beats expectations gold"),
    ("Dollar collapse gold",      "dollar weakness DXY decline gold surge safe"),

    # ── NEW: Gold demand / institutional / geopolitical ──────────────────────
    ("Central bank gold buying",  "central bank gold reserves buying"),
    ("Gold demand crisis",        "gold demand uncertainty crisis safe haven"),
    ("Geopolitical gold price",   "geopolitical tension conflict gold price"),
]

GNEWS_BASE = "https://news.google.com/rss/search"


def _fetch_gnews(query: str, label: str) -> list:
    """Fetch one Google News RSS query. Returns list of record dicts."""
    params = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    try:
        resp = requests.get(GNEWS_BASE, params=params, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        print(f"    [WARN] Google News '{label}': {exc}")
        return []

    try:
        soup = BeautifulSoup(resp.content, "xml")
    except Exception:
        soup = BeautifulSoup(resp.content, "html.parser")

    records = []
    for item in soup.find_all("item"):
        title_tag = item.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""
        publisher = ""
        if " - " in title:
            title, publisher = title.rsplit(" - ", 1)
            publisher = publisher.strip()
        title = title.strip()

        if not title or not FINANCE_PATTERN.search(title):
            continue

        pub_tag  = item.find("pubDate")
        pub_raw  = pub_tag.get_text(strip=True) if pub_tag else ""
        dt       = _parse_dt(pub_raw)
        dt_str   = dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None
        date_str = dt.strftime("%Y-%m-%d")           if dt else None

        link_tag = item.find("link")
        link     = link_tag.get_text(strip=True) if link_tag else ""

        source   = publisher if publisher else "Google News"
        category = classify_gold_category(title)
        quality  = get_source_quality(source)

        records.append({
            "Datetime":       dt_str,
            "Date":           date_str,
            "Headline":       title,
            "Source":         source,
            "URL":            link.strip(),
            "Gold_Category":  category,
            "Source_Quality": quality,
        })
    return records


def collect_google_news() -> pd.DataFrame:
    print("\n[Google News] Fetching headlines across targeted gold-impact queries ...")
    all_records = []
    for label, query in GOOGLE_NEWS_QUERIES:
        records = _fetch_gnews(query, label)
        print(f"  {label:35s}: {len(records):4d} headlines")
        all_records.extend(records)
        time.sleep(1.0)

    df = pd.DataFrame(all_records) if all_records else pd.DataFrame(
        columns=["Date", "Datetime", "Headline", "Source", "URL", "Gold_Category", "Source_Quality"])
    print(f"[Google News] Subtotal (pre-dedup): {len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SOURCES B–L: DIRECT RSS FEEDS (enhanced)
# ─────────────────────────────────────────────────────────────────────────────
RSS_SOURCES = [
    # ── Yahoo Finance ticker feeds ──────────────────────────────────────────
    ("https://finance.yahoo.com/rss/headline?s=GC=F",     "Yahoo Finance", False),
    ("https://finance.yahoo.com/rss/headline?s=SI=F",     "Yahoo Finance", False),
    ("https://finance.yahoo.com/rss/headline?s=CL=F",     "Yahoo Finance", False),
    ("https://finance.yahoo.com/rss/headline?s=DX-Y.NYB", "Yahoo Finance", False),
    # ── ForexLive ───────────────────────────────────────────────────────────
    ("https://www.forexlive.com/feed/news",               "ForexLive",     True),
    # ── FXStreet ────────────────────────────────────────────────────────────
    ("https://www.fxstreet.com/rss/news",                 "FXStreet",      True),
    # ── CNBC Economy ────────────────────────────────────────────────────────
    ("https://www.cnbc.com/id/20910258/device/rss/rss.html", "CNBC",      True),
    # ── Seeking Alpha (GLD ETF) ─────────────────────────────────────────────
    ("https://seekingalpha.com/api/sa/combined/GLD.xml",  "Seeking Alpha", True),
    # ── MarketWatch ─────────────────────────────────────────────────────────
    ("https://feeds.marketwatch.com/marketwatch/marketpulse/",        "MarketWatch", True),
    ("https://feeds.marketwatch.com/marketwatch/realtimeheadlines/",  "MarketWatch", True),
    # ── Financial Times ─────────────────────────────────────────────────────
    ("https://www.ft.com/rss/home",                       "Financial Times", True),

    # ── NEW: Reuters World (breaking geopolitical/war news) ─────────────────
    ("https://feeds.reuters.com/Reuters/worldNews",       "Reuters",       True),
    ("https://feeds.reuters.com/reuters/businessNews",    "Reuters",       True),
    ("https://feeds.reuters.com/reuters/topNews",         "Reuters",       True),

    # ── NEW: Kitco Gold News (dedicated gold outlet) ─────────────────────────
    ("https://www.kitco.com/rss/kitco-news.xml",          "Kitco",         False),

    # ── NEW: Investing.com gold & commodities ────────────────────────────────
    ("https://www.investing.com/rss/news_14.rss",         "Investing.com", True),  # commodities
    ("https://www.investing.com/rss/news_25.rss",         "Investing.com", True),  # forex
    ("https://www.investing.com/rss/news_1.rss",          "Investing.com", True),  # top news

    # ── NEW: AP News (war/conflict/US foreign policy) ────────────────────────
    ("https://rsshub.app/ap/topics/apf-topnews",          "AP News",       True),
    ("https://feeds.apnews.com/rss/topnews",              "AP News",       True),

    # ── GEOPOLITICAL / MILITARY (key gold safe-haven drivers) ────────────────
    ("https://www.aljazeera.com/xml/rss/all.xml",         "Al Jazeera",   True),   # geopolitical conflicts
]


def _parse_rss(url: str, source_name: str, apply_filter: bool) -> list:
    """Download and parse one RSS/Atom feed. Returns list of record dicts."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        print(f"    [WARN] {source_name}: {exc}")
        return []

    try:
        try:
            soup = BeautifulSoup(resp.content, "xml")
        except Exception:
            soup = BeautifulSoup(resp.content, "html.parser")
        items = soup.find_all("item") or soup.find_all("entry")
    except Exception as exc:
        print(f"    [WARN] {source_name} parse error: {exc}")
        return []

    records = []
    for item in items:
        title_tag = item.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""
        title = re.sub(r"<[^>]+>", "", title).strip()
        if not title:
            continue
        if apply_filter and not FINANCE_PATTERN.search(title):
            continue

        pub_tag = (item.find("pubDate") or item.find("published")
                   or item.find("updated") or item.find("dc:date"))
        pub_raw  = pub_tag.get_text(strip=True) if pub_tag else ""
        dt       = _parse_dt(pub_raw)
        dt_str   = dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None
        date_str = dt.strftime("%Y-%m-%d")           if dt else None

        link_tag = item.find("link")
        if link_tag:
            link = link_tag.get("href") or link_tag.get_text(strip=True) or ""
        else:
            link = ""

        category = classify_gold_category(title)
        quality  = get_source_quality(source_name)

        records.append({
            "Datetime":       dt_str,
            "Date":           date_str,
            "Headline":       title,
            "Source":         source_name,
            "URL":            link.strip(),
            "Gold_Category":  category,
            "Source_Quality": quality,
        })
    return records


def collect_rss_feeds() -> pd.DataFrame:
    """Collect headlines from all direct RSS feeds."""
    print("\n[RSS Feeds] Collecting from direct publisher feeds ...")
    all_records = []
    for url, name, apply_filter in RSS_SOURCES:
        records = _parse_rss(url, name, apply_filter)
        print(f"  {name:25s}: {len(records):4d} headlines")
        all_records.extend(records)
        time.sleep(1.5)

    df = pd.DataFrame(all_records) if all_records else pd.DataFrame(
        columns=["Date", "Datetime", "Headline", "Source", "URL", "Gold_Category", "Source_Quality"])
    print(f"[RSS Feeds] Subtotal (pre-dedup): {len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# COMBINE & CLEAN
# ─────────────────────────────────────────────────────────────────────────────
def combine_and_clean(*dfs: pd.DataFrame) -> pd.DataFrame:
    valid = [d for d in dfs if d is not None and not d.empty]
    if not valid:
        print("[WARN] No headlines collected from any source.")
        return pd.DataFrame(columns=["Date", "Datetime", "Headline", "Source",
                                     "URL", "Gold_Category", "Source_Quality"])

    df = pd.concat(valid, ignore_index=True)

    for col in ["Date", "Datetime", "Headline", "Source", "URL",
                "Gold_Category", "Source_Quality"]:
        if col not in df.columns:
            df[col] = None

    df = df[["Date", "Datetime", "Headline", "Source", "URL",
             "Gold_Category", "Source_Quality"]].copy()

    df = df[df["Headline"].notna() & (df["Headline"].str.strip() != "")]

    # Deduplicate on normalised headline text
    df["_key"] = df["Headline"].str.lower().str.strip()
    df.drop_duplicates(subset=["_key"], inplace=True)
    df.drop(columns=["_key"], inplace=True)

    df.dropna(subset=["Date"], inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df.dropna(subset=["Date"], inplace=True)

    df.sort_values("Date", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"\n{'='*65}")
    print(f"  COMBINED RESULTS — GOLD-IMPACT ENHANCED")
    print(f"{'='*65}")
    print(f"  Total unique headlines : {len(df):,}")
    if len(df) > 0:
        print(f"  Date range            : {df['Date'].min()} -> {df['Date'].max()}")
        print(f"\n  By Gold Category:")
        for cat, cnt in df["Gold_Category"].value_counts().items():
            print(f"    {cat:30s}: {cnt:,}")
        print(f"\n  By Source (top 10):")
        for src, cnt in df["Source"].value_counts().head(10).items():
            print(f"    {src:30s}: {cnt:,}")
    print(f"{'='*65}\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  STEP 3 (ENHANCED v2): Gold-Impact Financial News Collection")
    print(f"  Run date : {END_DATE}")
    print(f"  Output   : {OUTPUT_FILE}")
    print("=" * 65)

    # Source A: Google News RSS (18 targeted queries)
    try:
        df_gnews = collect_google_news()
    except Exception as exc:
        print(f"[ERROR] Google News collection failed: {exc}")
        df_gnews = pd.DataFrame()

    # Sources B–L: Direct RSS feeds
    try:
        df_rss = collect_rss_feeds()
    except Exception as exc:
        print(f"[ERROR] RSS collection failed: {exc}")
        df_rss = pd.DataFrame()

    # Merge, deduplicate, sort
    df_final = combine_and_clean(df_gnews, df_rss)

    # ── POST-FILTER: Remove OTHER_FINANCE rows with no direct gold keyword ──────
    # Headlines categorised as OTHER that also lack any gold-impact term add noise
    # to the training set.  Keep them only if they contain at least one gold term.
    _GOLD_DIRECT = re.compile(
        r"gold|xau|bullion|silver|platinum|precious metal|"
        r"federal reserve|fomc|rate hike|rate cut|powell|inflation|cpi|pce|"
        r"war|military|strike|attack|invasion|sanction|escalat|ceasefire|"
        r"safe haven|flight to safety|risk off|crisis|dollar|dxy|treasury yield|"
        r"iran|israel|ukraine|russia|china tension|north korea|middle east|"
        r"crude oil|opec|energy crisis|tariff|trade war|stagflation|recession",
        re.IGNORECASE,
    )
    pre_filter = len(df_final)
    mask_other = df_final["Gold_Category"] == "OTHER_FINANCE"
    mask_no_gold = ~df_final["Headline"].str.contains(_GOLD_DIRECT)
    df_final = df_final[~(mask_other & mask_no_gold)].reset_index(drop=True)
    dropped = pre_filter - len(df_final)
    if dropped:
        print(f"[POST-FILTER] Dropped {dropped:,} OTHER_FINANCE rows with no gold keyword.")

    # Save
    df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"[SAVED] {OUTPUT_FILE}")
    print(f"[TOTAL] {len(df_final):,} headlines written to CSV\n")


    if len(df_final) > 0:
        # Show war/geopolitical headlines (most impactful)
        war_news = df_final[df_final["Gold_Category"] == "WAR_GEOPOLITICAL"]
        print(f"War/Geopolitical headlines (gold impact): {len(war_news)}")
        if len(war_news) > 0:
            print(war_news[["Date", "Source", "Headline"]].head(10).to_string(index=False))
        print(f"\nSample recent headlines:")
        print(df_final[["Date", "Gold_Category", "Source", "Headline"]].head(15).to_string(index=False))
