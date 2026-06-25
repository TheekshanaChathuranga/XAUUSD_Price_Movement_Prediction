"""
STEP 3: Financial News Collection — Gold Price & Macro Sentiment
================================================================
Collects timestamped financial headlines for FinBERT sentiment scoring.

Data Sources (all free, no API key, confirmed working 2026-06-23):
─────────────────────────────────────────────────────────────────
  Source A │ Google News RSS     │ 11 targeted macro/gold queries
           │                     │ Aggregates Reuters, AP, Bloomberg, FT,
           │                     │ WSJ, CNBC, MarketWatch etc. — same corpus
           │                     │ as GDELT monitors, without rate limits.
           │                     │ ~700–1,100 articles per run.
  ─────────┼─────────────────────┼──────────────────────────────────────────
  Source B │ Yahoo Finance RSS   │ Ticker-targeted: Gold (GC=F), Silver
           │                     │ (SI=F), Crude Oil (CL=F), USD Index
  ─────────┼─────────────────────┼──────────────────────────────────────────
  Source C │ ForexLive RSS       │ Live FX / central bank / macro news
  ─────────┼─────────────────────┼──────────────────────────────────────────
  Source D │ FXStreet RSS        │ Gold, forex, ECB, Fed, inflation
  ─────────┼─────────────────────┼──────────────────────────────────────────
  Source E │ CNBC Economy RSS    │ Inflation, Fed policy, US economy
  ─────────┼─────────────────────┼──────────────────────────────────────────
  Source F │ Seeking Alpha RSS   │ Gold (GLD), silver, precious metals
  ─────────┼─────────────────────┼──────────────────────────────────────────
  Source G │ MarketWatch RSS     │ Real-time financial market headlines
  ─────────┼─────────────────────┼──────────────────────────────────────────
  Source H │ Financial Times RSS │ Premium macro / global finance headlines

WHY NOT GDELT DOC API:
  The GDELT 2.0 DOC search API has a minimum index date of ~2017 and enforces
  a hard rate limit (1 req/5 s via 429, even with back-off waits). It is not
  suitable as a reliable automated data source. Google News RSS provides the
  same underlying article corpus (GDELT itself indexes Google News) with no
  rate limits and no minimum date restriction for recent articles.

Output: financial_news_raw.csv
Columns: Date, Datetime, Headline, Source, URL
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

# ── FINANCE KEYWORD FILTER ────────────────────────────────────────────────────
# Applied to feeds that carry mixed content (CNBC, MarketWatch, FT, Google News)
FINANCE_KEYWORDS = [
    "gold", "xau", "federal reserve", "fed rate", "interest rate",
    "inflation", "cpi", "pce", "nonfarm", "payroll", "dollar", "dxy",
    "crude oil", "wti", "brent", "opec", "gdp", "recession", "treasury",
    "yield", "quantitative easing", "taper", "monetary policy", "central bank",
    "geopolit", "ukraine", "war", "sanction", "safe haven", "commodity",
    "powell", "yellen", "lagarde", "ecb", "fomc", "boe", "bank of england",
    "rate hike", "rate cut", "fed funds", "silver", "platinum", "bullion",
    "precious metal", "forex", "fx", "currency", "oil price", "energy",
    "stagflation", "disinflation", "tariff", "trade war",
]
FINANCE_PATTERN = re.compile("|".join(FINANCE_KEYWORDS), re.IGNORECASE)

# ── HTTP HEADERS ──────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: Date parser for RSS pub dates
# ─────────────────────────────────────────────────────────────────────────────
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
    # Last-resort: strip trailing tz abbreviation
    clean = re.sub(r"\s+[A-Z]{2,5}$", "", pub_raw.strip())
    try:
        return datetime.strptime(clean[:25], "%a, %d %b %Y %H:%M:%S")
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE A: GOOGLE NEWS RSS  (replaces GDELT DOC API)
# ─────────────────────────────────────────────────────────────────────────────
# 11 targeted queries covering gold, macro policy, FX, energy, geopolitics.
# Each query returns up to 100 articles (Google News RSS cap).
# No rate limit, no authentication, same underlying article corpus as GDELT.
#
# Query design rules:
#   - Use plain words (no quotes in URL — Google handles NLP)
#   - Avoid single-word queries (too broad, 0 results observed)
#   - Aim for 2-4 topic words per query
#
GOOGLE_NEWS_QUERIES = [
    # Topic                       Query string (URL-encoded via requests)
    ("Gold market",              "gold bullion commodity market"),
    ("Gold silver precious",     "gold silver precious metal"),
    ("Gold inflation dollar",    "gold inflation hedge dollar"),
    ("Federal Reserve policy",   "federal reserve interest rate"),
    ("FOMC Powell decision",     "FOMC rate decision Powell"),
    ("Inflation CPI economy",    "inflation CPI economy"),
    ("Treasury yield recession", "treasury yield bond recession"),
    ("Crude oil OPEC energy",    "crude oil OPEC energy market"),
    ("Dollar DXY forex",         "dollar DXY forex currency"),
    ("Geopolitical safe haven",  "geopolitical risk safe haven"),
    ("ECB Bank of England",      "ECB Bank of England central bank rate"),
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
        # Google News titles contain " - Publisher" suffix — strip it for clean text
        # but keep the publisher for attribution
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

        # Source name: use publisher extracted from title, else "Google News"
        source = publisher if publisher else "Google News"

        records.append({
            "Datetime": dt_str,
            "Date":     date_str,
            "Headline": title,
            "Source":   source,
            "URL":      link.strip(),
        })
    return records


def collect_google_news() -> pd.DataFrame:
    """
    Collect financial headlines from Google News RSS across all defined queries.
    Google News aggregates Reuters, AP, Bloomberg, FT, WSJ, CNBC, MarketWatch
    and hundreds of other outlets — effectively the same corpus GDELT monitors.
    """
    print("\n[Google News] Fetching headlines across targeted financial queries ...")
    all_records = []
    for label, query in GOOGLE_NEWS_QUERIES:
        records = _fetch_gnews(query, label)
        print(f"  {label:35s}: {len(records):4d} headlines")
        all_records.extend(records)
        time.sleep(1.0)   # polite delay

    df = pd.DataFrame(all_records) if all_records else pd.DataFrame(
        columns=["Date", "Datetime", "Headline", "Source", "URL"])
    print(f"[Google News] Subtotal (pre-dedup): {len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE B–H: DIRECT RSS FEEDS
# ─────────────────────────────────────────────────────────────────────────────
# Tested 2026-06-23. All return HTTP 200 with real content.
# apply_filter=False for ticker-targeted feeds (already on-topic).
# apply_filter=True  for mixed-topic feeds (CNBC, MarketWatch, FT).

RSS_SOURCES = [
    # ── Yahoo Finance ticker feeds ──────────────────────────────────────────
    # Gold futures (GC=F), Silver (SI=F), Crude Oil (CL=F), USD Index
    ("https://finance.yahoo.com/rss/headline?s=GC=F",    "Yahoo Finance", False),
    ("https://finance.yahoo.com/rss/headline?s=SI=F",    "Yahoo Finance", False),
    ("https://finance.yahoo.com/rss/headline?s=CL=F",    "Yahoo Finance", False),
    ("https://finance.yahoo.com/rss/headline?s=DX-Y.NYB","Yahoo Finance", False),
    # ── ForexLive ───────────────────────────────────────────────────────────
    ("https://www.forexlive.com/feed/news",              "ForexLive",     True),
    # ── FXStreet ────────────────────────────────────────────────────────────
    ("https://www.fxstreet.com/rss/news",                "FXStreet",      True),
    # ── CNBC Economy ────────────────────────────────────────────────────────
    ("https://www.cnbc.com/id/20910258/device/rss/rss.html", "CNBC",      True),
    # ── Seeking Alpha (GLD ETF — gold commentary) ───────────────────────────
    ("https://seekingalpha.com/api/sa/combined/GLD.xml", "Seeking Alpha", True),
    # ── MarketWatch ─────────────────────────────────────────────────────────
    ("https://feeds.marketwatch.com/marketwatch/marketpulse/", "MarketWatch", True),
    ("https://feeds.marketwatch.com/marketwatch/realtimeheadlines/", "MarketWatch", True),
    # ── Financial Times ─────────────────────────────────────────────────────
    ("https://www.ft.com/rss/home",                      "Financial Times", True),
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
        # Headline
        title_tag = item.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""
        title = re.sub(r"<[^>]+>", "", title).strip()
        if not title:
            continue
        if apply_filter and not FINANCE_PATTERN.search(title):
            continue

        # Date
        pub_tag = (item.find("pubDate") or item.find("published")
                   or item.find("updated") or item.find("dc:date"))
        pub_raw  = pub_tag.get_text(strip=True) if pub_tag else ""
        dt       = _parse_dt(pub_raw)
        dt_str   = dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None
        date_str = dt.strftime("%Y-%m-%d")           if dt else None

        # Link
        link_tag = item.find("link")
        if link_tag:
            link = link_tag.get("href") or link_tag.get_text(strip=True) or ""
        else:
            link = ""

        records.append({
            "Datetime": dt_str,
            "Date":     date_str,
            "Headline": title,
            "Source":   source_name,
            "URL":      link.strip(),
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
        columns=["Date", "Datetime", "Headline", "Source", "URL"])
    print(f"[RSS Feeds] Subtotal (pre-dedup): {len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# COMBINE & CLEAN
# ─────────────────────────────────────────────────────────────────────────────
def combine_and_clean(*dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all source DataFrames, deduplicate on headline text (case-insensitive),
    sort descending by date, and standardise column types.
    """
    valid = [d for d in dfs if d is not None and not d.empty]
    if not valid:
        print("[WARN] No headlines collected from any source.")
        return pd.DataFrame(columns=["Date", "Datetime", "Headline", "Source", "URL"])

    df = pd.concat(valid, ignore_index=True)

    # Ensure all required columns present
    for col in ["Date", "Datetime", "Headline", "Source", "URL"]:
        if col not in df.columns:
            df[col] = None

    df = df[["Date", "Datetime", "Headline", "Source", "URL"]].copy()

    # Drop empty headlines
    df = df[df["Headline"].notna() & (df["Headline"].str.strip() != "")]

    # Deduplicate on normalised headline text
    df["_key"] = df["Headline"].str.lower().str.strip()
    df.drop_duplicates(subset=["_key"], inplace=True)
    df.drop(columns=["_key"], inplace=True)

    # Drop rows with no valid date
    df.dropna(subset=["Date"], inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df.dropna(subset=["Date"], inplace=True)

    # Sort newest → oldest
    df.sort_values("Date", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"\n{'='*60}")
    print(f"  COMBINED RESULTS")
    print(f"{'='*60}")
    print(f"  Total unique headlines : {len(df):,}")
    if len(df) > 0:
        print(f"  Date range            : {df['Date'].min()} -> {df['Date'].max()}")
        print(f"\n  By source:")
        for src, cnt in df["Source"].value_counts().items():
            print(f"    {src:30s}: {cnt:,}")
    print(f"{'='*60}\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 3: Financial News Collection")
    print(f"  Run date : {END_DATE}")
    print(f"  Output   : {OUTPUT_FILE}")
    print("=" * 60)

    # ── Source A: Google News RSS (11 targeted financial queries) ──
    try:
        df_gnews = collect_google_news()
    except Exception as exc:
        print(f"[ERROR] Google News collection failed: {exc}")
        df_gnews = pd.DataFrame()

    # ── Source B–H: Direct RSS feeds ──
    try:
        df_rss = collect_rss_feeds()
    except Exception as exc:
        print(f"[ERROR] RSS collection failed: {exc}")
        df_rss = pd.DataFrame()

    # ── Merge, deduplicate, sort ──
    df_final = combine_and_clean(df_gnews, df_rss)

    # ── Save ──
    df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"[SAVED] {OUTPUT_FILE}")
    print(f"[TOTAL] {len(df_final):,} headlines written to CSV\n")

    if len(df_final) > 0:
        print("Sample headlines:")
        print(df_final[["Date", "Source", "Headline"]].head(15).to_string(index=False))
