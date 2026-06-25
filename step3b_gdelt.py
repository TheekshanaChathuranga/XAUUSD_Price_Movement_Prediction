"""
GDELT Project — Financial Headlines Collector
===============================================
Collects macro/financial news headlines from GDELT 2.0 DOC API.

Strategy:
  • Divides 2015-02-19 → today into 3-month (91-day) chunks
  • Starts from most recent chunk going BACKWARDS — so if interrupted,
    you already have the most valuable recent data
  • Saves progress after every chunk (incremental CSV append)
  • Skips chunks before GDELT 2.0 launch date (2015-02-19)
  • Handles 429 / plain-text rate-limit responses with back-off

GDELT API Rules:
  • Max 250 articles per call
  • 1 request per 5 seconds (we use 6s)
  • OR queries MUST be wrapped in parentheses
  • API only covers 2015-02-19 onward
  • Returns HTTP 200 with plain text on syntax errors (not JSON)

Output: gdelt_news_raw.csv  (Date, Datetime, Headline, Source, URL)
"""

import os, sys, time, re, requests, pandas as pd
from datetime import datetime, timedelta, date
from urllib.parse import urlencode

# ── Windows UTF-8 fix ──────────────────────────────────────────────────────────
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE  = os.path.join(OUTPUT_DIR, "gdelt_news_raw.csv")

# GDELT 2.0 launched 2015-02-19; requests before this return "Invalid query start date"
GDELT_START = datetime(2015, 2, 19)
GDELT_END   = datetime.now()

GDELT_ENDPOINT  = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_DELAY_SEC = 6.0    # API hard limit is 1 req / 5s — use 6s for safety
GDELT_MAX_RETRY = 5      # attempts per chunk before skipping
GDELT_CHUNK_DAYS = 91    # ~3 months per chunk
GDELT_MAX_RECORDS = 250  # articles per call (API maximum)

# ── FINANCE KEYWORD FILTER ────────────────────────────────────────────────────
FINANCE_KEYWORDS = [
    "gold", "xauusd", "bullion", "precious metal", "spot gold", "gold futures"
]
FINANCE_PATTERN = re.compile("|".join(FINANCE_KEYWORDS), re.IGNORECASE)

# ── GDELT QUERY — finance macro (OR terms in parentheses, required by API) ────
GDELT_QUERY = (
    "(gold OR XAUUSD) "
    "(domain:kitco.com OR domain:fxstreet.com OR domain:reuters.com "
    "OR domain:bloomberg.com OR domain:cnbc.com)"
)


def build_chunks(start: datetime, end: datetime, chunk_days: int = GDELT_CHUNK_DAYS):
    """Build date-range tuples newest→oldest (reversed for progressive collection)."""
    chunks = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        chunks.append((cur, nxt))
        cur = nxt
    chunks.reverse()   # most recent first
    return chunks


def fetch_chunk(start_dt: datetime, end_dt: datetime) -> list:
    """
    Query GDELT DOC 2.0 API for one 3-month window.
    Returns list of record dicts, or [] on failure.
    """
    params = {
        "query":         GDELT_QUERY,
        "mode":          "ArtList",
        "maxrecords":    GDELT_MAX_RECORDS,
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime":   end_dt.strftime("%Y%m%d%H%M%S"),
        "sort":          "DateDesc",
        "format":        "json",
        "sourcelang":    "English",
    }
    url = f"{GDELT_ENDPOINT}?{urlencode(params)}"

    data = None
    for attempt in range(1, GDELT_MAX_RETRY + 1):
        try:
            resp = requests.get(url, timeout=30)
        except requests.exceptions.RequestException as exc:
            wait = min(2 ** attempt, 60)
            print(f"      [NET ERR attempt {attempt}] {exc} - wait {wait}s")
            time.sleep(wait)
            continue

        body = resp.text.strip()

        # Detect 429 — GDELT sometimes returns it as HTTP 200 with plain text
        if resp.status_code == 429 or body.lower().startswith("please limit"):
            wait = 10 * attempt
            print(f"      [429 attempt {attempt}] rate-limited - wait {wait}s ...", flush=True)
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            print(f"      [HTTP {resp.status_code}] skipping chunk")
            return []

        # Detect query syntax errors (HTTP 200 but non-JSON body)
        if not body.startswith("{"):
            print(f"      [NON-JSON] {body[:80]}")
            return []    # No retry — it's a permanent API error for this chunk

        try:
            data = resp.json()
            break
        except Exception as exc:
            print(f"      [JSON ERR attempt {attempt}] {exc}")
            if attempt < GDELT_MAX_RETRY:
                time.sleep(GDELT_DELAY_SEC)
            continue
    else:
        print(f"      [SKIP] all {GDELT_MAX_RETRY} retries exhausted")
        return []

    if not data:
        return []

    articles = data.get("articles") or []
    results = []
    for art in articles:
        headline = (art.get("title") or "").strip()
        if not headline or not FINANCE_PATTERN.search(headline):
            continue

        dt_raw = art.get("seendate", "")
        dt = None
        try:
            dt = datetime.strptime(dt_raw, "%Y%m%dT%H%M%SZ")
        except Exception:
            if dt_raw and len(dt_raw) >= 8:
                try:
                    dt = datetime.strptime(dt_raw[:8], "%Y%m%d")
                except Exception:
                    pass

        results.append({
            "Datetime": dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None,
            "Date":     dt.strftime("%Y-%m-%d")           if dt else None,
            "Headline": headline,
            "Source":   art.get("domain") or "GDELT",
            "URL":      art.get("url") or "",
        })
    return results


def load_existing() -> pd.DataFrame:
    """Load previously saved progress, if any."""
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        print(f"[RESUME] Found existing file with {len(df):,} rows")
        return df
    return pd.DataFrame(columns=["Date", "Datetime", "Headline", "Source", "URL"])


def save_progress(df: pd.DataFrame):
    """Save current state to CSV (overwrites)."""
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")


def collect_gdelt():
    print("=" * 65)
    print("  GDELT 2.0 Financial Headlines Collector")
    print(f"  Period  : {GDELT_START.date()} -> {GDELT_END.date()}")
    print(f"  Chunks  : 3-month windows, newest first")
    print(f"  Output  : {OUTPUT_FILE}")
    print("=" * 65)

    chunks = build_chunks(GDELT_START, GDELT_END, GDELT_CHUNK_DAYS)
    total  = len(chunks)
    print(f"\n  Total chunks : {total}")
    print(f"  Est. time    : ~{total * GDELT_DELAY_SEC / 60:.0f} min  (ignoring retries)\n")

    # Load any previous progress
    df_existing = load_existing()
    all_records = df_existing.to_dict("records") if not df_existing.empty else []

    if not df_existing.empty:
        print(f"  [INFO] Existing data found ({len(df_existing)} rows). Only refreshing the most recent 2 chunks.")
        chunks_to_fetch = chunks[:2]
    else:
        chunks_to_fetch = chunks

    stats = {"ok": 0, "empty": 0, "error": 0}
    total_fetch = len(chunks_to_fetch)

    for i, (chunk_start, chunk_end) in enumerate(chunks_to_fetch, 1):
        label = f"{chunk_start.strftime('%Y-%m-%d')} -> {chunk_end.strftime('%Y-%m-%d')}"
        print(f"  [{i:3d}/{total_fetch}] {label}", end="  ", flush=True)

        records = fetch_chunk(chunk_start, chunk_end)

        if records:
            all_records.extend(records)
            stats["ok"] += 1
            print(f"-> {len(records):3d} articles  (total: {len(all_records):,})", flush=True)

            # Save progress after every successful chunk
            df_progress = pd.DataFrame(all_records)
            df_progress["_key"] = df_progress["Headline"].str.lower().str.strip()
            df_progress.drop_duplicates(subset=["_key"], inplace=True)
            df_progress.drop(columns=["_key"], inplace=True)
            save_progress(df_progress)
        else:
            stats["empty"] += 1
            print(f"-> 0 articles", flush=True)

        time.sleep(GDELT_DELAY_SEC)   # mandatory rate-limit delay

    # ── Final clean-up & save ─────────────────────────────────────────────────
    if all_records:
        df_final = pd.DataFrame(all_records)
        # Deduplicate
        df_final["_key"] = df_final["Headline"].str.lower().str.strip()
        df_final.drop_duplicates(subset=["_key"], inplace=True)
        df_final.drop(columns=["_key"], inplace=True)
        # Drop nulls
        df_final.dropna(subset=["Date"], inplace=True)
        df_final["Date"] = pd.to_datetime(df_final["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df_final.dropna(subset=["Date"], inplace=True)
        # Sort
        df_final.sort_values("Date", ascending=False, inplace=True)
        df_final.reset_index(drop=True, inplace=True)
        save_progress(df_final)

        print(f"\n{'=' * 65}")
        print(f"  GDELT COLLECTION COMPLETE")
        print(f"{'=' * 65}")
        print(f"  Chunks successful : {stats['ok']}")
        print(f"  Chunks empty      : {stats['empty']}")
        print(f"  Total unique rows : {len(df_final):,}")
        if len(df_final) > 0:
            print(f"  Date range        : {df_final['Date'].min()} -> {df_final['Date'].max()}")
        print(f"  Saved to          : {OUTPUT_FILE}")
        print(f"{'=' * 65}\n")

        print("  Sample (most recent 10):")
        print(df_final[["Date", "Source", "Headline"]].head(10).to_string(index=False))
    else:
        print("\n[WARN] No articles collected from any chunk.")

    return all_records


if __name__ == "__main__":
    collect_gdelt()
