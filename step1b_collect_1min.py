"""
STEP 1B: XAU/USD 1-Minute OHLCV Data Collection
=================================================
Collects 1-minute bars for XAU/USD via MetaTrader5 (primary, production)
with yfinance as a development stand-in (clearly flagged -- NOT for live trading).

Storage Strategy:
  In-memory rolling deque (maxlen=BUFFER_SIZE) + periodic SQLite flush.
  Rationale: 1-min data grows at ~390 bars/session; a full in-memory history
  would exhaust RAM within hours. The deque caps RAM usage while SQLite
  provides persistent history for back-testing / filter calibration.

Key Features:
  - MT5 Python API integration (production)
  - yfinance fallback for pipeline development (NON-PRODUCTION -- clearly flagged)
  - Rolling collections.deque (default 5,000 bars ~83 trading hours)
  - SQLite flush every FLUSH_INTERVAL seconds (default 300 s / 5 min)
  - Keeps last SQLITE_DAYS days in DB (default 30)
  - Gap/missing-bar detection + forward-fill (max 3 bars)
  - Data quality flag per bar: 0=OK, 1=FILLED, 2=SUSPECT (gap >3)
  - Native 1-min indicators: RSI-14, MACD 12/26/9, BB 20/2, ATR-14, EMA-9/21
    (NOT downsampled from daily -- that misrepresents intraday volatility)

Public API:
  get_buffer()                  -> pd.DataFrame
  get_5min_resample()           -> pd.DataFrame
  get_15min_resample()          -> pd.DataFrame
  get_daily_atr()               -> float
  start_collection_loop(60)     -> threading.Thread
  backfill_from_mt5(days=7)     -> pd.DataFrame
  run_test()                    via --test CLI flag
"""

import os, sys, time, math, sqlite3, logging, argparse, threading, collections
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import pandas_ta as ta

# ---- LOGGING ----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [1MIN] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("step1b")

# ---- CONFIG -----------------------------------------------------------------
OUTPUT_DIR     = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB      = os.path.join(OUTPUT_DIR, "xauusd_1min.db")
DAILY_PRICES   = os.path.join(OUTPUT_DIR, "xauusd_raw_prices.csv")

BUFFER_SIZE    = 5000   # max 1-min bars in RAM (~83 trading hours)
FLUSH_INTERVAL = 300    # seconds between SQLite flushes
SQLITE_DAYS    = 30     # days of 1-min history to keep in DB

TICKER_YF      = "GC=F"    # yfinance ticker (DEV FALLBACK)
TICKER_MT5     = "XAUUSD"  # MT5 symbol -- adjust to your broker

ATR_PERIOD     = 14
RSI_PERIOD     = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_PERIOD, BB_STD = 20, 2.0
EMA_FAST, EMA_SLOW = 9, 21

DQ_OK      = 0   # real bar
DQ_FILLED  = 1   # forward-filled gap (<= 3 bars)
DQ_SUSPECT = 2   # gap > 3 bars; synthetic bar

# ---- STATE ------------------------------------------------------------------
_buffer_lock        = threading.Lock()
_buffer             = collections.deque(maxlen=BUFFER_SIZE)
_last_flush_ts      = 0.0
_collection_running = False

# ---- MT5 IMPORT (optional) --------------------------------------------------
try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
    MT5_TIMEFRAME  = mt5.TIMEFRAME_M1
except ImportError:
    _MT5_AVAILABLE = False
    log.warning(
        "MetaTrader5 not installed / not available on this OS. "
        "Falling back to yfinance (NON-PRODUCTION)."
    )

# ---- INDICATORS -------------------------------------------------------------

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute native 1-min technical indicators. Never downsamples daily values."""
    df = df.copy()
    df["RSI_14"] = ta.rsi(df["Close"], length=RSI_PERIOD)

    macd = ta.macd(df["Close"], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    if macd is not None and not macd.empty:
        df = pd.concat([df, macd], axis=1)

    bb = ta.bbands(df["Close"], length=BB_PERIOD, std=BB_STD)
    if bb is not None and not bb.empty:
        df = pd.concat([df, bb], axis=1)

    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    df["ATR_14"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(ATR_PERIOD).mean()

    df["EMA_9"]  = ta.ema(df["Close"], length=EMA_FAST)
    df["EMA_21"] = ta.ema(df["Close"], length=EMA_SLOW)
    return df


def _resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample 1-min DataFrame to higher timeframe and recompute indicators."""
    if df.empty:
        return df
    if "Datetime" in df.columns:
        df = df.set_index("Datetime")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    agg = {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    r = df[["Open","High","Low","Close","Volume"]].resample(freq).agg(agg).dropna()
    return _compute_indicators(r).reset_index()

# ---- GAP DETECTION & FILL ---------------------------------------------------

def _fill_internal_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and forward-fill internal gaps within a batch; flag DQ."""
    if df.empty or len(df) < 2:
        if not df.empty and "data_quality" not in df.columns:
            df = df.copy()
            df["data_quality"] = DQ_OK
        return df

    df = df.copy()
    if "data_quality" not in df.columns:
        df["data_quality"] = DQ_OK

    dt_col = "Datetime" if "Datetime" in df.columns else None
    if dt_col:
        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.set_index(dt_col)
    df.index = pd.to_datetime(df.index)

    full_range   = pd.date_range(start=df.index.min(), end=df.index.max(), freq="1min")
    df_full      = df.reindex(full_range)
    newly_filled = df_full["Close"].isna()

    ohlcv_cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df_full.columns]
    df_full[ohlcv_cols] = df_full[ohlcv_cols].ffill(limit=3)

    if "data_quality" not in df_full.columns:
        df_full["data_quality"] = DQ_OK
    df_full.loc[newly_filled & df_full["Close"].notna(), "data_quality"] = DQ_FILLED

    still_empty = df_full["Close"].isna()
    if still_empty.any():
        n = still_empty.sum()
        log.warning(f"[GAP] Dropped {n} bars with gap >3 (unfillable).")
    df_full = df_full.dropna(subset=["Close"])
    df_full = df_full.reset_index().rename(columns={"index":"Datetime"})
    return df_full

# ---- MT5 FUNCTIONS ----------------------------------------------------------

def _init_mt5() -> bool:
    if not _MT5_AVAILABLE:
        return False
    if not mt5.initialize():
        log.error(f"MT5 initialize() failed: {mt5.last_error()}")
        return False
    log.info(f"MT5 connected. Version: {mt5.version()}")
    return True


def _fetch_mt5_bars(n_bars: int = 500) -> pd.DataFrame:
    if not _MT5_AVAILABLE:
        return pd.DataFrame()
    if not mt5.terminal_info():
        if not _init_mt5():
            return pd.DataFrame()
    try:
        rates = mt5.copy_rates_from_pos(TICKER_MT5, MT5_TIMEFRAME, 0, n_bars)
        if rates is None or len(rates) == 0:
            log.warning(f"MT5: no rates for {TICKER_MT5} ({mt5.last_error()})")
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df["Datetime"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"open":"Open","high":"High","low":"Low",
                                 "close":"Close","tick_volume":"Volume"})
        return df[["Datetime","Open","High","Low","Close","Volume"]].sort_values("Datetime").reset_index(drop=True)
    except Exception as e:
        log.error(f"MT5 fetch error: {e}")
        return pd.DataFrame()


def backfill_from_mt5(days: int = 7) -> pd.DataFrame:
    """
    Load historical 1-min bars.  MT5 if available, yfinance (DEV) otherwise.
    Returns DataFrame sorted ascending by Datetime.
    """
    n_bars = days * 390 + 200
    log.info(f"Backfilling {days} days (~{n_bars} bars)...")

    df = _fetch_mt5_bars(n_bars=n_bars)
    if not df.empty:
        log.info(f"[MT5] {len(df)} bars: {df['Datetime'].min()} -> {df['Datetime'].max()}")
        return _fill_internal_gaps(df)

    # ---- yfinance DEV fallback ----------------------------------------------
    log.warning(
        "\n"
        "  *** DEVELOPMENT FALLBACK: yfinance 1-min data ***\n"
        "  *** Coverage: last 7 days only.               ***\n"
        "  *** NOT suitable for live trading.            ***\n"
        "  *** Set up MT5 for production use.            ***"
    )
    import yfinance as yf
    all_dfs = []
    for i in range(min(days, 7)):
        end_dt   = datetime.now() - timedelta(days=i)
        start_dt = end_dt - timedelta(days=1)
        try:
            raw = yf.download(TICKER_YF, start=start_dt.strftime("%Y-%m-%d"),
                              end=end_dt.strftime("%Y-%m-%d"),
                              interval="1m", auto_adjust=True, progress=False)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw[["Open","High","Low","Close","Volume"]].copy()
            raw.index.name = "Datetime"
            raw = raw.reset_index()
            raw["Datetime"] = pd.to_datetime(raw["Datetime"]).dt.tz_localize(None)
            raw["data_quality"] = DQ_OK
            all_dfs.append(raw)
            time.sleep(0.3)
        except Exception as e:
            log.warning(f"yfinance day-{i} error: {e}")

    if not all_dfs:
        log.error("yfinance backfill returned no data.")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
    df = _fill_internal_gaps(df)
    log.info(f"[yfinance DEV] {len(df)} bars loaded.")
    return df

# ---- LIVE FETCH (collection loop) -------------------------------------------

def _fetch_latest_bars(n: int = 5) -> pd.DataFrame:
    if _MT5_AVAILABLE and mt5.terminal_info():
        df = _fetch_mt5_bars(n_bars=n)
        if not df.empty:
            return df
    import yfinance as yf
    try:
        raw = yf.download(TICKER_YF, period="1d", interval="1m",
                          auto_adjust=True, progress=False)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw[["Open","High","Low","Close","Volume"]].copy()
        raw.index.name = "Datetime"
        raw = raw.reset_index()
        raw["Datetime"] = pd.to_datetime(raw["Datetime"]).dt.tz_localize(None)
        raw["data_quality"] = DQ_OK
        return raw.tail(n).reset_index(drop=True)
    except Exception as e:
        log.warning(f"yfinance live fetch error: {e}")
        return pd.DataFrame()

# ---- BUFFER ACCESS ----------------------------------------------------------

def _buffer_to_df() -> pd.DataFrame:
    with _buffer_lock:
        if not _buffer:
            return pd.DataFrame()
        df = pd.DataFrame(list(_buffer))
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.sort_values("Datetime").reset_index(drop=True)
    return df


def get_buffer() -> pd.DataFrame:
    """Return current 1-min buffer as DataFrame with indicators computed."""
    df = _buffer_to_df()
    if df.empty:
        return df
    return _compute_indicators(df)


def get_5min_resample() -> pd.DataFrame:
    """Return buffer resampled to 5-min bars with indicators."""
    df = _buffer_to_df()
    return _resample(df, "5min") if not df.empty else df


def get_15min_resample() -> pd.DataFrame:
    """Return buffer resampled to 15-min bars with indicators."""
    df = _buffer_to_df()
    return _resample(df, "15min") if not df.empty else df


def get_daily_atr() -> float:
    """Median ATR(14) from daily prices file -- used for the 1-min ATR floor."""
    try:
        prices = pd.read_csv(DAILY_PRICES)
        hl  = prices["High"] - prices["Low"]
        hc  = (prices["High"] - prices["Close"].shift()).abs()
        lc  = (prices["Low"]  - prices["Close"].shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(ATR_PERIOD).mean().dropna()
        return float(atr.median())
    except Exception as e:
        log.warning(f"Could not compute daily ATR: {e}")
        return 0.0

# ---- SQLITE PERSISTENCE -----------------------------------------------------

def _init_db():
    con = sqlite3.connect(SQLITE_DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_1min (
            datetime      TEXT PRIMARY KEY,
            open          REAL, high REAL, low REAL, close REAL, volume REAL,
            data_quality  INTEGER DEFAULT 0,
            rsi_14        REAL, macd REAL, macd_signal REAL, macd_hist REAL,
            bb_upper      REAL, bb_mid REAL, bb_lower REAL,
            atr_14        REAL, ema_9 REAL, ema_21 REAL
        )
    """)
    con.commit()
    con.close()


def _safe_val(row, col):
    v = row.get(col)
    return None if (v is None or (isinstance(v, float) and math.isnan(v))) else v


def _flush_buffer_to_db():
    global _last_flush_ts
    df = get_buffer()
    if df.empty:
        return
    cutoff = (datetime.now() - timedelta(days=SQLITE_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    try:
        con = sqlite3.connect(SQLITE_DB)
        rows = []
        for _, row in df.iterrows():
            dt = str(row.get("Datetime", ""))
            if not dt:
                continue
            rows.append((
                dt,
                _safe_val(row,"Open"), _safe_val(row,"High"),
                _safe_val(row,"Low"),  _safe_val(row,"Close"), _safe_val(row,"Volume"),
                int(row.get("data_quality", DQ_OK)),
                _safe_val(row,"RSI_14"),
                _safe_val(row,"MACD_12_26_9"), _safe_val(row,"MACDs_12_26_9"),
                _safe_val(row,"MACDh_12_26_9"),
                _safe_val(row,f"BBU_{BB_PERIOD}_{BB_STD}"),
                _safe_val(row,f"BBM_{BB_PERIOD}_{BB_STD}"),
                _safe_val(row,f"BBL_{BB_PERIOD}_{BB_STD}"),
                _safe_val(row,"ATR_14"),
                _safe_val(row,"EMA_9"), _safe_val(row,"EMA_21"),
            ))
        con.executemany("""
            INSERT OR REPLACE INTO ohlcv_1min
            (datetime,open,high,low,close,volume,data_quality,
             rsi_14,macd,macd_signal,macd_hist,bb_upper,bb_mid,bb_lower,atr_14,ema_9,ema_21)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)
        con.execute("DELETE FROM ohlcv_1min WHERE datetime < ?", (cutoff,))
        pruned = con.execute("SELECT changes()").fetchone()[0]
        con.commit()
        con.close()
        log.info(f"[SQLite] Flushed {len(rows)} bars; pruned {pruned} old rows.")
        _last_flush_ts = time.time()
    except Exception as e:
        log.error(f"[SQLite] Flush error: {e}")

# ---- BUFFER UPDATE ----------------------------------------------------------

def _update_buffer(new_bars: pd.DataFrame):
    global _last_flush_ts
    if new_bars.empty:
        return
    if "Datetime" not in new_bars.columns and isinstance(new_bars.index, pd.DatetimeIndex):
        new_bars = new_bars.reset_index().rename(columns={"index":"Datetime"})
    new_bars["Datetime"] = pd.to_datetime(new_bars["Datetime"])
    new_bars = new_bars.sort_values("Datetime").reset_index(drop=True)

    with _buffer_lock:
        existing_ts = {str(r["Datetime"]) for r in _buffer}
        for _, row in new_bars.iterrows():
            ts = str(row["Datetime"])
            if ts not in existing_ts:
                _buffer.append(row.to_dict())
                existing_ts.add(ts)

    if time.time() - _last_flush_ts >= FLUSH_INTERVAL:
        _flush_buffer_to_db()

# ---- COLLECTION LOOP --------------------------------------------------------

def start_collection_loop(interval_sec: int = 60) -> threading.Thread:
    """
    Start background collection thread.
    - Initialises MT5 if available
    - Backfills last 3 days on startup
    - Fetches latest bars every interval_sec seconds
    - daemon=True so it exits when main process exits
    """
    global _collection_running
    if _collection_running:
        log.warning("Collection loop already running.")
        return None

    def _loop():
        global _collection_running
        _collection_running = True
        log.info("1-min collection loop started.")
        mt5_ok = _init_mt5() if _MT5_AVAILABLE else False
        if not mt5_ok and _MT5_AVAILABLE:
            log.warning("MT5 init failed -- using yfinance DEV fallback. "
                        "Ensure MT5 terminal is open and algo-trading enabled.")

        hist = backfill_from_mt5(days=3)
        if not hist.empty:
            _update_buffer(hist)
            log.info(f"Startup backfill: {len(hist)} bars loaded.")

        while _collection_running:
            try:
                bars = _fetch_latest_bars(n=5)
                if not bars.empty:
                    _update_buffer(bars)
                    latest = bars.iloc[-1]
                    log.info(
                        f"[TICK] {latest.get('Datetime','')}  "
                        f"C={latest.get('Close',0):.2f}  "
                        f"DQ={latest.get('data_quality',DQ_OK)}"
                    )
                else:
                    log.warning("No bars returned this cycle.")
            except Exception as e:
                log.error(f"Collection loop error: {e}")
            time.sleep(interval_sec)

        log.info("Collection loop stopped.")
        if mt5_ok:
            try: mt5.shutdown()
            except: pass

    t = threading.Thread(target=_loop, daemon=True, name="1min-collector")
    t.start()
    return t


def stop_collection_loop():
    global _collection_running
    _collection_running = False

# ---- SELF-TEST --------------------------------------------------------------

def run_test():
    print("=" * 60)
    print("  STEP 1B SELF-TEST")
    print("=" * 60)
    _init_db()
    print(f"[1] SQLite DB: {SQLITE_DB}")

    print("\n[2] Backfilling 2 days...")
    df = backfill_from_mt5(days=2)
    if df.empty:
        print("  ERROR: No data returned."); return
    print(f"  Bars: {len(df)}")
    print(f"  Range: {df['Datetime'].min()} -> {df['Datetime'].max()}")
    if "data_quality" in df.columns:
        vc = df["data_quality"].value_counts().to_dict()
        print(f"  Quality: OK={vc.get(0,0)} FILLED={vc.get(1,0)} SUSPECT={vc.get(2,0)}")

    print("\n[3] Loading buffer & computing indicators...")
    _update_buffer(df)
    buf = get_buffer()
    print(f"  Buffer rows: {len(buf)}")
    print(f"  RSI_14 (last 5):  {buf['RSI_14'].tail(5).round(2).tolist()}")
    print(f"  ATR_14 (last 5):  {buf['ATR_14'].tail(5).round(4).tolist()}")
    print(f"  EMA_9  (last 5):  {buf['EMA_9'].tail(5).round(2).tolist()}")

    print("\n[4] 5-min resample (last 3 bars):")
    r5 = get_5min_resample()
    if not r5.empty:
        cols = [c for c in ["Datetime","Open","High","Low","Close","ATR_14"] if c in r5.columns]
        print(r5.tail(3)[cols].to_string(index=False))

    print("\n[5] Daily ATR reference:")
    d_atr = get_daily_atr()
    floor = d_atr / math.sqrt(390) * 0.5
    print(f"  Median daily ATR(14): {d_atr:.2f}")
    print(f"  1-min ATR floor (x0.5): {floor:.4f}")

    print("\n[6] Flushing to SQLite...")
    _flush_buffer_to_db()
    con = sqlite3.connect(SQLITE_DB)
    cnt = con.execute("SELECT COUNT(*) FROM ohlcv_1min").fetchone()[0]
    con.close()
    print(f"  Rows in ohlcv_1min: {cnt}")

    print("\n[PASS] Self-test complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAU/USD 1-Min Data Collection")
    parser.add_argument("--test",     action="store_true", help="Run self-test")
    parser.add_argument("--run",      action="store_true", help="Start live collection loop")
    parser.add_argument("--backfill", type=int, default=7, metavar="DAYS",
                        help="Backfill N days and exit (default: 7)")
    args = parser.parse_args()
    _init_db()
    if args.test:
        run_test()
    elif args.run:
        log.info("Starting live collection. Press Ctrl+C to stop.")
        start_collection_loop(interval_sec=60)
        try:
            while True: time.sleep(5)
        except KeyboardInterrupt:
            stop_collection_loop()
            _flush_buffer_to_db()
            log.info("Stopped.")
    else:
        df = backfill_from_mt5(days=args.backfill)
        if not df.empty:
            _update_buffer(df)
            _flush_buffer_to_db()
            print(f"Backfilled {len(df)} bars saved to SQLite.")
