"""
STEP 2B: 1-Minute Signal Accuracy Enhancement
==============================================
Five sequential gates for filtering 1-min signals to fewer, higher-conviction trades.
All rejections are logged explicitly -- never silent drops.

Gates (applied in order):
  1. Confirmation-bar requirement   -- signal must hold for N consecutive bars
  2. ATR floor (minimum volatility) -- reject thin/dead-market hours
  3. Multi-timeframe (MTF) confluence -- 5-min AND 15-min trend must agree
  4. Probability threshold recalibration -- 1-min-specific P60/P40 thresholds
  5. Spread/slippage check -- used by Part 3 pre-trade check

Public API:
  filter_signal(signal, prob_up, buf_df) -> FilterResult (namedtuple)
  simulate_trade_frequency(df_1min)      -> dict
  load_1min_thresholds()                 -> (long_t, short_t)
  calibrate_thresholds(score_series)     -> (long_t, short_t)  [saves JSON]

CLI:
  python step2b_1min_signal_filter.py --simulate  [requires step1b buffer data]
"""

import os, sys, json, logging, math, argparse
from collections import deque
from typing import Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import pandas_ta as ta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FILTER] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("step2b")

# ---- CONFIG -----------------------------------------------------------------
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DAILY_PRICES     = os.path.join(OUTPUT_DIR, "xauusd_raw_prices.csv")
THRESHOLD_1MIN   = os.path.join(OUTPUT_DIR, "model_threshold_1min.json")

# Gate 1: confirmation bars
N_CONFIRM_BARS       = 3     # consecutive bars signal must hold before firing

# Gate 2: ATR floor
ATR_FLOOR_MULTIPLIER = 0.5   # fraction of scaled daily ATR
ATR_PERIOD           = 14
TRADING_MINS_PER_DAY = 390   # ~6.5 h for gold (Sun 22:00 - Fri 21:00 UTC approx)

# Gate 3: MTF confluence
MTF_REQUIRE_5MIN   = True
MTF_REQUIRE_15MIN  = True
EMA_FAST           = 9
EMA_SLOW           = 21

# Gate 4: 1-min probability thresholds
MIN_BARS_FOR_CALIBRATION = 500   # need at least this many scored bars to calibrate
LONG_THRESHOLD_1MIN      = 0.58  # fallback if insufficient data
SHORT_THRESHOLD_1MIN     = 0.42
LONG_PERCENTILE          = 60
SHORT_PERCENTILE         = 40

# Gate 5: spread check
MAX_SPREAD_PCT_OF_RISK   = 0.10  # reject if spread > 10% of $50 risk budget

# ---- RESULT TYPE ------------------------------------------------------------

@dataclass
class FilterResult:
    passed: bool
    signal: str          # "LONG" | "SHORT" | "NEUTRAL"
    reject_gate: Optional[str] = None   # which gate rejected, or None
    reject_reason: str = ""
    atr_1min: float = 0.0
    atr_floor: float = 0.0
    mtf_5min: str = "UNKNOWN"
    mtf_15min: str = "UNKNOWN"
    confirm_count: int = 0
    spread_pct_of_risk: float = 0.0
    estimated_spread_pips: float = 0.5  # default assumption

# ---- HELPERS ----------------------------------------------------------------

def _ema_trend(df: pd.DataFrame) -> str:
    """Return 'LONG' if EMA_9 > EMA_21 on last bar, 'SHORT' if below, else 'NEUTRAL'."""
    if df.empty or len(df) < EMA_SLOW + 5:
        return "NEUTRAL"
    ema9  = ta.ema(df["Close"], length=EMA_FAST)
    ema21 = ta.ema(df["Close"], length=EMA_SLOW)
    if ema9 is None or ema21 is None:
        return "NEUTRAL"
    last9  = ema9.dropna().iloc[-1] if not ema9.dropna().empty else None
    last21 = ema21.dropna().iloc[-1] if not ema21.dropna().empty else None
    if last9 is None or last21 is None:
        return "NEUTRAL"
    if last9 > last21:
        return "LONG"
    if last9 < last21:
        return "SHORT"
    return "NEUTRAL"


def _resample_df(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample 1-min OHLCV buffer to a higher timeframe."""
    if df.empty:
        return df
    df = df.copy()
    if "Datetime" in df.columns:
        df = df.set_index("Datetime")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    agg = {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    cols = [c for c in agg if c in df.columns]
    return df[cols].resample(freq).agg({k:v for k,v in agg.items() if k in cols}).dropna().reset_index()

# ---- GATE 4: THRESHOLDS -----------------------------------------------------

def load_1min_thresholds() -> Tuple[float, float]:
    """Load calibrated 1-min thresholds from JSON, or return defaults."""
    if os.path.exists(THRESHOLD_1MIN):
        try:
            with open(THRESHOLD_1MIN) as f:
                d = json.load(f)
            long_t  = float(d.get("long",  LONG_THRESHOLD_1MIN))
            short_t = float(d.get("short", SHORT_THRESHOLD_1MIN))
            log.info(f"[Gate4] Loaded 1-min thresholds: LONG>{long_t:.3f} SHORT<{short_t:.3f}")
            return long_t, short_t
        except Exception as e:
            log.warning(f"[Gate4] Could not load threshold file: {e}")
    log.info(f"[Gate4] Using default thresholds: LONG>{LONG_THRESHOLD_1MIN} SHORT<{SHORT_THRESHOLD_1MIN}")
    return LONG_THRESHOLD_1MIN, SHORT_THRESHOLD_1MIN


def calibrate_thresholds(score_series: pd.Series) -> Tuple[float, float]:
    """
    Walk-forward percentile calibration of 1-min probability thresholds.
    Uses P60/P40 (wider neutral band) appropriate for noisy 1-min data.
    Saves result to model_threshold_1min.json.
    """
    scores = score_series.dropna()
    if len(scores) < MIN_BARS_FOR_CALIBRATION:
        log.warning(
            f"[Gate4] Only {len(scores)} scored bars (need {MIN_BARS_FOR_CALIBRATION}). "
            f"Using default thresholds."
        )
        return LONG_THRESHOLD_1MIN, SHORT_THRESHOLD_1MIN

    long_t  = float(np.percentile(scores, LONG_PERCENTILE))
    short_t = float(np.percentile(scores, SHORT_PERCENTILE))

    with open(THRESHOLD_1MIN, "w") as f:
        json.dump({"long": long_t, "short": short_t, "n_bars": len(scores),
                   "percentile_long": LONG_PERCENTILE,
                   "percentile_short": SHORT_PERCENTILE}, f, indent=2)
    log.info(
        f"[Gate4] Calibrated thresholds from {len(scores)} bars: "
        f"LONG>{long_t:.4f} SHORT<{short_t:.4f}. Saved to {THRESHOLD_1MIN}"
    )
    return long_t, short_t

# ---- CONFIRM-BAR STATE MACHINE ----------------------------------------------

class ConfirmationTracker:
    """
    Tracks consecutive bars of the same signal direction.
    Call update(signal) each bar; check is_confirmed() before acting.
    """
    def __init__(self, n: int = N_CONFIRM_BARS):
        self.n         = n
        self._signal   = "NEUTRAL"
        self._count    = 0

    def update(self, signal: str):
        if signal == self._signal and signal != "NEUTRAL":
            self._count += 1
        else:
            self._signal = signal
            self._count  = 1 if signal != "NEUTRAL" else 0

    def is_confirmed(self) -> Tuple[bool, int]:
        """Returns (confirmed, count)."""
        return self._count >= self.n, self._count

    def reset(self):
        self._signal = "NEUTRAL"
        self._count  = 0

# ---- DAILY ATR (cached) -----------------------------------------------------
_cached_daily_atr: Optional[float] = None

def _get_daily_atr() -> float:
    global _cached_daily_atr
    if _cached_daily_atr is not None:
        return _cached_daily_atr
    try:
        prices = pd.read_csv(DAILY_PRICES)
        hl  = prices["High"] - prices["Low"]
        hc  = (prices["High"] - prices["Close"].shift()).abs()
        lc  = (prices["Low"]  - prices["Close"].shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().dropna()
        _cached_daily_atr = float(atr.median())
        return _cached_daily_atr
    except Exception:
        return 0.0

# ---- MAIN FILTER FUNCTION ---------------------------------------------------

def filter_signal(
    signal: str,
    prob_up: float,
    buf_df: pd.DataFrame,
    confirm_tracker: Optional[ConfirmationTracker] = None,
    spread_pips: float = 0.5,
    risk_usd: float = 50.0,
    pip_value_per_lot: float = 10.0,
    lot_size: float = 0.05,
) -> FilterResult:
    """
    Apply all five filter gates to a 1-min signal candidate.

    Parameters
    ----------
    signal           : "LONG" | "SHORT" | "NEUTRAL" from the ensemble
    prob_up          : probability of UP move (0-1) from ensemble
    buf_df           : current 1-min buffer DataFrame (with Datetime column)
    confirm_tracker  : optional ConfirmationTracker instance (shared across calls)
    spread_pips      : estimated broker spread in pips
    risk_usd         : target risk in USD per trade
    pip_value_per_lot: USD per pip per lot
    lot_size         : current position lot size (for spread cost calc)

    Returns
    -------
    FilterResult with passed=True only if all gates pass
    """
    result = FilterResult(passed=False, signal=signal)

    if signal == "NEUTRAL":
        result.reject_gate   = "neutral"
        result.reject_reason = "Signal is NEUTRAL -- no directional bias."
        log.debug(result.reject_reason)
        return result

    # ---- Gate 1: Confirmation bars ----------------------------------------
    if confirm_tracker is not None:
        confirm_tracker.update(signal)
        confirmed, count = confirm_tracker.is_confirmed()
        result.confirm_count = count
        if not confirmed:
            result.reject_gate   = "gate1_confirm"
            result.reject_reason = (
                f"[FILTER] Signal rejected: only {count}/{N_CONFIRM_BARS} confirm bars "
                f"(signal={signal}). Need {N_CONFIRM_BARS} consecutive bars."
            )
            log.info(result.reject_reason)
            return result
    else:
        result.confirm_count = N_CONFIRM_BARS  # assume confirmed if no tracker supplied

    # ---- Gate 2: ATR floor (thin market) -----------------------------------
    atr_1min  = 0.0
    atr_floor = 0.0
    if not buf_df.empty and len(buf_df) >= 15:
        close = buf_df["Close"].dropna()
        high  = buf_df["High"].dropna()
        low   = buf_df["Low"].dropna()
        if len(close) > ATR_PERIOD:
            hl = high - low
            hc = (high - close.shift()).abs()
            lc = (low  - close.shift()).abs()
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            atr_vals = tr.rolling(ATR_PERIOD).mean().dropna()
            if not atr_vals.empty:
                atr_1min = float(atr_vals.iloc[-1])

        daily_atr = _get_daily_atr()
        if daily_atr > 0:
            # Scale daily ATR to 1-min: daily_atr / sqrt(trading_minutes_per_day)
            atr_floor = daily_atr / math.sqrt(TRADING_MINS_PER_DAY) * ATR_FLOOR_MULTIPLIER

        result.atr_1min  = atr_1min
        result.atr_floor = atr_floor

        if atr_floor > 0 and atr_1min > 0 and atr_1min < atr_floor:
            result.reject_gate   = "gate2_atr_floor"
            result.reject_reason = (
                f"[FILTER] Signal rejected: ATR_1min={atr_1min:.4f} below floor={atr_floor:.4f} "
                f"-- thin market / low-liquidity hours. Skipping."
            )
            log.info(result.reject_reason)
            return result

    # ---- Gate 3: MTF confluence --------------------------------------------
    mtf_5min  = "NEUTRAL"
    mtf_15min = "NEUTRAL"
    if not buf_df.empty:
        r5  = _resample_df(buf_df, "5min")
        r15 = _resample_df(buf_df, "15min")
        mtf_5min  = _ema_trend(r5)
        mtf_15min = _ema_trend(r15)

    result.mtf_5min  = mtf_5min
    result.mtf_15min = mtf_15min

    # Check confluence: 5-min and 15-min must agree with signal direction
    failures = []
    if MTF_REQUIRE_5MIN and mtf_5min not in (signal, "NEUTRAL"):
        failures.append(f"5min={mtf_5min}")
    if MTF_REQUIRE_15MIN and mtf_15min not in (signal, "NEUTRAL"):
        failures.append(f"15min={mtf_15min}")

    if failures:
        disagreements = ", ".join(failures)
        result.reject_gate   = "gate3_mtf"
        result.reject_reason = (
            f"[MTF] {disagreements} disagree with 1-min signal={signal}. "
            f"Disagreement explicitly reported -- signal suppressed. "
            f"(5min={mtf_5min}, 15min={mtf_15min})"
        )
        log.info(result.reject_reason)
        return result

    # ---- Gate 4: 1-min probability threshold -------------------------------
    long_t, short_t = load_1min_thresholds()
    if signal == "LONG" and prob_up < long_t:
        result.reject_gate   = "gate4_threshold"
        result.reject_reason = (
            f"[FILTER] LONG signal rejected: prob_up={prob_up:.4f} < "
            f"1-min LONG threshold={long_t:.4f}."
        )
        log.info(result.reject_reason)
        return result
    if signal == "SHORT" and prob_up > short_t:
        result.reject_gate   = "gate4_threshold"
        result.reject_reason = (
            f"[FILTER] SHORT signal rejected: prob_up={prob_up:.4f} > "
            f"1-min SHORT threshold={short_t:.4f}."
        )
        log.info(result.reject_reason)
        return result

    # ---- Gate 5: Spread/slippage check ------------------------------------
    if spread_pips > 0 and lot_size > 0:
        spread_cost_usd   = spread_pips * pip_value_per_lot * lot_size
        spread_pct        = spread_cost_usd / risk_usd if risk_usd > 0 else 0.0
        result.spread_pct_of_risk      = spread_pct
        result.estimated_spread_pips   = spread_pips
        if spread_pct > MAX_SPREAD_PCT_OF_RISK:
            result.reject_gate   = "gate5_spread"
            result.reject_reason = (
                f"[FILTER] Trade rejected: spread cost ${spread_cost_usd:.2f} "
                f"({spread_pct*100:.1f}% of ${risk_usd} risk) exceeds "
                f"{MAX_SPREAD_PCT_OF_RISK*100:.0f}% limit."
            )
            log.info(result.reject_reason)
            return result

    # ---- All gates passed --------------------------------------------------
    result.passed = True
    log.info(
        f"[PASS] Signal={signal} prob={prob_up:.4f} | "
        f"Confirm={result.confirm_count}/{N_CONFIRM_BARS} | "
        f"ATR={atr_1min:.4f}/>{atr_floor:.4f} | "
        f"5min={mtf_5min} 15min={mtf_15min} | "
        f"Spread={result.spread_pct_of_risk*100:.1f}%"
    )
    return result

# ---- TRADE FREQUENCY SIMULATION ---------------------------------------------

def simulate_trade_frequency(df_1min: pd.DataFrame) -> dict:
    """
    Replay the full filter pipeline on a historical 1-min DataFrame to estimate
    trade frequency and filter effectiveness at current settings.

    The ensemble probability is approximated from EMA crossover strength
    (in the absence of a real 1-min model) to demonstrate filter mechanics.

    Returns dict:
      trades_per_hour, pct_filtered_confirm, pct_filtered_atr,
      pct_filtered_mtf, pct_filtered_threshold, pct_filtered_spread,
      total_candidates, passed
    """
    if df_1min.empty:
        return {"error": "Empty buffer -- run backfill first."}

    df = df_1min.copy().reset_index(drop=True)
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
    if "Close" not in df.columns:
        return {"error": "Missing Close column."}

    # Compute EMA crossover as a proxy signal
    ema9  = ta.ema(df["Close"], length=EMA_FAST)
    ema21 = ta.ema(df["Close"], length=EMA_SLOW)
    if ema9 is None or ema21 is None:
        return {"error": "Could not compute EMAs."}

    df["ema9"]  = ema9
    df["ema21"] = ema21
    df = df.dropna(subset=["ema9","ema21"]).reset_index(drop=True)

    # Proxy probability: sigmoid of ema spread normalised by ATR
    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift()).abs()
    lc  = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([hl.reset_index(drop=True),
                     hc.reset_index(drop=True),
                     lc.reset_index(drop=True)], axis=1).max(axis=1)
    atr = tr.rolling(ATR_PERIOD).mean().bfill()
    df["ATR_14"] = atr.values

    spread_norm = (df["ema9"] - df["ema21"]) / (df["ATR_14"] + 1e-9)
    df["prob_up"] = 1 / (1 + np.exp(-spread_norm * 3))  # sigmoid

    # Determine raw signal
    def _raw_sig(row):
        if row["ema9"] > row["ema21"]: return "LONG"
        if row["ema9"] < row["ema21"]: return "SHORT"
        return "NEUTRAL"

    df["raw_signal"] = df.apply(_raw_sig, axis=1)
    candidates = df[df["raw_signal"] != "NEUTRAL"].copy()
    total = len(candidates)
    if total == 0:
        return {"error": "No candidates generated."}

    tracker   = ConfirmationTracker(n=N_CONFIRM_BARS)
    long_t, short_t = load_1min_thresholds()

    filtered_confirm   = 0
    filtered_atr       = 0
    filtered_mtf       = 0
    filtered_threshold = 0
    filtered_spread    = 0
    passed_count       = 0

    daily_atr  = _get_daily_atr()
    atr_floor  = daily_atr / math.sqrt(TRADING_MINS_PER_DAY) * ATR_FLOOR_MULTIPLIER if daily_atr > 0 else 0.0

    for i, (_, row) in enumerate(candidates.iterrows()):
        sig      = row["raw_signal"]
        prob_up  = float(row["prob_up"])
        atr_1min = float(row["ATR_14"]) if not math.isnan(row["ATR_14"]) else 0.0

        # Gate 1
        tracker.update(sig)
        confirmed, count = tracker.is_confirmed()
        if not confirmed:
            filtered_confirm += 1
            continue

        # Gate 2
        if atr_floor > 0 and atr_1min > 0 and atr_1min < atr_floor:
            filtered_atr += 1
            continue

        # Gate 3 (simplified: use last 100 bars around this index)
        buf_slice = df.iloc[max(0, i-100):i+1].copy()
        r5  = _resample_df(buf_slice, "5min")
        r15 = _resample_df(buf_slice, "15min")
        t5  = _ema_trend(r5)
        t15 = _ema_trend(r15)
        fail = []
        if MTF_REQUIRE_5MIN  and t5  not in (sig, "NEUTRAL"): fail.append("5min")
        if MTF_REQUIRE_15MIN and t15 not in (sig, "NEUTRAL"): fail.append("15min")
        if fail:
            filtered_mtf += 1
            continue

        # Gate 4
        if sig == "LONG"  and prob_up < long_t:
            filtered_threshold += 1; continue
        if sig == "SHORT" and prob_up > short_t:
            filtered_threshold += 1; continue

        # Gate 5 (use default assumptions)
        spread_cost = 0.5 * 10.0 * 0.05  # 0.5 pip * $10/pip/lot * 0.05 lot
        if spread_cost / 50.0 > MAX_SPREAD_PCT_OF_RISK:
            filtered_spread += 1; continue

        passed_count += 1

    # Compute time span
    if "Datetime" in df.columns:
        span_hrs = (df["Datetime"].max() - df["Datetime"].min()).total_seconds() / 3600
    else:
        span_hrs = len(df) / 60  # assume 1-min bars
    tph = round(passed_count / span_hrs, 2) if span_hrs > 0 else 0.0

    def pct(n): return round(n / total * 100, 1) if total > 0 else 0.0

    result = {
        "total_candidates":      total,
        "passed":                passed_count,
        "trades_per_hour":       tph,
        "span_hours":            round(span_hrs, 1),
        "pct_filtered_confirm":  pct(filtered_confirm),
        "pct_filtered_atr":      pct(filtered_atr),
        "pct_filtered_mtf":      pct(filtered_mtf),
        "pct_filtered_threshold":pct(filtered_threshold),
        "pct_filtered_spread":   pct(filtered_spread),
        "settings": {
            "n_confirm_bars":        N_CONFIRM_BARS,
            "atr_floor_multiplier":  ATR_FLOOR_MULTIPLIER,
            "mtf_5min":              MTF_REQUIRE_5MIN,
            "mtf_15min":             MTF_REQUIRE_15MIN,
            "long_threshold":        long_t,
            "short_threshold":       short_t,
        }
    }

    print("\n" + "="*60)
    print("  1-MIN SIGNAL FILTER SIMULATION")
    print("="*60)
    print(f"  Total candidates      : {total}")
    print(f"  Passed all gates      : {passed_count}")
    print(f"  Span                  : {span_hrs:.1f} hours")
    print(f"  Trades per hour       : {tph}  <-- REVIEW BEFORE GO-LIVE")
    print(f"")
    print(f"  Filter breakdown:")
    print(f"    Gate 1 Confirm      : {filtered_confirm} ({pct(filtered_confirm)}%) rejected")
    print(f"    Gate 2 ATR floor    : {filtered_atr} ({pct(filtered_atr)}%) rejected")
    print(f"    Gate 3 MTF          : {filtered_mtf} ({pct(filtered_mtf)}%) rejected")
    print(f"    Gate 4 Threshold    : {filtered_threshold} ({pct(filtered_threshold)}%) rejected")
    print(f"    Gate 5 Spread       : {filtered_spread} ({pct(filtered_spread)}%) rejected")
    print("="*60)
    if tph > 4:
        print(f"  WARNING: {tph} trades/hr is HIGH. Consider tightening filters.")
    elif tph < 0.1:
        print(f"  WARNING: {tph} trades/hr is VERY LOW. Filters may be too tight.")
    else:
        print(f"  INFO: {tph} trades/hr is within a reasonable range (0.1 - 4/hr).")
    print()
    return result

# ---- CLI --------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1-Min Signal Filter")
    parser.add_argument("--simulate", action="store_true",
                        help="Run trade frequency simulation on buffered 1-min data")
    parser.add_argument("--calibrate", action="store_true",
                        help="Calibrate probability thresholds from buffered data")
    args = parser.parse_args()

    if args.simulate or args.calibrate:
        try:
            from step1b_collect_1min import backfill_from_mt5, _update_buffer, get_buffer, _init_db
            _init_db()
            print("Loading 1-min data for simulation (2-day backfill)...")
            df = backfill_from_mt5(days=2)
            if df.empty:
                print("No 1-min data available. Run step1b --backfill first.")
                sys.exit(1)
            _update_buffer(df)
            buf = get_buffer()
        except ImportError:
            print("step1b_collect_1min not found. Using yfinance directly.")
            import yfinance as yf
            raw = yf.download("GC=F", period="5d", interval="1m", auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw[["Open","High","Low","Close","Volume"]].copy()
            raw.index.name = "Datetime"
            buf = raw.reset_index()
            buf["Datetime"] = pd.to_datetime(buf["Datetime"]).dt.tz_localize(None)

        if args.simulate:
            simulate_trade_frequency(buf)
        if args.calibrate:
            prob_col = "prob_up" if "prob_up" in buf.columns else "Close"
            calibrate_thresholds(buf[prob_col])
    else:
        print("Usage: python step2b_1min_signal_filter.py --simulate | --calibrate")
