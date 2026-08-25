"""
agents/context_builder.py
=========================
Assembles live market data into structured context dicts for each analyst agent.

Each builder function returns a dict that is formatted into the agent's user prompt.
Data sources mirror the existing pipeline — no new data fetching logic, just
reading what is already collected by step1-step3.

Public API:
    build_technical_context()   -> dict  (RSI, MACD, BB, ATR, EMA, quant signal)
    build_macro_context()       -> dict  (oil, VIX, DXY, FRED series)
    build_sentiment_context()   -> dict  (FinBERT, VADER, GDELT categories)
    build_calendar_context()    -> dict  (upcoming macro events)
    check_event_trigger()       -> (bool, str)  (should activate, reason)
    build_quant_context()       -> dict  (ensemble probs, signal, SL/TP params)
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Import macro_calendar from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
import macro_calendar

from agents.config import cfg

log = logging.getLogger("agents.context")

# ── yfinance tickers ─────────────────────────────────────────────────────────
_OIL_TICKER = "CL=F"
_VIX_TICKER = "^VIX"
_DXY_TICKER = "DX-Y.NYB"
_GOLD_TICKER = "GC=F"


def _safe_read_csv(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    """Read CSV safely, return None on error."""
    try:
        if not path.exists():
            log.warning("File not found: %s", path)
            return None
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        log.warning("Failed to read %s: %s", path, e)
        return None


def _fetch_yf_price(ticker: str, period: str = "10d") -> Optional[pd.DataFrame]:
    """Fetch OHLCV data from yfinance, return None on failure."""
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period=period)
        if df.empty:
            return None
        return df
    except Exception as e:
        log.warning("yfinance %s failed: %s", ticker, e)
        return None


def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate ATR from OHLC DataFrame."""
    try:
        hl  = df["High"] - df["Low"]
        hpc = (df["High"] - df["Close"].shift()).abs()
        lpc = (df["Low"]  - df["Close"].shift()).abs()
        tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL CONTEXT — for Technical Analyst
# ══════════════════════════════════════════════════════════════════════════════

def build_technical_context() -> dict:
    """
    Reads live_inference_data.csv (the feature row going into the ensemble)
    and xauusd_raw_prices.csv for ATR/price context.

    Returns the raw indicator values the Technical Analyst reasons over.
    Also includes the quant ensemble's probability and signal.
    """
    ctx: dict = {}

    # ── Live inference feature row ─────────────────────────────────────────
    inf_df = _safe_read_csv(cfg.inference_data_path)
    if inf_df is not None and not inf_df.empty:
        row = inf_df.iloc[-1]
        ctx["inference_date"]   = str(row.get("Date", "unknown"))
        ctx["rsi_14"]           = float(row.get("RSI_14", 50.0))
        ctx["macd_line"]        = float(row.get("MACD_12_26_9", 0.0))
        ctx["macd_histogram"]   = float(row.get("MACDh_12_26_9", 0.0))
        ctx["macd_signal"]      = float(row.get("MACDs_12_26_9", 0.0))
        ctx["bb_width"]         = float(row.get("BB_Width", 0.0))
        ctx["bbl_ratio"]        = float(row.get("BBL_Ratio", 1.0))
        ctx["bbm_ratio"]        = float(row.get("BBM_Ratio", 1.0))
        ctx["bbu_ratio"]        = float(row.get("BBU_Ratio", 1.0))
        ctx["ema_50_ratio"]     = float(row.get("EMA_50_Ratio", 1.0))
        ctx["atr_14"]           = float(row.get("ATR_14", 0.0))
        ctx["high_vol_regime"]  = int(row.get("High_Vol_Regime", 0))
        ctx["rsi_regime"]       = int(row.get("RSI_Regime", 0))
        ctx["close_return"]     = float(row.get("Close_Return", 0.0))
        ctx["return_vol_20"]    = float(row.get("Return_Vol_20", 0.0))
        ctx["return_momentum_5"]= float(row.get("Return_Momentum_5", 0.0))

    # ── Raw prices for ATR context ─────────────────────────────────────────
    price_df = _safe_read_csv(cfg.raw_prices_path)
    if price_df is not None and not price_df.empty:
        price_df["Date"] = pd.to_datetime(price_df["Date"])
        price_df = price_df.sort_values("Date").reset_index(drop=True)
        ctx["latest_close"]     = float(price_df["Close"].iloc[-1])
        ctx["latest_high"]      = float(price_df["High"].iloc[-1])
        ctx["latest_low"]       = float(price_df["Low"].iloc[-1])
        ctx["atr_14_raw"]       = _calc_atr(price_df, 14)

        # ATR percentile rank vs 20-day rolling window
        price_df["ATR"] = price_df["High"].sub(price_df["Low"])  # simplified TR
        recent_atrs = price_df["ATR"].tail(22).values
        if len(recent_atrs) > 1:
            current_atr = recent_atrs[-1]
            ctx["atr_percentile_rank"] = float(
                np.mean(recent_atrs[:-1] < current_atr) * 100
            )
        else:
            ctx["atr_percentile_rank"] = 50.0

    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# MACRO CONTEXT — for Macro/Fundamental Analyst
# ══════════════════════════════════════════════════════════════════════════════

def build_macro_context() -> dict:
    """
    Fetches current oil, VIX, DXY via yfinance (real-time) and
    reads FRED macro series from fred_macro_raw.csv (slower-moving indicators).

    Chai et al. 2021 SVAR context is surfaced here for the agent's reasoning.
    """
    ctx: dict = {}

    # ── Crude Oil (CL=F) — dominant gold driver: ~89% variance per Chai 2021 ─
    oil_df = _fetch_yf_price(_OIL_TICKER, period="15d")
    if oil_df is not None and len(oil_df) >= 2:
        ctx["oil_price_latest"]  = float(oil_df["Close"].iloc[-1])
        ctx["oil_price_5d_ago"]  = float(oil_df["Close"].iloc[-min(5, len(oil_df)-1)])
        ctx["oil_return_5d"]     = float(
            (oil_df["Close"].iloc[-1] - oil_df["Close"].iloc[-min(5, len(oil_df)-1)])
            / oil_df["Close"].iloc[-min(5, len(oil_df)-1)] * 100
        )
        ctx["oil_return_1d"]     = float(oil_df["Close"].pct_change().iloc[-1] * 100)
        ctx["oil_shock_active"]  = abs(ctx["oil_return_5d"]) > 3.0
        ctx["oil_shock_direction"] = (
            "BULLISH" if ctx["oil_return_5d"] > 3.0 else
            "BEARISH" if ctx["oil_return_5d"] < -3.0 else
            "NEUTRAL"
        )
    else:
        # Fallback: use feature row WTI data
        inf_df = _safe_read_csv(cfg.inference_data_path)
        if inf_df is not None and not inf_df.empty:
            row = inf_df.iloc[-1]
            ctx["oil_return_1d"]    = float(row.get("WTI_Crude_Oil_Return", 0.0))
            ctx["oil_return_5d"]    = sum(
                float(row.get(f"WTI_Crude_Oil_Lag_{i}_Diff", 0.0)) for i in range(1, 6)
            )
            ctx["oil_shock_active"] = abs(ctx["oil_return_5d"]) > 3.0
            ctx["oil_shock_direction"] = (
                "BULLISH" if ctx.get("oil_return_5d", 0) > 3.0 else
                "BEARISH" if ctx.get("oil_return_5d", 0) < -3.0 else
                "NEUTRAL"
            )
            ctx["oil_price_latest"] = 0.0

    # ── VIX (^VIX) — positive relationship with gold, growing over 10 periods ─
    vix_df = _fetch_yf_price(_VIX_TICKER, period="10d")
    if vix_df is not None and len(vix_df) >= 2:
        ctx["vix_latest"]       = float(vix_df["Close"].iloc[-1])
        ctx["vix_5d_ago"]       = float(vix_df["Close"].iloc[-min(5, len(vix_df)-1)])
        ctx["vix_change_5d"]    = float(vix_df["Close"].iloc[-1] - vix_df["Close"].iloc[-min(5, len(vix_df)-1)])
        ctx["vix_direction"]    = "BULLISH" if ctx["vix_change_5d"] > 0 else "BEARISH"
        ctx["vix_above_panic"]  = ctx["vix_latest"] > cfg.vix_panic_threshold
    else:
        # Fallback: read from feature row
        inf_df = _safe_read_csv(cfg.inference_data_path)
        if inf_df is not None and not inf_df.empty:
            row = inf_df.iloc[-1]
            vix_diff = float(row.get("VIX_Index_Diff", 0.0))
            vix_lag1 = float(row.get("VIX_Index_Lag_1_Diff", 0.0))
            ctx["vix_change_5d"]   = vix_diff
            ctx["vix_direction"]   = "BULLISH" if vix_diff > 0 else "BEARISH"
            ctx["vix_latest"]      = 20.0  # unknown without live data
            ctx["vix_above_panic"] = False

    # ── DXY (DX-Y.NYB) — negative but WEAK driver (~0.5% variance, Chai 2021) ─
    dxy_df = _fetch_yf_price(_DXY_TICKER, period="10d")
    if dxy_df is not None and len(dxy_df) >= 2:
        ctx["dxy_latest"]       = float(dxy_df["Close"].iloc[-1])
        ctx["dxy_change_5d"]    = float(dxy_df["Close"].iloc[-1] - dxy_df["Close"].iloc[-min(5, len(dxy_df)-1)])
        ctx["dxy_direction"]    = "BEARISH" if ctx["dxy_change_5d"] > 0 else "BULLISH"
        ctx["dxy_return_5d"]    = float(
            (dxy_df["Close"].iloc[-1] - dxy_df["Close"].iloc[-min(5, len(dxy_df)-1)])
            / dxy_df["Close"].iloc[-min(5, len(dxy_df)-1)] * 100
        )
    else:
        inf_df = _safe_read_csv(cfg.inference_data_path)
        if inf_df is not None and not inf_df.empty:
            row = inf_df.iloc[-1]
            ctx["dxy_change_5d"]  = float(row.get("DXY_Index_Diff", 0.0))
            ctx["dxy_direction"]  = "BEARISH" if ctx["dxy_change_5d"] > 0 else "BULLISH"
            ctx["dxy_latest"]     = 100.0  # unknown
            ctx["dxy_return_5d"]  = 0.0

    # ── FRED Macro Series (slower-moving) ──────────────────────────────────
    fred_df = _safe_read_csv(cfg.fred_macro_path)
    if fred_df is not None and not fred_df.empty:
        # Get latest non-null value for each series
        fred_latest: dict = {}
        for col in fred_df.columns:
            if col.lower() in ("date", "index"):
                continue
            series = fred_df[col].dropna()
            if not series.empty:
                fred_latest[col] = float(series.iloc[-1])

        ctx["fed_funds_rate"]       = fred_latest.get("FedFunds_Rate",    fred_latest.get("Federal Funds Rate", None))
        ctx["us_10y_yield"]         = fred_latest.get("US_10Y_Yield",     fred_latest.get("10-Year Treasury Yield", None))
        ctx["cpi_us"]               = fred_latest.get("CPI_US",           fred_latest.get("CPI", None))
        ctx["unemployment_rate"]    = fred_latest.get("Unemployment_Rate", None)
        ctx["real_gdp_growth"]      = fred_latest.get("Real_GDP_Growth",   None)

    # Shock digestion window estimate (crude: based on how recent oil shock was)
    oil_return = ctx.get("oil_return_5d", 0.0)
    oil_ret_1d = ctx.get("oil_return_1d", 0.0)
    if abs(oil_return) > 3.0:
        if abs(oil_ret_1d) > 2.0:
            ctx["digestion_window"] = "EARLY"
            ctx["days_since_shock"] = 0
        elif abs(oil_ret_1d) > 0.5:
            ctx["digestion_window"] = "MID"
            ctx["days_since_shock"] = 2
        else:
            ctx["digestion_window"] = "LATE"
            ctx["days_since_shock"] = 4
    else:
        ctx["digestion_window"] = "NONE"
        ctx["days_since_shock"] = None

    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# SENTIMENT CONTEXT — for News/Sentiment Analyst
# ══════════════════════════════════════════════════════════════════════════════

def build_sentiment_context() -> dict:
    """
    Reads FinBERT scores from news_sentiment_cache.csv and
    GDELT categories from gdelt_news_raw.csv (last 4 hours).
    VADER scores are read from the live_inference_data.csv feature row.
    """
    ctx: dict = {}

    # ── FinBERT scores ────────────────────────────────────────────────────
    sent_df = _safe_read_csv(cfg.sentiment_cache_path)
    if sent_df is not None and not sent_df.empty and "Polarity_Score" in sent_df.columns:
        scores = sent_df["Polarity_Score"].dropna()
        ctx["finbert_mean"]       = float(scores.mean())
        ctx["finbert_latest_5"]   = float(scores.tail(5).mean())
        ctx["news_count"]         = len(scores)

        # Sample headlines
        if "Headline" in sent_df.columns:
            ctx["headline_sample"] = sent_df["Headline"].tail(5).tolist()
        else:
            ctx["headline_sample"] = []
    else:
        ctx["finbert_mean"]     = 0.0
        ctx["finbert_latest_5"] = 0.0
        ctx["news_count"]       = 0
        ctx["headline_sample"]  = []

    # ── Live VADER score from inference feature row ───────────────────────
    inf_df = _safe_read_csv(cfg.inference_data_path)
    if inf_df is not None and not inf_df.empty:
        row = inf_df.iloc[-1]
        ctx["sentiment_sma5"]    = float(row.get("Sentiment_SMA_5", 0.0))
        ctx["mean_sentiment"]    = float(row.get("Mean_Sentiment", 0.0))
        ctx["war_geo_impact"]    = float(row.get("WAR_GEOPOLITICAL_Impact", 0.0))
        ctx["war_geo_count"]     = int(row.get("WAR_GEOPOLITICAL_Count", 0))
        ctx["fed_policy_impact"] = float(row.get("FED_POLICY_Impact", 0.0))
        ctx["fed_policy_count"]  = int(row.get("FED_POLICY_Count", 0))
        ctx["inflation_impact"]  = float(row.get("INFLATION_Impact", 0.0))
        ctx["inflation_count"]   = int(row.get("INFLATION_Count", 0))
        ctx["dollar_fx_impact"]  = float(row.get("DOLLAR_FX_Impact", 0.0))
        ctx["gold_market_impact"]= float(row.get("GOLD_MARKET_Impact", 0.0))
        ctx["geo_surge_score"]   = float(row.get("Geo_Surge_Score", 0.0))
        ctx["news_surprise"]     = float(row.get("News_Surprise_Score", 0.0))
    else:
        for k in ["sentiment_sma5","mean_sentiment","war_geo_impact","war_geo_count",
                  "fed_policy_impact","fed_policy_count","inflation_impact","inflation_count",
                  "dollar_fx_impact","gold_market_impact","geo_surge_score","news_surprise"]:
            ctx[k] = 0.0

    # ── Blended score (matches step11_api_server.py formula) ─────────────
    ctx["vader_score"] = ctx.get("sentiment_sma5", 0.0)  # proxy: sentiment SMA
    ctx["blended_score"] = 0.7 * ctx.get("finbert_mean", 0.0) + 0.3 * ctx.get("vader_score", 0.0)

    # ── GDELT active categories (last 4 hours) ────────────────────────────
    active_categories: list[str] = []
    gdelt_df = _safe_read_csv(cfg.gdelt_path)
    if gdelt_df is not None and not gdelt_df.empty:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=cfg.gdelt_trigger_lookback_hours)
        # Try to find a datetime column
        dt_col = next((c for c in gdelt_df.columns if "date" in c.lower() or "time" in c.lower()), None)
        if dt_col:
            try:
                gdelt_df[dt_col] = pd.to_datetime(gdelt_df[dt_col], utc=True, errors="coerce")
                recent = gdelt_df[gdelt_df[dt_col] >= cutoff]
                if "Category" in recent.columns:
                    cats = recent["Category"].dropna().unique().tolist()
                    active_categories = [str(c) for c in cats]
                elif "EventCategory" in recent.columns:
                    cats = recent["EventCategory"].dropna().unique().tolist()
                    active_categories = [str(c) for c in cats]
            except Exception:
                pass

        # Fallback: check for category columns from feature row
        if not active_categories and ctx.get("war_geo_count", 0) > 0:
            active_categories.append("WAR_GEOPOLITICAL")
        if not active_categories and ctx.get("fed_policy_count", 0) > 0:
            active_categories.append("FED_POLICY")

    ctx["active_gdelt_categories"] = active_categories
    ctx["war_geo_active"]          = "WAR_GEOPOLITICAL" in active_categories or "WAR_MILITARY" in active_categories
    ctx["high_impact_event_active"] = ctx["war_geo_active"] or "FED_POLICY" in active_categories

    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# CALENDAR CONTEXT — for Event/Calendar Analyst
# ══════════════════════════════════════════════════════════════════════════════

def build_calendar_context() -> dict:
    """
    Reads upcoming macro events from macro_calendar.py.
    Computes caution_level and volatility_flag for the Risk team.
    """
    ctx: dict = {}

    try:
        all_events = macro_calendar.get_upcoming_events(days_ahead=2)
    except Exception as e:
        log.warning("macro_calendar failed: %s", e)
        all_events = []

    now = datetime.now()

    events_2h:  list[dict] = []
    events_24h: list[dict] = []
    next_high:  Optional[str] = None

    for ev in all_events:
        try:
            ev_date = datetime.fromisoformat(ev["date"])
            hours_until = (ev_date - now).total_seconds() / 3600.0
        except Exception:
            hours_until = ev.get("days_until", 99) * 24.0

        ev_enriched = {**ev, "hours_until": round(hours_until, 2)}

        if ev["impact"] == "HIGH" and abs(hours_until) <= cfg.event_window_hours:
            events_2h.append(ev_enriched)

        if hours_until >= 0 and hours_until <= 24:
            events_24h.append(ev_enriched)

        if ev["impact"] == "HIGH" and hours_until >= 0 and next_high is None:
            next_high = f"{ev['event']} in {hours_until:.1f}h"

    ctx["events_within_2h"]  = events_2h
    ctx["events_within_24h"] = events_24h
    ctx["event_window_active"] = len(events_2h) > 0
    ctx["next_high_impact_event"] = next_high

    # ── Volatility flag ────────────────────────────────────────────────────
    if events_2h:
        ctx["volatility_flag"] = "IMMINENT"
    elif any(e["impact"] == "HIGH" for e in events_24h):
        ctx["volatility_flag"] = "APPROACHING"
    else:
        ctx["volatility_flag"] = "CLEAR"

    # ── Caution level ──────────────────────────────────────────────────────
    if ctx["volatility_flag"] == "IMMINENT":
        ctx["caution_level"] = 3
    elif ctx["volatility_flag"] == "APPROACHING":
        ctx["caution_level"] = 2
    elif any(e["impact"] == "MEDIUM" for e in events_24h):
        ctx["caution_level"] = 1
    else:
        ctx["caution_level"] = 0

    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# QUANT CONTEXT — pass-through from existing ensemble
# ══════════════════════════════════════════════════════════════════════════════

def build_quant_context() -> dict:
    """
    Reads the existing quant ensemble's output from live_inference_data.csv
    and the test_predictions.csv for adaptive thresholds.

    This context is passed to the Trader and Fund Manager agents
    so they always know what the quant model said.
    """
    ctx: dict = {
        "p_cat": 0.5, "p_xgb": 0.5, "p_lgb": 0.5,
        "prob_up": 0.5, "quant_signal": "HOLD",
        "long_threshold": 0.65, "short_threshold": 0.35,
        "atr": 0.0, "entry_price": 0.0,
        "sl_mult": 1.5, "tp_mult": 3.0,
    }

    # Read adaptive thresholds from test_predictions.csv
    preds_df = _safe_read_csv(cfg.test_preds_path)
    if preds_df is not None and "Ensemble_Prob" in preds_df.columns and len(preds_df) > 20:
        probs = preds_df["Ensemble_Prob"].dropna()
        ctx["long_threshold"]  = float(np.percentile(probs, 70))
        ctx["short_threshold"] = float(np.percentile(probs, 30))

    # Read model threshold JSON for SL/TP multipliers
    threshold_path = cfg.root_dir / "model_threshold.json"
    if threshold_path.exists():
        import json
        try:
            with open(threshold_path) as f:
                t = json.load(f)
            ctx["sl_mult"] = t.get("sl_atr_mult", 1.5)
            ctx["tp_mult"] = t.get("tp_atr_mult", 3.0)
        except Exception:
            pass

    # Read latest prices for entry/SL/TP calculation
    price_df = _safe_read_csv(cfg.raw_prices_path)
    if price_df is not None and not price_df.empty:
        price_df["Date"] = pd.to_datetime(price_df["Date"])
        price_df = price_df.sort_values("Date").reset_index(drop=True)
        ctx["entry_price"] = float(price_df["Close"].iloc[-1])
        ctx["atr"]         = _calc_atr(price_df, 14)

    # NOTE: p_cat, p_xgb, p_lgb, prob_up, quant_signal are injected
    # by the orchestrator from the live inference run (step10/step11 output).
    # The defaults above are fallbacks if the orchestrator doesn't provide them.
    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# EVENT TRIGGER CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_event_trigger() -> tuple[bool, str]:
    """
    Determine whether the agent layer should activate.

    Returns:
        (should_activate: bool, reason: str)

    Activation conditions (OR logic):
      1. A HIGH-impact macro event (NFP/FOMC/CPI) is within ±event_window_hours
      2. An active WAR_GEOPOLITICAL/WAR_MILITARY GDELT event in past gdelt_lookback_hours
    """
    # Check calendar trigger
    cal = build_calendar_context()
    if cal["event_window_active"]:
        events = cal["events_within_2h"]
        names = [e["event"] for e in events]
        return True, f"Calendar trigger: {', '.join(names)} within {cfg.event_window_hours}h"

    # Check GDELT war/geopolitical trigger
    sent = build_sentiment_context()
    if sent.get("war_geo_active", False):
        return True, f"GDELT WAR_GEOPOLITICAL event active in past {cfg.gdelt_trigger_lookback_hours}h"

    return False, "No trigger condition met"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("\n[context_builder] Running context checks...")

    triggered, reason = check_event_trigger()
    print(f"\nTrigger active: {triggered}")
    print(f"Reason: {reason}")

    cal = build_calendar_context()
    print(f"\nCalendar: volatility_flag={cal['volatility_flag']}, caution_level={cal['caution_level']}")
    print(f"Events within 2h: {len(cal['events_within_2h'])}")
    print(f"Events within 24h: {len(cal['events_within_24h'])}")
    if cal.get("next_high_impact_event"):
        print(f"Next HIGH event: {cal['next_high_impact_event']}")

    sent = build_sentiment_context()
    print(f"\nSentiment: finbert={sent['finbert_mean']:.4f}, blended={sent['blended_score']:.4f}")
    print(f"Active GDELT categories: {sent['active_gdelt_categories']}")
    print(f"War active: {sent['war_geo_active']}")

    macro = build_macro_context()
    print(f"\nMacro: oil_return_5d={macro.get('oil_return_5d', 0):.2f}%")
    print(f"VIX: {macro.get('vix_latest', 'N/A')} (above_panic={macro.get('vix_above_panic', False)})")
    print(f"DXY: {macro.get('dxy_latest', 'N/A')} direction={macro.get('dxy_direction', 'N/A')}")
    print(f"Oil shock active: {macro.get('oil_shock_active', False)} ({macro.get('oil_shock_direction', 'N/A')})")
    print(f"Digestion window: {macro.get('digestion_window', 'N/A')}")
