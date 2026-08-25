"""
agents/analyst_team.py
======================
Seven parallel specialist analyst agents — Tier 1 (gemini-2.0-flash).

Each specialist receives micro-focused structured context, reasons over it,
and returns a validated Pydantic report. All seven run concurrently via asyncio.gather().
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from agents.context_builder import (
    build_technical_context,
    build_macro_context,
    build_sentiment_context,
    build_calendar_context,
)
from agents.llm_client import llm
from agents.schemas import (
    AgentCallMetadata,
    TechnicalSpecialistReport,
    QuantitativeSpecialistReport,
    MacroSVARSpecialistReport,
    PolicyFREDSpecialistReport,
    SentimentSpecialistReport,
    GeopoliticalGDELTSpecialistReport,
    CalendarSpecialistReport,
)

log = logging.getLogger("agents.analyst_team")


@dataclass
class AnalystTeamResult:
    technical:     TechnicalSpecialistReport
    quantitative:  QuantitativeSpecialistReport
    macro_svar:    MacroSVARSpecialistReport
    policy_fred:   PolicyFREDSpecialistReport
    sentiment:     SentimentSpecialistReport
    geopolitical:  GeopoliticalGDELTSpecialistReport
    calendar:      CalendarSpecialistReport
    metadata:      list[AgentCallMetadata]


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

_TECHNICAL_SPECIALIST_SYSTEM = """You are the Technical Specialist for an XAU/USD (gold) trading system.
Your role is to analyse the current state of technical indicators and produce a structured TechnicalSpecialistReport.

RULES:
1. Base ALL conclusions only on the numeric indicator values provided — no guessing.
2. RSI overbought = RSI > 70. RSI oversold = RSI < 30. Flag both explicitly.
3. MACD crossover = MACD line crossed above/below signal line in last 3 bars.
4. Bollinger Band squeeze = BB_Width below recent average (shrinking bands = breakout pending).
5. Volatility regime: HIGH_VOL_REGIME=1 means elevated volatility.
6. Under no circumstances should you mention or evaluate the quantitative machine learning models (CatBoost/XGBoost/LightGBM) here — that is another specialist's job.
7. Output valid JSON matching the schema exactly."""

_QUANTITATIVE_SPECIALIST_SYSTEM = """You are the Quantitative Specialist for an XAU/USD (gold) trading system.
Your role is to analyse the outputs of the machine learning ensemble (CatBoost, XGBoost, LightGBM, and Meta-Learner probability and signal) and produce a structured QuantitativeSpecialistReport.

RULES:
1. Report the ensemble P(up) and signal faithfully.
2. Do not evaluate standard technical charts or indicators (RSI, MACD, etc.) manually — focus on model probabilities, consensus (whether >=2/3 models agree), and long/short thresholds.
3. Output valid JSON matching the schema exactly."""

_MACRO_SVAR_SPECIALIST_SYSTEM = """You are the Macro/SVAR Specialist for an XAU/USD (gold) trading system.
Your analysis MUST be grounded in the structural relationships from Chai et al. 2021's SVAR analysis of gold price drivers.

MANDATORY CHAI ET AL. 2021 SVAR FRAMEWORK:
  1. CRUDE OIL RETURNS explain approximately 89% of gold's forecast error variance.
     → Oil price shocks are the DOMINANT driver of gold. Weight heavily.
     → An oil shock (>3% in 5 days) should dominate your entire analysis.
  
  2. VIX has a POSITIVE and GROWING relationship with gold returns.
     → VIX explains 4% of variance initially, rising to ~7.5% over 10 trading periods.
     → A rising VIX is BULLISH for gold. A falling VIX is mildly BEARISH.
  
  3. DXY (USD index) has a NEGATIVE but VERY WEAK relationship (~0.5% of variance).
     → Do NOT overweight dollar moves. Only mention DXY if its move is unusually large (>1% in a day).
  
  4. All oil/VIX/DXY shocks are typically digested by the market within ~5 trading days.
     → Always state whether you believe we are EARLY, MID, or LATE in that window.

RULES:
1. Focus on oil returns, oil shock, VIX level/direction, DXY trend.
2. Digestion window status must be EARLY/MID/LATE/NONE.
3. Do not analyze slower FRED statistics (like GDP, Unemployment) or headlines here.
4. Output valid JSON matching the schema exactly."""

_POLICY_FRED_SPECIALIST_SYSTEM = """You are the Policy/FRED Specialist for an XAU/USD (gold) trading system.
Your role is to analyse the slower-moving macroeconomic policy indicators (Fed Funds rate, 10Y yields, CPI inflation, GDP, unemployment rate) and identify the Fed policy stance (HAWKISH/DOVISH/NEUTRAL) and macro risk factors.

RULES:
1. Base your analysis on FRED data provided.
2. Focus on yield stack, interest rate direction, CPI trajectory.
3. Do not analyze technical indicators, short-term oil shocks, or news sentiment.
4. Output valid JSON matching the schema exactly."""

_SENTIMENT_SPECIALIST_SYSTEM = """You are the News/Sentiment Specialist for an XAU/USD (gold) trading system.
Your role is to assess gold news sentiment polarity using FinBERT, VADER, and blended scores, and provide headline evidence.

RULES:
1. Base sentiment direction (BULLISH/BEARISH/NEUTRAL) and intensity (STRONG/MODERATE/WEAK) on the blended sentiment score (70% FinBERT + 30% VADER).
2. Bullet the recent headlines (up to 5).
3. Do not analyze GDELT event categories, calendar events, or technical charts.
4. Output valid JSON matching the schema exactly."""

_GEOPOLITICAL_GDELT_SPECIALIST_SYSTEM = """You are the Geopolitical GDELT Specialist for an XAU/USD (gold) trading system.
Your role is to analyse active GDELT event categories, war geopolitical flags, surge scores, and news surprise scores.

RULES:
1. High impact event active = True only if WAR_MILITARY or FED_POLICY is active.
2. War geopolitical active = True only if WAR_MILITARY or WAR_GEOPOLITICAL is active.
3. Do not analyze FinBERT sentiment scores, oil prices, or calendar events.
4. Output valid JSON matching the schema exactly."""

_CALENDAR_SPECIALIST_SYSTEM = """You are the Calendar Specialist for an XAU/USD (gold) trading system.
Your role is to analyse the macroeconomic events calendar (CPI, NFP, FOMC) and recommend caution levels (0-3).

CAUTION LEVEL GUIDE:
  0 = CLEAR:     No HIGH-impact events in next 24 hours. Normal risk.
  1 = LOW:       A MEDIUM-impact event within 24 hours. Slight caution.
  2 = MEDIUM:    A HIGH-impact event within 24 hours. Reduce position if appropriate.
  3 = HIGH:      A HIGH-impact event within 2 hours (IMMINENT). Maximum caution.

RULES:
1. event_window_active = True only if a HIGH-impact event is within ±2 hours of NOW.
2. volatility_flag: IMMINENT if events_within_2h is non-empty, APPROACHING if a HIGH event is within 24h, CLEAR otherwise.
3. caution_level must match the volatility_flag exactly (3=IMMINENT, 2=APPROACHING, etc.).
4. Do not analyze technical metrics or news sentiment.
5. Output valid JSON matching the schema exactly."""


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL ANALYST RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

async def _run_technical_specialist() -> tuple[TechnicalSpecialistReport, AgentCallMetadata]:
    tech_ctx = build_technical_context()
    user_prompt = f"""Current XAU/USD Technical Indicator State:

Date: {tech_ctx.get('inference_date', 'latest')}
Gold Price: ${tech_ctx.get('latest_close', 0):.2f}
ATR(14): {tech_ctx.get('atr_14', tech_ctx.get('atr_14_raw', 0)):.2f}

MOMENTUM:
  RSI(14): {tech_ctx.get('rsi_14', 50):.2f}
  RSI Regime Flag: {tech_ctx.get('rsi_regime', 0)} (1=overbought, -1=oversold)
  MACD Line: {tech_ctx.get('macd_line', 0):.4f}
  MACD Histogram: {tech_ctx.get('macd_histogram', 0):.4f}
  MACD Signal: {tech_ctx.get('macd_signal', 0):.4f}

TREND:
  EMA50 Ratio (price/EMA50): {tech_ctx.get('ema_50_ratio', 1):.4f}
  Return Momentum (5d): {tech_ctx.get('return_momentum_5', 0):.4f}

VOLATILITY:
  BB Width: {tech_ctx.get('bb_width', 0):.4f}
  BBL Ratio: {tech_ctx.get('bbl_ratio', 1):.4f}
  BBM Ratio: {tech_ctx.get('bbm_ratio', 1):.4f}
  BBU Ratio: {tech_ctx.get('bbu_ratio', 1):.4f}
  High Vol Regime Flag: {tech_ctx.get('high_vol_regime', 0)}
  ATR Percentile Rank (vs 20d): {tech_ctx.get('atr_percentile_rank', 50):.1f}%
  Return Volatility (20d): {tech_ctx.get('return_vol_20', 0):.4f}

Produce a TechnicalSpecialistReport JSON following the schema exactly."""

    report, meta = await llm.call_analyst(
        "TechnicalSpecialist", _TECHNICAL_SPECIALIST_SYSTEM, user_prompt, TechnicalSpecialistReport
    )
    return report, meta


async def _run_quantitative_specialist(quant_ctx: dict) -> tuple[QuantitativeSpecialistReport, AgentCallMetadata]:
    user_prompt = f"""Current ML Quantitative Ensemble State:

QUANT ENSEMBLE OUTPUT (CatBoost + XGBoost + LightGBM):
  CatBoost P(up): {quant_ctx.get('p_cat', 0.5):.4f}
  XGBoost P(up):  {quant_ctx.get('p_xgb', 0.5):.4f}
  LightGBM P(up): {quant_ctx.get('p_lgb', 0.5):.4f}
  Ensemble P(up): {quant_ctx.get('prob_up', 0.5):.4f}
  Signal: {quant_ctx.get('quant_signal', 'HOLD')}
  Long Threshold: {quant_ctx.get('long_threshold', 0.65):.4f}
  Short Threshold: {quant_ctx.get('short_threshold', 0.35):.4f}

Produce a QuantitativeSpecialistReport JSON following the schema exactly."""

    report, meta = await llm.call_analyst(
        "QuantitativeSpecialist", _QUANTITATIVE_SPECIALIST_SYSTEM, user_prompt, QuantitativeSpecialistReport
    )
    return report, meta


async def _run_macro_svar_specialist() -> tuple[MacroSVARSpecialistReport, AgentCallMetadata]:
    macro_ctx = build_macro_context()
    user_prompt = f"""Current XAU/USD Macro SVAR Environment:

CRUDE OIL:
  Current Price: ${macro_ctx.get('oil_price_latest', 0):.2f}
  5-Day Return: {macro_ctx.get('oil_return_5d', 0):.2f}%
  1-Day Return: {macro_ctx.get('oil_return_1d', 0):.2f}%
  Shock Active (>3% in 5d): {macro_ctx.get('oil_shock_active', False)}
  Shock Direction: {macro_ctx.get('oil_shock_direction', 'NEUTRAL')}

VIX:
  Current Level: {macro_ctx.get('vix_latest', 0):.2f}
  5-Day Change: {macro_ctx.get('vix_change_5d', 0):.2f} points
  Direction: {macro_ctx.get('vix_direction', 'NEUTRAL')}
  Above Panic Threshold (>25): {macro_ctx.get('vix_above_panic', False)}

DXY (USD INDEX):
  Current Level: {macro_ctx.get('dxy_latest', 0):.2f}
  5-Day Change: {macro_ctx.get('dxy_change_5d', 0):.2f}
  Direction: {macro_ctx.get('dxy_direction', 'NEUTRAL')}

SHOCK DIGESTION WINDOW (~5 trading days):
  Status: {macro_ctx.get('digestion_window', 'NONE')}
  Days Since Last Significant Shock: {macro_ctx.get('days_since_shock', 'N/A')}

Produce a MacroSVARSpecialistReport JSON following the schema exactly."""

    report, meta = await llm.call_analyst(
        "MacroSVARSpecialist", _MACRO_SVAR_SPECIALIST_SYSTEM, user_prompt, MacroSVARSpecialistReport
    )
    return report, meta


async def _run_policy_fred_specialist() -> tuple[PolicyFREDSpecialistReport, AgentCallMetadata]:
    macro_ctx = build_macro_context()
    user_prompt = f"""Current FRED Macro Indicators:

FRED MACRO DATA:
  Fed Funds Rate: {macro_ctx.get('fed_funds_rate', 'N/A')}%
  US 10Y Yield: {macro_ctx.get('us_10y_yield', 'N/A')}%
  CPI: {macro_ctx.get('cpi_us', 'N/A')}
  Unemployment: {macro_ctx.get('unemployment_rate', 'N/A')}%
  Real GDP Growth: {macro_ctx.get('real_gdp_growth', 'N/A')}%

Produce a PolicyFREDSpecialistReport JSON following the schema exactly."""

    report, meta = await llm.call_analyst(
        "PolicyFREDSpecialist", _POLICY_FRED_SPECIALIST_SYSTEM, user_prompt, PolicyFREDSpecialistReport
    )
    return report, meta


async def _run_sentiment_specialist() -> tuple[SentimentSpecialistReport, AgentCallMetadata]:
    sent_ctx = build_sentiment_context()
    headline_list = "\n".join(f"  - {h}" for h in sent_ctx.get("headline_sample", [])[:5]) or "  (no recent headlines)"
    user_prompt = f"""Current News Sentiment:

SENTIMENT SCORES:
  FinBERT Mean Score: {sent_ctx.get('finbert_mean', 0):.4f}
  FinBERT Latest 5: {sent_ctx.get('finbert_latest_5', 0):.4f}
  VADER Live Score: {sent_ctx.get('vader_score', 0):.4f}
  BLENDED Score (70% FinBERT + 30% VADER): {sent_ctx.get('blended_score', 0):.4f}
  News Volume (24h): {sent_ctx.get('news_count', 0)}

RECENT HEADLINES:
{headline_list}

Produce a SentimentSpecialistReport JSON following the schema exactly."""

    report, meta = await llm.call_analyst(
        "SentimentSpecialist", _SENTIMENT_SPECIALIST_SYSTEM, user_prompt, SentimentSpecialistReport
    )
    return report, meta


async def _run_geopolitical_gdelt_specialist() -> tuple[GeopoliticalGDELTSpecialistReport, AgentCallMetadata]:
    sent_ctx = build_sentiment_context()
    gdelt_cats = sent_ctx.get("active_gdelt_categories", [])
    gdelt_str  = ", ".join(gdelt_cats) if gdelt_cats else "None detected"

    user_prompt = f"""Current Geopolitical GDELT State:

GDELT IMPACT SCORES:
  WAR_GEOPOLITICAL Impact: {sent_ctx.get('war_geo_impact', 0):.4f} (Count: {sent_ctx.get('war_geo_count', 0)})
  FED_POLICY Impact: {sent_ctx.get('fed_policy_impact', 0):.4f} (Count: {sent_ctx.get('fed_policy_count', 0)})
  INFLATION Impact: {sent_ctx.get('inflation_impact', 0):.4f} (Count: {sent_ctx.get('inflation_count', 0)})
  DOLLAR_FX Impact: {sent_ctx.get('dollar_fx_impact', 0):.4f}
  GOLD_MARKET Impact: {sent_ctx.get('gold_market_impact', 0):.4f}
  Geo Surge Score: {sent_ctx.get('geo_surge_score', 0):.4f}
  News Surprise Score: {sent_ctx.get('news_surprise', 0):.4f}

ACTIVE GDELT CATEGORIES (last 4 hours): {gdelt_str}

Produce a GeopoliticalGDELTSpecialistReport JSON following the schema exactly."""

    report, meta = await llm.call_analyst(
        "GeopoliticalGDELTSpecialist", _GEOPOLITICAL_GDELT_SPECIALIST_SYSTEM, user_prompt, GeopoliticalGDELTSpecialistReport
    )
    return report, meta


async def _run_calendar_specialist() -> tuple[CalendarSpecialistReport, AgentCallMetadata]:
    cal_ctx = build_calendar_context()
    def _fmt_events(evlist: list) -> str:
        if not evlist: return "  None"
        return "\n".join(f"  - {e['event']} [{e['impact']}] in {e.get('hours_until', 0.0):.1f}h" for e in evlist)

    user_prompt = f"""Current Calendar State:

EVENT WINDOW STATUS:
  Active (HIGH event within ±2h): {cal_ctx.get('event_window_active', False)}
  Volatility Flag: {cal_ctx.get('volatility_flag', 'CLEAR')}
  Next HIGH-Impact Event: {cal_ctx.get('next_high_impact_event', 'None')}

HIGH-IMPACT EVENTS WITHIN ±2 HOURS:
{_fmt_events(cal_ctx.get('events_within_2h', []))}

ALL TRACKED EVENTS WITHIN 24 HOURS:
{_fmt_events(cal_ctx.get('events_within_24h', []))}

RECOMMENDED CAUTION LEVEL: {cal_ctx.get('caution_level', 0)} / 3

Produce a CalendarSpecialistReport JSON following the schema exactly."""

    report, meta = await llm.call_analyst(
        "CalendarSpecialist", _CALENDAR_SPECIALIST_SYSTEM, user_prompt, CalendarSpecialistReport
    )
    return report, meta


# ══════════════════════════════════════════════════════════════════════════════
# PARALLEL TEAM RUNNER
# ══════════════════════════════════════════════════════════════════════════════

async def run_analyst_team(quant_context: dict) -> AnalystTeamResult:
    """Run all 7 specialist agents in parallel using asyncio.gather()."""
    log.info("[analyst_team] Running 7 specialists in parallel...")

    results = await asyncio.gather(
        _run_technical_specialist(),
        _run_quantitative_specialist(quant_context),
        _run_macro_svar_specialist(),
        _run_policy_fred_specialist(),
        _run_sentiment_specialist(),
        _run_geopolitical_gdelt_specialist(),
        _run_calendar_specialist(),
        return_exceptions=True,
    )

    reports = []
    metadata: list[AgentCallMetadata] = []
    names = [
        "TechnicalSpecialist", "QuantitativeSpecialist", "MacroSVARSpecialist",
        "PolicyFREDSpecialist", "SentimentSpecialist", "GeopoliticalGDELTSpecialist",
        "CalendarSpecialist"
    ]

    for i, (name, result) in enumerate(zip(names, results)):
        if isinstance(result, Exception):
            log.error("[analyst_team] %s FAILED: %s", name, result)
            raise RuntimeError(f"{name} failed: {result}") from result
        report, meta = result
        reports.append(report)
        metadata.append(meta)
        log.info("[analyst_team] %s complete (%dms)", name, meta.latency_ms)

    total_cost = sum(m.cost_usd_estimate for m in metadata)
    log.info("[analyst_team] All 7 specialists done. Total cost: $%.5f", total_cost)

    return AnalystTeamResult(
        technical=reports[0],
        quantitative=reports[1],
        macro_svar=reports[2],
        policy_fred=reports[3],
        sentiment=reports[4],
        geopolitical=reports[5],
        calendar=reports[6],
        metadata=metadata,
    )
