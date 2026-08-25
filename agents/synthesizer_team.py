"""
agents/synthesizer_team.py
==========================
Level 2 Synthesizers — Tier 2 (gemini-2.5-flash).

Aggregates Level 1 specialist reports into two clean syntheses:
  1. TechnicalQuantSynthesizer  -> TechnicalQuantSynthesis
  2. MacroSentimentSynthesizer  -> MacroSentimentSynthesis

Standardises down-stream communication for the Level 3 PM node.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from agents.analyst_team import AnalystTeamResult
from agents.llm_client import llm
from agents.schemas import (
    AgentCallMetadata,
    TechnicalQuantSynthesis,
    MacroSentimentSynthesis,
)

log = logging.getLogger("agents.synthesizer_team")


@dataclass
class SynthesizerTeamResult:
    tech_quant: TechnicalQuantSynthesis
    macro_sent: MacroSentimentSynthesis
    metadata:   list[AgentCallMetadata]


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

_TECHNICAL_QUANT_SYNTH_SYSTEM = """You are the Technical & Quantitative Synthesizer for an XAU/USD (gold) trading system.
Your role is to review and merge the reports of the Technical Specialist and the Quantitative Specialist, resolve conflicts, and produce a structured TechnicalQuantSynthesis.

RULES:
1. combined_bias must be BULLISH/BEARISH/NEUTRAL based on indicator stacked stacks and the ML ensemble probability.
2. Resolve any divergence. For example, if technical indicators stack bearish but the ML ensemble probability is 0.72 (bullish LONG), explain this divergence and choose the side with the higher statistical edge.
3. Output valid JSON matching the schema exactly."""

_MACRO_SENTIMENT_SYNTH_SYSTEM = """You are the Macro & Sentiment Synthesizer for an XAU/USD (gold) trading system.
Your role is to review and merge the Macro SVAR, Policy/FRED, Sentiment, Geopolitical, and Calendar Specialist reports into a structured MacroSentimentSynthesis.

RULES:
1. combined_bias must reflect the dominant SVAR variance drivers (OIL > VIX > DXY) aligned with blended news polarity and geopolitical stress.
2. event_risk_caution should summarize the event proximity and potential calendar shocks.
3. Resolve conflicts, e.g. macro conditions are bullish (oil shock pointing up) but blended news sentiment is negative.
4. Output valid JSON matching the schema exactly."""


# ══════════════════════════════════════════════════════════════════════════════
# RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

async def _run_tech_quant_synthesizer(
    tech_rep: any, quant_rep: any
) -> tuple[TechnicalQuantSynthesis, AgentCallMetadata]:
    user_prompt = f"""Review these Level 1 specialist reports:

TECHNICAL SPECIALIST REPORT:
  Trend Direction: {tech_rep.trend_direction.value}
  EMA Alignment: {tech_rep.ema_alignment}
  Momentum State: {tech_rep.momentum_state}
  MACD Crossover: {tech_rep.macd_crossover}
  Volatility Regime: {tech_rep.volatility_regime.value}
  BB Squeeze: {tech_rep.bb_squeeze}
  BB Position: {tech_rep.bb_position}
  RSI Value: {tech_rep.rsi_value:.1f} (overbought={tech_rep.rsi_overbought}, oversold={tech_rep.rsi_oversold})
  Conflicts: {tech_rep.signal_conflicts}
  Summary: {tech_rep.summary}

QUANTITATIVE SPECIALIST REPORT:
  Ensemble Signal: {quant_rep.quant_signal}
  Ensemble P(up): {quant_rep.quant_prob_up:.4f}
  Consensus: {quant_rep.quant_consensus}
  CatBoost: {quant_rep.catboost_prob:.4f} | XGBoost: {quant_rep.xgboost_prob:.4f} | LightGBM: {quant_rep.lightgbm_prob:.4f}
  Summary: {quant_rep.summary}

Synthesise them into a TechnicalQuantSynthesis JSON."""

    report, meta = await llm.call_reasoning_structured(
        "TechnicalQuantSynthesizer", _TECHNICAL_QUANT_SYNTH_SYSTEM, user_prompt, TechnicalQuantSynthesis
    )
    return report, meta


async def _run_macro_sentiment_synthesizer(
    macro_svar: any, policy_fred: any, sentiment: any, geopolitical: any, calendar: any
) -> tuple[MacroSentimentSynthesis, AgentCallMetadata]:
    user_prompt = f"""Review these Level 1 specialist reports:

MACRO SVAR SPECIALIST REPORT:
  Bias: {macro_svar.macro_bias.value} ({macro_svar.bias_strength.value})
  Dominant Driver: {macro_svar.dominant_driver.value}
  Oil Return (5d): {macro_svar.oil_return_5d:.2f}% | Shock Active: {macro_svar.oil_shock_active}
  VIX Level: {macro_svar.vix_level:.1f} (above_panic={macro_svar.vix_above_panic})
  DXY Trend: {macro_svar.dxy_direction.value}
  Digestion Status: {macro_svar.digestion_window_status.value}
  Chai SVAR Narrative: {macro_svar.chai_2021_narrative}
  Summary: {macro_svar.summary}

POLICY FRED SPECIALIST REPORT:
  Fed Funds: {policy_fred.fed_funds_rate}% | US 10Y: {policy_fred.us_10y_yield}%
  CPI: {policy_fred.cpi_us} | GDP: {policy_fred.real_gdp_growth}
  Stance: {policy_fred.fed_policy_stance} | Risk: {policy_fred.macro_risk_factor}
  Summary: {policy_fred.summary}

SENTIMENT SPECIALIST REPORT:
  Bias: {sentiment.sentiment_direction.value} ({sentiment.sentiment_intensity.value})
  Scores: FinBERT={sentiment.finbert_score:.4f} | VADER={sentiment.vader_score:.4f} | Blended={sentiment.blended_score:.4f}
  Volume: {sentiment.news_volume_24h} articles
  Summary: {sentiment.summary}

GEOPOLITICAL GDELT SPECIALIST REPORT:
  Bias: {geopolitical.geopolitical_bias.value}
  Active Categories: {geopolitical.active_gdelt_categories}
  War Active: {geopolitical.war_geopolitical_active} | Policy Impact: {geopolitical.fed_policy_impact_score:.2f}
  Geo Surge: {geopolitical.geo_surge_score:.2f} | Surprise: {geopolitical.news_surprise_score:.2f}
  Summary: {geopolitical.summary}

CALENDAR SPECIALIST REPORT:
  Caution Level: {calendar.caution_level}/3 | Window Active: {calendar.event_window_active}
  Next economic event: {calendar.next_high_impact_event}
  Summary: {calendar.summary}

Synthesise them into a MacroSentimentSynthesis JSON."""

    report, meta = await llm.call_reasoning_structured(
        "MacroSentimentSynthesizer", _MACRO_SENTIMENT_SYNTH_SYSTEM, user_prompt, MacroSentimentSynthesis
    )
    return report, meta


# ══════════════════════════════════════════════════════════════════════════════
# TEAM RUNNER
# ══════════════════════════════════════════════════════════════════════════════

async def run_synthesizer_team(
    analysts: AnalystTeamResult,
) -> SynthesizerTeamResult:
    """Run Level 2 synthesizers in parallel using asyncio.gather()."""
    log.info("[synthesizer_team] Running 2 synthesizers in parallel...")

    results = await asyncio.gather(
        _run_tech_quant_synthesizer(analysts.technical, analysts.quantitative),
        _run_macro_sentiment_synthesizer(
            analysts.macro_svar, analysts.policy_fred, analysts.sentiment,
            analysts.geopolitical, analysts.calendar
        ),
        return_exceptions=True,
    )

    metadata: list[AgentCallMetadata] = []
    names = ["TechnicalQuantSynthesizer", "MacroSentimentSynthesizer"]
    syntheses = []

    for name, result in zip(names, results):
        if isinstance(result, Exception):
            log.error("[synthesizer_team] %s FAILED: %s", name, result)
            raise RuntimeError(f"{name} failed: {result}") from result
        synth, meta = result
        syntheses.append(synth)
        metadata.append(meta)
        log.info("[synthesizer_team] %s complete (%dms)", name, meta.latency_ms)

    return SynthesizerTeamResult(
        tech_quant=syntheses[0],
        macro_sent=syntheses[1],
        metadata=metadata,
    )
