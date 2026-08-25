"""
agents/trader_agent.py
======================
Trader Agent — Tier 3 (gemini-2.5-pro).

Synthesises:
  - TechnicalQuantSynthesis
  - MacroSentimentSynthesis
  - Self-reflection lessons from reflection_memory.json
  - Current SL/TP/lot sizing constraints

Outputs: TraderProposal.
"""

from __future__ import annotations

import logging
import math

from agents.llm_client import llm
from agents.schemas import AgentCallMetadata, TraderProposal
from agents.synthesizer_team import SynthesizerTeamResult
from agents.reflection_agent import get_reflection_lessons_text

log = logging.getLogger("agents.trader_agent")

# Position sizing constants
RISK_USD          = 50.0
RR_RATIO          = 2.0
PIP_VALUE_PER_LOT = 1.0
MIN_LOT           = 0.01
LOT_INCREMENT     = 0.01
PIP_SIZE          = 0.01
MIN_PIPS          = 100.0
MAX_PIPS          = 500.0


def _compute_sl_tp_pips(atr: float, sl_mult: float = 1.5, tp_mult: float = 3.0) -> tuple[float, float]:
    if atr <= 0:
        atr = 5.0

    sl_price_dist = atr * sl_mult
    tp_price_dist = atr * tp_mult

    sl_pips = sl_price_dist / PIP_SIZE
    tp_pips = tp_price_dist / PIP_SIZE

    sl_pips = max(MIN_PIPS, min(MAX_PIPS, sl_pips))
    tp_pips = max(MIN_PIPS * RR_RATIO, min(MAX_PIPS * RR_RATIO, tp_pips))

    return round(sl_pips, 1), round(tp_pips, 1)


def _compute_lot_size(sl_pips: float) -> float:
    if sl_pips <= 0:
        return 0.0
    exact_lots = RISK_USD / (sl_pips * PIP_VALUE_PER_LOT)
    lot_size   = math.floor(exact_lots / LOT_INCREMENT) * LOT_INCREMENT
    return max(MIN_LOT, round(lot_size, 2))


_TRADER_SYSTEM = """You are the Trader Agent for an XAU/USD (gold) automated trading system.
You synthesise the Level 2 technical-quantitative and macro-sentiment syntheses and the reflection memory logs into a concrete trading proposal.

YOUR ROLE:
  - Base your proposal on the syntheses provided.
  - Review the reflection lessons. Do not repeat the same reasoning mistakes or enter trades in the same losing conditions.
  - Use the precomputed position parameters unless you have a specific reason to adjust them.
  - Output valid JSON matching the TraderProposal schema.

POSITION SIZING RULES:
  - Risk per trade: $50 USD
  - Minimum R:R ratio: 1:2
  - SL must be between 100 and 500 pips
  - TP must be between 200 and 1000 pips
  - lot_size = floor(50 / (sl_pips * 1.0) / 0.01) * 0.01 (minimum 0.01 lot)

CONFIDENCE SCORE (0-1):
  Start at the ensemble UP probability. Adjust:
    +0.05 if macro-sentiment bias matches technical-quant bias
    +0.05 if news blended sentiment is strong in the trade direction
    -0.05 if there are indicator divergences or conflicts
    -0.05 if calendar caution is high
  Cap at [0.1, 0.95].

Output valid JSON matching the TraderProposal schema exactly."""


async def run_trader(
    syntheses: SynthesizerTeamResult,
    quant_ctx: dict,
) -> tuple[TraderProposal, AgentCallMetadata]:
    """Run the Trader Agent using Level 2 syntheses and reflection memory."""
    tq = syntheses.tech_quant
    ms = syntheses.macro_sent

    # Pre-compute position parameters
    atr      = quant_ctx.get("atr", 5.0)
    sl_mult  = quant_ctx.get("sl_mult", 1.5)
    tp_mult  = quant_ctx.get("tp_mult", 3.0)
    sl_pips, tp_pips = _compute_sl_tp_pips(atr, sl_mult, tp_mult)
    lot_size = _compute_lot_size(sl_pips)
    rr_ratio = round(tp_pips / sl_pips, 2) if sl_pips > 0 else 0.0

    # Get reflection memory lessons
    lessons_text = get_reflection_lessons_text(limit=3)

    user_prompt = f"""Synthesise the following into a TraderProposal:

LEVEL 2 TECHNICAL-QUANT SYNTHESIS:
  Combined Bias: {tq.combined_bias.value}
  Confidence: {tq.overall_confidence:.2f}
  Confluence: {tq.confluence_indicators}
  Divergence: {tq.divergence_indicators}
  Rationale: {tq.synthesis_rationale}
  Summary: {tq.summary}

LEVEL 2 MACRO-SENTIMENT SYNTHESIS:
  Combined Bias: {ms.combined_bias.value}
  Confidence: {ms.overall_confidence:.2f}
  Macro Drivers: {ms.key_macro_drivers}
  Sentiment Alignment: {ms.sentiment_alignment}
  Event Risk Caution: {ms.event_risk_caution:.1f}
  Rationale: {ms.synthesis_rationale}
  Summary: {ms.summary}

QUANT ENSEMBLE INPUT DETAILS:
  Signal: {quant_ctx.get('quant_signal', 'HOLD')} | P(up): {quant_ctx.get('prob_up', 0.5):.4f}
  CatBoost: {quant_ctx.get('p_cat', 0.5):.4f} | XGBoost: {quant_ctx.get('p_xgb', 0.5):.4f} | LightGBM: {quant_ctx.get('p_lgb', 0.5):.4f}
  Long Threshold: {quant_ctx.get('long_threshold', 0.65):.4f} | Short Threshold: {quant_ctx.get('short_threshold', 0.35):.4f}

POSITION SIZING PARAMETERS:
  Entry Price: ${quant_ctx.get('entry_price', 0):.2f}
  Proposed SL: {sl_pips:.1f} pips | Proposed TP: {tp_pips:.1f} pips
  Proposed Lot Size: {lot_size:.2f} | Implied R:R: 1:{rr_ratio:.1f}

{lessons_text}

Produce a complete TraderProposal JSON. Follow your system rules exactly."""

    proposal, meta = await llm.call_decision(
        "TraderAgent", _TRADER_SYSTEM, user_prompt, TraderProposal
    )

    log.info(
        "[trader_agent] Proposal: %s (confidence=%.2f, override=%s, sl=%.1f, tp=%.1f, lot=%.2f)",
        proposal.direction.value,
        proposal.confidence_score,
        proposal.quant_signal_override,
        proposal.proposed_sl_pips,
        proposal.proposed_tp_pips,
        proposal.proposed_lot_size,
    )

    return proposal, meta
