"""
agents/risk_team.py
===================
Risk Management Team — Three perspectives — Tier 2 (gemini-2.5-flash).

Agents:
  - RISKY   Agent: argues for taking the trade at full proposed size
  - SAFE    Agent: argues for reducing size or skipping
  - NEUTRAL Agent: mediates between the two
"""

from __future__ import annotations

import asyncio
import logging
from typing import Tuple

from agents.config import cfg
from agents.llm_client import llm
from agents.schemas import (
    AgentCallMetadata,
    MarketRegime,
    RiskAgentType,
    RiskAssessment,
    TradeDirection,
    TraderProposal,
)
from agents.synthesizer_team import SynthesizerTeamResult

log = logging.getLogger("agents.risk_team")


def detect_regime(
    macro_svar: any,
    geopolitical: any,
    atr_percentile_rank: float = 50.0,
) -> MarketRegime:
    """
    Classify the current market regime for Risk team weighting.

    PANIC    → VIX above vix_panic_threshold OR active GDELT war event
    HIGH_VOL → ATR in top percentile of rolling window
    CALM     → default
    """
    vix_panic = getattr(macro_svar, "vix_above_panic", False)
    war_active = getattr(geopolitical, "war_geopolitical_active", False)

    if vix_panic or war_active:
        reason = []
        if vix_panic:    reason.append(f"VIX={macro_svar.vix_level:.1f} > {cfg.vix_panic_threshold}")
        if war_active:   reason.append("WAR_GEOPOLITICAL event active")
        log.info("[risk_team] Regime=PANIC (%s)", " + ".join(reason))
        return MarketRegime.PANIC

    if atr_percentile_rank >= cfg.atr_high_vol_percentile:
        log.info("[risk_team] Regime=HIGH_VOL (ATR at %.1f%%ile)", atr_percentile_rank)
        return MarketRegime.HIGH_VOL

    log.info("[risk_team] Regime=CALM")
    return MarketRegime.CALM


def _build_risk_system(agent_type: RiskAgentType, regime: MarketRegime, regime_weight: float) -> str:
    role_desc = {
        RiskAgentType.RISKY: (
            "You are the RISKY agent on the Risk Management team. "
            "Your role is to argue for taking the trade at full proposed size or larger. "
            "Focus on the opportunities: strong quant signals, solid technical stack, and high confidence."
        ),
        RiskAgentType.SAFE: (
            "You are the SAFE agent on the Risk Management team. "
            "Your role is to argue for reducing position size or skipping the trade. "
            "Focus on the risks: indicator conflicts, impending macro events, VIX panics, and high event_risk_caution."
        ),
        RiskAgentType.NEUTRAL: (
            "You are the NEUTRAL agent on the Risk Management team. "
            "Your role is to synthesise the Risky and Safe perspectives into a balanced size adjustment multiplier (0.5 to 1.0)."
        ),
    }[agent_type]

    return f"""{role_desc}

CURRENT REGIME: {regime.value}
YOUR WEIGHT IN FINAL DECISION: {regime_weight:.2f}

POSITION CONSTRAINTS:
  - Maximum risk per trade: $50 USD
  - size_adjustment_factor: 0.0=reject, 0.5=half size, 1.0=full, 1.5=aggressive (CALM only)

RULES:
1. key_concerns: List 1-3 specific concerns with numeric evidence.
2. Output valid JSON matching the RiskAssessment schema exactly."""


def _build_risk_user_prompt(
    proposal: TraderProposal,
    syntheses: SynthesizerTeamResult,
    regime: MarketRegime,
    weights: Tuple[float, float, float],
    agent_type: RiskAgentType,
) -> str:
    tq = syntheses.tech_quant
    ms = syntheses.macro_sent
    risky_w, neutral_w, safe_w = weights

    type_weight = {
        RiskAgentType.RISKY:   risky_w,
        RiskAgentType.NEUTRAL: neutral_w,
        RiskAgentType.SAFE:    safe_w,
    }[agent_type]

    return f"""TRADER PROPOSAL TO EVALUATE:
  Direction: {proposal.direction.value}
  Confidence: {proposal.confidence_score:.2f}
  Proposed SL: {proposal.proposed_sl_pips:.1f} pips | Proposed TP: {proposal.proposed_tp_pips:.1f} pips
  Lot Size: {proposal.proposed_lot_size:.2f} | R:R: 1:{proposal.implied_rr_ratio:.1f}
  Quant Signal: {proposal.quant_signal_in} (P_up={proposal.quant_prob_up:.4f})
  Rationale: {proposal.rationale[:300]}

LEVEL 2 TECHNICAL-QUANT SYNTHESIS:
  Bias: {tq.combined_bias.value} | Confidence: {tq.overall_confidence:.2f}
  Confluence: {tq.confluence_indicators}
  Divergence: {tq.divergence_indicators}
  Summary: {tq.summary}

LEVEL 2 MACRO-SENTIMENT SYNTHESIS:
  Bias: {ms.combined_bias.value} | Confidence: {ms.overall_confidence:.2f}
  Macro Drivers: {ms.key_macro_drivers}
  Sentiment Alignment: {ms.sentiment_alignment}
  Event Risk Caution: {ms.event_risk_caution:.1f}
  Summary: {ms.summary}

RISK CONTEXT:
  Market Regime: {regime.value}
  Your Weight: {type_weight:.2f}

Produce a RiskAssessment JSON."""


async def _run_risk_agent(
    agent_type: RiskAgentType,
    proposal: TraderProposal,
    syntheses: SynthesizerTeamResult,
    regime: MarketRegime,
    weights: Tuple[float, float, float],
) -> tuple[RiskAssessment, AgentCallMetadata]:
    risky_w, neutral_w, safe_w = weights
    agent_weight = {
        RiskAgentType.RISKY:   risky_w,
        RiskAgentType.NEUTRAL: neutral_w,
        RiskAgentType.SAFE:    safe_w,
    }[agent_type]

    system_prompt = _build_risk_system(agent_type, regime, agent_weight)
    user_prompt   = _build_risk_user_prompt(proposal, syntheses, regime, weights, agent_type)

    assessment, meta = await llm.call_reasoning_structured(
        f"Risk_{agent_type.value}", system_prompt, user_prompt, RiskAssessment
    )

    log.info(
        "[risk_team] %s: %s (size=%.2fx, weight=%.2f)",
        agent_type.value,
        assessment.recommendation.value,
        assessment.size_adjustment_factor,
        agent_weight,
    )

    return assessment, meta


async def run_risk_team(
    proposal: TraderProposal,
    syntheses: SynthesizerTeamResult,
    regime: MarketRegime,
    caution_level: int = 0,
) -> tuple[list[RiskAssessment], MarketRegime, list[AgentCallMetadata]]:
    """Run all three risk agents concurrently."""
    weights = cfg.get_regime_weights(regime.value, caution_level)
    risky_w, neutral_w, safe_w = weights

    log.info(
        "[risk_team] Running 3 risk agents (regime=%s, caution=%d) weights: Risky=%.2f Neutral=%.2f Safe=%.2f",
        regime.value, caution_level, risky_w, neutral_w, safe_w,
    )

    results = await asyncio.gather(
        _run_risk_agent(RiskAgentType.RISKY,   proposal, syntheses, regime, weights),
        _run_risk_agent(RiskAgentType.NEUTRAL,  proposal, syntheses, regime, weights),
        _run_risk_agent(RiskAgentType.SAFE,     proposal, syntheses, regime, weights),
        return_exceptions=True,
    )

    assessments: list[RiskAssessment] = []
    metadata:    list[AgentCallMetadata] = []

    agent_names = ["RISKY", "NEUTRAL", "SAFE"]
    for name, result in zip(agent_names, results):
        if isinstance(result, Exception):
            log.error("[risk_team] %s agent FAILED: %s", name, result)
            raise RuntimeError(f"Risk {name} agent failed: {result}") from result
        assessment, meta = result
        assessments.append(assessment)
        metadata.append(meta)

    return assessments, regime, metadata
