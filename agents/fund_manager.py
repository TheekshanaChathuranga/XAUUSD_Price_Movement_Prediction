"""
agents/fund_manager.py
======================
Portfolio Manager (Fund Manager) — Final Gate — Tier 3 (gemini-2.5-pro).

Reviews:
  - TechnicalQuantSynthesis
  - MacroSentimentSynthesis
  - TraderProposal
  - All three RiskAssessments + regime weights
  - Unique session ID and metadata
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Tuple

from agents.config import cfg
from agents.llm_client import llm
from agents.schemas import (
    AgentCallMetadata,
    FundManagerDecision,
    MarketRegime,
    RiskAssessment,
    RiskAgentType,
    TraderProposal,
)
from agents.synthesizer_team import SynthesizerTeamResult

log = logging.getLogger("agents.fund_manager")


_FUND_MANAGER_SYSTEM = """You are the Portfolio Manager (Fund Manager) — the final execution and decision gate for an XAU/USD (gold) trading system.
You review the Level 2 syntheses, the Trader's proposal, and the Risk Management team's weighted assessments, then make the final executive decision: APPROVE, REJECT, or RESIZE.

YOUR AUTHORITY:
  - APPROVE: Take the Trader's proposal at the proposed lot size.
  - REJECT:  Do not trade. Set final_direction=HOLD, final_lot_size=0.0.
  - RESIZE:  Trade in the proposed direction but adjust the lot size.
             Calculate final_lot_size = proposed_lot_size * weighted_size_factor (rounded to nearest 0.01).

WEIGHTED RISK SYNTHESIS:
  The three risk agents have weights based on the current regime.
  Compute: weighted_size = sum(agent.size_adjustment_factor * agent.regime_weight for all agents)
  If weighted_size < 0.4: lean toward REJECT or strong RESIZE.
  If 0.4–0.7: RESIZE to the weighted proportion.
  If > 0.7: APPROVE at full size (or weighted factor if slightly under 1.0).

REASONING CHAIN (always follow this order in full_reasoning):
  1. State the quant signal and ensemble probability.
  2. Summarise Level 2 Syntheses (technical-quant and macro-sentiment biases).
  3. Explain the risk team assessments and the weighted consensus factor.
  4. State your final decision (APPROVE/RESIZE/REJECT) and the final lot size.
  Keep under 1200 characters.

Output valid JSON matching the FundManagerDecision schema exactly."""


def _build_risk_consensus_summary(
    assessments: list[RiskAssessment],
    weights: Tuple[float, float, float],
) -> tuple[str, float]:
    risky_w, neutral_w, safe_w = weights
    weight_map = {
        RiskAgentType.RISKY:   risky_w,
        RiskAgentType.NEUTRAL: neutral_w,
        RiskAgentType.SAFE:    safe_w,
    }

    weighted_size = 0.0
    parts: list[str] = []

    for a in assessments:
        w = weight_map.get(a.agent_type, 1.0 / 3)
        weighted_size += a.size_adjustment_factor * w
        parts.append(
            f"{a.agent_type.value}(w={w:.2f}) → {a.recommendation.value} size={a.size_adjustment_factor:.2f}x"
        )

    summary = f"Weighted risk consensus: {' | '.join(parts)} → weighted_size_factor={weighted_size:.3f}"
    return summary, round(weighted_size, 3)


async def run_fund_manager(
    proposal: TraderProposal,
    risk_assessments: list[RiskAssessment],
    syntheses: SynthesizerTeamResult,
    regime: MarketRegime,
    session_id: str,
) -> tuple[FundManagerDecision, AgentCallMetadata]:
    """Run the Portfolio Manager (Fund Manager) decision agent."""
    tq = syntheses.tech_quant
    ms = syntheses.macro_sent

    caution_level = int(ms.event_risk_caution)
    weights = cfg.get_regime_weights(regime.value, caution_level)
    risk_consensus, weighted_size = _build_risk_consensus_summary(risk_assessments, weights)

    risk_summaries = "\n".join(
        f"  {a.agent_type.value} (weight={weights[i]:.2f}): {a.recommendation.value} size={a.size_adjustment_factor:.2f}x | Concerns: {a.key_concerns[:2]}"
        for i, a in enumerate(risk_assessments)
    )

    now_utc = datetime.now(timezone.utc).isoformat()

    user_prompt = f"""Make the final executive decision for this XAU/USD trade proposal.

TRADER PROPOSAL:
  Direction: {proposal.direction.value}
  Confidence: {proposal.confidence_score:.2f}
  SL: {proposal.proposed_sl_pips:.1f} pips | TP: {proposal.proposed_tp_pips:.1f} pips
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
  Drivers: {ms.key_macro_drivers}
  Sentiment: {ms.sentiment_alignment}
  Event Risk Caution: {ms.event_risk_caution:.1f}
  Summary: {ms.summary}

RISK MANAGEMENT ASSESSMENT ({regime.value} REGIME):
  Weights: Risky={weights[0]:.2f} | Neutral={weights[1]:.2f} | Safe={weights[2]:.2f}
{risk_summaries}
  
  Weighted Consensus: {risk_consensus}
  Weighted Size Factor: {weighted_size:.3f}

EXECUTION METADATA:
  Session ID: {session_id}
  Timestamp UTC: {now_utc}

Calculate the final decision.
If RESIZE: final_lot_size = {proposal.proposed_lot_size:.2f} * {weighted_size:.3f} = {proposal.proposed_lot_size * weighted_size:.4f} (round to nearest 0.01).
Output a complete FundManagerDecision JSON."""

    decision, meta = await llm.call_decision(
        "FundManager", _FUND_MANAGER_SYSTEM, user_prompt, FundManagerDecision
    )

    log.info(
        "[fund_manager] FINAL: %s | %s | lot=%.2f | regime=%s | shadow=%s",
        decision.final_decision.value,
        decision.final_direction.value,
        decision.final_lot_size,
        decision.current_regime.value,
        decision.shadow_mode,
    )

    return decision, meta
