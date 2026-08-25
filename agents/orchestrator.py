"""
agents/orchestrator.py
======================
Full pipeline runner — connects all 3 agent tiers into one call.

Pipeline:
  1. Build quant context (from existing ensemble output)
  2. Run Level 1 Specialists (7 parallel agents)
  3. Run Level 2 Synthesizers (2 parallel agents)
  4. Run Level 3 Trader Agent (proposes trade)
  5. Detect Market Regime
  6. Run Level 3 Risk Management Team (3 parallel agents)
  7. Run Level 3 Portfolio Manager (Fund Manager final gate)
  8. Assemble AgentSessionLog + write to audit database

Usage:
    from agents.orchestrator import run_pipeline, check_and_run
    session_log = await run_pipeline()
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone

from agents.analyst_team import run_analyst_team
from agents.synthesizer_team import run_synthesizer_team
from agents.trader_agent import run_trader
from agents.risk_team import detect_regime, run_risk_team
from agents.fund_manager import run_fund_manager
from agents.audit_logger import log_session
from agents.config import cfg
from agents.context_builder import (
    build_quant_context,
    build_technical_context,
    check_event_trigger,
)
from agents.schemas import AgentCallMetadata, AgentSessionLog, MarketRegime

log = logging.getLogger("agents.orchestrator")


async def run_pipeline(
    quant_override: dict | None = None,
    trigger_reason: str = "Manual run",
) -> AgentSessionLog:
    """Execute the upgraded 3-layer multi-agent trading pipeline."""
    session_id  = str(uuid.uuid4())
    timestamp   = datetime.now(timezone.utc).isoformat()
    all_metadata: list[AgentCallMetadata] = []
    wall_start  = time.monotonic()

    log.info("=" * 60)
    log.info("[orchestrator] SESSION START: %s", session_id)
    log.info("[orchestrator] Trigger: %s", trigger_reason)
    log.info("[orchestrator] Shadow mode: %s", cfg.shadow_mode)
    log.info("=" * 60)

    # ── Step 1: Build quant context ───────────────────────────────────────
    log.info("[orchestrator] [1/7] Building quant context...")
    quant_ctx = build_quant_context()
    if quant_override:
        quant_ctx.update(quant_override)
    log.info(
        "[orchestrator] Quant signal: %s (P_up=%.4f)",
        quant_ctx.get("quant_signal", "HOLD"),
        quant_ctx.get("prob_up", 0.5),
    )

    # ── Step 2: Run Level 1 Specialists (7 parallel) ─────────────────────
    log.info("[orchestrator] [2/7] Running 7 Specialist Analysts in parallel...")
    analyst_result = await run_analyst_team(quant_ctx)
    all_metadata.extend(analyst_result.metadata)

    # ── Step 3: Run Level 2 Synthesizers (2 parallel) ─────────────────────
    log.info("[orchestrator] [3/7] Running Level 2 Synthesizers in parallel...")
    synthesizer_result = await run_synthesizer_team(analyst_result)
    all_metadata.extend(synthesizer_result.metadata)
    log.info(
        "[orchestrator] Synthesizers complete. TechQuant=%s, MacroSent=%s",
        synthesizer_result.tech_quant.combined_bias.value,
        synthesizer_result.macro_sent.combined_bias.value,
    )

    # ── Step 4: Run Level 3 Trader Agent ──────────────────────────────────
    log.info("[orchestrator] [4/7] Running Level 3 Trader Agent...")
    trader_proposal, trader_meta = await run_trader(synthesizer_result, quant_ctx)
    all_metadata.append(trader_meta)
    log.info(
        "[orchestrator] Trader proposal: %s (confidence=%.2f)",
        trader_proposal.direction.value,
        trader_proposal.confidence_score,
    )

    # ── Step 5: Detect Market Regime ──────────────────────────────────────
    log.info("[orchestrator] [5/7] Detecting market regime...")
    tech_ctx    = build_technical_context()
    atr_pct_rank = tech_ctx.get("atr_percentile_rank", 50.0)
    
    regime = detect_regime(
        macro_svar=analyst_result.macro_svar,
        geopolitical=analyst_result.geopolitical,
        atr_percentile_rank=atr_pct_rank,
    )
    caution_level = analyst_result.calendar.caution_level
    log.info("[orchestrator] Regime=%s, caution=%d", regime.value, caution_level)

    # ── Step 6: Run Level 3 Risk Management (3 parallel) ──────────────────
    log.info("[orchestrator] [6/7] Running Risk Management assessments...")
    risk_assessments, regime_confirmed, risk_meta = await run_risk_team(
        proposal=trader_proposal,
        syntheses=synthesizer_result,
        regime=regime,
        caution_level=caution_level,
    )
    all_metadata.extend(risk_meta)

    # ── Step 7: Run Level 3 Portfolio Manager (final gate) ────────────────
    log.info("[orchestrator] [7/7] Running Portfolio Manager...")
    fm_decision, fm_meta = await run_fund_manager(
        proposal=trader_proposal,
        risk_assessments=risk_assessments,
        syntheses=synthesizer_result,
        regime=regime_confirmed,
        session_id=session_id,
    )
    all_metadata.append(fm_meta)
    log.info(
        "[orchestrator] PM final: %s | %s | lot=%.2f",
        fm_decision.final_decision.value,
        fm_decision.final_direction.value,
        fm_decision.final_lot_size,
    )

    # ── Step 8: Assemble session log & save ───────────────────────────────
    wall_ms      = int((time.monotonic() - wall_start) * 1000)
    total_cost   = sum(m.cost_usd_estimate for m in all_metadata)

    session_log = AgentSessionLog(
        session_id=session_id,
        timestamp_utc=timestamp,
        trigger_reason=trigger_reason,
        technical_specialist=analyst_result.technical,
        quantitative_specialist=analyst_result.quantitative,
        macro_svar_specialist=analyst_result.macro_svar,
        policy_fred_specialist=analyst_result.policy_fred,
        sentiment_specialist=analyst_result.sentiment,
        geopolitical_gdelt_specialist=analyst_result.geopolitical,
        calendar_specialist=analyst_result.calendar,
        technical_quant_synthesis=synthesizer_result.tech_quant,
        macro_sentiment_synthesis=synthesizer_result.macro_sent,
        trader_proposal=trader_proposal,
        risk_assessments=risk_assessments,
        market_regime=regime_confirmed,
        fund_manager_decision=fm_decision,
        api_calls=all_metadata,
        total_cost_usd=round(total_cost, 6),
        total_latency_ms=wall_ms,
    )

    log_session(session_log)

    log.info("=" * 60)
    log.info("[orchestrator] SESSION COMPLETE: %s", session_id)
    log.info("=" * 60)

    return session_log


async def check_and_run(quant_override: dict | None = None) -> AgentSessionLog | None:
    """Check trigger condition and run pipeline if active."""
    triggered, reason = check_event_trigger()

    if not triggered:
        log.debug("[orchestrator] No trigger condition — agent layer inactive.")
        return None

    log.info("[orchestrator] TRIGGER ACTIVE: %s", reason)
    return await run_pipeline(quant_override=quant_override, trigger_reason=reason)


def run_pipeline_sync(
    quant_override: dict | None = None,
    trigger_reason: str = "Manual run",
) -> AgentSessionLog:
    """Synchronous wrapper for run_pipeline()."""
    return asyncio.run(run_pipeline(quant_override=quant_override, trigger_reason=trigger_reason))


def check_and_run_sync(quant_override: dict | None = None) -> AgentSessionLog | None:
    """Synchronous wrapper for check_and_run()."""
    return asyncio.run(check_and_run(quant_override=quant_override))
