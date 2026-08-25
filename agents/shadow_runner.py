"""
agents/shadow_runner.py
=======================
Paper-trading / shadow-mode runner for the multi-agent LLM layer.

Runs as an APScheduler background job (every 5 minutes) alongside the
existing step11_api_server.py infrastructure. Checks the event trigger
condition and executes the full agent pipeline when active.

KEY SAFETY GUARANTEE:
  - In AGENT_SHADOW_MODE=true (default): decisions are logged to audit trail ONLY.
  - The Fund Manager's output does NOT trigger any real orders.
  - The existing quant ensemble, SL/TP logic, and position sizing are UNTOUCHED.
  - shadow_mode=false must be manually set in .env to enable live decisions.

Integration with step11_api_server.py:
  - The APScheduler instance from step11 calls shadow_runner_job() every 5 minutes.
  - Latest decision is available via /api/agent-decision endpoint.
  - Full session log via /api/agent-session-log/{session_id}.

Usage (from step11_api_server.py):
    from agents.shadow_runner import shadow_runner_job, get_runner_status
    scheduler.add_job(shadow_runner_job, 'interval', minutes=5, id='agent_shadow_runner')
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

from agents.audit_logger import get_latest_decision, get_recent_sessions
from agents.config import cfg
from agents.orchestrator import check_and_run_sync
from agents.schemas import AgentSessionLog

log = logging.getLogger("agents.shadow_runner")

# ── State ─────────────────────────────────────────────────────────────────────
_runner_lock    = threading.Lock()  # Prevent concurrent runs
_is_running     = False
_last_run_at:   str | None = None
_last_session:  AgentSessionLog | None = None
_last_error:    str | None = None
_run_count      = 0


def shadow_runner_job(quant_override: dict | None = None) -> None:
    """
    APScheduler job function — called every 5 minutes.

    Checks the event trigger condition and runs the full agent pipeline
    if active. Thread-safe: skips if a previous run is still in progress.

    Args:
        quant_override: Optional dict of quant ensemble values to inject.
                        If step11_api_server.py has just run inference,
                        pass the current {prob_up, p_cat, p_xgb, p_lgb, quant_signal}.
    """
    global _is_running, _last_run_at, _last_session, _last_error, _run_count

    if not _runner_lock.acquire(blocking=False):
        log.info("[shadow_runner] Previous run still in progress — skipping this tick.")
        return

    _is_running = True
    _last_run_at = datetime.now(timezone.utc).isoformat()

    try:
        log.info("[shadow_runner] Checking event trigger...")
        session = check_and_run_sync(quant_override=quant_override)

        if session is not None:
            _last_session = session
            _run_count   += 1
            _last_error   = None

            fm = session.fund_manager_decision
            mode_label = "SHADOW" if fm.shadow_mode else "LIVE"

            log.info(
                "[shadow_runner] [%s] Decision: %s | %s | lot=%.2f | cost=$%.5f | %dms",
                mode_label,
                fm.final_decision.value,
                fm.final_direction.value,
                fm.final_lot_size,
                session.total_cost_usd,
                session.total_latency_ms,
            )

            if not fm.shadow_mode:
                # LIVE MODE: this is where you would place the order.
                # Currently a safety stub — real order logic to be implemented here.
                log.warning(
                    "[shadow_runner] LIVE MODE ACTIVE — order would be placed: "
                    "%s %s lot=%.2f (NOT IMPLEMENTED — add broker API here)",
                    fm.final_direction.value,
                    fm.final_decision.value,
                    fm.final_lot_size,
                )
        else:
            log.info("[shadow_runner] No trigger condition — agent layer idle.")

    except Exception as e:
        _last_error = str(e)
        log.error("[shadow_runner] Pipeline failed: %s", e, exc_info=True)

    finally:
        _is_running = False
        _runner_lock.release()


def get_runner_status() -> dict:
    """
    Return current shadow runner status for the /api/agent-decision endpoint.
    """
    latest = get_latest_decision()

    return {
        "shadow_mode":        cfg.shadow_mode,
        "is_running":         _is_running,
        "last_run_at":        _last_run_at,
        "run_count":          _run_count,
        "last_error":         _last_error,
        "latest_decision":    latest,
        "event_window_hours": cfg.event_window_hours,
        "debate_rounds":      cfg.debate_rounds,
        "models": {
            "analyst":   cfg.analyst_model,
            "reasoning": cfg.reasoning_model,
            "decision":  cfg.decision_model,
        },
    }


def get_recent_decisions(limit: int = 10) -> list[dict]:
    """Return summary of the N most recent shadow decisions."""
    return get_recent_sessions(limit=limit)


def force_run(quant_override: dict | None = None) -> dict:
    """
    Force a pipeline run regardless of trigger condition.
    Used by the POST /api/agent-run endpoint for manual testing.

    Returns status dict with run result summary.
    """
    global _is_running, _last_run_at, _last_session, _last_error, _run_count

    if _is_running:
        return {"status": "busy", "message": "A pipeline run is already in progress."}

    if not _runner_lock.acquire(blocking=False):
        return {"status": "busy", "message": "Runner lock busy."}

    _is_running  = True
    _last_run_at = datetime.now(timezone.utc).isoformat()

    try:
        from agents.orchestrator import run_pipeline_sync
        session = run_pipeline_sync(
            quant_override=quant_override,
            trigger_reason="Manual force-run via /api/agent-run",
        )
        _last_session = session
        _run_count   += 1
        _last_error   = None

        fm = session.fund_manager_decision
        return {
            "status":         "success",
            "session_id":     session.session_id,
            "final_decision": fm.final_decision.value,
            "final_direction": fm.final_direction.value,
            "final_lot_size": fm.final_lot_size,
            "total_cost_usd": session.total_cost_usd,
            "total_latency_ms": session.total_latency_ms,
            "shadow_mode":    fm.shadow_mode,
        }
    except Exception as e:
        _last_error = str(e)
        log.error("[shadow_runner] force_run failed: %s", e, exc_info=True)
        return {"status": "error", "message": str(e)}
    finally:
        _is_running = False
        _runner_lock.release()
