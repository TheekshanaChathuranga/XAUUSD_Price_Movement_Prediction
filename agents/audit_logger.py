"""
agents/audit_logger.py
======================
Complete audit trail for every agent decision cycle.

Writes to:
  1. SQLite DB (agents_audit.db) — indexed, queryable, persistent
  2. JSONL files (agents_log/YYYY-MM-DD_HH-MM_<session_id>.jsonl) — human-readable

Schema (agents_audit.db table: agent_sessions):
  session_id      TEXT PRIMARY KEY
  timestamp_utc   TEXT
  trigger_reason  TEXT
  final_decision  TEXT  (APPROVE / REJECT / RESIZE)
  final_direction TEXT  (LONG / SHORT / HOLD)
  final_lot_size  REAL
  quant_signal_in TEXT
  quant_prob_up   REAL
  market_regime   TEXT
  shadow_mode     INTEGER (1=True)
  total_cost_usd  REAL
  total_latency_ms INTEGER
  full_log_json   TEXT  (complete AgentSessionLog as JSON)

Usage:
    from agents.audit_logger import log_session, get_latest_decision, get_session_log
    await log_session(session_log)
    decision = get_latest_decision()
    full_log  = get_session_log(session_id)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from agents.config import cfg
from agents.schemas import AgentSessionLog

log = logging.getLogger("agents.audit_logger")

_DB_PATH  = cfg.audit_db_path
_LOG_DIR  = cfg.audit_log_dir

# ── Ensure directories exist ──────────────────────────────────────────────────
_LOG_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════════════

def _init_db() -> None:
    """Create the agents_audit.db table if it doesn't exist."""
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_sessions (
                session_id       TEXT PRIMARY KEY,
                timestamp_utc    TEXT NOT NULL,
                trigger_reason   TEXT,
                final_decision   TEXT,
                final_direction  TEXT,
                final_lot_size   REAL,
                quant_signal_in  TEXT,
                quant_prob_up    REAL,
                market_regime    TEXT,
                shadow_mode      INTEGER,
                total_cost_usd   REAL,
                total_latency_ms INTEGER,
                full_log_json    TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON agent_sessions (timestamp_utc DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_decision
            ON agent_sessions (final_decision)
        """)
        conn.commit()


# Initialize on import
_init_db()


# ══════════════════════════════════════════════════════════════════════════════
# WRITE
# ══════════════════════════════════════════════════════════════════════════════

def log_session(session: AgentSessionLog) -> None:
    """
    Persist a completed AgentSessionLog to:
      1. SQLite DB (fast query / summary)
      2. JSONL file (full human-readable log)

    This is synchronous — call from the orchestrator after the pipeline completes.
    """
    fm  = session.fund_manager_decision
    log_json = session.model_dump_json(indent=2)

    # ── SQLite insert ─────────────────────────────────────────────────────
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO agent_sessions VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                session.session_id,
                session.timestamp_utc,
                session.trigger_reason,
                fm.final_decision.value,
                fm.final_direction.value,
                fm.final_lot_size,
                fm.quant_signal_in,
                fm.quant_prob_up,
                session.market_regime.value,
                int(fm.shadow_mode),
                session.total_cost_usd,
                session.total_latency_ms,
                log_json,
            ))
            conn.commit()
        log.info("[audit] Session %s saved to DB", session.session_id)
    except Exception as e:
        log.error("[audit] DB write failed for session %s: %s", session.session_id, e)

    # ── JSONL file write ──────────────────────────────────────────────────
    try:
        ts = session.timestamp_utc[:16].replace(":", "-").replace("T", "_")
        filename = f"{ts}_{session.session_id[:8]}.jsonl"
        filepath = _LOG_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            # One JSON object per agent call for easy grep/filtering
            f.write(f"// SESSION: {session.session_id}\n")
            f.write(f"// TRIGGER: {session.trigger_reason}\n")
            f.write(f"// FINAL: {fm.final_decision.value} | {fm.final_direction.value} | lot={fm.final_lot_size:.2f}\n")
            f.write(f"// COST: ${session.total_cost_usd:.5f} | LATENCY: {session.total_latency_ms}ms\n\n")

            # Write full session log as single JSON object
            f.write(log_json)

        log.info("[audit] Full log written: %s", filepath)
    except Exception as e:
        log.error("[audit] JSONL write failed for session %s: %s", session.session_id, e)


# ══════════════════════════════════════════════════════════════════════════════
# READ
# ══════════════════════════════════════════════════════════════════════════════

def get_latest_decision() -> dict | None:
    """
    Return the most recent Fund Manager decision as a plain dict.
    Used by the /api/agent-decision endpoint.
    Returns None if no decisions logged yet.
    """
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT session_id, timestamp_utc, trigger_reason,
                       final_decision, final_direction, final_lot_size,
                       quant_signal_in, quant_prob_up, market_regime,
                       shadow_mode, total_cost_usd, total_latency_ms
                FROM agent_sessions
                ORDER BY timestamp_utc DESC
                LIMIT 1
            """).fetchone()
        if row:
            return dict(row)
        return None
    except Exception as e:
        log.error("[audit] get_latest_decision failed: %s", e)
        return None


def get_session_log(session_id: str) -> dict | None:
    """
    Return the full AgentSessionLog JSON for a given session_id.
    Used by the /api/agent-session-log/{session_id} endpoint.
    """
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            row = conn.execute(
                "SELECT full_log_json FROM agent_sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()
        if row and row[0]:
            return json.loads(row[0])
        return None
    except Exception as e:
        log.error("[audit] get_session_log(%s) failed: %s", session_id, e)
        return None


def get_recent_sessions(limit: int = 10) -> list[dict]:
    """
    Return summary rows for the N most recent sessions (no full JSON).
    Used for dashboard display.
    """
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT session_id, timestamp_utc, trigger_reason,
                       final_decision, final_direction, final_lot_size,
                       quant_signal_in, quant_prob_up, market_regime,
                       shadow_mode, total_cost_usd, total_latency_ms
                FROM agent_sessions
                ORDER BY timestamp_utc DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        log.error("[audit] get_recent_sessions failed: %s", e)
        return []


def update_session_outcome(
    session_id: str,
    actual_direction: str,
    quant_correct: bool | None,
    agent_correct: bool | None,
) -> None:
    """
    Post-hoc update to fill in actual_next_day_direction and correctness flags
    for shadow-mode performance tracking.

    Called manually (or by a daily job) after the trading session closes.
    """
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            row = conn.execute(
                "SELECT full_log_json FROM agent_sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()
            if not row or not row[0]:
                log.warning("[audit] Session %s not found for outcome update", session_id)
                return

            log_data = json.loads(row[0])
            log_data["actual_next_day_direction"] = actual_direction
            log_data["quant_baseline_correct"]    = quant_correct
            log_data["agent_decision_correct"]    = agent_correct

            conn.execute(
                "UPDATE agent_sessions SET full_log_json = ? WHERE session_id = ?",
                (json.dumps(log_data, indent=2), session_id)
            )
            conn.commit()
        log.info("[audit] Outcome updated for session %s: actual=%s", session_id, actual_direction)
    except Exception as e:
        log.error("[audit] update_session_outcome(%s) failed: %s", session_id, e)
