"""
agents/reflection_agent.py
==========================
Self-Reflection Agent (Layer 3 — Performance Feedback Loop).

Acts as a continuous learning mechanism for the multi-agent system:
1. Inspects completed trade logs and audit database records.
2. When a trade results in a loss, it performs a diagnostic reflection ("Why did this fail?").
3. Saves structured reflection notes to `reflection_memory.json` in the project root.
4. Injects learned lessons into Bull/Bear Debater prompt contexts via get_reflection_lessons_text().

Designed to be FULLY SYNCHRONOUS so it integrates cleanly with both sync and async callers.
No async import chains — safe to import from debate_team, orchestrator, and api_server.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

from agents.config import cfg

log = logging.getLogger("agents.reflection_agent")

# Reflection memory stored in project root (same folder as all model files)
_MEMORY_FILE: Path = cfg.root_dir / "reflection_memory.json"

# ══════════════════════════════════════════════════════════════════════════════
# MEMORY PERSISTENCE (FILE-BACKED JSON)
# ══════════════════════════════════════════════════════════════════════════════

def load_reflection_memory(limit: int = 5) -> List[Dict[str, Any]]:
    """Load the most recent self-reflection entries from disk."""
    if not _MEMORY_FILE.exists():
        return []
    try:
        with open(_MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data[-limit:]
            return []
    except Exception as e:
        log.warning(f"Could not load reflection memory: {e}")
        return []


def save_reflection(reflection_entry: Dict[str, Any]) -> None:
    """Save a new self-reflection entry to reflection_memory.json."""
    existing: List[Dict[str, Any]] = []
    if _MEMORY_FILE.exists():
        try:
            with open(_MEMORY_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
        except Exception:
            existing = []

    existing.append(reflection_entry)
    # Keep at most 50 reflections
    existing = existing[-50:]
    try:
        with open(_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        log.info(f"Saved self-reflection for session '{reflection_entry.get('session_id', 'unknown')}'")
    except Exception as e:
        log.error(f"Failed to save reflection memory: {e}")


def get_reflection_lessons_text(limit: int = 3) -> str:
    """
    Format recent reflection entries into a concise text block.
    This is injected into Bull/Bear Debater prompts to prevent repeated mistakes.
    """
    reflections = load_reflection_memory(limit=limit)
    if not reflections:
        return "No past trade loss reflections recorded yet. Proceed with standard analysis."

    lines = [
        "",
        "=== SELF-REFLECTION LESSONS FROM PREVIOUS LOSING TRADES ===",
    ]
    for r in reflections:
        direction = r.get("direction", "HOLD")
        loss_bps  = r.get("loss_bps", 0)
        date_str  = r.get("date", "UNKNOWN")
        diagnosis = r.get("diagnosis", "N/A")
        lesson    = r.get("lesson", "N/A")
        lines.append(
            f"- [{date_str}] {direction} trade lost {loss_bps:.1f}bps\n"
            f"  Diagnosis: {diagnosis}\n"
            f"  Lesson: {lesson}"
        )
    lines.append("STRICT RULE: Do NOT repeat the reasoning fallacies noted above.\n")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT DB QUERY (SAFE — READ ONLY)
# ══════════════════════════════════════════════════════════════════════════════

def _get_session_memo(session_id: str) -> str:
    """Read fund manager justification from audit DB for a given session_id."""
    db_path = cfg.audit_db_path
    if not db_path.exists():
        return "No audit database found."
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT full_log_json FROM agent_sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()
            if not row:
                return "Session not found in audit database."
            log_data = json.loads(row[0])
            # Try to extract the fund manager's justification
            memo = (
                log_data.get("fund_manager_decision", {}).get("justification")
                or log_data.get("memo", "No justification captured.")
            )
            return str(memo)
    except Exception as e:
        log.warning(f"Could not read audit DB for session {session_id}: {e}")
        return "Audit read error."


# ══════════════════════════════════════════════════════════════════════════════
# RULE-BASED SELF-REFLECTION (API-COST FREE — No GPT call)
# ══════════════════════════════════════════════════════════════════════════════

_RULE_DIAGNOSES = [
    # (keyword_in_memo, diagnosis, lesson)
    ("sentiment", "Overweighted FinBERT/VADER sentiment on a day with low news volume — sentiment signal was noise.",
     "Only trust sentiment scores above 0.4 polarity when >10 headlines confirm the direction."),
    ("geopolit", "Geopolitical risk premium was priced in prior candle — the signal lagged the actual catalyst.",
     "Check if geopolitical event occurred >6h ago before trusting geo-risk as a fresh catalyst."),
    ("rsi", "RSI divergence was present on H1 but ignored — model entered on lagging price momentum.",
     "Confirm RSI is not in overbought (>70) or oversold (<30) territory before executing."),
    ("macd", "MACD histogram was already contracting before entry — momentum was fading at signal time.",
     "Require MACD histogram expansion in signal direction for at least 2 consecutive bars."),
    ("cpi", "Entry was taken within the 2-hour pre-CPI window — high-impact event caused reversal.",
     "Avoid entries within 2 hours of HIGH-impact macro events (CPI, NFP, FOMC)."),
    ("nfp", "NFP surprise overrode the technical signal — fundamental shift dominated.",
     "Reduce position size by 50% when NFP is scheduled within 24 hours."),
    ("dollar", "DXY correlation flipped unexpectedly — dollar strengthened despite bearish positioning.",
     "Verify DXY trend direction before entering LONG gold; a rising DXY requires strong confirmation."),
    ("vwap", "Price was trading below VWAP at entry — institutional order flow was bearish.",
     "Only enter LONG positions above VWAP; only enter SHORT positions below VWAP."),
]

def _rule_based_diagnose(direction: str, loss_bps: float, actual_move: str, memo: str) -> tuple[str, str]:
    """Perform rule-based diagnosis (no LLM, zero API cost)."""
    memo_lower = memo.lower()
    for keyword, diag, lesson in _RULE_DIAGNOSES:
        if keyword in memo_lower:
            return diag, lesson
    # Generic fallback
    diag   = (
        f"The {direction} trade lost {loss_bps:.1f}bps as gold moved {actual_move}. "
        f"The multi-agent consensus was overridden by an unanticipated market shift."
    )
    lesson = (
        "Enforce stricter multi-timeframe confirmation (1H + 4H aligned) "
        "and verify ATR is not in a compression regime before executing."
    )
    return diag, lesson


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — DIAGNOSE AND RECORD A LOSING TRADE
# ══════════════════════════════════════════════════════════════════════════════

def diagnose_losing_trade(
    session_id:         str,
    direction:          str,
    loss_bps:           float,
    actual_market_move: str,
    market_notes:       str = "",
    use_llm:            bool = False,
) -> Dict[str, Any]:
    """
    Diagnose why a trade resulted in a loss and record the lesson permanently.

    Args:
        session_id:          Audit session ID (e.g. "20260810-143022-a1b2")
        direction:           Trade direction that was taken: 'LONG' or 'SHORT'
        loss_bps:            Magnitude of loss in basis points
        actual_market_move:  Short description: e.g. "BEARISH (-1.2%)"
        market_notes:        Optional extra context (e.g. news headline that triggered move)
        use_llm:             If True, calls GPT to generate diagnosis. Default False (rule-based).

    Returns:
        The reflection entry dict that was saved.
    """
    memo = _get_session_memo(session_id)

    if use_llm:
        try:
            import openai
            client = openai.OpenAI(
                api_key=cfg.openai_api_key,
                base_url=cfg.openai_base_url,
            )
            prompt = f"""You are the Self-Reflection Agent for an XAU/USD Gold AI trading system.
A trade that was approved by the multi-agent system resulted in a LOSS. Analyze why.

TRADE DETAILS:
- Session ID: {session_id}
- Executed Signal: {direction}
- Loss Incurred: {loss_bps:.1f} bps
- Actual Market Outcome: Market moved {actual_market_move} instead.
- Fund Manager Justification: "{memo}"
- Market Context Notes: {market_notes}

TASK: Identify the core reasoning error and formulate a permanent lesson.

Return ONLY a JSON object with exactly these two keys:
{{
  "diagnosis": "2-sentence diagnosis of what specific reasoning failure occurred",
  "lesson": "1-sentence actionable rule to prevent this mistake in future debates"
}}"""

            response = client.chat.completions.create(
                model=cfg.decision_model,
                messages=[
                    {"role": "system", "content": "You are a strict quantitative trading auditor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            parsed = json.loads(raw)
            diagnosis = parsed.get("diagnosis", "")
            lesson    = parsed.get("lesson", "")
        except Exception as e:
            log.warning(f"LLM reflection failed ({e}), falling back to rule-based diagnosis.")
            diagnosis, lesson = _rule_based_diagnose(direction, loss_bps, actual_market_move, memo)
    else:
        diagnosis, lesson = _rule_based_diagnose(direction, loss_bps, actual_market_move, memo)

    entry = {
        "session_id":  session_id,
        "date":        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "direction":   direction,
        "loss_bps":    float(loss_bps),
        "actual_move": actual_market_move,
        "memo":        memo[:300],
        "diagnosis":   diagnosis,
        "lesson":      lesson,
    }

    save_reflection(entry)
    return entry
