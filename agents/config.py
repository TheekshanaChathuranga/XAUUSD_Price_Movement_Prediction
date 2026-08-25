"""
agents/config.py
================
Central configuration for the XAU/USD multi-agent LLM layer.
Reads all settings from .env — no hardcoded values.

Usage:
    from agents.config import cfg
    model = cfg.analyst_model
    weights = cfg.get_regime_weights(MarketRegime.PANIC, caution_level=2)

LLM Backend: OpenAI GPT-4o-mini via official OpenAI API endpoint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

# ── Load .env from project root ───────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=False)


# ── OpenAI pricing constants (USD per 1M tokens) ─────────────────────────
# Source: platform.openai.com/docs/pricing (2025)
_PRICING = {
    "gpt-4o-mini":      {"input": 0.15, "output": 0.60},   # $0.15/$0.60 per 1M tokens
    "gpt-4o":           {"input": 2.50, "output": 10.00},  # $2.50/$10.00 per 1M tokens
    "gpt-4-turbo":      {"input": 10.0, "output": 30.00},  # $10/$30 per 1M tokens
    # Fallback
    "default":          {"input": 0.15, "output": 0.60},
}


@dataclass
class AgentConfig:
    """All runtime configuration for the agent layer. Loaded once at import time."""

    # ── API Keys & Endpoint ───────────────────────────────────────────────────
    openai_api_key:  str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))

    # ── Model Routing ───────────────────────────────────────────────────
    analyst_model:   str = field(default_factory=lambda: os.getenv("ANALYST_MODEL",   "gpt-4o-mini"))
    reasoning_model: str = field(default_factory=lambda: os.getenv("REASONING_MODEL", "gpt-4o-mini"))
    decision_model:  str = field(default_factory=lambda: os.getenv("DECISION_MODEL",  "gpt-4o-mini"))

    # ── Agent Behaviour ───────────────────────────────────────────────────────
    debate_rounds: int = field(
        default_factory=lambda: int(os.getenv("DEBATE_ROUNDS", "2"))
    )
    shadow_mode: bool = field(
        default_factory=lambda: os.getenv("AGENT_SHADOW_MODE", "true").lower() == "true"
    )
    event_window_hours: float = field(
        default_factory=lambda: float(os.getenv("EVENT_WINDOW_HOURS", "2.0"))
    )
    gdelt_trigger_lookback_hours: float = field(
        default_factory=lambda: float(os.getenv("GDELT_TRIGGER_LOOKBACK_HOURS", "4.0"))
    )

    # ── Regime Detection ──────────────────────────────────────────────────────
    vix_panic_threshold: float = field(
        default_factory=lambda: float(os.getenv("VIX_PANIC_THRESHOLD", "25.0"))
    )
    atr_high_vol_percentile: float = field(
        default_factory=lambda: float(os.getenv("ATR_HIGH_VOL_PERCENTILE", "85.0"))
    )

    # ── Raw regime weight strings (parsed on demand) ──────────────────────────
    _regime_calm_str:     str = field(default_factory=lambda: os.getenv("REGIME_WEIGHTS_CALM",     "0.5,0.3,0.2"))
    _regime_high_vol_str: str = field(default_factory=lambda: os.getenv("REGIME_WEIGHTS_HIGH_VOL", "0.3,0.4,0.3"))
    _regime_panic_str:    str = field(default_factory=lambda: os.getenv("REGIME_WEIGHTS_PANIC",    "0.15,0.3,0.55"))

    # ── File Paths ────────────────────────────────────────────────────────────
    root_dir:             Path = field(default_factory=lambda: _ROOT)
    inference_data_path:  Path = field(default_factory=lambda: _ROOT / "live_inference_data.csv")
    raw_prices_path:      Path = field(default_factory=lambda: _ROOT / "xauusd_raw_prices.csv")
    fred_macro_path:      Path = field(default_factory=lambda: _ROOT / "fred_macro_raw.csv")
    gdelt_path:           Path = field(default_factory=lambda: _ROOT / "gdelt_news_raw.csv")
    sentiment_cache_path: Path = field(default_factory=lambda: _ROOT / "news_sentiment_cache.csv")
    test_preds_path:      Path = field(default_factory=lambda: _ROOT / "test_predictions.csv")
    audit_db_path:        Path = field(default_factory=lambda: _ROOT / "agents_audit.db")
    audit_log_dir:        Path = field(default_factory=lambda: _ROOT / "agents_log")

    def _parse_weights(self, s: str) -> Tuple[float, float, float]:
        """Parse 'risky,neutral,safe' string into float tuple."""
        parts = [float(x.strip()) for x in s.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Regime weights must have 3 values, got: {s!r}")
        total = sum(parts)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Regime weights must sum to 1.0, got {total:.3f}: {s!r}")
        return (parts[0], parts[1], parts[2])

    def get_regime_weights(self, regime: str, caution_level: int = 0) -> Tuple[float, float, float]:
        """
        Return (risky_weight, neutral_weight, safe_weight) for a given regime.

        caution_level (0-3) from CalendarAnalystReport shifts weight toward Safe:
          Each +1 caution reduces Risky by 0.05 and increases Safe by 0.05.

        Args:
            regime: MarketRegime enum value (as string: 'CALM', 'HIGH_VOL', 'PANIC')
            caution_level: 0-3 from calendar analyst

        Returns:
            Tuple of (risky_weight, neutral_weight, safe_weight) summing to 1.0
        """
        base_map = {
            "CALM":     self._parse_weights(self._regime_calm_str),
            "HIGH_VOL": self._parse_weights(self._regime_high_vol_str),
            "PANIC":    self._parse_weights(self._regime_panic_str),
        }
        risky, neutral, safe = base_map.get(regime, base_map["CALM"])

        # Apply caution level shift (each +1 caution = shift 0.05 from Risky to Safe)
        caution_shift = min(caution_level, 3) * 0.05
        risky  = max(0.0, risky  - caution_shift)
        safe   = min(1.0, safe   + caution_shift)

        # Renormalize to ensure sum = 1.0
        total = risky + neutral + safe
        return (round(risky/total, 4), round(neutral/total, 4), round(safe/total, 4))

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate API cost in USD for a single call."""
        pricing = _PRICING.get(model, _PRICING["default"])
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

    def validate(self) -> None:
        """Raise if critical config is missing."""
        if not self.openai_api_key or self.openai_api_key == "YOUR_OPENAI_API_KEY_HERE":
            raise ValueError(
                "OPENAI_API_KEY not set in .env. "
                "Set OPENAI_API_KEY=sk-... from your OpenAI account at platform.openai.com."
            )

    def __post_init__(self) -> None:
        # Ensure audit log directory exists
        self.audit_log_dir.mkdir(exist_ok=True)


# ── Singleton instance ────────────────────────────────────────────────────────
cfg = AgentConfig()


if __name__ == "__main__":
    try:
        cfg.validate()
        print("[config] OPENAI_API_KEY: set")
        print(f"[config] OPENAI_BASE_URL: {cfg.openai_base_url}")
    except ValueError as e:
        print(f"[config] WARNING: {e}")

    print(f"[config] Analyst model   : {cfg.analyst_model}")
    print(f"[config] Reasoning model : {cfg.reasoning_model}")
    print(f"[config] Decision model  : {cfg.decision_model}")
    print(f"[config] Shadow mode     : {cfg.shadow_mode}")
    print(f"[config] Debate rounds   : {cfg.debate_rounds}")
    print(f"[config] Event window    : +/-{cfg.event_window_hours}h")

    for regime in ["CALM", "HIGH_VOL", "PANIC"]:
        for caution in [0, 2]:
            w = cfg.get_regime_weights(regime, caution)
            print(f"[config] Regime={regime:<8s} caution={caution}  "
                  f"Risky={w[0]:.2f} Neutral={w[1]:.2f} Safe={w[2]:.2f}")
