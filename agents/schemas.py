"""
agents/schemas.py
=================
Structured communication protocol for the XAU/USD multi-agent LLM layer.

Design principle:
  Specialist reports, syntheses, and final decisions are STRUCTURED (Pydantic JSON schemas)
  so downstream agents query specific fields rather than parsing prose.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Optional, Any

from pydantic import BaseModel, Field, model_validator


# ══════════════════════════════════════════════════════════════════════════════
# SHARED ENUMERATIONS
# ══════════════════════════════════════════════════════════════════════════════

class Direction(str, Enum):
    """Primary directional bias."""
    BULLISH  = "BULLISH"
    BEARISH  = "BEARISH"
    NEUTRAL  = "NEUTRAL"


class SignalStrength(str, Enum):
    """Conviction level for a directional call."""
    STRONG   = "STRONG"
    MODERATE = "MODERATE"
    WEAK     = "WEAK"


class VolatilityRegime(str, Enum):
    """Market volatility classification."""
    LOW    = "LOW"
    NORMAL = "NORMAL"
    HIGH   = "HIGH"
    EXTREME = "EXTREME"


class MarketRegime(str, Enum):
    """Overall market regime used to weight the Risk Management team."""
    CALM     = "CALM"
    HIGH_VOL = "HIGH_VOL"
    PANIC    = "PANIC"


class GoldDriver(str, Enum):
    """Dominant macro driver per Chai et al. 2021 SVAR analysis."""
    OIL   = "OIL"
    VIX   = "VIX"
    DXY   = "DXY"
    MIXED = "MIXED"
    NONE  = "NONE"


class DigestionWindow(str, Enum):
    """Where we are in the ~5-trading-day shock digestion window (Chai et al. 2021)."""
    EARLY = "EARLY"
    MID   = "MID"
    LATE  = "LATE"
    NONE  = "NONE"


class VolatilityFlag(str, Enum):
    """Calendar event proximity flag."""
    IMMINENT   = "IMMINENT"    # HIGH-impact event within 2 hours
    APPROACHING = "APPROACHING" # HIGH-impact event within 24 hours
    CLEAR      = "CLEAR"       # No HIGH-impact event within 24 hours


class TradeDirection(str, Enum):
    """Final trade direction output."""
    LONG  = "LONG"
    SHORT = "SHORT"
    HOLD  = "HOLD"


class RiskAgentType(str, Enum):
    """Identity of each risk management agent."""
    RISKY   = "RISKY"
    SAFE    = "SAFE"
    NEUTRAL = "NEUTRAL"


class FinalDecision(str, Enum):
    """Fund Manager's gate decision."""
    APPROVE = "APPROVE"
    REJECT  = "REJECT"
    RESIZE  = "RESIZE"


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 1: SPECIALIST REPORTS (Tier 1 — gemini-2.0-flash)
# ══════════════════════════════════════════════════════════════════════════════

class TechnicalSpecialistReport(BaseModel):
    """Output of the Technical Specialist agent."""
    trend_direction: Direction = Field(description="Trend direction based on EMA stack and price action.")
    ema_alignment: str = Field(description="EMA stack state description.")
    momentum_state: str = Field(description="MACD momentum description.")
    macd_crossover: bool = Field(description="True if MACD crossover occurred in the last 3 bars.")
    volatility_regime: VolatilityRegime = Field(description="Current volatility regime.")
    bb_squeeze: bool = Field(description="True if Bollinger Bands are squeezing.")
    bb_position: str = Field(description="Price position within Bollinger Bands.")
    rsi_value: float = Field(ge=0.0, le=100.0, description="RSI(14) value.")
    rsi_overbought: bool = Field(description="True if RSI > 70.")
    rsi_oversold: bool = Field(description="True if RSI < 30.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Composite technical confidence score.")
    signal_conflicts: list[str] = Field(default_factory=list, description="Conflicts between indicators.")
    summary: str = Field(max_length=600, description="Factual technical summary under 500 characters.")


class QuantitativeSpecialistReport(BaseModel):
    """Output of the Quantitative Specialist agent."""
    quant_prob_up: float = Field(ge=0.0, le=1.0, description="Ensemble meta-learner probability of price going UP.")
    quant_signal: str = Field(description="Raw quant signal string (e.g. BUY, SELL, HOLD).")
    quant_consensus: bool = Field(description="True if at least 2/3 base models agree.")
    catboost_prob: float = Field(ge=0.0, le=1.0, description="CatBoost probability.")
    xgboost_prob: float = Field(ge=0.0, le=1.0, description="XGBoost probability.")
    lightgbm_prob: float = Field(ge=0.0, le=1.0, description="LightGBM probability.")
    long_threshold: float = Field(description="Minimum probability to take LONG.")
    short_threshold: float = Field(description="Maximum probability to take SHORT.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Quant model confidence score.")
    summary: str = Field(max_length=600, description="Summary of ML ensemble alignment under 500 characters.")


class MacroSVARSpecialistReport(BaseModel):
    """Output of the Macro/SVAR Specialist agent (Chai et al. 2021 SVAR grounded)."""
    macro_bias: Direction = Field(description="Macro directional bias based on oil/VIX/DXY.")
    bias_strength: SignalStrength = Field(description="Macro bias strength.")
    dominant_driver: GoldDriver = Field(description="Dominant driver per SVAR variance hierarchy (OIL > VIX > DXY).")
    oil_return_5d: float = Field(description="5-day crude oil return (%).")
    oil_shock_active: bool = Field(description="True if oil moved >3% in 5 days.")
    oil_shock_direction: Direction = Field(description="Oil shock direction.")
    vix_level: float = Field(description="Current VIX level.")
    vix_direction: Direction = Field(description="VIX trend direction.")
    vix_above_panic: bool = Field(description="True if VIX > 25.")
    dxy_level: float = Field(description="Current DXY level.")
    dxy_direction: Direction = Field(description="DXY trend direction.")
    dxy_impact_weight: str = Field(description="DXY weight reminder (LOW).")
    digestion_window_status: DigestionWindow = Field(description=" Digestion window status.")
    days_since_last_shock: Optional[int] = Field(default=None, description="Days since last shock.")
    chai_2021_narrative: str = Field(description="SVAR theory narrative mapping.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="SVAR macro confidence.")
    summary: str = Field(max_length=600, description="Macro SVAR summary under 500 characters.")


class PolicyFREDSpecialistReport(BaseModel):
    """Output of the Policy/FRED Specialist agent."""
    policy_bias: Direction = Field(description="Policy bias based on macroeconomic indicators.")
    fed_funds_rate: str = Field(description="Current Federal Funds Rate.")
    us_10y_yield: str = Field(description="Current US 10-Year yield.")
    cpi_us: str = Field(description="Latest CPI inflation reading.")
    unemployment_rate: str = Field(description="Latest unemployment rate.")
    real_gdp_growth: str = Field(description="Latest GDP growth.")
    fed_policy_stance: str = Field(description="Fed stance: HAWKISH, DOVISH, or NEUTRAL.")
    macro_risk_factor: str = Field(description="Primary slow-moving macroeconomic risk factor identified.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Policy analysis confidence.")
    summary: str = Field(max_length=600, description="FRED macro policy summary under 500 characters.")


class SentimentSpecialistReport(BaseModel):
    """Output of the News/Sentiment Specialist agent."""
    sentiment_direction: Direction = Field(description="Blended news sentiment direction.")
    sentiment_intensity: SignalStrength = Field(description="Sentiment intensity.")
    finbert_score: float = Field(ge=-1.0, le=1.0, description="Historical news polarity score.")
    vader_score: float = Field(ge=-1.0, le=1.0, description="Live news VADER compound score.")
    blended_score: float = Field(ge=-1.0, le=1.0, description="Blended sentiment score.")
    headline_sample: list[str] = Field(description="Sample headlines parsed.")
    news_volume_24h: int = Field(description="Headlines count in last 24h.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Sentiment confidence score.")
    summary: str = Field(max_length=600, description="Sentiment summary under 500 characters.")


class GeopoliticalGDELTSpecialistReport(BaseModel):
    """Output of the Geopolitical GDELT Specialist agent."""
    geopolitical_bias: Direction = Field(description="Geopolitical bias based on active events.")
    active_gdelt_categories: list[str] = Field(description="Active event categories.")
    high_impact_event_active: bool = Field(description="True if WAR_MILITARY or FED_POLICY active.")
    war_geopolitical_active: bool = Field(description="True if WAR_MILITARY/GEOPOLITICAL active.")
    war_geo_impact_score: float = Field(description="Geopolitical conflict intensity score.")
    fed_policy_impact_score: float = Field(description="Monetary policy event intensity score.")
    geo_surge_score: float = Field(description="Volume surge in geopolitical events.")
    news_surprise_score: float = Field(description="Deviations from news volume average.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Geopolitical confidence.")
    summary: str = Field(max_length=600, description="Geopolitical analysis summary under 500 characters.")


class CalendarEvent(BaseModel):
    """A single macro calendar event."""
    event:       str   = Field(description="Event name. e.g. 'US NFP (August)'")
    date:        str   = Field(description="Event date/time string. e.g. '2025-08-01 12:30 UTC'")
    impact:      str   = Field(description="Impact level: HIGH, MEDIUM, or LOW.")
    hours_until: float = Field(description="Hours until the event (negative = already passed).")


class CalendarSpecialistReport(BaseModel):
    """Output of the Calendar Specialist agent."""
    event_window_active: bool = Field(description="True if HIGH-impact event within ±2h.")
    volatility_flag: VolatilityFlag = Field(description="IMMINENT, APPROACHING, or CLEAR.")
    events_within_2h: list[CalendarEvent] = Field(default_factory=list, description="Events within 2h.")
    events_within_24h: list[CalendarEvent] = Field(default_factory=list, description="Events within 24h.")
    caution_level: int = Field(ge=0, le=3, description="Caution level (0-3).")
    next_high_impact_event: Optional[str] = Field(default=None, description="Next event and ETA.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Calendar caution confidence.")
    summary: str = Field(max_length=600, description="Calendar event caution summary under 500 characters.")


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 2: SYNTHESES (Tier 2 — gemini-2.5-flash)
# ══════════════════════════════════════════════════════════════════════════════

class TechnicalQuantSynthesis(BaseModel):
    """Level 2 Synthesis combining Technical Indicators and ML Models."""
    combined_bias: Direction = Field(description="Combined technical and quantitative bias.")
    synthesis_rationale: str = Field(description="Detailed rationale combining indicators + models.")
    confluence_indicators: list[str] = Field(description="Indicators aligning with the bias.")
    divergence_indicators: list[str] = Field(description="Indicators displaying divergence or conflicts.")
    overall_confidence: float = Field(ge=0.0, le=1.0, description="Synthesized confidence score.")
    summary: str = Field(max_length=800, description="Structured synthesis under 700 characters.")


class MacroSentimentSynthesis(BaseModel):
    """Level 2 Synthesis combining Macro Drivers, Calendar Risk, and Sentiment."""
    combined_bias: Direction = Field(description="Combined fundamental, sentiment, and news bias.")
    synthesis_rationale: str = Field(description="Detailed rationale combining macro SVAR, policy stance, headlines, and event risk.")
    key_macro_drivers: list[str] = Field(description="Dominant macro drivers and shock state.")
    sentiment_alignment: str = Field(description="Alignment description of blended sentiment vs macro trend.")
    event_risk_caution: float = Field(description="Synthesized caution score reflecting calendar events.")
    overall_confidence: float = Field(ge=0.0, le=1.0, description="Synthesized confidence score.")
    summary: str = Field(max_length=800, description="Structured synthesis under 700 characters.")


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 3: DECISION & EXECUTION (Tier 3 — gemini-2.5-pro)
# ══════════════════════════════════════════════════════════════════════════════

class TraderProposal(BaseModel):
    """Output of the Trader Agent (Level 3 Decision)."""
    direction: TradeDirection = Field(description="Proposed trade direction: LONG, SHORT, or HOLD.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Trader's composite confidence score.")
    proposed_sl_pips: float = Field(ge=0.0, description="Proposed stop-loss in pips.")
    proposed_tp_pips: float = Field(ge=0.0, description="Proposed take-profit in pips.")
    proposed_lot_size: float = Field(ge=0.0, description="Proposed lot size based on risk parameters.")
    implied_rr_ratio: float = Field(ge=0.0, description="Implied risk-to-reward ratio.")
    quant_signal_in: str = Field(description="Quant ensemble signal.")
    quant_prob_up: float = Field(ge=0.0, le=1.0, description="Quant probability UP.")
    quant_signal_override: bool = Field(description="True if overriding the quant signal.")
    override_rationale: Optional[str] = Field(default=None, description="Rationale for overriding quant signal.")
    synthesis_basis: list[str] = Field(description="Bullet points of Level 2 syntheses summarized.")
    rationale: str = Field(max_length=800, description="Detailed trading rationale under 700 characters.")


class RiskAssessment(BaseModel):
    """Output of one Risk Management agent (Risky / Safe / Neutral)."""
    agent_type: RiskAgentType = Field(description="Risk perspective: RISKY, SAFE, NEUTRAL.")
    regime_weight: float = Field(ge=0.0, le=1.0, description="Regime weight of this agent.")
    recommendation: TradeDirection = Field(description="Recommended action.")
    size_adjustment_factor: float = Field(ge=0.0, le=2.0, description="Lot size multiplier recommended.")
    key_concerns: list[str] = Field(default_factory=list, description="Primary risk concerns.")
    rationale: str = Field(max_length=800, description="Risk reasoning narrative under 700 characters.")


class FundManagerDecision(BaseModel):
    """Final executive decision from the Portfolio Manager."""
    final_decision: FinalDecision = Field(description="APPROVE, REJECT, or RESIZE.")
    final_direction: TradeDirection = Field(description="Final execution direction.")
    final_lot_size: float = Field(ge=0.0, description="Final lot size allocated.")
    final_sl_pips: float = Field(ge=0.0, description="Final stop-loss in pips.")
    final_tp_pips: float = Field(ge=0.0, description="Final take-profit in pips.")
    size_vs_proposal: str = Field(description="Sizing vs trader proposal description.")
    risk_team_consensus: str = Field(description="Summary of weighted risk team assessments.")
    current_regime: MarketRegime = Field(description="Market regime evaluated.")
    quant_signal_in: str = Field(description="Input quant signal.")
    quant_prob_up: float = Field(ge=0.0, le=1.0, description="Input quant probability.")
    agent_consensus: str = Field(description="Consensus summary of Level 1 and 2 layers.")
    full_reasoning: str = Field(max_length=1500, description="Reasoning chain under 1200 characters.")
    shadow_mode: bool = Field(description="True if shadow paper trading.")
    timestamp_utc: str = Field(description="Decision timestamp.")
    session_id: str = Field(description="UUID linking the session.")


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT TRAIL CONTAINER
# ══════════════════════════════════════════════════════════════════════════════

class AgentCallMetadata(BaseModel):
    """Token usage and latency metadata for a single LLM API call."""
    agent_name: str
    model_used: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    cost_usd_estimate: float = Field(ge=0.0)


class AgentSessionLog(BaseModel):
    """Complete audit trail for one decision cycle."""
    session_id: str
    timestamp_utc: str
    trigger_reason: str = Field(description="Why the agent layer activated.")

    # ── Specialist Reports (Level 1) ──────────────────────────────────────────
    technical_specialist:      TechnicalSpecialistReport
    quantitative_specialist:   QuantitativeSpecialistReport
    macro_svar_specialist:     MacroSVARSpecialistReport
    policy_fred_specialist:    PolicyFREDSpecialistReport
    sentiment_specialist:      SentimentSpecialistReport
    geopolitical_gdelt_specialist: GeopoliticalGDELTSpecialistReport
    calendar_specialist:       CalendarSpecialistReport

    # ── Syntheses (Level 2) ───────────────────────────────────────────────────
    technical_quant_synthesis:  TechnicalQuantSynthesis
    macro_sentiment_synthesis: MacroSentimentSynthesis

    # ── Trader Proposal ───────────────────────────────────────────────────────
    trader_proposal: TraderProposal

    # ── Risk Team ─────────────────────────────────────────────────────────────
    risk_assessments: list[RiskAssessment] = Field(description="All three risk agent assessments.")
    market_regime: MarketRegime

    # ── Final Decision ────────────────────────────────────────────────────────
    fund_manager_decision: FundManagerDecision

    # ── Performance Tracking (filled in post-hoc) ────────────────────────────
    actual_next_day_direction: Optional[str] = Field(default=None, description="Actual direction next session.")
    quant_baseline_correct: Optional[bool] = Field(default=None, description="Quant correct flag.")
    agent_decision_correct: Optional[bool] = Field(default=None, description="Agent decision correct flag.")

    # ── Cost & Performance ────────────────────────────────────────────────────
    api_calls: list[AgentCallMetadata] = Field(default_factory=list, description="Per-agent LLM call metadata.")
    total_cost_usd: float = Field(default=0.0, description="Total API cost (USD).")
    total_latency_ms: int = Field(default=0, description="Execution time (ms).")


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA EXPORT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

ALL_SCHEMAS: dict[str, type[BaseModel]] = {
    "TechnicalSpecialistReport": TechnicalSpecialistReport,
    "QuantitativeSpecialistReport": QuantitativeSpecialistReport,
    "MacroSVARSpecialistReport": MacroSVARSpecialistReport,
    "PolicyFREDSpecialistReport": PolicyFREDSpecialistReport,
    "SentimentSpecialistReport": SentimentSpecialistReport,
    "GeopoliticalGDELTSpecialistReport": GeopoliticalGDELTSpecialistReport,
    "CalendarSpecialistReport": CalendarSpecialistReport,
    "TechnicalQuantSynthesis": TechnicalQuantSynthesis,
    "MacroSentimentSynthesis": MacroSentimentSynthesis,
    "TraderProposal": TraderProposal,
    "RiskAssessment": RiskAssessment,
    "FundManagerDecision": FundManagerDecision,
    "AgentSessionLog": AgentSessionLog,
}


def export_all_schemas(output_path: Optional[str] = None) -> dict:
    """Export all agent JSON schemas as a single dict."""
    all_json = {
        name: cls.model_json_schema()
        for name, cls in ALL_SCHEMAS.items()
    }
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_json, f, indent=2)
        print(f"[schemas] Exported {len(all_json)} schemas -> {output_path}")
    return all_json
