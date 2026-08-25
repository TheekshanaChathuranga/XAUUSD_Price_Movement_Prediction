"""
agents/__init__.py
==================
Upgraded 3-Tier Multi-Agent LLM Decision Layer for XAU/USD.

Architecture:
  Level 1: Analyst Specialists (7 agents, parallel)
  Level 2: Sector/Macro Synthesizers (2 agents, parallel)
  Level 3: Portfolio Manager & Decision Team (Trader, Risk, Fund Manager)

LLM Stack: Gemini-only
  - Tier 1 (Analysts/Synthesizers): gemini-2.0-flash / gemini-2.5-flash
  - Tier 2 (Reasoning): gemini-2.5-flash
  - Tier 3 (Decision):  gemini-2.5-pro
"""

from agents.schemas import (
    TechnicalSpecialistReport,
    QuantitativeSpecialistReport,
    MacroSVARSpecialistReport,
    PolicyFREDSpecialistReport,
    SentimentSpecialistReport,
    GeopoliticalGDELTSpecialistReport,
    CalendarSpecialistReport,
    TechnicalQuantSynthesis,
    MacroSentimentSynthesis,
    TraderProposal,
    RiskAssessment,
    FundManagerDecision,
    AgentSessionLog,
)

__all__ = [
    "TechnicalSpecialistReport",
    "QuantitativeSpecialistReport",
    "MacroSVARSpecialistReport",
    "PolicyFREDSpecialistReport",
    "SentimentSpecialistReport",
    "GeopoliticalGDELTSpecialistReport",
    "CalendarSpecialistReport",
    "TechnicalQuantSynthesis",
    "MacroSentimentSynthesis",
    "TraderProposal",
    "RiskAssessment",
    "FundManagerDecision",
    "AgentSessionLog",
]
