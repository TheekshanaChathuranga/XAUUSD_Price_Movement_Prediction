"""Smoke test for agent layer imports and basic functionality."""
import sys
sys.path.insert(0, ".")

errors = []

# Test schemas
try:
    from agents.schemas import (
        TechnicalSpecialistReport, QuantitativeSpecialistReport,
        MacroSVARSpecialistReport, PolicyFREDSpecialistReport,
        SentimentSpecialistReport, GeopoliticalGDELTSpecialistReport,
        CalendarSpecialistReport, TechnicalQuantSynthesis,
        MacroSentimentSynthesis, TraderProposal, RiskAssessment,
        FundManagerDecision, AgentSessionLog,
    )
    print("[OK] schemas - all new models imported successfully")
except Exception as e:
    errors.append(f"[FAIL] schemas: {e}")

# Test config
try:
    from agents.config import cfg
    print(f"[OK] config: analyst={cfg.analyst_model}, decision={cfg.decision_model}")
    print(f"     shadow_mode={cfg.shadow_mode}, debate_rounds={cfg.debate_rounds}")
    w_calm  = cfg.get_regime_weights("CALM", 0)
    w_panic = cfg.get_regime_weights("PANIC", 3)
    print(f"[OK] CALM  weights: Risky={w_calm[0]:.2f} Neutral={w_calm[1]:.2f} Safe={w_calm[2]:.2f}")
    print(f"[OK] PANIC+caution=3: Risky={w_panic[0]:.2f} Neutral={w_panic[1]:.2f} Safe={w_panic[2]:.2f}")
except Exception as e:
    errors.append(f"[FAIL] config: {e}")

# Test context builder
try:
    from agents.context_builder import check_event_trigger, build_calendar_context
    triggered, reason = check_event_trigger()
    cal = build_calendar_context()
    print(f"[OK] context_builder: triggered={triggered}")
    print(f"     reason: {reason[:80]}")
    print(f"[OK] calendar: flag={cal['volatility_flag']}, caution={cal['caution_level']}")
    if cal.get("next_high_impact_event"):
        print(f"     next HIGH event: {cal['next_high_impact_event']}")
    print(f"     events_within_24h: {len(cal.get('events_within_24h', []))}")
except Exception as e:
    errors.append(f"[FAIL] context_builder: {e}")

# Test audit logger
try:
    from agents.audit_logger import get_latest_decision, get_recent_sessions
    latest = get_latest_decision()
    recent = get_recent_sessions(3)
    print(f"[OK] audit_logger: DB accessible, latest={latest}, sessions={len(recent)}")
except Exception as e:
    errors.append(f"[FAIL] audit_logger: {e}")

# Test shadow runner
try:
    from agents.shadow_runner import get_runner_status
    status = get_runner_status()
    print(f"[OK] shadow_runner: shadow_mode={status['shadow_mode']}, analyst={status['models']['analyst']}")
except Exception as e:
    errors.append(f"[FAIL] shadow_runner: {e}")

# Test LLM client import (does not make any API calls)
try:
    from agents.llm_client import llm
    print(f"[OK] llm_client: KimiClient instantiated (base_url={llm._get_client().base_url})")
except Exception as e:
    errors.append(f"[FAIL] llm_client: {e}")

# Test analyst team import
try:
    from agents.analyst_team import run_analyst_team
    print("[OK] analyst_team: imported")
except Exception as e:
    errors.append(f"[FAIL] analyst_team: {e}")

# Test synthesizer team import
try:
    from agents.synthesizer_team import run_synthesizer_team
    print("[OK] synthesizer_team: imported")
except Exception as e:
    errors.append(f"[FAIL] synthesizer_team: {e}")

# Test trader agent import
try:
    from agents.trader_agent import run_trader
    print("[OK] trader_agent: imported")
except Exception as e:
    errors.append(f"[FAIL] trader_agent: {e}")

# Test risk team import
try:
    from agents.risk_team import run_risk_team, detect_regime
    print("[OK] risk_team: imported")
except Exception as e:
    errors.append(f"[FAIL] risk_team: {e}")

# Test fund manager import
try:
    from agents.fund_manager import run_fund_manager
    print("[OK] fund_manager: imported")
except Exception as e:
    errors.append(f"[FAIL] fund_manager: {e}")

# Test orchestrator import
try:
    from agents.orchestrator import run_pipeline, check_and_run
    print("[OK] orchestrator: imported")
except Exception as e:
    errors.append(f"[FAIL] orchestrator: {e}")

# Summary
print()
print("=" * 50)
if errors:
    print(f"FAILED ({len(errors)} errors):")
    for err in errors:
        print(f"  {err}")
    sys.exit(1)
else:
    print("ALL IMPORTS OK - Agent layer ready")
    print("=" * 50)
