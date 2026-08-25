"""
end_to_end_test.py
==================
Full system verification for Kimi K3 integration.

Runs 5 levels of checks:
  L1 - Config & credentials loaded correctly
  L2 - Connectivity: live ping to Kimi API endpoint
  L3 - Unstructured call (plain text from Tier 2 reasoning model)
  L4 - Structured call (JSON output parsed into Pydantic via Tier 1 analyst model)
  L5 - Full agent pipeline (force-run, all 8 steps, shadow mode)
"""

import asyncio
import sys
import json
import time
import traceback

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

results = []

def record(level, name, ok, detail=""):
    results.append((level, name, ok, detail))
    icon = PASS if ok else FAIL
    print(f"{icon} {level} | {name}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"       {line}")

print()
print("=" * 65)
print("  XAU/USD Multi-Agent System — End-to-End Verification")
print("  Backend: Kimi K3 (moonshotai/kimi-k3-free)")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────────
# L1 — Config & credentials
# ─────────────────────────────────────────────────────────────────────────────
print("─── L1: Config & Credentials ───────────────────────────────")
try:
    from agents.config import cfg
    cfg.validate()
    record("L1", "OPENAI_API_KEY set",    True,  f"key prefix: {cfg.openai_api_key[:12]}...")
    record("L1", "OPENAI_BASE_URL set",   True,  cfg.openai_base_url)
    record("L1", "Analyst model",         True,  cfg.analyst_model)
    record("L1", "Reasoning model",       True,  cfg.reasoning_model)
    record("L1", "Decision model",        True,  cfg.decision_model)
    record("L1", "Shadow mode",           True,  f"shadow_mode={cfg.shadow_mode}")
    record("L1", "Debate rounds",         True,  f"debate_rounds={cfg.debate_rounds}")
    weights = cfg.get_regime_weights("PANIC", 2)
    record("L1", "Regime weights (PANIC)", True,
           f"Risky={weights[0]:.2f} Neutral={weights[1]:.2f} Safe={weights[2]:.2f}")
except Exception as e:
    record("L1", "Config load", False, traceback.format_exc())
    print(f"\n{FAIL} L1 failed — cannot continue.\n")
    sys.exit(1)

print()

# ─────────────────────────────────────────────────────────────────────────────
# L2 — Connectivity: raw HTTP probe to the base URL
# ─────────────────────────────────────────────────────────────────────────────
print("─── L2: API Connectivity ────────────────────────────────────")
try:
    import httpx
    t0 = time.monotonic()
    # Simple OPTIONS/GET to check the endpoint is reachable
    r = httpx.get(f"{cfg.openai_base_url}/models",
                  headers={"Authorization": f"Bearer {cfg.openai_api_key}"},
                  timeout=15)
    ms = int((time.monotonic() - t0) * 1000)
    ok = r.status_code in (200, 401, 403, 404)  # any HTTP response = server reachable
    try:
        body = r.json()
        model_ids = [m.get("id", "") for m in body.get("data", [])]
        kimi_visible = any("kimi" in m.lower() or "moonshot" in m.lower() for m in model_ids)
        detail = f"HTTP {r.status_code} in {ms}ms | models listed: {len(model_ids)}"
        if kimi_visible:
            detail += f"\n  ✓ kimi-k3-free visible in model list"
        else:
            detail += f"\n  (model list: {model_ids[:5]})"
    except Exception:
        detail = f"HTTP {r.status_code} in {ms}ms"
    record("L2", "Endpoint reachable", ok, detail)
except Exception as e:
    record("L2", "Endpoint reachable", False, str(e))

print()

# ─────────────────────────────────────────────────────────────────────────────
# L3 — Unstructured call (plain text)
# ─────────────────────────────────────────────────────────────────────────────
print("─── L3: Live Unstructured API Call ─────────────────────────")
async def test_unstructured():
    from agents.llm_client import llm
    t0 = time.monotonic()
    text, meta = await llm.call_reasoning_text(
        agent_name="VerificationTest",
        system_prompt="You are a concise assistant. Respond in one sentence only.",
        user_prompt="What is XAU/USD and why do traders watch it?",
    )
    ms = int((time.monotonic() - t0) * 1000)
    ok = isinstance(text, str) and len(text) > 10
    detail = (
        f"Latency: {ms}ms | tokens: {meta.input_tokens}+{meta.output_tokens}\n"
        f"Response: {text[:120]}{'...' if len(text) > 120 else ''}"
    )
    return ok, detail

try:
    ok, detail = asyncio.run(test_unstructured())
    record("L3", "Unstructured text call", ok, detail)
except Exception as e:
    record("L3", "Unstructured text call", False, traceback.format_exc())

print()

# ─────────────────────────────────────────────────────────────────────────────
# L4 — Structured call (JSON → Pydantic)
# ─────────────────────────────────────────────────────────────────────────────
print("─── L4: Structured JSON Output (Pydantic parsing) ──────────")
async def test_structured():
    from agents.llm_client import llm
    from agents.schemas import TechnicalSpecialistReport

    t0 = time.monotonic()
    report, meta = await llm.call_analyst(
        agent_name="StructuredTest",
        system_prompt=(
            "You are a technical specialist for XAU/USD. "
            "Return a dummy/placeholder analysis. "
            "All numeric fields should be realistic gold price values near 2400."
        ),
        user_prompt=(
            "Generate a sample TechnicalSpecialistReport for XAU/USD. "
            "Use placeholder values — this is a system verification call."
        ),
        schema=TechnicalSpecialistReport,
    )
    ms = int((time.monotonic() - t0) * 1000)
    ok = isinstance(report, TechnicalSpecialistReport)
    detail = (
        f"Latency: {ms}ms | tokens: {meta.input_tokens}+{meta.output_tokens}\n"
        f"trend={report.trend_direction.value}  "
        f"confidence={report.confidence_score:.2f}  rsi={report.rsi_value:.1f}  "
        f"summary={report.summary[:80]}..."
    )
    return ok, detail

try:
    ok, detail = asyncio.run(test_structured())
    record("L4", "Structured JSON → TechnicalSpecialistReport", ok, detail)
except Exception as e:
    record("L4", "Structured JSON → TechnicalSpecialistReport", False, traceback.format_exc())

print()

# ─────────────────────────────────────────────────────────────────────────────
# L5 — Full pipeline (force-run, all 8 steps)
# ─────────────────────────────────────────────────────────────────────────────
print("─── L5: Full Agent Pipeline (Force-Run) ─────────────────────")
print(f"  {INFO} Running all 8 pipeline steps — this may take 60-180s...")
print(f"  {INFO} Shadow mode: {cfg.shadow_mode} (no real orders will be placed)")
print()

try:
    from agents.shadow_runner import force_run

    t0 = time.monotonic()
    result = force_run(quant_override={
        "prob_up": 0.62,
        "p_cat": 0.60,
        "p_xgb": 0.63,
        "p_lgb": 0.61,
        "quant_signal": "BUY",
    })
    ms = int((time.monotonic() - t0) * 1000)

    ok = result.get("status") == "success"
    if ok:
        detail = (
            f"Session ID  : {result.get('session_id')}\n"
            f"Decision    : {result.get('final_decision')}\n"
            f"Direction   : {result.get('final_direction')}\n"
            f"Lot size    : {result.get('final_lot_size')}\n"
            f"Shadow mode : {result.get('shadow_mode')}\n"
            f"Total cost  : ${result.get('total_cost_usd', 0):.5f}\n"
            f"Wall time   : {ms}ms"
        )
    else:
        detail = f"status={result.get('status')} | {result.get('message', result)}"

    record("L5", "Full pipeline (all 8 steps)", ok, detail)

except Exception as e:
    record("L5", "Full pipeline (all 8 steps)", False, traceback.format_exc())

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
passed = sum(1 for _, _, ok, _ in results if ok)
failed = sum(1 for _, _, ok, _ in results if not ok)
total  = len(results)
print(f"  RESULT: {passed}/{total} checks passed  |  {failed} failed")
print("=" * 65)

if failed == 0:
    print("\n  ✅  ALL SYSTEMS GO — Kimi K3 integration fully operational.\n")
    sys.exit(0)
else:
    print(f"\n  ❌  {failed} check(s) failed — review output above.\n")
    sys.exit(1)
