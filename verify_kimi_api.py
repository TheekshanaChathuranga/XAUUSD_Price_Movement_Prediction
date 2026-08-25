"""
verify_kimi_api.py  — Diagnose kimi-k3-free on tokenrouter
Tests: streaming, non-streaming (max_tokens), non-streaming (max_completion_tokens)
"""
import httpx, json, sys, os, time
os.chdir(r'd:\UOR\L04\Research\Macroeconomic Financial News Technical\Data_Collection')
sys.path.insert(0, '.')
from agents.config import cfg

BASE  = cfg.kimi_base_url
KEY   = cfg.kimi_api_key
MODEL = "moonshotai/kimi-k3-free"
HDRS  = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

def sep(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

# ─── Test 1: STREAMING ───────────────────────────────────────
sep("Test 1: Streaming (stream=True, max_tokens=20)")
payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 20,
    "stream": True
}
try:
    t0 = time.monotonic()
    chunks = []
    with httpx.stream("POST", f"{BASE}/chat/completions",
                      headers=HDRS, json=payload, timeout=90) as r:
        print(f"HTTP {r.status_code}")
        for line in r.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    d = json.loads(line[6:])
                    delta = d.get("choices",[{}])[0].get("delta",{}).get("content","")
                    if delta:
                        chunks.append(delta)
                    finish = d.get("choices",[{}])[0].get("finish_reason")
                    if finish:
                        print(f"  finish_reason: {finish}")
                        break
                except Exception:
                    pass
            if time.monotonic() - t0 > 75:
                print("  [75s limit hit]")
                break
    ms = int((time.monotonic()-t0)*1000)
    text = "".join(chunks)
    print(f"  Response : {repr(text)}")
    print(f"  Elapsed  : {ms}ms")
    print(f"  RESULT   : {'PASS' if text else 'FAIL - empty content'}")
except Exception as e:
    print(f"  ERROR: {e}")

# ─── Test 2: NON-STREAMING with max_tokens ────────────────────
sep("Test 2: Non-streaming (stream=False, max_tokens=20)")
payload2 = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 20,
    "stream": False
}
try:
    t0 = time.monotonic()
    r = httpx.post(f"{BASE}/chat/completions",
                   headers=HDRS, json=payload2, timeout=90)
    ms = int((time.monotonic()-t0)*1000)
    print(f"HTTP {r.status_code}  ({ms}ms)")
    data = r.json()
    print(json.dumps(data, indent=2)[:1000])
except Exception as e:
    print(f"  ERROR: {e}")

# ─── Test 3: NON-STREAMING with max_completion_tokens ─────────
sep("Test 3: Non-streaming (stream=False, max_completion_tokens=20)")
payload3 = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Hi"}],
    "max_completion_tokens": 20,
    "stream": False
}
try:
    t0 = time.monotonic()
    r = httpx.post(f"{BASE}/chat/completions",
                   headers=HDRS, json=payload3, timeout=90)
    ms = int((time.monotonic()-t0)*1000)
    print(f"HTTP {r.status_code}  ({ms}ms)")
    data = r.json()
    print(json.dumps(data, indent=2)[:1000])
except Exception as e:
    print(f"  ERROR: {e}")

print()
print("=" * 60)
print("  Probe complete")
print("=" * 60)
