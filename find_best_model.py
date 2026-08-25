"""
find_best_model.py — Test all fast tokenrouter models and pick the best one
"""
import httpx, json, sys, os, time
os.chdir(r'd:\UOR\L04\Research\Macroeconomic Financial News Technical\Data_Collection')
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv('.env', override=True)
from agents.config import AgentConfig
cfg = AgentConfig()

BASE = cfg.kimi_base_url
KEY  = cfg.kimi_api_key
HDRS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

# Best candidates for a reasoning/trading agent — free or very cheap
# Selected from tokenrouter list that previously responded
CANDIDATES = [
    "qwen/qwen3.5-9b",
    "qwen/qwen3.7-max",
    "google/gemini-3.5-flash-lite",
    "openai/gpt-5.4-nano",
    "deepseek/deepseek-v4-pro",
    "moonshotai/kimi-k2-free",
    "moonshotai/kimi-k3-free",
]

TEST_MSG = [
    {"role": "system", "content": "You are a financial analyst. Answer concisely."},
    {"role": "user",   "content": "In one sentence: what is the main risk when trading gold (XAU/USD)?"}
]

print(f"Testing {len(CANDIDATES)} models on {BASE}")
print("=" * 65)

results = []
for model in CANDIDATES:
    payload = {"model": model, "messages": TEST_MSG, "max_tokens": 80, "temperature": 0.1, "stream": False}
    try:
        t0 = time.monotonic()
        r = httpx.post(f"{BASE}/chat/completions", headers=HDRS, json=payload, timeout=25)
        ms = int((time.monotonic() - t0) * 1000)
        if r.status_code == 200:
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            tin  = usage.get("prompt_tokens", 0)
            tout = usage.get("completion_tokens", 0)
            print(f"PASS  {model}")
            print(f"      {ms}ms | in={tin} out={tout}")
            print(f"      Reply: {text[:120]}")
            results.append((model, ms, "PASS"))
        else:
            err = r.json().get("error", {}).get("message", r.text[:80])
            print(f"FAIL  {model}  HTTP {r.status_code}: {err}")
            results.append((model, 0, f"HTTP {r.status_code}"))
    except httpx.ReadTimeout:
        print(f"TOUT  {model}  (>25s timeout)")
        results.append((model, 99999, "TIMEOUT"))
    except Exception as e:
        print(f"ERR   {model}  {e}")
        results.append((model, 0, str(e)[:60]))
    print()

print("=" * 65)
print("SUMMARY (fastest passing models first):")
passing = [(m, ms) for m, ms, s in results if s == "PASS"]
passing.sort(key=lambda x: x[1])
for m, ms in passing:
    print(f"  {ms:5d}ms  {m}")
if not passing:
    print("  No models passed within 25s timeout.")
