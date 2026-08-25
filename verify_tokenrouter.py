"""
verify_tokenrouter.py — Test if tokenrouter endpoint itself works with other models
"""
import httpx, json, sys, os, time
os.chdir(r'd:\UOR\L04\Research\Macroeconomic Financial News Technical\Data_Collection')
sys.path.insert(0, '.')
from agents.config import cfg

BASE = cfg.kimi_base_url
KEY  = cfg.kimi_api_key
HDRS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

# Models to try in order — all were visible in /models list
CANDIDATES = [
    "qwen/qwen3.5-9b",
    "google/gemini-3.5-flash-lite",
    "openai/gpt-5.4-nano",
    "moonshotai/kimi-k3-free",
]

MSG = [{"role": "user", "content": "Say only the word: hello"}]

for model in CANDIDATES:
    print(f"\n>>> Testing: {model}")
    payload = {"model": model, "messages": MSG, "max_tokens": 20, "stream": False}
    try:
        t0 = time.monotonic()
        r = httpx.post(f"{BASE}/chat/completions", headers=HDRS,
                       json=payload, timeout=30)
        ms = int((time.monotonic() - t0) * 1000)
        if r.status_code == 200:
            data = r.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            print(f"    PASS  {ms}ms | reply: {repr(text[:80])} | tokens: {usage}")
        else:
            print(f"    FAIL  HTTP {r.status_code}  {ms}ms | {r.text[:200]}")
    except httpx.ReadTimeout:
        ms = int((time.monotonic() - t0) * 1000)
        print(f"    TIMEOUT after {ms}ms — model queued / unavailable")
    except Exception as e:
        print(f"    ERROR: {e}")

print("\nDone.")
