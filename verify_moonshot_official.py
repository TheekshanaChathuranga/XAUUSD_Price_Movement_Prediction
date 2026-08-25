"""
verify_moonshot_official.py — Test official api.moonshot.cn endpoint
"""
import httpx, json, sys, os, time
os.chdir(r'd:\UOR\L04\Research\Macroeconomic Financial News Technical\Data_Collection')
sys.path.insert(0, '.')

# Force reload env since dotenv caches
from dotenv import load_dotenv
load_dotenv('.env', override=True)

from agents.config import AgentConfig
cfg = AgentConfig()  # fresh instance

KEY   = cfg.kimi_api_key
BASE  = cfg.kimi_base_url
HDRS  = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

print(f"Base URL : {BASE}")
print(f"API Key  : {KEY[:16]}...")
print()

# Step 1: List models
print("=== GET /models ===")
try:
    t0 = time.monotonic()
    r = httpx.get(f"{BASE}/models", headers=HDRS, timeout=15)
    ms = int((time.monotonic()-t0)*1000)
    print(f"HTTP {r.status_code}  ({ms}ms)")
    data = r.json()
    ids = [m.get("id","") for m in data.get("data",[])]
    print(f"Models: {ids[:10]}")
except Exception as e:
    print(f"ERROR: {e}")

print()

# Step 2: Simple chat completion
print("=== POST /chat/completions (moonshot-v1-8k) ===")
payload = {
    "model": "moonshot-v1-8k",
    "messages": [{"role": "user", "content": "Say only the word: hello"}],
    "max_tokens": 20,
    "temperature": 0.1,
    "stream": False
}
try:
    t0 = time.monotonic()
    r = httpx.post(f"{BASE}/chat/completions",
                   headers=HDRS, json=payload, timeout=60)
    ms = int((time.monotonic()-t0)*1000)
    print(f"HTTP {r.status_code}  ({ms}ms)")
    if r.status_code == 200:
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        print(f"Response : {repr(text)}")
        print(f"Usage    : {usage}")
        print("PASS")
    else:
        print(f"FAIL: {r.text[:500]}")
except Exception as e:
    print(f"ERROR: {e}")
