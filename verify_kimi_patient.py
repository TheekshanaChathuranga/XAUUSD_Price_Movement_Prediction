"""
verify_kimi_patient.py — Wait up to 5 minutes for kimi-k3-free on tokenrouter
"""
import httpx, json, sys, os, time
os.chdir(r'd:\UOR\L04\Research\Macroeconomic Financial News Technical\Data_Collection')
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv('.env', override=True)
from agents.config import AgentConfig
cfg = AgentConfig()

BASE  = cfg.kimi_base_url
KEY   = cfg.kimi_api_key
MODEL = cfg.analyst_model
HDRS  = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

print(f"Base URL : {BASE}")
print(f"Model    : {MODEL}")
print(f"API Key  : {KEY[:16]}...")
print()

TIMEOUT = 300  # 5 minutes
print(f"Sending request — waiting up to {TIMEOUT}s...")

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Say only the word: hello"}],
    "max_tokens": 20,
    "temperature": 0.1,
    "stream": False
}

t0 = time.monotonic()
try:
    r = httpx.post(f"{BASE}/chat/completions",
                   headers=HDRS, json=payload, timeout=TIMEOUT)
    ms = int((time.monotonic() - t0) * 1000)
    print(f"HTTP {r.status_code}  ({ms}ms / {ms//1000}s)")
    if r.status_code == 200:
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        print(f"Response : {repr(text)}")
        print(f"Usage    : {usage}")
        print()
        print("==> KIMI API WORKING - PASS")
    else:
        print(f"Error body: {r.text[:500]}")
        print("==> FAIL")
except httpx.ReadTimeout:
    ms = int((time.monotonic() - t0) * 1000)
    print(f"TIMEOUT after {ms//1000}s — model did not respond")
    print("==> kimi-k3-free is queued / overloaded on tokenrouter")
except Exception as e:
    print(f"ERROR: {e}")
