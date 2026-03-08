"""Test HuggingFace Inference Providers (new 2025 router endpoint)."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN", "")

if not token:
    print("❌ No HF_TOKEN in .env")
    exit(1)

# ══════════════════════════════════════════════════════════════════
#  NEW URL — HF router (replaces dead api-inference.huggingface.co)
#  OpenAI-compatible chat completions format
# ══════════════════════════════════════════════════════════════════
NEW_URL = "https://router.huggingface.co/v1/chat/completions"

candidates = [
    # ── DeepSeek R1 ───────────────────────────────────────────────
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1",
    # ── Qwen3 ─────────────────────────────────────────────────────
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/QwQ-32B",
    # ── Qwen2.5 ───────────────────────────────────────────────────
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    # ── Meta Llama ────────────────────────────────────────────────
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    # ── Mistral ───────────────────────────────────────────────────
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    # ── Microsoft Phi ─────────────────────────────────────────────
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-4-mini-instruct",
    # ── Google Gemma ──────────────────────────────────────────────
    "google/gemma-2-9b-it",
    "google/gemma-3-12b-it",
    # ── Others ────────────────────────────────────────────────────
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

messages = [
    {
        "role": "user",
        "content": "Bitcoin RSI=72, MACD bearish crossover, funding rate 0.01%. One word: LONG, SHORT, or HOLD?"
    }
]

print(f"\nTesting {len(candidates)} models via NEW HF Router...")
print(f"URL: {NEW_URL}\n")
print(f"{'Model':<55} {'Status':<10} {'Response'}")
print("─" * 100)

working    = []
gated      = []
dead       = []
no_credits = []

for model in candidates:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 30,
        "temperature": 0.3,
    }

    try:
        resp = requests.post(NEW_URL, headers=headers, json=payload, timeout=30)
        code = resp.status_code

        if code == 200:
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()[:70]
            print(f"  ✅ {model:<53} {code:<10} {text}")
            working.append(model)

        elif code == 401:
            print(f"  🔒 {model:<53} {code:<10} Auth failed — check token")

        elif code == 403:
            print(f"  🔒 {model:<53} {code:<10} Gated — request access at HF")
            gated.append(model)

        elif code == 402:
            print(f"  💳 {model:<53} {code:<10} Out of free credits")
            no_credits.append(model)

        elif code == 404:
            print(f"  ❌ {model:<53} {code:<10} Not on router")
            dead.append(model)

        elif code == 429:
            print(f"  ⚠️  {model:<53} {code:<10} Rate limited (exists)")
            working.append(model)

        else:
            try:
                msg = resp.json().get("error", {})
                if isinstance(msg, dict):
                    msg = msg.get("message", resp.text[:80])
            except Exception:
                msg = resp.text[:80]
            print(f"  ❓ {model:<53} {code:<10} {msg}")

    except requests.exceptions.Timeout:
        print(f"  ⏰ {model:<53} {'timeout':<10}")
    except Exception as e:
        print(f"  💥 {model:<53} {'error':<10} {e}")

# ── Summary ───────────────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"  ✅ Working    : {len(working)}")
print(f"  🔒 Gated      : {len(gated)}")
print(f"  💳 No credits : {len(no_credits)}")
print(f"  ❌ Dead/404   : {len(dead)}")
print(f"  Total tested  : {len(candidates)}")

# ── Auto suggest config ───────────────────────────────────────────
print(f"\n{'='*100}")
print("  PASTE THIS INTO config.py:\n")

priority_analyst    = ["Qwen/Qwen3-8B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                       "Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3",
                       "meta-llama/Llama-3.1-8B-Instruct"]
priority_risk       = ["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "Qwen/Qwen3-8B",
                       "NousResearch/Hermes-3-Llama-3.1-8B", "mistralai/Mixtral-8x7B-Instruct-v0.1"]
priority_strategist = ["Qwen/QwQ-32B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                       "Qwen/Qwen3-32B", "Qwen/Qwen3-14B",
                       "meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen2.5-72B-Instruct"]

def pick(priority_list, working_list):
    for m in priority_list:
        if m in working_list:
            return m
    return working_list[0] if working_list else "NONE"

if working:
    analyst    = pick(priority_analyst, working)
    risk_mgr   = pick(priority_risk, working)
    strategist = pick(priority_strategist, working)
    fallbacks  = [m for m in working if m not in (analyst, risk_mgr, strategist)][:5]

    print(f'  AI_CONFIG = {{')
    print(f'      "enabled": True,')
    print(f'      "models": {{')
    print(f'          "analyst":      "{analyst}",')
    print(f'          "risk_manager": "{risk_mgr}",')
    print(f'          "strategist":   "{strategist}",')
    print(f'      }},')
    print(f'      "fallback_models": {fallbacks},')
    print(f'      "max_new_tokens": 500,')
    print(f'      "temperature": 0.6,')
    print(f'      "retry_attempts": 3,')
    print(f'      "retry_delay_seconds": 5,')
    print(f'      "timeout_seconds": 45,')
    print(f'  }}')
else:
    print("  ❌ No working models found.")
    print("  Fix: Go to https://huggingface.co/settings/tokens")
    print("  Create token → Fine-grained → tick 'Make calls to serverless Inference API'")

print()