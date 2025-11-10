import os
import sys
import json
import time
from typing import Optional
import requests

# -----------------------------
# Config: YandexGPT HTTP adapter
# -----------------------------
YAGPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YAC_MODEL = os.getenv("YAC_MODEL", "yandexgpt")
YAC_API_KEY = os.getenv("YAC_API_KEY")  # required
YAC_FOLDER = os.getenv("YAC_FOLDER")  # required


def call_yandex_gpt(prompt: str, temperature: float) -> Optional[str]:
    """
    Calls YandexGPT (completion endpoint) with a simple single-message prompt.
    Expects YAC_FOLDER and YAC_API_KEY in environment.
    Returns text or None on error.
    """
    if not (YAC_FOLDER and YAC_API_KEY):
        print("[warn] YAC_FOLDER or YAC_API_KEY not set; running in dry-run mode.")
        return f"(dry-run) Pretend LLM answer for T={temperature:.2f}:\n" \
               f"I would explain: {prompt[:120]}..."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YAC_API_KEY}",
        "x-folder-id": YAC_FOLDER,
    }

    payload = {
        "modelUri": f"gpt://{YAC_FOLDER}/{YAC_MODEL}",
        "completionOptions": {
            "temperature": float(temperature),
            "maxTokens": 1200,
            "stream": False
        },
        "messages": [
            {"role": "system", "text": "You are a helpful, concise assistant."},
            {"role": "user", "text": prompt}
        ]
    }

    try:
        resp = requests.post(YAGPT_URL, headers=headers, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Yandex returns choices[0].message.text (or similar). We’ll try to be safe:
        # Common shapes:
        #   { "result": { "alternatives": [ { "message": { "text": "..." } } ] } }
        # or for legacy: { "choices":[{"message":{"content":"..."}}] }
        # We'll probe both.
        text = None
        if isinstance(data, dict):
            # Newer Yandex schemas often:
            alt = (
                data.get("result", {})
                    .get("alternatives", [{}])
            )
            if alt and isinstance(alt, list):
                text = alt[0].get("message", {}).get("text")
        if not text:
            # Fallback to OpenAI-like
            choices = data.get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content")

        return text or "(No text in response)"
    except requests.RequestException as e:
        print(f"[error] HTTP error: {e}")
    except Exception as e:
        print(f"[error] Parse error: {e}")
    return None

# -------------------------------------------------
# Swap-in point for other providers if you want one
# -------------------------------------------------
def llm_complete(prompt: str, temperature: float) -> str:
    """
    Single switch to change providers. Currently using YandexGPT adapter.
    """
    text = call_yandex_gpt(prompt, temperature)
    if text is None:
        text = "(error) Could not obtain a response from the LLM."
    return text

# ---------------
# CLI interaction
# ---------------
def read_temperature() -> Optional[float]:
    """
    Ask the user for a temperature. If they enter a non-number, return None (signals exit).
    If it's out of [0, 2], we clamp and tell them.
    """
    raw = input("What temperature to set (0–2)? (Type any non-number to exit): ").strip()
    try:
        t = float(raw)
    except ValueError:
        return None

    if t < 0:
        print("Temperature cannot be negative. Clamping to 0.0.")
        t = 0.0
    if t > 2:
        print("Temperature above 2 is uncommon. Clamping to 2.0.")
        t = 2.0
    return t

def main():
    print("=== Tiny Temperature Playground ===")
    topic = input("What would you like to know? ").strip()
    if not topic:
        print("No question provided. Exiting.")
        return

    while True:
        t = read_temperature()
        if t is None:
            print("Exiting. Bye!")
            break

        print(f"\n→ Asking LLM (temperature={t:.2f})...")
        answer = llm_complete(topic, t)
        print("\n--- LLM Answer ---")
        print(answer)
        print("------------------\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Bye!")
