import os
import time
import json
import requests
from typing import Tuple, Optional

# =========================
# Config
# =========================

# ---- Yandex Cloud LLM (native) ----
YAGPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YAC_API_KEY = os.getenv("YAC_API_KEY")
YAC_FOLDER  = os.getenv("YAC_FOLDER")
YAC_MODEL   = "yandexgpt"  # <- as requested

# ---- Hugging Face Router (OpenAI-compatible) ----
from openai import OpenAI
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_CLIENT = OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_API_TOKEN)

# Full model IDs (printed exactly as below and used as-is)
HF_MODELS = [
    "moonshotai/Kimi-K2-Thinking:novita",
    "zai-org/GLM-4.6:novita",
    "meta-llama/Llama-3.1-8B-Instruct:novita",
    "meta-llama/Llama-3.1-8B-Instruct:nebius",
    "deepseek-ai/DeepSeek-R1",
    "openai/gpt-oss-120b",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
]

# Generation params
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 400

# =========================
# Utilities
# =========================

def approx_tokens(text: str) -> int:
    """Crude token estimate ≈ characters/4."""
    return max(1, len(text or "") // 4)

def print_divider():
    print("\n" + "-" * 80 + "\n")

# =========================
# Backends
# =========================

def call_yandex(prompt: str, model: str = YAC_MODEL) -> Tuple[Optional[str], int, int]:
    """
    Returns: (answer, input_tokens, output_tokens)
    Uses Yandex usage fields if present; otherwise approximates.
    """
    if not (YAC_API_KEY and YAC_FOLDER):
        raise RuntimeError("Yandex credentials missing: set YAC_API_KEY and YAC_FOLDER.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YAC_API_KEY}",
        "x-folder-id": YAC_FOLDER,
    }
    payload = {
        "modelUri": f"gpt://{YAC_FOLDER}/{model}",
        "completionOptions": {
            "stream": False,
            "temperature": TEMPERATURE,
            "maxTokens": MAX_NEW_TOKENS
        },
        "messages": [{"role": "user", "text": prompt}],
    }

    r = requests.post(YAGPT_URL, headers=headers, json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Yandex API error {r.status_code}: {r.text}")

    data = r.json()
    result = data.get("result", {})
    alts = result.get("alternatives", [])
    answer = (alts[0].get("message", {}) or {}).get("text") if alts else None

    usage = result.get("usage") or {}
    in_tok  = usage.get("inputTextTokens")
    out_tok = usage.get("completionTokens")

    if in_tok is None:  in_tok  = approx_tokens(prompt)
    if out_tok is None: out_tok = approx_tokens(answer or "")
    return answer, in_tok, out_tok


def call_hf_chat(prompt: str, model_id: str) -> Tuple[Optional[str], int, int]:
    """
    Calls HF Router (OpenAI-compatible /v1) chat.completions.
    Returns: (answer, input_tokens_est_or_real, output_tokens_est_or_real)
    """
    if not HF_API_TOKEN:
        raise RuntimeError("Hugging Face token missing: set HF_API_TOKEN.")

    # Some backends provide usage; others don't.
    resp = HF_CLIENT.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
    )

    answer = resp.choices[0].message.content if resp.choices else None

    in_tok = getattr(getattr(resp, "usage", None), "prompt_tokens", None)
    out_tok = getattr(getattr(resp, "usage", None), "completion_tokens", None)

    if in_tok is None:  in_tok  = approx_tokens(prompt)
    if out_tok is None: out_tok = approx_tokens(answer or "")

    return answer, in_tok, out_tok

# =========================
# CLI
# =========================

def main():
    print("=== Day 6 — Multi-Model Prompt Runner ===")
    print("Backends: Yandex (native) + Hugging Face Router (OpenAI-compatible).")
    print("Env needed: YAC_API_KEY, YAC_FOLDER, HF_API_TOKEN\n")

    task = input("What would you like to ask the models?\n> ").strip()
    if not task:
        print("No task entered. Exiting.")
        return

    while True:
        print_divider()
        print("Choose a model:")
        print(f"1 - YandexGPT ({YAC_MODEL})")
        # list all HF models with their full names as requested
        for idx, mid in enumerate(HF_MODELS, start=2):
            print(f"{idx} - {mid}")
        exit_num = 2 + len(HF_MODELS)
        print(f"{exit_num} - Exit")

        choice_raw = input("> ").strip()
        if not choice_raw.isdigit():
            print("Please enter a number.")
            continue

        choice = int(choice_raw)
        if choice == 1:
            backend = f"Yandex • {YAC_MODEL}"
            try:
                start = time.perf_counter()
                answer, in_tok, out_tok = call_yandex(task, model=YAC_MODEL)
                elapsed = time.perf_counter() - start
            except Exception as e:
                print_divider()
                print("Error:", str(e))
                print_divider()
                continue

        elif 2 <= choice < exit_num:
            model_id = HF_MODELS[choice - 2]
            backend = f"HF Router • {model_id}"
            try:
                start = time.perf_counter()
                answer, in_tok, out_tok = call_hf_chat(task, model_id)
                elapsed = time.perf_counter() - start
            except Exception as e:
                print_divider()
                print("Error:", str(e))
                print_divider()
                continue

        elif choice == exit_num:
            print("Bye!")
            break
        else:
            print("Unknown choice.")
            continue

        # Print result
        print_divider()
        print(f"Model:  {backend}")
        print(f"Time:   {elapsed:.2f} s")
        print(f"Tokens: input ≈ {in_tok}, output ≈ {out_tok}, total ≈ {in_tok + out_tok}")
        print_divider()
        print("Answer:\n")
        print(answer or "[No answer]")
        print_divider()


if __name__ == "__main__":
    main()
