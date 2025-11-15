# llm_client.py
import os
from typing import Any, Dict, List, Tuple

import requests

# =========================
# YandexGPT CONFIG
# =========================

YAGPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

YAC_FOLDER = os.getenv("YAC_FOLDER")
YAC_API_KEY = os.getenv("YAC_API_KEY")

# Prefer full yandexgpt over lite in this AI challenge
YAC_MODEL = os.getenv("YAC_MODEL", "yandexgpt")


class LlmError(RuntimeError):
    pass


def call_yandex_llm(
    messages: List[Dict[str, str]],
    temperature: float = 0.6,
    max_tokens: int = 1500,
) -> Tuple[str, Dict[str, Any]]:
    """
    messages: list of {"role": "system"|"user"|"assistant", "text": "..."}

    Returns (reply_text, usage_dict)
    """
    if not YAC_FOLDER or not YAC_API_KEY:
        raise LlmError("Missing YAC_FOLDER or YAC_API_KEY environment variables")

    payload = {
        "modelUri": f"gpt://{YAC_FOLDER}/{YAC_MODEL}",
        "completionOptions": {
            "temperature": temperature,
            "maxTokens": max_tokens,
            "stream": False,
        },
        "messages": messages,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YAC_API_KEY}",
    }

    resp = requests.post(YAGPT_URL, headers=headers, json=payload)
    resp.raise_for_status()

    data = resp.json()

    # Extract text
    try:
        reply_text = data["result"]["alternatives"][0]["message"]["text"]
    except (KeyError, IndexError) as e:
        raise LlmError(f"Unexpected Yandex response format: {data}") from e

    # Get usage from either top-level or result.usage
    usage = data.get("usage") or data.get("result", {}).get("usage", {}) or {}

    return reply_text, usage
