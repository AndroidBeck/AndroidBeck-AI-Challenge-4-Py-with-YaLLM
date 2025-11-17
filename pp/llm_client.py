# llm_client.py
import os
from typing import Any, Dict, List, Tuple, Optional

import requests


# =========================
# YandexGPT CONFIG
# =========================

YAGPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

YAC_FOLDER = os.getenv("YAC_FOLDER")
YAC_API_KEY = os.getenv("YAC_API_KEY")

# Prefer full yandexgpt over lite in this AI challenge
YAC_MODEL_ENV = os.getenv("YAC_MODEL", "yandexgpt")


# =========================
# MODEL MAP
# =========================

MODEL_MAP: Dict[int, str] = {
    1: "yandexgpt",
    2: "yandexgpt-lite",
    3: "qwen2.5-7b-instruct", # "qwen3-235b-a22b-fp8/latest",
    4: "gpt-oss-120b/latest",
}


class LlmError(RuntimeError):
    pass


def get_default_model_name() -> str:
    """
    Default model used if none explicitly selected.
    Priority:
      1) YAC_MODEL env, if present
      2) model #1 from MODEL_MAP (yandexgpt)
    """
    if YAC_MODEL_ENV:
        return YAC_MODEL_ENV

    return MODEL_MAP[1]


def get_model_name_by_index(index: int) -> Optional[str]:
    """
    Returns model name for index (1..5) or None if invalid.
    """
    return MODEL_MAP.get(index)


def call_yandex_llm(
    messages: List[Dict[str, str]],
    temperature: float = 0.6,
    max_tokens: int = 1500,
    model_name: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    messages: list of {"role": "system"|"user"|"assistant", "text": "..."}

    Returns (reply_text, usage_dict)
    """
    if not YAC_FOLDER or not YAC_API_KEY:
        raise LlmError("Missing YAC_FOLDER or YAC_API_KEY environment variables")

    if model_name is None:
        model_name = get_default_model_name()

    payload = {
        "modelUri": f"gpt://{YAC_FOLDER}/{model_name}",
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
