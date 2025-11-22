import os
from typing import Any, Dict, List, Tuple, Optional

import requests

from config import (
    load_config,
    get_model_map,
    get_chat_temperature,
    get_chat_max_tokens,
)

# =========================
# YandexGPT CONFIG
# =========================

YAGPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

YAC_FOLDER = os.getenv("YAC_FOLDER")
YAC_API_KEY = os.getenv("YAC_API_KEY")

# Если задано через env — это имеет приоритет как "форс-модель"
YAC_MODEL_ENV = os.getenv("YAC_MODEL")

# =========================
# MODEL MAP (из конфига)
# =========================

MODEL_MAP: Dict[int, str] = get_model_map()


class LlmError(RuntimeError):
    pass


def get_default_model_name() -> str:
    """
    Дефолтная модель, если явно не указана.
    Приоритет:
      1) YAC_MODEL env, если задан
      2) default_model из конфига
      3) первая модель из MODEL_MAP
      4) "yandexgpt" как совсем последний fallback
    """
    if YAC_MODEL_ENV:
        return YAC_MODEL_ENV

    cfg = load_config()
    model_from_cfg = cfg.get("default_model")
    if model_from_cfg:
        return model_from_cfg

    if MODEL_MAP:
        # первая модель по индексу
        return next(iter(MODEL_MAP.values()))

    return "yandexgpt"


def get_model_name_by_index(index: int) -> Optional[str]:
    """
    Возвращает имя модели по индексу или None, если индекс неверный.
    """
    return MODEL_MAP.get(index)


def call_yandex_llm(
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model_name: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    messages: list of {"role": "system"|"user"|"assistant", "text": "..."}

    Returns (reply_text, usage_dict)

    Если temperature или max_tokens не заданы — берутся из конфига.
    Если model_name не задано — берётся дефолтная модель.
    """
    if not YAC_FOLDER or not YAC_API_KEY:
        raise LlmError("Missing YAC_FOLDER or YAC_API_KEY environment variables")

    if model_name is None:
        model_name = get_default_model_name()

    if temperature is None:
        temperature = get_chat_temperature()

    if max_tokens is None:
        max_tokens = get_chat_max_tokens()

    payload = {
        "modelUri": f"gpt://{YAC_FOLDER}/{model_name}",
        "completionOptions": {
            "temperature": float(temperature),
            "maxTokens": int(max_tokens),
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
