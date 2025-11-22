import json
import os
from typing import Any, Dict, List, Optional

# Путь к конфигу рядом с кодом (а не в случайной текущей директории)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "pp_config.json")

DEFAULT_CONFIG: Dict[str, Any] = {
    # Параметры для обычного чата
    "chat_temperature": 0.7,
    "chat_max_tokens": 1500,

    # Дефолтный размер summary (когда пользователь явно не указал X)
    "summary_default_max_tokens": 400,

    # Дефолтная модель (по имени, как в Yandex)
    "default_model": "yandexgpt",

    # Список моделей (редактируемый пользователем)
    # порядок = индексы 1..N
    "models": [
        "yandexgpt",
        "yandexgpt-lite"
    ],
}

_config_cache: Optional[Dict[str, Any]] = None


def load_config() -> Dict[str, Any]:
    """
    Загружает конфиг из pp_config.json.
    Если файла нет — создаёт с DEFAULT_CONFIG.
    Если файл битый — использует DEFAULT_CONFIG, но файл не перезаписывает.
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    if not os.path.exists(CONFIG_PATH):
        cfg = DEFAULT_CONFIG.copy()
        save_config(cfg)
        _config_cache = cfg
        return cfg

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # Файл битый / не читается — fallback
        data = {}

    # Мерджим с дефолтами, чтобы новые поля появлялись автоматически
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(data)

    _config_cache = cfg
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    """Сохраняет конфиг и обновляет кэш."""
    global _config_cache
    _config_cache = cfg
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# ======= helpers для моделей =======

def get_model_map() -> Dict[int, str]:
    """
    Строит dict вида {1: 'yandexgpt', 2: 'yandexgpt-lite', ...}
    из списка models в конфиге.
    """
    cfg = load_config()
    models: List[str] = cfg.get("models", [])
    return {i + 1: name for i, name in enumerate(models)}


def set_default_model(model_name: str) -> None:
    cfg = load_config()
    cfg["default_model"] = model_name
    save_config(cfg)


# ======= helpers для параметров чата =======

def get_chat_temperature() -> float:
    cfg = load_config()
    return float(cfg.get("chat_temperature", DEFAULT_CONFIG["chat_temperature"]))


def set_chat_temperature(temp: float) -> None:
    cfg = load_config()
    cfg["chat_temperature"] = float(temp)
    save_config(cfg)


def get_chat_max_tokens() -> int:
    cfg = load_config()
    return int(cfg.get("chat_max_tokens", DEFAULT_CONFIG["chat_max_tokens"]))


def set_chat_max_tokens(max_tokens: int) -> None:
    cfg = load_config()
    cfg["chat_max_tokens"] = int(max_tokens)
    save_config(cfg)


def get_summary_default_max_tokens() -> int:
    cfg = load_config()
    return int(cfg.get("summary_default_max_tokens", DEFAULT_CONFIG["summary_default_max_tokens"]))
