# news_weather_server.py
from __future__ import annotations

from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

import news_weather_core as core

mcp = FastMCP("NewsWeatherAgent")


@mcp.tool()
def set_digest_interval(minutes: int) -> Dict[str, Any]:
    """
    Установить интервал (в минутах) генерации дайджеста.
    Это наш reminder.
    """
    if minutes <= 0:
        raise ValueError("minutes must be positive")

    core.init_db()
    settings = core.load_settings()
    settings.digest_interval_minutes = minutes
    core.save_settings(settings)

    return {
        "message": "Digest interval updated",
        "digest_interval_minutes": settings.digest_interval_minutes,
        "last_digest_at": settings.last_digest_at.isoformat()
        if settings.last_digest_at
        else None,
    }


@mcp.tool()
def get_settings() -> Dict[str, Any]:
    """
    Текущие настройки планировщика.
    """
    core.init_db()
    s = core.load_settings()
    return {
        "digest_interval_minutes": s.digest_interval_minutes,
        "last_digest_at": s.last_digest_at.isoformat() if s.last_digest_at else None,
    }


@mcp.tool()
def get_latest_digest() -> Dict[str, Any]:
    """
    Последний дайджест новостей и погоды.
    """
    core.init_db()
    d = core.get_latest_digest()
    if d is None:
        return {"message": "No digests yet"}
    return d


@mcp.tool()
def list_digests(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Список последних N дайджестов (укороченный вид).
    """
    core.init_db()
    return core.list_digests(limit=limit)


@mcp.tool()
def get_raw_data(hours: int = 24) -> Dict[str, Any]:
    """
    Сырые логи новостей и погоды за последние X часов.
    """
    core.init_db()
    return core.get_raw_data(hours=hours)


if __name__ == "__main__":
    # Для запуска как MCP-сервер (например, из ChatGPT или `mcp dev`)
    core.init_db()
    core.start_scheduler_thread()
    mcp.run(transport="stdio")
