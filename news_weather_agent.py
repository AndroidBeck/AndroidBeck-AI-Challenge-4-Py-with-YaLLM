"""
News + Weather Monitor MCP server.

Функции:
- Периодически собирает погоду и новости, сохраняет в SQLite и делает дайджест.
- Печатает дайджест в stdout (эффект "агент 24/7").
- Даёт MCP-инструменты для управления и просмотра дайджестов.

Зависимости:
    pip install "mcp[cli]" requests

Запуск в режиме stdio (для ChatGPT/Claude Desktop и mcp dev):
    python news_weather_agent.py
    # или
    python -m news_weather_agent

Установка в MCP (через mcp CLI, см. оф. доку):
    mcp dev news_weather_agent.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Константы и настройки
# ---------------------------------------------------------------------------

DB_PATH = os.path.join(os.path.dirname(__file__), "news_weather_agent.db")

# Локация по умолчанию — Амстердам
DEFAULT_LOCATION_NAME = os.environ.get("NWA_LOCATION_NAME", "Amsterdam, NL")
DEFAULT_LAT = float(os.environ.get("NWA_LAT", "52.3740"))
DEFAULT_LON = float(os.environ.get("NWA_LON", "4.8897"))

# Интервал по умолчанию (минуты) для генерации дайджеста
DEFAULT_DIGEST_INTERVAL_MINUTES = int(os.environ.get("NWA_DEFAULT_INTERVAL", "60"))

# Частота, с которой фоновой поток проверяет, пора ли делать дайджест (сек)
SCHEDULER_LOOP_SLEEP_SECONDS = int(os.environ.get("NWA_LOOP_SLEEP", "60"))

# Новостной источник — Hacker News front page (без API-ключа)
HN_API_URL = "https://hn.algolia.com/api/v1/search?tags=front_page"

# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("NewsWeatherAgent")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def get_db() -> sqlite3.Connection:
    """Создаёт новое соединение с БД (отдельное для каждого потока)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Создание таблиц, если их ещё нет."""
    with get_db() as conn:
        cur = conn.cursor()

        # Погода
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS weather_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                location TEXT NOT NULL,
                temperature REAL,
                wind_speed REAL,
                description TEXT
            )
            """
        )

        # Новости
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS news_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                title TEXT NOT NULL,
                url TEXT,
                source TEXT,
                score INTEGER
            )
            """
        )

        # Дайджесты
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS digests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                text TEXT NOT NULL,
                weather_snapshot TEXT,
                news_snapshot TEXT
            )
            """
        )

        # Настройки (reminder)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                digest_interval_minutes INTEGER NOT NULL,
                last_digest_at TEXT
            )
            """
        )

        # Гарантируем, что есть строка с id=1
        cur.execute("SELECT id FROM settings WHERE id = 1")
        row = cur.fetchone()
        if row is None:
            cur.execute(
                """
                INSERT INTO settings (id, digest_interval_minutes, last_digest_at)
                VALUES (1, ?, NULL)
                """,
                (DEFAULT_DIGEST_INTERVAL_MINUTES,),
            )

        conn.commit()


# ---------------------------------------------------------------------------
# Модель настроек
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    digest_interval_minutes: int
    last_digest_at: Optional[datetime]


def load_settings() -> Settings:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT digest_interval_minutes, last_digest_at FROM settings WHERE id = 1"
        )
        row = cur.fetchone()
        if row is None:
            # На всякий случай, если таблицу почистили
            return Settings(
                digest_interval_minutes=DEFAULT_DIGEST_INTERVAL_MINUTES,
                last_digest_at=None,
            )

        last_digest = (
            datetime.fromisoformat(row["last_digest_at"])
            if row["last_digest_at"]
            else None
        )
        return Settings(
            digest_interval_minutes=row["digest_interval_minutes"],
            last_digest_at=last_digest,
        )


def save_settings(settings: Settings) -> None:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE settings
            SET digest_interval_minutes = ?, last_digest_at = ?
            WHERE id = 1
            """,
            (
                settings.digest_interval_minutes,
                settings.last_digest_at.isoformat() if settings.last_digest_at else None,
            ),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Внешние запросы: погода и новости
# ---------------------------------------------------------------------------


def fetch_weather() -> Dict[str, Any]:
    """
    Берём текущую погоду с Open-Meteo для заданных координат.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={DEFAULT_LAT}&longitude={DEFAULT_LON}"
        "&current_weather=true"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    current = data.get("current_weather", {}) or {}
    temperature = current.get("temperature")
    windspeed = current.get("windspeed")
    # В Open-Meteo код погоды, для простоты превратим в строку
    desc = f"code={current.get('weathercode')}"

    # Логируем в БД
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO weather_logs (created_at, location, temperature, wind_speed, description)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                DEFAULT_LOCATION_NAME,
                temperature,
                windspeed,
                desc,
            ),
        )
        conn.commit()

    return {
        "location": DEFAULT_LOCATION_NAME,
        "temperature": temperature,
        "wind_speed": windspeed,
        "description": desc,
    }


def fetch_news(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Берём топовые новости с Hacker News.
    """
    resp = requests
