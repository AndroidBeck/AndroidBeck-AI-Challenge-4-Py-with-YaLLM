# news_weather_core.py
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

# -----------------------------
# Константы и настройки
# -----------------------------

DB_PATH = os.path.join(os.path.dirname(__file__), "news_weather_agent.db")

DEFAULT_LOCATION_NAME = os.environ.get("NWA_LOCATION_NAME", "Amsterdam, NL")
DEFAULT_LAT = float(os.environ.get("NWA_LAT", "52.3740"))
DEFAULT_LON = float(os.environ.get("NWA_LON", "4.8897"))

DEFAULT_DIGEST_INTERVAL_MINUTES = int(os.environ.get("NWA_DEFAULT_INTERVAL", "60"))
SCHEDULER_LOOP_SLEEP_SECONDS = int(os.environ.get("NWA_LOOP_SLEEP", "60"))

HN_API_URL = "https://hn.algolia.com/api/v1/search?tags=front_page"


# -----------------------------
# База данных
# -----------------------------

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db() as conn:
        cur = conn.cursor()

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

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                digest_interval_minutes INTEGER NOT NULL,
                last_digest_at TEXT
            )
            """
        )

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


# -----------------------------
# Настройки
# -----------------------------

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


# -----------------------------
# Внешние запросы
# -----------------------------

def fetch_weather() -> Dict[str, Any]:
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
    desc = f"code={current.get('weathercode')}"

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
    resp = requests.get(HN_API_URL, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    hits = data.get("hits", [])[:limit]

    now = datetime.now().isoformat(timespec="seconds")
    items: List[Dict[str, Any]] = []

    with get_db() as conn:
        cur = conn.cursor()
        for h in hits:
            title = h.get("title") or h.get("story_title") or "No title"
            url = h.get("url") or h.get("story_url")
            score = h.get("points", 0)
            item = {
                "title": title,
                "url": url,
                "score": score,
                "source": "HackerNews",
            }
            items.append(item)

            cur.execute(
                """
                INSERT INTO news_logs (created_at, title, url, source, score)
                VALUES (?, ?, ?, ?, ?)
                """,
                (now, title, url, "HackerNews", score),
            )

        conn.commit()

    return items


# -----------------------------
# Дайджест
# -----------------------------

def build_digest_text(
    weather: Dict[str, Any],
    news: List[Dict[str, Any]],
    interval_minutes: int,
) -> str:
    parts: List[str] = []

    loc = weather.get("location", DEFAULT_LOCATION_NAME)
    t = weather.get("temperature")
    w = weather.get("wind_speed")
    desc = weather.get("description")

    parts.append("=== Weather Summary ===")
    parts.append(f"Location: {loc}")
    if t is not None:
        parts.append(f"Temperature: {t}°C")
    if w is not None:
        parts.append(f"Wind: {w} m/s")
    if desc:
        parts.append(f"Conditions: {desc}")

    parts.append("")
    parts.append(f"=== News Summary (interval ≈ {interval_minutes} min) ===")

    if not news:
        parts.append("No news items were fetched.")
    else:
        for i, item in enumerate(news, start=1):
            title = item.get("title", "No title")
            url = item.get("url") or ""
            score = item.get("score", 0)
            parts.append(f"{i}. {title} (score: {score})")
            if url:
                parts.append(f"   {url}")

    parts.append("")
    parts.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    return "\n".join(parts)


def generate_and_store_digest() -> Optional[Dict[str, Any]]:
    try:
        settings = load_settings()

        weather = fetch_weather()
        news = fetch_news(limit=5)

        text = build_digest_text(weather, news, settings.digest_interval_minutes)
        created_at = datetime.now().isoformat(timespec="seconds")

        weather_snapshot = json.dumps(weather, ensure_ascii=False)
        news_snapshot = json.dumps(news, ensure_ascii=False)

        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO digests (created_at, text, weather_snapshot, news_snapshot)
                VALUES (?, ?, ?, ?)
                """,
                (created_at, text, weather_snapshot, news_snapshot),
            )
            conn.commit()

        settings.last_digest_at = datetime.fromisoformat(created_at)
        save_settings(settings)

        banner = "=" * 60
        print(
            f"\n{banner}\n[NewsWeatherAgent] New digest at {created_at}:\n"
            f"{banner}\n{text}\n",
            flush=True,
        )

        return {
            "created_at": created_at,
            "text": text,
            "weather": weather,
            "news": news,
        }

    except Exception as e:  # noqa: BLE001
        print(f"[NewsWeatherAgent] Error while generating digest: {e}", flush=True)
        return None


# -----------------------------
# Чтение данных для MCP
# -----------------------------

def get_latest_digest() -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, created_at, text, weather_snapshot, news_snapshot
            FROM digests
            ORDER BY id DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()

    if row is None:
        return None

    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "text": row["text"],
        "weather": json.loads(row["weather_snapshot"])
        if row["weather_snapshot"]
        else None,
        "news": json.loads(row["news_snapshot"])
        if row["news_snapshot"]
        else None,
    }


def list_digests(limit: int = 5) -> List[Dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be positive")

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, created_at, substr(text, 1, 200) AS preview
            FROM digests
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()

    return [
        {"id": r["id"], "created_at": r["created_at"], "preview": r["preview"]}
        for r in rows
    ]


def get_raw_data(hours: int = 24) -> Dict[str, Any]:
    if hours <= 0:
        raise ValueError("hours must be positive")

    cutoff = datetime.now() - timedelta(hours=hours)
    cutoff_iso = cutoff.isoformat(timespec="seconds")

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT created_at, location, temperature, wind_speed, description
            FROM weather_logs
            WHERE created_at >= ?
            ORDER BY created_at ASC
            """,
            (cutoff_iso,),
        )
        weather_rows = [dict(r) for r in cur.fetchall()]

        cur.execute(
            """
            SELECT created_at, title, url, source, score
            FROM news_logs
            WHERE created_at >= ?
            ORDER BY created_at ASC
            """,
            (cutoff_iso,),
        )
        news_rows = [dict(r) for r in cur.fetchall()]

    return {
        "from": cutoff_iso,
        "to": datetime.now().isoformat(timespec="seconds"),
        "weather_logs": weather_rows,
        "news_logs": news_rows,
    }


# -----------------------------
# Планировщик
# -----------------------------

def scheduler_loop() -> None:
    init_db()
    print("[NewsWeatherAgent] Scheduler loop started", flush=True)

    while True:
        try:
            settings = load_settings()
            now = datetime.now()

            if settings.last_digest_at is None:
                generate_and_store_digest()
            else:
                delta = now - settings.last_digest_at
                if delta >= timedelta(minutes=settings.digest_interval_minutes):
                    generate_and_store_digest()

        except Exception as e:  # noqa: BLE001
            print(f"[NewsWeatherAgent] Scheduler error: {e}", flush=True)

        time.sleep(SCHEDULER_LOOP_SLEEP_SECONDS)


def start_scheduler_thread() -> threading.Thread:
    thread = threading.Thread(target=scheduler_loop, daemon=True)
    thread.start()
    return thread
