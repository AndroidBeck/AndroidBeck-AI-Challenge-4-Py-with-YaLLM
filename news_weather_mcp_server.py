# news_weather_mcp_server.py
import sys
import traceback
from datetime import datetime
from typing import Optional, Literal, List, Dict, Any

import httpx
from mcp.server.fastmcp import FastMCP

# --- Open-Meteo forecast endpoint (БЕЗ geocoding) ---
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# --- Hacker News frontpage ---
HN_API_URL = "https://hn.algolia.com/api/v1/search?tags=front_page"

# Захардкоженные координаты Амстердама
DEFAULT_LAT = 52.3740
DEFAULT_LON = 4.8897

mcp = FastMCP("news_weather_server")


async def _fetch_current_weather(
    client: httpx.AsyncClient,
    lat: float,
    lon: float,
    units: Literal["metric", "imperial"] = "metric",
) -> dict:
    """
    Call Open-Meteo Forecast API for current weather at given coordinates.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
    }

    if units == "imperial":
        params.update(
            {
                "temperature_unit": "fahrenheit",
                "windspeed_unit": "mph",
            }
        )
    else:
        params.update(
            {
                "temperature_unit": "celsius",
                "windspeed_unit": "kmh",
            }
        )

    resp = await client.get(FORECAST_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if "current_weather" not in data:
        raise RuntimeError("Open-Meteo response has no 'current_weather' field")

    return data["current_weather"]


async def _fetch_news(
    client: httpx.AsyncClient,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Fetch top front-page stories from Hacker News.
    """
    resp = await client.get(HN_API_URL, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    hits = data.get("hits") or []
    hits = hits[:limit]

    items: List[Dict[str, Any]] = []
    for h in hits:
        title = h.get("title") or h.get("story_title") or "No title"
        url = h.get("url") or h.get("story_url")
        points = h.get("points", 0)
        author = h.get("author")
        created_at = h.get("created_at")

        items.append(
            {
                "title": title,
                "url": url,
                "points": points,
                "author": author,
                "created_at": created_at,
                "source": "HackerNews",
            }
        )

    return items


@mcp.tool()
async def get_news_and_weather(
    location: str = "Amsterdam",
    country_code: Optional[str] = "NL",
    units: Literal["metric", "imperial"] = "metric",
    news_limit: int = 5,
) -> dict:
    """
    Get current weather and top news headlines.

    В этой версии мы:
    - НЕ вызываем geocoding API (используем фиксированные координаты Амстердама)
    - Стараемся вернуть хотя бы частичные данные, даже если что-то упало.
    """
    fetched_at = datetime.utcnow().isoformat() + "Z"

    weather_result: Dict[str, Any] = {}
    news_result: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}

    async with httpx.AsyncClient() as client:
        # 1) Погода
        try:
            current = await _fetch_current_weather(
                client,
                DEFAULT_LAT,
                DEFAULT_LON,
                units,
            )
            weather_result = {
                "temperature": current.get("temperature"),
                "windspeed": current.get("windspeed"),
                "winddirection": current.get("winddirection"),
                "weathercode": current.get("weathercode"),
                "time": current.get("time"),
            }
        except Exception as e:
            tb = traceback.format_exc()
            print(
                "[news_weather_mcp_server] Error while fetching weather:",
                tb,
                file=sys.stderr,
                flush=True,
            )
            errors["weather"] = str(e) or "Unknown weather error (see stderr)"

        # 2) Новости
        try:
            news_result = await _fetch_news(client, news_limit)
        except Exception as e:
            tb = traceback.format_exc()
            print(
                "[news_weather_mcp_server] Error while fetching news:",
                tb,
                file=sys.stderr,
                flush=True,
            )
            errors["news"] = str(e) or "Unknown news error (see stderr)"

    # Если вообще всё умерло — отдаём общий error
    if not weather_result and not news_result:
        return {
            "error": "Failed to fetch both weather and news",
            "errors": errors,
            "location": location,
            "country_code": country_code,
            "units": units,
        }

    # Иначе возвращаем то, что получилось, плюс возможные частичные ошибки
    result: Dict[str, Any] = {
        "fetched_at": fetched_at,
        "location": {
            "query": location,
            "resolved_name": "Amsterdam (fixed coords)",
            "country_code": country_code,
            "latitude": DEFAULT_LAT,
            "longitude": DEFAULT_LON,
        },
        "units": units,
        "current_weather": weather_result or None,
        "news": news_result,
        "source": {
            "weather": "open-meteo.com (direct, no geocoding)",
            "news": "hn.algolia.com",
        },
    }
    if errors:
        result["partial_errors"] = errors

    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
