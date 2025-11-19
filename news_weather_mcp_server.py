# news_weather_mcp_server.py
import asyncio
import sys
import traceback
from datetime import datetime
from typing import Optional, Literal, List, Dict, Any

import httpx
from mcp.server.fastmcp import FastMCP

# --- Open-Meteo endpoints ---
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# --- Hacker News frontpage ---
HN_API_URL = "https://hn.algolia.com/api/v1/search?tags=front_page"

# Create the MCP server instance
mcp = FastMCP("news_weather_server")


async def _geocode_location(
    client: httpx.AsyncClient,
    location: str,
    country_code: Optional[str] = None,
) -> dict:
    """
    Use Open-Meteo Geocoding API to turn a location name into lat/lon.
    """
    params = {
        "name": location,
        "count": 1,
        "language": "en",
        "format": "json",
    }
    if country_code:
        params["country"] = country_code

    resp = await client.get(GEOCODING_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    results = data.get("results") or []
    if not results:
        raise ValueError(f"No geocoding results for location='{location}'")

    return results[0]


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
        # default: metric
        params.update(
            {
                "temperature_unit": "celsius",
                "windspeed_unit": "kmh",
            }
        )

    resp = await client.get(FORECAST_URL, params=params, timeout=10)
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
    resp = await client.get(HN_API_URL, timeout=10)
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

    Args:
        location: City or place name (e.g. "Amsterdam").
        country_code: Optional ISO 3166 country code (e.g. "NL", "US").
        units: "metric" or "imperial" for temperature & wind.
        news_limit: How many top news items to return.

    Returns:
        JSON-serializable dict with weather + news.
    """
    try:
        async with httpx.AsyncClient() as client:
            # 1) Geocode
            geo = await _geocode_location(client, location, country_code)
            lat = geo["latitude"]
            lon = geo["longitude"]
            resolved_name = geo.get("name")
            resolved_country = geo.get("country_code") or geo.get("country")

            # 2) Weather
            current = await _fetch_current_weather(client, lat, lon, units)

            # 3) News
            news_items = await _fetch_news(client, news_limit)

            return {
                "fetched_at": datetime.utcnow().isoformat() + "Z",
                "location": {
                    "query": location,
                    "resolved_name": resolved_name,
                    "country_code": resolved_country,
                    "latitude": lat,
                    "longitude": lon,
                },
                "units": units,
                "current_weather": {
                    "temperature": current.get("temperature"),
                    "windspeed": current.get("windspeed"),
                    "winddirection": current.get("winddirection"),
                    "weathercode": current.get("weathercode"),
                    "time": current.get("time"),
                },
                "news": news_items,
                "source": {
                    "weather": "open-meteo.com",
                    "news": "hn.algolia.com",
                },
            }
    except Exception as e:
        tb = traceback.format_exc()
        print(
            "[news_weather_mcp_server] Error in get_news_and_weather:",
            tb,
            file=sys.stderr,
            flush=True,
        )
        return {
            "error": str(e),
            "location": location,
            "country_code": country_code,
            "units": units,
        }


if __name__ == "__main__":
    mcp.run(transport="stdio")
