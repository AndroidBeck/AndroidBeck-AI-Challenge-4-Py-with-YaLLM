# task14_news_weather_mcp_server.py
"""
Day 14 – MCP server: news + weather (separate tools)

Tools:
  - get_weather(location, country_code?, units?)
  - get_news(topic, language?, page_size?)

Environment:
  - NEWS_API_KEY  (for newsapi.org)
"""

import os
import sys
import datetime
from typing import Any, Dict

import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("task14-news-weather")


def _debug_log(*args: Any) -> None:
    """Simple stderr logger to not pollute tool outputs."""
    print("[task14_news_weather]", *args, file=sys.stderr)


# -----------------------------
# Weather (Open-Meteo)
# -----------------------------

AMSTERDAM_FALLBACK = {
    "name": "Amsterdam",
    "country": "Netherlands",
    "country_code": "NL",
    "latitude": 52.37403,
    "longitude": 4.88969,
}


def _geocode_location(name: str) -> Dict[str, Any]:
    """
    Use Open-Meteo geocoding to resolve a city name into coordinates.

    If the API fails or returns no results, fall back to Amsterdam and
    include a note about this in the output.
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": name,
        "count": 1,
        "language": "en",
        "format": "json",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        _debug_log(f"Geocoding error for {name!r}: {e!r}")
        _debug_log("Falling back to Amsterdam coordinates.")
        fb = dict(AMSTERDAM_FALLBACK)
        fb["note"] = f"Geocoding failed for {name!r}, using Amsterdam as fallback."
        return fb

    results = data.get("results") or []
    if not results:
        _debug_log(f"Geocoding returned no results for {name!r}. Falling back to Amsterdam.")
        fb = dict(AMSTERDAM_FALLBACK)
        fb["note"] = (
            f"Geocoding returned no results for {name!r}, "
            "using Amsterdam as fallback."
        )
        return fb

    r0 = results[0]
    loc = {
        "name": r0.get("name") or AMSTERDAM_FALLBACK["name"],
        "country": r0.get("country") or AMSTERDAM_FALLBACK["country"],
        "country_code": r0.get("country_code") or AMSTERDAM_FALLBACK["country_code"],
        "latitude": r0.get("latitude") or AMSTERDAM_FALLBACK["latitude"],
        "longitude": r0.get("longitude") or AMSTERDAM_FALLBACK["longitude"],
    }

    # If for some reason lat/lon are missing, mark fallback as well
    if loc["latitude"] is None or loc["longitude"] is None:
        _debug_log(f"Geocoding missing coords for {name!r}. Using Amsterdam fallback coords.")
        loc["latitude"] = AMSTERDAM_FALLBACK["latitude"]
        loc["longitude"] = AMSTERDAM_FALLBACK["longitude"]
        loc["note"] = (
            f"Geocoding succeeded but coordinates were missing for {name!r}, "
            "using Amsterdam fallback coordinates."
        )

    return loc


def _fetch_current_weather(lat: float, lon: float, units: str = "metric") -> Dict[str, Any]:
    url = "https://api.open-meteo.com/v1/forecast"

    if units == "imperial":
        temp_unit = "fahrenheit"
        wind_unit = "mph"
    else:
        temp_unit = "celsius"
        wind_unit = "kmh"

    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "temperature_unit": temp_unit,
        "windspeed_unit": wind_unit,
        "timezone": "auto",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    cw = data.get("current_weather") or {}
    return {
        "temperature": cw.get("temperature"),
        "windspeed": cw.get("windspeed"),
        "winddirection": cw.get("winddirection"),
        "weathercode": cw.get("weathercode"),
        "time": cw.get("time"),
    }


@mcp.tool()
def get_weather(
    location: str,
    country_code: str = "",
    units: str = "metric",
) -> Dict[str, Any]:
    """
    Get current weather for a location using Open-Meteo.

    Args:
        location: City or place name (e.g., "Amsterdam").
        country_code: Optional ISO country code (not strictly required).
        units: "metric" or "imperial".

    Returns:
        A JSON object with resolved location info and current weather.
        If geocoding fails, falls back to Amsterdam and includes a note.
    """
    try:
        _debug_log(f"get_weather called: location={location!r}, country_code={country_code!r}, units={units!r}")

        loc = _geocode_location(location)
        lat = loc["latitude"]
        lon = loc["longitude"]

        current = _fetch_current_weather(lat, lon, units=units)

        return {
            "location": {
                "query": location,
                "resolved_name": loc.get("name"),
                "country": loc.get("country"),
                "country_code": loc.get("country_code"),
                "latitude": lat,
                "longitude": lon,
                "note": loc.get("note", ""),  # will mention Amsterdam fallback if used
            },
            "units": units,
            "current_weather": current,
            "source": "open-meteo.com",
            "fetched_at": datetime.datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        _debug_log("Error in get_weather:", repr(e))
        return {
            "error": str(e),
            "location": location,
        }


# -----------------------------
# News (NewsAPI.org)
# -----------------------------

def _get_news_api_key() -> str:
    key = os.getenv("NEWS_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "NEWS_API_KEY env var is missing. "
            "Register at newsapi.org and set NEWS_API_KEY."
        )
    return key


@mcp.tool()
def get_news(
    topic: str,
    language: str = "en",
    page_size: int = 5,
) -> Dict[str, Any]:
    """
    Get recent news articles related to a topic using NewsAPI.org.

    Args:
        topic: Search query, e.g. "AI regulation Europe".
        language: News language code (e.g., "en").
        page_size: Number of articles to fetch (1–100).

    Returns:
        A JSON object with a list of simplified articles.
    """
    try:
        _debug_log(f"get_news called: topic={topic!r}, language={language!r}, page_size={page_size}")
        api_key = _get_news_api_key()

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": topic,
            "language": language,
            "pageSize": max(1, min(page_size, 100)),
            "sortBy": "publishedAt",
        }
        headers = {"Authorization": api_key}
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        articles_out = []
        for art in data.get("articles", []):
            articles_out.append(
                {
                    "title": art.get("title"),
                    "description": art.get("description"),
                    "url": art.get("url"),
                    "source": (art.get("source") or {}).get("name"),
                    "published_at": art.get("publishedAt"),
                }
            )

        return {
            "topic": topic,
            "language": language,
            "total_results": data.get("totalResults", len(articles_out)),
            "articles": articles_out,
            "fetched_at": datetime.datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        _debug_log("Error in get_news:", repr(e))
        return {
            "error": str(e),
            "topic": topic,
        }


if __name__ == "__main__":
    mcp.run()
