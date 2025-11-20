import asyncio
import sys
import traceback
from typing import Optional, Literal

import httpx
from mcp.server.fastmcp import FastMCP

# --- Open-Meteo endpoints ---
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Create the MCP server instance
mcp = FastMCP("weather_server")


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
        # 'country' param filters by country code (e.g. 'NL', 'US')
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


@mcp.tool()
async def get_current_weather(
    location: str,
    country_code: Optional[str] = None,
    units: Literal["metric", "imperial"] = "metric",
) -> dict:
    """
    Get current weather for a given location using the Open-Meteo API.

    Args:
        location: City or place name (e.g. "Amsterdam", "New York").
        country_code: Optional ISO 3166 country code (e.g. "NL", "US")
                      to disambiguate common city names.
        units: "metric" or "imperial" for temperature & wind.

    Returns:
        A JSON-serializable dict with resolved location and current weather,
        or an error description.
    """
    try:
        async with httpx.AsyncClient() as client:
            geo = await _geocode_location(client, location, country_code)

            lat = geo["latitude"]
            lon = geo["longitude"]
            resolved_name = geo.get("name")
            resolved_country = geo.get("country_code") or geo.get("country")

            current = await _fetch_current_weather(client, lat, lon, units)

            return {
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
                "source": "open-meteo.com",
            }
    except Exception as e:
        # Print full traceback to stderr (visible if you run the server directly)
        tb = traceback.format_exc()
        print(
            "[weather_mcp_server] Error in get_current_weather:",
            tb,
            file=sys.stderr,
            flush=True,
        )
        # Return error details to the client so you can see them from the agent
        return {
            "error": str(e),
            "location": location,
            "country_code": country_code,
            "units": units,
        }


if __name__ == "__main__":
    mcp.run(transport="stdio")
