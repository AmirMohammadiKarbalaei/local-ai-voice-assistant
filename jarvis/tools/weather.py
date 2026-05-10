import requests

from jarvis.config import AssistantConfig


def schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get current weather and today's temperature. Use this for weather, "
                "temperature, rain, wind, umbrella, cold, hot, or forecast questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Optional city or place, for example Liverpool, London, Newcastle, or Tokyo.",
                    },
                    "latitude": {"type": "number", "description": "Optional latitude."},
                    "longitude": {"type": "number", "description": "Optional longitude."},
                },
                "required": [],
            },
        },
    }


def geocode_location(location: str) -> dict:
    try:
        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en", "format": "json"},
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results") or []
        if not results:
            return {"ok": False, "error": f"Could not find location: {location}"}

        place = results[0]
        display_parts = [place.get("name"), place.get("admin1"), place.get("country")]
        display_name = ", ".join(part for part in display_parts if part)

        return {
            "ok": True,
            "name": display_name or location,
            "latitude": place.get("latitude"),
            "longitude": place.get("longitude"),
            "timezone": place.get("timezone"),
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_weather(
    config: AssistantConfig,
    location: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> dict:
    display_location = location or config.user_location

    if location and (latitude is None or longitude is None):
        geo = geocode_location(location)
        if not geo.get("ok"):
            return geo

        latitude = geo["latitude"]
        longitude = geo["longitude"]
        display_location = geo["name"]

    latitude = latitude if latitude is not None else config.user_latitude
    longitude = longitude if longitude is not None else config.user_longitude

    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,"
                    "apparent_temperature,"
                    "precipitation,"
                    "rain,"
                    "wind_speed_10m"
                ),
                "daily": (
                    "temperature_2m_max,"
                    "temperature_2m_min,"
                    "precipitation_probability_max"
                ),
                "timezone": "auto",
                "forecast_days": 1,
            },
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()

        current = data.get("current", {})
        daily = data.get("daily", {})

        return {
            "ok": True,
            "location": display_location,
            "latitude": latitude,
            "longitude": longitude,
            "timezone": data.get("timezone"),
            "temperature_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "precipitation_mm": current.get("precipitation"),
            "rain_mm": current.get("rain"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "max_temp_c": (daily.get("temperature_2m_max") or [None])[0],
            "min_temp_c": (daily.get("temperature_2m_min") or [None])[0],
            "precipitation_probability_percent": (
                daily.get("precipitation_probability_max") or [None]
            )[0],
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}
