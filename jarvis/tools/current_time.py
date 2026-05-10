from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from jarvis.config import AssistantConfig
from jarvis.timezones import timezone_from_location


def schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Get the current date and time. Use this for questions like "
                "what time is it, what day is it, current date, or time in a city."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Optional city or country, for example London, Tokyo, New York, or China.",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "Optional IANA timezone, for example Europe/London or Asia/Tokyo.",
                    },
                },
                "required": [],
            },
        },
    }


def get_current_time(
    config: AssistantConfig,
    location: str | None = None,
    timezone: str | None = None,
) -> dict:
    tz_name = (
        timezone
        or timezone_from_location(location, config.user_timezone)
        or config.user_timezone
    )
    display_location = location or config.user_location

    try:
        now = datetime.now(ZoneInfo(tz_name))
    except ZoneInfoNotFoundError:
        return {
            "ok": False,
            "error": f"Unknown timezone: {tz_name}",
            "hint": "Ask for a city I know, or use an IANA timezone like Europe/London.",
        }

    return {
        "ok": True,
        "location": display_location,
        "timezone": tz_name,
        "date": now.strftime("%A, %d %B %Y"),
        "time_24h": now.strftime("%H:%M"),
        "time_12h": now.strftime("%I:%M %p").lstrip("0"),
        "iso": now.isoformat(timespec="seconds"),
    }
