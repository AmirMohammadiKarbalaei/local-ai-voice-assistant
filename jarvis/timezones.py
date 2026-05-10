import re


LOCATION_TIMEZONE_ALIASES = {
    # UK
    "uk": "Europe/London",
    "united kingdom": "Europe/London",
    "england": "Europe/London",
    "london": "Europe/London",
    "newcastle": "Europe/London",
    "newcastle upon tyne": "Europe/London",
    "liverpool": "Europe/London",
    "bath": "Europe/London",
    "manchester": "Europe/London",

    # Europe
    "paris": "Europe/Paris",
    "france": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "germany": "Europe/Berlin",
    "rome": "Europe/Rome",
    "italy": "Europe/Rome",
    "madrid": "Europe/Madrid",
    "spain": "Europe/Madrid",
    "amsterdam": "Europe/Amsterdam",
    "netherlands": "Europe/Amsterdam",

    # US / Canada
    "new york": "America/New_York",
    "nyc": "America/New_York",
    "washington": "America/New_York",
    "washington dc": "America/New_York",
    "chicago": "America/Chicago",
    "texas": "America/Chicago",
    "los angeles": "America/Los_Angeles",
    "la": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles",
    "seattle": "America/Los_Angeles",
    "toronto": "America/Toronto",
    "vancouver": "America/Vancouver",

    # Asia-Pacific
    "china": "Asia/Shanghai",
    "beijing": "Asia/Shanghai",
    "shanghai": "Asia/Shanghai",
    "hong kong": "Asia/Hong_Kong",
    "taiwan": "Asia/Taipei",
    "taipei": "Asia/Taipei",
    "tokyo": "Asia/Tokyo",
    "japan": "Asia/Tokyo",
    "seoul": "Asia/Seoul",
    "korea": "Asia/Seoul",
    "singapore": "Asia/Singapore",
    "sydney": "Australia/Sydney",
    "melbourne": "Australia/Melbourne",
    "australia": "Australia/Sydney",

    # Middle East
    "dubai": "Asia/Dubai",
    "uae": "Asia/Dubai",
    "istanbul": "Europe/Istanbul",
    "turkey": "Europe/Istanbul",
}


def normalise_location_key(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def timezone_from_location(location: str | None, default_timezone: str) -> str | None:
    if not location:
        return default_timezone

    location = location.strip()

    # If the model passes an IANA timezone as location, accept it.
    if "/" in location:
        return location

    key = normalise_location_key(location)
    if key in LOCATION_TIMEZONE_ALIASES:
        return LOCATION_TIMEZONE_ALIASES[key]

    # Partial match, e.g. "Newcastle UK" -> "newcastle".
    for alias, timezone in LOCATION_TIMEZONE_ALIASES.items():
        if alias in key:
            return timezone

    return None
