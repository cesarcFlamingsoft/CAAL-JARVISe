"""Fresh, user-scoped local clock that follows the weather-location policy."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from caal.user_scope import is_valid_user_id

_DEFAULT_TIMEZONE = "America/Edmonton"


def _configured_zone() -> tuple[str, ZoneInfo]:
    name = os.environ.get("TIMEZONE", _DEFAULT_TIMEZONE)
    try:
        return name, ZoneInfo(name)
    except (KeyError, ValueError, OSError):
        return _DEFAULT_TIMEZONE, ZoneInfo(_DEFAULT_TIMEZONE)


def _result(now: datetime, *, timezone_id: str, source: str) -> dict:
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    try:
        zone = ZoneInfo(timezone_id)
    except (KeyError, ValueError, OSError):
        timezone_id, zone = _configured_zone()
        source = "configured_fallback"
    local = now.astimezone(zone).replace(microsecond=0)
    return {
        "status": "ok",
        "message": (
            "This is a fresh clock read for this turn. Answer from it, never from a time "
            "mentioned earlier in the session or system prompt."
        ),
        "data": {
            "local_time": local.isoformat(),
            "timezone": timezone_id,
            "timezone_abbreviation": local.tzname(),
            "location_source": source,
        },
    }


def _runtime():
    from caal import user_api, weather_api

    return weather_api.get_weather_runtime(user_api.get_runtime())


async def _timezone_for_user(user_id: str) -> tuple[str, str] | None:
    """Use the same selected-city/browser-location precedence as weather.

    No model input selects a place. For an active browser fix, the weather client
    resolves its timezone through its existing bounded cache/upstream path.
    """
    runtime = _runtime()
    if runtime is None:
        return None
    try:
        preference = runtime.store.preferences(user_id, now=runtime.now(), purge_expired=False)
        location = preference.resolved
        if location is None:
            return None
        if location.timezone:
            return location.timezone, "weather_city"
        snapshot = await runtime.client.current(user_id)
        timezone_id = getattr(getattr(snapshot, "location", None), "timezone", None)
        if isinstance(timezone_id, str) and timezone_id:
            return timezone_id, "weather_browser"
    except Exception:
        # A clock must still answer if location/weather is temporarily unavailable.
        return None
    return None


async def current_time(*, user_id: str | None = None) -> dict:
    """Read the current time for the verified user's effective weather location."""
    resolved = (
        await _timezone_for_user(user_id)
        if user_id is not None and is_valid_user_id(user_id)
        else None
    )
    if resolved is None:
        timezone_id, _ = _configured_zone()
        source = "configured_fallback"
    else:
        timezone_id, source = resolved
    return _result(datetime.now(timezone.utc), timezone_id=timezone_id, source=source)


def current_time_for_testing(
    now: datetime, *, timezone_id: str = _DEFAULT_TIMEZONE, source: str = "weather_city"
) -> dict:
    """Inject a UTC instant and resolved timezone for deterministic tests."""
    return _result(now, timezone_id=timezone_id, source=source)
