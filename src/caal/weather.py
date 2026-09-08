"""Current weather for one user's place, from Open-Meteo, at most once an hour.

JARVIS reads two Open-Meteo endpoints and nothing else:

* ``GET https://api.open-meteo.com/v1/forecast`` with the user's resolved
  latitude and longitude, the documented ``current=`` field list and
  ``timezone=auto``;
* ``GET https://geocoding-api.open-meteo.com/v1/search`` with a bounded
  ``name``, a bounded ``count``, ``language=en`` and ``format=json``, to turn
  something a person typed into a real place.

Neither needs a key and neither is told who is asking. Both are absolute
https URLs on an exact host allowlist, sent with redirects disabled, a socket
timeout and a response byte cap, and their answers are parsed into the small
set of fields this module declares -- never forwarded, never logged.

When the place is a position the browser shared rather than a city somebody
picked, :mod:`caal.reverse_geocode` is asked -- server-side, with the same
rounded position and under the public Nominatim usage policy -- for the name
of the nearby place, so the widget can say where it thinks the user is and a
wrong fix can be seen and replaced. A city chosen by hand always wins and is
never reverse geocoded; a reverse lookup that fails changes nothing, and the
weather is read either way.

Every successful forecast is cached per user and resolved place for
:data:`CACHE_TTL_SECONDS`, so a place is asked upstream at most once per
rolling hour however many times the dashboard refreshes; concurrent readers
of the same place wait on one call rather than starting several. When the
upstream cannot be reached the last answer is returned *and said to be
stale*; with nothing cached the answer is an honest unavailable state. A city
lookup is cached for a day, keyed by the query alone.

Nothing here logs a coordinate, a city, a user, or an upstream body.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlsplit

import httpx

from .geo_logging import install_log_redaction
from .reverse_geocode import NearbyPlace, ReverseGeocoder
from .weather_store import (
    COORDINATE_PRECISION,
    MAX_QUERY_LENGTH,
    City,
    ResolvedLocation,
    WeatherPreference,
    WeatherStore,
    is_valid_latitude,
    is_valid_longitude,
    normalize_query,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CACHE_TTL_SECONDS",
    "CURRENT_FIELDS",
    "FORECAST_URL",
    "GEOCODE_CACHE_TTL_SECONDS",
    "GEOCODING_URL",
    "MAX_RESPONSE_BYTES",
    "MAX_SEARCH_RESULTS",
    "OBSERVATION_FIELDS",
    "observation_of",
    "TIMEOUT_SECONDS",
    "WEATHER_REASONS",
    "WeatherClient",
    "WeatherError",
    "WeatherSnapshot",
    "describe_code",
    "install_log_redaction",
    "is_allowed_url",
]

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
# The only two hosts this module may contact, ever.
ALLOWED_HOSTS = frozenset(("api.open-meteo.com", "geocoding-api.open-meteo.com"))

# The documented ``current=`` fields, in the order they are sent.
CURRENT_FIELDS: tuple[str, ...] = (
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "is_day",
    "precipitation",
    "rain",
    "showers",
    "snowfall",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
)

# What a reader of this module gets, and all it gets. Units are Open-Meteo's
# documented defaults: Celsius, millimetres, centimetres of snow, km/h.
OBSERVATION_FIELDS: tuple[str, ...] = (
    "observed_at",
    "temperature_c",
    "apparent_temperature_c",
    "relative_humidity_pct",
    "is_day",
    "precipitation_mm",
    "rain_mm",
    "showers_mm",
    "snowfall_cm",
    "weather_code",
    "cloud_cover_pct",
    "wind_speed_kmh",
    "wind_direction_deg",
    "wind_gusts_kmh",
)

# One upstream forecast call per user and place per rolling hour.
CACHE_TTL_SECONDS = 3600
GEOCODE_CACHE_TTL_SECONDS = 86400
_MAX_CACHE_TTL_SECONDS = 6 * 3600

TIMEOUT_SECONDS = 10.0
CONNECT_TIMEOUT_SECONDS = 5.0
_MAX_TIMEOUT_SECONDS = 30.0
# A current-conditions answer is a couple of kilobytes; a bounded city lookup
# is a few more. This is generous and still a hard stop.
MAX_RESPONSE_BYTES = 64 * 1024
_MIN_RESPONSE_BYTES = 1024
_MAX_RESPONSE_BYTES_LIMIT = 512 * 1024

# Open-Meteo documents count as 1..100; the dashboard never needs more.
MAX_SEARCH_RESULTS = 10

_USER_AGENT = "CAAL-JARVIS weather"
_LOCAL_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$")

WEATHER_REASONS: tuple[str, ...] = (
    "transport",
    "provider_refused",
    "malformed_response",
    "unknown_city",
)

# The WMO present-weather codes Open-Meteo documents for ``weather_code``. A
# code outside this table is not guessed at: it becomes no reading at all.
WMO_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

_SNOW_CODES = frozenset((71, 73, 75, 77, 85, 86))
_FREEZING_CODES = frozenset((56, 57, 66, 67))
_THUNDER_CODES = frozenset((95, 96, 99))
_WET_CODES = frozenset((51, 53, 55, 61, 63, 65, 80, 81, 82))
_FOG_CODES = frozenset((45, 48))


def describe_code(code: object) -> str | None:
    """The documented words for a WMO code, or ``None`` for anything else."""
    if isinstance(code, bool) or not isinstance(code, int):
        return None
    return WMO_CODES.get(code)


# --- keeping coordinates out of the log --------------------------------------------------

# ``httpx`` logs every request line, URL and all, at INFO, and for a browser
# fix that URL carries a position. :mod:`caal.geo_logging` redacts those
# values from the records of both place upstreams; importing this module is
# enough to have it installed.
install_log_redaction()


def is_allowed_url(url: object) -> bool:
    """True only for an absolute https URL on the exact Open-Meteo host allowlist."""
    if not isinstance(url, str):
        return False
    parts = urlsplit(url)
    if parts.scheme != "https" or "@" in parts.netloc:
        return False
    return parts.hostname in ALLOWED_HOSTS and parts.netloc == parts.hostname


class WeatherError(Exception):
    """One bounded reason a weather read failed. The message never travels."""

    def __init__(self, message: str = "The weather could not be read", *, reason: str) -> None:
        if reason not in WEATHER_REASONS:
            raise ValueError("Unknown weather failure reason")
        super().__init__(message)
        self.reason = reason


# --- bounded readings --------------------------------------------------------------------


def _number(value: object, *, low: float, high: float) -> float | None:
    """A finite measurement inside a physically possible range, else nothing."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        return None
    return number


def _integer(value: object, *, low: int, high: int) -> int | None:
    number = _number(value, low=low, high=high)
    if number is None or number != int(number):
        return None
    return int(number)


def _flag(value: object) -> bool | None:
    """Open-Meteo answers ``is_day`` as 1 or 0; anything else is not a claim."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    return None


def _local_time(value: object) -> str | None:
    if not isinstance(value, str) or _LOCAL_TIME.fullmatch(value) is None:
        return None
    return value


def _code(value: object) -> int | None:
    code = _integer(value, low=0, high=99)
    return code if code is not None and code in WMO_CODES else None


# Each declared reading: the upstream field it comes from, and the only
# values that can be true of it. Used both when the upstream answers and when
# a cached reading is read back, so nothing enters through the database that
# could not have entered through the network.
_READINGS: dict[str, tuple[str, Callable[[object], Any]]] = {
    "observed_at": ("time", _local_time),
    "temperature_c": ("temperature_2m", lambda v: _number(v, low=-100, high=100)),
    "apparent_temperature_c": ("apparent_temperature", lambda v: _number(v, low=-150, high=150)),
    "relative_humidity_pct": ("relative_humidity_2m", lambda v: _integer(v, low=0, high=100)),
    "is_day": ("is_day", _flag),
    "precipitation_mm": ("precipitation", lambda v: _number(v, low=0, high=2000)),
    "rain_mm": ("rain", lambda v: _number(v, low=0, high=2000)),
    "showers_mm": ("showers", lambda v: _number(v, low=0, high=2000)),
    "snowfall_cm": ("snowfall", lambda v: _number(v, low=0, high=1000)),
    "weather_code": ("weather_code", _code),
    "cloud_cover_pct": ("cloud_cover", lambda v: _integer(v, low=0, high=100)),
    "wind_speed_kmh": ("wind_speed_10m", lambda v: _number(v, low=0, high=600)),
    "wind_direction_deg": ("wind_direction_10m", lambda v: _integer(v, low=0, high=360)),
    "wind_gusts_kmh": ("wind_gusts_10m", lambda v: _number(v, low=0, high=900)),
}


def observation_of(current: object) -> dict[str, Any]:
    """Reduce the upstream ``current`` block to the declared fields, bounded.

    Every field is present; a value the upstream did not give, or gave in a
    form or a range that cannot be true, is ``None`` rather than a number.
    """
    block = current if isinstance(current, dict) else {}
    return {
        field: check(block.get(source)) for field, (source, check) in _READINGS.items()
    }


def reread_observation(stored: object) -> dict[str, Any]:
    """The same reduction applied to a reading read back out of the cache."""
    block = stored if isinstance(stored, dict) else {}
    return {field: check(block.get(field)) for field, (_, check) in _READINGS.items()}


def advice_for(observation: dict[str, Any]) -> str:
    """One short, practical sentence about the conditions as read. Never a forecast."""
    code = observation.get("weather_code")
    snow = observation.get("snowfall_cm") or 0
    rain = (observation.get("precipitation_mm") or 0) + (observation.get("showers_mm") or 0)
    gusts = observation.get("wind_gusts_kmh") or 0
    feels = observation.get("apparent_temperature_c")
    if code in _THUNDER_CODES:
        return "Thunderstorms about: stay under cover."
    if snow > 0 or code in _SNOW_CODES:
        return "Snow is falling: allow extra time."
    if code in _FREEZING_CODES:
        return "Freezing rain: surfaces will be icy."
    if rain > 0 or code in _WET_CODES:
        return "Rain about: take a coat."
    if gusts >= 60:
        return "Strong gusts: secure anything loose outside."
    if code in _FOG_CODES:
        return "Fog: visibility will be poor."
    if feels is not None and feels <= 0:
        return "Below freezing as it feels: wrap up warm."
    if feels is not None and feels >= 28:
        return "Hot as it feels: keep water to hand."
    return "Dry for now."


# --- what a reader gets ------------------------------------------------------------------


@dataclass(frozen=True)
class WeatherSnapshot:
    """The current conditions for one user's place, and how true they are.

    ``state`` is one of ``ok`` (read within the hour), ``stale`` (the upstream
    could not be reached and this is the last answer), ``unavailable`` (it
    could not be reached and there is no last answer) or ``no_location`` (the
    user has neither picked a city nor shared a position).
    """

    state: str
    location: ResolvedLocation | None = None
    observation: dict[str, Any] | None = None
    conditions: str | None = None
    advice: str | None = None
    cached_at: int | None = None
    expires_at: int | None = None
    reason: str | None = None

    @property
    def stale(self) -> bool:
        return self.state == "stale"

    def view(self) -> dict[str, Any]:
        """What the API answers. No coordinates, no upstream payload, ever."""
        return {
            "state": self.state,
            "stale": self.stale,
            "reason": self.reason,
            "location": None if self.location is None else self.location.view(),
            "observation": None if self.observation is None else dict(self.observation),
            "conditions": self.conditions,
            "advice": self.advice,
            "cached_at": self.cached_at,
            "expires_at": self.expires_at,
        }


def _snapshot(
    state: str,
    location: ResolvedLocation | None,
    payload: dict[str, Any] | None = None,
    *,
    cached_at: int | None = None,
    expires_at: int | None = None,
    reason: str | None = None,
) -> WeatherSnapshot:
    observation = None if payload is None else reread_observation(payload.get("observation"))
    if observation is None:
        return WeatherSnapshot(state=state, location=location, reason=reason)
    return WeatherSnapshot(
        state=state,
        location=location,
        observation=observation,
        conditions=describe_code(observation["weather_code"]),
        advice=advice_for(observation),
        cached_at=cached_at,
        expires_at=expires_at,
        reason=reason,
    )


# --- the client --------------------------------------------------------------------------


class WeatherClient:
    """Bounded Open-Meteo reads for one user's place, capped to one call an hour.

    ``transport`` is for tests (``httpx.MockTransport``); production uses the
    default. ``clock`` decides freshness and is the only source of time here.
    """

    def __init__(
        self,
        store: WeatherStore,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = TIMEOUT_SECONDS,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        cache_ttl_seconds: int = CACHE_TTL_SECONDS,
        geocode_ttl_seconds: int = GEOCODE_CACHE_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
        reverse_geocoder: ReverseGeocoder | None = None,
    ) -> None:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise ValueError("timeout_seconds must be a number of seconds")
        if not 0 < float(timeout_seconds) <= _MAX_TIMEOUT_SECONDS:
            raise ValueError(f"timeout_seconds must be within (0, {_MAX_TIMEOUT_SECONDS}]")
        if isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, int):
            raise ValueError("max_response_bytes must be a whole number of bytes")
        if not _MIN_RESPONSE_BYTES <= max_response_bytes <= _MAX_RESPONSE_BYTES_LIMIT:
            raise ValueError("max_response_bytes is out of bounds")
        for name, ttl, upper in (
            ("cache_ttl_seconds", cache_ttl_seconds, _MAX_CACHE_TTL_SECONDS),
            ("geocode_ttl_seconds", geocode_ttl_seconds, 7 * 86400),
        ):
            if isinstance(ttl, bool) or not isinstance(ttl, int) or not 0 < ttl <= upper:
                raise ValueError(f"{name} must be within (0, {upper}] seconds")
        self._store = store
        self._transport = transport
        self._timeout = float(timeout_seconds)
        self._max_bytes = int(max_response_bytes)
        self._ttl = int(cache_ttl_seconds)
        self._geocode_ttl = int(geocode_ttl_seconds)
        self._clock = clock
        # One in-flight upstream call per cache key: readers of the same place
        # wait rather than each starting a call of their own.
        self._locks: dict[str, asyncio.Lock] = {}
        # Turns a shared browser position into a coarse place name, so a wrong
        # position can be seen. Never asked about a city somebody picked.
        self._reverse = reverse_geocoder or ReverseGeocoder(
            transport=transport, clock=clock
        )

    def __repr__(self) -> str:
        return (
            f"WeatherClient(timeout_seconds={self._timeout}, "
            f"cache_ttl_seconds={self._ttl}, max_response_bytes={self._max_bytes})"
        )

    @property
    def store(self) -> WeatherStore:
        return self._store

    @property
    def cache_ttl_seconds(self) -> int:
        return self._ttl

    def now(self) -> int:
        return int(self._clock())

    def _lock_for(self, key: str) -> asyncio.Lock:
        # Created and read only from the event loop thread, so no await may
        # come between the lookup and the insert.
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    # --- transport -----------------------------------------------------------------

    def _http(self) -> httpx.AsyncClient:
        timeout = httpx.Timeout(self._timeout, connect=min(CONNECT_TIMEOUT_SECONDS, self._timeout))
        return httpx.AsyncClient(
            transport=self._transport,
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            headers=dict([("Accept", "application/json"), ("User-Agent", _USER_AGENT)]),
        )

    async def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """One bounded GET at an allowlisted Open-Meteo host. Redirects are refused."""
        if not is_allowed_url(url):
            raise WeatherError("Refusing a request outside Open-Meteo", reason="transport")
        async with self._http() as client:
            request = client.build_request("GET", url, params=params)
            status, body = await self._send(client, request)
        if 300 <= status < 400:
            raise WeatherError("The upstream redirected", reason="malformed_response")
        if not 200 <= status < 300:
            raise WeatherError("The upstream refused the read", reason="provider_refused")
        try:
            loaded = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise WeatherError(
                "The upstream did not answer JSON", reason="malformed_response"
            ) from exc
        if not isinstance(loaded, dict):
            raise WeatherError(
                "The upstream answered an unexpected shape", reason="malformed_response"
            )
        return loaded

    async def _send(self, client: httpx.AsyncClient, request: httpx.Request) -> tuple[int, bytes]:
        """Send one request; return the status and a body read within the byte budget."""
        try:
            response = await client.send(request, stream=True)
        except (httpx.HTTPError, OSError) as exc:
            raise WeatherError(
                f"Transport failure: {type(exc).__name__}", reason="transport"
            ) from exc
        try:
            declared = response.headers.get("content-length")
            if declared is not None and (not declared.isdigit() or int(declared) > self._max_bytes):
                raise WeatherError("The upstream answer is too large", reason="malformed_response")
            chunks: list[bytes] = []
            total = 0
            try:
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > self._max_bytes:
                        raise WeatherError(
                            "The upstream answer is too large", reason="malformed_response"
                        )
                    chunks.append(chunk)
            except (httpx.HTTPError, OSError) as exc:
                raise WeatherError(
                    f"Transport failure: {type(exc).__name__}", reason="transport"
                ) from exc
        finally:
            await response.aclose()
        return response.status_code, b"".join(chunks)


    # --- the forecast --------------------------------------------------------------

    async def current(self, user_id: str) -> WeatherSnapshot:
        """This user's current conditions: cached within the hour, truthful otherwise."""
        now = self.now()
        preference = self._store.preferences(user_id, now=now)
        location = preference.resolved
        if location is None:
            return WeatherSnapshot(state="no_location")

        key = self._store.cache_key(user_id, location)
        fresh = self._store.cached_forecast(key)
        if fresh is not None and fresh.is_fresh(now):
            return await self._named(self._cached_snapshot(location, fresh, "ok"))

        async with self._lock_for(key):
            # Another reader of the same place may have filled the cache while
            # this one waited; a second upstream call would be for nothing.
            now = self.now()
            cached = self._store.cached_forecast(key)
            if cached is not None and cached.is_fresh(now):
                return await self._named(self._cached_snapshot(location, cached, "ok"))
            try:
                payload = await self._fetch_forecast(location)
            except WeatherError as exc:
                logger.info("The weather upstream could not be read (%s)", exc.reason)
                if cached is not None:
                    stale = self._cached_snapshot(location, cached, "stale", reason=exc.reason)
                    return await self._named(stale)
                return await self._named(
                    WeatherSnapshot(state="unavailable", location=location, reason=exc.reason)
                )
            entry = self._store.store_forecast(
                key, user_id, payload, now=now, ttl_seconds=self._ttl
            )
        return await self._named(self._cached_snapshot(location, entry, "ok"))

    async def _fetch_forecast(self, location: ResolvedLocation) -> dict[str, Any]:
        """One documented forecast call. What is kept is the reduced reading only."""
        data = await self._get_json(
            FORECAST_URL,
            {
                "latitude": f"{location.latitude:.2f}",
                "longitude": f"{location.longitude:.2f}",
                "current": ",".join(CURRENT_FIELDS),
                "timezone": "auto",
            },
        )
        current = data.get("current")
        if not isinstance(current, dict):
            raise WeatherError(
                "The upstream answered no current block", reason="malformed_response"
            )
        return {
            "observation": observation_of(current),
            "timezone": _timezone_of(data.get("timezone")),
        }

    def _cached_snapshot(
        self,
        location: ResolvedLocation,
        entry: Any,
        state: str,
        *,
        reason: str | None = None,
    ) -> WeatherSnapshot:
        payload = entry.payload
        # A browser fix has no place name of its own; the timezone the upstream
        # resolved is the one safe, coarse thing that can be said about it.
        timezone = location.timezone or payload.get("timezone")
        placed = ResolvedLocation(
            source=location.source,
            latitude=location.latitude,
            longitude=location.longitude,
            label=location.label,
            timezone=timezone if isinstance(timezone, str) else None,
            country=location.country,
            admin1=location.admin1,
        )
        return _snapshot(
            state,
            placed,
            payload,
            cached_at=entry.fetched_at,
            expires_at=entry.expires_at,
            reason=reason,
        )

    async def _named(self, snapshot: WeatherSnapshot) -> WeatherSnapshot:
        """Give a shared browser position the name of the place it falls in.

        Only a browser fix is ever looked up: a city somebody chose already
        carries the name Open-Meteo returned for it, and that choice wins. If
        the lookup cannot name the place -- it is unreachable, it is over the
        usage policy, or it answers something unreadable -- the generic label
        stands, which is truthful, and the reading itself is untouched.
        """
        location = snapshot.location
        if location is None or location.source != "browser":
            return snapshot
        place: NearbyPlace | None = await self._reverse.nearby(
            location.latitude, location.longitude
        )
        if place is None:
            return snapshot
        return replace(
            snapshot,
            location=replace(
                location,
                label=place.label,
                admin1=place.region or location.admin1,
                country=place.country or location.country,
            ),
        )

    # --- the city lookup -----------------------------------------------------------

    async def search_cities(self, query: object, count: object = 5) -> list[City]:
        """Places matching what someone typed, bounded, cached for a day.

        The query is bounded before anything is sent and the upstream is always
        asked for the same maximum, so however many results a caller wants,
        one place name is one cache entry and one upstream call per day.
        """
        typed = _display_query(query)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("count must be a whole number of results, 1 or more")
        wanted = min(count, MAX_SEARCH_RESULTS)
        key = normalize_query(typed)
        now = self.now()

        cached = self._store.cached_geocode(key)
        if cached is not None and cached.is_fresh(now):
            rows = cached.results
        else:
            # A failed lookup raises rather than being cached as "no such place".
            rows = await self._fetch_cities(typed)
            self._store.store_geocode(key, rows, now=now, ttl_seconds=self._geocode_ttl)

        cities = [City.from_view(row) for row in rows]
        return [city for city in cities if city is not None][:wanted]

    async def _fetch_cities(self, name: str) -> list[dict[str, Any]]:
        """One documented city lookup, reduced before it is ever stored."""
        data = await self._get_json(
            GEOCODING_URL,
            {
                "name": name,
                "count": str(MAX_SEARCH_RESULTS),
                "language": "en",
                "format": "json",
            },
        )
        raw = data.get("results")
        rows = raw[:MAX_SEARCH_RESULTS] if isinstance(raw, list) else []
        found: list[dict[str, Any]] = []
        for row in rows:
            city = City.from_view(row)
            if city is not None:
                found.append(city.view())
        logger.info("A city lookup matched %d place(s)", len(found))
        return found

    async def choose_city(
        self, user_id: str, *, name: object, latitude: object, longitude: object
    ) -> WeatherPreference:
        """Save a city the *lookup* returned. A place it did not is refused.

        The client names a candidate and its coordinates; nothing it says about
        the place is kept. The label, region, country and timezone stored are
        the ones Open-Meteo returned for the matching row, so a browser can
        never make JARVIS show one place under another place's name.
        """
        if not is_valid_latitude(latitude) or not is_valid_longitude(longitude):
            raise ValueError("Coordinates are out of range")
        target = (
            round(float(latitude), COORDINATE_PRECISION),  # type: ignore[arg-type]
            round(float(longitude), COORDINATE_PRECISION),  # type: ignore[arg-type]
        )
        for city in await self.search_cities(name, count=MAX_SEARCH_RESULTS):
            if (city.latitude, city.longitude) == target:
                now = self.now()
                self._store.set_city(user_id, city, now=now)
                # A hand-picked city replaces the browser fix outright rather
                # than leaving a position lying around that nothing reads.
                return self._store.clear_browser_location(user_id, now=now)
        raise WeatherError("No such place in the city lookup", reason="unknown_city")


def _display_query(value: object) -> str:
    """What is sent upstream: trimmed and collapsed, in the case it was typed."""
    if not isinstance(value, str):
        raise ValueError("A city query must be text")
    typed = " ".join(value.split())
    if not typed or len(typed) > MAX_QUERY_LENGTH:
        raise ValueError(f"A city query must be 1 to {MAX_QUERY_LENGTH} characters")
    return typed


def _timezone_of(value: object) -> str | None:
    if not isinstance(value, str) or not 0 < len(value) <= 64:
        return None
    if not all(ch.isalnum() or ch in "_+-/" for ch in value):
        return None
    return value
