"""Where a user's weather comes from, and what was last read for that place.

Two small tables in the shared ``assistant.sqlite3`` (schema version 5 in
:mod:`caal.user_store`), owned by this module:

* ``weather_preferences``  at most one row per user: the city they picked by
  hand, and -- separately -- the last position their *browser* offered with
  their explicit consent. The city is durable; the browser fix carries an
  expiry and is forgotten the first time it is read past it, so an exact
  position never quietly becomes a long-lived record of where somebody is.
  Both are stored rounded to :data:`COORDINATE_PRECISION` decimal places
  (about a kilometre) because that is all a forecast needs -- the precise
  coordinates a browser reports are never written down at all.
* ``weather_forecasts`` and ``weather_geocodes``  what the upstream last
  answered. A forecast entry is keyed by user *and* resolved place, so one
  user's cache can never be served to another; a city lookup is keyed by the
  query alone because it is public reference data about place names and holds
  nothing personal.

Nothing here contacts a network, and nothing here logs a coordinate, a city,
or a user id.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from typing import Any

from .user_scope import is_valid_user_id
from .user_store import UserStore

logger = logging.getLogger(__name__)

__all__ = [
    "BROWSER_LOCATION_TTL_SECONDS",
    "COORDINATE_PRECISION",
    "FORECAST_RETENTION_SECONDS",
    "MAX_PLACE_NAME_LENGTH",
    "BrowserFix",
    "CachedForecast",
    "CachedGeocode",
    "City",
    "ResolvedLocation",
    "WeatherPreference",
    "WeatherStore",
    "is_valid_latitude",
    "is_valid_longitude",
]

# About 1.1 km at the equator: enough for a forecast, and deliberately not
# enough to say which building somebody is in.
COORDINATE_PRECISION = 2
# A browser fix is a session-lifetime convenience, not a stored address.
BROWSER_LOCATION_TTL_SECONDS = 6 * 3600
MAX_BROWSER_LOCATION_TTL_SECONDS = 24 * 3600
# A stale forecast is worth keeping for a while so an upstream outage can be
# answered truthfully; beyond this it is only clutter.
FORECAST_RETENTION_SECONDS = 7 * 86400

MAX_PLACE_NAME_LENGTH = 120
MAX_TIMEZONE_LENGTH = 64
MAX_QUERY_LENGTH = 80
MAX_PAYLOAD_BYTES = 16 * 1024
_MAX_GEOCODE_BYTES = 32 * 1024

_TIMEZONE = re.compile(r"^[A-Za-z0-9_+/-]{1,64}$")
_BROWSER_LABEL = "Your current location"


def is_valid_latitude(value: object) -> bool:
    return _is_finite(value) and -90.0 <= float(value) <= 90.0  # type: ignore[arg-type]


def is_valid_longitude(value: object) -> bool:
    return _is_finite(value) and -180.0 <= float(value) <= 180.0  # type: ignore[arg-type]


def _is_finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _round(value: float) -> float:
    return round(float(value), COORDINATE_PRECISION)


def _require_user(user_id: object) -> str:
    if not is_valid_user_id(user_id):
        raise ValueError("User id has an invalid shape")
    return user_id  # type: ignore[return-value]


def _require_coordinates(latitude: object, longitude: object) -> tuple[float, float]:
    if not is_valid_latitude(latitude) or not is_valid_longitude(longitude):
        raise ValueError("Coordinates are out of range")
    return _round(latitude), _round(longitude)  # type: ignore[arg-type]


def _place_text(value: object, *, name: str, required: bool = False) -> str | None:
    """Plain, single-line, bounded place text; ``None`` when there is none."""
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return None
    if not isinstance(value, str) or not value.isprintable():
        raise ValueError(f"{name} must be plain text")
    text = " ".join(value.split())
    if not text:
        if required:
            raise ValueError(f"{name} is required")
        return None
    if len(text) > MAX_PLACE_NAME_LENGTH:
        raise ValueError(f"{name} is too long")
    return text


def _timezone_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _TIMEZONE.fullmatch(value) is None:
        raise ValueError("Timezone has an unexpected shape")
    return value


def normalize_query(value: object) -> str:
    """The key a city lookup is cached under: trimmed, collapsed, case-folded."""
    if not isinstance(value, str):
        raise ValueError("A city query must be text")
    query = " ".join(value.split()).casefold()
    if not query or len(query) > MAX_QUERY_LENGTH:
        raise ValueError("A city query must be 1 to 80 characters")
    return query


# --- values ------------------------------------------------------------------------------


@dataclass(frozen=True)
class City:
    """One place from the Open-Meteo city lookup. Never invented by a client."""

    name: str
    latitude: float
    longitude: float
    timezone: str | None = None
    country: str | None = None
    admin1: str | None = None

    def view(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": self.timezone,
            "country": self.country,
            "admin1": self.admin1,
        }

    def validated(self) -> City:
        """The same city with every field checked and the coordinates rounded."""
        latitude, longitude = _require_coordinates(self.latitude, self.longitude)
        return City(
            name=_place_text(self.name, name="City name", required=True),  # type: ignore[arg-type]
            latitude=latitude,
            longitude=longitude,
            timezone=_timezone_text(self.timezone),
            country=_place_text(self.country, name="Country"),
            admin1=_place_text(self.admin1, name="Region"),
        )

    @classmethod
    def from_view(cls, data: object) -> City | None:
        """Rebuild a city from a cached lookup row; ``None`` for anything else."""
        if not isinstance(data, dict):
            return None
        try:
            return cls(
                name=data["name"],
                latitude=data["latitude"],
                longitude=data["longitude"],
                timezone=data.get("timezone"),
                country=data.get("country"),
                admin1=data.get("admin1"),
            ).validated()
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class BrowserFix:
    """A position the browser offered, rounded, with the hour it stops counting."""

    latitude: float
    longitude: float
    updated_at: int
    expires_at: int


@dataclass(frozen=True)
class ResolvedLocation:
    """Where this user's weather is read from right now."""

    source: str
    latitude: float
    longitude: float
    label: str
    timezone: str | None = None
    country: str | None = None
    admin1: str | None = None

    def view(self) -> dict[str, Any]:
        """What a browser may see. Deliberately no coordinates: the label is enough."""
        return {
            "source": self.source,
            "label": self.label,
            "region": self.admin1,
            "country": self.country,
            "timezone": self.timezone,
        }


@dataclass(frozen=True)
class WeatherPreference:
    """One user's weather location settings, as they stand at a moment."""

    city: City | None = None
    browser: BrowserFix | None = None

    @property
    def resolved(self) -> ResolvedLocation | None:
        """The city if one was chosen, else an unexpired browser fix, else nothing."""
        if self.city is not None:
            return ResolvedLocation(
                source="city",
                latitude=self.city.latitude,
                longitude=self.city.longitude,
                label=self.city.name,
                timezone=self.city.timezone,
                country=self.city.country,
                admin1=self.city.admin1,
            )
        if self.browser is not None:
            # A generic label: the position itself is never shown back.
            return ResolvedLocation(
                source="browser",
                latitude=self.browser.latitude,
                longitude=self.browser.longitude,
                label=_BROWSER_LABEL,
            )
        return None


@dataclass(frozen=True)
class CachedForecast:
    """What the upstream last answered for one user and one place."""

    payload: dict[str, Any]
    fetched_at: int
    expires_at: int

    def is_fresh(self, now: int) -> bool:
        return int(now) < self.expires_at


@dataclass(frozen=True)
class CachedGeocode:
    """What the city lookup last answered for one query. Public reference data."""

    results: list[dict[str, Any]]
    fetched_at: int
    expires_at: int

    def is_fresh(self, now: int) -> bool:
        return int(now) < self.expires_at


# --- store -------------------------------------------------------------------------------


class WeatherStore:
    """Per-user weather preferences and the bounded caches behind them."""

    def __init__(
        self, users: UserStore, *, browser_ttl_seconds: int = BROWSER_LOCATION_TTL_SECONDS
    ) -> None:
        if isinstance(browser_ttl_seconds, bool) or not isinstance(browser_ttl_seconds, int):
            raise ValueError("browser_ttl_seconds must be a whole number of seconds")
        if not 0 < browser_ttl_seconds <= MAX_BROWSER_LOCATION_TTL_SECONDS:
            raise ValueError("browser_ttl_seconds is out of bounds")
        self._users = users
        self._browser_ttl = browser_ttl_seconds

    @property
    def browser_ttl_seconds(self) -> int:
        return self._browser_ttl

    def _connect(self) -> sqlite3.Connection:
        return self._users.connect()

    # --- preferences ---------------------------------------------------------------

    def preferences(self, user_id: object, *, now: int) -> WeatherPreference:
        """This user's city and unexpired browser fix. Expired fixes are erased."""
        owner = _require_user(user_id)
        moment = int(now)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM weather_preferences WHERE user_id = ?", (owner,)
            ).fetchone()
            if row is None:
                return WeatherPreference()
            browser = self._browser_of(row)
            if browser is not None and browser.expires_at <= moment:
                connection.execute(
                    "UPDATE weather_preferences SET browser_latitude = NULL, "
                    "browser_longitude = NULL, browser_updated_at = NULL, "
                    "browser_expires_at = NULL, updated_at = ? WHERE user_id = ?",
                    (moment, owner),
                )
                browser = None
        return WeatherPreference(city=self._city_of(row), browser=browser)

    @staticmethod
    def _city_of(row: sqlite3.Row) -> City | None:
        if row["city_name"] is None:
            return None
        return City(
            name=row["city_name"],
            latitude=row["city_latitude"],
            longitude=row["city_longitude"],
            timezone=row["city_timezone"],
            country=row["city_country"],
            admin1=row["city_admin1"],
        )

    @staticmethod
    def _browser_of(row: sqlite3.Row) -> BrowserFix | None:
        if row["browser_latitude"] is None or row["browser_expires_at"] is None:
            return None
        return BrowserFix(
            latitude=row["browser_latitude"],
            longitude=row["browser_longitude"],
            updated_at=row["browser_updated_at"] or 0,
            expires_at=row["browser_expires_at"],
        )

    def _ensure_row(self, connection: sqlite3.Connection, user_id: str, now: int) -> None:
        connection.execute(
            "INSERT INTO weather_preferences (user_id, updated_at) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO NOTHING",
            (user_id, now),
        )

    def set_city(self, user_id: object, city: City, *, now: int) -> WeatherPreference:
        """Replace this user's manual city. The caller must have validated it upstream."""
        owner = _require_user(user_id)
        if not isinstance(city, City):
            raise ValueError("A city is required")
        checked = city.validated()
        moment = int(now)
        with closing(self._connect()) as connection:
            self._ensure_row(connection, owner, moment)
            connection.execute(
                "UPDATE weather_preferences SET city_name = ?, city_latitude = ?, "
                "city_longitude = ?, city_timezone = ?, city_country = ?, city_admin1 = ?, "
                "city_updated_at = ?, updated_at = ? WHERE user_id = ?",
                (
                    checked.name,
                    checked.latitude,
                    checked.longitude,
                    checked.timezone,
                    checked.country,
                    checked.admin1,
                    moment,
                    moment,
                    owner,
                ),
            )
        logger.info("A user chose a weather city by hand")
        return self.preferences(owner, now=moment)

    def clear_city(self, user_id: object, *, now: int) -> WeatherPreference:
        owner = _require_user(user_id)
        moment = int(now)
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE weather_preferences SET city_name = NULL, city_latitude = NULL, "
                "city_longitude = NULL, city_timezone = NULL, city_country = NULL, "
                "city_admin1 = NULL, city_updated_at = NULL, updated_at = ? WHERE user_id = ?",
                (moment, owner),
            )
        return self.preferences(owner, now=moment)

    def set_browser_location(
        self, user_id: object, *, latitude: object, longitude: object, now: int
    ) -> WeatherPreference:
        """Record a consented browser fix, rounded, with a short expiry."""
        owner = _require_user(user_id)
        rounded_lat, rounded_lon = _require_coordinates(latitude, longitude)
        moment = int(now)
        with closing(self._connect()) as connection:
            self._ensure_row(connection, owner, moment)
            connection.execute(
                "UPDATE weather_preferences SET browser_latitude = ?, browser_longitude = ?, "
                "browser_updated_at = ?, browser_expires_at = ?, updated_at = ? "
                "WHERE user_id = ?",
                (
                    rounded_lat,
                    rounded_lon,
                    moment,
                    moment + self._browser_ttl,
                    moment,
                    owner,
                ),
            )
        logger.info("A user shared a browser location for weather (kept %ds)", self._browser_ttl)
        return self.preferences(owner, now=moment)

    def clear_browser_location(self, user_id: object, *, now: int) -> WeatherPreference:
        owner = _require_user(user_id)
        moment = int(now)
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE weather_preferences SET browser_latitude = NULL, "
                "browser_longitude = NULL, browser_updated_at = NULL, "
                "browser_expires_at = NULL, updated_at = ? WHERE user_id = ?",
                (moment, owner),
            )
        return self.preferences(owner, now=moment)

    # --- caches --------------------------------------------------------------------

    def cache_key(self, user_id: object, location: ResolvedLocation | None) -> str:
        """The key one user's forecast for one place is cached under."""
        owner = _require_user(user_id)
        if not isinstance(location, ResolvedLocation):
            raise ValueError("A resolved location is required")
        latitude, longitude = _require_coordinates(location.latitude, location.longitude)
        return f"{owner}|{location.source}|{latitude:.2f}|{longitude:.2f}"

    def cached_forecast(self, cache_key: str) -> CachedForecast | None:
        """The last answer for this key, fresh or stale. Never another key's."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload, fetched_at, expires_at FROM weather_forecasts "
                "WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        payload = _loads_object(row["payload"])
        if payload is None:
            return None
        return CachedForecast(
            payload=payload, fetched_at=row["fetched_at"], expires_at=row["expires_at"]
        )

    def store_forecast(
        self,
        cache_key: str,
        user_id: object,
        payload: dict[str, Any],
        *,
        now: int,
        ttl_seconds: int,
    ) -> CachedForecast:
        owner = _require_user(user_id)
        if not isinstance(payload, dict):
            raise ValueError("A forecast payload must be an object")
        body = json.dumps(payload, separators=(",", ":"))
        if len(body.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            raise ValueError("The forecast payload is too large to cache")
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive whole number of seconds")
        moment = int(now)
        expires_at = moment + ttl_seconds
        with closing(self._connect()) as connection:
            connection.execute(
                "INSERT INTO weather_forecasts (cache_key, user_id, payload, fetched_at, "
                "expires_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT(cache_key) DO UPDATE SET "
                "payload = excluded.payload, fetched_at = excluded.fetched_at, "
                "expires_at = excluded.expires_at",
                (cache_key, owner, body, moment, expires_at),
            )
            connection.execute(
                "DELETE FROM weather_forecasts WHERE fetched_at < ?",
                (moment - FORECAST_RETENTION_SECONDS,),
            )
        return CachedForecast(payload=payload, fetched_at=moment, expires_at=expires_at)

    def cached_geocode(self, query: object) -> CachedGeocode | None:
        key = normalize_query(query)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT results, fetched_at, expires_at FROM weather_geocodes WHERE query = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        results = _loads_array(row["results"])
        if results is None:
            return None
        return CachedGeocode(
            results=results, fetched_at=row["fetched_at"], expires_at=row["expires_at"]
        )

    def store_geocode(
        self, query: object, results: list[dict[str, Any]], *, now: int, ttl_seconds: int
    ) -> CachedGeocode:
        key = normalize_query(query)
        if not isinstance(results, list):
            raise ValueError("City lookup results must be a list")
        body = json.dumps(results, separators=(",", ":"))
        if len(body.encode("utf-8")) > _MAX_GEOCODE_BYTES:
            raise ValueError("The city lookup results are too large to cache")
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive whole number of seconds")
        moment = int(now)
        expires_at = moment + ttl_seconds
        with closing(self._connect()) as connection:
            connection.execute(
                "INSERT INTO weather_geocodes (query, results, fetched_at, expires_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(query) DO UPDATE SET "
                "results = excluded.results, fetched_at = excluded.fetched_at, "
                "expires_at = excluded.expires_at",
                (key, body, moment, expires_at),
            )
            connection.execute("DELETE FROM weather_geocodes WHERE expires_at < ?", (moment,))
        return CachedGeocode(results=results, fetched_at=moment, expires_at=expires_at)

    def forget_user(self, user_id: object) -> None:
        """Erase everything this module holds for one user."""
        owner = _require_user(user_id)
        with closing(self._connect()) as connection:
            connection.execute("DELETE FROM weather_preferences WHERE user_id = ?", (owner,))
            connection.execute("DELETE FROM weather_forecasts WHERE user_id = ?", (owner,))


def _loads_object(body: object) -> dict[str, Any] | None:
    if not isinstance(body, str):
        return None
    try:
        loaded = json.loads(body)
    except ValueError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _loads_array(body: object) -> list[dict[str, Any]] | None:
    if not isinstance(body, str):
        return None
    try:
        loaded = json.loads(body)
    except ValueError:
        return None
    if not isinstance(loaded, list) or not all(isinstance(item, dict) for item in loaded):
        return None
    return loaded
