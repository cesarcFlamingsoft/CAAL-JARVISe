"""The name of the place a shared browser position falls in, and nothing finer.

When somebody deliberately shares this browser location for the weather,
JARVIS knows a position and no name for it, so the Weather widget can only say
"Your current location" -- which is truthful but gives nobody a way to notice
that the fix is wrong. This module turns that position into a coarse, public
place name -- a nearby city, its region, its country -- so a wrong location can
be seen and overridden by hand.

It reads exactly one endpoint:

* ``GET https://nominatim.openstreetmap.org/reverse`` with the position
  *already rounded to two decimals* by :mod:`caal.weather_store` (about a
  kilometre), ``zoom=10`` for a city-level answer, ``addressdetails=1``,
  ``format=jsonv2`` and ``accept-language=en``.

The request is an absolute https URL on an exact one-host allowlist, sent with
redirects disabled, a socket timeout, a response byte cap and a distinctive
User-Agent that names the application and nobody in it. No key, no cookie, no
identity: Nominatim is asked about a point and told nothing about who is
asking.

The public Nominatim usage policy allows an absolute maximum of one request a
second from one application, so every lookup in this process -- whatever the
user -- takes a slot from one global schedule; a lookup whose slot is further
away than :data:`MAX_QUEUE_WAIT_SECONDS` is skipped rather than queued, and the
widget keeps its generic label. Answers are held in memory only, for no longer
than the browser fix that prompted them, so a repeated dashboard read never
becomes a repeated upstream call and nothing new is written down anywhere.

A lookup that fails, or answers something this module cannot read, is worth
nothing at all: it returns ``None``, the caller keeps the truthful generic
label, and the weather itself is unaffected.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

from .geo_logging import install_log_redaction
from .weather_store import (
    BROWSER_LOCATION_TTL_SECONDS,
    COORDINATE_PRECISION,
    MAX_PLACE_NAME_LENGTH,
    is_valid_latitude,
    is_valid_longitude,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CACHE_TTL_SECONDS",
    "FAILURE_CACHE_TTL_SECONDS",
    "MAX_CACHED_PLACES",
    "MAX_QUEUE_WAIT_SECONDS",
    "MAX_RESPONSE_BYTES",
    "MIN_REQUEST_INTERVAL_SECONDS",
    "NOMINATIM_URL",
    "USER_AGENT",
    "NearbyPlace",
    "ReverseGeocoder",
    "is_allowed_url",
]

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
# The only host this module may contact, ever.
ALLOWED_HOSTS = frozenset(("nominatim.openstreetmap.org",))

# Distinctive enough for the operators of a free service to recognise, and
# deliberately not a person: no email, no user, no instance name.
USER_AGENT = "CAAL-JARVIS-weather/1.0 (self-hosted assistant; coarse reverse geocoding)"

# The absolute maximum the public Nominatim usage policy allows one
# application, shared by every lookup in this process.
MIN_REQUEST_INTERVAL_SECONDS = 1.0
# Longer than this and a dashboard read stops waiting: a generic label now
# beats a precise one late.
MAX_QUEUE_WAIT_SECONDS = 3.0

# A name is only ever held as long as the browser fix that prompted it.
CACHE_TTL_SECONDS = BROWSER_LOCATION_TTL_SECONDS
# A failure is remembered briefly so an outage is not asked about repeatedly.
FAILURE_CACHE_TTL_SECONDS = 600
MAX_CACHED_PLACES = 256

TIMEOUT_SECONDS = 8.0
CONNECT_TIMEOUT_SECONDS = 4.0
_MAX_TIMEOUT_SECONDS = 30.0
# A city-level reverse answer is well under a kilobyte.
MAX_RESPONSE_BYTES = 32 * 1024
_MIN_RESPONSE_BYTES = 1024

# In the order they are preferred: the nearest thing to a settlement name,
# never a street, a house number or a postcode.
_LABEL_FIELDS = (
    "city",
    "town",
    "village",
    "municipality",
    "borough",
    "hamlet",
    "suburb",
    "county",
    "state_district",
    "state",
    "region",
)
_REGION_FIELDS = ("state", "region", "state_district", "county")

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

install_log_redaction()


def is_allowed_url(url: object) -> bool:
    """True only for an absolute https URL on the exact Nominatim host allowlist."""
    if not isinstance(url, str):
        return False
    parts = urlsplit(url)
    if parts.scheme != "https" or "@" in parts.netloc:
        return False
    return parts.hostname in ALLOWED_HOSTS and parts.netloc == parts.hostname


@dataclass(frozen=True)
class NearbyPlace:
    """A coarse public place name. Never a street, a number, or a coordinate."""

    label: str
    region: str | None = None
    country: str | None = None


@dataclass(frozen=True)
class _CachedPlace:
    place: NearbyPlace | None
    expires_at: float


def _place_text(value: object) -> str | None:
    """Plain, single-line, bounded place text; ``None`` for anything else."""
    if not isinstance(value, str):
        return None
    plain = " ".join(_CONTROL.sub(" ", value).split())
    return plain[:MAX_PLACE_NAME_LENGTH] if plain else None


def _first_named(address: dict[str, Any], fields: tuple[str, ...]) -> str | None:
    for field in fields:
        found = _place_text(address.get(field))
        if found is not None:
            return found
    return None


def place_of(payload: object) -> NearbyPlace | None:
    """Reduce one Nominatim answer to a coarse place, or to nothing at all."""
    if not isinstance(payload, dict):
        return None
    address = payload.get("address")
    if not isinstance(address, dict):
        return None
    label = _first_named(address, _LABEL_FIELDS)
    if label is None:
        return None
    region = _first_named(address, _REGION_FIELDS)
    return NearbyPlace(
        label=label,
        region=None if region == label else region,
        country=_place_text(address.get("country")),
    )


class ReverseGeocoder:
    """Bounded, paced, cached reverse lookups against one exact Nominatim host.

    ``transport`` is for tests (``httpx.MockTransport``); production uses the
    default. ``clock`` and ``sleep`` are the only sources of time here.
    """

    # One free slot at a time for the whole process: two runtimes, or two
    # users, are still one caller as far as the usage policy is concerned.
    _next_slot: float = 0.0

    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = TIMEOUT_SECONDS,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        cache_ttl_seconds: int = CACHE_TTL_SECONDS,
        min_interval_seconds: float = MIN_REQUEST_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise ValueError("timeout_seconds must be a number of seconds")
        if not 0 < float(timeout_seconds) <= _MAX_TIMEOUT_SECONDS:
            raise ValueError(f"timeout_seconds must be within (0, {_MAX_TIMEOUT_SECONDS}]")
        if isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, int):
            raise ValueError("max_response_bytes must be a whole number of bytes")
        if not _MIN_RESPONSE_BYTES <= max_response_bytes <= MAX_RESPONSE_BYTES:
            raise ValueError("max_response_bytes is out of bounds")
        if isinstance(cache_ttl_seconds, bool) or not isinstance(cache_ttl_seconds, int):
            raise ValueError("cache_ttl_seconds must be a whole number of seconds")
        if not 0 < cache_ttl_seconds <= CACHE_TTL_SECONDS:
            # A name may never outlive the browser fix that prompted it.
            raise ValueError(f"cache_ttl_seconds must be within (0, {CACHE_TTL_SECONDS}]")
        if isinstance(min_interval_seconds, bool) or not isinstance(
            min_interval_seconds, (int, float)
        ):
            raise ValueError("min_interval_seconds must be a number of seconds")
        if float(min_interval_seconds) < MIN_REQUEST_INTERVAL_SECONDS:
            raise ValueError("min_interval_seconds may not be under the Nominatim policy")
        self._transport = transport
        self._timeout = float(timeout_seconds)
        self._max_bytes = int(max_response_bytes)
        self._ttl = int(cache_ttl_seconds)
        self._min_interval = float(min_interval_seconds)
        self._clock = clock
        self._sleep = sleep
        self._cache: OrderedDict[str, _CachedPlace] = OrderedDict()

    def __repr__(self) -> str:
        return (
            f"ReverseGeocoder(timeout_seconds={self._timeout}, "
            f"cache_ttl_seconds={self._ttl}, max_response_bytes={self._max_bytes})"
        )

    @property
    def cached_places(self) -> int:
        return len(self._cache)

    @classmethod
    def reset_pacing(cls) -> None:
        """Forget the global slot. For tests and for a fresh process only."""
        cls._next_slot = 0.0

    # --- the lookup ----------------------------------------------------------------

    async def nearby(self, latitude: object, longitude: object) -> NearbyPlace | None:
        """The coarse place around this position, or ``None`` if it cannot be named.

        The position must already be a position this process was given; it is
        rounded again here so nothing finer than a kilometre is ever sent.
        """
        if not is_valid_latitude(latitude) or not is_valid_longitude(longitude):
            return None
        lat = round(float(latitude), COORDINATE_PRECISION)  # type: ignore[arg-type]
        lon = round(float(longitude), COORDINATE_PRECISION)  # type: ignore[arg-type]
        key = f"{lat:.2f}|{lon:.2f}"

        now = self._clock()
        cached = self._cached(key, now)
        if cached is not None:
            return cached.place

        wait = self._reserve_slot(now)
        if wait is None:
            logger.info("A reverse lookup was skipped to stay within the usage policy")
            return None
        if wait > 0:
            await self._sleep(wait)

        place = await self._lookup(lat, lon)
        self._remember(key, place, now=self._clock())
        return place

    def _cached(self, key: str, now: float) -> _CachedPlace | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry.expires_at <= now:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return entry

    def _remember(self, key: str, place: NearbyPlace | None, *, now: float) -> None:
        ttl = self._ttl if place is not None else FAILURE_CACHE_TTL_SECONDS
        self._cache[key] = _CachedPlace(place=place, expires_at=now + ttl)
        self._cache.move_to_end(key)
        while len(self._cache) > MAX_CACHED_PLACES:
            self._cache.popitem(last=False)

    def _reserve_slot(self, now: float) -> float | None:
        """Take the next free slot and say how long to wait, or ``None`` to skip.

        Read and written without an await in between, so two concurrent
        lookups can never be handed the same slot.
        """
        start = max(now, ReverseGeocoder._next_slot)
        if start - now > MAX_QUEUE_WAIT_SECONDS:
            return None
        ReverseGeocoder._next_slot = start + self._min_interval
        return start - now

    # --- transport -----------------------------------------------------------------

    def _http(self) -> httpx.AsyncClient:
        timeout = httpx.Timeout(self._timeout, connect=min(CONNECT_TIMEOUT_SECONDS, self._timeout))
        return httpx.AsyncClient(
            transport=self._transport,
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            headers=dict([("Accept", "application/json"), ("User-Agent", USER_AGENT)]),
        )

    async def _lookup(self, latitude: float, longitude: float) -> NearbyPlace | None:
        """One bounded reverse call. A failure is nothing, never a guess."""
        if not is_allowed_url(NOMINATIM_URL):
            return None
        params = {
            "lat": f"{latitude:.2f}",
            "lon": f"{longitude:.2f}",
            "format": "jsonv2",
            "zoom": "10",
            "addressdetails": "1",
            "accept-language": "en",
        }
        try:
            async with self._http() as client:
                request = client.build_request("GET", NOMINATIM_URL, params=params)
                status, body = await self._send(client, request)
        except (httpx.HTTPError, OSError) as exc:
            logger.info("A reverse lookup could not be made (%s)", type(exc).__name__)
            return None
        if not 200 <= status < 300:
            # A redirect is a refusal here: this module follows nothing.
            logger.info("A reverse lookup was refused upstream (status %d)", status)
            return None
        try:
            loaded = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            logger.info("A reverse lookup did not answer JSON")
            return None
        place = place_of(loaded)
        if place is None:
            logger.info("A reverse lookup named no place")
        return place

    async def _send(self, client: httpx.AsyncClient, request: httpx.Request) -> tuple[int, bytes]:
        """Send one request; return the status and a body read within the byte budget."""
        response = await client.send(request, stream=True)
        try:
            declared = response.headers.get("content-length")
            if declared is not None and (not declared.isdigit() or int(declared) > self._max_bytes):
                raise _TooLargeError()
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > self._max_bytes:
                    raise _TooLargeError()
                chunks.append(chunk)
        except _TooLargeError:
            logger.info("A reverse lookup answer was over the byte cap")
            return 0, b""
        finally:
            await response.aclose()
        return response.status_code, b"".join(chunks)


class _TooLargeError(Exception):
    """The upstream answered more bytes than this module will read."""
