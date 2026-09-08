"""The dashboard's weather: one user's place, and the conditions there.

``GET /users/me/weather``
    the current conditions for wherever this user's weather is read from --
    the city they picked, else the position their browser offered with their
    consent -- as a bounded reading, the words for it, one practical
    sentence, and how true it is: read within the hour, stale after an
    upstream outage, unavailable, or no place set at all. For a browser
    position the ``label``, ``region`` and ``country`` are the nearby place a
    server-side reverse lookup resolved, so a wrong position can be seen and
    overridden; when that lookup cannot name it, the label stays the generic
    "Your current location" and the reading is unaffected. The position
    itself is never in the answer.
``GET /users/me/weather/cities?q=&count=``
    places matching what someone typed, bounded and reduced to a name, a
    region, a country, a timezone and coordinates. Cached upstream for a day.
``PUT`` / ``DELETE /users/me/weather/city``
    choose one of those places, or go back to the browser position. A place
    the lookup did not return is refused: the label JARVIS shows always comes
    from Open-Meteo's own data, never from what a browser claimed.
``PUT`` / ``DELETE /users/me/weather/browser-location``
    record or forget a position the browser offered. The request must carry
    an explicit ``consent: true`` and coordinates that could be real; the
    position is stored rounded, expires within hours, and is never echoed
    back or logged.

Every route lives under ``/users/me`` so it inherits the identity boundary of
:mod:`caal.user_api`: a single-use ``caal-backend`` principal from the BFF
names the user, the user is loaded from the database on every call, and the
identity middleware makes every response uncacheable and free of the app-wide
CORS policy. No response here carries a coordinate, an upstream body, or
anything belonging to another user.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt

from . import user_api
from .user_api import CurrentUser, IdentityRuntime, require_user, throttle_mutation
from .weather import MAX_SEARCH_RESULTS, WeatherClient, WeatherError
from .weather_store import MAX_QUERY_LENGTH, WeatherPreference, WeatherStore

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_SEARCH_COUNT",
    "WeatherRuntime",
    "get_weather_runtime",
    "require_weather",
    "reset_weather_runtime",
    "router",
]

# Open-Meteo documents ``count`` as 1..100. A caller may ask within that
# range; what is actually requested and answered is capped much lower.
MAX_SEARCH_COUNT = 100

_NOT_CONFIGURED = "Multi-user identity is not configured on this CAAL backend."
_UPSTREAM_UNAVAILABLE = "weather_upstream_unavailable"
# Starlette renamed this constant; the number is the stable spelling.
UNPROCESSABLE = 422


# --- runtime -----------------------------------------------------------------------------


class WeatherRuntime:
    """The identity runtime plus this user's weather store and bounded client.

    Tests build one directly with a scripted transport; production builds one
    lazily per identity runtime. There is nothing to configure: Open-Meteo
    needs no key, so weather works wherever multi-user identity does.
    """

    def __init__(
        self,
        identity: IdentityRuntime,
        *,
        store: WeatherStore | None = None,
        client: WeatherClient | None = None,
    ) -> None:
        self.identity = identity
        self.store = store or WeatherStore(identity.store)
        self.client = client or WeatherClient(self.store, clock=lambda: float(identity.now()))

    def now(self) -> int:
        return self.identity.now()


_runtime_lock = threading.Lock()
_cached: tuple[IdentityRuntime, WeatherRuntime] | None = None


def get_weather_runtime(
    identity: IdentityRuntime | None = Depends(user_api.get_runtime),
) -> WeatherRuntime | None:
    """The process-wide runtime, or ``None`` while multi-user identity is unconfigured."""
    global _cached
    if identity is None:
        return None
    with _runtime_lock:
        if _cached is None or _cached[0] is not identity:
            _cached = (identity, WeatherRuntime(identity))
        return _cached[1]


def reset_weather_runtime() -> None:
    global _cached
    with _runtime_lock:
        _cached = None


def require_weather(
    runtime: WeatherRuntime | None = Depends(get_weather_runtime),
) -> WeatherRuntime:
    if runtime is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_NOT_CONFIGURED)
    return runtime


# --- schemas -----------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


# A coordinate is a number a browser or a lookup produced, never text that
# happens to look like one, and never a boolean.
Latitude = StrictFloat | StrictInt
Longitude = StrictFloat | StrictInt


class LocationResponse(_Strict):
    """Where the weather is read from. Deliberately without coordinates."""

    source: str
    label: str
    region: str | None
    country: str | None
    timezone: str | None


class ObservationResponse(_Strict):
    """One reading, in Open-Meteo's documented default units."""

    observed_at: str | None
    temperature_c: float | None
    apparent_temperature_c: float | None
    relative_humidity_pct: int | None
    is_day: bool | None
    precipitation_mm: float | None
    rain_mm: float | None
    showers_mm: float | None
    snowfall_cm: float | None
    weather_code: int | None
    cloud_cover_pct: int | None
    wind_speed_kmh: float | None
    wind_direction_deg: int | None
    wind_gusts_kmh: float | None


class WeatherResponse(_Strict):
    generated_at: int
    state: str
    stale: bool
    reason: str | None
    location: LocationResponse | None
    observation: ObservationResponse | None
    conditions: str | None
    advice: str | None
    cached_at: int | None
    expires_at: int | None


class CityResponse(_Strict):
    """One place from the lookup. The coordinates are the lookup's own."""

    name: str
    latitude: float
    longitude: float
    timezone: str | None
    country: str | None
    region: str | None


class CitySearchResponse(_Strict):
    query: str
    results: list[CityResponse]


class SavedCityResponse(_Strict):
    name: str
    region: str | None
    country: str | None
    timezone: str | None


class BrowserFixResponse(_Strict):
    """When the browser position was offered and when it stops counting. No position."""

    updated_at: int
    expires_at: int


class WeatherLocationResponse(_Strict):
    source: str | None
    label: str | None
    city: SavedCityResponse | None
    browser: BrowserFixResponse | None


class ChooseCityRequest(_Strict):
    """A candidate from the lookup. Every label is taken from the lookup, not from here."""

    name: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    latitude: Latitude = Field(ge=-90, le=90)
    longitude: Longitude = Field(ge=-180, le=180)


class BrowserLocationRequest(_Strict):
    """A position the browser offered, with the user's explicit say-so."""

    consent: StrictBool
    latitude: Latitude = Field(ge=-90, le=90)
    longitude: Longitude = Field(ge=-180, le=180)


# --- helpers -----------------------------------------------------------------------------


def _location_view(preference: WeatherPreference) -> WeatherLocationResponse:
    resolved = preference.resolved
    city = preference.city
    browser = preference.browser
    return WeatherLocationResponse(
        source=None if resolved is None else resolved.source,
        label=None if resolved is None else resolved.label,
        city=(
            None
            if city is None
            else SavedCityResponse(
                name=city.name,
                region=city.admin1,
                country=city.country,
                timezone=city.timezone,
            )
        ),
        browser=(
            None
            if browser is None
            else BrowserFixResponse(
                updated_at=browser.updated_at, expires_at=browser.expires_at
            )
        ),
    )


def _upstream_failure(exc: WeatherError) -> HTTPException:
    """One bounded refusal per failure. The upstream's own words never travel."""
    if exc.reason == "unknown_city":
        return HTTPException(
            status_code=UNPROCESSABLE, detail="unknown_city"
        )
    logger.info("A weather lookup could not be answered (%s)", exc.reason)
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_UPSTREAM_UNAVAILABLE
    )


# --- routes ------------------------------------------------------------------------------

router = APIRouter(tags=["weather"])


@router.get("/users/me/weather", response_model=WeatherResponse)
async def current_weather(
    user: CurrentUser = Depends(require_user),
    runtime: WeatherRuntime = Depends(require_weather),
) -> WeatherResponse:
    snapshot = await runtime.client.current(user.profile.user_id)
    view: dict[str, Any] = snapshot.view()
    observation = view.pop("observation")
    location = view.pop("location")
    return WeatherResponse(
        generated_at=runtime.now(),
        location=None if location is None else LocationResponse(**location),
        observation=None if observation is None else ObservationResponse(**observation),
        **view,
    )


@router.get("/users/me/weather/location", response_model=WeatherLocationResponse)
async def read_location(
    user: CurrentUser = Depends(require_user),
    runtime: WeatherRuntime = Depends(require_weather),
) -> WeatherLocationResponse:
    """Where this user has said their weather should come from. No reading is taken."""
    preference = runtime.store.preferences(user.profile.user_id, now=runtime.now())
    return _location_view(preference)


@router.get("/users/me/weather/cities", response_model=CitySearchResponse)
async def search_cities(
    q: str = Query(min_length=1, max_length=MAX_QUERY_LENGTH),
    count: int = Query(default=5, ge=1, le=MAX_SEARCH_COUNT),
    user: CurrentUser = Depends(require_user),
    runtime: WeatherRuntime = Depends(require_weather),
) -> CitySearchResponse:
    typed = " ".join(q.split())
    if not typed:
        raise HTTPException(status_code=UNPROCESSABLE, detail="invalid")
    try:
        found = await runtime.client.search_cities(typed, count=min(count, MAX_SEARCH_RESULTS))
    except ValueError as exc:
        raise HTTPException(
            status_code=UNPROCESSABLE, detail="invalid"
        ) from exc
    except WeatherError as exc:
        raise _upstream_failure(exc) from exc
    return CitySearchResponse(
        query=typed,
        results=[
            CityResponse(
                name=city.name,
                latitude=city.latitude,
                longitude=city.longitude,
                timezone=city.timezone,
                country=city.country,
                region=city.admin1,
            )
            for city in found
        ],
    )


@router.put("/users/me/weather/city", response_model=WeatherLocationResponse)
async def choose_city(
    body: ChooseCityRequest,
    user: CurrentUser = Depends(throttle_mutation),
    runtime: WeatherRuntime = Depends(require_weather),
) -> WeatherLocationResponse:
    try:
        preference = await runtime.client.choose_city(
            user.profile.user_id,
            name=body.name,
            latitude=body.latitude,
            longitude=body.longitude,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=UNPROCESSABLE, detail="invalid"
        ) from exc
    except WeatherError as exc:
        raise _upstream_failure(exc) from exc
    return _location_view(preference)


@router.delete("/users/me/weather/city", response_model=WeatherLocationResponse)
async def clear_city(
    user: CurrentUser = Depends(throttle_mutation),
    runtime: WeatherRuntime = Depends(require_weather),
) -> WeatherLocationResponse:
    preference = runtime.store.clear_city(user.profile.user_id, now=runtime.now())
    return _location_view(preference)


@router.put("/users/me/weather/browser-location", response_model=WeatherLocationResponse)
async def set_browser_location(
    body: BrowserLocationRequest,
    user: CurrentUser = Depends(throttle_mutation),
    runtime: WeatherRuntime = Depends(require_weather),
) -> WeatherLocationResponse:
    # Consent is a decision the person made in the browser, not a default this
    # route may assume: only a literal true is a yes.
    if body.consent is not True:
        raise HTTPException(
            status_code=UNPROCESSABLE, detail="consent_required"
        )
    try:
        preference = runtime.store.set_browser_location(
            user.profile.user_id,
            latitude=body.latitude,
            longitude=body.longitude,
            now=runtime.now(),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=UNPROCESSABLE, detail="invalid"
        ) from exc
    return _location_view(preference)


@router.delete("/users/me/weather/browser-location", response_model=WeatherLocationResponse)
async def clear_browser_location(
    user: CurrentUser = Depends(throttle_mutation),
    runtime: WeatherRuntime = Depends(require_weather),
) -> WeatherLocationResponse:
    preference = runtime.store.clear_browser_location(user.profile.user_id, now=runtime.now())
    return _location_view(preference)
