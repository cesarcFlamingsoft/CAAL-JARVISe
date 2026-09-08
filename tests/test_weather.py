"""The per-user weather store and the bounded Open-Meteo client.

Nothing here talks to Open-Meteo: every upstream conversation goes through an
injected ``httpx`` transport that records exactly what was asked for. What is
being pinned down is the contract the dashboard depends on -- one manual city
that survives a restart, a browser fix that does not, at most one forecast
call per resolved user location per rolling hour, a truthful stale answer when
the upstream is down, and bounded requests to two exact https hosts.
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest

from caal import profile_crypto
from caal.profile_crypto import KeyRing
from caal.reverse_geocode import NOMINATIM_URL, ReverseGeocoder
from caal.user_store import MEMBER, Actor, UserStore
from caal.weather import (
    CACHE_TTL_SECONDS,
    CURRENT_FIELDS,
    FORECAST_URL,
    GEOCODE_CACHE_TTL_SECONDS,
    GEOCODING_URL,
    MAX_RESPONSE_BYTES,
    MAX_SEARCH_RESULTS,
    OBSERVATION_FIELDS,
    WeatherClient,
    WeatherError,
    is_allowed_url,
)
from caal.weather_store import (
    BROWSER_LOCATION_TTL_SECONDS,
    COORDINATE_PRECISION,
    City,
    WeatherStore,
)

NOW = 1_700_000_000  # 2023-11-14T22:13:20Z

PARIS = City(
    name="Paris",
    latitude=48.8566,
    longitude=2.3522,
    timezone="Europe/Paris",
    country="France",
    admin1="Ile-de-France",
)
TOKYO = City(
    name="Tokyo",
    latitude=35.6895,
    longitude=139.6917,
    timezone="Asia/Tokyo",
    country="Japan",
    admin1="Tokyo",
)


@pytest.fixture
def users(tmp_path) -> UserStore:
    keyring = KeyRing.from_env(profile_crypto.generate_key_material(version=1))
    return UserStore(tmp_path / "assistant.sqlite3", keyring=keyring)


@pytest.fixture
def store(users: UserStore) -> WeatherStore:
    return WeatherStore(users)


def _user(users: UserStore, email: str) -> str:
    return users.create_user(
        email=email,
        display_name=email.split("@")[0],
        role=MEMBER,
        actor=Actor.system(),
        now=NOW,
    ).user_id


@pytest.fixture
def ana(users: UserStore) -> str:
    return _user(users, "ana@example.com")


@pytest.fixture
def bo(users: UserStore) -> str:
    return _user(users, "bo@example.com")


# --- the manual city ---------------------------------------------------------------------


def test_no_preference_resolves_to_nothing(store: WeatherStore, ana: str):
    preference = store.preferences(ana, now=NOW)
    assert preference.city is None
    assert preference.browser is None
    assert preference.resolved is None


def test_a_saved_city_is_durable_and_rounded(users: UserStore, store: WeatherStore, ana: str):
    store.set_city(ana, PARIS, now=NOW)

    # A brand new store over the same database: the city survives a restart.
    reopened = WeatherStore(users).preferences(ana, now=NOW + 400 * 86400)
    assert reopened.city is not None
    assert reopened.city.name == "Paris"
    assert reopened.city.country == "France"
    assert reopened.city.timezone == "Europe/Paris"
    # Stored at the precision the forecast needs and no more.
    assert reopened.city.latitude == round(PARIS.latitude, COORDINATE_PRECISION)
    assert reopened.city.longitude == round(PARIS.longitude, COORDINATE_PRECISION)
    resolved = reopened.resolved
    assert resolved is not None
    assert resolved.source == "city"
    assert resolved.label == "Paris"


def test_saving_a_second_city_replaces_the_first(store: WeatherStore, ana: str):
    store.set_city(ana, PARIS, now=NOW)
    store.set_city(ana, TOKYO, now=NOW + 60)
    city = store.preferences(ana, now=NOW + 60).city
    assert city is not None and city.name == "Tokyo"


def test_clearing_the_city_leaves_no_city(store: WeatherStore, ana: str):
    store.set_city(ana, PARIS, now=NOW)
    store.clear_city(ana, now=NOW + 5)
    assert store.preferences(ana, now=NOW + 5).city is None


# --- the browser fix ---------------------------------------------------------------------


def test_browser_location_is_rounded_and_expires(store: WeatherStore, ana: str):
    store.set_browser_location(ana, latitude=51.507351, longitude=-0.127758, now=NOW)

    fresh = store.preferences(ana, now=NOW + 60)
    assert fresh.browser is not None
    # The exact coordinates the browser reported are never written down.
    assert fresh.browser.latitude == 51.51
    assert fresh.browser.longitude == -0.13
    assert fresh.browser.expires_at == NOW + BROWSER_LOCATION_TTL_SECONDS
    assert fresh.resolved is not None and fresh.resolved.source == "browser"

    # Past its short life it is gone, and it is gone from the database too.
    expired = store.preferences(ana, now=NOW + BROWSER_LOCATION_TTL_SECONDS + 1)
    assert expired.browser is None
    assert expired.resolved is None


def test_browser_ttl_is_short():
    assert 0 < BROWSER_LOCATION_TTL_SECONDS <= 24 * 3600


def test_a_saved_city_wins_over_the_browser_fix(store: WeatherStore, ana: str):
    store.set_browser_location(ana, latitude=51.5, longitude=-0.13, now=NOW)
    store.set_city(ana, PARIS, now=NOW)
    resolved = store.preferences(ana, now=NOW).resolved
    assert resolved is not None and resolved.source == "city" and resolved.label == "Paris"

    # Clearing the city falls back to the browser fix rather than to nothing.
    store.clear_city(ana, now=NOW)
    fallback = store.preferences(ana, now=NOW).resolved
    assert fallback is not None and fallback.source == "browser"


def test_clearing_the_browser_location_forgets_it(store: WeatherStore, ana: str):
    store.set_browser_location(ana, latitude=51.5, longitude=-0.13, now=NOW)
    store.clear_browser_location(ana, now=NOW)
    assert store.preferences(ana, now=NOW).browser is None


@pytest.mark.parametrize(
    "latitude, longitude",
    [
        (91.0, 0.0),
        (-90.5, 0.0),
        (0.0, 180.5),
        (0.0, -181.0),
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (0.0, float("-inf")),
        ("51.5", 0.0),
        (None, 0.0),
        (True, 0.0),
    ],
)
def test_impossible_coordinates_are_refused(store: WeatherStore, ana: str, latitude, longitude):
    with pytest.raises(ValueError):
        store.set_browser_location(ana, latitude=latitude, longitude=longitude, now=NOW)
    assert store.preferences(ana, now=NOW).browser is None


# --- isolation ---------------------------------------------------------------------------


def test_one_users_preference_is_never_anothers(store: WeatherStore, ana: str, bo: str):
    store.set_city(ana, PARIS, now=NOW)
    store.set_browser_location(bo, latitude=35.68, longitude=139.69, now=NOW)

    assert store.preferences(bo, now=NOW).city is None
    assert store.preferences(ana, now=NOW).browser is None
    bo_resolved = store.preferences(bo, now=NOW).resolved
    assert bo_resolved is not None and bo_resolved.source == "browser"


def test_an_unknown_user_id_shape_is_refused(store: WeatherStore):
    for bad in ("", "../ana", "usr_nothex", 17, None):
        with pytest.raises(ValueError):
            store.preferences(bad, now=NOW)
        with pytest.raises(ValueError):
            store.set_city(bad, PARIS, now=NOW)


# --- the forecast cache ------------------------------------------------------------------


def test_a_cached_forecast_belongs_to_one_user_and_one_place(
    store: WeatherStore, ana: str, bo: str
):
    ana_key = store.cache_key(ana, store.set_city(ana, PARIS, now=NOW).resolved)
    bo_key = store.cache_key(bo, store.set_city(bo, PARIS, now=NOW).resolved)
    assert ana_key != bo_key

    store.store_forecast(ana_key, ana, {"temperature_c": 7.0}, now=NOW, ttl_seconds=3600)
    assert store.cached_forecast(bo_key) is None
    entry = store.cached_forecast(ana_key)
    assert entry is not None
    assert entry.payload == {"temperature_c": 7.0}
    assert entry.fetched_at == NOW
    assert entry.expires_at == NOW + 3600
    assert entry.is_fresh(NOW + 3599) is True
    assert entry.is_fresh(NOW + 3600) is False


def test_an_expired_forecast_is_still_readable_as_stale(store: WeatherStore, ana: str):
    key = store.cache_key(ana, store.set_city(ana, PARIS, now=NOW).resolved)
    store.store_forecast(key, ana, {"temperature_c": 7.0}, now=NOW, ttl_seconds=3600)
    entry = store.cached_forecast(key)
    assert entry is not None and entry.is_fresh(NOW + 7200) is False


def test_the_geocode_cache_is_keyed_by_the_query_alone(store: WeatherStore):
    store.store_geocode("paris", [PARIS.view()], now=NOW, ttl_seconds=86400)
    entry = store.cached_geocode("  PARIS ")
    assert entry is not None
    assert entry.results == [PARIS.view()]
    assert entry.is_fresh(NOW + 86399) is True
    assert entry.is_fresh(NOW + 86400) is False
    assert store.cached_geocode("tokyo") is None


def run(coro):
    return asyncio.run(coro)


# --- the forecast client -----------------------------------------------------------------


class FakeUpstream:
    """Plays Open-Meteo and Nominatim. Records every request; answers per host."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.answers: list[object] = []
        self.reverse_answers: list[object] = []
        self.default: object = httpx.Response(404, json={"error": True, "reason": "unrouted"})
        self.reverse_default: object = httpx.Response(404, json={"error": "unrouted"})

    def add(self, answer: object, times: int = 1) -> None:
        self.answers.extend([answer] * times)

    def add_reverse(self, answer: object, times: int = 1) -> None:
        self.reverse_answers.extend([answer] * times)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == "nominatim.openstreetmap.org":
            queue, fallback = self.reverse_answers, self.reverse_default
        else:
            queue, fallback = self.answers, self.default
        answer = queue.pop(0) if queue else fallback
        if isinstance(answer, Exception):
            raise answer
        return answer  # type: ignore[return-value]

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    @property
    def forecast_calls(self) -> list[httpx.Request]:
        return [r for r in self.requests if str(r.url).startswith(FORECAST_URL)]

    @property
    def reverse_calls(self) -> list[httpx.Request]:
        return [r for r in self.requests if str(r.url).startswith(NOMINATIM_URL)]


def _forecast_body(**overrides) -> dict:
    current = {
        "time": "2023-11-14T23:00",
        "interval": 900,
        "temperature_2m": 7.4,
        "apparent_temperature": 4.2,
        "relative_humidity_2m": 81,
        "is_day": 0,
        "precipitation": 0.1,
        "rain": 0.1,
        "showers": 0.0,
        "snowfall": 0.0,
        "weather_code": 61,
        "cloud_cover": 100,
        "wind_speed_10m": 14.4,
        "wind_direction_10m": 230,
        "wind_gusts_10m": 31.0,
    }
    current.update(overrides)
    return {
        "latitude": 48.86,
        "longitude": 2.35,
        "generationtime_ms": 0.06,
        "utc_offset_seconds": 3600,
        "timezone": "Europe/Paris",
        "timezone_abbreviation": "CET",
        "elevation": 43.0,
        "current_units": {"temperature_2m": "C"},
        "current": current,
    }


def _ok() -> httpx.Response:
    return httpx.Response(200, json=_forecast_body())


class Clock:
    def __init__(self, now: int = NOW) -> None:
        self.now = float(now)

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def upstream() -> FakeUpstream:
    return FakeUpstream()


@pytest.fixture
def clock() -> Clock:
    return Clock()


class Sleeper:
    """Stands in for asyncio.sleep so the global lookup spacing costs no time."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.slept: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.clock.now += seconds


@pytest.fixture(autouse=True)
def unspaced():
    ReverseGeocoder.reset_pacing()
    yield
    ReverseGeocoder.reset_pacing()


@pytest.fixture
def geocoder(upstream: FakeUpstream, clock: Clock) -> ReverseGeocoder:
    return ReverseGeocoder(transport=upstream.transport, clock=clock, sleep=Sleeper(clock))


@pytest.fixture
def client(
    store: WeatherStore, upstream: FakeUpstream, clock: Clock, geocoder: ReverseGeocoder
) -> WeatherClient:
    return WeatherClient(
        store, transport=upstream.transport, clock=clock, reverse_geocoder=geocoder
    )


def _reverse_ok() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "address": {
                "road": "Westminster Bridge Road",
                "suburb": "Lambeth",
                "city": "London",
                "state": "England",
                "postcode": "SE1 7PB",
                "country": "United Kingdom",
            }
        },
    )


def test_current_asks_open_meteo_for_exactly_the_documented_request(
    store: WeatherStore, client: WeatherClient, upstream: FakeUpstream, ana: str
):
    store.set_city(ana, PARIS, now=NOW)
    upstream.add(_ok())

    snapshot = run(client.current(ana))

    assert snapshot.state == "ok"
    assert len(upstream.forecast_calls) == 1
    request = upstream.forecast_calls[0]
    assert request.method == "GET"
    url = request.url
    assert url.scheme == "https"
    assert url.host == "api.open-meteo.com"
    assert url.path == "/v1/forecast"
    assert dict(url.params) == {
        "latitude": "48.86",
        "longitude": "2.35",
        "current": ",".join(CURRENT_FIELDS),
        "timezone": "auto",
    }
    # No key, no token, no identity: Open-Meteo is asked about a place only.
    assert "authorization" not in {name.lower() for name in request.headers}
    assert ana not in str(url)


def test_current_reduces_the_answer_to_bounded_values(
    store: WeatherStore, client: WeatherClient, upstream: FakeUpstream, ana: str
):
    store.set_city(ana, PARIS, now=NOW)
    upstream.add(_ok())

    snapshot = run(client.current(ana))
    observation = snapshot.observation
    assert observation is not None
    assert observation["temperature_c"] == 7.4
    assert observation["apparent_temperature_c"] == 4.2
    assert observation["relative_humidity_pct"] == 81
    assert observation["is_day"] is False
    assert observation["precipitation_mm"] == 0.1
    assert observation["weather_code"] == 61
    assert observation["cloud_cover_pct"] == 100
    assert observation["wind_speed_kmh"] == 14.4
    assert observation["wind_direction_deg"] == 230
    assert observation["wind_gusts_kmh"] == 31.0
    assert observation["observed_at"] == "2023-11-14T23:00"
    # Only the fields this module declares; nothing the upstream volunteered.
    assert set(observation) == set(OBSERVATION_FIELDS)
    # A practical, human summary the widget can show without inventing anything.
    assert snapshot.conditions == "Slight rain"
    assert isinstance(snapshot.advice, str) and snapshot.advice
    assert snapshot.location is not None and snapshot.location.label == "Paris"
    assert snapshot.cached_at == NOW
    assert snapshot.expires_at == NOW + CACHE_TTL_SECONDS


def test_nonsense_measurements_become_nothing_rather_than_a_number(
    store: WeatherStore, client: WeatherClient, upstream: FakeUpstream, ana: str
):
    store.set_city(ana, PARIS, now=NOW)
    upstream.add(
        httpx.Response(
            200,
            json=_forecast_body(
                temperature_2m="warm",
                relative_humidity_2m=180,
                wind_speed_10m=-4,
                wind_direction_10m=900,
                weather_code=4321,
                cloud_cover=None,
            ),
        )
    )

    snapshot = run(client.current(ana))
    observation = snapshot.observation
    assert observation is not None
    assert observation["temperature_c"] is None
    assert observation["relative_humidity_pct"] is None
    assert observation["wind_speed_kmh"] is None
    assert observation["wind_direction_deg"] is None
    assert observation["weather_code"] is None
    assert observation["cloud_cover_pct"] is None
    assert snapshot.conditions is None


def test_a_user_with_no_location_is_never_a_request(
    client: WeatherClient, upstream: FakeUpstream, ana: str
):
    snapshot = run(client.current(ana))
    assert snapshot.state == "no_location"
    assert snapshot.location is None
    assert snapshot.observation is None
    assert upstream.requests == []


def test_a_browser_fix_is_read_without_naming_the_place(
    store: WeatherStore, client: WeatherClient, upstream: FakeUpstream, ana: str
):
    store.set_browser_location(ana, latitude=51.507351, longitude=-0.127758, now=NOW)
    upstream.add(_ok())

    snapshot = run(client.current(ana))
    assert snapshot.state == "ok"
    assert snapshot.location is not None
    assert snapshot.location.source == "browser"
    # A generic label, and the timezone the upstream resolved -- never a coordinate.
    assert snapshot.location.label == "Your current location"
    assert snapshot.location.timezone == "Europe/Paris"
    assert "51.507351" not in str(upstream.forecast_calls[0].url)
    assert "51.51" not in json.dumps(snapshot.view())


# --- the hourly cap ----------------------------------------------------------------------


def test_a_second_read_within_the_hour_never_reaches_open_meteo(
    store: WeatherStore, client: WeatherClient, upstream: FakeUpstream, clock: Clock, ana: str
):
    store.set_city(ana, PARIS, now=NOW)
    upstream.add(_ok())

    first = run(client.current(ana))
    for offset in (1, 60, 1800, CACHE_TTL_SECONDS - 1):
        clock.now = NOW + offset
        again = run(client.current(ana))
        assert again.state == "ok"
        assert again.stale is False
        assert again.cached_at == first.cached_at
        assert again.observation == first.observation
    assert len(upstream.forecast_calls) == 1


def test_the_cap_survives_a_restart_of_the_client(
    store: WeatherStore, upstream: FakeUpstream, clock: Clock, ana: str
):
    store.set_city(ana, PARIS, now=NOW)
    upstream.add(_ok(), times=2)

    run(WeatherClient(store, transport=upstream.transport, clock=clock).current(ana))
    clock.now = NOW + 120
    # A fresh process, the same database: still within the hour, still one call.
    second = run(WeatherClient(store, transport=upstream.transport, clock=clock).current(ana))
    assert second.state == "ok"
    assert len(upstream.forecast_calls) == 1


def test_once_the_hour_is_up_the_forecast_is_read_again(
    store: WeatherStore, client: WeatherClient, upstream: FakeUpstream, clock: Clock, ana: str
):
    store.set_city(ana, PARIS, now=NOW)
    upstream.add(_ok(), times=2)

    run(client.current(ana))
    clock.now = NOW + CACHE_TTL_SECONDS
    refreshed = run(client.current(ana))
    assert len(upstream.forecast_calls) == 2
    assert refreshed.cached_at == NOW + CACHE_TTL_SECONDS
    assert refreshed.expires_at == NOW + 2 * CACHE_TTL_SECONDS


def test_the_cache_ttl_is_never_longer_than_an_hour(store: WeatherStore):
    assert CACHE_TTL_SECONDS == 3600
    with pytest.raises(ValueError):
        WeatherClient(store, cache_ttl_seconds=0)


def test_concurrent_reads_of_one_place_make_one_upstream_call(
    store: WeatherStore, clock: Clock, ana: str
):
    calls: list[httpx.Request] = []

    async def slow(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        # Long enough that every reader is waiting before the first answers.
        await asyncio.sleep(0.05)
        return httpx.Response(200, json=_forecast_body())

    store.set_city(ana, PARIS, now=NOW)
    client = WeatherClient(store, transport=httpx.MockTransport(slow), clock=clock)

    async def five_at_once():
        return await asyncio.gather(*(client.current(ana) for _ in range(5)))

    snapshots = run(five_at_once())
    assert len(calls) == 1
    assert [s.state for s in snapshots] == ["ok"] * 5


# --- when the upstream is down -----------------------------------------------------------


def test_an_outage_returns_the_last_answer_and_says_it_is_stale(
    store: WeatherStore, client: WeatherClient, upstream: FakeUpstream, clock: Clock, ana: str
):
    store.set_city(ana, PARIS, now=NOW)
    upstream.add(_ok())
    good = run(client.current(ana))

    clock.now = NOW + 2 * CACHE_TTL_SECONDS
    upstream.add(httpx.ConnectError("no route"))
    stale = run(client.current(ana))

    assert stale.state == "stale"
    assert stale.stale is True
    assert stale.reason == "transport"
    assert stale.observation == good.observation
    # The reading is honestly dated to when it was actually read.
    assert stale.cached_at == NOW
    assert stale.expires_at == NOW + CACHE_TTL_SECONDS
    assert stale.view()["stale"] is True


@pytest.mark.parametrize(
    "answer, reason",
    [
        (httpx.ConnectError("no route"), "transport"),
        (httpx.ReadTimeout("slow"), "transport"),
        (httpx.Response(500, json={"error": True, "reason": "boom"}), "provider_refused"),
        (httpx.Response(429, json={"error": True}), "provider_refused"),
        (
            httpx.Response(302, headers={"location": "https://elsewhere.example/x"}),
            "malformed_response",
        ),
        (httpx.Response(200, content=b"not json"), "malformed_response"),
        (httpx.Response(200, json={"latitude": 48.86}), "malformed_response"),
        (httpx.Response(200, content=b"x" * (MAX_RESPONSE_BYTES + 1)), "malformed_response"),
    ],
)
def test_every_upstream_failure_is_one_bounded_unavailable_state(
    store: WeatherStore,
    client: WeatherClient,
    upstream: FakeUpstream,
    ana: str,
    answer,
    reason,
):
    store.set_city(ana, PARIS, now=NOW)
    upstream.add(answer)

    snapshot = run(client.current(ana))
    assert snapshot.state == "unavailable"
    assert snapshot.reason == reason
    assert snapshot.observation is None
    # A redirect is refused, not followed.
    assert len(upstream.requests) == 1


def test_a_failure_never_carries_the_upstream_body_into_a_log_or_an_answer(
    store: WeatherStore, client: WeatherClient, upstream: FakeUpstream, ana: str, caplog
):
    secret = "UPSTREAM-BODY-MARKER"
    store.set_city(ana, PARIS, now=NOW)
    upstream.add(httpx.Response(503, json={"error": True, "reason": secret}))

    with caplog.at_level(logging.DEBUG):
        snapshot = run(client.current(ana))

    assert snapshot.state == "unavailable"
    rendered = json.dumps(snapshot.view())
    assert secret not in rendered
    assert secret not in caplog.text
    # Nor the user, nor the place, nor a coordinate.
    assert ana not in caplog.text
    assert "48.86" not in caplog.text
    assert "Paris" not in caplog.text


def test_only_the_two_open_meteo_hosts_may_be_contacted():
    assert is_allowed_url(FORECAST_URL) is True
    assert is_allowed_url(GEOCODING_URL) is True
    for url in (
        "http://api.open-meteo.com/v1/forecast",
        "https://api.open-meteo.com.evil.example/v1/forecast",
        "https://open-meteo.com/v1/forecast",
        "https://user@api.open-meteo.com/v1/forecast",
        "https://127.0.0.1/v1/forecast",
        "file:///etc/passwd",
        "/v1/forecast",
    ):
        assert is_allowed_url(url) is False, url


# --- isolation ---------------------------------------------------------------------------


def test_two_users_in_one_city_never_share_a_cached_reading(
    store: WeatherStore, client: WeatherClient, upstream: FakeUpstream, ana: str, bo: str
):
    store.set_city(ana, PARIS, now=NOW)
    store.set_city(bo, PARIS, now=NOW)
    upstream.add(httpx.Response(200, json=_forecast_body(temperature_2m=7.4)))
    upstream.add(httpx.Response(200, json=_forecast_body(temperature_2m=21.0)))

    for_ana = run(client.current(ana))
    for_bo = run(client.current(bo))
    assert for_ana.observation is not None and for_ana.observation["temperature_c"] == 7.4
    assert for_bo.observation is not None and for_bo.observation["temperature_c"] == 21.0
    assert len(upstream.forecast_calls) == 2


def test_one_users_outage_is_never_answered_with_anothers_reading(
    store: WeatherStore, client: WeatherClient, upstream: FakeUpstream, ana: str, bo: str
):
    store.set_city(ana, PARIS, now=NOW)
    store.set_city(bo, PARIS, now=NOW)
    upstream.add(_ok())
    run(client.current(ana))

    upstream.add(httpx.ConnectError("no route"))
    for_bo = run(client.current(bo))
    assert for_bo.state == "unavailable"
    assert for_bo.observation is None


def test_changing_the_place_is_a_new_reading(
    store: WeatherStore, client: WeatherClient, upstream: FakeUpstream, ana: str
):
    store.set_city(ana, PARIS, now=NOW)
    upstream.add(httpx.Response(200, json=_forecast_body(temperature_2m=7.4)))
    run(client.current(ana))

    store.set_city(ana, TOKYO, now=NOW)
    upstream.add(httpx.Response(200, json=_forecast_body(temperature_2m=21.0)))
    moved = run(client.current(ana))
    assert moved.location is not None and moved.location.label == "Tokyo"
    assert moved.observation is not None and moved.observation["temperature_c"] == 21.0
    assert dict(upstream.forecast_calls[1].url.params)["latitude"] == "35.69"


# --- the city lookup ---------------------------------------------------------------------


def _geocode_body(*results) -> dict:
    return {"results": list(results), "generationtime_ms": 0.6}


PARIS_ROW = {
    "id": 2988507,
    "name": "Paris",
    "latitude": 48.85341,
    "longitude": 2.3488,
    "elevation": 42.0,
    "feature_code": "PPLC",
    "country_code": "FR",
    "timezone": "Europe/Paris",
    "population": 2138551,
    "country_id": 3017382,
    "country": "France",
    "admin1": "Ile-de-France",
    "admin2": "Paris",
}
PARIS_TX_ROW = {
    "id": 4717560,
    "name": "Paris",
    "latitude": 33.66094,
    "longitude": -95.55551,
    "timezone": "America/Chicago",
    "country": "United States",
    "admin1": "Texas",
}


def _geocode_calls(upstream: FakeUpstream) -> list[httpx.Request]:
    return [r for r in upstream.requests if str(r.url).startswith(GEOCODING_URL)]


def test_a_city_lookup_is_one_bounded_documented_request(
    client: WeatherClient, upstream: FakeUpstream
):
    upstream.add(httpx.Response(200, json=_geocode_body(PARIS_ROW, PARIS_TX_ROW)))

    results = run(client.search_cities("  Paris  ", count=2))

    assert [city.name for city in results] == ["Paris", "Paris"]
    request = _geocode_calls(upstream)[0]
    assert request.method == "GET"
    assert request.url.scheme == "https"
    assert request.url.host == "geocoding-api.open-meteo.com"
    assert request.url.path == "/v1/search"
    assert dict(request.url.params) == {
        "name": "Paris",
        "count": str(MAX_SEARCH_RESULTS),
        "language": "en",
        "format": "json",
    }


def test_a_looked_up_city_keeps_only_the_fields_the_dashboard_needs(
    client: WeatherClient, upstream: FakeUpstream
):
    upstream.add(httpx.Response(200, json=_geocode_body(PARIS_ROW)))

    city = run(client.search_cities("paris", count=5))[0]
    assert city.name == "Paris"
    assert city.country == "France"
    assert city.admin1 == "Ile-de-France"
    assert city.timezone == "Europe/Paris"
    # Rounded on the way in, like everything else that gets stored.
    assert city.latitude == 48.85
    assert city.longitude == 2.35
    assert set(city.view()) == {
        "name",
        "latitude",
        "longitude",
        "timezone",
        "country",
        "admin1",
    }


def test_unusable_lookup_rows_are_dropped_rather_than_shown(
    client: WeatherClient, upstream: FakeUpstream
):
    upstream.add(
        httpx.Response(
            200,
            json=_geocode_body(
                {"name": "Nowhere", "latitude": 999.0, "longitude": 0.0},
                {"latitude": 1.0, "longitude": 1.0},
                {"name": "", "latitude": 1.0, "longitude": 1.0},
                "not an object",
                PARIS_ROW,
            ),
        )
    )
    results = run(client.search_cities("paris", count=10))
    assert [city.name for city in results] == ["Paris"]


def test_a_lookup_with_no_match_is_an_empty_answer(
    client: WeatherClient, upstream: FakeUpstream
):
    upstream.add(httpx.Response(200, json={"generationtime_ms": 0.4}))
    assert run(client.search_cities("zzzzzzz", count=5)) == []


def test_the_lookup_is_bounded_before_anything_is_sent(
    client: WeatherClient, upstream: FakeUpstream
):
    for bad in ("", "   ", "x" * 200, 17, None):
        with pytest.raises(ValueError):
            run(client.search_cities(bad, count=5))
    for bad_count in (0, -1, "5", None, True):
        with pytest.raises(ValueError):
            run(client.search_cities("paris", count=bad_count))
    assert upstream.requests == []


def test_a_lookup_asks_for_no_more_than_the_bound(
    client: WeatherClient, upstream: FakeUpstream
):
    upstream.add(httpx.Response(200, json=_geocode_body(*[PARIS_ROW] * 40)))
    results = run(client.search_cities("paris", count=99))
    assert len(results) == MAX_SEARCH_RESULTS


def test_a_repeated_lookup_does_not_reach_open_meteo_again(
    client: WeatherClient, upstream: FakeUpstream, clock: Clock
):
    upstream.add(httpx.Response(200, json=_geocode_body(PARIS_ROW)))
    first = run(client.search_cities("Paris", count=5))

    clock.now = NOW + 3600
    # The same place, typed differently: one cache entry, one upstream call.
    again = run(client.search_cities("  paris ", count=3))
    assert [c.view() for c in again] == [c.view() for c in first]
    assert len(_geocode_calls(upstream)) == 1

    clock.now = NOW + GEOCODE_CACHE_TTL_SECONDS
    upstream.add(httpx.Response(200, json=_geocode_body(PARIS_ROW)))
    run(client.search_cities("paris", count=5))
    assert len(_geocode_calls(upstream)) == 2


def test_a_failed_lookup_is_not_cached_as_an_empty_answer(
    client: WeatherClient, upstream: FakeUpstream
):
    upstream.add(httpx.ConnectError("no route"))
    with pytest.raises(WeatherError) as excinfo:
        run(client.search_cities("paris", count=5))
    assert excinfo.value.reason == "transport"

    upstream.add(httpx.Response(200, json=_geocode_body(PARIS_ROW)))
    assert [c.name for c in run(client.search_cities("paris", count=5))] == ["Paris"]


# --- choosing a city ---------------------------------------------------------------------


def test_a_chosen_city_must_be_one_the_lookup_returned(
    store: WeatherStore, client: WeatherClient, upstream: FakeUpstream, ana: str
):
    upstream.add(httpx.Response(200, json=_geocode_body(PARIS_ROW, PARIS_TX_ROW)))
    preference = run(client.choose_city(ana, name="Paris", latitude=48.85341, longitude=2.3488))

    assert preference.city is not None
    assert preference.city.name == "Paris"
    # Every label comes from the lookup, never from whatever the client sent.
    assert preference.city.country == "France"
    assert preference.city.admin1 == "Ile-de-France"
    assert preference.city.timezone == "Europe/Paris"
    assert store.preferences(ana, now=NOW).city == preference.city


def test_the_second_match_can_be_chosen_by_its_own_coordinates(
    client: WeatherClient, upstream: FakeUpstream, ana: str
):
    upstream.add(httpx.Response(200, json=_geocode_body(PARIS_ROW, PARIS_TX_ROW)))
    preference = run(client.choose_city(ana, name="Paris", latitude=33.66094, longitude=-95.55551))
    assert preference.city is not None and preference.city.admin1 == "Texas"


def test_a_place_the_lookup_never_returned_is_refused(
    store: WeatherStore, client: WeatherClient, upstream: FakeUpstream, ana: str
):
    upstream.add(httpx.Response(200, json=_geocode_body(PARIS_ROW)), times=3)

    for latitude, longitude in ((0.0, 0.0), (48.85341, 100.0), (51.5, -0.13)):
        with pytest.raises(WeatherError) as excinfo:
            run(client.choose_city(ana, name="Paris", latitude=latitude, longitude=longitude))
        assert excinfo.value.reason == "unknown_city"
    assert store.preferences(ana, now=NOW).city is None


def test_choosing_a_city_reuses_the_cached_lookup(
    client: WeatherClient, upstream: FakeUpstream, ana: str
):
    upstream.add(httpx.Response(200, json=_geocode_body(PARIS_ROW)))
    run(client.search_cities("Paris", count=5))
    run(client.choose_city(ana, name="Paris", latitude=48.85341, longitude=2.3488))
    # The search the user just did is the one that validates their choice.
    assert len(_geocode_calls(upstream)) == 1


def test_choosing_a_city_clears_any_browser_fix(
    store: WeatherStore, client: WeatherClient, upstream: FakeUpstream, ana: str
):
    store.set_browser_location(ana, latitude=51.5, longitude=-0.13, now=NOW)
    upstream.add(httpx.Response(200, json=_geocode_body(PARIS_ROW)))
    preference = run(client.choose_city(ana, name="Paris", latitude=48.85341, longitude=2.3488))
    assert preference.browser is None
    assert store.preferences(ana, now=NOW).browser is None


# --- naming a shared browser position ----------------------------------------------------


def test_a_shared_browser_position_is_named_by_the_reverse_lookup(
    store: WeatherStore, client: WeatherClient, upstream: FakeUpstream, ana: str
):
    store.set_browser_location(ana, latitude=51.507351, longitude=-0.127758, now=NOW)
    upstream.add(_ok())
    upstream.add_reverse(_reverse_ok())

    snapshot = run(client.current(ana))

    assert snapshot.state == "ok"
    assert snapshot.location is not None
    assert snapshot.location.source == "browser"
    # Named well enough to notice a wrong location, coarse enough to be nobody.
    assert snapshot.location.label == "London"
    assert snapshot.location.admin1 == "England"
    assert snapshot.location.country == "United Kingdom"
    view = json.dumps(snapshot.view())
    assert "51.51" not in view and "-0.13" not in view and "51.507351" not in view
    assert "Westminster Bridge Road" not in view and "SE1 7PB" not in view


def test_the_reverse_lookup_is_told_only_the_rounded_position(
    store: WeatherStore, client: WeatherClient, upstream: FakeUpstream, ana: str
):
    store.set_browser_location(ana, latitude=51.507351, longitude=-0.127758, now=NOW)
    upstream.add(_ok())
    upstream.add_reverse(_reverse_ok())

    run(client.current(ana))

    assert len(upstream.reverse_calls) == 1
    sent = upstream.reverse_calls[0].url
    assert sent.host == "nominatim.openstreetmap.org"
    assert dict(sent.params)["lat"] == "51.51"
    assert dict(sent.params)["lon"] == "-0.13"
    assert "51.507351" not in str(sent)
    assert ana not in str(sent)


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(500, text="down"),
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json={"address": {"road": "Westminster Bridge Road"}}),
        httpx.ConnectError("no route"),
    ],
)
def test_an_unusable_reverse_lookup_leaves_the_truthful_generic_label(
    store: WeatherStore, client: WeatherClient, upstream: FakeUpstream, ana: str, answer: object
):
    store.set_browser_location(ana, latitude=51.507351, longitude=-0.127758, now=NOW)
    upstream.add(_ok())
    upstream.add_reverse(answer)

    snapshot = run(client.current(ana))

    # The weather still works; only the name of the place is missing.
    assert snapshot.state == "ok"
    assert snapshot.observation is not None
    assert snapshot.observation["temperature_c"] == 7.4
    assert snapshot.location is not None
    assert snapshot.location.label == "Your current location"
    assert snapshot.location.admin1 is None and snapshot.location.country is None


def test_a_saved_city_is_never_reverse_geocoded(
    store: WeatherStore, client: WeatherClient, upstream: FakeUpstream, ana: str
):
    store.set_city(ana, PARIS, now=NOW)
    upstream.add(_ok())

    snapshot = run(client.current(ana))

    assert upstream.reverse_calls == []
    assert snapshot.location is not None
    assert snapshot.location.label == "Paris"
    assert snapshot.location.admin1 == "Ile-de-France"


def test_a_city_chosen_by_hand_replaces_a_reverse_geocoded_name(
    store: WeatherStore, client: WeatherClient, upstream: FakeUpstream, ana: str
):
    store.set_browser_location(ana, latitude=51.507351, longitude=-0.127758, now=NOW)
    upstream.add(_ok(), times=2)
    upstream.add_reverse(_reverse_ok())
    assert run(client.current(ana)).location.label == "London"  # type: ignore[union-attr]

    store.set_city(ana, PARIS, now=NOW)
    named = run(client.current(ana))

    assert named.location is not None and named.location.label == "Paris"
    assert named.location.source == "city"
    assert len(upstream.reverse_calls) == 1


def test_repeated_reads_of_one_position_reverse_geocode_it_once(
    store: WeatherStore, client: WeatherClient, upstream: FakeUpstream, clock: Clock, ana: str
):
    store.set_browser_location(ana, latitude=51.507351, longitude=-0.127758, now=NOW)
    upstream.add(_ok(), times=3)
    upstream.add_reverse(_reverse_ok())

    for offset in (0, 60, CACHE_TTL_SECONDS + 1):
        clock.now = NOW + offset
        snapshot = run(client.current(ana))
        assert snapshot.location is not None and snapshot.location.label == "London"

    assert len(upstream.reverse_calls) == 1


def test_a_reverse_geocoded_name_never_reaches_a_log_with_a_position(
    store: WeatherStore, client: WeatherClient, upstream: FakeUpstream, ana: str, caplog
):
    store.set_browser_location(ana, latitude=51.507351, longitude=-0.127758, now=NOW)
    upstream.add(_ok())
    upstream.add_reverse(_reverse_ok())

    with caplog.at_level(logging.DEBUG):
        run(client.current(ana))

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "51.507351" not in logged
    assert "lat=51.51" not in logged and "latitude=51.51" not in logged
