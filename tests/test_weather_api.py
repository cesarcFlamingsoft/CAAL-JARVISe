"""HTTP contract for the per-user weather routes.

``/users/me/weather`` and its siblings sit behind the same internal trust
boundary as the rest of the identity API: the BFF proves itself with a
single-use signed principal, the backend loads the user from its own
database, and only that user's own place is read. Nothing here talks to
Open-Meteo: the upstream is an injected ``httpx`` transport that records what
it was asked.

Pinned properties: an unauthenticated caller gets nothing; a place is asked
upstream at most once an hour however often the dashboard reloads; a city can
only be saved if the city lookup returned it; a browser position needs
explicit consent and plausible coordinates and is never echoed back; an
upstream outage is answered truthfully rather than as a server error; and one
user's place is never another's.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from caal import profile_crypto, user_api, weather_api, webhooks
from caal.internal_auth import AUDIENCE_BACKEND, RateLimiter, mint_principal
from caal.profile_crypto import KeyRing
from caal.reverse_geocode import ReverseGeocoder
from caal.security_config import MultiUserConfig
from caal.user_api import IdentityRuntime
from caal.user_store import MEMBER, Actor, UserStore
from caal.weather import MAX_SEARCH_RESULTS, WeatherClient
from caal.weather_api import WeatherRuntime
from caal.weather_store import BROWSER_LOCATION_TTL_SECONDS, WeatherStore

SECRET = "s" * 48
BOOTSTRAP = "cesarc@mexcantech.com"
NOW = 1_700_000_000

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
    "country": "France",
    "admin1": "Ile-de-France",
}


def _forecast_body(**overrides) -> dict:
    current = {
        "time": "2023-11-14T23:00",
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
    return {"latitude": 48.85, "longitude": 2.35, "timezone": "Europe/Paris", "current": current}


class FakeUpstream:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.answers: list[object] = []
        self.reverse_answers: list[object] = []

    def add(self, answer: object, times: int = 1) -> None:
        self.answers.extend([answer] * times)

    def add_reverse(self, answer: object, times: int = 1) -> None:
        self.reverse_answers.extend([answer] * times)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reverse = request.url.host == "nominatim.openstreetmap.org"
        queue = self.reverse_answers if reverse else self.answers
        answer = queue.pop(0) if queue else httpx.Response(404, json={"e": 1})
        if isinstance(answer, Exception):
            raise answer
        return answer  # type: ignore[return-value]

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def forecasts(self) -> list[httpx.Request]:
        return [r for r in self.requests if "v1/forecast" in str(r.url)]

    def searches(self) -> list[httpx.Request]:
        return [r for r in self.requests if "v1/search" in str(r.url)]

    def reverses(self) -> list[httpx.Request]:
        return [r for r in self.requests if "nominatim" in str(r.url)]


class Harness:
    def __init__(self, tmp_path) -> None:
        self.now = NOW
        self.keyring = KeyRing.from_env(profile_crypto.generate_key_material(version=1))
        self.store = UserStore(tmp_path / "assistant.sqlite3", keyring=self.keyring)
        self.config = MultiUserConfig(
            internal_auth_secret=SECRET,
            keyring=self.keyring,
            bootstrap_admin_email=BOOTSTRAP,
            store_path=tmp_path / "assistant.sqlite3",
        )
        self.identity = IdentityRuntime(
            self.config,
            store=self.store,
            mutation_limiter=RateLimiter(limit=100, window_seconds=60),
            clock=lambda: self.now,
        )
        self.upstream = FakeUpstream()
        self.weather_store = WeatherStore(self.store)
        self.runtime = WeatherRuntime(
            self.identity,
            store=self.weather_store,
            client=WeatherClient(
                self.weather_store,
                transport=self.upstream.transport,
                clock=lambda: self.now,
                reverse_geocoder=ReverseGeocoder(
                    transport=self.upstream.transport,
                    clock=lambda: self.now,
                    sleep=self._skip_wait,
                ),
            ),
        )
        self.ana = self.user("ana@example.com")
        self.bo = self.user("bo@example.com")

    async def _skip_wait(self, seconds: float) -> None:
        """The global lookup spacing is real; waiting through it in a test is not."""
        self.now += int(seconds) + 1

    def user(self, email: str) -> str:
        return self.store.create_user(
            email=email,
            display_name=email.split("@")[0],
            role=MEMBER,
            actor=Actor.system(),
            now=self.now,
        ).user_id

    def bearer(self, user_id: str) -> dict[str, str]:
        token = mint_principal(
            secret=SECRET, subject=user_id, audience=AUDIENCE_BACKEND, now=self.now
        )
        return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def unspaced():
    ReverseGeocoder.reset_pacing()
    yield
    ReverseGeocoder.reset_pacing()


@pytest.fixture
def harness(tmp_path):
    return Harness(tmp_path)


@pytest.fixture
def client(harness):
    webhooks.app.dependency_overrides[user_api.get_runtime] = lambda: harness.identity
    webhooks.app.dependency_overrides[weather_api.get_weather_runtime] = lambda: harness.runtime
    try:
        with TestClient(webhooks.app) as test_client:
            yield test_client
    finally:
        webhooks.app.dependency_overrides.pop(user_api.get_runtime, None)
        webhooks.app.dependency_overrides.pop(weather_api.get_weather_runtime, None)


def _save_paris(harness, client) -> httpx.Response:
    harness.upstream.add(httpx.Response(200, json={"results": [PARIS_ROW]}))
    return client.put(
        "/users/me/weather/city",
        headers=harness.bearer(harness.ana),
        json={"name": "Paris", "latitude": 48.85341, "longitude": 2.3488},
    )


# --- the identity boundary ---------------------------------------------------------------


def test_every_weather_route_refuses_an_unauthenticated_caller(client):
    calls = [
        ("get", "/users/me/weather", None),
        ("get", "/users/me/weather/location", None),
        ("get", "/users/me/weather/cities?q=paris", None),
        ("put", "/users/me/weather/city", {"name": "Paris", "latitude": 1.0, "longitude": 1.0}),
        ("delete", "/users/me/weather/city", None),
        (
            "put",
            "/users/me/weather/browser-location",
            {"consent": True, "latitude": 1.0, "longitude": 1.0},
        ),
        ("delete", "/users/me/weather/browser-location", None),
    ]
    for method, path, body in calls:
        response = getattr(client, method)(path, **({"json": body} if body else {}))
        assert response.status_code == 401, path


def test_a_forged_principal_is_refused(harness, client):
    forged = mint_principal(
        secret="w" * 48, subject=harness.ana, audience=AUDIENCE_BACKEND, now=NOW
    )
    response = client.get("/users/me/weather", headers={"Authorization": f"Bearer {forged}"})
    assert response.status_code == 401


# --- current weather ---------------------------------------------------------------------


def test_a_user_with_no_place_is_told_so_rather_than_shown_a_guess(harness, client):
    response = client.get("/users/me/weather", headers=harness.bearer(harness.ana))
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "no_location"
    assert body["stale"] is False
    assert body["location"] is None
    assert body["observation"] is None
    assert harness.upstream.requests == []


def test_current_weather_is_the_bounded_summary_and_no_coordinates(harness, client):
    assert _save_paris(harness, client).status_code == 200
    harness.upstream.add(httpx.Response(200, json=_forecast_body()))

    response = client.get("/users/me/weather", headers=harness.bearer(harness.ana))
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "ok"
    assert body["stale"] is False
    assert body["location"] == {
        "source": "city",
        "label": "Paris",
        "region": "Ile-de-France",
        "country": "France",
        "timezone": "Europe/Paris",
    }
    assert body["observation"]["temperature_c"] == 7.4
    assert body["conditions"] == "Slight rain"
    assert body["advice"]
    assert body["cached_at"] == NOW
    assert body["expires_at"] == NOW + 3600
    assert body["generated_at"] == NOW
    # No coordinate, no upstream payload, no user id anywhere in the answer.
    rendered = json.dumps(body)
    for forbidden in ("48.85", "2.3488", "latitude", "generationtime", harness.ana):
        assert forbidden not in rendered, forbidden
    assert response.headers["cache-control"].startswith("no-store")


def test_the_route_asks_open_meteo_once_an_hour_however_often_it_is_called(harness, client):
    _save_paris(harness, client)
    harness.upstream.add(httpx.Response(200, json=_forecast_body()))

    for _ in range(6):
        again = client.get("/users/me/weather", headers=harness.bearer(harness.ana))
        assert again.status_code == 200
    assert len(harness.upstream.forecasts()) == 1

    harness.now = NOW + 3600
    harness.upstream.add(httpx.Response(200, json=_forecast_body()))
    client.get("/users/me/weather", headers=harness.bearer(harness.ana))
    assert len(harness.upstream.forecasts()) == 2


def test_an_upstream_outage_answers_the_stale_reading_truthfully(harness, client):
    _save_paris(harness, client)
    harness.upstream.add(httpx.Response(200, json=_forecast_body()))
    client.get("/users/me/weather", headers=harness.bearer(harness.ana))

    harness.now = NOW + 7200
    harness.upstream.add(httpx.ConnectError("no route"))
    response = client.get("/users/me/weather", headers=harness.bearer(harness.ana))

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "stale"
    assert body["stale"] is True
    assert body["reason"] == "transport"
    assert body["observation"]["temperature_c"] == 7.4
    assert body["cached_at"] == NOW


def test_an_outage_with_nothing_cached_is_an_honest_unavailable(harness, client, caplog):
    _save_paris(harness, client)
    harness.upstream.add(httpx.Response(500, json={"error": True, "reason": "SECRET-BODY"}))

    with caplog.at_level(logging.DEBUG):
        response = client.get("/users/me/weather", headers=harness.bearer(harness.ana))

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "unavailable"
    assert body["observation"] is None
    assert body["reason"] == "provider_refused"
    assert "SECRET-BODY" not in json.dumps(body)
    assert "SECRET-BODY" not in caplog.text


def test_one_users_weather_is_never_anothers(harness, client):
    _save_paris(harness, client)
    harness.upstream.add(httpx.Response(200, json=_forecast_body()))
    client.get("/users/me/weather", headers=harness.bearer(harness.ana))

    for_bo = client.get("/users/me/weather", headers=harness.bearer(harness.bo))
    assert for_bo.status_code == 200
    assert for_bo.json()["state"] == "no_location"
    assert len(harness.upstream.forecasts()) == 1


# --- the city lookup ---------------------------------------------------------------------


def test_the_city_search_returns_bounded_places_only(harness, client):
    harness.upstream.add(httpx.Response(200, json={"results": [PARIS_ROW], "generationtime_ms": 1}))

    response = client.get(
        "/users/me/weather/cities?q=Paris&count=5", headers=harness.bearer(harness.ana)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "Paris"
    assert body["results"] == [
        {
            "name": "Paris",
            "latitude": 48.85,
            "longitude": 2.35,
            "timezone": "Europe/Paris",
            "country": "France",
            "region": "Ile-de-France",
        }
    ]
    # Everything Open-Meteo volunteered about the place is dropped.
    rendered = json.dumps(body)
    for forbidden in ("population", "feature_code", "elevation", "country_code", "2988507"):
        assert forbidden not in rendered, forbidden


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        "?q=" + "x" * 200,
        "count=0",
        "count=101",
        "count=abc",
    ],
)
def test_a_city_search_outside_its_bounds_is_refused_before_anything_is_sent(
    harness, client, query
):
    path = (
        f"/users/me/weather/cities?q=paris&{query}"
        if query.startswith("count")
        else f"/users/me/weather/cities?q={query}"
    )
    response = client.get(path, headers=harness.bearer(harness.ana))
    assert response.status_code == 422, path
    assert harness.upstream.searches() == []


def test_a_city_search_is_capped_at_the_documented_bound(harness, client):
    harness.upstream.add(httpx.Response(200, json={"results": [PARIS_ROW] * 40}))
    response = client.get(
        "/users/me/weather/cities?q=paris&count=100", headers=harness.bearer(harness.ana)
    )
    assert response.status_code == 200
    assert len(response.json()["results"]) == MAX_SEARCH_RESULTS
    assert dict(harness.upstream.searches()[0].url.params)["count"] == str(MAX_SEARCH_RESULTS)


def test_a_repeated_search_does_not_reach_open_meteo_again(harness, client):
    harness.upstream.add(httpx.Response(200, json={"results": [PARIS_ROW]}))
    for _ in range(4):
        client.get("/users/me/weather/cities?q=paris", headers=harness.bearer(harness.ana))
    assert len(harness.upstream.searches()) == 1


def test_a_failed_city_search_is_a_bounded_refusal(harness, client):
    harness.upstream.add(httpx.ConnectError("no route"))
    response = client.get(
        "/users/me/weather/cities?q=paris", headers=harness.bearer(harness.ana)
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "weather_upstream_unavailable"


# --- choosing and clearing a city --------------------------------------------------------


def test_saving_a_city_answers_the_settings_view_without_coordinates(harness, client):
    response = _save_paris(harness, client)
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "city"
    assert body["label"] == "Paris"
    assert body["city"] == {
        "name": "Paris",
        "region": "Ile-de-France",
        "country": "France",
        "timezone": "Europe/Paris",
    }
    assert body["browser"] is None
    assert "48.85" not in json.dumps(body)


def test_a_city_the_lookup_never_returned_is_refused(harness, client):
    harness.upstream.add(httpx.Response(200, json={"results": [PARIS_ROW]}))
    response = client.put(
        "/users/me/weather/city",
        headers=harness.bearer(harness.ana),
        json={"name": "Paris", "latitude": 0.0, "longitude": 0.0},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "unknown_city"
    assert client.get("/users/me/weather", headers=harness.bearer(harness.ana)).json()["state"] == (
        "no_location"
    )


@pytest.mark.parametrize(
    "body",
    [
        {"name": "Paris", "latitude": 91.0, "longitude": 2.35},
        {"name": "Paris", "latitude": 48.85, "longitude": 181.0},
        {"name": "", "latitude": 48.85, "longitude": 2.35},
        {"name": "Paris", "latitude": "48.85", "longitude": 2.35},
        {"name": "Paris", "latitude": 48.85},
        {"name": "Paris", "latitude": 48.85, "longitude": 2.35, "country": "Atlantis"},
    ],
)
def test_an_impossible_city_choice_is_refused(harness, client, body):
    response = client.put(
        "/users/me/weather/city", headers=harness.bearer(harness.ana), json=body
    )
    assert response.status_code == 422, body


def test_clearing_the_city_falls_back_to_the_browser_fix(harness, client):
    _save_paris(harness, client)
    client.put(
        "/users/me/weather/browser-location",
        headers=harness.bearer(harness.ana),
        json={"consent": True, "latitude": 51.5074, "longitude": -0.1278},
    )
    cleared = client.delete("/users/me/weather/city", headers=harness.bearer(harness.ana))
    assert cleared.status_code == 200
    body = cleared.json()
    assert body["city"] is None
    assert body["source"] == "browser"
    assert body["label"] == "Your current location"


def test_one_users_city_is_never_anothers(harness, client):
    _save_paris(harness, client)
    response = client.get("/users/me/weather", headers=harness.bearer(harness.bo))
    assert response.json()["location"] is None


# --- the browser position ----------------------------------------------------------------


def test_a_browser_position_needs_explicit_consent(harness, client):
    for body in (
        {"latitude": 51.5074, "longitude": -0.1278},
        {"consent": False, "latitude": 51.5074, "longitude": -0.1278},
        {"consent": "yes", "latitude": 51.5074, "longitude": -0.1278},
        {"consent": 1, "latitude": 51.5074, "longitude": -0.1278},
    ):
        response = client.put(
            "/users/me/weather/browser-location", headers=harness.bearer(harness.ana), json=body
        )
        assert response.status_code == 422, body
    assert client.get("/users/me/weather", headers=harness.bearer(harness.ana)).json()["state"] == (
        "no_location"
    )


@pytest.mark.parametrize(
    "body",
    [
        {"consent": True, "latitude": 91.0, "longitude": 0.0},
        {"consent": True, "latitude": -90.001, "longitude": 0.0},
        {"consent": True, "latitude": 0.0, "longitude": 180.5},
        {"consent": True, "latitude": "51.5", "longitude": 0.0},
        {"consent": True, "latitude": None, "longitude": 0.0},
        {"consent": True, "latitude": 51.5},
        {"consent": True, "latitude": 51.5, "longitude": 0.0, "accuracy": 12},
    ],
)
def test_an_impossible_browser_position_is_refused(harness, client, body):
    response = client.put(
        "/users/me/weather/browser-location", headers=harness.bearer(harness.ana), json=body
    )
    assert response.status_code == 422, body


def test_a_consented_browser_position_is_kept_briefly_and_never_echoed(harness, client):
    response = client.put(
        "/users/me/weather/browser-location",
        headers=harness.bearer(harness.ana),
        json={"consent": True, "latitude": 51.507351, "longitude": -0.127758},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "browser"
    assert body["label"] == "Your current location"
    assert body["city"] is None
    assert body["browser"] == {
        "updated_at": NOW,
        "expires_at": NOW + BROWSER_LOCATION_TTL_SECONDS,
    }
    # Neither the position the browser gave nor the rounded one it was kept as.
    rendered = json.dumps(body)
    for forbidden in ("51.507351", "51.51", "-0.127758", "-0.13", "latitude"):
        assert forbidden not in rendered, forbidden


def test_a_browser_position_stops_counting_when_it_expires(harness, client):
    client.put(
        "/users/me/weather/browser-location",
        headers=harness.bearer(harness.ana),
        json={"consent": True, "latitude": 51.5074, "longitude": -0.1278},
    )
    harness.upstream.add(httpx.Response(200, json=_forecast_body()))
    assert client.get("/users/me/weather", headers=harness.bearer(harness.ana)).json()["state"] == (
        "ok"
    )

    harness.now = NOW + BROWSER_LOCATION_TTL_SECONDS + 1
    later = client.get("/users/me/weather", headers=harness.bearer(harness.ana))
    assert later.json()["state"] == "no_location"


def test_clearing_the_browser_position_forgets_it(harness, client):
    client.put(
        "/users/me/weather/browser-location",
        headers=harness.bearer(harness.ana),
        json={"consent": True, "latitude": 51.5074, "longitude": -0.1278},
    )
    response = client.delete(
        "/users/me/weather/browser-location", headers=harness.bearer(harness.ana)
    )
    assert response.status_code == 200
    assert response.json()["browser"] is None
    assert response.json()["source"] is None


def test_a_browser_position_never_reaches_the_log(harness, client, caplog):
    with caplog.at_level(logging.DEBUG):
        client.put(
            "/users/me/weather/browser-location",
            headers=harness.bearer(harness.ana),
            json={"consent": True, "latitude": 51.507351, "longitude": -0.127758},
        )
        harness.upstream.add(httpx.Response(200, json=_forecast_body()))
        client.get("/users/me/weather", headers=harness.bearer(harness.ana))
    for forbidden in ("51.507351", "51.51", "-0.127758", harness.ana):
        assert forbidden not in caplog.text, forbidden


def test_the_location_setting_can_be_read_back(harness, client):
    empty = client.get("/users/me/weather/location", headers=harness.bearer(harness.ana))
    assert empty.status_code == 200
    assert empty.json() == {"source": None, "label": None, "city": None, "browser": None}

    _save_paris(harness, client)
    saved = client.get("/users/me/weather/location", headers=harness.bearer(harness.ana))
    assert saved.status_code == 200
    body = saved.json()
    assert body["source"] == "city"
    assert body["city"]["name"] == "Paris"
    assert body["browser"] is None
    assert "48.85" not in json.dumps(body)
    # Reading the setting is not reading the weather: no upstream call.
    assert harness.upstream.forecasts() == []


def test_the_location_setting_is_read_per_user(harness, client):
    _save_paris(harness, client)
    for_bo = client.get("/users/me/weather/location", headers=harness.bearer(harness.bo))
    assert for_bo.json()["city"] is None


# --- naming a shared browser position ----------------------------------------------------


def _share_london(harness, client) -> httpx.Response:
    return client.put(
        "/users/me/weather/browser-location",
        headers=harness.bearer(harness.ana),
        json={"consent": True, "latitude": 51.507351, "longitude": -0.127758},
    )


def _reverse_ok() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "address": {
                "road": "Westminster Bridge Road",
                "city": "London",
                "state": "England",
                "postcode": "SE1 7PB",
                "country": "United Kingdom",
            }
        },
    )


def test_a_shared_position_is_answered_with_the_nearby_place(harness, client):
    assert _share_london(harness, client).status_code == 200
    harness.upstream.add(httpx.Response(200, json=_forecast_body()))
    harness.upstream.add_reverse(_reverse_ok())

    response = client.get("/users/me/weather", headers=harness.bearer(harness.ana))

    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "ok"
    assert body["location"]["source"] == "browser"
    assert body["location"]["label"] == "London"
    assert body["location"]["region"] == "England"
    assert body["location"]["country"] == "United Kingdom"
    # Nothing finer than the place, and no position, ever comes back.
    raw = response.text
    assert "51.5" not in raw and "-0.12" not in raw and "51.507351" not in raw
    assert "Westminster Bridge Road" not in raw and "SE1 7PB" not in raw


def test_an_unavailable_reverse_lookup_keeps_the_generic_label(harness, client):
    assert _share_london(harness, client).status_code == 200
    harness.upstream.add(httpx.Response(200, json=_forecast_body()))
    harness.upstream.add_reverse(httpx.Response(503, text="nominatim is down"))

    body = client.get("/users/me/weather", headers=harness.bearer(harness.ana)).json()

    assert body["state"] == "ok"
    assert body["observation"]["temperature_c"] == 7.4
    assert body["location"]["label"] == "Your current location"
    assert body["location"]["region"] is None


def test_a_saved_city_still_wins_and_is_never_reverse_geocoded(harness, client):
    assert _share_london(harness, client).status_code == 200
    assert _save_paris(harness, client).status_code == 200
    harness.upstream.add(httpx.Response(200, json=_forecast_body()))

    body = client.get("/users/me/weather", headers=harness.bearer(harness.ana)).json()

    assert body["location"]["source"] == "city"
    assert body["location"]["label"] == "Paris"
    assert harness.upstream.reverses() == []


def test_the_position_never_reaches_a_log_through_the_reverse_lookup(harness, client, caplog):
    assert _share_london(harness, client).status_code == 200
    harness.upstream.add(httpx.Response(200, json=_forecast_body()))
    harness.upstream.add_reverse(_reverse_ok())

    with caplog.at_level(logging.DEBUG):
        client.get("/users/me/weather", headers=harness.bearer(harness.ana))

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "51.507351" not in logged and "51.51" not in logged and "-0.13" not in logged
