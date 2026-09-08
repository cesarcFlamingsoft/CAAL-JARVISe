"""Naming the place a browser position falls in, without saying where it is.

Nothing here talks to Nominatim: the upstream is an injected ``httpx``
transport that records exactly what was asked for. What is being pinned down
is the contract the Weather widget depends on -- one bounded request to one
exact https host, a nearby place name or an honest nothing, a cache that stops
the same position being asked about twice, and a global spacing that keeps
JARVIS inside the public Nominatim usage policy.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
import pytest

from caal.reverse_geocode import (
    CACHE_TTL_SECONDS,
    FAILURE_CACHE_TTL_SECONDS,
    MAX_CACHED_PLACES,
    MAX_QUEUE_WAIT_SECONDS,
    MAX_RESPONSE_BYTES,
    MIN_REQUEST_INTERVAL_SECONDS,
    NOMINATIM_URL,
    USER_AGENT,
    NearbyPlace,
    ReverseGeocoder,
    is_allowed_url,
)

LONDON = (51.51, -0.13)


def run(coro):
    return asyncio.run(coro)


class FakeNominatim:
    """Plays Nominatim. Records every request; answers from a scripted queue."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.answers: list[object] = []
        self.default: object = httpx.Response(200, json=_address_body())

    def add(self, answer: object, times: int = 1) -> None:
        self.answers.extend([answer] * times)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.answers.pop(0) if self.answers else self.default
        if isinstance(answer, Exception):
            raise answer
        return answer  # type: ignore[return-value]

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def _address_body(**overrides) -> dict:
    address = {
        "road": "Westminster Bridge Road",
        "house_number": "12",
        "suburb": "Lambeth",
        "city": "London",
        "state": "England",
        "postcode": "SE1 7PB",
        "country": "United Kingdom",
        "country_code": "gb",
    }
    address.update(overrides)
    return {
        "place_id": 1234,
        "lat": "51.5074",
        "lon": "-0.1278",
        "display_name": "12, Westminster Bridge Road, Lambeth, London, England, SE1 7PB",
        "address": address,
    }


class Clock:
    """Wall-clock seconds under the control of the test, advanced by the sleeper."""

    def __init__(self, now: float = 1_700_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class Sleeper:
    """Stands in for asyncio.sleep: records the wait and advances the clock."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.slept: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.clock.now += seconds


@pytest.fixture(autouse=True)
def unspaced():
    """Each test starts with no earlier lookup holding the global slot."""
    ReverseGeocoder.reset_pacing()
    yield
    ReverseGeocoder.reset_pacing()


@pytest.fixture
def upstream() -> FakeNominatim:
    return FakeNominatim()


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def sleeper(clock: Clock) -> Sleeper:
    return Sleeper(clock)


@pytest.fixture
def geocoder(upstream: FakeNominatim, clock: Clock, sleeper: Sleeper) -> ReverseGeocoder:
    return ReverseGeocoder(transport=upstream.transport, clock=clock, sleep=sleeper)


# --- the request -------------------------------------------------------------------------


def test_the_lookup_is_one_bounded_documented_request(
    geocoder: ReverseGeocoder, upstream: FakeNominatim
):
    place = run(geocoder.nearby(*LONDON))

    assert place is not None
    assert len(upstream.requests) == 1
    request = upstream.requests[0]
    assert request.method == "GET"
    url = request.url
    assert url.scheme == "https"
    assert url.host == "nominatim.openstreetmap.org"
    assert url.path == "/reverse"
    # Only the position already rounded to two decimals, and the documented
    # switches for a city-level answer.
    assert dict(url.params) == {
        "lat": "51.51",
        "lon": "-0.13",
        "format": "jsonv2",
        "zoom": "10",
        "addressdetails": "1",
        "accept-language": "en",
    }
    headers = {name.lower(): value for name, value in request.headers.items()}
    assert "authorization" not in headers
    assert "cookie" not in headers
    assert headers["user-agent"] == USER_AGENT


def test_the_user_agent_is_distinctive_and_names_nobody():
    # The Nominatim policy asks for an identifiable application; it does not
    # ask for a person, and this one names none.
    assert "CAAL" in USER_AGENT and "JARVIS" in USER_AGENT
    assert "@" not in USER_AGENT
    assert len(USER_AGENT) <= 120


def test_the_precise_position_is_never_sent(geocoder: ReverseGeocoder, upstream: FakeNominatim):
    run(geocoder.nearby(51.5073509, -0.1277583))

    sent = str(upstream.requests[0].url)
    assert "51.5073509" not in sent and "-0.1277583" not in sent
    assert "lat=51.51" in sent


@pytest.mark.parametrize(
    "url",
    [
        "http://nominatim.openstreetmap.org/reverse",
        "https://nominatim.openstreetmap.org.evil.example/reverse",
        "https://evil.example/reverse",
        "https://user@nominatim.openstreetmap.org/reverse",
        "https://nominatim.openstreetmap.org:8443/reverse",
        "//nominatim.openstreetmap.org/reverse",
        None,
    ],
)
def test_only_the_one_exact_host_may_be_contacted(url):
    assert is_allowed_url(url) is False
    assert is_allowed_url(NOMINATIM_URL) is True


def test_a_redirect_is_refused_rather_than_followed(
    geocoder: ReverseGeocoder, upstream: FakeNominatim
):
    upstream.add(httpx.Response(302, headers={"location": "https://evil.example/reverse"}))

    assert run(geocoder.nearby(*LONDON)) is None
    assert len(upstream.requests) == 1


def test_an_answer_over_the_byte_cap_is_dropped(upstream: FakeNominatim, clock: Clock):
    geocoder = ReverseGeocoder(transport=upstream.transport, clock=clock, max_response_bytes=2048)
    upstream.add(httpx.Response(200, json=_address_body(city="L" * 4096)))

    assert run(geocoder.nearby(*LONDON)) is None


def test_the_byte_cap_is_bounded():
    assert MAX_RESPONSE_BYTES <= 64 * 1024


# --- what a lookup says ------------------------------------------------------------------


def test_the_nearby_place_is_named_from_the_address(geocoder: ReverseGeocoder):
    place = run(geocoder.nearby(*LONDON))

    assert place == NearbyPlace(label="London", region="England", country="United Kingdom")


def test_the_street_and_house_number_are_never_part_of_the_name(geocoder: ReverseGeocoder):
    place = run(geocoder.nearby(*LONDON))

    assert place is not None
    for field in (place.label, place.region or "", place.country or ""):
        assert "Westminster Bridge Road" not in field
        assert "12" not in field
        assert "SE1 7PB" not in field


@pytest.mark.parametrize(
    "address, expected",
    [
        ({"town": "Ashford"}, "Ashford"),
        ({"village": "Grasmere"}, "Grasmere"),
        ({"municipality": "Kortrijk"}, "Kortrijk"),
        ({"county": "Kent"}, "Kent"),
        ({"state": "England"}, "England"),
    ],
)
def test_a_place_without_a_city_falls_back_to_the_coarser_name(
    upstream: FakeNominatim, clock: Clock, address: dict, expected: str
):
    geocoder = ReverseGeocoder(transport=upstream.transport, clock=clock)
    body = _address_body()
    body["address"] = {"country": "Somewhere", **address}
    upstream.add(httpx.Response(200, json=body))

    place = run(geocoder.nearby(*LONDON))

    assert place is not None and place.label == expected


def test_a_place_name_is_plain_and_bounded(upstream: FakeNominatim, clock: Clock):
    geocoder = ReverseGeocoder(transport=upstream.transport, clock=clock)
    upstream.add(httpx.Response(200, json=_address_body(city="Lon\ndon" + " x" * 200, state=42)))

    place = run(geocoder.nearby(*LONDON))

    assert place is not None
    assert "\n" not in place.label and len(place.label) <= 120
    # A region that is not text at all is no region, never a stringified number.
    assert place.region is None


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json=["a", "list"]),
        httpx.Response(200, json={"address": "not an object"}),
        httpx.Response(200, json={"address": {"country": "France"}}),
        httpx.Response(200, json={"error": "Unable to geocode"}),
        httpx.Response(429, json={"error": "Too Many Requests"}),
        httpx.Response(500, text="upstream on fire"),
        httpx.ConnectError("no route"),
        httpx.ReadTimeout("too slow"),
    ],
)
def test_an_unusable_answer_is_nothing_rather_than_a_guess(
    upstream: FakeNominatim, clock: Clock, answer: object
):
    geocoder = ReverseGeocoder(transport=upstream.transport, clock=clock)
    upstream.add(answer)

    assert run(geocoder.nearby(*LONDON)) is None


@pytest.mark.parametrize("latitude, longitude", [(91.0, 0.0), (0.0, 181.0), ("51", 0.0), (None, 0)])
def test_an_impossible_position_is_never_a_request(
    geocoder: ReverseGeocoder, upstream: FakeNominatim, latitude, longitude
):
    assert run(geocoder.nearby(latitude, longitude)) is None
    assert upstream.requests == []


def test_a_failure_never_carries_the_upstream_body_into_a_log(
    geocoder: ReverseGeocoder, upstream: FakeNominatim, caplog
):
    upstream.add(httpx.Response(500, text="secret upstream detail"))

    with caplog.at_level(logging.DEBUG, logger="caal.reverse_geocode"):
        assert run(geocoder.nearby(*LONDON)) is None

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "secret upstream detail" not in logged
    assert "51.51" not in logged and "-0.13" not in logged


# --- the cache ---------------------------------------------------------------------------


def test_the_same_position_is_never_asked_about_twice(
    geocoder: ReverseGeocoder, upstream: FakeNominatim
):
    first = run(geocoder.nearby(*LONDON))
    for _ in range(5):
        # The same place once rounded, so the same one cache entry.
        assert run(geocoder.nearby(51.5137, -0.1338)) == first

    assert len(upstream.requests) == 1


def test_a_cached_name_is_forgotten_when_it_expires(
    geocoder: ReverseGeocoder, upstream: FakeNominatim, clock: Clock
):
    run(geocoder.nearby(*LONDON))
    clock.now += CACHE_TTL_SECONDS + 1
    run(geocoder.nearby(*LONDON))

    assert len(upstream.requests) == 2


def test_the_cache_never_outlives_the_browser_position_it_describes():
    from caal.weather_store import BROWSER_LOCATION_TTL_SECONDS

    assert 0 < CACHE_TTL_SECONDS <= BROWSER_LOCATION_TTL_SECONDS


def test_a_failed_lookup_is_not_retried_immediately(
    geocoder: ReverseGeocoder, upstream: FakeNominatim, clock: Clock
):
    upstream.add(httpx.Response(500, text="down"))

    assert run(geocoder.nearby(*LONDON)) is None
    assert run(geocoder.nearby(*LONDON)) is None
    assert len(upstream.requests) == 1

    clock.now += FAILURE_CACHE_TTL_SECONDS + 1
    assert run(geocoder.nearby(*LONDON)) is not None
    assert len(upstream.requests) == 2


def test_the_cache_cannot_grow_without_bound(
    geocoder: ReverseGeocoder, upstream: FakeNominatim, clock: Clock
):
    for step in range(MAX_CACHED_PLACES * 2):
        clock.now += MIN_REQUEST_INTERVAL_SECONDS
        run(geocoder.nearby(round(0.01 * step, 2), 0.0))

    assert geocoder.cached_places <= MAX_CACHED_PLACES


def test_one_position_never_becomes_the_name_of_another(
    geocoder: ReverseGeocoder, upstream: FakeNominatim, clock: Clock
):
    body = _address_body(city="Paris", state="Ile-de-France", country="France")
    upstream.add(httpx.Response(200, json=body))
    first = run(geocoder.nearby(48.86, 2.35))

    clock.now += MIN_REQUEST_INTERVAL_SECONDS
    second = run(geocoder.nearby(*LONDON))

    assert first is not None and first.label == "Paris"
    assert second is not None and second.label == "London"


# --- the usage policy --------------------------------------------------------------------


def test_lookups_of_different_places_are_spaced_a_second_apart(
    geocoder: ReverseGeocoder, upstream: FakeNominatim, sleeper: Sleeper
):
    assert MIN_REQUEST_INTERVAL_SECONDS >= 1.0

    run(geocoder.nearby(*LONDON))
    run(geocoder.nearby(48.86, 2.35))

    assert len(upstream.requests) == 2
    assert sleeper.slept == [MIN_REQUEST_INTERVAL_SECONDS]


def test_a_lookup_that_would_queue_too_long_is_skipped_rather_than_stalled(
    upstream: FakeNominatim, clock: Clock, sleeper: Sleeper
):
    geocoder = ReverseGeocoder(
        transport=upstream.transport,
        clock=clock,
        sleep=sleeper,
        min_interval_seconds=MAX_QUEUE_WAIT_SECONDS + 2,
    )

    assert run(geocoder.nearby(*LONDON)) is not None
    # The next free slot is further off than a dashboard read is willing to
    # wait, so the widget keeps its truthful generic label instead.
    assert run(geocoder.nearby(48.86, 2.35)) is None
    assert len(upstream.requests) == 1


def test_a_cached_answer_never_waits_on_the_rate_limit(
    geocoder: ReverseGeocoder, sleeper: Sleeper
):
    run(geocoder.nearby(*LONDON))
    run(geocoder.nearby(*LONDON))

    assert sleeper.slept == []


def test_the_spacing_is_global_across_every_geocoder(
    upstream: FakeNominatim, clock: Clock, sleeper: Sleeper
):
    # Two clients in one process are still one caller as far as Nominatim is
    # concerned, so the spacing is shared rather than per-instance.
    one = ReverseGeocoder(transport=upstream.transport, clock=clock, sleep=sleeper)
    two = ReverseGeocoder(transport=upstream.transport, clock=clock, sleep=sleeper)

    run(one.nearby(*LONDON))
    run(two.nearby(48.86, 2.35))

    assert sleeper.slept == [MIN_REQUEST_INTERVAL_SECONDS]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_seconds": 0},
        {"timeout_seconds": 120},
        {"max_response_bytes": 10},
        {"cache_ttl_seconds": 0},
        {"cache_ttl_seconds": 30 * 86400},
        {"min_interval_seconds": 0.1},
    ],
)
def test_bad_construction_is_refused(kwargs: dict):
    with pytest.raises(ValueError):
        ReverseGeocoder(**kwargs)
