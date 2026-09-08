"""HTTP contract for the dashboard's connected-account feeds.

``/users/me/dashboard/calendar`` and ``/users/me/dashboard/inbox`` sit behind
the same internal trust boundary as the rest of the identity API: the BFF
proves itself with a single-use signed principal, the backend loads the user
from its own database, and only that user's live connections are read. Every
account is reported on its own -- items, or the bounded reason there are
none -- and one account's failure never hides another's data. Nothing in a
response or a log line is a token, a body, or an address that was not the
account's own label.

No test here talks to a real provider: the reads go through an injected
``httpx`` transport that plays Google and Microsoft.
"""

from __future__ import annotations

import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from caal import connections_api, profile_crypto, user_api, webhooks
from caal.connections_api import ConnectionsRuntime
from caal.internal_auth import AUDIENCE_BACKEND, RateLimiter, mint_principal
from caal.oauth_providers import load_provider_registry
from caal.profile_crypto import KeyRing
from caal.provider_data import ProviderDataClient
from caal.security_config import MultiUserConfig
from caal.user_api import IdentityRuntime
from caal.user_store import MEMBER, Actor, UserStore

SECRET = "s" * 48
BOOTSTRAP = "cesarc@mexcantech.com"
NOW = 1_700_000_000  # 2023-11-14T22:13:20Z
ACCESS = "ya29.ACCESS-TOKEN-PLAINTEXT"
REFRESH = "1//REFRESH-TOKEN-PLAINTEXT"
CLIENT_SECRET = "client-secret-PLAINTEXT-value"
ENV = {
    "CAAL_OAUTH_GOOGLE_CLIENT_ID": "google-client-id.apps.googleusercontent.com",
    "CAAL_OAUTH_GOOGLE_CLIENT_SECRET": CLIENT_SECRET,
    "CAAL_OAUTH_MICROSOFT_CLIENT_ID": "ms-client-id",
    "CAAL_OAUTH_MICROSOFT_CLIENT_SECRET": CLIENT_SECRET,
    "CAAL_PUBLIC_ORIGIN": "https://jarvis.example.com",
}
GOOGLE_CALENDAR = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
GMAIL_MESSAGES = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
GRAPH_CALENDAR = "https://graph.microsoft.com/v1.0/me/calendarView"
GRAPH_MESSAGES = "https://graph.microsoft.com/v1.0/me/mailFolders/inbox/messages"
GOOGLE_SCOPES = (
    "openid",
    "email",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
)
MICROSOFT_SCOPES = ("openid", "email", "offline_access", "Mail.Read", "Calendars.Read")


class FakeProvider:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.routes: list[tuple[str, str, object]] = []

    def add(self, method: str, url_prefix: str, answer: object) -> None:
        self.routes.append((method, url_prefix, answer))

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        for method, prefix, answer in self.routes:
            if request.method == method and str(request.url).startswith(prefix):
                if isinstance(answer, Exception):
                    raise answer
                return answer  # type: ignore[return-value]
        return httpx.Response(404, json=dict(error="unrouted"))

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


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
        self.runtime = IdentityRuntime(
            self.config,
            store=self.store,
            mutation_limiter=RateLimiter(limit=100, window_seconds=60),
            clock=lambda: self.now,
        )
        self.registry = load_provider_registry(ENV)
        self.provider = FakeProvider()
        self.connections = ConnectionsRuntime(self.runtime, providers=self.registry)
        self.connections.data_client = ProviderDataClient(
            self.connections.store,
            self.registry,
            transport=self.provider.transport,
            clock=lambda: self.now,
        )
        self.ana = self.user("ana@example.com")
        self.bo = self.user("bo@example.com")

    def user(self, email: str) -> str:
        return self.store.create_user(
            email=email,
            display_name=email.split("@")[0],
            role=MEMBER,
            actor=Actor.system(),
            now=self.now,
        ).user_id

    def connect(self, user_id: str, provider: str = "google", **overrides):
        params = dict(
            access_token=ACCESS,
            refresh_token=REFRESH,
            expires_in=3600,
            scopes=GOOGLE_SCOPES if provider == "google" else MICROSOFT_SCOPES,
            provider_account_id=f"{provider}-account-1",
            account_label=f"ana@{provider}.example",
            now=self.now,
        )
        params.update(overrides)
        return self.connections.store.complete_authorization(user_id, provider, **params)

    def bearer(self, user_id: str) -> dict[str, str]:
        token = mint_principal(
            secret=SECRET, subject=user_id, audience=AUDIENCE_BACKEND, now=self.now
        )
        return dict(Authorization=f"Bearer {token}")


def _install(harness: Harness) -> None:
    webhooks.app.dependency_overrides[user_api.get_runtime] = lambda: harness.runtime
    webhooks.app.dependency_overrides[connections_api.get_connections_runtime] = (
        lambda: harness.connections
    )


def _uninstall() -> None:
    webhooks.app.dependency_overrides.pop(user_api.get_runtime, None)
    webhooks.app.dependency_overrides.pop(connections_api.get_connections_runtime, None)


@pytest.fixture(autouse=True)
def capture_logs(caplog):
    """Attach caplog directly: ``voice_agent`` stops ``caal`` records propagating."""
    names = (
        "caal.dashboard_api",
        "caal.provider_data",
        "caal.connections_api",
        "caal.provider_connections",
        "httpx",
    )
    targets = [logging.getLogger(name) for name in names]
    previous = [(t.level, t.propagate) for t in targets]
    for target in targets:
        target.setLevel(logging.DEBUG)
        target.addHandler(caplog.handler)
    try:
        yield
    finally:
        for target, (level, propagate) in zip(targets, previous):
            target.removeHandler(caplog.handler)
            target.setLevel(level)
            target.propagate = propagate


@pytest.fixture
def harness(tmp_path):
    return Harness(tmp_path)


@pytest.fixture
def client(harness):
    _install(harness)
    try:
        with TestClient(webhooks.app) as test_client:
            yield test_client
    finally:
        _uninstall()


def _event(event_id: str, start: str, end: str, title: str) -> dict:
    return dict(id=event_id, summary=title, start=dict(dateTime=start), end=dict(dateTime=end))


def _graph_event(event_id: str, start: str, end: str, title: str) -> dict:
    return dict(
        id=event_id,
        subject=title,
        start=dict(dateTime=start, timeZone="UTC"),
        end=dict(dateTime=end, timeZone="UTC"),
        isAllDay=False,
        isCancelled=False,
        webLink="https://outlook.office365.com/owa/?itemid=" + event_id,
    )


# --- gate and authentication ------------------------------------------------------------

PATHS = ("/users/me/dashboard/calendar", "/users/me/dashboard/inbox")
CALENDAR_KEYS = set(["generated_at", "window_start", "window_end", "accounts", "events"])
INBOX_KEYS = set(["generated_at", "accounts", "messages", "unread_count"])
EVENT_KEYS = set(
    ["id", "connection_id", "provider", "title", "start", "end", "all_day", "location", "link"]
    + ["status"]
)
MESSAGE_KEYS = set(
    ["id", "connection_id", "provider", "subject", "sender", "preview", "received_at", "unread"]
    + ["link"]
)


def test_feeds_fail_closed_without_configuration_and_require_a_single_use_principal(
    harness,
) -> None:
    webhooks.app.dependency_overrides[user_api.get_runtime] = lambda: None
    try:
        with TestClient(webhooks.app) as client:
            for path in PATHS:
                assert client.get(path, headers=harness.bearer(harness.ana)).status_code == 503
    finally:
        _uninstall()
    _install(harness)
    try:
        with TestClient(webhooks.app) as client:
            forged = mint_principal(secret="x" * 48, subject=harness.ana, audience=AUDIENCE_BACKEND)
            for path in PATHS:
                assert client.get(path).status_code == 401, path
                bad = dict(Authorization="Bearer nope")
                assert client.get(path, headers=bad).status_code == 401, path
                stolen = dict(Authorization="Bearer " + forged)
                assert client.get(path, headers=stolen).status_code == 401, path
                headers = harness.bearer(harness.ana)
                assert client.get(path, headers=headers).status_code == 200, path
                assert client.get(path, headers=headers).status_code == 401, path  # single use
            assert harness.provider.requests == []
    finally:
        _uninstall()


# --- calendar -----------------------------------------------------------------------------


def test_calendar_feed_merges_every_account_in_time_order_and_reports_each_account(
    harness, client, caplog
) -> None:
    google = harness.connect(harness.ana)
    microsoft = harness.connect(harness.ana, provider="microsoft")
    expired = harness.connect(
        harness.ana,
        provider_account_id="google-account-2",
        account_label="two@google.example",
        refresh_token=None,
        expires_in=1,
    )
    zoho = harness.connect(
        harness.ana,
        provider="zoho",
        scopes=("ZohoCalendar.event.READ",),
        account_label="ana@zoho.example",
    )
    harness.connect(harness.bo, account_label="bo@google.example")
    harness.now += 600
    items = [
        _event("g-later", "2023-11-16T09:00:00Z", "2023-11-16T10:00:00Z", "Google later"),
        _event("g-soon", "2023-11-15T01:00:00Z", "2023-11-15T02:00:00Z", "Google soon"),
    ]
    harness.provider.add("GET", GOOGLE_CALENDAR, httpx.Response(200, json=dict(items=items)))
    graph = [
        _graph_event("m-mid", "2023-11-15T12:00:00.0000000", "2023-11-15T13:00:00.0000000", "Mid"),
    ]
    harness.provider.add("GET", GRAPH_CALENDAR, httpx.Response(200, json=dict(value=graph)))

    with caplog.at_level(logging.DEBUG):
        response = client.get(
            "/users/me/dashboard/calendar?days=3&limit=10", headers=harness.bearer(harness.ana)
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == CALENDAR_KEYS
    assert body["generated_at"] == harness.now
    assert body["window_start"] == "2023-11-14T22:23:20Z"
    assert body["window_end"] == "2023-11-17T22:23:20Z"
    assert [(e["id"], e["provider"]) for e in body["events"]] == [
        ("g-soon", "google"),
        ("m-mid", "microsoft"),
        ("g-later", "google"),
    ]
    soon = body["events"][0]
    assert set(soon) == EVENT_KEYS
    assert soon["connection_id"] == google.connection_id and soon["title"] == "Google soon"
    assert body["events"][1]["connection_id"] == microsoft.connection_id
    assert body["events"][1]["link"].startswith("https://outlook.office365.com/")

    accounts = dict((a["connection_id"], a) for a in body["accounts"])
    assert list(accounts) == [
        google.connection_id,
        microsoft.connection_id,
        expired.connection_id,
        zoho.connection_id,
    ]
    assert accounts[google.connection_id] == dict(
        connection_id=google.connection_id,
        provider="google",
        account_label="ana@google.example",
        status="ok",
        reason=None,
        count=2,
    )
    assert accounts[microsoft.connection_id]["status"] == "ok"
    assert accounts[microsoft.connection_id]["count"] == 1
    assert accounts[expired.connection_id]["status"] == "reconnect_required"
    assert accounts[expired.connection_id]["reason"] == "reconnect_required"
    assert accounts[expired.connection_id]["count"] == 0
    # This deployment has no Zoho application configured, so the account's
    # stored access cannot be renewed and it says so rather than inventing a
    # read. The Zoho adapters themselves are covered in test_provider_data.
    assert accounts[zoho.connection_id]["status"] == "not_configured"
    assert accounts[zoho.connection_id]["reason"] == "not_configured"

    # The providers were asked for exactly this window, with this user's tokens only.
    calls = harness.provider.requests
    called = sorted(str(r.url).split("?")[0] for r in calls)
    assert called == sorted([GOOGLE_CALENDAR, GRAPH_CALENDAR])
    for call in calls:
        assert call.headers["authorization"] == "Bearer " + ACCESS
    google_call = next(r for r in calls if str(r.url).startswith(GOOGLE_CALENDAR))
    params = dict(google_call.url.params)
    assert params["timeMin"] == "2023-11-14T22:23:20Z"
    assert params["timeMax"] == "2023-11-17T22:23:20Z"
    assert params["maxResults"] == "10"

    # Nothing of bo's, and nothing secret, anywhere observable.
    assert "bo@google.example" not in response.text
    for secret in (ACCESS, REFRESH, CLIENT_SECRET):
        assert secret not in response.text and secret not in caplog.text
    hidden = ("Google soon", "Mid", "ana@google.example", "ana@zoho.example", "two@google.example")
    for word in hidden:
        assert word not in caplog.text, word
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "access-control-allow-origin" not in [k.lower() for k in response.headers]


# --- inbox ----------------------------------------------------------------------------------


def test_inbox_feed_merges_newest_first_and_counts_unread(harness, client) -> None:
    google = harness.connect(harness.ana)
    microsoft = harness.connect(harness.ana, provider="microsoft")
    harness.provider.add(
        "GET", GMAIL_MESSAGES + "?", httpx.Response(200, json=dict(messages=[dict(id="g1")]))
    )
    headers = [
        dict(name="Subject", value="From Gmail"),
        dict(name="From", value="Ana <ana@friend.example>"),
    ]
    gmail = dict(
        id="g1",
        labelIds=["INBOX", "UNREAD"],
        snippet="Gmail snippet",
        internalDate="1700000000000",
        payload=dict(headers=headers),
    )
    harness.provider.add("GET", GMAIL_MESSAGES + "/g1", httpx.Response(200, json=gmail))
    newer = dict(
        id="m1",
        subject="From Outlook",
        receivedDateTime="2023-11-14T22:20:00Z",
        isRead=True,
        bodyPreview="Outlook preview",
        webLink="https://outlook.office365.com/owa/?ItemID=m1",
    )
    newer["from"] = dict(emailAddress=dict(name="Bo", address="bo@contoso.example"))
    harness.provider.add("GET", GRAPH_MESSAGES, httpx.Response(200, json=dict(value=[newer])))

    response = client.get("/users/me/dashboard/inbox?limit=5", headers=harness.bearer(harness.ana))

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == INBOX_KEYS
    assert [(m["id"], m["provider"]) for m in body["messages"]] == [
        ("m1", "microsoft"),
        ("g1", "google"),
    ]
    outlook = body["messages"][0]
    assert set(outlook) == MESSAGE_KEYS
    assert outlook["connection_id"] == microsoft.connection_id and outlook["sender"] == "Bo"
    assert outlook["unread"] is False
    assert outlook["link"].startswith("https://outlook.office365.com/")
    mail = body["messages"][1]
    assert mail["connection_id"] == google.connection_id and mail["subject"] == "From Gmail"
    assert mail["unread"] is True and mail["preview"] == "Gmail snippet" and mail["link"] is None
    assert body["unread_count"] == 1
    assert [a["status"] for a in body["accounts"]] == ["ok", "ok"]
    assert [a["count"] for a in body["accounts"]] == [1, 1]
    graph_call = next(r for r in harness.provider.requests if str(r.url).startswith(GRAPH_MESSAGES))
    assert dict(graph_call.url.params)["$top"] == "5"
    assert "bo@contoso.example" not in response.text  # the name labels the sender


# --- bounds ---------------------------------------------------------------------------------


def test_query_bounds_are_enforced_and_defaults_apply(harness, client) -> None:
    harness.connect(harness.ana)
    harness.provider.add("GET", GOOGLE_CALENDAR, httpx.Response(200, json=dict(items=[])))
    harness.provider.add("GET", GMAIL_MESSAGES + "?", httpx.Response(200, json=dict(messages=[])))
    for query in ("days=0", "days=32", "limit=0", "limit=51", "limit=abc", "days=1.5"):
        response = client.get(
            "/users/me/dashboard/calendar?" + query, headers=harness.bearer(harness.ana)
        )
        assert response.status_code == 422, query
    for query in ("limit=0", "limit=51", "limit=x"):
        response = client.get(
            "/users/me/dashboard/inbox?" + query, headers=harness.bearer(harness.ana)
        )
        assert response.status_code == 422, query
    assert harness.provider.requests == []

    calendar = client.get("/users/me/dashboard/calendar", headers=harness.bearer(harness.ana))
    assert calendar.status_code == 200, calendar.text
    assert calendar.json()["window_end"] == "2023-11-21T22:13:20Z"  # seven days by default
    assert calendar.json()["events"] == []
    (google_call,) = harness.provider.requests
    assert dict(google_call.url.params)["maxResults"] == "25"

    inbox = client.get("/users/me/dashboard/inbox", headers=harness.bearer(harness.ana))
    assert inbox.status_code == 200, inbox.text
    assert inbox.json()["messages"] == [] and inbox.json()["unread_count"] == 0
    listing = harness.provider.requests[-1]
    assert dict(listing.url.params)["maxResults"] == "15"  # Gmail metadata fan-out is capped


def test_the_connections_runtime_builds_a_bounded_data_client_when_none_is_injected(
    harness,
) -> None:
    runtime = ConnectionsRuntime(harness.runtime, providers=harness.registry)
    assert isinstance(runtime.data, ProviderDataClient)
    assert runtime.data is runtime.data
    assert runtime.data.timeout_seconds == 10.0
    assert runtime.data.fetch_budget_seconds == 20.0
    assert harness.connections.data is harness.connections.data_client
