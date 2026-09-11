"""Reading calendar events and inbox messages from a user's connected accounts.

No test here reaches a network: every provider is played by an injected
``httpx`` transport that records what it was asked and answers on script.
Pinned properties:

* each provider is asked its own documented endpoint, over https only, with
  the stored access token as a bearer, within the window and limit requested;
* what comes back is reduced to bounded, plain-text summaries -- never a
  body, an attendee, an address list, HTML, or the provider's raw payload;
* an expired token is renewed with the stored refresh token when there is
  one, and reported as ``reconnect_required`` when there is not, without a
  provider call; a provider that refuses the renewed token is reported the
  same way, never guessed around;
* refusals, timeouts, oversized and malformed answers each become one
  :class:`ProviderDataError` with a bounded reason;
* Zoho is read through its own documented Calendar and Mail APIs, with its own
  authorization scheme, and a payload field Zoho does not document -- a place,
  a browser link, a read state in an unknown form -- is left empty rather than
  invented;
* nothing observable -- results, ``repr``, the log -- carries a token, a
  secret, or an email address that was not the account's own label.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs

import httpx
import pytest

from caal import profile_crypto
from caal.oauth_providers import load_provider_registry
from caal.profile_crypto import KeyRing
from caal.provider_connections import ConnectionStore
from caal.provider_data import (
    MAX_PREVIEW_LENGTH,
    MAX_TITLE_LENGTH,
    ProviderDataClient,
    ProviderDataError,
)
from caal.user_store import MEMBER, Actor, UserStore

SECRET = "s" * 48
NOW = 1_700_000_000  # 2023-11-14T22:13:20Z
START = datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)
END = START + timedelta(days=7)
ACCESS = "ya29.ACCESS-TOKEN-PLAINTEXT"
REFRESH = "1//REFRESH-TOKEN-PLAINTEXT"
CLIENT_SECRET = "client-secret-PLAINTEXT-value"
ENV = {
    "CAAL_OAUTH_GOOGLE_CLIENT_ID": "google-client-id.apps.googleusercontent.com",
    "CAAL_OAUTH_GOOGLE_CLIENT_SECRET": CLIENT_SECRET,
    "CAAL_OAUTH_MICROSOFT_CLIENT_ID": "ms-client-id",
    "CAAL_OAUTH_MICROSOFT_CLIENT_SECRET": CLIENT_SECRET,
    "CAAL_OAUTH_ZOHO_CLIENT_ID": "zoho-client-id",
    "CAAL_OAUTH_ZOHO_CLIENT_SECRET": CLIENT_SECRET,
    "CAAL_PUBLIC_ORIGIN": "https://jarvis.example.com",
}
GOOGLE_CALENDAR = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
GMAIL_MESSAGES = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
GRAPH_CALENDAR = "https://graph.microsoft.com/v1.0/me/calendarView"
GRAPH_MESSAGES = "https://graph.microsoft.com/v1.0/me/mailFolders/inbox/messages"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
MICROSOFT_TOKEN = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
GOOGLE_SCOPES = (
    "openid",
    "email",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
)
MICROSOFT_SCOPES = ("openid", "email", "offline_access", "Mail.Read", "Calendars.Read")
ZOHO_SCOPES = (
    "AaaServer.profile.READ",
    "ZohoMail.accounts.READ",
    "ZohoMail.messages.READ",
    "ZohoCalendar.calendar.READ",
    "ZohoCalendar.event.READ",
)
ZOHO_CALENDARS = "https://calendar.zoho.com/api/v1/calendars"
ZOHO_MAIL_ACCOUNTS = "https://mail.zoho.com/api/accounts"
ZOHO_TOKEN = "https://accounts.zoho.com/oauth/v2/token"
SECRETS = (ACCESS, REFRESH, CLIENT_SECRET)


class FakeProvider:
    """Scripted provider endpoints behind ``httpx.MockTransport``."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.routes: list[tuple[str, str, object]] = []

    def add(self, method: str, url_prefix: str, answer: object) -> None:
        self.routes.append((method, url_prefix, answer))

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        for method, prefix, answer in self.routes:
            if request.method == method and str(request.url).startswith(prefix):
                if callable(answer):
                    answer = answer(request)
                if isinstance(answer, Exception):
                    raise answer
                return answer  # type: ignore[return-value]
        return httpx.Response(404, json={"error": "unrouted"})

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def sent(self, method: str, url_prefix: str) -> list[httpx.Request]:
        return [
            r for r in self.requests if r.method == method and str(r.url).startswith(url_prefix)
        ]


def _default_scopes(provider: str) -> tuple[str, ...]:
    if provider == "google":
        return GOOGLE_SCOPES
    return ZOHO_SCOPES if provider == "zoho" else MICROSOFT_SCOPES


class Harness:
    def __init__(self, tmp_path, env: dict | None = None) -> None:
        self.now = NOW
        self.ring = KeyRing.from_env(profile_crypto.generate_key_material(version=1))
        self.users = UserStore(tmp_path / "assistant.sqlite3", keyring=self.ring)
        self.store = ConnectionStore(self.users, keyring=self.ring, state_secret=SECRET)
        self.registry = load_provider_registry(ENV if env is None else env)
        self.provider = FakeProvider()
        self.ana = self._user("ana@example.com")
        self.bo = self._user("bo@example.com")

    def _user(self, email: str) -> str:
        return self.users.create_user(
            email=email,
            display_name=email.split("@")[0],
            role=MEMBER,
            actor=Actor.system(),
            now=self.now,
        ).user_id

    def connect(self, user_id: str, provider: str = "google", **overrides):
        params = {
            "access_token": ACCESS,
            "refresh_token": REFRESH,
            "expires_in": 3600,
            "scopes": _default_scopes(provider),
            "provider_account_id": f"{provider}-account-1",
            "account_label": f"ana@{provider}.example",
            "now": self.now,
        }
        params.update(overrides)
        return self.store.complete_authorization(user_id, provider, **params)

    def client(self, **overrides) -> ProviderDataClient:
        params = {
            "transport": self.provider.transport,
            "clock": lambda: self.now,
        }
        params.update(overrides)
        return ProviderDataClient(self.store, self.registry, **params)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def capture_logs(caplog):
    """Attach caplog directly: ``voice_agent`` stops ``caal`` records propagating."""
    targets = [
        logging.getLogger(name)
        for name in ("caal.provider_data", "caal.provider_connections", "httpx", "httpcore")
    ]
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
def h(tmp_path):
    return Harness(tmp_path)


def assert_no_secret(text: str, *extra: str) -> None:
    for secret in (*SECRETS, *extra):
        assert secret not in text, secret


# --- Google Calendar ---------------------------------------------------------------------


def test_google_calendar_asks_the_primary_calendar_for_the_window_and_returns_summaries(
    h, caplog
) -> None:
    connection = h.connect(h.ana)
    h.provider.add(
        "GET",
        GOOGLE_CALENDAR,
        httpx.Response(
            200,
            json={
                "kind": "calendar#events",
                "items": [
                    {
                        "id": "evt1",
                        "summary": "  Standup\n with   team ",
                        "start": {"dateTime": "2023-11-15T09:00:00-07:00"},
                        "end": {"dateTime": "2023-11-15T09:30:00-07:00"},
                        "location": "Room 4",
                        "status": "confirmed",
                        "htmlLink": "https://www.google.com/calendar/event?eid=abc",
                        "description": "<b>secret body</b> with notes",
                        "attendees": [{"email": "someone@example.com"}],
                    },
                    {
                        "id": "evt2",
                        "summary": "Holiday",
                        "start": {"date": "2023-11-16"},
                        "end": {"date": "2023-11-17"},
                        "status": "tentative",
                    },
                    {
                        "id": "evt3",
                        "summary": "Gone",
                        "start": {"dateTime": "2023-11-15T12:00:00Z"},
                        "end": {"dateTime": "2023-11-15T13:00:00Z"},
                        "status": "cancelled",
                    },
                    {
                        "id": "evt4",
                        "start": {"dateTime": "2023-11-15T20:00:00Z"},
                        "end": {"dateTime": "2023-11-15T21:00:00Z"},
                        "htmlLink": "http://insecure.example/x",
                    },
                    {
                        "id": "evt5",
                        "summary": "T" * 300 + "\x00\x07" + chr(0x200B),
                        "start": {"dateTime": "2023-11-15T22:00:00Z"},
                        "end": {"dateTime": "2023-11-15T22:15:00.1234567Z"},
                        "location": "L" * 300,
                    },
                    {"id": "bad", "summary": "no start"},
                    "not-an-object",
                ],
            },
        ),
    )

    with caplog.at_level(logging.DEBUG):
        events = run(h.client().calendar_events(h.ana, connection, start=START, end=END, limit=25))

    (request,) = h.provider.requests
    assert request.method == "GET"
    assert request.headers["authorization"] == f"Bearer {ACCESS}"
    params = dict(request.url.params)
    assert params["timeMin"] == "2023-11-14T22:13:20Z"
    assert params["timeMax"] == "2023-11-21T22:13:20Z"
    assert params["singleEvents"] == "true"
    assert params["orderBy"] == "startTime"
    assert params["maxResults"] == "25"
    assert params["showDeleted"] == "false"

    # Soonest first, whatever order the provider answered in: the all-day event
    # on the 16th comes after everything on the 15th.
    assert [event.id for event in events] == ["evt1", "evt4", "evt5", "evt2"]
    first = events[0]
    assert first.title == "Standup with team"
    assert first.start == "2023-11-15T16:00:00Z" and first.end == "2023-11-15T16:30:00Z"
    assert first.all_day is False and first.location == "Room 4"
    assert first.link == "https://www.google.com/calendar/event?eid=abc"
    assert first.status == "confirmed"
    holiday = events[3]
    assert holiday.all_day is True
    assert holiday.start == "2023-11-16" and holiday.end == "2023-11-17"
    assert holiday.status == "tentative"
    untitled = events[1]
    assert untitled.title is None and untitled.link is None and untitled.status is None
    long = events[2]
    assert long.title == "T" * MAX_TITLE_LENGTH
    assert long.location == "L" * MAX_TITLE_LENGTH
    assert long.end == "2023-11-15T22:15:00Z"

    view = first.view()
    assert set(view) == {"id", "title", "start", "end", "all_day", "location", "link", "status"}
    wire = json.dumps([event.view() for event in events])
    assert "secret body" not in wire and "someone@example.com" not in wire
    assert "description" not in wire and "attendees" not in wire
    assert_no_secret(caplog.text, "Standup", "Room 4", "someone@example.com")
    assert_no_secret(repr(events))


def test_a_google_connection_without_the_calendar_scope_is_reported_without_a_call(h) -> None:
    connection = h.connect(
        h.ana,
        scopes=("openid", "email", "https://www.googleapis.com/auth/gmail.readonly"),
    )
    with pytest.raises(ProviderDataError) as excinfo:
        run(h.client().calendar_events(h.ana, connection, start=START, end=END, limit=10))
    assert excinfo.value.reason == "insufficient_scope"
    assert h.provider.requests == []

    # A row whose granted scopes are unknown is not second-guessed: the
    # provider decides, and its refusal is classified.
    legacy = h.connect(h.ana, provider_account_id="google-account-2", scopes=())
    h.provider.add(
        "GET",
        GOOGLE_CALENDAR,
        httpx.Response(
            403,
            json={
                "error": {
                    "code": 403,
                    "message": "Request had insufficient authentication scopes.",
                    "status": "PERMISSION_DENIED",
                    "errors": [{"reason": "insufficientPermissions"}],
                }
            },
        ),
    )
    with pytest.raises(ProviderDataError) as excinfo:
        run(h.client().calendar_events(h.ana, legacy, start=START, end=END, limit=10))
    assert excinfo.value.reason == "insufficient_scope"
    assert len(h.provider.requests) == 1


# --- Gmail ---------------------------------------------------------------------------------


def _gmail_detail(message_id, *, subject, sender, snippet, internal_date, unread):
    labels = ["INBOX", "UNREAD"] if unread else ["INBOX"]
    headers = [
        dict(name="Subject", value=subject),
        dict(name="From", value=sender),
        dict(name="To", value="ana@gmail.example"),
    ]
    payload = dict(mimeType="text/html", headers=headers, body=dict(data="SECRET-BODY-BASE64"))
    return httpx.Response(
        200,
        json=dict(
            id=message_id,
            threadId="t" + message_id,
            labelIds=labels,
            snippet=snippet,
            internalDate=str(internal_date),
            payload=payload,
        ),
    )


def test_gmail_lists_the_inbox_then_reads_metadata_only_and_returns_bounded_previews(
    h, caplog
) -> None:
    connection = h.connect(h.ana)
    h.provider.add(
        "GET",
        GMAIL_MESSAGES + "?",
        httpx.Response(
            200,
            json=dict(
                messages=[dict(id="m1"), dict(id="m2"), dict(id="m3"), dict(id="gone")],
                resultSizeEstimate=4,
            ),
        ),
    )
    h.provider.add(
        "GET",
        GMAIL_MESSAGES + "/m1",
        _gmail_detail(
            "m1",
            subject="Invoice & receipt",
            sender="Billing Team <billing@vendor.example>",
            snippet="Your invoice is attached &#39;thanks&#39; <b>bold</b>",
            internal_date=1_700_000_000_000,
            unread=True,
        ),
    )
    h.provider.add(
        "GET",
        GMAIL_MESSAGES + "/m2",
        _gmail_detail(
            "m2",
            subject="",
            sender="noreply@vendor.example",
            snippet="S" * 400,
            internal_date=1_699_990_000_000,
            unread=False,
        ),
    )
    h.provider.add(
        "GET",
        GMAIL_MESSAGES + "/m3",
        _gmail_detail(
            "m3",
            subject="Older",
            sender="Ana <ana@friend.example>",
            snippet="hi",
            internal_date=1_699_980_000_000,
            unread=False,
        ),
    )
    h.provider.add(
        "GET",
        GMAIL_MESSAGES + "/gone",
        httpx.Response(404, json=dict(error=dict(code=404, status="NOT_FOUND"))),
    )

    with caplog.at_level(logging.DEBUG):
        messages = run(h.client().inbox_messages(h.ana, connection, limit=10))

    (listing,) = h.provider.sent("GET", GMAIL_MESSAGES + "?")
    assert listing.headers["authorization"] == f"Bearer {ACCESS}"
    params = dict(listing.url.params)
    assert params["maxResults"] == "10" and params["labelIds"] == "INBOX"
    details = h.provider.sent("GET", GMAIL_MESSAGES + "/")
    assert len(details) == 4
    for request in details:
        assert request.url.params["format"] == "metadata"
        assert request.url.params.get_list("metadataHeaders") == ["Subject", "From"]
        assert "format=full" not in str(request.url) and "format=raw" not in str(request.url)

    assert [m.id for m in messages] == ["m1", "m2", "m3"]
    first = messages[0]
    assert first.subject == "Invoice & receipt"
    assert first.sender == "Billing Team"
    assert first.preview == "Your invoice is attached 'thanks' bold"
    assert first.received_at == "2023-11-14T22:13:20Z"
    assert first.unread is True and first.link is None
    second = messages[1]
    assert second.subject is None and second.sender == "noreply@vendor.example"
    assert second.preview == "S" * MAX_PREVIEW_LENGTH and second.unread is False
    assert messages[2].sender == "Ana"

    assert set(first.view()) == {
        "id",
        "subject",
        "sender",
        "preview",
        "received_at",
        "unread",
        "link",
    }
    wire = json.dumps([m.view() for m in messages])
    assert "SECRET-BODY" not in wire and "ana@gmail.example" not in wire and "<b>" not in wire
    assert_no_secret(caplog.text, "Invoice", "billing@vendor.example", "SECRET-BODY")
    assert_no_secret(repr(messages))


def test_a_google_connection_without_the_mail_scope_is_reported_without_a_call(h) -> None:
    connection = h.connect(
        h.ana, scopes=("openid", "email", "https://www.googleapis.com/auth/calendar.readonly")
    )
    with pytest.raises(ProviderDataError) as excinfo:
        run(h.client().inbox_messages(h.ana, connection, limit=5))
    assert excinfo.value.reason == "insufficient_scope"
    assert h.provider.requests == []


# --- Microsoft Graph -----------------------------------------------------------------------


def _graph_time(value: str, zone: str = "UTC") -> dict:
    return dict(dateTime=value, timeZone=zone)


def test_graph_calendar_view_is_asked_in_utc_and_reduced_to_summaries(h, caplog) -> None:
    connection = h.connect(h.ana, provider="microsoft")
    later = dict(
        id="AAMk-later",
        subject="Later",
        start=_graph_time("2023-11-16T09:00:00.0000000"),
        end=_graph_time("2023-11-16T10:00:00.0000000"),
        isAllDay=False,
        isCancelled=False,
        location=dict(displayName="Teams", address=dict(street="1 Secret St")),
        webLink="https://outlook.office365.com/owa/?itemid=AAMk-later",
        showAs="busy",
        attendees=[dict(emailAddress=dict(address="bo@contoso.example"))],
        body=dict(content="<p>SECRET</p>"),
        bodyPreview="SECRET preview",
    )
    first = dict(
        id="AAMk-first",
        subject="Planning",
        start=_graph_time("2023-11-15T14:00:00.0000000"),
        end=_graph_time("2023-11-15T15:00:00.0000000"),
        isAllDay=False,
        isCancelled=False,
        showAs="tentative",
        location=dict(displayName=""),
    )
    all_day = dict(
        id="AAMk-allday",
        subject="Offsite",
        start=_graph_time("2023-11-17T00:00:00.0000000"),
        end=_graph_time("2023-11-18T00:00:00.0000000"),
        isAllDay=True,
    )
    cancelled = dict(
        id="AAMk-cancelled",
        subject="Nope",
        start=_graph_time("2023-11-15T10:00:00.0000000"),
        isCancelled=True,
    )
    zoned = dict(
        id="AAMk-zone",
        subject="Local",
        start=_graph_time("2023-11-15T08:00:00.0000000", "America/Edmonton"),
        end=_graph_time("2023-11-15T09:00:00.0000000", "America/Edmonton"),
    )
    unknown_zone = dict(
        id="AAMk-windows",
        subject="Unknown zone",
        start=_graph_time("2023-11-15T08:00:00.0000000", "Mountain Standard Time"),
    )
    h.provider.add(
        "GET",
        GRAPH_CALENDAR,
        httpx.Response(
            200, json=dict(value=[later, first, all_day, cancelled, zoned, unknown_zone])
        ),
    )

    with caplog.at_level(logging.DEBUG):
        events = run(h.client().calendar_events(h.ana, connection, start=START, end=END, limit=20))

    (request,) = h.provider.requests
    assert request.headers["authorization"] == f"Bearer {ACCESS}"
    assert request.headers["prefer"] == 'outlook.timezone="UTC"'
    params = dict(request.url.params)
    assert params["startDateTime"] == "2023-11-14T22:13:20Z"
    assert params["endDateTime"] == "2023-11-21T22:13:20Z"
    assert params["$top"] == "20"
    selected = params["$select"].split(",")
    assert "body" not in selected and "attendees" not in selected and "bodyPreview" not in selected

    assert [e.id for e in events] == ["AAMk-first", "AAMk-zone", "AAMk-later", "AAMk-allday"]
    planning = events[0]
    assert planning.title == "Planning" and planning.status == "tentative"
    assert planning.start == "2023-11-15T14:00:00Z" and planning.end == "2023-11-15T15:00:00Z"
    assert planning.location is None and planning.all_day is False
    local = events[1]
    assert local.start == "2023-11-15T15:00:00Z"  # 08:00 in Edmonton is 15:00 UTC in November
    assert events[2].location == "Teams" and events[2].status == "confirmed"
    assert events[2].link == "https://outlook.office365.com/owa/?itemid=AAMk-later"
    offsite = events[3]
    assert offsite.all_day is True and offsite.start == "2023-11-17" and offsite.end == "2023-11-18"
    wire = json.dumps([e.view() for e in events])
    assert "SECRET" not in wire and "Secret St" not in wire and "bo@contoso.example" not in wire
    assert_no_secret(caplog.text, "Planning", "Teams")


def test_graph_inbox_is_read_newest_first_with_a_select_and_reduced_to_summaries(h) -> None:
    connection = h.connect(h.ana, provider="microsoft")
    newest = dict(
        id="AAMk-new",
        subject="Quarterly numbers",
        sender=dict(emailAddress=dict(name="Bo Chen", address="bo@contoso.example")),
        receivedDateTime="2023-11-14T21:00:00Z",
        isRead=False,
        bodyPreview="Here are the numbers\r\nfor Q3",
        webLink="https://outlook.office365.com/owa/?ItemID=AAMk-new",
        toRecipients=[dict(emailAddress=dict(address="ana@microsoft.example"))],
        body=dict(content="<html>SECRET</html>"),
    )
    newest["from"] = newest.pop("sender")
    older = dict(
        id="AAMk-old",
        subject=None,
        receivedDateTime="2023-11-14T20:00:00Z",
        isRead=True,
        bodyPreview="",
        webLink="javascript:alert(1)",
    )
    older["from"] = dict(emailAddress=dict(address="alerts@contoso.example"))
    h.provider.add(
        "GET",
        GRAPH_MESSAGES,
        httpx.Response(200, json=dict(value=[older, newest, dict(id="", subject="no id")])),
    )

    messages = run(h.client().inbox_messages(h.ana, connection, limit=25))

    (request,) = h.provider.requests
    params = dict(request.url.params)
    assert params["$top"] == "25"
    assert params["$orderby"] == "receivedDateTime desc"
    assert params["$select"] == "id,subject,from,receivedDateTime,isRead,bodyPreview,webLink"
    assert [m.id for m in messages] == ["AAMk-new", "AAMk-old"]
    new = messages[0]
    assert new.subject == "Quarterly numbers" and new.sender == "Bo Chen"
    assert new.preview == "Here are the numbers for Q3" and new.unread is True
    assert new.received_at == "2023-11-14T21:00:00Z"
    assert new.link == "https://outlook.office365.com/owa/?ItemID=AAMk-new"
    old = messages[1]
    assert old.subject is None and old.sender == "alerts@contoso.example"
    assert old.preview is None and old.unread is False and old.link is None
    wire = json.dumps([m.view() for m in messages])
    assert "SECRET" not in wire and "ana@microsoft.example" not in wire and "javascript" not in wire


def test_a_microsoft_connection_missing_a_scope_is_reported_without_a_call(h) -> None:
    mail_only = h.connect(
        h.ana, provider="microsoft", scopes=("openid", "email", "offline_access", "Mail.Read")
    )
    with pytest.raises(ProviderDataError) as excinfo:
        run(h.client().calendar_events(h.ana, mail_only, start=START, end=END, limit=5))
    assert excinfo.value.reason == "insufficient_scope"
    # Graph may echo scopes as full resource URIs; those count.
    full = h.connect(
        h.ana,
        provider="microsoft",
        provider_account_id="microsoft-account-2",
        scopes=("https://graph.microsoft.com/Mail.Read",),
    )
    with pytest.raises(ProviderDataError) as excinfo:
        run(h.client().calendar_events(h.ana, full, start=START, end=END, limit=5))
    assert excinfo.value.reason == "insufficient_scope"
    assert h.provider.requests == []


# --- token renewal -------------------------------------------------------------------------


def _token_answer(access: str, *, refresh: str | None = None, expires_in: int = 3600):
    body = dict(access_token=access, token_type="Bearer", expires_in=expires_in)
    if refresh is not None:
        body["refresh_token"] = refresh
    return httpx.Response(200, json=body)


def test_an_expired_token_is_renewed_with_the_refresh_token_and_the_renewal_is_stored(
    h, caplog
) -> None:
    connection = h.connect(h.ana)
    h.now += 3600 - 30  # inside the expiry skew: renew before sending
    h.provider.add("POST", GOOGLE_TOKEN, _token_answer("ya29.RENEWED", expires_in=1800))
    h.provider.add("GET", GOOGLE_CALENDAR, httpx.Response(200, json=dict(items=[])))

    with caplog.at_level(logging.DEBUG):
        events = run(h.client().calendar_events(h.ana, connection, start=START, end=END, limit=5))

    assert events == []
    renewal, read = h.provider.requests
    assert renewal.method == "POST" and str(renewal.url) == GOOGLE_TOKEN
    form = parse_qs(renewal.content.decode("utf-8"), strict_parsing=True)
    assert form["grant_type"] == ["refresh_token"]
    assert form["refresh_token"] == [REFRESH]
    assert form["client_id"] == [ENV["CAAL_OAUTH_GOOGLE_CLIENT_ID"]]
    assert form["client_secret"] == [CLIENT_SECRET]
    assert "authorization" not in [name.lower() for name in renewal.headers]
    assert read.headers["authorization"] == "Bearer ya29.RENEWED"
    stored = h.store.credentials(h.ana, connection.connection_id)
    assert stored.access_token == "ya29.RENEWED" and stored.refresh_token == REFRESH
    assert stored.token_expires_at == h.now + 1800
    assert_no_secret(caplog.text, "ya29.RENEWED")
    assert "Renewed the access token for a google connection" in caplog.text


def test_a_refused_fresh_token_is_renewed_once_and_a_rotated_refresh_token_is_kept(h) -> None:
    connection = h.connect(h.ana, provider="microsoft")
    refused = httpx.Response(401, json=dict(error=dict(code="InvalidAuthenticationToken")))
    answers = iter([refused, httpx.Response(200, json=dict(value=[]))])
    h.provider.add("GET", GRAPH_MESSAGES, lambda request: next(answers))
    h.provider.add(
        "POST",
        MICROSOFT_TOKEN,
        _token_answer("ms.RENEWED", refresh="ms.ROTATED", expires_in=4800),
    )

    messages = run(h.client().inbox_messages(h.ana, connection, limit=5))

    assert messages == []
    assert [(r.method, str(r.url).split("?")[0]) for r in h.provider.requests] == [
        ("GET", GRAPH_MESSAGES),
        ("POST", MICROSOFT_TOKEN),
        ("GET", GRAPH_MESSAGES),
    ]
    assert h.provider.requests[0].headers["authorization"] == f"Bearer {ACCESS}"
    assert h.provider.requests[2].headers["authorization"] == "Bearer ms.RENEWED"
    stored = h.store.credentials(h.ana, connection.connection_id)
    assert stored.access_token == "ms.RENEWED" and stored.refresh_token == "ms.ROTATED"
    assert stored.token_expires_at == h.now + 4800

    # A second refusal after the renewal is final: reconnect, never a loop.
    h.provider.routes.clear()
    h.provider.requests.clear()
    h.provider.add("GET", GRAPH_MESSAGES, refused)
    h.provider.add("POST", MICROSOFT_TOKEN, _token_answer("ms.AGAIN"))
    with pytest.raises(ProviderDataError) as excinfo:
        run(h.client().inbox_messages(h.ana, connection, limit=5))
    assert excinfo.value.reason == "reconnect_required"
    assert [r.method for r in h.provider.requests] == ["GET", "POST", "GET"]


def test_no_refresh_token_a_dead_grant_or_no_client_configuration_means_reconnect(
    h, caplog
) -> None:
    no_refresh = h.connect(h.ana, refresh_token=None)
    h.now += 3600
    with pytest.raises(ProviderDataError) as excinfo:
        run(h.client().calendar_events(h.ana, no_refresh, start=START, end=END, limit=5))
    assert excinfo.value.reason == "reconnect_required"
    assert h.provider.requests == []

    revoked = h.connect(
        h.ana, provider_account_id="google-account-2", account_label="two@google.example"
    )
    h.now += 3600
    h.provider.add(
        "POST",
        GOOGLE_TOKEN,
        httpx.Response(
            400,
            json=dict(error="invalid_grant", error_description="Token revoked. sk-live-LEAK"),
        ),
    )
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ProviderDataError) as excinfo:
            run(h.client().calendar_events(h.ana, revoked, start=START, end=END, limit=5))
    assert excinfo.value.reason == "reconnect_required"
    assert excinfo.value.provider_code == "invalid_grant"
    assert len(h.provider.requests) == 1
    assert "LEAK" not in caplog.text and "LEAK" not in str(excinfo.value)
    # A failed renewal leaves the stored tokens untouched.
    assert h.store.credentials(h.ana, revoked.connection_id).access_token == ACCESS

    # Without the application's client credentials nothing can renew; say so by name.
    unconfigured = ProviderDataClient(
        h.store, load_provider_registry(dict()), transport=h.provider.transport, clock=lambda: h.now
    )
    with pytest.raises(ProviderDataError) as excinfo:
        run(unconfigured.calendar_events(h.ana, revoked, start=START, end=END, limit=5))
    assert excinfo.value.reason == "not_configured"
    assert len(h.provider.requests) == 1


# --- bounded failures ----------------------------------------------------------------------


def test_refusals_timeouts_and_oversized_or_malformed_answers_are_bounded_reasons(
    h, caplog
) -> None:
    connection = h.connect(h.ana)
    too_big = h.client().max_response_bytes + 1
    cases = [
        (
            httpx.Response(500, json=dict(error=dict(code=500, status="INTERNAL"))),
            "provider_refused",
            "internal",
        ),
        (
            httpx.Response(429, json=dict(error=dict(errors=[dict(reason="rateLimitExceeded")]))),
            "provider_refused",
            "ratelimitexceeded",
        ),
        (
            httpx.Response(403, json=dict(error=dict(errors=[dict(reason="accessNotConfigured")]))),
            "provider_refused",
            "accessnotconfigured",
        ),
        (
            httpx.Response(302, headers=dict(location="https://elsewhere.example/")),
            "malformed_response",
            None,
        ),
        (httpx.Response(200, content=b"<html>not json</html>"), "malformed_response", None),
        (httpx.Response(200, content=b"x" * too_big), "malformed_response", None),
        (httpx.ReadTimeout("slow"), "transport", None),
        (httpx.ConnectError("refused"), "transport", None),
    ]
    for answer, reason, code in cases:
        h.provider.routes.clear()
        h.provider.requests.clear()
        h.provider.add("GET", GOOGLE_CALENDAR, answer)
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(ProviderDataError) as excinfo:
                run(h.client().calendar_events(h.ana, connection, start=START, end=END, limit=5))
        assert excinfo.value.reason == reason, answer
        assert excinfo.value.provider_code == code, answer
        assert len(h.provider.requests) == 1, answer
    assert_no_secret(caplog.text)


def test_a_read_that_exceeds_its_wall_clock_budget_is_a_transport_failure(h) -> None:
    connection = h.connect(h.ana)

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.5)
        return httpx.Response(200, json=dict(items=[]))

    client = ProviderDataClient(
        h.store,
        h.registry,
        transport=httpx.MockTransport(slow),
        clock=lambda: h.now,
        fetch_budget_seconds=0.05,
    )
    with pytest.raises(ProviderDataError) as excinfo:
        run(client.calendar_events(h.ana, connection, start=START, end=END, limit=5))
    assert excinfo.value.reason == "transport"


# --- Zoho Calendar ---------------------------------------------------------------------------


def _zoho_calendars() -> httpx.Response:
    """``GET /api/v1/calendars``: the user's own calendars."""
    return httpx.Response(
        200,
        json=dict(
            calendars=[
                dict(uid="cal-one", name="Personal", isdefault=True, category="own"),
                dict(uid="cal two/slash", name="Bad uid", isdefault=False),
                dict(name="No uid at all", isdefault=False),
            ]
        ),
    )


def test_zoho_calendar_lists_calendars_then_reads_upcoming_events_from_each(h, caplog) -> None:
    connection = h.connect(h.ana, provider="zoho", account_label="ana@zoho.example")
    h.provider.add("GET", ZOHO_CALENDARS + "?", _zoho_calendars())
    h.provider.add(
        "GET",
        ZOHO_CALENDARS + "/cal-one/events",
        httpx.Response(
            200,
            json=dict(
                events=[
                    dict(
                        uid="z2",
                        title="  Later\n  meeting ",
                        isallday=False,
                        start="20231116T090000",
                        end="20231116T100000",
                        dateandtime=dict(
                            timezone="Asia/Calcutta",
                            start="20231116T090000+0530",
                            end="20231116T100000+0530",
                        ),
                        attendees=[dict(email="someone@example.com", status="ACCEPTED")],
                        description="SECRET body of the event",
                    ),
                    dict(
                        uid="z1",
                        title="Standup",
                        isallday=False,
                        start="20231115T093000",
                        end="20231115T094500",
                        dateandtime=dict(
                            timezone="UTC", start="20231115T093000Z", end="20231115T094500Z"
                        ),
                    ),
                    dict(
                        uid="z3",
                        title="Holiday",
                        isallday=True,
                        start="20231117",
                        end="20231118",
                        dateandtime=dict(timezone="UTC", start="20231117", end="20231118"),
                    ),
                    dict(
                        uid="no-zone",
                        title="Unanchored",
                        isallday=False,
                        start="20231115T120000",
                        end="20231115T130000",
                    ),
                    dict(title="no uid", isallday=False, start="20231115T120000"),
                    "not-an-object",
                ]
            ),
        ),
    )

    with caplog.at_level(logging.DEBUG):
        events = run(h.client().calendar_events(h.ana, connection, start=START, end=END, limit=25))

    listing, *event_reads = h.provider.requests
    assert str(listing.url).startswith(ZOHO_CALENDARS)
    assert listing.headers["authorization"] == f"Zoho-oauthtoken {ACCESS}"
    assert [str(r.url).split("?")[0] for r in event_reads] == [ZOHO_CALENDARS + "/cal-one/events"]
    window = json.loads(dict(event_reads[0].url.params)["range"])
    assert window["start"] == "20231114T221320Z"
    assert window["end"] == "20231121T221320Z"

    assert [event.id for event in events] == ["z1", "z2", "z3"]
    first = events[0]
    assert first.start == "2023-11-15T09:30:00Z" and first.end == "2023-11-15T09:45:00Z"
    assert first.all_day is False and first.title == "Standup"
    offset = events[1]
    assert offset.title == "Later meeting"
    assert offset.start == "2023-11-16T03:30:00Z" and offset.end == "2023-11-16T04:30:00Z"
    holiday = events[2]
    assert holiday.all_day is True
    assert holiday.start == "2023-11-17" and holiday.end == "2023-11-18"

    wire = json.dumps([event.view() for event in events])
    assert "SECRET" not in wire and "someone@example.com" not in wire
    assert "attendees" not in wire and "description" not in wire
    assert_no_secret(caplog.text, "Standup", "someone@example.com", "ana@zoho.example")


def test_zoho_calendar_never_asks_for_more_than_the_documented_range(h) -> None:
    connection = h.connect(h.ana, provider="zoho")
    h.provider.add("GET", ZOHO_CALENDARS + "?", _zoho_calendars())
    h.provider.add(
        "GET", ZOHO_CALENDARS + "/cal-one/events", httpx.Response(200, json=dict(events=[]))
    )
    far = START + timedelta(days=61)
    run(h.client().calendar_events(h.ana, connection, start=START, end=far, limit=5))
    window = json.loads(dict(h.provider.requests[1].url.params)["range"])
    start = datetime.strptime(window["start"], "%Y%m%dT%H%M%SZ")
    end = datetime.strptime(window["end"], "%Y%m%dT%H%M%SZ")
    assert end - start <= timedelta(days=31), "Zoho documents a 31-day maximum range"


def test_a_zoho_connection_without_the_calendar_scopes_is_reported_without_a_call(h) -> None:
    connection = h.connect(
        h.ana, provider="zoho", scopes=("AaaServer.profile.READ", "ZohoMail.messages.READ")
    )
    with pytest.raises(ProviderDataError) as excinfo:
        run(h.client().calendar_events(h.ana, connection, start=START, end=END, limit=10))
    assert excinfo.value.reason == "insufficient_scope"
    assert h.provider.requests == []


def test_zoho_calendar_answering_an_unexpected_shape_is_reported_not_invented(h) -> None:
    connection = h.connect(h.ana, provider="zoho")
    h.provider.add("GET", ZOHO_CALENDARS + "?", httpx.Response(200, json=dict(calendars="nope")))
    with pytest.raises(ProviderDataError) as excinfo:
        run(h.client().calendar_events(h.ana, connection, start=START, end=END, limit=5))
    assert excinfo.value.reason == "malformed_response"


# --- Zoho Mail -------------------------------------------------------------------------------

ZOHO_ACCOUNT_ID = "2560636000000008002"
ZOHO_SEARCH = ZOHO_MAIL_ACCOUNTS + "/" + ZOHO_ACCOUNT_ID + "/messages/search"


def _zoho_mail_accounts() -> httpx.Response:
    return httpx.Response(
        200,
        json=dict(
            status=dict(code=200, description="success"),
            data=[
                dict(
                    accountId=ZOHO_ACCOUNT_ID,
                    primaryEmailAddress="ana@zoho.example",
                    accountDisplayName="Ana",
                    enabled=True,
                )
            ],
        ),
    )


def test_zoho_mail_discovers_the_account_then_reads_recent_mail_as_bounded_summaries(
    h, caplog
) -> None:
    connection = h.connect(h.ana, provider="zoho", account_label="ana@zoho.example")
    h.provider.add(
        "GET",
        ZOHO_SEARCH,
        httpx.Response(
            200,
            json=dict(
                status=dict(code=200, description="success"),
                data=[
                    dict(
                        messageId="1709887058769100001",
                        subject="Invoice &amp; receipt",
                        sender="Billing Team",
                        fromAddress="billing@vendor.example",
                        summary="Your invoice is attached &#39;thanks&#39; <b>bold</b>",
                        receivedTime="1700000000000",
                        status="unread",
                        toAddress="&quot;ana&quot;&lt;ana@zoho.example&gt;",
                        ccAddress="cc@vendor.example",
                        URI="https://mail.zoho.com/api/accounts/1/messages/1",
                    ),
                    dict(
                        messageId="1709883097133100003",
                        subject="",
                        sender="",
                        fromAddress="noreply@vendor.example",
                        summary="S" * 400,
                        receivedtime="1699990000000",
                        status="read",
                    ),
                    dict(subject="no id", receivedTime="1699980000000"),
                    "not-an-object",
                ],
            ),
        ),
    )
    h.provider.add("GET", ZOHO_MAIL_ACCOUNTS, _zoho_mail_accounts())

    with caplog.at_level(logging.DEBUG):
        messages = run(h.client().inbox_messages(h.ana, connection, limit=20))

    discovery, search = h.provider.requests
    assert str(discovery.url) == ZOHO_MAIL_ACCOUNTS
    assert discovery.headers["authorization"] == f"Zoho-oauthtoken {ACCESS}"
    params = dict(search.url.params)
    assert params["searchKey"] == "newMails"
    assert params["start"] == "1"
    assert params["limit"] == "20"

    assert [message.id for message in messages] == ["1709887058769100001", "1709883097133100003"]
    first = messages[0]
    assert first.subject == "Invoice & receipt"
    assert first.sender == "Billing Team"
    assert first.preview == "Your invoice is attached 'thanks' bold"
    assert first.received_at == "2023-11-14T22:13:20Z"
    assert first.unread is True
    second = messages[1]
    assert second.subject is None and second.sender == "noreply@vendor.example"
    assert second.unread is False
    assert len(second.preview) == MAX_PREVIEW_LENGTH

    wire = json.dumps([message.view() for message in messages])
    assert "ana@zoho.example" not in wire and "cc@vendor.example" not in wire
    assert "toAddress" not in wire and "ccAddress" not in wire
    assert_no_secret(caplog.text, "Invoice", "billing@vendor.example")


def test_a_zoho_connection_without_the_mail_scopes_is_reported_without_a_call(h) -> None:
    connection = h.connect(
        h.ana, provider="zoho", scopes=("AaaServer.profile.READ", "ZohoCalendar.event.READ")
    )
    with pytest.raises(ProviderDataError) as excinfo:
        run(h.client().inbox_messages(h.ana, connection, limit=10))
    assert excinfo.value.reason == "insufficient_scope"
    assert h.provider.requests == []


def test_zoho_mail_without_a_usable_account_is_reported_rather_than_guessed(h) -> None:
    connection = h.connect(h.ana, provider="zoho")
    h.provider.add("GET", ZOHO_MAIL_ACCOUNTS, httpx.Response(200, json=dict(data=[])))
    with pytest.raises(ProviderDataError) as excinfo:
        run(h.client().inbox_messages(h.ana, connection, limit=5))
    assert excinfo.value.reason == "malformed_response"
    assert len(h.provider.requests) == 1


def test_zoho_renews_an_expired_token_before_reading_like_every_other_provider(h) -> None:
    connection = h.connect(h.ana, provider="zoho", expires_in=1)
    h.now += 600
    h.provider.add(
        "POST",
        ZOHO_TOKEN,
        httpx.Response(200, json=dict(access_token="zoho-renewed-token", expires_in=3600)),
    )
    h.provider.add("GET", ZOHO_SEARCH, httpx.Response(200, json=dict(data=[])))
    h.provider.add("GET", ZOHO_MAIL_ACCOUNTS, _zoho_mail_accounts())
    messages = run(h.client().inbox_messages(h.ana, connection, limit=5))
    assert messages == []
    assert len(h.provider.sent("POST", ZOHO_TOKEN)) == 1
    discovery = h.provider.sent("GET", ZOHO_MAIL_ACCOUNTS)[0]
    assert discovery.headers["authorization"] == "Zoho-oauthtoken zoho-renewed-token"


# --- feeds across accounts -----------------------------------------------------------------


def test_feeds_read_every_account_and_report_each_outcome_without_hiding_the_others(
    h, caplog
) -> None:
    google = h.connect(h.ana)
    expired = h.connect(
        h.ana,
        provider_account_id="google-account-2",
        account_label="two@google.example",
        refresh_token=None,
        expires_in=1,
    )
    microsoft = h.connect(h.ana, provider="microsoft")
    zoho = h.connect(h.ana, provider="zoho", account_label="ana@zoho.example")
    theirs = h.connect(h.bo, account_label="bo@google.example")
    h.now += 600
    event = dict(
        id="g1",
        summary="Google",
        start=dict(dateTime="2023-11-15T10:00:00Z"),
        end=dict(dateTime="2023-11-15T11:00:00Z"),
    )
    h.provider.add("GET", GOOGLE_CALENDAR, httpx.Response(200, json=dict(items=[event])))
    h.provider.add(
        "GET", GRAPH_CALENDAR, httpx.Response(503, json=dict(error=dict(code="ServiceUnavailable")))
    )
    h.provider.add("GET", ZOHO_CALENDARS + "?", httpx.Response(200, json=dict(calendars=[])))

    with caplog.at_level(logging.DEBUG):
        results = run(
            h.client().calendar_feed(
                h.ana, [google, expired, microsoft, zoho], start=START, end=END, limit=10
            )
        )

    by_id = dict((r.connection.connection_id, r) for r in results)
    assert list(by_id) == [c.connection_id for c in (google, expired, microsoft, zoho)]
    assert by_id[google.connection_id].status == "ok"
    assert [e.id for e in by_id[google.connection_id].events] == ["g1"]
    assert by_id[expired.connection_id].status == "reconnect_required"
    assert by_id[expired.connection_id].reason == "reconnect_required"
    assert by_id[microsoft.connection_id].status == "unavailable"
    assert by_id[microsoft.connection_id].reason == "provider_refused"
    assert by_id[microsoft.connection_id].error.provider_code == "serviceunavailable"
    assert by_id[zoho.connection_id].status == "ok"
    assert by_id[zoho.connection_id].events == ()
    assert theirs.connection_id not in by_id
    called = sorted(str(r.url).split("?")[0] for r in h.provider.requests)
    assert called == sorted([GOOGLE_CALENDAR, GRAPH_CALENDAR, ZOHO_CALENDARS])
    assert "Reading calendar from microsoft failed: provider_refused" in caplog.text
    assert_no_secret(
        caplog.text, "Google", "two@google.example", "ana@zoho.example", "bo@google.example"
    )

    # The inbox feed follows the same rules.
    h.provider.routes.clear()
    h.provider.requests.clear()
    h.provider.add("GET", GMAIL_MESSAGES + "?", httpx.Response(200, json=dict(messages=[])))
    h.provider.add("GET", GRAPH_MESSAGES, httpx.Response(200, json=dict(value=[])))
    h.provider.add("GET", ZOHO_SEARCH, httpx.Response(200, json=dict(data=[])))
    h.provider.add("GET", ZOHO_MAIL_ACCOUNTS, _zoho_mail_accounts())
    inbox = run(h.client().inbox_feed(h.ana, [google, microsoft, zoho], limit=10))
    assert [r.status for r in inbox] == ["ok", "ok", "ok"]
    assert all(r.messages == () for r in inbox)


def test_windows_limits_and_client_bounds_are_validated(h) -> None:
    connection = h.connect(h.ana)
    client = h.client()
    naive = datetime(2023, 11, 14)
    far = START + timedelta(days=90)
    for start, end, limit in (
        (naive, END, 5),
        (START, START, 5),
        (START, far, 5),
        (START, END, 0),
        (START, END, 51),
        (START, END, True),
    ):
        with pytest.raises(ValueError):
            run(client.calendar_events(h.ana, connection, start=start, end=end, limit=limit))
    with pytest.raises(ValueError):
        run(client.inbox_messages(h.ana, connection, limit=0))
    for kwargs in (
        dict(timeout_seconds=0),
        dict(timeout_seconds=31),
        dict(max_response_bytes=10),
        dict(fetch_budget_seconds=0),
        dict(fetch_budget_seconds=True),
    ):
        with pytest.raises(ValueError):
            ProviderDataClient(h.store, h.registry, **kwargs)
    with pytest.raises(ValueError):
        ProviderDataError(reason="nope")
    with pytest.raises(ValueError):
        ProviderDataError(provider_code="Bad Code")
    assert h.provider.requests == []
