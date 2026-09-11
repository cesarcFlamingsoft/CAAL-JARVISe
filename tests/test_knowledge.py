"""The knowledge layer JARVIS answers email and calendar questions from, and its tools.

No test here reaches a network: every provider is played by an injected
httpx transport that records what it was asked. Pinned properties:

* a question is answered from the per-user index while the freshness ledger
  says the index is fresh and covers the window; otherwise the provider is
  read once, within a bounded budget, and concurrent askers share that read;
* a failed or slow refresh falls back to what was indexed and says so, and
  no connected accounts is an honest empty answer, never invented data;
* nothing but the bounded dashboard-safe fields is ever persisted: a body a
  provider volunteers is not written, not returned, and not spoken;
* one user never hears another, and a revoked connection vanishes;
* the tools are user-scoped, speak short truthful summaries without ids or
  links, and are awaited by the LLM node.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest

from caal import profile_crypto
from caal.knowledge import (
    CALENDAR_TTL_SECONDS,
    INBOX_TTL_SECONDS,
    KnowledgeService,
    matches,
    resolve_account,
)
from caal.knowledge_store import KnowledgeStore
from caal.oauth_providers import load_provider_registry
from caal.profile_crypto import KeyRing
from caal.provider_connections import ConnectionStore, ProviderConnection
from caal.provider_data import ProviderDataClient
from caal.tools import knowledge_tools
from caal.tools.registry import create_default_registry
from caal.user_scope import UserScope
from caal.user_store import MEMBER, Actor, UserStore

# caal.llm re-exports the llm_node *function*; load the module itself.
llm_node_module = importlib.import_module("caal.llm.llm_node")

SECRET = "s" * 48
NOW = 1_700_000_000  # 2023-11-14T22:13:20Z, a Tuesday
ACCESS = "ya29.ACCESS-TOKEN-PLAINTEXT"
REFRESH = "1//REFRESH-TOKEN-PLAINTEXT"
CLIENT_SECRET = "client-secret-PLAINTEXT-value"
BODY = "SECRET-BODY-TEXT-NEVER-STORED"
ENV = dict(
    CAAL_OAUTH_GOOGLE_CLIENT_ID="google-client-id.apps.googleusercontent.com",
    CAAL_OAUTH_GOOGLE_CLIENT_SECRET=CLIENT_SECRET,
    CAAL_OAUTH_MICROSOFT_CLIENT_ID="ms-client-id",
    CAAL_OAUTH_MICROSOFT_CLIENT_SECRET=CLIENT_SECRET,
    CAAL_PUBLIC_ORIGIN="https://jarvis.example.com",
)
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
START = datetime.fromtimestamp(NOW, tz=timezone.utc)


def run(coro):
    return asyncio.run(coro)


class FakeProvider:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.routes: list[tuple[str, str, object]] = []

    def add(self, method: str, url_prefix: str, answer: object) -> None:
        self.routes.insert(0, (method, url_prefix, answer))

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        for method, prefix, answer in self.routes:
            if request.method == method and str(request.url).startswith(prefix):
                if isinstance(answer, Exception):
                    raise answer
                return answer  # type: ignore[return-value]
        return httpx.Response(404, json=dict(error="unrouted"))

    def calls(self, prefix: str) -> list[httpx.Request]:
        return [r for r in self.requests if str(r.url).startswith(prefix)]

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


class Harness:
    def __init__(self, tmp_path, timezone_name: str = "UTC") -> None:
        self.now = NOW
        self.db = tmp_path / "assistant.sqlite3"
        self.keyring = KeyRing.from_env(profile_crypto.generate_key_material(version=1))
        self.users = UserStore(self.db, keyring=self.keyring)
        self.connections = ConnectionStore(self.users, keyring=self.keyring, state_secret=SECRET)
        self.registry = load_provider_registry(ENV)
        self.provider = FakeProvider()
        self.data = ProviderDataClient(
            self.connections,
            self.registry,
            transport=self.provider.transport,
            clock=lambda: self.now,
        )
        self.index = KnowledgeStore(self.users, keyring=self.keyring)
        self.service = KnowledgeService(
            self.connections,
            self.data,
            self.index,
            clock=lambda: self.now,
            timezone=timezone_name,
        )
        self.ana = self.user("ana@example.com")
        self.bo = self.user("bo@example.com")

    def user(self, email: str) -> str:
        return self.users.create_user(
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
            expires_in=36000,
            scopes=GOOGLE_SCOPES if provider == "google" else MICROSOFT_SCOPES,
            provider_account_id=f"{provider}-account-1",
            account_label=f"ana@{provider}.example",
            now=self.now,
        )
        params.update(overrides)
        return self.connections.complete_authorization(user_id, provider, **params)

    def google_events(self, *items: dict) -> None:
        self.provider.add("GET", GOOGLE_CALENDAR, httpx.Response(200, json=dict(items=list(items))))

    def gmail(self, *messages: dict) -> None:
        refs = [dict(id=m["id"]) for m in messages]
        self.provider.add(
            "GET", GMAIL_MESSAGES + "?", httpx.Response(200, json=dict(messages=refs))
        )
        for m in messages:
            self.provider.add("GET", GMAIL_MESSAGES + "/" + m["id"], httpx.Response(200, json=m))


def gevent(event_id: str, start: str, end: str, title: str, **extra) -> dict:
    item = dict(id=event_id, summary=title, start=dict(dateTime=start), end=dict(dateTime=end))
    item.update(extra)
    return item


def gmail_message(message_id: str, subject: str, sender: str, received: str, **extra) -> dict:
    stamp = int(datetime.fromisoformat(received.replace("Z", "+00:00")).timestamp() * 1000)
    headers = [dict(name="Subject", value=subject), dict(name="From", value=sender)]
    item = dict(
        id=message_id,
        labelIds=["INBOX", "UNREAD"],
        snippet=f"Preview of {subject}",
        internalDate=str(stamp),
        payload=dict(headers=headers, body=dict(data=BODY), parts=[dict(body=dict(data=BODY))]),
        raw=BODY,
    )
    item.update(extra)
    return item


@pytest.fixture(autouse=True)
def capture_logs(caplog):
    names = (
        "caal.knowledge",
        "caal.knowledge_store",
        "caal.provider_data",
        "caal.tools",
        "caal.llm.llm_node",
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
def h(tmp_path):
    return Harness(tmp_path)


@pytest.fixture(autouse=True)
def unbind_tools():
    knowledge_tools.reset()
    yield
    knowledge_tools.reset()


# --- freshness --------------------------------------------------------------------------


def test_upcoming_events_read_the_provider_once_then_answer_from_the_index(h) -> None:
    google = h.connect(h.ana)
    h.google_events(
        gevent("g-later", "2023-11-16T09:00:00Z", "2023-11-16T10:00:00Z", "Dentist"),
        gevent("g-soon", "2023-11-15T01:00:00Z", "2023-11-15T02:00:00Z", "Standup"),
    )
    end = START + timedelta(days=7)

    first = run(h.service.upcoming_events(h.ana, start=START, end=end, limit=10))
    second = run(h.service.upcoming_events(h.ana, start=START, end=end, limit=10))

    assert [e.item.id for e in first.events] == ["g-soon", "g-later"]
    assert [e.item.id for e in second.events] == ["g-soon", "g-later"]
    assert first.connected and not first.stale
    assert [a.connection_id for a in first.accounts] == [google.connection_id]
    account = first.accounts[0]
    assert account.status == "ok" and account.stale is False and account.count == 2
    assert account.account_label == "ana@google.example" and account.provider == "google"
    assert len(h.provider.calls(GOOGLE_CALENDAR)) == 1, "the second answer came from the index"
    # The one read asked from the start of today, so the morning is answerable too,
    # and well past the window asked about, so a wider question is still covered.
    (call,) = h.provider.calls(GOOGLE_CALENDAR)
    params = dict(call.url.params)
    assert params["timeMin"] == "2023-11-14T00:00:00Z"
    assert params["timeMax"] >= "2023-11-28T00:00:00Z"

    # Past the TTL the provider is asked again, and a vanished event vanishes.
    h.now += CALENDAR_TTL_SECONDS
    h.google_events(gevent("g-later", "2023-11-16T09:00:00Z", "2023-11-16T10:00:00Z", "Dentist"))
    third = run(h.service.upcoming_events(h.ana, start=START, end=end, limit=10))
    assert [e.item.id for e in third.events] == ["g-later"]
    assert len(h.provider.calls(GOOGLE_CALENDAR)) == 2

    # A window the last read did not cover is read again even while fresh.
    far = run(
        h.service.upcoming_events(h.ana, start=START, end=START + timedelta(days=31), limit=10)
    )
    assert len(h.provider.calls(GOOGLE_CALENDAR)) == 3
    assert far.window_end_ts >= int((START + timedelta(days=31)).timestamp())


def test_recent_messages_are_indexed_without_bodies_and_searchable(h, caplog) -> None:
    h.connect(h.ana)
    h.gmail(
        gmail_message(
            "m2", "Quarterly numbers", "Bo Example <bo@contoso.example>", "2023-11-14T22:00:00Z"
        ),
        gmail_message(
            "m1",
            "Lunch tomorrow?",
            "ana.friend@example.com",
            "2023-11-14T20:00:00Z",
            labelIds=["INBOX"],
        ),
    )

    answer = run(h.service.recent_messages(h.ana, limit=10))

    assert [m.item.id for m in answer.messages] == ["m2", "m1"]
    assert answer.unread_count == 1 and answer.connected and not answer.stale
    assert answer.messages[0].item.sender == "Bo Example"
    assert answer.messages[0].item.preview == "Preview of Quarterly numbers"
    assert answer.messages[1].item.sender == "ana.friend@example.com"

    # Only the bounded summary exists anywhere: not in the database, not in logs.
    raw = h.db.read_bytes()
    assert BODY.encode() not in raw
    assert b"Quarterly numbers" not in raw and b"Preview of" not in raw
    assert BODY not in caplog.text and "Quarterly" not in caplog.text
    for secret in (ACCESS, REFRESH, CLIENT_SECRET):
        assert secret.encode() not in raw  # tokens are stored encrypted
        assert secret not in caplog.text

    # Search answers from the index by subject, sender or preview, without a provider call.
    gmail_calls = len(h.provider.calls(GMAIL_MESSAGES))
    found = run(h.service.recent_messages(h.ana, limit=10, query="quarterly"))
    assert [m.item.id for m in found.messages] == ["m2"]
    by_sender = run(h.service.recent_messages(h.ana, limit=10, query="Bo"))
    assert [m.item.id for m in by_sender.messages] == ["m2"]
    by_preview = run(h.service.recent_messages(h.ana, limit=10, query="preview of lunch"))
    assert [m.item.id for m in by_preview.messages] == ["m1"]
    unread = run(h.service.recent_messages(h.ana, limit=10, unread_only=True))
    assert [m.item.id for m in unread.messages] == ["m2"]
    one = run(h.service.recent_messages(h.ana, limit=10, message_id="m1"))
    assert [m.item.id for m in one.messages] == ["m1"]
    assert run(h.service.recent_messages(h.ana, limit=10, query="invoice")).messages == []
    assert len(h.provider.calls(GMAIL_MESSAGES)) == gmail_calls

    h.now += INBOX_TTL_SECONDS
    run(h.service.recent_messages(h.ana, limit=10))
    assert len(h.provider.calls(GMAIL_MESSAGES)) > gmail_calls


def test_a_failed_or_slow_refresh_falls_back_to_the_index_and_says_so(h) -> None:
    google = h.connect(h.ana)
    h.google_events(gevent("g1", "2023-11-15T09:00:00Z", "2023-11-15T10:00:00Z", "Standup"))
    end = START + timedelta(days=7)
    run(h.service.upcoming_events(h.ana, start=START, end=end, limit=10))

    h.now += CALENDAR_TTL_SECONDS
    h.provider.add("GET", GOOGLE_CALENDAR, httpx.Response(503, json=dict(error="down")))
    answer = run(h.service.upcoming_events(h.ana, start=START, end=end, limit=10))
    assert [e.item.id for e in answer.events] == ["g1"]
    assert answer.stale is True
    (account,) = answer.accounts
    assert account.connection_id == google.connection_id
    assert account.status == "unavailable" and account.reason == "provider_refused"
    assert account.stale is True and account.count == 1
    assert account.indexed_at == NOW

    class Slow:
        async def calendar_feed(self, *args, **kwargs):
            await asyncio.sleep(0.5)
            raise AssertionError("never reached")

        inbox_feed = calendar_feed

    slow = KnowledgeService(
        h.connections, Slow(), h.index, clock=lambda: h.now, refresh_budget_seconds=0.01
    )
    h.now += CALENDAR_TTL_SECONDS
    answer = run(slow.upcoming_events(h.ana, start=START, end=end, limit=10))
    assert [e.item.id for e in answer.events] == ["g1"] and answer.stale is True
    assert answer.accounts[0].reason == "refresh_timeout"
    # A permanent failure is reported as such, and no data is invented.
    h.provider.add("GET", GOOGLE_CALENDAR, httpx.Response(401, json=dict(error="bad")))
    h.provider.add(
        "POST",
        "https://oauth2.googleapis.com/token",
        httpx.Response(400, json=dict(error="invalid_grant")),
    )
    h.now += CALENDAR_TTL_SECONDS
    answer = run(h.service.upcoming_events(h.ana, start=START, end=end, limit=10))
    assert answer.accounts[0].status == "reconnect_required" and answer.stale is True


# --- isolation and honesty ----------------------------------------------------------------


def test_answers_are_isolated_per_user_and_a_revoked_connection_vanishes(h) -> None:
    ana_google = h.connect(h.ana)
    h.connect(h.bo, account_label="bo@google.example", provider_account_id="google-bo")
    h.google_events(gevent("shared-id", "2023-11-15T09:00:00Z", "2023-11-15T10:00:00Z", "Ana only"))
    end = START + timedelta(days=7)
    ana = run(h.service.upcoming_events(h.ana, start=START, end=end, limit=10))
    assert [e.item.title for e in ana.events] == ["Ana only"]

    h.google_events(gevent("shared-id", "2023-11-15T11:00:00Z", "2023-11-15T12:00:00Z", "Bo only"))
    bo = run(h.service.upcoming_events(h.bo, start=START, end=end, limit=10))
    assert [e.item.title for e in bo.events] == ["Bo only"]
    assert [a.account_label for a in bo.accounts] == ["bo@google.example"]
    # Each user saw only their own account read, with their own connection.
    again = run(h.service.upcoming_events(h.ana, start=START, end=end, limit=10))
    assert [e.item.title for e in again.events] == ["Ana only"]
    assert [a.connection_id for a in again.accounts] == [ana_google.connection_id]

    h.connections.revoke_connection(h.ana, ana_google.connection_id, now=h.now)
    h.service.forget_connection(h.ana, ana_google.connection_id)
    gone = run(h.service.upcoming_events(h.ana, start=START, end=end, limit=10))
    assert gone.events == [] and gone.accounts == [] and gone.connected is False
    assert (
        h.index.events(h.ana, [ana_google.connection_id], start_ts=0, end_ts=NOW * 2, limit=5) == []
    )
    # Rows that were never forgotten still cannot surface once the connection is not live.
    bo_google = h.connections.list_connections(h.bo)[0]
    h.connections.revoke_connection(h.bo, bo_google.connection_id, now=h.now)
    assert run(h.service.upcoming_events(h.bo, start=START, end=end, limit=10)).events == []


def test_no_connected_accounts_is_an_honest_empty_answer_without_a_provider_call(h) -> None:
    events = run(
        h.service.upcoming_events(h.ana, start=START, end=START + timedelta(days=1), limit=5)
    )
    mail = run(h.service.recent_messages(h.ana, limit=5))
    assert events.connected is False and events.events == [] and events.accounts == []
    assert mail.connected is False and mail.messages == [] and mail.unread_count == 0
    assert h.provider.requests == []
    with pytest.raises(ValueError):
        run(h.service.recent_messages("nobody", limit=5))
    with pytest.raises(ValueError):
        run(h.service.recent_messages(h.ana, limit=0))


def test_concurrent_askers_share_one_provider_read(h) -> None:
    h.connect(h.ana)
    h.google_events(gevent("g1", "2023-11-15T09:00:00Z", "2023-11-15T10:00:00Z", "Standup"))
    end = START + timedelta(days=7)

    async def both():
        return await asyncio.gather(
            h.service.upcoming_events(h.ana, start=START, end=end, limit=10),
            h.service.upcoming_events(h.ana, start=START, end=end, limit=10),
        )

    first, second = run(both())
    assert [e.item.id for e in first.events] == ["g1"] == [e.item.id for e in second.events]
    assert len(h.provider.calls(GOOGLE_CALENDAR)) == 1


def test_dashboard_reads_write_through_to_the_index(h) -> None:
    google = h.connect(h.ana)
    microsoft = h.connect(h.ana, provider="microsoft")
    h.google_events(gevent("g1", "2023-11-15T09:00:00Z", "2023-11-15T10:00:00Z", "Standup"))
    h.provider.add("GET", GRAPH_CALENDAR, httpx.Response(503, json=dict(error="down")))
    h.gmail(gmail_message("m1", "Hello", "Bo <bo@contoso.example>", "2023-11-14T21:00:00Z"))
    h.provider.add("GET", GRAPH_MESSAGES, httpx.Response(200, json=dict(value=[])))
    connections = h.connections.list_connections(h.ana)
    end = START + timedelta(days=7)

    fetches = run(h.service.calendar_feed(h.ana, connections, start=START, end=end, limit=25))
    assert [(f.connection.connection_id, f.status) for f in fetches] == [
        (google.connection_id, "ok"),
        (microsoft.connection_id, "unavailable"),
    ]
    inbox = run(h.service.inbox_feed(h.ana, connections, limit=20))
    assert [f.status for f in inbox] == ["ok", "ok"]
    calls = len(h.provider.requests)

    answer = run(h.service.upcoming_events(h.ana, start=START, end=end, limit=10))
    assert [e.item.id for e in answer.events] == ["g1"]
    statuses = dict((a.connection_id, a.status) for a in answer.accounts)
    assert statuses == dict(
        [(google.connection_id, "ok"), (microsoft.connection_id, "unavailable")]
    )
    mail = run(h.service.recent_messages(h.ana, limit=10))
    assert [m.item.id for m in mail.messages] == ["m1"]
    assert len(h.provider.requests) == calls, "answered from what the dashboard indexed"

    # The dashboard fallback: what was last indexed for one account, with its ledger entry.
    events, state = h.service.cached_events(h.ana, google, start=START, end=end, limit=5)
    assert [e.id for e in events] == ["g1"] and state.status == "ok"
    messages, state = h.service.cached_messages(h.ana, microsoft, limit=5)
    assert messages == [] and state.status == "ok"
    nothing, state = h.service.cached_events(h.ana, microsoft, start=START, end=end, limit=5)
    assert nothing == [] and state.status == "unavailable"


@pytest.mark.parametrize(
    "query, haystack, expected",
    [
        ("dentist", "Dentist appointment", True),
        ("DENTIST Appointment", "dentist appointment", True),
        ("appt", "dentist appointment", False),
        ("cafe", "Meet at Cafe Central", True),
        ("the meeting with bo", "Weekly meeting - Bo", True),
        ("invoice", "Quarterly numbers", False),
        ("", "anything", True),
        ("bo", "Kimbo Slice", True),
    ],
)
def test_text_matching_is_forgiving_but_never_invents(query, haystack, expected) -> None:
    assert matches(query, haystack) is expected
    assert matches(query, None, haystack) is expected


def test_text_matching_folds_accents() -> None:
    assert matches("cafe", "Meet at Café Central") is True
    assert matches("café", "Meet at Cafe Central") is True


# --- tools ------------------------------------------------------------------------------


def _bind(h) -> None:
    knowledge_tools.configure(lambda: h.service)


def test_tools_refuse_unidentified_sessions_and_report_missing_accounts_truthfully(h) -> None:
    _bind(h)
    for call in (
        knowledge_tools.recent_email(user_id=None),
        knowledge_tools.search_email(query="invoice", user_id=None),
        knowledge_tools.read_email_summary(user_id=None),
        knowledge_tools.upcoming_schedule(user_id=None),
        knowledge_tools.find_event(query="dentist", user_id=None),
    ):
        result = run(call)
        assert result["status"] == "unavailable"
        assert "signed in" in result["message"] and result["data"] == {}

    none = run(knowledge_tools.recent_email(user_id=h.ana))
    assert none["status"] == "no_accounts"
    assert "No email or calendar accounts are connected" in none["message"]
    assert "Settings" in none["message"]
    assert h.provider.requests == []

    knowledge_tools.reset()
    unbound = run(knowledge_tools.upcoming_schedule(user_id=h.ana))
    assert unbound["status"] == "unavailable" and "not available" in unbound["message"]


def test_email_tools_speak_short_summaries_without_ids_links_or_bodies(h) -> None:
    _bind(h)
    google = h.connect(h.ana)
    h.connect(h.ana, provider="microsoft", account_label="work@outlook.example")
    h.gmail(
        gmail_message(
            "m-quarter",
            "Quarterly numbers",
            "Bo Example <bo@contoso.example>",
            "2023-11-14T22:03:20Z",
        ),
        gmail_message(
            "m-lunch",
            "Lunch tomorrow?",
            "Ana Friend <ana@friend.example>",
            "2023-11-14T20:13:20Z",
            labelIds=["INBOX"],
        ),
    )
    outlook = dict(
        id="m-outlook",
        subject="Invoice 42 is due",
        receivedDateTime="2023-11-13T09:00:00Z",
        isRead=False,
        bodyPreview="Please pay invoice 42 by Friday",
        webLink="https://outlook.office365.com/owa/?ItemID=m-outlook",
    )
    outlook["from"] = dict(emailAddress=dict(name="Acme Billing", address="billing@acme.example"))
    h.provider.add("GET", GRAPH_MESSAGES, httpx.Response(200, json=dict(value=[outlook])))

    recent = run(knowledge_tools.recent_email(limit=5, user_id=h.ana))
    assert recent["status"] == "ok"
    message = recent["message"]
    assert message.startswith("You have 2 unread of 3 recent emails across 2 accounts.")
    assert "Bo Example about Quarterly numbers, 10 minutes ago" in message
    assert "Ana Friend about Lunch tomorrow?, 2 hours ago" in message
    assert "Acme Billing about Invoice 42 is due, yesterday at 9:00 AM" in message
    for hidden in ("m-quarter", "m-outlook", "https://", h.ana, google.connection_id, BODY):
        assert hidden not in message, hidden
    assert [m["id"] for m in recent["data"]["messages"]] == ["m-quarter", "m-lunch", "m-outlook"]
    assert recent["data"]["messages"][0]["account"] == "ana@google.example"
    assert set(recent["data"]["messages"][0]) == set(
        [
            "id",
            "account",
            "provider",
            "subject",
            "sender",
            "preview",
            "received_at",
            "when",
            "unread",
            "link",
        ]
    )
    assert recent["data"]["unread_count"] == 2 and recent["data"]["stale"] is False
    assert BODY not in str(recent)

    unread = run(knowledge_tools.recent_email(unread_only=True, user_id=h.ana))
    assert unread["message"].startswith("You have 2 unread emails across 2 accounts.")

    found = run(knowledge_tools.search_email(query="invoice", user_id=h.ana))
    assert found["status"] == "ok"
    assert found["message"].startswith("I found 1 email matching invoice.")
    assert "Acme Billing about Invoice 42 is due, yesterday at 9:00 AM" in found["message"]
    missing = run(knowledge_tools.search_email(query="tax return", user_id=h.ana))
    assert missing["status"] == "not_found"
    assert (
        missing["message"]
        == "I could not find any recent email matching tax return in your connected accounts."
    )

    only_work = run(knowledge_tools.recent_email(account="outlook", user_id=h.ana))
    assert [m["id"] for m in only_work["data"]["messages"]] == ["m-outlook"]
    unknown = run(knowledge_tools.recent_email(account="yahoo", user_id=h.ana))
    assert unknown["status"] == "not_found" and "yahoo" in unknown["message"]

    summary = run(knowledge_tools.read_email_summary(message_id="m-quarter", user_id=h.ana))
    assert summary["status"] == "ok"
    assert summary["message"] == (
        "Email from Bo Example, subject Quarterly numbers, received 10 minutes ago, unread. "
        "Preview: Preview of Quarterly numbers."
    )
    by_words = run(knowledge_tools.read_email_summary(query="lunch", user_id=h.ana))
    assert by_words["data"]["message"]["id"] == "m-lunch"
    newest = run(knowledge_tools.read_email_summary(user_id=h.ana))
    assert newest["data"]["message"]["id"] == "m-quarter"
    assert (
        run(knowledge_tools.read_email_summary(message_id="nope", user_id=h.ana))["status"]
        == "not_found"
    )
    # Model-supplied limits are clamped, never trusted.
    assert len(run(knowledge_tools.recent_email(limit=999, user_id=h.ana))["data"]["messages"]) == 3
    assert run(knowledge_tools.recent_email(limit="two", user_id=h.ana))["status"] == "ok"
    long_query = run(knowledge_tools.search_email(query="x" * 5000, user_id=h.ana))
    assert long_query["status"] == "not_found"


def test_calendar_tools_summarize_the_schedule_and_check_whether_an_event_exists(h) -> None:
    _bind(h)
    h.connect(h.ana)
    h.google_events(
        gevent("standup", "2023-11-14T23:00:00Z", "2023-11-14T23:30:00Z", "Standup"),
        gevent(
            "dentist", "2023-11-15T09:00:00Z", "2023-11-15T09:30:00Z", "Dentist", location="Room 1"
        ),
        dict(
            id="offsite",
            summary="Team offsite",
            start=dict(date="2023-11-17"),
            end=dict(date="2023-11-18"),
        ),
        gevent("far", "2023-11-24T09:00:00Z", "2023-11-24T10:00:00Z", "Far away"),
    )

    week = run(knowledge_tools.upcoming_schedule(days=7, user_id=h.ana))
    assert week["status"] == "ok"
    assert week["message"].startswith("You have 3 events in the next 7 days.")
    assert "Today at 11:00 PM, Standup" in week["message"]
    assert "Tomorrow at 9:00 AM, Dentist at Room 1" in week["message"]
    assert "Friday, all day, Team offsite" in week["message"]
    assert "Far away" not in week["message"] and "standup" not in week["message"]
    assert [e["id"] for e in week["data"]["events"]] == ["standup", "dentist", "offsite"]
    assert set(week["data"]["events"][1]) == set(
        [
            "id",
            "account",
            "provider",
            "title",
            "start",
            "end",
            "all_day",
            "location",
            "when",
            "status",
            "link",
        ]
    )
    assert week["data"]["window"]["days"] == 7

    today = run(knowledge_tools.upcoming_schedule(day="today", user_id=h.ana))
    assert today["message"].startswith("You have 1 event today.") and "Standup" in today["message"]
    tomorrow = run(knowledge_tools.upcoming_schedule(day="tomorrow", user_id=h.ana))
    assert tomorrow["message"].startswith("You have 1 event tomorrow.")
    friday = run(knowledge_tools.upcoming_schedule(day="friday", user_id=h.ana))
    assert [e["id"] for e in friday["data"]["events"]] == ["offsite"]
    dated = run(knowledge_tools.upcoming_schedule(day="2023-11-24", user_id=h.ana))
    assert [e["id"] for e in dated["data"]["events"]] == ["far"]
    empty = run(knowledge_tools.upcoming_schedule(day="2023-11-20", user_id=h.ana))
    assert empty["status"] == "ok"
    assert (
        empty["message"] == "You have nothing on your connected calendars on Monday, November 20."
    )

    yes = run(knowledge_tools.find_event(query="dentist", user_id=h.ana))
    assert yes["status"] == "ok" and yes["data"]["exists"] is True
    assert yes["message"] == "Yes. Dentist is tomorrow at 9:00 AM at Room 1."
    no = run(knowledge_tools.find_event(query="board meeting", days=14, user_id=h.ana))
    assert no["status"] == "not_found" and no["data"]["exists"] is False
    assert no["message"] == (
        "No. I do not see any event matching board meeting in the next 14 days "
        "on your connected calendars."
    )
    on_day = run(knowledge_tools.find_event(query="offsite", day="friday", user_id=h.ana))
    assert on_day["data"]["exists"] is True and "Friday, all day" in on_day["message"]
    assert run(knowledge_tools.find_event(query="", user_id=h.ana))["status"] == "not_found"
    assert run(knowledge_tools.upcoming_schedule(day="someday", user_id=h.ana))["status"] == "ok"
    assert len(h.provider.calls(GOOGLE_CALENDAR)) <= 2, "a wider read at most once, then the index"


def test_tools_say_when_an_answer_is_from_a_stale_index(h) -> None:
    _bind(h)
    h.connect(h.ana)
    h.google_events(gevent("g1", "2023-11-15T09:00:00Z", "2023-11-15T10:00:00Z", "Standup"))
    run(knowledge_tools.upcoming_schedule(days=7, user_id=h.ana))
    h.now += CALENDAR_TTL_SECONDS
    h.provider.add("GET", GOOGLE_CALENDAR, httpx.Response(503, json=dict(error="down")))

    stale = run(knowledge_tools.upcoming_schedule(days=7, user_id=h.ana))
    assert stale["status"] == "ok" and stale["data"]["stale"] is True
    assert "Standup" in stale["message"]
    assert stale["message"].endswith(
        "Note: your Google account ana@google.example could not be refreshed just now, "
        "so this is as of 10:13 PM."
    )
    h.provider.add("GET", GOOGLE_CALENDAR, httpx.Response(401, json=dict(error="bad")))
    h.provider.add(
        "POST",
        "https://oauth2.googleapis.com/token",
        httpx.Response(400, json=dict(error="invalid_grant")),
    )
    h.now += CALENDAR_TTL_SECONDS
    broken = run(knowledge_tools.upcoming_schedule(days=7, user_id=h.ana))
    assert "needs to be reconnected under Settings" in broken["message"]
    assert broken["data"]["accounts"][0]["status"] == "reconnect_required"


def test_registry_marks_the_knowledge_tools_user_scoped_and_the_llm_node_awaits_them(h) -> None:
    _bind(h)
    registry = create_default_registry()
    names = set(
        [
            "inbox.recent",
            "inbox.search",
            "inbox.read_summary",
            "schedule.upcoming",
            "schedule.find_event",
        ]
    )
    assert names <= set(registry.names())
    for name in names:
        tool = registry.get(name)
        assert tool.user_scoped is True and tool.requires_confirmation is False
        assert "user_id" not in tool.parameters["properties"]
        assert tool.parameters["additionalProperties"] is False
        assert "connected" in tool.description.lower()
    assert registry.get("inbox.search").parameters["required"] == ["query"]
    assert registry.get("schedule.find_event").parameters["required"] == ["query"]

    agent = SimpleNamespace(
        _native_tool_registry=registry,
        _user_scope=UserScope.for_user(SimpleNamespace(user_id=h.ana, display_name="Ana")),
    )
    result = run(llm_node_module._execute_single_tool(agent, "inbox.recent", dict(user_id=h.bo)))
    assert isinstance(result, dict) and result["status"] == "no_accounts"
    anonymous = SimpleNamespace(_native_tool_registry=registry, _user_scope=UserScope.anonymous())
    refused = run(llm_node_module._execute_single_tool(anonymous, "schedule.upcoming", {}))
    assert refused["status"] == "unauthorized"
    assert h.provider.requests == []


def test_a_named_account_is_matched_exactly_and_an_unknown_name_never_widens(h) -> None:
    """The label a user gave one of their own accounts narrows an answer, or nothing does."""
    _bind(h)
    google = h.connect(h.ana, account_label="ana@google.example")
    work = h.connect(h.ana, provider="microsoft", account_label="Vertex Ops")
    h.gmail(
        gmail_message("m-google", "Lunch?", "Ana Friend <ana@f.example>", "2023-11-14T21:00:00Z")
    )
    outlook = dict(
        id="m-vertex",
        subject="Invoice 42 is due",
        receivedDateTime="2023-11-14T09:00:00Z",
        isRead=False,
        bodyPreview="Please pay invoice 42",
    )
    outlook["from"] = dict(emailAddress=dict(name="Acme Billing", address="billing@acme.example"))
    h.provider.add("GET", GRAPH_MESSAGES, httpx.Response(200, json=dict(value=[outlook])))

    named = run(h.service.recent_messages(h.ana, limit=10, account="vertex ops"))
    assert [a.connection_id for a in named.accounts] == [work.connection_id]
    assert [m.item.id for m in named.messages] == ["m-vertex"]
    # A word of the label is enough; the provider name still names the provider.
    by_word = run(h.service.recent_messages(h.ana, limit=10, account="Vertex"))
    assert [a.connection_id for a in by_word.accounts] == [work.connection_id]
    by_provider = run(h.service.recent_messages(h.ana, limit=10, account="google"))
    assert [a.connection_id for a in by_provider.accounts] == [google.connection_id]

    # An unknown label answers about nothing at all: never about every account.
    unknown = run(h.service.recent_messages(h.ana, limit=10, account="Acme Holdings"))
    assert unknown.accounts == [] and unknown.messages == [] and unknown.connected is True
    tool = run(knowledge_tools.recent_email(account="Acme Holdings", user_id=h.ana))
    assert tool["status"] == "not_found" and tool["data"] == {}
    assert tool["message"] == "I could not find a connected account matching Acme Holdings."
    schedule = run(
        knowledge_tools.upcoming_schedule(day="today", account="Acme Holdings", user_id=h.ana)
    )
    assert schedule["status"] == "not_found" and schedule["data"] == {}
    only = run(knowledge_tools.recent_email(account="Vertex Ops", user_id=h.ana))
    assert [m["id"] for m in only["data"]["messages"]] == ["m-vertex"]
    assert [a["account"] for a in only["data"]["accounts"]] == ["Vertex Ops"]
    assert work.connection_id not in str(only)


# --- the names a user gives their own accounts ------------------------------------------


def _name(h, connection, label=None, aliases=()):
    return h.connections.set_labels(
        connection.user_id, connection.connection_id, user_label=label, aliases=list(aliases)
    )


def _fake(connection_id: str, provider: str, account_label: str, label=None, aliases=()):
    return ProviderConnection(
        connection_id=connection_id,
        user_id="usr_" + "1" * 24,
        provider=provider,
        status="connected",
        account_label=account_label,
        user_label=label,
        aliases=tuple(aliases),
    )


def test_account_resolution_follows_one_deterministic_precedence() -> None:
    """A name is read the same way every time, and never mixes two readings."""
    google = _fake("con_" + "a" * 24, "google", "ana@university.example", "university", ["campus"])
    work = _fake("con_" + "b" * 24, "microsoft", "ana@vertex.example", "work", ["university"])
    live = [google, work]

    # No name at all, or nothing matchable in one: every account.
    assert resolve_account(live, None).connections == live
    assert resolve_account(live, "  ").connections == live
    # A provider names a provider, even though "google" is nobody's user name.
    assert resolve_account(live, "Google").connections == [google]
    # A user name beats a provider label that also contains the word: "university"
    # is the name of the first account and only an alias of the second, so it is
    # ambiguous between the two user names rather than resolved by the address.
    assert resolve_account(live, "university").ambiguous is True
    assert resolve_account(live, "campus").connections == [google]
    assert resolve_account(live, "work").connections == [work]
    # The provider's own label resolves whole, then by part.
    assert resolve_account(live, "ana@vertex.example").connections == [work]
    assert resolve_account(live, "vertex").connections == [work]
    # An apostrophe is not a word boundary, so "wife" finds "wife's".
    wife = _fake("con_" + "c" * 24, "zoho", "her@zoho.example", "wife's")
    assert resolve_account([wife], "wife").connections == [wife]
    assert resolve_account([wife], "wifes").connections == [wife]
    # Nothing matched is nothing, never everything.
    nothing = resolve_account(live, "holiday home")
    assert nothing.connections == [] and nothing.ambiguous is False


def test_a_user_name_resolves_an_account_and_never_replaces_the_provider_label(h) -> None:
    _bind(h)
    google = h.connect(h.ana, account_label="ana@google.example")
    work = h.connect(
        h.ana,
        provider="microsoft",
        account_label="ana@vertex.example",
        provider_account_id="ms-work",
    )
    _name(h, google, "university", ["school", "campus"])
    _name(h, work, "work", ["office"])
    h.gmail(
        gmail_message("m-google", "Reading list", "Prof <prof@u.example>", "2023-11-14T21:00:00Z")
    )
    outlook = dict(
        id="m-vertex",
        subject="Invoice 42 is due",
        receivedDateTime="2023-11-14T09:00:00Z",
        isRead=False,
        bodyPreview="Please pay invoice 42",
    )
    outlook["from"] = dict(emailAddress=dict(name="Acme Billing", address="billing@acme.example"))
    h.provider.add("GET", GRAPH_MESSAGES, httpx.Response(200, json=dict(value=[outlook])))

    for said in ("university", "  UNIVERSITY ", "Campus", "school"):
        answer = run(h.service.recent_messages(h.ana, limit=10, account=said))
        assert [a.connection_id for a in answer.accounts] == [google.connection_id], said
        assert [m.item.id for m in answer.messages] == ["m-google"]
    by_alias = run(h.service.recent_messages(h.ana, limit=10, account="office"))
    assert [a.connection_id for a in by_alias.accounts] == [work.connection_id]

    # The provider's own label still resolves the account, and is still shown.
    by_address = run(h.service.recent_messages(h.ana, limit=10, account="ana@vertex.example"))
    assert [a.account_label for a in by_address.accounts] == ["ana@vertex.example"]
    assert [a.user_label for a in by_address.accounts] == ["work"]
    # A provider word still names the provider, not a user name.
    by_provider = run(h.service.recent_messages(h.ana, limit=10, account="google"))
    assert [a.connection_id for a in by_provider.accounts] == [google.connection_id]

    # An unknown name is still no match at all, never every account.
    unknown = run(h.service.recent_messages(h.ana, limit=10, account="holiday home"))
    assert unknown.accounts == [] and unknown.messages == [] and unknown.ambiguous is False


def test_one_name_on_two_accounts_asks_which_one_rather_than_guessing(h) -> None:
    _bind(h)
    first = h.connect(h.ana, account_label="ana@google.example")
    second = h.connect(
        h.ana,
        provider="microsoft",
        account_label="ana@vertex.example",
        provider_account_id="ms-two",
    )
    _name(h, first, "work")
    _name(h, second, "the job", ["work"])
    h.gmail(gmail_message("m1", "Hello", "Bo <bo@x.example>", "2023-11-14T21:00:00Z"))

    answer = run(h.service.recent_messages(h.ana, limit=10, account="work"))
    assert answer.ambiguous is True
    assert answer.accounts == [] and answer.messages == []
    assert h.provider.requests == [], "an ambiguous name reads no provider at all"

    spoken = run(knowledge_tools.recent_email(account="work", user_id=h.ana))
    assert spoken["status"] == "ambiguous" and spoken["data"] == dict()
    assert "work" in spoken["message"] and "which one" in spoken["message"]
    for hidden in (first.connection_id, second.connection_id, h.ana, ACCESS):
        assert hidden not in str(spoken), hidden

    schedule = run(knowledge_tools.upcoming_schedule(day="today", account="work", user_id=h.ana))
    assert schedule["status"] == "ambiguous"

    # Naming one of them uniquely answers again.
    resolved = run(h.service.recent_messages(h.ana, limit=10, account="the job"))
    assert [a.connection_id for a in resolved.accounts] == [second.connection_id]


def test_a_user_name_never_reaches_another_users_account(h) -> None:
    _bind(h)
    theirs = h.connect(h.bo, account_label="bo@google.example", provider_account_id="google-bo")
    _name(h, theirs, "work", ["office"])
    mine = h.connect(h.ana, account_label="ana@google.example")
    _name(h, mine, "personal")

    crossed = run(h.service.recent_messages(h.ana, limit=10, account="office"))
    assert crossed.accounts == [] and crossed.ambiguous is False
    assert h.provider.requests == []
    refused = run(knowledge_tools.recent_email(account="office", user_id=h.ana))
    assert refused["status"] == "not_found"
    assert h.bo not in str(refused) and theirs.connection_id not in str(refused)
    # And Bo's own question about "work" is answered from Bo's account only.
    h.gmail(gmail_message("m-bo", "Hi", "Bo <bo@x.example>", "2023-11-14T21:00:00Z"))
    ok = run(h.service.recent_messages(h.bo, limit=10, account="work"))
    assert [a.connection_id for a in ok.accounts] == [theirs.connection_id]


def test_a_user_name_is_never_written_to_the_logs(h, caplog) -> None:
    _bind(h)
    connection = h.connect(h.ana, account_label="ana@google.example")
    _name(h, connection, "Zebulon", ["Quixote"])
    h.gmail(gmail_message("m1", "Hello", "Bo <bo@x.example>", "2023-11-14T21:00:00Z"))

    with caplog.at_level(logging.DEBUG):
        answered = run(knowledge_tools.recent_email(account="Quixote", user_id=h.ana))
    assert answered["status"] == "ok"
    for hidden in ("Zebulon", "zebulon", "Quixote", "quixote", h.ana):
        assert hidden not in caplog.text, hidden


def test_a_stale_account_is_spoken_of_by_the_name_its_owner_gave_it(h) -> None:
    _bind(h)
    connection = h.connect(h.ana, account_label="ana@google.example")
    _name(h, connection, "university")
    h.provider.add("GET", GMAIL_MESSAGES, httpx.Response(503, json=dict(error="down")))

    spoken = run(knowledge_tools.recent_email(user_id=h.ana))
    assert "university" in spoken["message"]
    assert "ana@google.example" not in spoken["message"]
    assert spoken["data"]["accounts"][0]["user_label"] == "university"


def test_a_named_account_never_reaches_across_users(h) -> None:
    """Bo's account label names nothing of Ana's, whoever says it."""
    _bind(h)
    h.connect(h.bo, account_label="Bo Private", provider_account_id="google-bo")
    h.connect(h.ana, account_label="ana@google.example")

    crossed = run(h.service.recent_messages(h.ana, limit=10, account="Bo Private"))
    assert crossed.accounts == [] and crossed.messages == []
    assert h.provider.requests == [], "an unmatched label reads no provider at all"
    refused = run(knowledge_tools.recent_email(account="Bo Private", user_id=h.ana))
    assert refused["status"] == "not_found"
    assert h.bo not in str(refused) and "Bo Private" in refused["message"]


def test_a_named_account_is_never_written_to_the_logs(h, caplog) -> None:
    _bind(h)
    h.connect(h.ana, account_label="Zebulon Quixote")
    h.gmail(gmail_message("m1", "Hello", "Bo <bo@x.example>", "2023-11-14T21:00:00Z"))

    with caplog.at_level(logging.DEBUG):
        run(knowledge_tools.recent_email(account="Zebulon Quixote", user_id=h.ana))

    for hidden in ("Zebulon", "zebulon", "Quixote", "quixote", h.ana):
        assert hidden not in caplog.text, hidden


def test_the_tool_loop_logs_knowledge_calls_without_queries_or_contents(h, caplog) -> None:
    _bind(h)
    h.connect(h.ana)
    h.gmail(
        gmail_message(
            "m1", "Quarterly numbers", "Bo Example <bo@x.example>", "2023-11-14T22:00:00Z"
        )
    )
    registry = create_default_registry()
    agent = SimpleNamespace(
        _native_tool_registry=registry,
        _user_scope=UserScope.for_user(SimpleNamespace(user_id=h.ana, display_name="Ana")),
    )

    class Provider:
        def format_tool_call_message(self, *, content, tool_calls):
            return dict(role="assistant", content=content)

        def format_tool_result(self, *, content, tool_call_id, tool_name):
            return dict(role="tool", content=content, name=tool_name)

    call = SimpleNamespace(
        id="call-1", name="inbox.search", arguments=dict(query="quarterly numbers")
    )
    with caplog.at_level(logging.DEBUG):
        messages, _ = run(
            llm_node_module._execute_tool_calls(agent, [], [call], None, Provider(), None)
        )

    assert "Quarterly numbers" in messages[-1]["content"], "the model still gets the answer"
    assert "inbox.search" in caplog.text
    for hidden in ("quarterly", "Quarterly", "Bo Example", h.ana):
        assert hidden not in caplog.text, hidden
