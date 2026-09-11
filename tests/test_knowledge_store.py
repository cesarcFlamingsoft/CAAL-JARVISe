"""The per-user knowledge index behind the email and calendar answers JARVIS gives.

Pinned properties of :mod:caal.knowledge_store:

* a row is only ever written and read under the user *and* connection it
  came from; the text of a row is encrypted with the profile key ring and
  bound to that exact (user, connection, item), so nothing copied between
  rows or users decrypts;
* only the bounded dashboard-safe fields are kept: never a body, a
  recipient list, HTML, or the raw payload of a provider;
* the index is bounded per connection and window queries are answered from
  indexed numeric columns;
* revoking a connection or forgetting a user erases every row they own.
"""

from __future__ import annotations

import sqlite3

import pytest

from caal import profile_crypto
from caal.knowledge_store import (
    MAX_INDEXED_EVENTS,
    MAX_INDEXED_MESSAGES,
    KnowledgeStore,
)
from caal.profile_crypto import KeyRing
from caal.provider_data import CalendarEvent, InboxMessage
from caal.user_store import UserStore

NOW = 1_700_000_000  # 2023-11-14T22:13:20Z
ANA = "usr_" + "a" * 24
BO = "usr_" + "b" * 24
CON_A = "con_" + "a" * 24
CON_B = "con_" + "b" * 24
CON_C = "con_" + "c" * 24
FAR = NOW + 90 * 86400


@pytest.fixture
def db(tmp_path):
    return tmp_path / "assistant.sqlite3"


@pytest.fixture
def store(db):
    keyring = KeyRing.from_env(profile_crypto.generate_key_material(version=1))
    users = UserStore(db, keyring=keyring)
    return KnowledgeStore(users, keyring=keyring)


def message(message_id: str, received_at: str, **overrides) -> InboxMessage:
    params = dict(
        id=message_id,
        received_at=received_at,
        unread=False,
        subject=f"Subject {message_id}",
        sender="Bo Example",
        preview=f"Preview {message_id}",
        link=None,
    )
    params.update(overrides)
    return InboxMessage(**params)


def event(event_id: str, start: str, end: str | None = None, **overrides) -> CalendarEvent:
    params = dict(
        id=event_id,
        start=start,
        end=end,
        all_day=False,
        title=f"Title {event_id}",
        location="Room 1",
        link=None,
        status="confirmed",
    )
    params.update(overrides)
    return CalendarEvent(**params)


def ids(rows) -> list[str]:
    return [row.item.id for row in rows]


def test_schema_version_six_creates_the_knowledge_tables(store, db) -> None:
    with sqlite3.connect(db) as connection:
        tables = set(
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = ?", ("table",)
            )
        )
        version = connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
    assert set(["knowledge_messages", "knowledge_events", "knowledge_sync"]) <= tables
    assert version >= 6


def test_messages_are_stored_encrypted_bounded_and_read_back_newest_first(store, db) -> None:
    older = message("m-old", "2023-11-14T20:00:00Z")
    newer = message("m-new", "2023-11-14T22:00:00Z", unread=True, subject="Quarterly numbers")
    store.upsert_messages(ANA, CON_A, "google", [older, newer], now=NOW)

    rows = store.messages(ANA, [CON_A], limit=10)
    assert [(row.item.id, row.connection_id) for row in rows] == [
        ("m-new", CON_A),
        ("m-old", CON_A),
    ]
    assert rows[0].item.subject == "Quarterly numbers" and rows[0].item.unread is True
    assert rows[0].provider == "google" and rows[0].indexed_at == NOW

    raw = db.read_bytes()
    for plaintext in (b"Quarterly numbers", b"Preview m-new", b"Bo Example", b"Subject m-old"):
        assert plaintext not in raw, plaintext


def test_message_upsert_updates_read_state_and_keeps_only_the_newest_per_connection(
    store,
) -> None:
    first = [message(f"m{i:03d}", f"2023-11-01T{i:02d}:00:00Z") for i in range(10)]
    store.upsert_messages(ANA, CON_A, "google", first, now=NOW)
    store.upsert_messages(
        ANA, CON_A, "google", [message("m000", "2023-11-01T00:00:00Z", unread=True)], now=NOW + 1
    )
    assert store.messages(ANA, [CON_A], limit=50)[-1].item.unread is True

    flood = [
        message(f"f{i:03d}", f"2023-12-{1 + i // 24:02d}T{i % 24:02d}:00:00Z")
        for i in range(MAX_INDEXED_MESSAGES + 5)
    ]
    store.upsert_messages(ANA, CON_A, "google", flood, now=NOW + 2)
    kept = store.messages(ANA, [CON_A], limit=500)
    assert len(kept) == MAX_INDEXED_MESSAGES
    assert all(row.item.id.startswith("f") for row in kept)
    assert kept[0].item.received_at > kept[-1].item.received_at


def test_events_replace_the_window_of_the_connection_and_answer_range_queries(store) -> None:
    window = (NOW, NOW + 7 * 86400)
    inside = event("e-in", "2023-11-15T09:00:00Z", "2023-11-15T10:00:00Z")
    later = event("e-late", "2023-11-17T09:00:00Z", "2023-11-17T09:30:00Z", title="Dentist")
    all_day = event("e-day", "2023-11-16", "2023-11-17", all_day=True, title="Offsite")
    store.replace_events(ANA, CON_A, "google", [inside, later, all_day], window=window, now=NOW)

    rows = store.events(ANA, [CON_A], start_ts=NOW, end_ts=NOW + 3 * 86400, limit=10)
    assert ids(rows) == ["e-in", "e-day", "e-late"]
    assert rows[1].item.all_day is True and rows[1].item.start == "2023-11-16"
    assert rows[0].item.title == "Title e-in" and rows[0].item.location == "Room 1"

    # A later read of the same window that no longer lists an event drops it.
    store.replace_events(ANA, CON_A, "google", [inside], window=window, now=NOW + 60)
    assert ids(store.events(ANA, [CON_A], start_ts=NOW, end_ts=FAR, limit=10)) == ["e-in"]

    # The cap holds even when a provider answers more than expected.
    many = [
        event(f"x{i:03d}", f"2023-11-{15 + i // 24:02d}T{i % 24:02d}:00:00Z")
        for i in range(MAX_INDEXED_EVENTS + 3)
    ]
    store.replace_events(ANA, CON_A, "google", many, window=window, now=NOW + 120)
    assert len(store.events(ANA, [CON_A], start_ts=0, end_ts=FAR, limit=500)) == (
        MAX_INDEXED_EVENTS
    )


def test_rows_never_cross_users_or_connections(store) -> None:
    day = (NOW, NOW + 86400)
    store.upsert_messages(ANA, CON_A, "google", [message("a1", "2023-11-14T22:00:00Z")], now=NOW)
    store.upsert_messages(BO, CON_B, "microsoft", [message("b1", "2023-11-14T22:00:00Z")], now=NOW)
    store.replace_events(
        ANA, CON_A, "google", [event("ea", "2023-11-15T09:00:00Z")], window=day, now=NOW
    )
    store.replace_events(
        BO, CON_B, "microsoft", [event("eb", "2023-11-15T09:00:00Z")], window=day, now=NOW
    )

    # Asking under the wrong user for a connection that exists yields nothing.
    assert store.messages(ANA, [CON_B], limit=10) == []
    assert ids(store.messages(BO, [CON_A, CON_B], limit=10)) == ["b1"]
    assert store.events(ANA, [CON_B], start_ts=0, end_ts=FAR, limit=10) == []
    assert ids(store.events(BO, [CON_A, CON_B], start_ts=0, end_ts=FAR, limit=10)) == ["eb"]
    assert store.messages(ANA, [], limit=10) == []
    # Identifiers shaped wrong are refused before any query.
    with pytest.raises(ValueError):
        store.upsert_messages("ana", CON_A, "google", [], now=NOW)
    with pytest.raises(ValueError):
        store.messages(ANA, ["con_x"], limit=10)
    with pytest.raises(ValueError):
        store.upsert_messages(ANA, CON_A, "imap", [], now=NOW)


def test_a_row_moved_to_another_user_fails_to_decrypt_and_is_skipped(store, db) -> None:
    store.upsert_messages(ANA, CON_A, "google", [message("a1", "2023-11-14T22:00:00Z")], now=NOW)
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE knowledge_messages SET user_id = ? WHERE user_id = ?", (BO, ANA))
    assert store.messages(BO, [CON_A], limit=10) == []


def test_sync_state_is_recorded_per_connection_and_kind(store) -> None:
    store.record_sync(ANA, CON_A, "inbox", status="ok", item_count=3, now=NOW, ttl_seconds=300)
    store.record_sync(
        ANA, CON_A, "calendar", status="unavailable", reason="transport", now=NOW, ttl_seconds=300
    )
    states = store.sync_states(ANA, [CON_A, CON_C], "inbox")
    assert set(states) == set([CON_A])
    inbox = states[CON_A]
    assert inbox.status == "ok" and inbox.synced_at == NOW and inbox.expires_at == NOW + 300
    assert inbox.item_count == 3 and inbox.reason is None
    assert inbox.is_fresh(NOW + 299) and not inbox.is_fresh(NOW + 300)
    calendar = store.sync_states(ANA, [CON_A], "calendar")[CON_A]
    assert calendar.status == "unavailable" and calendar.reason == "transport"
    assert store.sync_states(BO, [CON_A], "inbox") == {}
    with pytest.raises(ValueError):
        store.record_sync(ANA, CON_A, "sms", status="ok", now=NOW, ttl_seconds=10)
    # When a connection was last read successfully survives later failures.
    assert inbox.last_ok_at == NOW and calendar.last_ok_at is None
    failed = store.record_sync(
        ANA, CON_A, "inbox", status="unavailable", reason="transport", now=NOW + 5, ttl_seconds=60
    )
    assert failed.last_ok_at == NOW and failed.status == "unavailable"
    assert store.sync_states(ANA, [CON_A], "inbox")[CON_A].last_ok_at == NOW


def test_forgetting_a_connection_or_a_user_erases_every_row(store) -> None:
    day = (NOW, NOW + 86400)
    for user, con in ((ANA, CON_A), (ANA, CON_C), (BO, CON_B)):
        store.upsert_messages(user, con, "google", [message("m", "2023-11-14T22:00:00Z")], now=NOW)
        store.replace_events(
            user, con, "google", [event("e", "2023-11-15T09:00:00Z")], window=day, now=NOW
        )
        store.record_sync(user, con, "inbox", status="ok", item_count=1, now=NOW, ttl_seconds=60)

    store.forget_connection(ANA, CON_A)
    assert store.messages(ANA, [CON_A], limit=10) == []
    assert store.events(ANA, [CON_A], start_ts=0, end_ts=FAR, limit=10) == []
    assert store.sync_states(ANA, [CON_A], "inbox") == {}
    assert len(store.messages(ANA, [CON_C], limit=10)) == 1
    # A connection id that belongs to somebody else changes nothing under the wrong user.
    store.forget_connection(ANA, CON_B)
    assert len(store.messages(BO, [CON_B], limit=10)) == 1

    store.forget_user(BO)
    assert store.messages(BO, [CON_B], limit=10) == []
    assert store.events(BO, [CON_B], start_ts=0, end_ts=FAR, limit=10) == []
    assert store.sync_states(BO, [CON_B], "inbox") == {}
    assert len(store.messages(ANA, [CON_C], limit=10)) == 1


def test_prune_drops_old_messages_and_finished_events(store) -> None:
    store.upsert_messages(
        ANA,
        CON_A,
        "google",
        [message("old", "2023-10-01T00:00:00Z"), message("recent", "2023-11-14T20:00:00Z")],
        now=NOW,
    )
    store.replace_events(
        ANA,
        CON_A,
        "google",
        [
            event("done", "2023-11-10T09:00:00Z", "2023-11-10T10:00:00Z"),
            event("ahead", "2023-11-15T09:00:00Z", "2023-11-15T10:00:00Z"),
        ],
        window=(NOW - 30 * 86400, NOW + 7 * 86400),
        now=NOW,
    )
    store.prune(now=NOW)
    assert ids(store.messages(ANA, [CON_A], limit=10)) == ["recent"]
    assert ids(store.events(ANA, [CON_A], start_ts=0, end_ts=FAR, limit=10)) == ["ahead"]
