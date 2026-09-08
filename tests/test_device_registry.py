"""Coverage for the durable device session and handoff registry (Phase 1)."""

from __future__ import annotations

import json
import sqlite3
import threading

import pytest

from caal import device_registry
from caal.device_registry import Transport


@pytest.fixture(autouse=True)
def store(monkeypatch, tmp_path):
    """Point every registry write at an isolated SQLite file."""
    path = tmp_path / "assistant.sqlite3"
    monkeypatch.setattr(device_registry, "STORE_PATH", path)
    return path


def _register(device_id="phone-1", label="Cesar's iPhone", transport="mobile", now=1000):
    return device_registry.register_device_session(
        device_id=device_id,
        room_name="jarvis-room",
        label=label,
        transport=transport,
        now=now,
    )


def test_normalize_label_trims_and_collapses_whitespace() -> None:
    assert device_registry.normalize_label("  Living   Room  Echo ") == "Living Room Echo"


@pytest.mark.parametrize(
    "label",
    ["", "   ", "\t\n", "x" * (device_registry.MAX_LABEL_LENGTH + 1), "bad\x00label", "bell\x07"],
)
def test_normalize_label_rejects_unsafe_labels(label: str) -> None:
    with pytest.raises(ValueError):
        device_registry.normalize_label(label)


def test_transport_enum_covers_supported_transports() -> None:
    assert {member.value for member in Transport} == {"web", "mobile", "phone"}
    assert device_registry.coerce_transport("PHONE") is Transport.PHONE
    assert device_registry.coerce_transport(Transport.WEB) is Transport.WEB
    with pytest.raises(ValueError):
        device_registry.coerce_transport("carrier-pigeon")


def test_register_device_session_persists_a_safe_model() -> None:
    session = _register()

    assert session.device_id == "phone-1"
    assert session.room_name == "jarvis-room"
    assert session.label == "Cesar's iPhone"
    assert session.transport is Transport.MOBILE
    assert session.last_seen == 1000
    assert len(session.session_id) >= 32


def test_session_ids_are_opaque_unique_and_kept_out_of_reprs() -> None:
    first = _register(device_id="phone-1")
    second = _register(device_id="laptop-1", transport="web")

    assert first.session_id != second.session_id
    assert first.session_id not in repr(first)
    assert "phone-1" in repr(first)


def test_register_upserts_one_row_per_device_and_retires_the_old_session() -> None:
    first = _register(device_id="phone-1", label="Old Label", now=1000)
    second = device_registry.register_device_session(
        device_id="phone-1",
        room_name="jarvis-room-2",
        label="New Label",
        transport="mobile",
        now=1050,
    )

    assert first.session_id != second.session_id
    assert device_registry.get_session(first.session_id, now=1050) is None

    active = device_registry.list_active_sessions(now=1050)
    assert len(active) == 1
    assert active[0].label == "New Label"
    assert active[0].room_name == "jarvis-room-2"


def test_register_validates_label_room_and_device_id() -> None:
    with pytest.raises(ValueError):
        _register(label="")
    with pytest.raises(ValueError):
        _register(device_id="  ")
    with pytest.raises(ValueError):
        device_registry.register_device_session(
            device_id="phone-1", room_name="", label="Phone", transport="mobile", now=1000
        )


def test_heartbeat_refreshes_last_seen_and_keeps_the_session_active() -> None:
    session = _register(now=1000)

    beat = device_registry.heartbeat(session.session_id, now=1200)

    assert beat is not None
    assert beat.session_id == session.session_id
    assert beat.last_seen == 1200
    assert device_registry.list_active_sessions(now=1400)[0].last_seen == 1200


def test_heartbeat_rejects_unknown_and_expired_sessions() -> None:
    session = _register(now=1000)
    expired_at = 1000 + device_registry.SESSION_TTL_SECONDS + 1

    assert device_registry.heartbeat("not-a-session", now=1000) is None
    assert device_registry.heartbeat(session.session_id, now=expired_at) is None


def test_listing_excludes_expired_sessions_and_orders_by_last_seen() -> None:
    stale = _register(device_id="phone-1", label="Stale Phone", now=1000)
    fresh = _register(device_id="laptop-1", label="Fresh Laptop", transport="web", now=1200)
    now = 1000 + device_registry.SESSION_TTL_SECONDS + 1

    active = device_registry.list_active_sessions(now=now)

    assert [item.session_id for item in active] == [fresh.session_id]
    assert stale.session_id not in {item.session_id for item in active}


def test_listing_is_bounded_by_a_hard_cap() -> None:
    for index in range(device_registry.MAX_ACTIVE_SESSIONS + 5):
        _register(device_id=f"device-{index}", label=f"Device {index}", now=1000 + index)

    assert len(device_registry.list_active_sessions(now=1100)) == (
        device_registry.MAX_ACTIVE_SESSIONS
    )
    assert len(device_registry.list_active_sessions(now=1100, limit=3)) == 3
    assert len(device_registry.list_active_sessions(now=1100, limit=10_000)) == (
        device_registry.MAX_ACTIVE_SESSIONS
    )
    with pytest.raises(ValueError):
        device_registry.list_active_sessions(now=1100, limit=0)


def test_purge_expired_sessions_deletes_only_stale_rows() -> None:
    _register(device_id="phone-1", now=1000)
    fresh = _register(device_id="laptop-1", transport="web", now=1200)
    now = 1000 + device_registry.SESSION_TTL_SECONDS + 1

    removed = device_registry.purge_expired_sessions(now=now)

    assert removed == 1
    assert [item.session_id for item in device_registry.list_active_sessions(now=now)] == [
        fresh.session_id
    ]


def test_create_handoff_snapshots_context_without_leaking_the_session_id() -> None:
    session = _register(now=1000)

    handoff = device_registry.create_handoff(
        session.session_id,
        context={"summary": "Discussing the deploy", "turns": 4},
        now=1000,
    )

    assert handoff.device_id == "phone-1"
    assert handoff.room_name == "jarvis-room"
    assert handoff.context == {"summary": "Discussing the deploy", "turns": 4}
    assert handoff.claimed_at is None
    assert len(handoff.handoff_id) >= 32
    assert handoff.handoff_id not in repr(handoff)

    with sqlite3.connect(device_registry.STORE_PATH) as connection:
        stored = connection.execute("SELECT * FROM device_handoffs").fetchall()
    assert session.session_id not in json.dumps(stored, default=str)


def test_create_handoff_requires_a_live_session() -> None:
    session = _register(now=1000)
    expired_at = 1000 + device_registry.SESSION_TTL_SECONDS + 1

    with pytest.raises(LookupError):
        device_registry.create_handoff("not-a-session", context={}, now=1000)
    with pytest.raises(LookupError):
        device_registry.create_handoff(session.session_id, context={}, now=expired_at)


def test_handoff_context_snapshot_is_bounded() -> None:
    session = _register(now=1000)

    truncated = device_registry.create_handoff(
        session.session_id,
        context={"summary": "y" * (device_registry.MAX_CONTEXT_VALUE_CHARS + 200)},
        now=1000,
    )
    assert len(truncated.context["summary"]) == device_registry.MAX_CONTEXT_VALUE_CHARS

    with pytest.raises(ValueError):
        device_registry.create_handoff(
            session.session_id,
            context={f"key-{index}": index for index in range(20)},
            now=1000,
        )
    with pytest.raises(ValueError):
        device_registry.create_handoff(
            session.session_id, context={"nested": {"not": "allowed"}}, now=1000
        )
    with pytest.raises(ValueError):
        device_registry.create_handoff(session.session_id, context=["not", "a", "map"], now=1000)


def test_handoff_is_claimable_exactly_once() -> None:
    session = _register(now=1000)
    handoff = device_registry.create_handoff(
        session.session_id, context={"summary": "hand me over"}, now=1000
    )

    claimed = device_registry.claim_handoff(handoff.handoff_id, device_id="laptop-1", now=1010)
    replayed = device_registry.claim_handoff(handoff.handoff_id, device_id="laptop-1", now=1011)

    assert claimed is not None
    assert claimed.context == {"summary": "hand me over"}
    assert claimed.claimed_at == 1010
    assert claimed.claimed_by == "laptop-1"
    assert replayed is None


def test_claim_rejects_unknown_and_expired_handoffs() -> None:
    session = _register(now=1000)
    handoff = device_registry.create_handoff(session.session_id, context={}, now=1000)
    expired_at = 1000 + device_registry.HANDOFF_TTL_SECONDS + 1

    assert device_registry.claim_handoff("not-a-handoff", device_id="laptop-1", now=1000) is None
    assert (
        device_registry.claim_handoff(handoff.handoff_id, device_id="laptop-1", now=expired_at)
        is None
    )


def test_concurrent_claims_award_the_handoff_to_a_single_device() -> None:
    session = _register(now=1000)
    handoff = device_registry.create_handoff(session.session_id, context={}, now=1000)
    winners: list[str] = []
    lock = threading.Lock()
    start = threading.Barrier(8)

    def claim(index: int) -> None:
        start.wait()
        record = device_registry.claim_handoff(
            handoff.handoff_id, device_id=f"device-{index}", now=1010
        )
        if record is not None:
            with lock:
                winners.append(record.claimed_by or "")

    threads = [threading.Thread(target=claim, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(winners) == 1
