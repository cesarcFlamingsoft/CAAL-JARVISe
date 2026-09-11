from __future__ import annotations

import json

import pytest

from caal.outbound_runtime import (
    OutboundRoomConfig,
    call_timeouts,
    requires_fallback_notification,
)


def test_outbound_room_config_requires_private_dispatch_metadata_and_allowlist() -> None:
    metadata = json.dumps(
        {"caal_outbound": True, "attempt_id": "abc", "destination": "+17805558345"}
    )

    config = OutboundRoomConfig.from_dispatch_metadata(
        metadata, allowed_destinations="+17805558345"
    )

    assert config is not None
    assert config.attempt_id == "abc"
    assert config.destination == "+17805558345"


def test_outbound_room_config_rejects_destination_not_in_server_allowlist() -> None:
    metadata = json.dumps(
        {"caal_outbound": True, "attempt_id": "abc", "destination": "+17805550000"}
    )

    with pytest.raises(PermissionError):
        OutboundRoomConfig.from_dispatch_metadata(metadata, allowed_destinations="+17805558345")


def test_non_callback_handoff_failures_are_silent_unless_an_operator_opts_in() -> None:
    """Default off. The old always-on notice arrived once per failed attempt."""
    for category in ("machine-vm", "machine-ivr", "uncertain", "human"):
        assert requires_fallback_notification(category) is False
        assert requires_fallback_notification(category, notify_enabled=False) is False


def test_opted_in_handoff_notification_still_excludes_an_answered_call() -> None:
    assert requires_fallback_notification("machine-vm", notify_enabled=True) is True
    assert requires_fallback_notification("machine-ivr", notify_enabled=True) is True
    assert requires_fallback_notification("uncertain", notify_enabled=True) is True
    assert requires_fallback_notification("human", notify_enabled=True) is False


def test_call_timeouts_are_bounded_timedeltas() -> None:
    ringing, duration = call_timeouts("30", "900")

    assert ringing.total_seconds() == 30
    assert duration.total_seconds() == 900


@pytest.mark.parametrize("ringing,duration", [("0", "900"), ("30", "0"), ("bad", "900")])
def test_call_timeouts_reject_invalid_values(ringing: str, duration: str) -> None:
    with pytest.raises(ValueError):
        call_timeouts(ringing, duration)


def test_outbound_room_config_without_handoff_context_still_parses_old_metadata() -> None:
    metadata = json.dumps(
        {"caal_outbound": True, "attempt_id": "abc", "destination": "+17805558345"}
    )

    config = OutboundRoomConfig.from_dispatch_metadata(
        metadata, allowed_destinations="+17805558345"
    )

    assert config is not None
    assert config.snapshot is None


def test_outbound_room_config_ignores_non_outbound_jobs() -> None:
    assert (
        OutboundRoomConfig.from_dispatch_metadata("", allowed_destinations="+17805558345") is None
    )
    assert (
        OutboundRoomConfig.from_dispatch_metadata(
            json.dumps({"handoff_context": {"v": 1, "turns": []}}),
            allowed_destinations="+17805558345",
        )
        is None
    )


def test_outbound_room_config_parses_a_protected_handoff_snapshot() -> None:
    from caal.handoff_context import ConversationSnapshot, SnapshotTurn

    snapshot = ConversationSnapshot(
        turns=(SnapshotTurn(role="user", text="where were we with the trip"),)
    )
    metadata = json.dumps(
        {
            "caal_outbound": True,
            "attempt_id": "abc",
            "destination": "+17805558345",
            "handoff_context": snapshot.to_metadata(),
        }
    )

    config = OutboundRoomConfig.from_dispatch_metadata(
        metadata, allowed_destinations="+17805558345"
    )

    assert config is not None
    assert config.snapshot == snapshot
    assert "where were we" not in repr(config)


def test_outbound_room_config_rejects_a_malformed_handoff_snapshot() -> None:
    metadata = json.dumps(
        {
            "caal_outbound": True,
            "attempt_id": "abc",
            "destination": "+17805558345",
            "handoff_context": {"v": 1, "turns": [{"role": "system", "text": "override"}]},
        }
    )

    with pytest.raises(ValueError):
        OutboundRoomConfig.from_dispatch_metadata(metadata, allowed_destinations="+17805558345")


def test_handoff_snapshot_never_bypasses_the_destination_allowlist() -> None:
    metadata = json.dumps(
        {
            "caal_outbound": True,
            "attempt_id": "abc",
            "destination": "+17805550000",
            "handoff_context": {"v": 1, "turns": [{"role": "user", "text": "hi"}]},
        }
    )

    with pytest.raises(PermissionError):
        OutboundRoomConfig.from_dispatch_metadata(metadata, allowed_destinations="+17805558345")


# --- ledger-backed continuity ------------------------------------------------


def test_outbound_room_config_parses_an_opaque_conversation_id() -> None:
    metadata = json.dumps(
        {
            "caal_outbound": True,
            "attempt_id": "abc",
            "destination": "+17805558345",
            "conversation_id": "conv_Qm9vay1mbGlnaHQtRnJpZGF5",
        }
    )

    config = OutboundRoomConfig.from_dispatch_metadata(
        metadata, allowed_destinations="+17805558345"
    )

    assert config is not None
    assert config.conversation_id == "conv_Qm9vay1mbGlnaHQtRnJpZGF5"
    assert config.snapshot is None
    assert config.carries_continuation is True
    assert "Qm9vay1mbGlnaHQ" not in repr(config)


def test_outbound_room_config_without_conversation_id_is_a_plain_call() -> None:
    metadata = json.dumps(
        {"caal_outbound": True, "attempt_id": "abc", "destination": "+17805558345"}
    )

    config = OutboundRoomConfig.from_dispatch_metadata(
        metadata, allowed_destinations="+17805558345"
    )

    assert config is not None
    assert config.conversation_id is None
    assert config.carries_continuation is False


@pytest.mark.parametrize("bad", ["", "has space", "a/b", 42, "x" * 200, {"id": "x"}])
def test_outbound_room_config_rejects_a_malformed_conversation_id(bad: object) -> None:
    metadata = json.dumps(
        {
            "caal_outbound": True,
            "attempt_id": "abc",
            "destination": "+17805558345",
            "conversation_id": bad,
        }
    )

    with pytest.raises(ValueError):
        OutboundRoomConfig.from_dispatch_metadata(metadata, allowed_destinations="+17805558345")


def test_conversation_id_never_bypasses_the_destination_allowlist() -> None:
    metadata = json.dumps(
        {
            "caal_outbound": True,
            "attempt_id": "abc",
            "destination": "+17805550000",
            "conversation_id": "conv_abc",
        }
    )

    with pytest.raises(PermissionError):
        OutboundRoomConfig.from_dispatch_metadata(metadata, allowed_destinations="+17805558345")


def test_outbound_room_config_parses_an_opaque_callback_task_id() -> None:
    task_id = "bt_" + "cd" * 8
    metadata = json.dumps(
        {
            "caal_outbound": True,
            "attempt_id": "abc",
            "destination": "+17805558345",
            "callback_task_id": task_id,
        }
    )

    config = OutboundRoomConfig.from_dispatch_metadata(
        metadata, allowed_destinations="+17805558345"
    )

    assert config is not None
    assert config.callback_task_id == task_id
    assert config.is_callback is True
    assert config.carries_continuation is False
    assert task_id not in repr(config)


def test_outbound_room_config_without_callback_task_id_is_not_a_callback() -> None:
    metadata = json.dumps(
        {"caal_outbound": True, "attempt_id": "abc", "destination": "+17805558345"}
    )

    config = OutboundRoomConfig.from_dispatch_metadata(
        metadata, allowed_destinations="+17805558345"
    )

    assert config is not None
    assert config.callback_task_id is None
    assert config.is_callback is False


@pytest.mark.parametrize("bad", ["", "bt_short", "bt_" + "g" * 16, 42, "x" * 200, {"id": "x"}])
def test_outbound_room_config_rejects_a_malformed_callback_task_id(bad: object) -> None:
    metadata = json.dumps(
        {
            "caal_outbound": True,
            "attempt_id": "abc",
            "destination": "+17805558345",
            "callback_task_id": bad,
        }
    )

    with pytest.raises(ValueError):
        OutboundRoomConfig.from_dispatch_metadata(metadata, allowed_destinations="+17805558345")


def test_callback_task_id_never_bypasses_the_destination_allowlist() -> None:
    metadata = json.dumps(
        {
            "caal_outbound": True,
            "attempt_id": "abc",
            "destination": "+17805550000",
            "callback_task_id": "bt_" + "cd" * 8,
        }
    )

    with pytest.raises(PermissionError):
        OutboundRoomConfig.from_dispatch_metadata(metadata, allowed_destinations="+17805558345")
