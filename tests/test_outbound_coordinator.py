from __future__ import annotations

import json

import pytest

from caal.outbound_calls import OutboundCallCoordinator, OutboundCallPolicy


class _Rooms:
    def __init__(self) -> None:
        self.created = []

    async def create_room(self, request):
        self.created.append(request)
        return object()


class _Dispatch:
    def __init__(self) -> None:
        self.created = []

    async def create_dispatch(self, request):
        self.created.append(request)
        return object()


class _LiveKit:
    def __init__(self) -> None:
        self.room = _Rooms()
        self.agent_dispatch = _Dispatch()


@pytest.mark.asyncio
async def test_coordinator_creates_isolated_room_then_dispatches_outbound_agent() -> None:
    livekit = _LiveKit()
    coordinator = OutboundCallCoordinator(
        policy=OutboundCallPolicy.from_csv("+17805558345"), livekit=livekit, agent_name="caal"
    )

    request = await coordinator.start("+17805558345")

    assert request.attempt_id
    assert len(livekit.room.created) == 1
    assert len(livekit.agent_dispatch.created) == 1
    room_request = livekit.room.created[0]
    dispatch_request = livekit.agent_dispatch.created[0]
    room_metadata = json.loads(room_request.metadata)
    assert room_metadata == {"caal_outbound": True, "attempt_id": request.attempt_id}
    assert dispatch_request.room == room_request.name
    assert dispatch_request.agent_name == "caal"
    assert json.loads(dispatch_request.metadata) == {
        **request.dispatch_metadata(),
        "destination": "+17805558345",
    }


@pytest.mark.asyncio
async def test_coordinator_does_not_create_room_for_unapproved_destination() -> None:
    livekit = _LiveKit()
    coordinator = OutboundCallCoordinator(
        policy=OutboundCallPolicy.from_csv("+17805558345"), livekit=livekit, agent_name="caal"
    )

    with pytest.raises(PermissionError):
        await coordinator.start("+17805550000")

    assert livekit.room.created == []
    assert livekit.agent_dispatch.created == []


@pytest.mark.asyncio
async def test_handoff_context_travels_only_in_private_dispatch_metadata() -> None:
    from caal.handoff_context import ConversationSnapshot, SnapshotTurn

    livekit = _LiveKit()
    coordinator = OutboundCallCoordinator(
        policy=OutboundCallPolicy.from_csv("+17805558345"), livekit=livekit, agent_name="caal"
    )
    snapshot = ConversationSnapshot(
        turns=(
            SnapshotTurn(role="user", text="where were we with the trip"),
            SnapshotTurn(role="assistant", text="Booking the Friday flight."),
        )
    )

    request = await coordinator.start("+17805558345", context=snapshot)

    room_request = livekit.room.created[0]
    dispatch_request = livekit.agent_dispatch.created[0]
    assert json.loads(room_request.metadata) == {
        "caal_outbound": True,
        "attempt_id": request.attempt_id,
    }
    assert "where were we" not in room_request.metadata
    assert "Friday flight" not in room_request.metadata
    dispatch_metadata = json.loads(dispatch_request.metadata)
    assert dispatch_metadata["handoff_context"] == snapshot.to_metadata()
    assert dispatch_metadata["destination"] == "+17805558345"
    assert "where were we" not in repr(request)


@pytest.mark.asyncio
async def test_coordinator_without_context_keeps_the_legacy_dispatch_shape() -> None:
    livekit = _LiveKit()
    coordinator = OutboundCallCoordinator(
        policy=OutboundCallPolicy.from_csv("+17805558345"), livekit=livekit, agent_name="caal"
    )

    request = await coordinator.start("+17805558345")

    assert json.loads(livekit.agent_dispatch.created[0].metadata) == {
        "caal_outbound": True,
        "attempt_id": request.attempt_id,
        "destination": "+17805558345",
    }


# --- ledger-backed continuity ------------------------------------------------


@pytest.fixture
def ledger(monkeypatch, tmp_path):
    from caal import conversation_ledger

    monkeypatch.setattr(conversation_ledger, "STORE_PATH", tmp_path / "assistant.sqlite3")
    return conversation_ledger


@pytest.mark.asyncio
async def test_confirmed_continuity_dispatches_only_the_opaque_conversation_id(ledger) -> None:
    livekit = _LiveKit()
    coordinator = OutboundCallCoordinator(
        policy=OutboundCallPolicy.from_csv("+17805558345"), livekit=livekit, agent_name="caal"
    )
    conversation_id = ledger.open_conversation(session_key="web-room-1", now=1000)
    ledger.append_turn(conversation_id, "user", "where were we with the trip", now=1001)
    ledger.append_turn(conversation_id, "assistant", "Booking the Friday flight.", now=1002)

    request = await coordinator.start("+17805558345", conversation_id=conversation_id)

    room_request = livekit.room.created[0]
    dispatch_request = livekit.agent_dispatch.created[0]
    assert json.loads(room_request.metadata) == {
        "caal_outbound": True,
        "attempt_id": request.attempt_id,
    }
    assert conversation_id not in room_request.metadata
    dispatch_metadata = json.loads(dispatch_request.metadata)
    assert dispatch_metadata == {
        "caal_outbound": True,
        "attempt_id": request.attempt_id,
        "destination": "+17805558345",
        "conversation_id": conversation_id,
    }
    assert "handoff_context" not in dispatch_metadata
    assert "where were we" not in dispatch_request.metadata
    assert "Friday flight" not in dispatch_request.metadata
    assert conversation_id not in repr(request)


@pytest.mark.asyncio
async def test_continuity_is_linked_as_pending_before_the_worker_is_dispatched(ledger) -> None:
    """The origin session may close while the phone rings; the ledger must survive."""
    livekit = _LiveKit()
    coordinator = OutboundCallCoordinator(
        policy=OutboundCallPolicy.from_csv("+17805558345"), livekit=livekit, agent_name="caal"
    )
    conversation_id = ledger.open_conversation(session_key="web-room-1", now=1000)
    ledger.append_turn(conversation_id, "user", "hello there", now=1001)

    request = await coordinator.start("+17805558345", conversation_id=conversation_id)

    assert ledger.close_session(conversation_id, session_key="web-room-1") is False
    context = ledger.claim_continuation(conversation_id, session_key=request.attempt_id)
    assert context is not None
    assert [turn.text for turn in context.turns] == ["hello there"]


@pytest.mark.asyncio
async def test_unknown_or_malformed_conversation_id_never_dispatches(ledger) -> None:
    livekit = _LiveKit()
    coordinator = OutboundCallCoordinator(
        policy=OutboundCallPolicy.from_csv("+17805558345"), livekit=livekit, agent_name="caal"
    )

    with pytest.raises(LookupError):
        await coordinator.start("+17805558345", conversation_id="never-opened-id")
    with pytest.raises(ValueError):
        await coordinator.start("+17805558345", conversation_id="not a valid/id")

    assert livekit.room.created == []
    assert livekit.agent_dispatch.created == []


@pytest.mark.asyncio
async def test_conversation_id_never_bypasses_the_destination_allowlist(ledger) -> None:
    livekit = _LiveKit()
    coordinator = OutboundCallCoordinator(
        policy=OutboundCallPolicy.from_csv("+17805558345"), livekit=livekit, agent_name="caal"
    )
    conversation_id = ledger.open_conversation(session_key="web-room-1", now=1000)

    with pytest.raises(PermissionError):
        await coordinator.start("+17805550000", conversation_id=conversation_id)

    assert livekit.room.created == []
    # Nothing was linked either: no pending continuation holds the ledger open.
    assert ledger.close_session(conversation_id, session_key="web-room-1") is True


# --- background task callback -------------------------------------------------


@pytest.mark.asyncio
async def test_callback_task_id_travels_only_in_private_dispatch_metadata() -> None:
    livekit = _LiveKit()
    coordinator = OutboundCallCoordinator(
        policy=OutboundCallPolicy.from_csv("+17805558345"), livekit=livekit, agent_name="caal"
    )
    task_id = "bt_" + "ab" * 8

    request = await coordinator.start("+17805558345", callback_task_id=task_id)

    room_request = livekit.room.created[0]
    dispatch_request = livekit.agent_dispatch.created[0]
    assert json.loads(room_request.metadata) == {
        "caal_outbound": True,
        "attempt_id": request.attempt_id,
    }
    assert task_id not in room_request.metadata
    assert json.loads(dispatch_request.metadata)["callback_task_id"] == task_id
    assert task_id not in repr(request)


@pytest.mark.asyncio
async def test_callback_never_bypasses_the_allowlist_or_accepts_a_bad_task_id() -> None:
    livekit = _LiveKit()
    coordinator = OutboundCallCoordinator(
        policy=OutboundCallPolicy.from_csv("+17805558345"), livekit=livekit, agent_name="caal"
    )

    with pytest.raises(PermissionError):
        await coordinator.start("+17805550000", callback_task_id="bt_" + "ab" * 8)
    with pytest.raises(ValueError):
        await coordinator.start("+17805558345", callback_task_id="not-a-task")

    assert livekit.room.created == []
    assert livekit.agent_dispatch.created == []
