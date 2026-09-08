"""Per-user phone policy: only the verified user's own approved number is ever dialed.

Under multi-user, the outbound destination is never a shared allowlist and
never anything the caller said or typed. It is the single approved E.164
number stored encrypted on the user's profile, resolved server-side at the
moment of dialing and re-resolved by the outbound worker before the SIP call
is placed. Users without a number are declined with an explanation. The
existing allowlist path is kept for legacy single-user deployments.
"""

from __future__ import annotations

import json

import pytest

from caal import background_tasks
from caal.background_task_session import (
    CALLBACK_ARMED_REPLY,
    CALLBACK_NOTHING_RUNNING_REPLY,
    BackgroundTaskBridge,
)
from caal.background_tasks import SUCCEEDED
from caal.handoff_intent import (
    HANDOFF_CONTROL_REPLIES,
    NO_CALLBACK_NUMBER_REPLY,
    STARTING_REPLY,
    PhoneHandoffController,
)
from caal.outbound_calls import OutboundCallCoordinator, OutboundCallPolicy
from caal.outbound_runtime import OutboundRoomConfig

ANA = "usr_" + "a" * 24
BO = "usr_" + "b" * 24
ANA_NUMBER = "+17805558345"
LEGACY_ALLOWLIST = "+17805550000"


@pytest.fixture(autouse=True)
def store(monkeypatch, tmp_path):
    monkeypatch.setattr(background_tasks, "STORE_PATH", tmp_path / "assistant.sqlite3")


class _LiveKit:
    def __init__(self) -> None:
        self.rooms: list = []
        self.dispatches: list = []
        self.room = self
        self.agent_dispatch = self

    async def create_room(self, request):
        self.rooms.append(request)

    async def create_dispatch(self, request):
        self.dispatches.append(request)


# --- policy -------------------------------------------------------------------------------


def test_policy_for_a_single_users_number_allows_only_that_number() -> None:
    policy = OutboundCallPolicy.for_destination(ANA_NUMBER)

    request = policy.authorize(ANA_NUMBER, user_id=ANA)

    assert request.destination == ANA_NUMBER
    assert request.user_id == ANA
    assert request.dispatch_metadata()["user_id"] == ANA
    with pytest.raises(PermissionError):
        policy.authorize(LEGACY_ALLOWLIST, user_id=ANA)
    with pytest.raises(ValueError):
        policy.authorize(ANA_NUMBER, user_id="not-a-user")
    with pytest.raises(ValueError):
        OutboundCallPolicy.for_destination("780-555-8345")


def test_legacy_requests_carry_no_user_id() -> None:
    request = OutboundCallPolicy.from_csv(LEGACY_ALLOWLIST).authorize(LEGACY_ALLOWLIST)
    assert request.user_id is None
    assert "user_id" not in request.dispatch_metadata()


@pytest.mark.asyncio
async def test_coordinator_dispatches_the_user_id_privately_and_never_in_room_metadata() -> None:
    livekit = _LiveKit()
    coordinator = OutboundCallCoordinator(
        policy=OutboundCallPolicy.for_destination(ANA_NUMBER), livekit=livekit, agent_name="caal"
    )

    request = await coordinator.start(ANA_NUMBER, user_id=ANA)

    assert request.user_id == ANA
    dispatch = json.loads(livekit.dispatches[0].metadata)
    assert dispatch["user_id"] == ANA and dispatch["destination"] == ANA_NUMBER
    assert ANA not in livekit.rooms[0].metadata
    assert ANA_NUMBER not in livekit.rooms[0].metadata


# --- worker-side revalidation ------------------------------------------------------------------


def _metadata(**extra) -> str:
    return json.dumps(
        {"caal_outbound": True, "attempt_id": "abc", "destination": ANA_NUMBER, **extra}
    )


def test_worker_revalidates_the_users_current_number_instead_of_the_allowlist() -> None:
    resolved: list[str] = []

    def resolve(user_id: str) -> str | None:
        resolved.append(user_id)
        return ANA_NUMBER

    config = OutboundRoomConfig.from_dispatch_metadata(
        _metadata(user_id=ANA),
        allowed_destinations=LEGACY_ALLOWLIST,
        resolve_user_destination=resolve,
    )

    assert config is not None
    assert config.user_id == ANA and config.destination == ANA_NUMBER
    assert resolved == [ANA]
    assert ANA_NUMBER not in repr(config)


def test_worker_refuses_when_the_number_was_cleared_changed_or_cannot_be_resolved() -> None:
    with pytest.raises(PermissionError):
        OutboundRoomConfig.from_dispatch_metadata(
            _metadata(user_id=ANA),
            allowed_destinations=LEGACY_ALLOWLIST,
            resolve_user_destination=lambda user_id: None,
        )
    with pytest.raises(PermissionError):
        OutboundRoomConfig.from_dispatch_metadata(
            _metadata(user_id=ANA),
            allowed_destinations=LEGACY_ALLOWLIST,
            resolve_user_destination=lambda user_id: "+17805559999",
        )
    # No resolver means no way to revalidate: fail closed, even if the legacy
    # allowlist happens to contain the number.
    with pytest.raises(PermissionError):
        OutboundRoomConfig.from_dispatch_metadata(
            _metadata(user_id=ANA), allowed_destinations=ANA_NUMBER
        )


@pytest.mark.parametrize("bad", ["", "usr_x", 42, "' OR 1=1", "x" * 300])
def test_worker_rejects_malformed_user_ids(bad) -> None:
    with pytest.raises(ValueError):
        OutboundRoomConfig.from_dispatch_metadata(
            _metadata(user_id=bad),
            allowed_destinations=LEGACY_ALLOWLIST,
            resolve_user_destination=lambda user_id: ANA_NUMBER,
        )


def test_legacy_dispatch_without_a_user_still_uses_the_allowlist() -> None:
    config = OutboundRoomConfig.from_dispatch_metadata(
        json.dumps({"caal_outbound": True, "attempt_id": "abc", "destination": LEGACY_ALLOWLIST}),
        allowed_destinations=LEGACY_ALLOWLIST,
        resolve_user_destination=lambda user_id: ANA_NUMBER,
    )
    assert config is not None and config.user_id is None


# --- handoff controller -------------------------------------------------------------------


class _Session:
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.history = None

    async def say(self, text: str, **_) -> None:
        self.spoken.append(text)


class _Call:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, destination: str, **kwargs) -> object:
        self.calls.append((destination, kwargs))
        return object()


@pytest.mark.asyncio
async def test_handoff_dials_the_users_own_number_resolved_at_confirmation_time() -> None:
    session, call = _Session(), _Call()
    numbers = {"current": None}
    controller = PhoneHandoffController(
        start_call=call,
        allowed_destinations=LEGACY_ALLOWLIST,  # must be ignored for a user-scoped session
        conversation_id=None,
        user_id=ANA,
        destination_resolver=lambda: numbers["current"],
    )

    await controller.handle_final_transcript("continue this conversation on my phone", session)
    assert await controller.handle_final_transcript("yes", session) is True
    assert call.calls == []
    assert session.spoken[-1] == NO_CALLBACK_NUMBER_REPLY
    assert NO_CALLBACK_NUMBER_REPLY in HANDOFF_CONTROL_REPLIES
    assert "administrator" in NO_CALLBACK_NUMBER_REPLY.lower()
    assert "profile" in NO_CALLBACK_NUMBER_REPLY.lower()

    numbers["current"] = ANA_NUMBER
    await controller.handle_final_transcript("continue this conversation on my phone", session)
    assert await controller.handle_final_transcript("yes", session) is True
    assert [destination for destination, _ in call.calls] == [ANA_NUMBER]
    assert call.calls[0][1]["user_id"] == ANA
    assert session.spoken[-1] == STARTING_REPLY
    assert all(ANA_NUMBER not in text and "8345" not in text for text in session.spoken)


@pytest.mark.asyncio
async def test_handoff_never_reads_a_number_from_the_transcript() -> None:
    session, call = _Session(), _Call()
    controller = PhoneHandoffController(
        start_call=call,
        allowed_destinations="",
        user_id=ANA,
        destination_resolver=lambda: ANA_NUMBER,
    )

    consumed = await controller.handle_final_transcript(
        "call me at 780 555 9999 to continue this conversation", session
    )

    assert call.calls == []
    assert consumed is False or session.spoken[-1] != STARTING_REPLY


# --- callbacks bound to the user --------------------------------------------------------------


class _Execute:
    async def __call__(self, request: str, context: str) -> str:
        return "done"


@pytest.mark.asyncio
async def test_user_bound_callback_is_dialed_through_the_user_dialer_only() -> None:
    dialed: list[tuple[str, str]] = []
    legacy_dialed: list[tuple[str, str]] = []

    async def dial_user(user_id: str, task_id: str) -> None:
        dialed.append((user_id, task_id))

    async def dial_legacy(destination: str, task_id: str) -> None:
        legacy_dialed.append((destination, task_id))

    bridge = BackgroundTaskBridge(
        execute=_Execute(),
        session_key="phone-room",
        user_id=ANA,
        dial_callback=dial_legacy,
        dial_user_callback=dial_user,
    )
    # Another user's bridge in the same process. Started before any work is
    # queued so its runner idles and cannot be the one to execute the task.
    other = BackgroundTaskBridge(execute=_Execute(), session_key="phone-room", user_id=BO)
    await bridge.start()
    await other.start()
    try:
        session = _Session()
        assert await bridge.arm_callback(None, session, user_id=ANA) is False
        assert session.spoken == [CALLBACK_NOTHING_RUNNING_REPLY]

        task = background_tasks.enqueue("find fares", session_key="phone-room", user_id=ANA)
        assert await bridge.arm_callback(None, session, user_id=ANA) is True
        assert session.spoken[-1] == CALLBACK_ARMED_REPLY
        # Another user's bridge can never arm it, even naming the right task owner.
        assert await other.arm_callback(None, _Session(), user_id=ANA) is False
        assert await other.arm_callback(None, _Session(), user_id=BO) is False
        await other.close()
        await other.abandon()

        background_tasks._mark_running(task.task_id)
        background_tasks._finish(task.task_id, SUCCEEDED, result="answer")
        await bridge._on_settled(task.task_id)

        assert dialed == [(ANA, task.task_id)]
        assert legacy_dialed == []
        assert background_tasks.callback_armed(task.task_id) is False
    finally:
        await bridge.close()
        await bridge.abandon()


@pytest.mark.asyncio
async def test_user_bound_callback_without_a_user_dialer_falls_back_without_dialing() -> None:
    sent: list[str] = []

    async def fallback(text: str) -> None:
        sent.append(text)

    bridge = BackgroundTaskBridge(
        execute=_Execute(), session_key="phone-room", user_id=ANA, fallback=fallback
    )
    await bridge.start()
    try:
        task = background_tasks.enqueue("find fares", session_key="phone-room", user_id=ANA)
        assert await bridge.arm_callback(None, _Session(), user_id=ANA) is True
        background_tasks._mark_running(task.task_id)
        background_tasks._finish(task.task_id, SUCCEEDED, result="answer")
        await bridge._on_settled(task.task_id)
        assert background_tasks.callback_armed(task.task_id) is False
        assert len(sent) == 1 and "answer" in sent[0]
    finally:
        await bridge.close()
        await bridge.abandon()
