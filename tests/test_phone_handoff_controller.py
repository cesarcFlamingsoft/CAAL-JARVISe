"""Integration coverage for the confirm-then-dial phone handoff controller.

These tests drive the controller with fakes for the voice session and for the
real OutboundCallCoordinator seam, proving that a call only ever happens after
an explicit confirmation and only ever to the single approved destination.
"""

from __future__ import annotations

import json
import logging

import pytest
from livekit.agents import llm

from caal.handoff_context import REDACTED, ConversationSnapshot
from caal.handoff_intent import ASK_CONFIRMATION_REPLY, PhoneHandoffController
from caal.outbound_calls import OutboundCallCoordinator, OutboundCallPolicy

APPROVED = "+17805558345"


class FakeSession:
    """Records everything the agent would speak and keeps a LiveKit-style history."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.history = llm.ChatContext.empty()
        self.history.add_message(role="system", content="You are JARVIS. Operator prompt.")

    async def say(self, text: str) -> None:
        self.spoken.append(text)
        self.history.add_message(role="assistant", content=text)

    def user_said(self, text: str) -> None:
        self.history.add_message(role="user", content=text)


class FakeCall:
    """Records every destination the controller tried to dial."""

    def __init__(self, *, fail: Exception | None = None) -> None:
        self.destinations: list[str] = []
        self.contexts: list[object] = []
        self._fail = fail

    async def __call__(self, destination: str, *, context: object = None) -> object:
        self.destinations.append(destination)
        self.contexts.append(context)
        if self._fail is not None:
            raise self._fail
        return object()


class _Rooms:
    def __init__(self) -> None:
        self.created: list[object] = []

    async def create_room(self, request):
        self.created.append(request)
        return object()


class _Dispatch:
    def __init__(self) -> None:
        self.created: list[object] = []

    async def create_dispatch(self, request):
        self.created.append(request)
        return object()


class _LiveKit:
    def __init__(self) -> None:
        self.room = _Rooms()
        self.agent_dispatch = _Dispatch()


@pytest.mark.asyncio
async def test_request_then_confirmation_places_exactly_one_approved_call() -> None:
    session, call = FakeSession(), FakeCall()
    controller = PhoneHandoffController(start_call=call, allowed_destinations=APPROVED)

    consumed = await controller.handle_final_transcript(
        "continue this conversation on my phone", session
    )

    assert consumed is True
    assert call.destinations == []
    assert len(session.spoken) == 1

    assert await controller.handle_final_transcript("yes", session)

    assert call.destinations == [APPROVED]
    assert len(session.spoken) == 2


@pytest.mark.asyncio
async def test_denied_confirmation_places_no_call() -> None:
    session, call = FakeSession(), FakeCall()
    controller = PhoneHandoffController(start_call=call, allowed_destinations=APPROVED)

    await controller.handle_final_transcript("continue this conversation on my phone", session)

    assert await controller.handle_final_transcript("no", session)
    assert call.destinations == []
    assert controller.awaiting_confirmation is False


@pytest.mark.asyncio
async def test_missing_confirmation_places_no_call_and_releases_the_turn() -> None:
    session, call = FakeSession(), FakeCall()
    controller = PhoneHandoffController(start_call=call, allowed_destinations=APPROVED)

    await controller.handle_final_transcript("continue this conversation on my phone", session)
    consumed = await controller.handle_final_transcript("what time is it", session)

    assert consumed is False
    assert call.destinations == []
    assert controller.awaiting_confirmation is False


@pytest.mark.asyncio
async def test_confirmation_alone_never_dials_without_a_prior_request() -> None:
    session, call = FakeSession(), FakeCall()
    controller = PhoneHandoffController(start_call=call, allowed_destinations=APPROVED)

    consumed = await controller.handle_final_transcript("yes call me now", session)

    assert consumed is False
    assert call.destinations == []
    assert session.spoken == []


@pytest.mark.asyncio
async def test_no_approved_destination_declines_safely_without_dialing() -> None:
    session, call = FakeSession(), FakeCall()
    controller = PhoneHandoffController(start_call=call, allowed_destinations="")

    await controller.handle_final_transcript("continue this conversation on my phone", session)
    consumed = await controller.handle_final_transcript("yes", session)

    assert consumed is True
    assert call.destinations == []
    assert session.spoken[-1]


@pytest.mark.asyncio
async def test_multiple_approved_destinations_decline_rather_than_guess() -> None:
    session, call = FakeSession(), FakeCall()
    controller = PhoneHandoffController(
        start_call=call, allowed_destinations=f"{APPROVED},+17805550000"
    )

    await controller.handle_final_transcript("continue this conversation on my phone", session)
    await controller.handle_final_transcript("yes", session)

    assert call.destinations == []


@pytest.mark.asyncio
async def test_spoken_replies_never_disclose_the_destination_number() -> None:
    session, call = FakeSession(), FakeCall()
    controller = PhoneHandoffController(start_call=call, allowed_destinations=APPROVED)

    await controller.handle_final_transcript("continue this conversation on my phone", session)
    await controller.handle_final_transcript("yes", session)

    assert all(APPROVED not in text for text in session.spoken)
    assert all("7805558345" not in text.replace(" ", "") for text in session.spoken)


@pytest.mark.asyncio
async def test_dial_failure_is_reported_without_raising_into_the_session() -> None:
    session = FakeSession()
    call = FakeCall(fail=PermissionError("Destination is not approved for outbound calling"))
    controller = PhoneHandoffController(start_call=call, allowed_destinations=APPROVED)

    await controller.handle_final_transcript("continue this conversation on my phone", session)
    consumed = await controller.handle_final_transcript("yes", session)

    assert consumed is True
    assert len(session.spoken) == 2


@pytest.mark.asyncio
async def test_confirmed_handoff_dispatches_the_named_caal_worker_via_coordinator() -> None:
    livekit = _LiveKit()
    coordinator = OutboundCallCoordinator(
        policy=OutboundCallPolicy.from_csv(APPROVED), livekit=livekit, agent_name="caal"
    )
    session = FakeSession()
    controller = PhoneHandoffController(start_call=coordinator.start, allowed_destinations=APPROVED)

    await controller.handle_final_transcript("continue this conversation on my phone", session)
    await controller.handle_final_transcript("yes", session)

    assert len(livekit.room.created) == 1
    assert len(livekit.agent_dispatch.created) == 1
    dispatch = livekit.agent_dispatch.created[0]
    assert dispatch.agent_name == "caal"
    assert json.loads(dispatch.metadata)["destination"] == APPROVED


@pytest.mark.asyncio
async def test_unconfirmed_handoff_never_reaches_the_coordinator() -> None:
    livekit = _LiveKit()
    coordinator = OutboundCallCoordinator(
        policy=OutboundCallPolicy.from_csv(APPROVED), livekit=livekit, agent_name="caal"
    )
    session = FakeSession()
    controller = PhoneHandoffController(start_call=coordinator.start, allowed_destinations=APPROVED)

    await controller.handle_final_transcript("continue this conversation on my phone", session)
    await controller.handle_final_transcript("no", session)

    assert livekit.room.created == []
    assert livekit.agent_dispatch.created == []


# --- conversation-context continuity ----------------------------------------


@pytest.mark.asyncio
async def test_confirmed_handoff_captures_recent_visible_history_at_the_boundary() -> None:
    session, call = FakeSession(), FakeCall()
    controller = PhoneHandoffController(start_call=call, allowed_destinations=APPROVED)
    session.user_said("How long should the bread proof?")
    await session.say("About an hour at room temperature.")

    session.user_said("continue this conversation on my phone")
    await controller.handle_final_transcript("continue this conversation on my phone", session)
    assert call.contexts == []

    # History can still grow between the question and the answer.
    session.user_said("wait, and what oven temperature?")
    await session.say("Two hundred and thirty degrees.")
    session.user_said("yes")
    await controller.handle_final_transcript("yes", session)

    assert call.destinations == [APPROVED]
    assert len(call.contexts) == 1
    snapshot = call.contexts[0]
    assert isinstance(snapshot, ConversationSnapshot)
    texts = [turn.text for turn in snapshot.turns]
    assert texts[-1] == "Two hundred and thirty degrees."
    assert "wait, and what oven temperature?" in texts
    assert "How long should the bread proof?" in texts
    assert ASK_CONFIRMATION_REPLY not in texts
    assert all("Operator prompt" not in text for text in texts)
    assert all(turn.role in {"user", "assistant"} for turn in snapshot.turns)


@pytest.mark.asyncio
async def test_captured_context_is_redacted_and_never_logged(caplog) -> None:
    session, call = FakeSession(), FakeCall()
    controller = PhoneHandoffController(start_call=call, allowed_destinations=APPROVED)
    session.user_said("the garage code is 8675309 and my password is Tr0ub4dor&3")
    await session.say("Understood.")

    with caplog.at_level(logging.DEBUG):
        await controller.handle_final_transcript("continue this conversation on my phone", session)
        await controller.handle_final_transcript("yes", session)

    snapshot = call.contexts[0]
    flat = " ".join(turn.text for turn in snapshot.turns)
    assert "8675309" not in flat
    assert "Tr0ub4dor&3" not in flat
    assert REDACTED in flat
    for record in caplog.records:
        assert "garage code" not in record.getMessage()
        assert "Understood." not in record.getMessage()


@pytest.mark.asyncio
async def test_handoff_without_history_still_dials_with_no_context() -> None:
    session, call = FakeSession(), FakeCall()
    del session.history
    controller = PhoneHandoffController(start_call=call, allowed_destinations=APPROVED)

    await controller.handle_final_transcript("continue this conversation on my phone", session)
    await controller.handle_final_transcript("yes", session)

    assert call.destinations == [APPROVED]
    assert call.contexts == [None]


@pytest.mark.asyncio
async def test_context_capture_failure_never_blocks_the_confirmed_call() -> None:
    session, call = FakeSession(), FakeCall()
    controller = PhoneHandoffController(start_call=call, allowed_destinations=APPROVED)

    class _Broken:
        @property
        def items(self):
            raise RuntimeError("history unavailable")

    session.history = _Broken()

    await controller.handle_final_transcript("continue this conversation on my phone", session)
    await controller.handle_final_transcript("yes", session)

    assert call.destinations == [APPROVED]
    assert call.contexts == [None]


@pytest.mark.asyncio
async def test_denied_handoff_captures_nothing() -> None:
    session, call = FakeSession(), FakeCall()
    controller = PhoneHandoffController(start_call=call, allowed_destinations=APPROVED)
    session.user_said("private planning detail")

    await controller.handle_final_transcript("continue this conversation on my phone", session)
    await controller.handle_final_transcript("no", session)

    assert call.contexts == []


# --- ledger-backed continuity ------------------------------------------------


@pytest.fixture
def ledger(monkeypatch, tmp_path):
    from caal import conversation_ledger

    monkeypatch.setattr(conversation_ledger, "STORE_PATH", tmp_path / "assistant.sqlite3")
    return conversation_ledger


class LedgerCall(FakeCall):
    """Accepts the opaque conversation ID the way the coordinator does."""

    def __init__(self) -> None:
        super().__init__()
        self.conversation_ids: list[object] = []

    async def __call__(self, destination, *, context=None, conversation_id=None):
        self.conversation_ids.append(conversation_id)
        return await super().__call__(destination, context=context)


@pytest.mark.asyncio
async def test_confirmed_handoff_sends_the_conversation_id_instead_of_a_snapshot(ledger) -> None:
    session, call = FakeSession(), LedgerCall()
    conversation_id = ledger.open_conversation(session_key="web-room-1", now=1000)
    controller = PhoneHandoffController(
        start_call=call, allowed_destinations=APPROVED, conversation_id=conversation_id
    )
    session.user_said("How long should the bread proof?")
    await session.say("About an hour at room temperature.")

    await controller.handle_final_transcript("continue this conversation on my phone", session)
    await controller.handle_final_transcript("yes", session)

    assert call.destinations == [APPROVED]
    assert call.conversation_ids == [conversation_id]
    assert call.contexts == [None]


@pytest.mark.asyncio
async def test_controller_without_a_conversation_keeps_the_snapshot_path(ledger) -> None:
    session, call = FakeSession(), LedgerCall()
    controller = PhoneHandoffController(start_call=call, allowed_destinations=APPROVED)
    session.user_said("How long should the bread proof?")

    await controller.handle_final_transcript("continue this conversation on my phone", session)
    await controller.handle_final_transcript("yes", session)

    assert call.conversation_ids == [None]
    assert isinstance(call.contexts[0], ConversationSnapshot)


@pytest.mark.asyncio
async def test_handoff_control_turns_never_land_in_the_ledger(ledger) -> None:
    """The recorder sees what LiveKit adds to history; control replies must be skipped."""
    from caal.conversation_ledger import ConversationRecorder
    from caal.handoff_intent import HANDOFF_CONTROL_REPLIES

    conversation_id = ledger.open_conversation(session_key="web-room-1", now=1000)
    recorder = ConversationRecorder(exclude_assistant_texts=HANDOFF_CONTROL_REPLIES)
    recorder.bind(conversation_id)

    class RecordingSession(FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.history.add_message(role="user", content="How long should the bread proof?")
            for item in self.history.items:
                recorder.record(item)

        async def say(self, text: str) -> None:
            await super().say(text)
            recorder.record(self.history.items[-1])

        def user_said(self, text: str) -> None:
            super().user_said(text)
            recorder.record(self.history.items[-1])

    session, call = RecordingSession(), LedgerCall()
    await session.say("About an hour at room temperature.")
    controller = PhoneHandoffController(
        start_call=call, allowed_destinations=APPROVED, conversation_id=conversation_id
    )
    # A consumed user turn is never added to LiveKit history (StopResponse), so the
    # recorder only ever sees the assistant's control replies, which it must skip.
    await controller.handle_final_transcript("continue this conversation on my phone", session)
    await controller.handle_final_transcript("yes", session)

    ledger.link_continuation(conversation_id, session_key="attempt-x", now=1010)
    context = ledger.claim_continuation(conversation_id, session_key="attempt-x", now=1011)
    assert context is not None
    texts = [turn.text for turn in context.turns]
    assert texts == ["How long should the bread proof?", "About an hour at room temperature."]
    assert all(reply not in texts for reply in HANDOFF_CONTROL_REPLIES)
