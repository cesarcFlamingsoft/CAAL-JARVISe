"""Regression coverage for typed LiveKit chat reaching CAAL's local handoff.

A spoken turn starts local handling from ``user_input_transcribed``. Typed chat
never fires that event, so "move the conversation to my phone" used to travel
all the way to Hermes, which has no outbound-call tool and answers that it
cannot hand a chat off. These tests pin the second entry point,
``VoiceAssistant.on_user_turn_completed``, and pin that a spoken turn is never
handled twice through it.
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from caal.handoff_intent import (
    ASK_CLARIFICATION_REPLY,
    ASK_CONFIRMATION_REPLY,
    STARTING_REPLY,
    PhoneHandoffController,
)

APPROVED = "+17805558345"
TYPED_REQUEST = "move the conversation to my phone"
NATURAL_REQUEST = "I need to take this on the road"
SUPPLIED_VOICE_REQUEST = "okay cool uh can we continue this conversation over the phone"
PARTIAL_REQUEST = "I'm leaving soon"

_voice_agent = None


def _load_voice_agent():
    """Load voice_agent.py once; it is a script, not an importable package."""
    global _voice_agent
    if _voice_agent is None:
        module_path = Path(__file__).parents[1] / "voice_agent.py"
        spec = importlib.util.spec_from_file_location("voice_agent_typed_chat_test", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _voice_agent = module
    return _voice_agent


class FakeSession:
    """Records everything the agent would speak."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def say(self, text: str) -> None:
        self.spoken.append(text)


class FakeCall:
    """Records every destination the controller tried to dial."""

    def __init__(self) -> None:
        self.destinations: list[str] = []

    async def __call__(self, destination: str, *, context: object = None) -> object:
        self.destinations.append(destination)
        return object()


class FakeEndCall:
    """Records every caller-requested termination of the current room."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self) -> None:
        self.calls += 1


class TypedMessage:
    """Stands in for the LiveKit ChatMessage handed to on_user_turn_completed."""

    def __init__(self, text: str) -> None:
        self.text_content = text


def _build_handler(session: FakeSession, call: FakeCall, end_call: FakeEndCall):
    voice_agent = _load_voice_agent()
    controller = PhoneHandoffController(start_call=call, allowed_destinations=APPROVED)
    return voice_agent.LocalTurnHandler(
        phone_handoff=controller, session=session, end_call=end_call
    )


async def _run_turn(handler, text: str, reached_llm: list[str]) -> None:
    """Mimic the LiveKit turn: a StopResponse means Hermes never sees the text."""
    voice_agent = _load_voice_agent()
    assistant = SimpleNamespace(_turn_consumed=handler.turn_consumed, _sync_return_context=None)
    try:
        await voice_agent.VoiceAssistant.on_user_turn_completed(
            assistant, SimpleNamespace(), TypedMessage(text)
        )
    except voice_agent.StopResponse:
        return
    reached_llm.append(text)


@pytest.mark.asyncio
async def test_typed_handoff_request_is_consumed_locally_and_never_reaches_hermes() -> None:
    session, call, end_call = FakeSession(), FakeCall(), FakeEndCall()
    handler = _build_handler(session, call, end_call)
    reached_llm: list[str] = []

    await _run_turn(handler, TYPED_REQUEST, reached_llm)

    assert reached_llm == []
    assert session.spoken == [ASK_CONFIRMATION_REPLY]
    assert call.destinations == []


@pytest.mark.asyncio
async def test_typed_confirmation_dials_the_single_approved_destination() -> None:
    session, call, end_call = FakeSession(), FakeCall(), FakeEndCall()
    handler = _build_handler(session, call, end_call)
    reached_llm: list[str] = []

    await _run_turn(handler, TYPED_REQUEST, reached_llm)
    await _run_turn(handler, "yes", reached_llm)

    assert reached_llm == []
    assert call.destinations == [APPROVED]
    assert session.spoken == [ASK_CONFIRMATION_REPLY, STARTING_REPLY]


@pytest.mark.asyncio
async def test_ordinary_typed_message_still_reaches_the_llm() -> None:
    session, call, end_call = FakeSession(), FakeCall(), FakeEndCall()
    handler = _build_handler(session, call, end_call)
    reached_llm: list[str] = []

    await _run_turn(handler, "what time is it", reached_llm)

    assert reached_llm == ["what time is it"]
    assert session.spoken == []
    assert call.destinations == []


@pytest.mark.asyncio
async def test_spoken_handoff_request_is_answered_once_not_twice() -> None:
    session, call, end_call = FakeSession(), FakeCall(), FakeEndCall()
    handler = _build_handler(session, call, end_call)
    reached_llm: list[str] = []

    # The STT event claims the turn; the same text then arrives as the new message.
    handler.on_final_transcript(TYPED_REQUEST)
    await _run_turn(handler, TYPED_REQUEST, reached_llm)

    assert reached_llm == []
    assert session.spoken == [ASK_CONFIRMATION_REPLY]


@pytest.mark.asyncio
async def test_voice_turn_waits_for_a_late_final_transcript_before_hermes() -> None:
    """Slow final STT must still let a local handoff consume the voice turn."""
    session, call, end_call = FakeSession(), FakeCall(), FakeEndCall()
    handler = _build_handler(session, call, end_call)
    reached_llm: list[str] = []

    handler.on_speech_transcript_started()
    committed_turn = asyncio.create_task(_run_turn(handler, "", reached_llm))
    await asyncio.sleep(0)
    handler.on_final_transcript(SUPPLIED_VOICE_REQUEST)
    await committed_turn

    assert reached_llm == []
    assert session.spoken == [ASK_CONFIRMATION_REPLY]
    assert call.destinations == []


@pytest.mark.asyncio
async def test_spoken_end_call_request_is_not_terminated_twice() -> None:
    session, call, end_call = FakeSession(), FakeCall(), FakeEndCall()
    handler = _build_handler(session, call, end_call)
    reached_llm: list[str] = []

    handler.on_final_transcript("hang up")
    await _run_turn(handler, "hang up", reached_llm)
    await asyncio.sleep(0)

    assert end_call.calls == 1
    assert call.destinations == []


@pytest.mark.asyncio
async def test_typed_end_call_request_outranks_a_pending_handoff_confirmation() -> None:
    session, call, end_call = FakeSession(), FakeCall(), FakeEndCall()
    handler = _build_handler(session, call, end_call)
    reached_llm: list[str] = []

    await _run_turn(handler, TYPED_REQUEST, reached_llm)
    await _run_turn(handler, "hang up", reached_llm)
    await asyncio.sleep(0)

    assert end_call.calls == 1
    assert call.destinations == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "expected_reply"),
    [
        (NATURAL_REQUEST, ASK_CONFIRMATION_REPLY),
        (SUPPLIED_VOICE_REQUEST, ASK_CONFIRMATION_REPLY),
        (PARTIAL_REQUEST, ASK_CLARIFICATION_REPLY),
    ],
)
async def test_inferred_intent_is_answered_locally_on_the_typed_path(
    text: str, expected_reply: str
) -> None:
    session, call, end_call = FakeSession(), FakeCall(), FakeEndCall()
    handler = _build_handler(session, call, end_call)
    reached_llm: list[str] = []

    await _run_turn(handler, text, reached_llm)

    assert reached_llm == []
    assert session.spoken == [expected_reply]
    assert call.destinations == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "expected_reply"),
    [
        (NATURAL_REQUEST, ASK_CONFIRMATION_REPLY),
        (SUPPLIED_VOICE_REQUEST, ASK_CONFIRMATION_REPLY),
        (PARTIAL_REQUEST, ASK_CLARIFICATION_REPLY),
    ],
)
async def test_spoken_and_typed_paths_answer_inferred_intent_identically(
    text: str, expected_reply: str
) -> None:
    """The STT event and typed chat must reach the same local decision, once."""
    session, call, end_call = FakeSession(), FakeCall(), FakeEndCall()
    handler = _build_handler(session, call, end_call)
    reached_llm: list[str] = []

    handler.on_final_transcript(text)
    await _run_turn(handler, text, reached_llm)

    assert reached_llm == []
    assert session.spoken == [expected_reply]
    assert call.destinations == []


@pytest.mark.asyncio
async def test_natural_request_still_needs_an_explicit_confirmation_before_dialing() -> None:
    session, call, end_call = FakeSession(), FakeCall(), FakeEndCall()
    handler = _build_handler(session, call, end_call)
    reached_llm: list[str] = []

    await _run_turn(handler, NATURAL_REQUEST, reached_llm)

    assert call.destinations == []

    await _run_turn(handler, "yes", reached_llm)

    assert reached_llm == []
    assert call.destinations == [APPROVED]


@pytest.mark.asyncio
async def test_clarified_intent_dials_only_after_a_second_explicit_confirmation() -> None:
    session, call, end_call = FakeSession(), FakeCall(), FakeEndCall()
    handler = _build_handler(session, call, end_call)
    reached_llm: list[str] = []

    await _run_turn(handler, PARTIAL_REQUEST, reached_llm)
    await _run_turn(handler, "yes", reached_llm)

    assert call.destinations == []

    await _run_turn(handler, "yes", reached_llm)

    assert reached_llm == []
    assert call.destinations == [APPROVED]
    assert session.spoken == [ASK_CLARIFICATION_REPLY, ASK_CONFIRMATION_REPLY, STARTING_REPLY]


# --- hang up now, call back when the background task is done -----------------

CALLBACK_REQUEST = "hang up and call me back when you're done"


class FakeArmCallback:
    """Records callback arming; stays pending until released, like a real hang-up."""

    def __init__(self) -> None:
        self.calls = 0
        self.release = asyncio.Event()

    async def __call__(self) -> None:
        self.calls += 1
        await self.release.wait()


def _build_callback_handler(session, call, end_call, arm):
    voice_agent = _load_voice_agent()
    controller = PhoneHandoffController(start_call=call, allowed_destinations=APPROVED)
    return voice_agent.LocalTurnHandler(
        phone_handoff=controller,
        session=session,
        end_call=end_call,
        arm_callback_and_end_call=arm,
    )


@pytest.mark.asyncio
async def test_callback_request_arms_callback_instead_of_plain_hang_up() -> None:
    session, call, end_call, arm = FakeSession(), FakeCall(), FakeEndCall(), FakeArmCallback()
    handler = _build_callback_handler(session, call, end_call, arm)
    reached_llm: list[str] = []

    await _run_turn(handler, CALLBACK_REQUEST, reached_llm)
    await asyncio.sleep(0)

    assert arm.calls == 1
    assert end_call.calls == 0
    assert call.destinations == []
    assert session.spoken == []
    arm.release.set()


@pytest.mark.asyncio
async def test_spoken_callback_request_is_armed_once_across_both_entry_points() -> None:
    session, call, end_call, arm = FakeSession(), FakeCall(), FakeEndCall(), FakeArmCallback()
    handler = _build_callback_handler(session, call, end_call, arm)
    reached_llm: list[str] = []

    handler.on_final_transcript(CALLBACK_REQUEST)
    await _run_turn(handler, CALLBACK_REQUEST, reached_llm)
    await asyncio.sleep(0)

    assert arm.calls == 1
    assert end_call.calls == 0
    arm.release.set()


@pytest.mark.asyncio
async def test_plain_hang_up_while_callback_is_closing_does_not_terminate_twice() -> None:
    session, call, end_call, arm = FakeSession(), FakeCall(), FakeEndCall(), FakeArmCallback()
    handler = _build_callback_handler(session, call, end_call, arm)
    reached_llm: list[str] = []

    await _run_turn(handler, CALLBACK_REQUEST, reached_llm)
    await asyncio.sleep(0)
    await _run_turn(handler, "hang up", reached_llm)
    await _run_turn(handler, CALLBACK_REQUEST, reached_llm)
    await asyncio.sleep(0)

    assert arm.calls == 1
    assert end_call.calls == 0
    arm.release.set()


@pytest.mark.asyncio
async def test_callback_request_is_left_to_hermes_when_nothing_can_arm_it() -> None:
    session, call, end_call = FakeSession(), FakeCall(), FakeEndCall()
    handler = _build_handler(session, call, end_call)
    reached_llm: list[str] = []

    await _run_turn(handler, CALLBACK_REQUEST, reached_llm)
    await asyncio.sleep(0)

    assert reached_llm == [CALLBACK_REQUEST]
    assert end_call.calls == 0
    assert call.destinations == []
    assert session.spoken == []


@pytest.mark.asyncio
async def test_typed_turn_is_released_when_no_handoff_is_configured() -> None:
    session, end_call = FakeSession(), FakeEndCall()
    voice_agent = _load_voice_agent()
    handler = voice_agent.LocalTurnHandler(phone_handoff=None, session=session, end_call=end_call)
    reached_llm: list[str] = []

    await _run_turn(handler, TYPED_REQUEST, reached_llm)

    assert reached_llm == [TYPED_REQUEST]
    assert session.spoken == []
