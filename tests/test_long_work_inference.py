"""Regression coverage for inferred long-running work.

The reported failure: a user asked JARVIS to create a PDF, then asked for
status, and the background_tasks table stayed empty. Only explicit "in the
background" phrasing was recognised, so a plain "create a PDF of ..." went to
Hermes synchronously and nothing was scheduled or acknowledged.

A clear long-running request (build a PDF, report, or document; research,
investigate, or compile something) is now scheduled on the durable queue
before Hermes sees the turn. The reply is a concise acknowledgement. On an
outbound phone leg it also asks whether to call back when the work is done;
only a fresh, explicit "yes" in the next turn arms the existing protected
callback and ends that call. "No" keeps the task running and says so. Any
other answer, silence, or a timeout fails closed: no callback, the task keeps
running, and normal conversation proceeds. Small ordinary questions are never
backgrounded, and the existing explicit commands are unchanged.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from caal import background_tasks, conversation_ledger
from caal.background_task_session import (
    BACKGROUND_ACK_REPLY,
    BACKGROUND_CONTROL_REPLIES,
    BACKGROUND_STATUS_WORKING_REPLY,
    CALLBACK_ARMED_REPLY,
    LONG_WORK_ACK_REPLY,
    LONG_WORK_OFFER_CALLBACK_REPLY,
    BackgroundTaskBridge,
    BackgroundTurnOutcome,
)
from caal.background_tasks import (
    RUNNING,
    background_status_requested,
    background_task_requested,
    callback_armed,
    list_tasks,
    long_running_work_inferred,
)
from caal.end_call_intent import (
    ASK_CALLBACK_REPLY,
    CALLBACK_OFFER_DECLINED_REPLY,
    END_CALL_CONTROL_REPLIES,
    NO_TASK_REPLY,
    EndCallAction,
    EndCallIntentMachine,
)
from caal.handoff_intent import ASK_CONFIRMATION_REPLY, PhoneHandoffController
from caal.outbound_runtime import OutboundRoomConfig

APPROVED = "+17805558345"
WEB_ROOM = "room-web-1"
PHONE_ROOM = "caal-outbound-abc"
ATTEMPT_ID = "abc"
PDF_REQUEST = "create a PDF summarizing everything we discussed about the Lisbon trip"
RESEARCH_REQUEST = "research the best electric bikes under two thousand dollars"
EXPLICIT_REQUEST = "look into hotel prices in the background and let me know when it's done"

_voice_agent = None


def _load_voice_agent():
    """Load voice_agent.py once; it is a script, not an importable package."""
    global _voice_agent
    if _voice_agent is None:
        module_path = Path(__file__).parents[1] / "voice_agent.py"
        spec = importlib.util.spec_from_file_location("voice_agent_long_work_test", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _voice_agent = module
    return _voice_agent


# --- fakes -------------------------------------------------------------------


class FakeSession:
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.history = SimpleNamespace(items=[])

    async def say(self, text: str, **_: object) -> None:
        self.spoken.append(text)


class FakeExecute:
    """Injected worker: holds every task open until released."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.gate = asyncio.Event()

    async def __call__(self, request: str, context: str) -> str:
        self.calls.append((request, context))
        await self.gate.wait()
        return "The PDF is ready and saved to your documents folder."


class FakeEndCall:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self) -> None:
        self.calls += 1


class FakeCall:
    def __init__(self) -> None:
        self.destinations: list[str] = []

    async def __call__(self, destination: str, *, context: object = None) -> object:
        self.destinations.append(destination)
        return object()


class FakeRoomService:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def delete_room(self, request) -> None:
        self.deleted.append(request.room)


class FakeContext:
    def __init__(self, *, room_name: str, metadata: str) -> None:
        self.room = SimpleNamespace(name=room_name)
        self.job = SimpleNamespace(metadata=metadata)
        self.api = SimpleNamespace(room=FakeRoomService())


class TypedMessage:
    def __init__(self, text: str) -> None:
        self.text_content = text


# --- fixtures and helpers ----------------------------------------------------


@pytest.fixture
def store(monkeypatch, tmp_path):
    path = tmp_path / "assistant.sqlite3"
    monkeypatch.setattr(background_tasks, "STORE_PATH", path)
    monkeypatch.setattr(conversation_ledger, "STORE_PATH", path)
    return path


def _dispatch_metadata() -> str:
    return json.dumps({"caal_outbound": True, "attempt_id": ATTEMPT_ID, "destination": APPROVED})


def _outbound_config() -> OutboundRoomConfig:
    config = OutboundRoomConfig.from_dispatch_metadata(
        _dispatch_metadata(), allowed_destinations=APPROVED
    )
    assert config is not None
    return config


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0.01)


async def _teardown(*bridges: BackgroundTaskBridge) -> None:
    for bridge in bridges:
        await bridge.close()
    await _settle()
    for bridge in bridges:
        await bridge.abandon()


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


async def _speak_turn(handler, text: str, reached_llm: list[str]) -> None:
    """A spoken turn: the final STT event lands, then the turn commits."""
    handler.on_final_transcript(text)
    await _run_turn(handler, text, reached_llm)


async def _web_session(store):
    """A web session: background work is possible, a callback is not."""
    voice_agent = _load_voice_agent()
    execute = FakeExecute()
    bridge = BackgroundTaskBridge(execute=execute, session_key=WEB_ROOM, max_concurrency=1)
    await bridge.start()
    session, end_call = FakeSession(), FakeEndCall()
    handler = voice_agent.LocalTurnHandler(
        phone_handoff=None, session=session, end_call=end_call, background=bridge
    )
    return handler, session, bridge, execute, end_call


async def _phone_leg(store, *, phone_handoff: PhoneHandoffController | None = None):
    """An answered outbound phone leg, wired exactly as the entrypoint wires it."""
    voice_agent = _load_voice_agent()
    execute = FakeExecute()
    bridge = BackgroundTaskBridge(execute=execute, session_key=PHONE_ROOM, max_concurrency=2)
    await bridge.start()
    session = FakeSession()
    ctx = FakeContext(room_name=PHONE_ROOM, metadata=_dispatch_metadata())
    config = _outbound_config()
    arm = voice_agent.build_callback_arming(
        ctx, session=session, bridge=bridge, outbound_config=lambda: config
    )
    assert arm is not None
    end_call = FakeEndCall()
    handler = voice_agent.LocalTurnHandler(
        phone_handoff=phone_handoff,
        session=session,
        end_call=end_call,
        background=bridge,
        arm_callback_and_end_call=arm,
    )
    return handler, session, bridge, execute, ctx, end_call


def _assert_opaque(spoken: list[str]) -> None:
    """Nothing spoken ever carries a task id, a destination, or the raw request."""
    for line in spoken:
        assert "bt_" not in line
        assert APPROVED not in line and APPROVED.lstrip("+") not in line
        assert "Lisbon" not in line and "electric bikes" not in line
    for task in list_tasks():
        assert all(task.task_id not in line for line in spoken)


# --- classifier ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        PDF_REQUEST,
        "Create a PDF of our conversation.",
        "can you generate a pdf report of the quarterly numbers",
        "Actually, can you create a PDF explaining what you are? It can be short.",
        "Jarvis, please prepare a PDF with the meeting notes",
        "write a report on the competitors in the smart speaker market",
        "I need a PDF summary of everything we talked about today",
        "make me a document that outlines the project plan",
        "put together a presentation about our Q3 results",
        RESEARCH_REQUEST,
        "investigate why our website traffic dropped last month",
        "Could you do some research on solar panel installers in Edmonton?",
        "compile a list of every vendor we used this year with their contact info",
        "look into flights to Lisbon in May and what they cost",
        "do a deep dive into the history of the Roman aqueducts",
    ],
)
def test_clear_long_running_work_is_inferred(text: str) -> None:
    assert long_running_work_inferred(text) is True
    assert background_task_requested(text) is False  # inferred, not explicit


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "what time is it",
        "what's the weather tomorrow",
        "set a timer for five minutes",
        "turn off the kitchen lights",
        "what is a pdf",
        "What is a PDF file?",
        "how do I create a pdf on my phone",
        "explain how to generate a pdf in python",
        "did you create the pdf",
        "is the pdf ready yet",
        "don't create a pdf, just tell me",
        "open the pdf I sent you",
        "write an email to Bob about lunch",
        "write a haiku about autumn",
        "tell me about the research on sleep",
        "what does the report say",
        "he said he would investigate it",
        "if I asked you to write a report, could you?",
        "look into it",
        "research",
        "yes",
        "no",
        "how's the background task going",
        "cancel the background task",
    ],
)
def test_ordinary_turns_are_not_inferred_as_long_work(text: str) -> None:
    assert long_running_work_inferred(text) is False


def test_inference_is_deterministic_and_bounded() -> None:
    assert all(long_running_work_inferred(PDF_REQUEST) for _ in range(5))
    assert long_running_work_inferred(("blah " * 10_000) + PDF_REQUEST) is False
    assert long_running_work_inferred(PDF_REQUEST + (" blah" * 10_000)) is True
    assert long_running_work_inferred(None) is False  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "text",
    [
        "is the pdf ready yet",
        "how's the report coming along",
        "what's the status of the research",
        "is the task done",
    ],
)
def test_natural_status_questions_about_the_work_are_status_requests(text: str) -> None:
    assert background_status_requested(text) is True
    assert long_running_work_inferred(text) is False


def test_control_replies_are_registered_as_mechanics() -> None:
    assert LONG_WORK_ACK_REPLY in BACKGROUND_CONTROL_REPLIES
    assert LONG_WORK_OFFER_CALLBACK_REPLY in BACKGROUND_CONTROL_REPLIES
    assert CALLBACK_OFFER_DECLINED_REPLY in END_CALL_CONTROL_REPLIES
    assert LONG_WORK_OFFER_CALLBACK_REPLY.endswith(
        "This will take a while. Would you like me to call you when it's done?"
    )
    assert "background" in CALLBACK_OFFER_DECLINED_REPLY.lower()


# --- the bridge alone ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_bridge_schedules_inferred_work_and_reports_the_offer(store) -> None:
    execute = FakeExecute()
    bridge = BackgroundTaskBridge(execute=execute, session_key=WEB_ROOM)
    session = FakeSession()
    await bridge.start()
    try:
        outcome = await bridge.process_turn(PDF_REQUEST, session, offer_callback=True)
        assert outcome == BackgroundTurnOutcome(
            consumed=True, scheduled=True, callback_offered=True
        )
        outcome = await bridge.process_turn(RESEARCH_REQUEST, session, offer_callback=False)
        assert outcome == BackgroundTurnOutcome(consumed=True, scheduled=True)
        # Explicit commands keep their own acknowledgement and never offer.
        outcome = await bridge.process_turn(EXPLICIT_REQUEST, session, offer_callback=True)
        assert outcome == BackgroundTurnOutcome(consumed=True, scheduled=True)
        assert await bridge.process_turn("what time is it", session) == BackgroundTurnOutcome(
            consumed=False
        )
        await _settle()
    finally:
        execute.gate.set()
        await _teardown(bridge)

    assert session.spoken == [
        LONG_WORK_OFFER_CALLBACK_REPLY,
        LONG_WORK_ACK_REPLY,
        BACKGROUND_ACK_REPLY,
    ]
    assert [t.request for t in list_tasks()] == [PDF_REQUEST, RESEARCH_REQUEST, EXPLICIT_REQUEST]
    # handle_turn keeps its boolean contract for existing callers.
    assert await bridge.handle_turn("what time is it", session) is False


# --- the end-call machine holds the offered question ------------------------------------


def test_offered_callback_question_only_arms_on_a_fresh_yes() -> None:
    now = [100.0]
    machine = EndCallIntentMachine(clock=lambda: now[0])

    machine.offer_callback()
    assert machine.awaiting_confirmation is True
    decision = machine.observe("yes please", callback_available=True)
    assert decision.action is EndCallAction.ARM_CALLBACK and decision.consumed is True
    assert machine.awaiting_confirmation is False

    machine.offer_callback()
    decision = machine.observe("no thanks", callback_available=True)
    assert decision.action is EndCallAction.CANCEL and decision.consumed is True
    assert decision.reply == CALLBACK_OFFER_DECLINED_REPLY

    machine.offer_callback()
    decision = machine.observe("what time is it", callback_available=True)
    assert decision.action is EndCallAction.CANCEL and decision.consumed is False
    assert decision.reply is None
    # The question is gone; a "yes" a turn later is ordinary conversation.
    assert machine.observe("yes", callback_available=True).action is EndCallAction.NONE

    machine.offer_callback()
    assert machine.observe("   ", callback_available=True).action is EndCallAction.CANCEL
    assert machine.observe("yes", callback_available=True).action is EndCallAction.NONE

    machine.offer_callback()
    now[0] += EndCallIntentMachine.CONFIRMATION_TIMEOUT_SECONDS + 1
    assert machine.awaiting_confirmation is False
    assert machine.observe("yes", callback_available=True).action is EndCallAction.NONE


# --- the reported failure, on the web session ----------------------------------------------


@pytest.mark.asyncio
async def test_pdf_request_is_scheduled_and_acknowledged_before_hermes(store) -> None:
    handler, session, bridge, execute, end_call = await _web_session(store)
    reached_llm: list[str] = []
    try:
        await _run_turn(handler, PDF_REQUEST, reached_llm)
        await _settle()
        assert reached_llm == []
        # No phone to call back on a web session: acknowledge without the offer.
        assert session.spoken == [LONG_WORK_ACK_REPLY]
        (task,) = list_tasks(session_key=WEB_ROOM)
        assert task.status == RUNNING and task.request == PDF_REQUEST
        assert execute.calls[0][0] == PDF_REQUEST

        # Status after automatic scheduling reports the work, exactly as an
        # explicit background task would, in both the literal and natural forms.
        await _run_turn(handler, "how's the background task going", reached_llm)
        await _run_turn(handler, "is the PDF ready yet?", reached_llm)
        assert session.spoken[1:] == [
            BACKGROUND_STATUS_WORKING_REPLY,
            BACKGROUND_STATUS_WORKING_REPLY,
        ]

        # A stray "yes" with nothing pending is ordinary conversation.
        await _run_turn(handler, "yes", reached_llm)
        await _run_turn(handler, "what time is it", reached_llm)
        assert reached_llm == ["yes", "what time is it"]
        assert end_call.calls == 0
        assert callback_armed(task.task_id) is False
        _assert_opaque(session.spoken)
    finally:
        execute.gate.set()
        await _teardown(bridge)


@pytest.mark.asyncio
async def test_normal_prompts_are_never_hijacked(store) -> None:
    handler, session, bridge, execute, _ = await _web_session(store)
    reached_llm: list[str] = []
    prompts = [
        "what time is it",
        "what is a pdf",
        "write an email to Bob about lunch",
        "how do I create a pdf on my phone",
        "turn off the kitchen lights",
    ]
    try:
        for prompt in prompts:
            await _run_turn(handler, prompt, reached_llm)
    finally:
        execute.gate.set()
        await _teardown(bridge)

    assert reached_llm == prompts
    assert session.spoken == []
    assert list_tasks() == []


@pytest.mark.asyncio
async def test_explicit_background_commands_are_unchanged(store) -> None:
    handler, session, bridge, execute, _ = await _web_session(store)
    reached_llm: list[str] = []
    try:
        await _run_turn(handler, EXPLICIT_REQUEST, reached_llm)
        await _settle()
    finally:
        execute.gate.set()
        await _teardown(bridge)
    assert reached_llm == []
    assert session.spoken == [BACKGROUND_ACK_REPLY]
    assert len(list_tasks()) == 1


# --- on the phone: the callback offer -----------------------------------------------------


@pytest.mark.asyncio
async def test_yes_arms_the_protected_callback_and_ends_the_call(store) -> None:
    handler, session, bridge, execute, ctx, end_call = await _phone_leg(store)
    reached_llm: list[str] = []
    try:
        await _speak_turn(handler, PDF_REQUEST, reached_llm)
        await _settle()
        assert session.spoken == [LONG_WORK_OFFER_CALLBACK_REPLY]
        (task,) = list_tasks(statuses=[RUNNING])
        assert callback_armed(task.task_id) is False
        assert ctx.api.room.deleted == []

        await _speak_turn(handler, "yes", reached_llm)
        await _settle()

        assert reached_llm == []
        assert session.spoken == [LONG_WORK_OFFER_CALLBACK_REPLY, CALLBACK_ARMED_REPLY]
        assert callback_armed(task.task_id) is True
        assert ctx.api.room.deleted == [PHONE_ROOM]
        assert end_call.calls == 0  # the callback path ends the room itself, once
        assert list_tasks(statuses=[RUNNING])[0].task_id == task.task_id
        _assert_opaque(session.spoken)
    finally:
        execute.gate.set()
        await _teardown(bridge)


@pytest.mark.asyncio
async def test_no_keeps_the_task_running_and_the_conversation_open(store) -> None:
    handler, session, bridge, execute, ctx, end_call = await _phone_leg(store)
    reached_llm: list[str] = []
    try:
        await _speak_turn(handler, RESEARCH_REQUEST, reached_llm)
        await _settle()
        await _speak_turn(handler, "no thanks", reached_llm)
        await _settle()
        assert reached_llm == []
        assert session.spoken == [LONG_WORK_OFFER_CALLBACK_REPLY, CALLBACK_OFFER_DECLINED_REPLY]
        (task,) = list_tasks(statuses=[RUNNING])
        assert callback_armed(task.task_id) is False
        assert ctx.api.room.deleted == [] and end_call.calls == 0

        # Normal conversation continues, and a late "yes" no longer means anything.
        await _speak_turn(handler, "what time is it", reached_llm)
        await _speak_turn(handler, "yes", reached_llm)
        assert reached_llm == ["what time is it", "yes"]
        assert callback_armed(task.task_id) is False
        assert ctx.api.room.deleted == []
        _assert_opaque(session.spoken)
    finally:
        execute.gate.set()
        await _teardown(bridge)


@pytest.mark.asyncio
async def test_unrelated_answer_fails_closed_and_proceeds_normally(store) -> None:
    handler, session, bridge, execute, ctx, end_call = await _phone_leg(store)
    reached_llm: list[str] = []
    try:
        await _speak_turn(handler, PDF_REQUEST, reached_llm)
        await _settle()
        await _speak_turn(handler, "what's the weather like tomorrow", reached_llm)
        await _speak_turn(handler, "yes", reached_llm)
        await _settle()
        assert reached_llm == ["what's the weather like tomorrow", "yes"]
        assert session.spoken == [LONG_WORK_OFFER_CALLBACK_REPLY]
        (task,) = list_tasks(statuses=[RUNNING])
        assert callback_armed(task.task_id) is False
        assert ctx.api.room.deleted == [] and end_call.calls == 0
    finally:
        execute.gate.set()
        await _teardown(bridge)


@pytest.mark.asyncio
async def test_blank_answer_fails_closed(store) -> None:
    handler, session, bridge, execute, ctx, _ = await _phone_leg(store)
    reached_llm: list[str] = []
    try:
        await _speak_turn(handler, PDF_REQUEST, reached_llm)
        await _settle()
        await _speak_turn(handler, "   ", reached_llm)
        await _speak_turn(handler, "yes", reached_llm)
        await _settle()
        assert "yes" in reached_llm
        (task,) = list_tasks(statuses=[RUNNING])
        assert callback_armed(task.task_id) is False
        assert ctx.api.room.deleted == []
        assert session.spoken == [LONG_WORK_OFFER_CALLBACK_REPLY]
    finally:
        execute.gate.set()
        await _teardown(bridge)


@pytest.mark.asyncio
async def test_timed_out_offer_fails_closed(store, monkeypatch) -> None:
    monkeypatch.setattr(EndCallIntentMachine, "CONFIRMATION_TIMEOUT_SECONDS", 0.0)
    handler, session, bridge, execute, ctx, _ = await _phone_leg(store)
    reached_llm: list[str] = []
    try:
        await _speak_turn(handler, PDF_REQUEST, reached_llm)
        await _settle()
        await asyncio.sleep(0.02)
        await _speak_turn(handler, "yes", reached_llm)
        await _settle()
        assert reached_llm == ["yes"]
        (task,) = list_tasks(statuses=[RUNNING])
        assert callback_armed(task.task_id) is False
        assert ctx.api.room.deleted == []
        assert session.spoken == [LONG_WORK_OFFER_CALLBACK_REPLY]
    finally:
        execute.gate.set()
        await _teardown(bridge)


@pytest.mark.asyncio
async def test_typed_and_spoken_turns_behave_identically_on_the_phone(store) -> None:
    handler, session, bridge, execute, ctx, _ = await _phone_leg(store)
    reached_llm: list[str] = []
    try:
        await _run_turn(handler, PDF_REQUEST, reached_llm)
        await _settle()
        await _run_turn(handler, "yes", reached_llm)
        await _settle()
        assert reached_llm == []
        assert session.spoken == [LONG_WORK_OFFER_CALLBACK_REPLY, CALLBACK_ARMED_REPLY]
        (task,) = list_tasks(statuses=[RUNNING])
        assert callback_armed(task.task_id) is True
        assert ctx.api.room.deleted == [PHONE_ROOM]
    finally:
        execute.gate.set()
        await _teardown(bridge)


@pytest.mark.asyncio
async def test_long_work_with_call_me_when_done_asks_once_and_schedules(store) -> None:
    """ "Research X and call me when you're done" used to be told nothing was running."""
    handler, session, bridge, execute, ctx, _ = await _phone_leg(store)
    reached_llm: list[str] = []
    try:
        await _speak_turn(
            handler, "research flights to Lisbon in May and call me when you're done", reached_llm
        )
        await _settle()
        assert reached_llm == []
        assert session.spoken == [LONG_WORK_OFFER_CALLBACK_REPLY]
        assert NO_TASK_REPLY not in session.spoken and ASK_CALLBACK_REPLY not in session.spoken
        (task,) = list_tasks(statuses=[RUNNING])
        await _speak_turn(handler, "yes", reached_llm)
        await _settle()
        assert callback_armed(task.task_id) is True
        assert ctx.api.room.deleted == [PHONE_ROOM]
    finally:
        execute.gate.set()
        await _teardown(bridge)


@pytest.mark.asyncio
async def test_direct_hang_up_still_outranks_long_work_and_the_offer(store) -> None:
    handler, session, bridge, execute, ctx, end_call = await _phone_leg(store)
    reached_llm: list[str] = []
    try:
        await _speak_turn(handler, PDF_REQUEST, reached_llm)
        await _settle()
        await _speak_turn(handler, "hang up", reached_llm)
        await asyncio.sleep(0)
        assert end_call.calls == 1
        (task,) = list_tasks(statuses=[RUNNING])
        assert callback_armed(task.task_id) is False
        assert session.spoken == [LONG_WORK_OFFER_CALLBACK_REPLY]
        assert reached_llm == []
    finally:
        execute.gate.set()
        await _teardown(bridge)


@pytest.mark.asyncio
async def test_pending_handoff_confirmation_keeps_its_priority(store) -> None:
    """A "yes" answers whichever question is actually pending, never a stale one."""
    call = FakeCall()
    controller = PhoneHandoffController(start_call=call, allowed_destinations=APPROVED)
    handler, session, bridge, execute, ctx, _ = await _phone_leg(store, phone_handoff=controller)
    reached_llm: list[str] = []
    try:
        await _speak_turn(handler, "move the conversation to my phone", reached_llm)
        assert session.spoken == [ASK_CONFIRMATION_REPLY]
        # An unrelated long-work request withdraws the handoff question, is
        # scheduled, and asks the callback question instead.
        await _speak_turn(handler, PDF_REQUEST, reached_llm)
        await _settle()
        assert session.spoken == [ASK_CONFIRMATION_REPLY, LONG_WORK_OFFER_CALLBACK_REPLY]
        await _speak_turn(handler, "yes", reached_llm)
        await _settle()
        (task,) = list_tasks(statuses=[RUNNING])
        assert call.destinations == []  # the stale handoff question never dials
        assert callback_armed(task.task_id) is True
        assert ctx.api.room.deleted == [PHONE_ROOM]
        assert reached_llm == []
    finally:
        execute.gate.set()
        await _teardown(bridge)


# --- the voice STT race --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_late_final_transcript_schedules_once_and_offers_once(store) -> None:
    """The VAD turn commits before the slow STT final; the request still wins, once."""
    handler, session, bridge, execute, ctx, _ = await _phone_leg(store)
    reached_llm: list[str] = []
    try:
        handler.on_speech_transcript_started()
        committed_turn = asyncio.create_task(_run_turn(handler, "", reached_llm))
        await asyncio.sleep(0)
        handler.on_final_transcript(PDF_REQUEST)
        await committed_turn
        await _settle()
        assert reached_llm == []
        assert session.spoken == [LONG_WORK_OFFER_CALLBACK_REPLY]
        assert len(list_tasks()) == 1

        # The answer arrives the same way and arms exactly once.
        handler.on_speech_transcript_started()
        committed_turn = asyncio.create_task(_run_turn(handler, "", reached_llm))
        await asyncio.sleep(0)
        handler.on_final_transcript("yes")
        await committed_turn
        await _settle()
        assert reached_llm == []
        assert session.spoken == [LONG_WORK_OFFER_CALLBACK_REPLY, CALLBACK_ARMED_REPLY]
        (task,) = list_tasks(statuses=[RUNNING])
        assert callback_armed(task.task_id) is True
        assert ctx.api.room.deleted == [PHONE_ROOM]
    finally:
        execute.gate.set()
        await _teardown(bridge)


@pytest.mark.asyncio
async def test_stt_timeout_without_a_transcript_schedules_nothing(store) -> None:
    handler, session, bridge, execute, _, _ = await _phone_leg(store)
    monkeypatch_grace = 0.05
    handler._FINAL_STT_GRACE_SECONDS = monkeypatch_grace
    reached_llm: list[str] = []
    try:
        handler.on_speech_transcript_started()
        await _run_turn(handler, "", reached_llm)
        assert reached_llm == [""]
        assert session.spoken == []
        assert list_tasks() == []
    finally:
        execute.gate.set()
        await _teardown(bridge)
