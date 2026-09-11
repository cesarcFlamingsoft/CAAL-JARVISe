"""Coverage for background tasks reaching JARVIS through the voice agent.

A recognised "do this in the background" turn is answered locally with a
fixed acknowledgement and scheduled on the durable queue; Hermes never sees
that turn. The task runs through an injected worker (in production the
configured LLM provider) with a bounded, private snapshot of the recent
conversation. Its outcome is spoken once to the session that asked, or sent
through the Telegram fallback when that session is gone. Status and cancel
phrases are exact, and hang-up and phone-handoff keep their priority.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from caal import background_task_session, background_tasks
from caal import settings as settings_module
from caal.background_task_session import (
    BACKGROUND_ACK_REPLY,
    BACKGROUND_BUSY_REPLY,
    BACKGROUND_CANCELLED_REPLY,
    BACKGROUND_CONTROL_REPLIES,
    BACKGROUND_NOTHING_TO_CANCEL_REPLY,
    BACKGROUND_STATUS_IDLE_REPLY,
    BACKGROUND_STATUS_WORKING_REPLY,
    MAX_FALLBACK_CHARS,
    MAX_SPOKEN_RESULT_CHARS,
    MAX_TASK_CONTEXT_CHARS,
    BackgroundTaskBridge,
    LLMBackgroundWorker,
    capture_task_context,
    fallback_notification,
    spoken_notification,
)
from caal.background_tasks import (
    CANCELLED,
    FAILED,
    QUEUED,
    RUNNING,
    SUCCEEDED,
    background_cancel_requested,
    background_status_requested,
    background_task_requested,
    get_task,
    list_tasks,
    pending_notifications,
)
from caal.handoff_intent import (
    ASK_CONFIRMATION_REPLY,
    CANCELLED_REPLY,
    PhoneHandoffController,
)
from caal.llm.providers.base import LLMResponse
from caal.telegram_notify import TelegramCallNotifier

SESSION_KEY = "room-web-1"
BACKGROUND_REQUEST = (
    "research flight prices to Lisbon in the background and let me know when it's done"
)
SECRET = "sk-live-ZZZ0SUPERSECRET0ZZZ"

_voice_agent = None


def _load_voice_agent():
    """Load voice_agent.py once; it is a script, not an importable package."""
    global _voice_agent
    if _voice_agent is None:
        module_path = Path(__file__).parents[1] / "voice_agent.py"
        spec = importlib.util.spec_from_file_location("voice_agent_background_test", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _voice_agent = module
    return _voice_agent


class FakeSession:
    """Records everything the agent would speak and exposes a chat history."""

    def __init__(self, history: list[object] | None = None) -> None:
        self.spoken: list[str] = []
        self.history = SimpleNamespace(items=list(history or []))

    async def say(self, text: str) -> None:
        self.spoken.append(text)


class FakeExecute:
    """Injected worker adapter: records what it was asked and answers on demand."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.gate = asyncio.Event()
        self.gate.set()
        self.result = "Cheapest fare is 180 euros on Tuesday."
        self.error: Exception | None = None

    async def __call__(self, request: str, context: str) -> str:
        self.calls.append((request, context))
        await self.gate.wait()
        if self.error is not None:
            raise self.error
        return self.result


class FakeFallback:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def __call__(self, text: str) -> None:
        self.texts.append(text)


class FakeCall:
    def __init__(self) -> None:
        self.destinations: list[str] = []

    async def __call__(self, destination: str, *, context: object = None) -> object:
        self.destinations.append(destination)
        return object()


class FakeEndCall:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self) -> None:
        self.calls += 1


class TypedMessage:
    def __init__(self, text: str) -> None:
        self.text_content = text


def _message(role: str, content: str) -> SimpleNamespace:
    return SimpleNamespace(type="message", role=role, content=content)


@pytest.fixture
def store(monkeypatch, tmp_path):
    path = tmp_path / "assistant.sqlite3"
    monkeypatch.setattr(background_tasks, "STORE_PATH", path)
    return path


@pytest.fixture
def bridge_parts(store):
    execute = FakeExecute()
    fallback = FakeFallback()
    bridge = BackgroundTaskBridge(
        execute=execute,
        session_key=SESSION_KEY,
        fallback=fallback,
        max_concurrency=1,
    )
    return bridge, execute, fallback


def _build_handler(session, bridge, *, phone_handoff=None, end_call=None):
    voice_agent = _load_voice_agent()
    return voice_agent.LocalTurnHandler(
        phone_handoff=phone_handoff,
        session=session,
        end_call=end_call or FakeEndCall(),
        background=bridge,
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


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0.01)


async def _teardown(bridge: BackgroundTaskBridge) -> None:
    """End the session, let quick work land, then stop the runner as a dying process would."""
    await bridge.close()
    await _settle()
    await bridge.abandon()


# ---------------------------------------------------------------------------
# Scheduling through the local turn path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_typed_background_request_is_acknowledged_and_scheduled(bridge_parts) -> None:
    bridge, execute, _ = bridge_parts
    session = FakeSession(
        [
            _message("user", "I'm flying to Lisbon in May"),
            _message("assistant", "Nice, want me to check fares?"),
        ]
    )
    handler = _build_handler(session, bridge)
    reached_llm: list[str] = []
    await bridge.start(context_source=lambda: capture_task_context(session))
    try:
        await _run_turn(handler, BACKGROUND_REQUEST, reached_llm)
        await _settle()
    finally:
        await _teardown(bridge)

    assert reached_llm == []
    assert session.spoken[0] == BACKGROUND_ACK_REPLY
    assert BACKGROUND_ACK_REPLY in BACKGROUND_CONTROL_REPLIES
    tasks = list_tasks(session_key=SESSION_KEY)
    assert len(tasks) == 1 and tasks[0].status == SUCCEEDED
    assert execute.calls[0][0] == BACKGROUND_REQUEST
    assert "Lisbon in May" in execute.calls[0][1]
    assert "private" in execute.calls[0][1].lower()
    assert len(execute.calls[0][1]) <= MAX_TASK_CONTEXT_CHARS


@pytest.mark.asyncio
async def test_spoken_background_request_is_answered_once(bridge_parts) -> None:
    bridge, execute, _ = bridge_parts
    session = FakeSession()
    handler = _build_handler(session, bridge)
    reached_llm: list[str] = []
    await bridge.start()
    try:
        handler.on_final_transcript(BACKGROUND_REQUEST)
        await _run_turn(handler, BACKGROUND_REQUEST, reached_llm)
        await _settle()
    finally:
        await _teardown(bridge)

    assert reached_llm == []
    assert session.spoken.count(BACKGROUND_ACK_REPLY) == 1
    assert len(list_tasks(session_key=SESSION_KEY)) == 1


@pytest.mark.asyncio
async def test_ordinary_turns_reach_the_llm_while_a_task_runs(bridge_parts) -> None:
    bridge, execute, _ = bridge_parts
    execute.gate.clear()
    session = FakeSession()
    handler = _build_handler(session, bridge)
    reached_llm: list[str] = []
    await bridge.start()
    try:
        await _run_turn(handler, BACKGROUND_REQUEST, reached_llm)
        await _settle()
        assert get_task(list_tasks()[0].task_id).status == RUNNING
        await _run_turn(handler, "what time is it", reached_llm)
        await _run_turn(handler, "and what's the weather", reached_llm)
    finally:
        execute.gate.set()
        await _teardown(bridge)

    assert reached_llm == ["what time is it", "and what's the weather"]
    assert session.spoken == [BACKGROUND_ACK_REPLY]


@pytest.mark.asyncio
async def test_without_a_bridge_background_requests_still_reach_the_llm() -> None:
    session = FakeSession()
    handler = _build_handler(session, None)
    reached_llm: list[str] = []
    await _run_turn(handler, BACKGROUND_REQUEST, reached_llm)
    assert reached_llm == [BACKGROUND_REQUEST]
    assert session.spoken == []


@pytest.mark.asyncio
async def test_hang_up_outranks_background_and_handoff(bridge_parts) -> None:
    bridge, execute, _ = bridge_parts
    session, call, end_call = FakeSession(), FakeCall(), FakeEndCall()
    controller = PhoneHandoffController(start_call=call, allowed_destinations="+17805558345")
    handler = _build_handler(session, bridge, phone_handoff=controller, end_call=end_call)
    reached_llm: list[str] = []
    await bridge.start()
    try:
        await _run_turn(handler, BACKGROUND_REQUEST, reached_llm)
        await _run_turn(handler, "hang up", reached_llm)
        await asyncio.sleep(0)
    finally:
        await _teardown(bridge)

    assert end_call.calls == 1
    assert call.destinations == []
    assert session.spoken == [BACKGROUND_ACK_REPLY]
    assert len(list_tasks()) == 1


@pytest.mark.asyncio
async def test_pending_handoff_confirmation_outranks_a_background_request(bridge_parts) -> None:
    bridge, execute, _ = bridge_parts
    session, call = FakeSession(), FakeCall()
    controller = PhoneHandoffController(start_call=call, allowed_destinations="+17805558345")
    handler = _build_handler(session, bridge, phone_handoff=controller)
    reached_llm: list[str] = []
    await bridge.start()
    try:
        await _run_turn(handler, "move the conversation to my phone", reached_llm)
        # The pending confirmation sees this first and withdraws; only then may
        # the background bridge take the turn. A later "yes" must not dial.
        await _run_turn(handler, "no, just look into it in the background", reached_llm)
        await _run_turn(handler, "yes", reached_llm)
        await _run_turn(handler, "move the conversation to my phone", reached_llm)
        await _run_turn(handler, "no", reached_llm)
        await _settle()
    finally:
        await _teardown(bridge)

    assert session.spoken == [ASK_CONFIRMATION_REPLY, BACKGROUND_ACK_REPLY] + [
        ASK_CONFIRMATION_REPLY,
        CANCELLED_REPLY,
    ]
    assert call.destinations == []
    assert len(list_tasks()) == 1
    assert reached_llm == ["yes"]


@pytest.mark.asyncio
async def test_full_queue_is_declined_locally(bridge_parts, monkeypatch) -> None:
    bridge, execute, _ = bridge_parts
    monkeypatch.setattr(background_tasks, "MAX_QUEUE_DEPTH", 1)
    execute.gate.clear()
    session = FakeSession()
    handler = _build_handler(session, bridge)
    reached_llm: list[str] = []
    await bridge.start()
    try:
        await _run_turn(handler, BACKGROUND_REQUEST, reached_llm)
        await _run_turn(handler, "also check hotels in the background", reached_llm)
    finally:
        execute.gate.set()
        await _teardown(bridge)

    assert session.spoken == [BACKGROUND_ACK_REPLY, BACKGROUND_BUSY_REPLY]
    assert reached_llm == []


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_result_is_spoken_once_to_the_original_session(bridge_parts) -> None:
    bridge, execute, fallback = bridge_parts
    execute.result = f"Cheapest fare is 180 euros. Booking token={SECRET}"
    session = FakeSession()
    handler = _build_handler(session, bridge)
    reached_llm: list[str] = []
    await bridge.start()
    try:
        await _run_turn(handler, BACKGROUND_REQUEST + f" my password: {SECRET}", reached_llm)
        await _settle()
        assert await bridge.deliver_pending(session) == 1
        assert await bridge.deliver_pending(session) == 0
    finally:
        await _teardown(bridge)

    task = list_tasks()[0]
    assert session.spoken == [BACKGROUND_ACK_REPLY, spoken_notification(task)]
    announcement = session.spoken[1]
    assert "180 euros" in announcement
    assert task.task_id not in announcement
    assert "Lisbon" not in announcement  # raw request text stays private
    assert SECRET not in announcement
    assert SECRET not in execute.calls[0][0]
    assert fallback.texts == []
    assert pending_notifications() == []


@pytest.mark.asyncio
async def test_failure_is_reported_concisely_without_internals(bridge_parts) -> None:
    bridge, execute, _ = bridge_parts
    execute.error = RuntimeError(f"upstream 502 at https://hermes.internal token={SECRET}")
    session = FakeSession()
    handler = _build_handler(session, bridge)
    await bridge.start()
    try:
        await _run_turn(handler, BACKGROUND_REQUEST, [])
        await _settle()
        assert await bridge.deliver_pending(session) == 1
    finally:
        await _teardown(bridge)

    announcement = session.spoken[1]
    assert "couldn't finish" in announcement.lower()
    assert "hermes.internal" not in announcement
    assert SECRET not in announcement
    assert "RuntimeError" not in announcement


def test_spoken_and_fallback_notifications_are_bounded_and_opaque(store) -> None:
    task = background_tasks.enqueue("x" * 500, session_key=SESSION_KEY)
    background_tasks._mark_running(task.task_id)
    background_tasks._finish(task.task_id, SUCCEEDED, result="r" * 5000)
    finished = get_task(task.task_id)
    spoken = spoken_notification(finished)
    sent = fallback_notification(finished)
    assert len(spoken) <= MAX_SPOKEN_RESULT_CHARS + 120
    assert len(sent) <= MAX_FALLBACK_CHARS
    assert task.task_id not in spoken and task.task_id not in sent
    assert "xxxxx" not in spoken and "xxxxx" not in sent

    interrupted = background_tasks.enqueue("y" * 50, session_key=SESSION_KEY)
    background_tasks._mark_running(interrupted.task_id)
    background_tasks.recover_interrupted()
    assert "interrupted" in spoken_notification(get_task(interrupted.task_id)).lower()

    cancelled = background_tasks.enqueue("z" * 50, session_key=SESSION_KEY)
    background_tasks.cancel(cancelled.task_id)
    assert spoken_notification(get_task(cancelled.task_id)) == ""


@pytest.mark.asyncio
async def test_session_close_keeps_a_held_job_running_and_delivers_once(bridge_parts) -> None:
    """Ending the session must neither block on nor interrupt background work."""
    bridge, execute, fallback = bridge_parts
    execute.gate.clear()
    session = FakeSession()
    handler = _build_handler(session, bridge)
    await bridge.start()
    await _run_turn(handler, BACKGROUND_REQUEST, [])
    await _settle()
    task_id = list_tasks()[0].task_id
    assert get_task(task_id).status == RUNNING

    started = time.monotonic()
    await bridge.close()
    assert time.monotonic() - started < 0.5
    assert get_task(task_id).status == RUNNING
    assert fallback.texts == []
    # The closed session can no longer schedule or answer background commands.
    assert await bridge.handle_turn("how's the background task going", session) is False

    execute.gate.set()
    await _settle()

    assert get_task(task_id).status == SUCCEEDED
    assert len(fallback.texts) == 1
    assert "180 euros" in fallback.texts[0]
    assert task_id not in fallback.texts[0]
    assert "Lisbon" not in fallback.texts[0]
    assert pending_notifications() == []
    # Exactly once: a later live delivery or flush finds nothing left.
    assert await bridge.deliver_pending(session) == 0
    assert session.spoken == [BACKGROUND_ACK_REPLY]
    await bridge.abandon()


@pytest.mark.asyncio
async def test_failure_after_session_close_goes_to_fallback_once(bridge_parts) -> None:
    bridge, execute, fallback = bridge_parts
    execute.gate.clear()
    session = FakeSession()
    handler = _build_handler(session, bridge)
    await bridge.start()
    await _run_turn(handler, BACKGROUND_REQUEST, [])
    await _settle()
    await bridge.close()

    execute.error = RuntimeError(f"upstream 502 token={SECRET}")
    execute.gate.set()
    await _settle()

    assert len(fallback.texts) == 1
    assert "couldn't finish" in fallback.texts[0].lower()
    assert SECRET not in fallback.texts[0]
    assert pending_notifications() == []
    await bridge.abandon()


@pytest.mark.asyncio
async def test_queued_work_still_starts_after_session_close(bridge_parts) -> None:
    bridge, execute, fallback = bridge_parts
    execute.gate.clear()
    session = FakeSession()
    handler = _build_handler(session, bridge)
    await bridge.start(context_source=lambda: capture_task_context(session))
    await _run_turn(handler, BACKGROUND_REQUEST, [])
    await _run_turn(handler, "also check hotels in the background", [])
    await _settle()
    await bridge.close()
    assert sorted(t.status for t in list_tasks()) == sorted([RUNNING, QUEUED])

    execute.gate.set()
    await _settle()

    assert [t.status for t in list_tasks()] == [SUCCEEDED, SUCCEEDED]
    assert len(execute.calls) == 2
    assert len(fallback.texts) == 2
    await bridge.abandon()


@pytest.mark.asyncio
async def test_outcome_after_close_without_fallback_waits_for_the_room(store) -> None:
    execute = FakeExecute()
    execute.gate.clear()
    bridge = BackgroundTaskBridge(execute=execute, session_key=SESSION_KEY, fallback=None)
    session = FakeSession()
    handler = _build_handler(session, bridge)
    await bridge.start()
    await _run_turn(handler, BACKGROUND_REQUEST, [])
    await _settle()
    await bridge.close()
    execute.gate.set()
    await _settle()
    await bridge.abandon()

    assert [t.status for t in list_tasks()] == [SUCCEEDED]
    assert len(pending_notifications(session_key=SESSION_KEY)) == 1

    reconnected = BackgroundTaskBridge(execute=FakeExecute(), session_key=SESSION_KEY)
    later_session = FakeSession()
    await reconnected.start()
    try:
        assert await reconnected.deliver_pending(later_session) == 1
        assert "180 euros" in later_session.spoken[0]
    finally:
        await _teardown(reconnected)


@pytest.mark.asyncio
async def test_abandon_at_process_teardown_returns_work_to_the_durable_queue(
    bridge_parts,
) -> None:
    """Teardown hands work back; the durable work service picks it up from there."""
    bridge, execute, fallback = bridge_parts
    execute.gate.clear()
    session = FakeSession()
    handler = _build_handler(session, bridge)
    await bridge.start()
    await _run_turn(handler, BACKGROUND_REQUEST, [])
    await _settle()
    await bridge.close()
    await bridge.abandon()

    task = list_tasks()[0]
    assert task.status == QUEUED
    assert task.finished_at is None
    # Nothing was announced, because nothing finished.
    assert fallback.texts == []
    assert background_tasks.pending_notifications() == []


@pytest.mark.asyncio
async def test_no_task_id_or_request_text_ever_reaches_the_logs(bridge_parts, caplog) -> None:
    bridge, execute, fallback = bridge_parts
    captured: list[logging.Logger] = [
        logging.getLogger("caal.background_tasks"),
        logging.getLogger("caal.background_task_session"),
    ]
    for target in captured:
        target.setLevel(logging.DEBUG)
        target.addHandler(caplog.handler)
    session = FakeSession()
    handler = _build_handler(session, bridge)
    try:
        await bridge.start()
        await _run_turn(handler, BACKGROUND_REQUEST, [])
        await _settle()
        await bridge.deliver_pending(session)
        execute.gate.clear()
        await _run_turn(handler, "look into hotels in the background too", [])
        await _settle()
        await _run_turn(handler, "cancel the background task", [])
        execute.error = RuntimeError("boom")
        execute.gate.set()
        await _run_turn(handler, "check trains in the background as well", [])
        await _settle()
        await bridge.close()
        await bridge.abandon()
    finally:
        for target in captured:
            target.removeHandler(caplog.handler)

    text = caplog.text
    assert text  # the capture is live
    assert not re.search(r"bt_[0-9a-f]{16}", text)
    for task in list_tasks():
        assert task.task_id not in text
    assert "Lisbon" not in text and "hotels" not in text and "trains" not in text


@pytest.mark.asyncio
async def test_stale_notifications_from_other_sessions_go_to_fallback(bridge_parts) -> None:
    bridge, execute, fallback = bridge_parts
    other = background_tasks.enqueue("earlier request", session_key="room-old")
    background_tasks._mark_running(other.task_id)
    background_tasks._finish(other.task_id, SUCCEEDED, result="the old answer")
    mine = background_tasks.enqueue("my request", session_key=SESSION_KEY)
    background_tasks._mark_running(mine.task_id)
    background_tasks._finish(mine.task_id, SUCCEEDED, result="my answer")

    await bridge.start()
    try:
        # Fresh outcomes are left for their own session's loop; only stale ones move.
        assert await bridge.flush_stale_to_fallback() == 0
        assert await bridge.flush_stale_to_fallback(min_age_seconds=0) == 1
        assert fallback.texts == [fallback_notification(get_task(other.task_id))]
        assert "the old answer" in fallback.texts[0]
        session = FakeSession()
        assert await bridge.deliver_pending(session) == 1
        assert "my answer" in session.spoken[0]
    finally:
        await _teardown(bridge)


@pytest.mark.asyncio
async def test_fallback_failure_does_not_lose_the_claim_forever(store) -> None:
    async def broken(text: str) -> None:
        raise ConnectionError("telegram down")

    bridge = BackgroundTaskBridge(execute=FakeExecute(), session_key=SESSION_KEY, fallback=broken)
    task = background_tasks.enqueue("earlier request", session_key="room-old")
    background_tasks._mark_running(task.task_id)
    background_tasks._finish(task.task_id, SUCCEEDED, result="answer")
    await bridge.start()
    try:
        assert await bridge.flush_stale_to_fallback(min_age_seconds=0) == 0
    finally:
        await _teardown(bridge)
    # Still claimed exactly once: a broken channel must not spam retries forever,
    # but the task remains inspectable.
    assert get_task(task.task_id).notified_at is not None


# ---------------------------------------------------------------------------
# Status and cancel phrases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "cancel the background task",
        "Jarvis, cancel that background task.",
        "stop the background job",
        "never mind the background task",
        "cancel my background tasks",
    ],
)
def test_exact_cancel_phrases(text: str) -> None:
    assert background_cancel_requested(text) is True
    assert background_task_requested(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "how's the background task going",
        "What's the status of my background task?",
        "is the background task done yet",
        "any update on the background job",
        "background task status",
    ],
)
def test_exact_status_phrases(text: str) -> None:
    assert background_status_requested(text) is True
    assert background_task_requested(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "cancel my 3pm meeting",
        "stop the music",
        "what's the status of my order",
        "is dinner done yet",
        "cancel the background task and also book the flight",
        "never mind",
        "how's it going",
    ],
)
def test_status_and_cancel_have_no_false_positives(text: str) -> None:
    assert background_cancel_requested(text) is False
    assert background_status_requested(text) is False


@pytest.mark.asyncio
async def test_status_and_cancel_are_answered_locally(bridge_parts) -> None:
    bridge, execute, _ = bridge_parts
    execute.gate.clear()
    session = FakeSession()
    handler = _build_handler(session, bridge)
    reached_llm: list[str] = []
    await bridge.start()
    try:
        await _run_turn(handler, "how's the background task going", reached_llm)
        await _run_turn(handler, "cancel the background task", reached_llm)
        await _run_turn(handler, BACKGROUND_REQUEST, reached_llm)
        await _settle()
        await _run_turn(handler, "how's the background task going", reached_llm)
        await _run_turn(handler, "cancel my 3pm meeting", reached_llm)
        await _run_turn(handler, "cancel the background task", reached_llm)
        await _settle()
        assert get_task(list_tasks()[0].task_id).status == CANCELLED
        await _run_turn(handler, "cancel the background task", reached_llm)
        # A cancelled task is settled by its acknowledgement, never announced again.
        assert await bridge.deliver_pending(session) == 0
    finally:
        execute.gate.set()
        await _teardown(bridge)

    assert session.spoken == [
        BACKGROUND_STATUS_IDLE_REPLY,
        BACKGROUND_NOTHING_TO_CANCEL_REPLY,
        BACKGROUND_ACK_REPLY,
        BACKGROUND_STATUS_WORKING_REPLY,
        BACKGROUND_CANCELLED_REPLY,
        BACKGROUND_NOTHING_TO_CANCEL_REPLY,
    ]
    assert reached_llm == ["cancel my 3pm meeting"]
    assert pending_notifications() == []


# ---------------------------------------------------------------------------
# Worker adapter, context capture, Telegram, settings
# ---------------------------------------------------------------------------


class FakeProvider:
    def __init__(self, *, content: str | None = "done", delay: float = 0.0) -> None:
        self.content = content
        self.delay = delay
        self.messages: list[list[dict]] = []

    async def chat(self, messages, tools=None, **_):
        self.messages.append(messages)
        await asyncio.sleep(self.delay)
        return LLMResponse(content=self.content, tool_calls=[])


@pytest.mark.asyncio
async def test_llm_worker_sends_request_with_private_context() -> None:
    provider = FakeProvider(content="  The answer.  ")
    worker = LLMBackgroundWorker(provider, timeout_seconds=5)
    result = await worker("find the fares", "User: I'm flying in May")
    assert result == "The answer."
    messages = provider.messages[0]
    assert messages[0]["role"] == "system" and "User: I'm flying in May" in messages[0]["content"]
    assert "concise" in messages[0]["content"].lower()
    assert messages[-1] == {"role": "user", "content": "find the fares"}
    assert all("tools" not in message for message in messages)

    without_context = await worker("find the fares", "")
    assert without_context == "The answer."
    assert "[" not in provider.messages[1][0]["content"].split("\n")[0]


@pytest.mark.asyncio
async def test_llm_worker_times_out_and_rejects_empty_answers() -> None:
    slow = LLMBackgroundWorker(FakeProvider(delay=1), timeout_seconds=0.05)
    with pytest.raises(asyncio.TimeoutError):
        await slow("x", "")
    empty = LLMBackgroundWorker(FakeProvider(content="   "), timeout_seconds=1)
    with pytest.raises(RuntimeError):
        await empty("x", "")


def test_capture_task_context_is_private_bounded_and_redacted() -> None:
    history = [
        _message("system", "hidden system prompt"),
        _message("user", f"my api key is {SECRET}"),
        _message("assistant", BACKGROUND_ACK_REPLY),
        _message("assistant", "Sure."),
        SimpleNamespace(type="function_call", role="assistant", content="tool"),
        _message("user", "w" * 10_000),
    ]
    context = capture_task_context(FakeSession(history))
    assert SECRET not in context
    assert "hidden system prompt" not in context
    assert BACKGROUND_ACK_REPLY not in context
    assert "Sure." in context
    assert len(context) <= MAX_TASK_CONTEXT_CHARS
    assert capture_task_context(FakeSession()) == ""
    assert capture_task_context(SimpleNamespace()) == ""


@pytest.mark.asyncio
async def test_telegram_text_notification_is_bounded() -> None:
    class _Response:
        def raise_for_status(self) -> None:
            return None

    class _Client:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def post(self, url: str, *, json: dict) -> _Response:
            self.calls.append((url, json))
            return _Response()

    client = _Client()
    notifier = TelegramCallNotifier(token="tok", chat_id="42", client=client)
    await notifier.notify_text("m" * (MAX_FALLBACK_CHARS * 2))
    url, payload = client.calls[0]
    assert url.endswith("/bottok/sendMessage")
    assert payload["chat_id"] == "42"
    assert len(payload["text"]) <= MAX_FALLBACK_CHARS
    with pytest.raises(ValueError):
        await notifier.notify_text("   ")


def test_settings_defaults_and_redaction() -> None:
    defaults = settings_module.DEFAULT_SETTINGS
    assert defaults["background_tasks_enabled"] is True
    concurrency = defaults["background_task_max_concurrency"]
    assert 1 <= concurrency <= background_tasks.MAX_CONCURRENCY_LIMIT
    assert defaults["background_task_timeout_seconds"] > 0
    assert "background_task_shutdown_grace_seconds" not in defaults
    assert defaults["telegram_bot_token"] == ""
    assert defaults["telegram_chat_id"] == ""
    assert settings_module.is_sensitive_key("telegram_bot_token")
    redacted = settings_module.redact_sensitive_values({"telegram_bot_token": "123:abc"})
    assert redacted["telegram_bot_token"] == settings_module.REDACTED_SECRET


def test_voice_agent_runtime_settings_carry_background_config(monkeypatch) -> None:
    voice_agent = _load_voice_agent()
    monkeypatch.setattr(
        voice_agent.settings_module,
        "load_settings",
        lambda: {
            **settings_module.DEFAULT_SETTINGS,
            "background_task_max_concurrency": 99,
            "telegram_bot_token": "from-settings",
        },
    )
    monkeypatch.setattr(voice_agent.settings_module, "load_user_settings", lambda: {})
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "env-chat")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    runtime = voice_agent.get_runtime_settings()
    assert runtime["background_tasks_enabled"] is True
    assert runtime["background_task_max_concurrency"] == background_tasks.MAX_CONCURRENCY_LIMIT
    assert runtime["telegram_bot_token"] == "from-settings"
    assert runtime["telegram_chat_id"] == "env-chat"


def test_build_background_bridge_uses_provider_and_optional_telegram(monkeypatch) -> None:
    voice_agent = _load_voice_agent()
    runtime = {
        **settings_module.DEFAULT_SETTINGS,
        "background_tasks_enabled": True,
        "telegram_bot_token": "",
        "telegram_chat_id": "",
    }
    bridge = voice_agent.build_background_task_bridge(
        runtime, provider=FakeProvider(), session_key=SESSION_KEY
    )
    assert isinstance(bridge, BackgroundTaskBridge)
    assert bridge.has_fallback is False
    runtime["telegram_bot_token"], runtime["telegram_chat_id"] = "t", "c"
    assert (
        voice_agent.build_background_task_bridge(
            runtime, provider=FakeProvider(), session_key=SESSION_KEY
        ).has_fallback
        is True
    )
    runtime["background_tasks_enabled"] = False
    assert (
        voice_agent.build_background_task_bridge(
            runtime, provider=FakeProvider(), session_key=SESSION_KEY
        )
        is None
    )
    assert isinstance(background_task_session.LLMBackgroundWorker, type)


def test_voice_agent_wires_document_worker_when_telegram_is_configured() -> None:
    """The production bridge, not only an isolated worker, must deliver PDFs."""
    from caal.document_work import DocumentWorker

    voice_agent = _load_voice_agent()
    runtime = {
        **settings_module.DEFAULT_SETTINGS,
        "background_tasks_enabled": True,
        "telegram_bot_token": "t",
        "telegram_chat_id": "c",
    }
    bridge = voice_agent.build_background_task_bridge(
        runtime, provider=FakeProvider(), session_key=SESSION_KEY
    )

    assert isinstance(bridge._execute, DocumentWorker)


def test_voice_agent_binds_global_telegram_documents_to_the_configured_user() -> None:
    """A signed-in user's document must not leak to the operator Telegram chat."""
    from caal.document_work import DocumentWorker
    from caal.user_scope import UserScope

    voice_agent = _load_voice_agent()
    owner = "usr_" + "a" * 24
    runtime = {
        **settings_module.DEFAULT_SETTINGS,
        "background_tasks_enabled": True,
        "telegram_bot_token": "t",
        "telegram_chat_id": "c",
        "telegram_owner_user_id": owner,
    }
    owner_bridge = voice_agent.build_background_task_bridge(
        runtime,
        provider=FakeProvider(),
        session_key=SESSION_KEY,
        user_scope=UserScope(user_id=owner, identity_configured=True),
    )
    stranger_bridge = voice_agent.build_background_task_bridge(
        runtime,
        provider=FakeProvider(),
        session_key="other",
        user_scope=UserScope(user_id="usr_" + "b" * 24, identity_configured=True),
    )

    assert isinstance(owner_bridge._execute, DocumentWorker)
    assert owner_bridge._execute._deliver is not None
    assert stranger_bridge._execute._deliver is None


# ---------------------------------------------------------------------------
# Documents: a PDF asked for by voice becomes a real file delivered on Telegram
# ---------------------------------------------------------------------------

PDF_REQUEST = "Create a short PDF about yourself and send it to me on Telegram."


class FakeCompose:
    """Stands in for the bounded LLM worker that writes the document text."""

    def __init__(self, body: str = "JARVIS is a local-first voice assistant.") -> None:
        self.calls: list[tuple[str, str]] = []
        self.body = body
        self.error: Exception | None = None

    async def __call__(self, request: str, context: str) -> str:
        self.calls.append((request, context))
        if self.error is not None:
            raise self.error
        return self.body


class FakeDocumentChannel:
    """Records the documents Telegram was asked to deliver."""

    def __init__(self) -> None:
        self.documents: list[tuple[str, bytes, str]] = []
        self.error: Exception | None = None

    async def __call__(self, *, filename: str, content: bytes, caption: str) -> None:
        if self.error is not None:
            raise self.error
        self.documents.append((filename, content, caption))


async def _run_document_task(bridge: BackgroundTaskBridge, request: str = PDF_REQUEST):
    session = FakeSession()
    await bridge.start()
    try:
        assert await bridge.handle_turn(request, session) is True
        await _settle()
    finally:
        await _teardown(bridge)
    tasks = list_tasks(session_key=SESSION_KEY)
    assert len(tasks) == 1
    return tasks[0], session


@pytest.mark.asyncio
async def test_pdf_by_telegram_request_delivers_a_real_pdf_document(store, tmp_path) -> None:
    from caal.document_work import DocumentWorker

    compose = FakeCompose()
    channel = FakeDocumentChannel()
    bridge = BackgroundTaskBridge(
        execute=DocumentWorker(compose=compose, deliver=channel, artifact_dir=tmp_path / "art"),
        session_key=SESSION_KEY,
        max_concurrency=1,
    )

    task, _ = await _run_document_task(bridge)

    assert task.status == SUCCEEDED
    assert len(channel.documents) == 1
    filename, content, caption = channel.documents[0]
    # A real PDF, not a description of one.
    assert filename.endswith(".pdf")
    assert content.startswith(b"%PDF-") and content.rstrip().endswith(b"%%EOF")
    assert compose.body.encode("latin-1") in content or b"JARVIS" in content
    assert caption and len(caption) <= 200
    # The spoken outcome truthfully reports what was made and where it went.
    assert "pdf" in task.result.lower() and "telegram" in task.result.lower()


@pytest.mark.asyncio
async def test_phone_pdf_task_delivers_to_telegram_then_calls_back_once(store, tmp_path) -> None:
    """A file task accepted on the phone leaves the call and returns only after delivery."""
    from caal.document_work import DocumentWorker

    compose = FakeCompose()
    channel = FakeDocumentChannel()
    callbacks: list[tuple[str, str]] = []
    ended: list[bool] = []

    async def dial(destination: str, task_id: str) -> None:
        assert channel.documents, "the callback must follow Telegram file delivery"
        callbacks.append((destination, task_id))

    bridge = BackgroundTaskBridge(
        execute=DocumentWorker(compose=compose, deliver=channel, artifact_dir=tmp_path / "art"),
        session_key=SESSION_KEY,
        dial_callback=dial,
        max_concurrency=1,
    )
    session = FakeSession()

    async def arm_callback_and_end_call() -> None:
        assert await bridge.arm_callback("+178****8345", session) is True
        ended.append(True)

    await bridge.start()
    try:
        outcome = await bridge.process_turn(
            PDF_REQUEST, session, auto_callback=arm_callback_and_end_call
        )
        await _settle()
        (task,) = list_tasks(session_key=SESSION_KEY)

        assert outcome.consumed is True
        assert outcome.scheduled is True
        assert outcome.callback_offered is False
        assert ended == [True]
        assert len(channel.documents) == 1
        assert callbacks == [("+178****8345", task.task_id)]
    finally:
        await _teardown(bridge)


@pytest.mark.asyncio
async def test_document_delivery_failure_settles_the_task_without_leaking_the_cause(
    store, tmp_path, caplog
) -> None:
    from caal.document_work import DocumentWorker

    channel = FakeDocumentChannel()
    channel.error = RuntimeError("401 from api.telegram.org/bot9999:SUPERSECRETTOKEN")
    artifacts = tmp_path / "art"
    bridge = BackgroundTaskBridge(
        execute=DocumentWorker(compose=FakeCompose(), deliver=channel, artifact_dir=artifacts),
        session_key=SESSION_KEY,
        max_concurrency=1,
    )

    with caplog.at_level(logging.DEBUG):
        task, _ = await _run_document_task(bridge)

    # Settled, not stranded, and the reason is a sentence we wrote ourselves.
    assert task.status == FAILED
    assert task.result is None
    assert "SUPERSECRETTOKEN" not in (task.error or "")
    assert "telegram" in (task.error or "").lower()
    assert "SUPERSECRETTOKEN" not in caplog.text
    # No half-delivered artifact is left on disk.
    assert list(artifacts.iterdir()) == []


@pytest.mark.asyncio
async def test_document_request_fails_explicitly_when_no_channel_is_configured(
    store, tmp_path
) -> None:
    from caal.document_work import DocumentWorker

    compose = FakeCompose()
    bridge = BackgroundTaskBridge(
        execute=DocumentWorker(compose=compose, deliver=None, artifact_dir=tmp_path / "art"),
        session_key=SESSION_KEY,
        max_concurrency=1,
    )

    task, _ = await _run_document_task(bridge)

    assert task.status == FAILED
    assert "not set up" in (task.error or "")
    # It fails before spending a model call on a document it cannot deliver.
    assert compose.calls == []


@pytest.mark.asyncio
async def test_ordinary_background_work_never_reaches_the_document_path(store, tmp_path) -> None:
    from caal.document_work import DocumentWorker

    compose = FakeCompose(body="Cheapest fare is 180 euros on Tuesday.")
    channel = FakeDocumentChannel()
    bridge = BackgroundTaskBridge(
        execute=DocumentWorker(compose=compose, deliver=channel, artifact_dir=tmp_path / "art"),
        session_key=SESSION_KEY,
        max_concurrency=1,
    )

    task, _ = await _run_document_task(bridge, BACKGROUND_REQUEST)

    assert task.status == SUCCEEDED
    assert task.result == "Cheapest fare is 180 euros on Tuesday."
    assert channel.documents == []
    # The request reached the ordinary worker unchanged.
    assert compose.calls[0][0] == BACKGROUND_REQUEST


def test_document_reading_is_narrow_and_names_its_channel() -> None:
    from caal.document_work import PDF_KIND, detect_document_request

    request = detect_document_request(PDF_REQUEST)
    assert request is not None
    assert (request.kind, request.channel) == (PDF_KIND, "telegram")
    assert request.deliverable is True

    unsupported = detect_document_request("Make a PDF of my notes and email it to me")
    assert unsupported is not None and unsupported.deliverable is False

    for ordinary in (
        BACKGROUND_REQUEST,
        "Write a PDF summary of the Lisbon trip",
        "Write a PDF explaining how people send files on Telegram",
        "Send me a message on Telegram when you're done",
    ):
        assert detect_document_request(ordinary) is None


def test_rendered_pdf_is_a_structurally_complete_document() -> None:
    from caal import pdf_document

    content = pdf_document.render_pdf(
        title="About JARVIS (draft)",
        body="Long body. " * 4_000,
    )

    assert content.startswith(b"%PDF-1.4")
    assert content.rstrip().endswith(b"%%EOF")
    assert b"/Type /Catalog" in content and b"/Type /Pages" in content
    assert b"/BaseFont /Helvetica" in content
    # The cross reference table points at the real byte offset of every object.
    body = content.split(b"\nxref\n", 1)
    assert len(body) == 2
    start = int(content.rsplit(b"startxref\n", 1)[1].split(b"\n", 1)[0])
    assert content[start : start + 4] == b"xref"
    for number, offset in enumerate(re.findall(rb"^(\d{10}) 00000 n $", content, re.M), start=1):
        assert content[int(offset) :].startswith(f"{number} 0 obj".encode())
    # Parentheses in the title cannot break out of the PDF string literal.
    assert rb"(About JARVIS \(draft\)) Tj" in content
    assert content.count(b"/Type /Page\n") <= pdf_document.MAX_PAGES


def test_stored_artifacts_are_private_opaque_and_reclaimable(tmp_path) -> None:
    from caal import pdf_document

    directory = tmp_path / "artifacts"
    path = pdf_document.store_artifact(b"%PDF-1.4 tiny", directory=directory)

    assert path.parent == directory
    assert oct(path.stat().st_mode)[-3:] == "600"
    assert oct(directory.stat().st_mode)[-3:] == "700"
    assert re.fullmatch(r"[0-9a-f]{32}\.pdf", path.name)

    assert pdf_document.purge_artifacts(directory, max_age_seconds=3_600) == 0
    assert pdf_document.purge_artifacts(directory, max_age_seconds=-1) == 1
    assert list(directory.iterdir()) == []
