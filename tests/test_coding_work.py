"""A coding request becomes durable background work, never a silent voice turn.

A delegated coding job takes far longer than a voice turn, so it is queued in
the same durable store as every other background task, acknowledged out loud
straight away, and announced only once the agent runtime has actually returned
a verified result. The queue keeps its own authority: cancel and status stay
deterministic control commands, and ordinary turns are untouched.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from caal import background_tasks
from caal.background_task_session import (
    BACKGROUND_CANCELLED_REPLY,
    BACKGROUND_STATUS_WORKING_REPLY,
    CODING_ACK_REPLY,
    BackgroundTaskBridge,
)
from caal.background_tasks import QUEUED, RUNNING, list_tasks

ROOM = "room-coding"
CODING_TURN = "fix the retry bug in the ollama provider"


class FakeSession:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def say(self, text: str, **_: object) -> None:
        self.spoken.append(text)


class GatedWorker:
    def __init__(self, answer: str = "I fixed the retry and the tests pass.") -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []
        self.gate = asyncio.Event()

    async def __call__(self, request: str, context: str) -> str:
        self.calls.append((request, context))
        await self.gate.wait()
        return self.answer


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setattr(background_tasks, "STORE_PATH", tmp_path / "assistant.sqlite3")
    yield


def make_bridge(*, llm: GatedWorker, coding: GatedWorker | None) -> BackgroundTaskBridge:
    return BackgroundTaskBridge(execute=llm, session_key=ROOM, coding_execute=coding)


async def teardown(bridge: BackgroundTaskBridge) -> None:
    await bridge.abandon()
    await bridge.close()


@pytest.mark.asyncio
async def test_a_coding_turn_is_queued_and_acknowledged(store) -> None:
    llm, coding = GatedWorker(), GatedWorker()
    bridge = make_bridge(llm=llm, coding=coding)
    await bridge.start()
    session = FakeSession()

    outcome = await bridge.process_turn(CODING_TURN, session)
    await asyncio.sleep(0)

    assert outcome.consumed is True
    assert outcome.scheduled is True
    assert session.spoken == [CODING_ACK_REPLY]
    assert len(list_tasks(session_key=bridge.owner_key, statuses=[QUEUED, RUNNING])) == 1
    assert coding.calls and coding.calls[0][0] == CODING_TURN
    assert llm.calls == []
    await teardown(bridge)


@pytest.mark.asyncio
async def test_the_outcome_is_only_spoken_once_the_job_really_finished(store) -> None:
    llm, coding = GatedWorker(), GatedWorker()
    bridge = make_bridge(llm=llm, coding=coding)
    await bridge.start()
    session = FakeSession()

    await bridge.process_turn(CODING_TURN, session)
    await asyncio.sleep(0)
    # Still running: a status question must not claim it is done.
    await bridge.process_turn("how is the background task going", session)
    assert session.spoken[-1] == BACKGROUND_STATUS_WORKING_REPLY

    coding.gate.set()
    await asyncio.sleep(0.05)
    delivered = await bridge.deliver_pending(session)

    assert delivered == 1
    assert "retry" in session.spoken[-1]
    await teardown(bridge)


@pytest.mark.asyncio
async def test_a_coding_job_can_still_be_cancelled(store) -> None:
    llm, coding = GatedWorker(), GatedWorker()
    bridge = make_bridge(llm=llm, coding=coding)
    await bridge.start()
    session = FakeSession()

    await bridge.process_turn(CODING_TURN, session)
    await asyncio.sleep(0)
    await bridge.process_turn("cancel the background task", session)

    assert session.spoken[-1] == BACKGROUND_CANCELLED_REPLY
    assert list_tasks(session_key=bridge.owner_key, statuses=[QUEUED, RUNNING]) == []
    await teardown(bridge)


@pytest.mark.asyncio
async def test_an_ordinary_turn_is_left_alone(store) -> None:
    llm, coding = GatedWorker(), GatedWorker()
    bridge = make_bridge(llm=llm, coding=coding)
    await bridge.start()
    session = FakeSession()

    outcome = await bridge.process_turn("what time is it", session)

    assert outcome.consumed is False
    assert coding.calls == []
    assert session.spoken == []
    await teardown(bridge)


@pytest.mark.asyncio
async def test_without_a_coding_worker_nothing_claims_the_turn(store) -> None:
    llm = GatedWorker()
    bridge = make_bridge(llm=llm, coding=None)
    await bridge.start()
    session = FakeSession()

    outcome = await bridge.process_turn(CODING_TURN, session)

    assert outcome.consumed is False
    assert session.spoken == []
    await teardown(bridge)


@pytest.mark.asyncio
async def test_a_failed_coding_job_is_reported_as_unfinished(store) -> None:
    class Failing:
        async def __call__(self, request: str, context: str) -> str:
            raise RuntimeError("hermes returned 502 from 10.0.0.4")

    bridge = make_bridge(llm=GatedWorker(), coding=Failing())
    await bridge.start()
    session = FakeSession()

    await bridge.process_turn(CODING_TURN, session)
    await asyncio.sleep(0.05)
    await bridge.deliver_pending(session)

    assert "couldn" in session.spoken[-1].lower()
    assert "10.0.0.4" not in session.spoken[-1]
    await teardown(bridge)


# --- the real delegate, on the real queue -------------------------------------


class RecordingHermes:
    """The agent runtime, recording exactly what CAAL asked of it."""

    provider_name = "hermes"
    manages_own_tools = True

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[list[dict[str, object]]] = []

    async def chat(self, messages, **_: object):
        self.calls.append(messages)
        return type("R", (), dict(content=self._content, tool_calls=[]))()


def hermes_bridge(hermes) -> BackgroundTaskBridge:
    from caal.coding_delegation import build_coding_delegate

    return BackgroundTaskBridge(
        execute=GatedWorker(),
        session_key=ROOM,
        coding_execute=build_coding_delegate(dict(), provider=hermes),
    )


@pytest.mark.asyncio
async def test_a_queued_coding_job_is_carried_out_by_the_agent_runtime(store, monkeypatch) -> None:
    """No local process is involved: the container has no Claude Code to run."""
    from caal.coding_delegation import CLAUDE_CODE_EFFORT, COMPLETION_MARKER

    def no_subprocesses(*args, **kwargs):
        raise AssertionError("a coding job must never spawn a local process")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", no_subprocesses)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", no_subprocesses)
    hermes = RecordingHermes(f"I fixed the retry and reran the suite. {COMPLETION_MARKER}")
    bridge = hermes_bridge(hermes)
    await bridge.start()
    session = FakeSession()

    await bridge.process_turn(CODING_TURN, session)
    await asyncio.sleep(0.05)
    delivered = await bridge.deliver_pending(session)

    assert delivered == 1
    assert "reran the suite" in session.spoken[-1]
    assert COMPLETION_MARKER not in session.spoken[-1]
    system = str(hermes.calls[0][0]["content"]).lower()
    assert "claude code" in system and CLAUDE_CODE_EFFORT in system and "default model" in system
    await teardown(bridge)


@pytest.mark.asyncio
async def test_an_unverified_hermes_answer_is_reported_as_unfinished(store) -> None:
    """Hermes talking about the work is not Hermes having done it."""
    hermes = RecordingHermes("That sounds like it is probably in the retry helper somewhere.")
    bridge = hermes_bridge(hermes)
    await bridge.start()
    session = FakeSession()

    await bridge.process_turn(CODING_TURN, session)
    await asyncio.sleep(0.05)
    await bridge.deliver_pending(session)

    assert "couldn" in session.spoken[-1].lower()
    assert "retry helper" not in session.spoken[-1]
    await teardown(bridge)


@pytest.mark.asyncio
async def test_the_conversation_never_crosses_to_the_agent_runtime(store) -> None:
    from caal.coding_delegation import COMPLETION_MARKER

    hermes = RecordingHermes(f"Done. {COMPLETION_MARKER}")
    bridge = hermes_bridge(hermes)
    await bridge.start(context_source=lambda: "User: my door code is 4821\nAssistant: noted")
    session = FakeSession()

    await bridge.process_turn(CODING_TURN, session)
    await asyncio.sleep(0.05)

    sent = " ".join(str(message["content"]) for message in hermes.calls[0])
    assert "4821" not in sent
    await teardown(bridge)


@pytest.mark.asyncio
async def test_nothing_of_the_job_reaches_the_log(store, caplog) -> None:
    from caal.coding_delegation import COMPLETION_MARKER

    caplog.set_level(logging.DEBUG)
    hermes = RecordingHermes(f"I renamed the retry helper. {COMPLETION_MARKER}")
    bridge = hermes_bridge(hermes)
    await bridge.start()
    session = FakeSession()

    await bridge.process_turn(CODING_TURN, session)
    await asyncio.sleep(0.05)
    await bridge.deliver_pending(session)

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "ollama provider" not in logged
    assert "retry helper" not in logged
    await teardown(bridge)
