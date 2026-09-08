"""Coverage for bringing a phone leg's turns back into the origin web session.

When the outbound phone leg ends while the web session that started it is
still open, the ledger keeps a one-time return-sync marker for that session.
The web session consumes it atomically on its next final user turn, before
Hermes sees that turn, and injects the ledger context through the existing
restore seam: silently, exactly once, and never when the origin link closed.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import logging
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from livekit.agents import llm

from caal import conversation_ledger
from caal.conversation_ledger import ConversationRecorder
from caal.handoff_context import CONTINUATION_MESSAGE_ID

ORIGIN = "web-room-1"
PHONE = "attempt-abc"
PHONE_TEXT = "on the phone: pick the oak flooring"

_voice_agent = None


def _load_voice_agent():
    global _voice_agent
    if _voice_agent is None:
        module_path = Path(__file__).parents[1] / "voice_agent.py"
        spec = importlib.util.spec_from_file_location("voice_agent_return_sync_test", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _voice_agent = module
    return _voice_agent


@pytest.fixture(autouse=True)
def store(monkeypatch, tmp_path):
    path = tmp_path / "assistant.sqlite3"
    monkeypatch.setattr(conversation_ledger, "STORE_PATH", path)
    return path


class _FakeAgent:
    """Enough of ``VoiceAssistant`` for ``on_user_turn_completed`` and the restore seam."""

    def __init__(self, *, sync_return_context, turn_consumed=None) -> None:
        self.chat_ctx = llm.ChatContext.empty()
        self.chat_ctx.add_message(role="system", content="You are JARVIS.")
        self.updates: list[llm.ChatContext] = []
        self._turn_consumed = turn_consumed
        self._sync_return_context = sync_return_context

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        self.updates.append(chat_ctx)
        self.chat_ctx = chat_ctx


class _TypedMessage:
    def __init__(self, text: str) -> None:
        self.text_content = text


def _open_origin() -> tuple[ConversationRecorder, str]:
    """A web session with a bound recorder and one visible turn."""
    conversation_id = conversation_ledger.open_conversation(session_key=ORIGIN)
    recorder = ConversationRecorder()
    recorder.bind(conversation_id)
    conversation_ledger.append_turn(conversation_id, "user", "let's plan the garage renovation")
    return recorder, conversation_id


def _run_phone_leg(
    conversation_id: str, *, phone_key: str = PHONE, text: str = PHONE_TEXT
) -> ConversationRecorder:
    """A claimed phone leg that spoke, then ended through the real close path."""
    voice_agent = _load_voice_agent()
    conversation_ledger.link_continuation(conversation_id, session_key=phone_key)
    assert (
        conversation_ledger.claim_continuation(conversation_id, session_key=phone_key) is not None
    )
    phone = ConversationRecorder()
    phone.bind(conversation_id)
    conversation_ledger.append_turn(conversation_id, "user", text)
    conversation_ledger.append_turn(conversation_id, "assistant", "Noted.")
    voice_agent.end_phone_leg_conversation(phone, session_key=phone_key)
    return phone


def _marker_rows(store) -> int:
    import sqlite3

    raw = sqlite3.connect(store)
    try:
        return raw.execute("SELECT COUNT(*) FROM conversation_return_sync").fetchone()[0]
    finally:
        raw.close()


async def _turn(agent: _FakeAgent, text: str) -> tuple[bool, llm.ChatContext]:
    """Mimic LiveKit: a throwaway turn context, StopResponse means Hermes never sees it."""
    voice_agent = _load_voice_agent()
    turn_ctx = agent.chat_ctx.copy()
    try:
        await voice_agent.VoiceAssistant.on_user_turn_completed(
            agent, turn_ctx, _TypedMessage(text)
        )
    except voice_agent.StopResponse:
        return False, turn_ctx
    return True, turn_ctx


# --- phone leg end -----------------------------------------------------------


def test_phone_leg_end_marks_a_return_only_while_the_origin_is_live(store) -> None:
    voice_agent = _load_voice_agent()
    recorder, conversation_id = _open_origin()

    _run_phone_leg(conversation_id)

    assert conversation_ledger.conversation_exists(conversation_id) is True
    assert conversation_ledger.consume_return_sync(conversation_id, session_key=ORIGIN) is not None

    # A second leg after the web tab closed neither marks nor holds the ledger.
    conversation_ledger.link_continuation(conversation_id, session_key="attempt-2")
    conversation_ledger.claim_continuation(conversation_id, session_key="attempt-2")
    voice_agent.close_session_conversation(recorder, session_key=ORIGIN)
    phone = ConversationRecorder()
    phone.bind(conversation_id)
    voice_agent.end_phone_leg_conversation(phone, session_key="attempt-2")

    assert conversation_ledger.conversation_exists(conversation_id) is False


def test_phone_leg_end_is_inert_without_a_bound_conversation(monkeypatch) -> None:
    voice_agent = _load_voice_agent()
    calls: list[str] = []
    monkeypatch.setattr(
        conversation_ledger, "mark_return_sync", lambda *a, **k: calls.append("mark")
    )
    monkeypatch.setattr(conversation_ledger, "close_session", lambda *a, **k: calls.append("close"))

    voice_agent.end_phone_leg_conversation(ConversationRecorder(), session_key=PHONE)

    assert calls == []


def test_phone_leg_end_still_closes_when_marking_fails(monkeypatch) -> None:
    voice_agent = _load_voice_agent()
    _, conversation_id = _open_origin()
    conversation_ledger.link_continuation(conversation_id, session_key=PHONE)
    conversation_ledger.claim_continuation(conversation_id, session_key=PHONE)
    phone = ConversationRecorder()
    phone.bind(conversation_id)

    def _boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(conversation_ledger, "mark_return_sync", _boom)
    voice_agent.end_phone_leg_conversation(phone, session_key=PHONE)

    # The phone link is closed, so the origin closing now deletes the ledger.
    assert conversation_ledger.close_session(conversation_id, session_key=ORIGIN) is True


# --- next web turn -----------------------------------------------------------


@pytest.mark.asyncio
async def test_next_web_turn_hydrates_once_before_hermes_and_silently(caplog) -> None:
    voice_agent = _load_voice_agent()
    recorder, conversation_id = _open_origin()
    _run_phone_leg(conversation_id)
    hydrator = voice_agent.ReturnSyncHydrator(recorder=recorder, session_key=ORIGIN)
    agent = _FakeAgent(sync_return_context=hydrator.hydrate)

    with caplog.at_level(logging.DEBUG):
        reached_llm, turn_ctx = await _turn(agent, "so what did we decide")

    assert reached_llm is True
    assert len(agent.updates) == 1
    injected = agent.chat_ctx.get_by_id(CONTINUATION_MESSAGE_ID)
    assert injected is not None and injected.role == "system"
    assert PHONE_TEXT in injected.text_content
    # The turn Hermes is about to get already carries the context too.
    in_turn = turn_ctx.get_by_id(CONTINUATION_MESSAGE_ID)
    assert in_turn is not None and PHONE_TEXT in in_turn.text_content
    assert turn_ctx.index_by_id(CONTINUATION_MESSAGE_ID) < len(turn_ctx.items)
    # Nothing was spoken or logged about it.
    for record in caplog.records:
        assert PHONE_TEXT not in record.getMessage()
        assert conversation_id not in record.getMessage()

    reached_llm, turn_ctx = await _turn(agent, "and the budget?")

    assert reached_llm is True
    assert len(agent.updates) == 1
    assert sum(1 for item in agent.chat_ctx.items if item.id == CONTINUATION_MESSAGE_ID) == 1
    assert sum(1 for item in turn_ctx.items if item.id == CONTINUATION_MESSAGE_ID) == 1


@pytest.mark.asyncio
async def test_locally_consumed_turn_leaves_the_marker_for_the_next_hermes_turn() -> None:
    voice_agent = _load_voice_agent()
    recorder, conversation_id = _open_origin()
    _run_phone_leg(conversation_id)
    hydrator = voice_agent.ReturnSyncHydrator(recorder=recorder, session_key=ORIGIN)

    async def _consumed(text: str) -> bool:
        return True

    agent = _FakeAgent(sync_return_context=hydrator.hydrate, turn_consumed=_consumed)

    reached_llm, _ = await _turn(agent, "hang up")

    assert reached_llm is False
    assert agent.updates == []
    assert conversation_ledger.consume_return_sync(conversation_id, session_key=ORIGIN) is not None


@pytest.mark.asyncio
async def test_web_turn_never_hydrates_when_the_origin_closed_before_the_phone_ended() -> None:
    voice_agent = _load_voice_agent()
    recorder, conversation_id = _open_origin()
    conversation_ledger.link_continuation(conversation_id, session_key=PHONE)
    conversation_ledger.claim_continuation(conversation_id, session_key=PHONE)
    voice_agent.close_session_conversation(recorder, session_key=ORIGIN)
    phone = ConversationRecorder()
    phone.bind(conversation_id)
    conversation_ledger.append_turn(conversation_id, "user", PHONE_TEXT)
    voice_agent.end_phone_leg_conversation(phone, session_key=PHONE)
    hydrator = voice_agent.ReturnSyncHydrator(recorder=recorder, session_key=ORIGIN)
    agent = _FakeAgent(sync_return_context=hydrator.hydrate)

    reached_llm, turn_ctx = await _turn(agent, "so what did we decide")

    assert reached_llm is True
    assert agent.updates == []
    assert turn_ctx.get_by_id(CONTINUATION_MESSAGE_ID) is None


@pytest.mark.asyncio
async def test_web_turn_without_a_pending_return_touches_nothing() -> None:
    voice_agent = _load_voice_agent()
    recorder, _ = _open_origin()
    hydrator = voice_agent.ReturnSyncHydrator(recorder=recorder, session_key=ORIGIN)
    agent = _FakeAgent(sync_return_context=hydrator.hydrate)

    reached_llm, turn_ctx = await _turn(agent, "what time is it")

    assert reached_llm is True
    assert agent.updates == []
    assert turn_ctx.get_by_id(CONTINUATION_MESSAGE_ID) is None


@pytest.mark.asyncio
async def test_ledger_failure_never_breaks_the_web_turn(monkeypatch, caplog) -> None:
    voice_agent = _load_voice_agent()
    recorder, conversation_id = _open_origin()
    _run_phone_leg(conversation_id)

    def _boom(*args, **kwargs):
        raise RuntimeError(f"disk full while reading {PHONE_TEXT}")

    monkeypatch.setattr(conversation_ledger, "claim_return_sync", _boom)
    hydrator = voice_agent.ReturnSyncHydrator(recorder=recorder, session_key=ORIGIN)
    agent = _FakeAgent(sync_return_context=hydrator.hydrate)

    with caplog.at_level(logging.DEBUG):
        reached_llm, _ = await _turn(agent, "so what did we decide")

    assert reached_llm is True
    assert agent.updates == []
    assert all(PHONE_TEXT not in record.getMessage() for record in caplog.records)


# --- at-least-once delivery --------------------------------------------------


class _FlakyAgent(_FakeAgent):
    """Fails ``update_chat_ctx`` a scripted number of times, then behaves."""

    def __init__(self, *, sync_return_context, failures: int) -> None:
        super().__init__(sync_return_context=sync_return_context)
        self.failures = failures

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError(f"history update failed while holding {PHONE_TEXT}")
        await super().update_chat_ctx(chat_ctx)


@pytest.mark.asyncio
async def test_a_failed_restore_keeps_the_marker_so_the_next_turn_retries(store, caplog) -> None:
    voice_agent = _load_voice_agent()
    recorder, conversation_id = _open_origin()
    _run_phone_leg(conversation_id)
    hydrator = voice_agent.ReturnSyncHydrator(recorder=recorder, session_key=ORIGIN)
    agent = _FlakyAgent(sync_return_context=hydrator.hydrate, failures=1)

    with caplog.at_level(logging.DEBUG):
        reached_llm, turn_ctx = await _turn(agent, "so what did we decide")

    assert reached_llm is True
    assert agent.updates == []
    assert turn_ctx.get_by_id(CONTINUATION_MESSAGE_ID) is None
    assert _marker_rows(store) == 1
    assert all(PHONE_TEXT not in record.getMessage() for record in caplog.records)

    reached_llm, turn_ctx = await _turn(agent, "hello again")

    assert reached_llm is True
    assert len(agent.updates) == 1
    assert PHONE_TEXT in agent.chat_ctx.get_by_id(CONTINUATION_MESSAGE_ID).text_content
    assert PHONE_TEXT in turn_ctx.get_by_id(CONTINUATION_MESSAGE_ID).text_content
    assert _marker_rows(store) == 0

    reached_llm, _ = await _turn(agent, "and again")
    assert len(agent.updates) == 1


@pytest.mark.asyncio
async def test_a_failed_turn_injection_keeps_the_marker_without_duplicating_history(
    store, monkeypatch
) -> None:
    voice_agent = _load_voice_agent()
    recorder, conversation_id = _open_origin()
    _run_phone_leg(conversation_id)
    hydrator = voice_agent.ReturnSyncHydrator(recorder=recorder, session_key=ORIGIN)
    agent = _FakeAgent(sync_return_context=hydrator.hydrate)
    real_inject = voice_agent.inject_continuation_preamble
    calls = {"n": 0}

    def _inject_then_fail(chat_ctx, context):
        # Only the per-turn injection goes through this name; the durable
        # history update inside the restore seam has already succeeded.
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("turn context rejected the preamble")
        return real_inject(chat_ctx, context)

    monkeypatch.setattr(voice_agent, "inject_continuation_preamble", _inject_then_fail)

    reached_llm, turn_ctx = await _turn(agent, "so what did we decide")

    assert reached_llm is True
    assert _marker_rows(store) == 1
    assert turn_ctx.get_by_id(CONTINUATION_MESSAGE_ID) is None

    reached_llm, turn_ctx = await _turn(agent, "hello again")

    assert reached_llm is True
    assert _marker_rows(store) == 0
    assert sum(1 for item in agent.chat_ctx.items if item.id == CONTINUATION_MESSAGE_ID) == 1
    assert sum(1 for item in turn_ctx.items if item.id == CONTINUATION_MESSAGE_ID) == 1
    assert PHONE_TEXT in turn_ctx.get_by_id(CONTINUATION_MESSAGE_ID).text_content


@pytest.mark.asyncio
async def test_concurrent_turns_apply_the_return_exactly_once(store) -> None:
    import asyncio

    voice_agent = _load_voice_agent()
    recorder, conversation_id = _open_origin()
    _run_phone_leg(conversation_id)
    hydrator = voice_agent.ReturnSyncHydrator(recorder=recorder, session_key=ORIGIN)
    started = asyncio.Event()
    release = asyncio.Event()

    class _SlowAgent(_FakeAgent):
        async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
            started.set()
            await release.wait()
            await super().update_chat_ctx(chat_ctx)

    agent = _SlowAgent(sync_return_context=hydrator.hydrate)
    first = asyncio.create_task(_turn(agent, "first"))
    await started.wait()
    # A second turn arrives while the first is still mid-restore.
    reached_llm, second_ctx = await _turn(agent, "second")
    assert reached_llm is True
    assert second_ctx.get_by_id(CONTINUATION_MESSAGE_ID) is None
    release.set()
    reached_llm, first_ctx = await first

    assert reached_llm is True
    assert len(agent.updates) == 1
    assert first_ctx.get_by_id(CONTINUATION_MESSAGE_ID) is not None
    assert _marker_rows(store) == 0
    # Replaying after the ack changes nothing.
    await _turn(agent, "third")
    assert len(agent.updates) == 1
    assert sum(1 for item in agent.chat_ctx.items if item.id == CONTINUATION_MESSAGE_ID) == 1


# --- repeated handoffs -------------------------------------------------------


@pytest.mark.asyncio
async def test_a_second_phone_leg_refreshes_the_continuation_with_its_new_turns(store) -> None:
    voice_agent = _load_voice_agent()
    recorder, conversation_id = _open_origin()
    hydrator = voice_agent.ReturnSyncHydrator(recorder=recorder, session_key=ORIGIN)
    agent = _FakeAgent(sync_return_context=hydrator.hydrate)

    _run_phone_leg(conversation_id, phone_key="attempt-1", text=PHONE_TEXT)
    reached_llm, _ = await _turn(agent, "so what did we decide")
    assert reached_llm is True
    assert len(agent.updates) == 1

    second_text = "second call: the budget is approved"
    _run_phone_leg(conversation_id, phone_key="attempt-2", text=second_text)
    reached_llm, turn_ctx = await _turn(agent, "and after that call?")

    assert reached_llm is True
    assert len(agent.updates) == 2
    assert _marker_rows(store) == 0
    for chat_ctx in (agent.chat_ctx, turn_ctx):
        preambles = [item for item in chat_ctx.items if item.id == CONTINUATION_MESSAGE_ID]
        assert len(preambles) == 1
        assert second_text in preambles[0].text_content
        assert PHONE_TEXT in preambles[0].text_content

    # Idle turns afterwards leave everything alone.
    reached_llm, _ = await _turn(agent, "thanks")
    assert len(agent.updates) == 2


@pytest.mark.asyncio
async def test_assistant_without_a_return_hook_behaves_as_before() -> None:
    voice_agent = _load_voice_agent()
    agent = SimpleNamespace(_turn_consumed=None, _sync_return_context=None)

    await voice_agent.VoiceAssistant.on_user_turn_completed(
        agent, llm.ChatContext.empty(), _TypedMessage("what time is it")
    )


# --- wiring ------------------------------------------------------------------


def test_entrypoint_wires_the_return_sync_into_the_real_lifecycle() -> None:
    voice_agent = _load_voice_agent()
    source = inspect.getsource(voice_agent.entrypoint)

    assert "ReturnSyncHydrator(" in source
    assert "sync_return_context=" in source
    assert "end_phone_leg_conversation(" in source
    # The phone leg marks the return on the way out; the web session merely closes.
    assert source.index("end_phone_leg_conversation(") < source.index("close_session_conversation(")


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Attribute):
                names.add(func.attr)
            elif isinstance(func, ast.Name):
                names.add(func.id)
    return names


def test_entrypoint_releases_the_ledger_however_the_session_lifecycle_ends() -> None:
    """One outer ``finally`` covers session start, the greeting and the wait."""
    voice_agent = _load_voice_agent()
    source = textwrap.dedent(inspect.getsource(voice_agent.entrypoint))
    tree = ast.parse(source)

    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and "end_phone_leg_conversation" in {n for h in node.finalbody for n in _called_names(h)}
    ]
    assert len(guards) == 1, "exactly one cleanup guard, so the ledger is released once"
    guard = guards[0]
    finalizer = {name for handler in guard.finalbody for name in _called_names(handler)}
    assert "close_session_conversation" in finalizer

    body_calls = {name for statement in guard.body for name in _called_names(statement)}
    assert "start" in body_calls, "session.start must run inside the guard"
    assert "generate_reply" in body_calls, "the greeting must run inside the guard"
    assert "wait" in body_calls, "waiting for close must run inside the guard"
    assert "run_outbound_call" in body_calls, "the outbound dial must run inside the guard"
    # Cleanup is not duplicated anywhere in the body of the guard or outside it.
    outside = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"end_phone_leg_conversation", "close_session_conversation"}
    ]
    assert len(outside) == 2


@pytest.mark.asyncio
async def test_session_liveness_heartbeat_touches_the_bound_conversation(monkeypatch) -> None:
    voice_agent = _load_voice_agent()
    recorder, conversation_id = _open_origin()
    touched: list[tuple[str, str]] = []
    monkeypatch.setattr(
        conversation_ledger,
        "touch_session",
        lambda cid, *, session_key, now=None: touched.append((cid, session_key)) or True,
    )

    voice_agent.touch_session_conversation(recorder, session_key=ORIGIN)
    voice_agent.touch_session_conversation(ConversationRecorder(), session_key=ORIGIN)

    assert touched == [(conversation_id, ORIGIN)]

    def _boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(conversation_ledger, "touch_session", _boom)
    voice_agent.touch_session_conversation(recorder, session_key=ORIGIN)


def test_entrypoint_runs_a_liveness_heartbeat_inside_the_guard() -> None:
    voice_agent = _load_voice_agent()
    source = inspect.getsource(voice_agent.entrypoint)

    assert "touch_session_conversation(" in source
    assert "SESSION_LIVENESS_INTERVAL_SECONDS" in source
