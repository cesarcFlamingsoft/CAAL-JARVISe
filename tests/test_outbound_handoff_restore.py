"""Integration coverage for restoring handoff context inside the outbound worker.

The snapshot rides in trusted dispatch metadata, but it must only ever reach
the outbound session after AMD has positively classified a human. Voicemail,
IVR, uncertain verdicts and dial failures stay silent and context-free, and
nothing about the snapshot is ever logged.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from livekit.agents import llm

from caal.handoff_context import CONTINUATION_MESSAGE_ID, ConversationSnapshot, SnapshotTurn
from caal.outbound_runtime import OutboundRoomConfig

APPROVED = "+17805558345"
PRIVATE_TEXT = "the garage renovation budget"

_voice_agent = None


def _load_voice_agent():
    global _voice_agent
    if _voice_agent is None:
        module_path = Path(__file__).parents[1] / "voice_agent.py"
        spec = importlib.util.spec_from_file_location("voice_agent_restore_test", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _voice_agent = module
    return _voice_agent


class _FakeAMDResult:
    def __init__(self, category: str) -> None:
        self.category = SimpleNamespace(value=category)


class _FakeAMDFactory:
    """Stands in for livekit.agents.AMD with a scripted verdict."""

    def __init__(self, category: str) -> None:
        self.category = category

    def __call__(self, session, **kwargs):
        factory = self

        class _Detector:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def execute(self):
                return _FakeAMDResult(factory.category)

        return _Detector()


class _FakeSip:
    def __init__(self, *, fail: Exception | None = None) -> None:
        self.requests: list[object] = []
        self._fail = fail

    async def create_sip_participant(self, request):
        self.requests.append(request)
        if self._fail is not None:
            raise self._fail
        return object()


class _FakeContext:
    def __init__(self, *, sip: _FakeSip) -> None:
        self.api = SimpleNamespace(sip=sip)
        self.room = SimpleNamespace(name="caal-outbound-abc")
        self.shutdown_reasons: list[str] = []

    async def shutdown(self, reason: str = "") -> None:
        self.shutdown_reasons.append(reason)


class _FakeAgent:
    def __init__(self) -> None:
        self.chat_ctx = llm.ChatContext.empty()
        self.chat_ctx.add_message(role="system", content="You are JARVIS.")
        self.updates: list[llm.ChatContext] = []

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        self.updates.append(chat_ctx)
        self.chat_ctx = chat_ctx


def _config(*, with_snapshot: bool) -> OutboundRoomConfig:
    metadata = {"caal_outbound": True, "attempt_id": "abc", "destination": APPROVED}
    if with_snapshot:
        snapshot = ConversationSnapshot(
            turns=(
                SnapshotTurn(role="user", text=f"let's talk about {PRIVATE_TEXT}"),
                SnapshotTurn(role="assistant", text="Sure, where do you want to start?"),
            )
        )
        metadata["handoff_context"] = snapshot.to_metadata()
    config = OutboundRoomConfig.from_dispatch_metadata(
        json.dumps(metadata), allowed_destinations=APPROVED
    )
    assert config is not None
    return config


@pytest.fixture
def outbound_env(monkeypatch):
    monkeypatch.setenv("LIVEKIT_OUTBOUND_TRUNK_ID", "ST_test")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("CAAL_OUTBOUND_RING_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("CAAL_OUTBOUND_MAX_DURATION_SECONDS", raising=False)


async def _run(monkeypatch, *, category: str, sip: _FakeSip, config: OutboundRoomConfig):
    voice_agent = _load_voice_agent()
    monkeypatch.setattr(voice_agent.agents, "AMD", _FakeAMDFactory(category))
    ctx = _FakeContext(sip=sip)
    agent = _FakeAgent()
    answered = await voice_agent.run_outbound_call(ctx, object(), config, agent=agent)
    return answered, ctx, agent


@pytest.mark.asyncio
async def test_human_answer_restores_the_snapshot_once_before_the_greeting(
    monkeypatch, outbound_env, caplog
) -> None:
    with caplog.at_level(logging.DEBUG):
        answered, ctx, agent = await _run(
            monkeypatch, category="human", sip=_FakeSip(), config=_config(with_snapshot=True)
        )

    assert answered is True
    assert ctx.shutdown_reasons == []
    assert len(agent.updates) == 1
    injected = agent.chat_ctx.get_by_id(CONTINUATION_MESSAGE_ID)
    assert injected is not None
    assert injected.role == "system"
    assert PRIVATE_TEXT in injected.text_content
    assert all(PRIVATE_TEXT not in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    "category", ["machine-vm", "machine-ivr", "machine-unavailable", "uncertain"]
)
@pytest.mark.asyncio
async def test_non_human_verdicts_never_restore_the_snapshot(
    monkeypatch, outbound_env, category: str
) -> None:
    answered, ctx, agent = await _run(
        monkeypatch, category=category, sip=_FakeSip(), config=_config(with_snapshot=True)
    )

    assert answered is False
    assert agent.updates == []
    assert agent.chat_ctx.get_by_id(CONTINUATION_MESSAGE_ID) is None
    assert ctx.shutdown_reasons == ["outbound call not answered by a human"]


@pytest.mark.asyncio
async def test_dial_failure_never_restores_the_snapshot(monkeypatch, outbound_env) -> None:
    answered, ctx, agent = await _run(
        monkeypatch,
        category="human",
        sip=_FakeSip(fail=RuntimeError("SIP trunk rejected the call")),
        config=_config(with_snapshot=True),
    )

    assert answered is False
    assert agent.updates == []
    assert ctx.shutdown_reasons == ["outbound call not answered by a human"]


@pytest.mark.asyncio
async def test_legacy_outbound_call_without_snapshot_still_answers_normally(
    monkeypatch, outbound_env
) -> None:
    answered, ctx, agent = await _run(
        monkeypatch, category="human", sip=_FakeSip(), config=_config(with_snapshot=False)
    )

    assert answered is True
    assert agent.updates == []
    assert ctx.shutdown_reasons == []


def test_handoff_greeting_acknowledges_continuity_without_disclosing_details() -> None:
    voice_agent = _load_voice_agent()

    handoff = voice_agent.greeting_instructions(_config(with_snapshot=True))
    plain = voice_agent.greeting_instructions(_config(with_snapshot=False))
    fresh = voice_agent.greeting_instructions(None)

    assert plain == fresh
    assert handoff != plain
    lowered = handoff.lower()
    assert "phone" in lowered
    assert "continu" in lowered or "pick" in lowered
    assert "do not repeat" in lowered or "don't repeat" in lowered or "without repeating" in lowered
    assert PRIVATE_TEXT not in handoff


# --- ledger-backed continuity ------------------------------------------------


@pytest.fixture
def ledger(monkeypatch, tmp_path):
    from caal import conversation_ledger

    monkeypatch.setattr(conversation_ledger, "STORE_PATH", tmp_path / "assistant.sqlite3")
    return conversation_ledger


def _ledger_config(ledger, *, attempt_id: str = "abc") -> tuple[OutboundRoomConfig, str]:
    conversation_id = ledger.open_conversation(session_key="web-room-1", now=1000)
    ledger.append_turn(conversation_id, "user", f"let's talk about {PRIVATE_TEXT}", now=1001)
    ledger.append_turn(conversation_id, "assistant", "Sure, where do you want to start?", now=1002)
    # Linked on the wall clock: the worker claims with the wall clock too, and a
    # continuation linked at logical second 1003 would be outside its claim window.
    ledger.link_continuation(conversation_id, session_key=attempt_id)
    metadata = {
        "caal_outbound": True,
        "attempt_id": attempt_id,
        "destination": APPROVED,
        "conversation_id": conversation_id,
    }
    config = OutboundRoomConfig.from_dispatch_metadata(
        json.dumps(metadata), allowed_destinations=APPROVED
    )
    assert config is not None
    return config, conversation_id


async def _run_with_recorder(monkeypatch, *, category, sip, config):
    from caal.conversation_ledger import ConversationRecorder

    voice_agent = _load_voice_agent()
    monkeypatch.setattr(voice_agent.agents, "AMD", _FakeAMDFactory(category))
    ctx = _FakeContext(sip=sip)
    agent = _FakeAgent()
    recorder = ConversationRecorder()
    answered = await voice_agent.run_outbound_call(
        ctx, object(), config, agent=agent, recorder=recorder
    )
    return answered, ctx, agent, recorder


@pytest.mark.asyncio
async def test_human_answer_hydrates_from_the_ledger_before_the_greeting(
    monkeypatch, outbound_env, ledger, caplog
) -> None:
    config, conversation_id = _ledger_config(ledger)

    with caplog.at_level(logging.DEBUG):
        answered, ctx, agent, recorder = await _run_with_recorder(
            monkeypatch, category="human", sip=_FakeSip(), config=config
        )

    assert answered is True
    assert ctx.shutdown_reasons == []
    assert len(agent.updates) == 1
    injected = agent.chat_ctx.get_by_id(CONTINUATION_MESSAGE_ID)
    assert injected is not None and injected.role == "system"
    assert PRIVATE_TEXT in injected.text_content
    # The phone session now records into the same logical conversation.
    assert recorder.conversation_id == conversation_id
    # Claimed exactly once: a second claim for the same attempt must fail.
    assert ledger.claim_continuation(conversation_id, session_key="abc") is None
    for record in caplog.records:
        assert PRIVATE_TEXT not in record.getMessage()
        assert conversation_id not in record.getMessage()


@pytest.mark.parametrize(
    "category", ["machine-vm", "machine-ivr", "machine-unavailable", "uncertain"]
)
@pytest.mark.asyncio
async def test_non_human_verdicts_never_read_the_ledger_and_release_the_claim(
    monkeypatch, outbound_env, ledger, category: str
) -> None:
    config, conversation_id = _ledger_config(ledger)
    reads: list[str] = []
    real_claim = ledger.claim_continuation

    def _spy(*args, **kwargs):
        reads.append("claim")
        return real_claim(*args, **kwargs)

    monkeypatch.setattr(ledger, "claim_continuation", _spy)

    answered, ctx, agent, recorder = await _run_with_recorder(
        monkeypatch, category=category, sip=_FakeSip(), config=config
    )

    assert answered is False
    assert reads == []
    assert agent.updates == []
    assert recorder.conversation_id is None
    assert ctx.shutdown_reasons == ["outbound call not answered by a human"]
    # The pending link is gone, so the origin closing now deletes the ledger.
    assert ledger.close_session(conversation_id, session_key="web-room-1") is True


@pytest.mark.asyncio
async def test_dial_failure_never_reads_the_ledger(monkeypatch, outbound_env, ledger) -> None:
    config, conversation_id = _ledger_config(ledger)

    answered, ctx, agent, recorder = await _run_with_recorder(
        monkeypatch,
        category="human",
        sip=_FakeSip(fail=RuntimeError("SIP trunk rejected the call")),
        config=config,
    )

    assert answered is False
    assert agent.updates == []
    assert recorder.conversation_id is None
    assert ledger.close_session(conversation_id, session_key="web-room-1") is True


@pytest.mark.asyncio
async def test_expired_or_unknown_continuation_still_answers_without_context(
    monkeypatch, outbound_env, ledger
) -> None:
    metadata = {
        "caal_outbound": True,
        "attempt_id": "abc",
        "destination": APPROVED,
        "conversation_id": "conv_never_linked",
    }
    config = OutboundRoomConfig.from_dispatch_metadata(
        json.dumps(metadata), allowed_destinations=APPROVED
    )

    answered, ctx, agent, recorder = await _run_with_recorder(
        monkeypatch, category="human", sip=_FakeSip(), config=config
    )

    assert answered is True
    assert agent.updates == []
    assert recorder.conversation_id is None
    assert ctx.shutdown_reasons == []


@pytest.mark.asyncio
async def test_legacy_outbound_call_without_conversation_id_ignores_the_ledger(
    monkeypatch, outbound_env, ledger
) -> None:
    answered, ctx, agent, recorder = await _run_with_recorder(
        monkeypatch, category="human", sip=_FakeSip(), config=_config(with_snapshot=False)
    )

    assert answered is True
    assert agent.updates == []
    assert recorder.conversation_id is None
    assert ctx.shutdown_reasons == []


def test_ledger_continuation_uses_the_handoff_greeting(ledger) -> None:
    voice_agent = _load_voice_agent()
    config, _ = _ledger_config(ledger)

    assert voice_agent.greeting_instructions(config) == voice_agent.HANDOFF_GREETING_INSTRUCTIONS
    assert PRIVATE_TEXT not in voice_agent.greeting_instructions(config)
