"""Regression checks for runtime names used by the LiveKit agent paths."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import typing
from pathlib import Path

import httpx

from caal.call_termination import end_call_requested
from caal.handoff_intent import PhoneHandoffController, handoff_requested


def _load_voice_agent_module():
    module_path = Path(__file__).parents[1] / "voice_agent.py"
    spec = importlib.util.spec_from_file_location("voice_agent_runtime_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_voice_agent_exposes_livekit_packet_and_settings_dependencies():
    """Webhook commands can resolve their packet and settings dependencies."""
    voice_agent = _load_voice_agent_module()

    assert voice_agent.rtc.DataPacket
    assert callable(voice_agent.get_setting)


def test_wake_word_recognition_annotations_resolve():
    """The non-streaming STT path has a concrete audio-buffer annotation."""
    from caal.stt.wake_word_gated import WakeWordGatedSTT

    hints = typing.get_type_hints(WakeWordGatedSTT._recognize_impl)

    assert "buffer" in hints


class _FakeJobContext:
    """Minimal stand-in for the LiveKit job context the agent runs inside."""

    def __init__(self, metadata: str) -> None:
        self.job = type("_Job", (), {"metadata": metadata})()
        self.api = object()


def test_phone_handoff_is_offered_for_an_inbound_job_with_one_approved_number(monkeypatch):
    """A web caller gets the handoff path, dispatched to the named caal worker."""
    voice_agent = _load_voice_agent_module()
    monkeypatch.setenv("CAAL_OUTBOUND_ALLOWED_DESTINATIONS", "+17805558345")
    monkeypatch.delenv("CAAL_AGENT_NAME", raising=False)

    controller = voice_agent.build_phone_handoff_controller(_FakeJobContext(""))

    assert isinstance(controller, PhoneHandoffController)
    assert controller._start_call.__self__._agent_name == "caal"


def test_phone_handoff_is_disabled_inside_an_outbound_job(monkeypatch):
    """An outbound call must never be able to dial itself again."""
    voice_agent = _load_voice_agent_module()
    monkeypatch.setenv("CAAL_OUTBOUND_ALLOWED_DESTINATIONS", "+17805558345")
    metadata = json.dumps(
        {"caal_outbound": True, "attempt_id": "abc", "destination": "+17805558345"}
    )

    assert voice_agent.build_phone_handoff_controller(_FakeJobContext(metadata)) is None


def test_phone_handoff_is_disabled_without_exactly_one_approved_number(monkeypatch):
    """Without a single approved destination there is nothing safe to dial."""
    voice_agent = _load_voice_agent_module()

    monkeypatch.setenv("CAAL_OUTBOUND_ALLOWED_DESTINATIONS", "")
    assert voice_agent.build_phone_handoff_controller(_FakeJobContext("")) is None

    monkeypatch.setenv("CAAL_OUTBOUND_ALLOWED_DESTINATIONS", "+178****8345,+178****0000")
    assert voice_agent.build_phone_handoff_controller(_FakeJobContext("")) is None

    # Bad configuration must disable handoff rather than crash the voice session.
    monkeypatch.setenv("CAAL_OUTBOUND_ALLOWED_DESTINATIONS", "not-a-phone-number")
    assert voice_agent.build_phone_handoff_controller(_FakeJobContext("")) is None


def test_is_outbound_job_tolerates_missing_or_malformed_metadata():
    voice_agent = _load_voice_agent_module()

    assert voice_agent.is_outbound_job(json.dumps({"caal_outbound": True})) is True
    assert voice_agent.is_outbound_job("") is False
    assert voice_agent.is_outbound_job("not json") is False
    assert voice_agent.is_outbound_job(json.dumps({"caal_outbound": "yes"})) is False


def test_termination_phrases_outrank_the_phone_handoff_path():
    """The transcript callback checks termination first; the two must not overlap."""
    for phrase in ("hang up", "JARVIS, end the call", "goodbye"):
        assert end_call_requested(phrase)
        assert not handoff_requested(phrase)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeAsyncClient:
    payload = {}

    def __init__(self, *, timeout):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, url, *, headers, json):
        return _FakeResponse(self.payload)


def test_integration_payloads_are_not_logged(monkeypatch):
    """Provider responses must not put tokens or private content in logs."""
    voice_agent = _load_voice_agent_module()
    _FakeAsyncClient.payload = {
        "conversation_id": "conversation-1",
        "access_token": "never-log-this-token",
        "response": {"speech": {"plain": {"speech": "The garage is closed."}}},
    }
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    messages = []
    monkeypatch.setattr(voice_agent.logger, "info", lambda message: messages.append(str(message)))

    _, tools = voice_agent.create_hass_tools("http://home.test", "token", "agent")
    result = asyncio.run(tools["hass_assist"]("Is the garage closed?"))

    assert result == "The garage is closed."
    assert "never-log-this-token" not in "\n".join(messages)


def test_friday_payloads_are_not_logged(monkeypatch):
    """Delegated-assistant responses must not put private content in logs."""
    voice_agent = _load_voice_agent_module()
    _FakeAsyncClient.payload = {
        "access_token": "never-log-this-token",
        "choices": [{"message": {"content": "Everything is ready."}}],
    }
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    messages = []
    monkeypatch.setattr(voice_agent.logger, "info", lambda message: messages.append(str(message)))

    _, tools = voice_agent.create_friday_tools("http://friday.test", "token", "main")
    result = asyncio.run(tools["friday"]("Status update"))

    assert result == "Everything is ready."
    assert "never-log-this-token" not in "\n".join(messages)


# --- conversation ledger wiring ---------------------------------------------


def test_phone_handoff_controller_carries_the_session_conversation_id(monkeypatch):
    voice_agent = _load_voice_agent_module()
    monkeypatch.setenv("CAAL_OUTBOUND_ALLOWED_DESTINATIONS", "+17805558345")

    controller = voice_agent.build_phone_handoff_controller(
        _FakeJobContext(""), conversation_id="conv_opaque"
    )

    assert isinstance(controller, PhoneHandoffController)
    assert controller.conversation_id == "conv_opaque"


def test_conversation_capture_records_every_added_history_item(monkeypatch, tmp_path):
    from livekit.agents import llm

    from caal import conversation_ledger

    monkeypatch.setattr(conversation_ledger, "STORE_PATH", tmp_path / "assistant.sqlite3")
    voice_agent = _load_voice_agent_module()

    class _Session:
        def __init__(self) -> None:
            self.handlers = {}

        def on(self, name, handler=None):
            def _register(fn):
                self.handlers[name] = fn
                return fn

            return _register(handler) if handler is not None else _register

    session = _Session()
    recorder = conversation_ledger.ConversationRecorder()
    conversation_id = conversation_ledger.open_conversation(session_key="room-1", now=1000)
    recorder.bind(conversation_id)

    voice_agent.attach_conversation_capture(session, recorder)

    handler = session.handlers["conversation_item_added"]
    handler(type("Ev", (), {"item": llm.ChatMessage(role="user", content=["hello there"])})())
    handler(type("Ev", (), {"item": llm.ChatMessage(role="system", content=["hidden"])})())
    handler(type("Ev", (), {})())  # malformed event must not raise

    assert recorder.recorded == 1


def test_session_conversation_opens_for_inbound_and_not_for_outbound_jobs(monkeypatch, tmp_path):
    from caal import conversation_ledger

    monkeypatch.setattr(conversation_ledger, "STORE_PATH", tmp_path / "assistant.sqlite3")
    voice_agent = _load_voice_agent_module()
    outbound = json.dumps(
        {"caal_outbound": True, "attempt_id": "abc", "destination": "+17805558345"}
    )

    inbound_id = voice_agent.open_session_conversation("", session_key="room-1")
    outbound_id = voice_agent.open_session_conversation(outbound, session_key="room-2")

    assert inbound_id is not None
    assert conversation_ledger.conversation_exists(inbound_id)
    assert outbound_id is None
