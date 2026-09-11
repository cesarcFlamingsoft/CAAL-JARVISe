"""Behavioural tests for the Hermes-first latency work (phase 1).

Each test states a latency or safety property of the Hermes runtime:
per-session integration setup, HTTP client reuse, VAD prewarming, a fixed
greeting, keypad authentication that never blocks the audio loop, and tool
dispatch restricted to an explicit allowlist.
"""

from __future__ import annotations

import asyncio
import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VOICE_AGENT = None


def _load_voice_agent():
    """Load voice_agent.py once; it is a script, not an importable package."""
    global _VOICE_AGENT
    if _VOICE_AGENT is None:
        module_path = Path(__file__).parents[1] / "voice_agent.py"
        spec = importlib.util.spec_from_file_location("voice_agent_phase1_test", module_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _VOICE_AGENT = module
    return _VOICE_AGENT


class _FakeProvider:
    """Minimal LLMProvider stand-in for llm_node tests."""

    def __init__(self, *, manages_own_tools: bool) -> None:
        self.manages_own_tools = manages_own_tools
        self.stream_calls: list[list[dict]] = []

    @property
    def provider_name(self) -> str:
        return "hermes" if self.manages_own_tools else "ollama"

    @property
    def model(self) -> str:
        return "test-model"

    async def chat(self, messages, tools=None, **_):
        raise AssertionError("chat() must not be used when no tools are discovered")

    async def chat_stream(self, messages, tools=None, **_):
        self.stream_calls.append(messages)
        yield "Ready."


def _chat_ctx(text: str = "Hello"):
    class ChatMessage:
        def __init__(self, role: str, content: str) -> None:
            self.role = role
            self.text_content = content

    return SimpleNamespace(items=[ChatMessage("user", text)])


# ---------------------------------------------------------------------------
# (1) Hermes skips per-session MCP / n8n / native-tool discovery
# ---------------------------------------------------------------------------


def test_hermes_session_skips_mcp_and_n8n_initialization(monkeypatch) -> None:
    """Hermes owns its own tool runtime, so a session must not pay MCP setup cost."""
    voice_agent = _load_voice_agent()

    def _unexpected(*args, **kwargs):
        raise AssertionError("MCP/n8n setup must be skipped for the Hermes runtime")

    monkeypatch.setattr(voice_agent, "load_mcp_config", _unexpected)
    monkeypatch.setattr(voice_agent, "initialize_mcp_servers", _unexpected)
    monkeypatch.setattr(voice_agent, "discover_n8n_workflows", _unexpected)

    ctx = SimpleNamespace(room=SimpleNamespace(local_participant=None))
    integrations = asyncio.run(
        voice_agent.initialize_session_integrations(ctx, {"llm_provider": "hermes"})
    )

    assert integrations.mcp_servers == {}
    assert integrations.n8n_workflow_tools == []
    assert integrations.n8n_workflow_name_map == {}
    assert integrations.n8n_base_url is None


def test_non_hermes_session_still_initializes_mcp_and_n8n(monkeypatch) -> None:
    """Ollama has no tool runtime of its own and still needs discovered tools."""
    voice_agent = _load_voice_agent()

    n8n_server = object()
    config = SimpleNamespace(name="n8n", url="http://n8n.test:5678/mcp-server/http")

    async def _initialize(configs):
        return {"n8n": n8n_server}, []

    async def _discover(server, base_url):
        assert server is n8n_server
        assert base_url == "http://n8n.test:5678"
        return [{"function": {"name": "run_backup"}}], {"run_backup": "Backup"}

    monkeypatch.setattr(voice_agent, "load_mcp_config", lambda: [config])
    monkeypatch.setattr(voice_agent, "initialize_mcp_servers", _initialize)
    monkeypatch.setattr(voice_agent, "discover_n8n_workflows", _discover)

    ctx = SimpleNamespace(room=SimpleNamespace(local_participant=None))
    integrations = asyncio.run(
        voice_agent.initialize_session_integrations(ctx, {"llm_provider": "ollama"})
    )

    assert integrations.mcp_servers == {"n8n": n8n_server}
    assert integrations.n8n_workflow_name_map == {"run_backup": "Backup"}
    assert integrations.n8n_base_url == "http://n8n.test:5678"


def test_llm_node_skips_tool_discovery_for_hermes(monkeypatch) -> None:
    """Native/MCP tool discovery is dead weight when the provider runs its own loop."""
    import importlib

    llm_node_module = importlib.import_module("caal.llm.llm_node")

    def _unexpected(*args, **kwargs):
        raise AssertionError("tool discovery must be skipped for the Hermes runtime")

    monkeypatch.setattr(llm_node_module, "_discover_tools", _unexpected)
    provider = _FakeProvider(manages_own_tools=True)

    async def _run() -> list[str]:
        return [
            chunk
            async for chunk in llm_node_module.llm_node(
                SimpleNamespace(), _chat_ctx(), provider=provider
            )
        ]

    assert asyncio.run(_run()) == ["Ready."]
    assert len(provider.stream_calls) == 1


def test_llm_node_still_discovers_tools_for_other_providers(monkeypatch) -> None:
    """Providers without their own tool runtime keep the discovery path."""
    import importlib

    llm_node_module = importlib.import_module("caal.llm.llm_node")

    calls = []

    async def _discover(agent):
        calls.append(agent)
        return None

    monkeypatch.setattr(llm_node_module, "_discover_tools", _discover)
    provider = _FakeProvider(manages_own_tools=False)

    async def _run() -> list[str]:
        return [
            chunk
            async for chunk in llm_node_module.llm_node(
                SimpleNamespace(), _chat_ctx(), provider=provider
            )
        ]

    assert asyncio.run(_run()) == ["Ready."]
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# (2) Hermes reuses one httpx client and closes it on shutdown
# ---------------------------------------------------------------------------


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": "Ready, Cesar.", "tool_calls": []}}]}


class _CountingClient:
    instances: list["_CountingClient"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.closed = False
        self.posts = 0
        _CountingClient.instances.append(self)

    async def post(self, url, *, json, headers, **kwargs):
        if self.closed:
            raise AssertionError("request issued on a closed client")
        self.posts += 1
        return _FakeResponse()

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def counting_client(monkeypatch):
    _CountingClient.instances = []
    monkeypatch.setattr(
        "caal.llm.providers.hermes_provider.httpx.AsyncClient",
        _CountingClient,
    )
    return _CountingClient


def test_hermes_provider_reuses_one_httpx_client_across_requests(counting_client) -> None:
    """A fresh TCP/TLS connection per turn adds avoidable latency to every reply."""
    from caal.llm.providers.hermes_provider import HermesProvider

    provider = HermesProvider(api_key="test-key")

    async def _run() -> None:
        await provider.chat([{"role": "user", "content": "one"}])
        await provider.chat([{"role": "user", "content": "two"}])

    asyncio.run(_run())

    assert len(counting_client.instances) == 1
    assert counting_client.instances[0].posts == 2


def test_hermes_provider_aclose_closes_the_shared_client(counting_client) -> None:
    """The shared client must be released when the session shuts down."""
    from caal.llm.providers.hermes_provider import HermesProvider

    provider = HermesProvider(api_key="test-key")

    async def _run() -> None:
        await provider.chat([{"role": "user", "content": "one"}])
        await provider.aclose()
        await provider.aclose()  # idempotent

    asyncio.run(_run())

    assert len(counting_client.instances) == 1
    assert counting_client.instances[0].closed is True


def test_caal_llm_aclose_closes_the_provider() -> None:
    """LiveKit closes the LLM; that must reach the provider's HTTP client."""
    from caal.llm.caal_llm import CAALLLM

    closed = []

    class _Provider(_FakeProvider):
        async def aclose(self) -> None:
            closed.append(True)

    caal_llm = CAALLLM(provider=_Provider(manages_own_tools=True))
    asyncio.run(caal_llm.aclose())

    assert closed == [True]


# ---------------------------------------------------------------------------
# (3) Silero VAD is prewarmed in the worker process
# ---------------------------------------------------------------------------


def test_prewarm_loads_silero_vad_into_process_userdata(monkeypatch) -> None:
    """Loading Silero on the first job delays the first reply of every cold worker."""
    voice_agent = _load_voice_agent()
    sentinel = object()
    monkeypatch.setattr(voice_agent.silero.VAD, "load", staticmethod(lambda **_: sentinel))

    proc = SimpleNamespace(userdata={})
    voice_agent.prewarm(proc)

    assert proc.userdata["vad"] is sentinel


def test_tuned_vad_reuses_the_prewarmed_instance(monkeypatch) -> None:
    """Session tuning is applied to the prewarmed model instead of reloading it."""
    voice_agent = _load_voice_agent()

    def _unexpected(**_):
        raise AssertionError("a prewarmed VAD must not be reloaded")

    monkeypatch.setattr(voice_agent.silero.VAD, "load", staticmethod(_unexpected))

    updates: list[dict] = []
    prewarmed = SimpleNamespace(update_options=lambda **kwargs: updates.append(kwargs))
    runtime = {
        "vad_min_speech_duration": 0.2,
        "vad_min_silence_duration": 0.3,
        "vad_prefix_padding": 0.4,
        "vad_activation_threshold": 0.8,
    }

    vad = voice_agent.load_tuned_vad(runtime, prewarmed=prewarmed)

    assert vad is prewarmed
    assert updates == [
        {
            "min_speech_duration": 0.2,
            "min_silence_duration": 0.3,
            "prefix_padding_duration": 0.4,
            "activation_threshold": 0.8,
        }
    ]


def test_tuned_vad_falls_back_to_loading_when_not_prewarmed(monkeypatch) -> None:
    """Without a prewarmed model the session still gets a correctly tuned VAD."""
    voice_agent = _load_voice_agent()
    loads: list[dict] = []
    sentinel = object()

    def _load(**kwargs):
        loads.append(kwargs)
        return sentinel

    monkeypatch.setattr(voice_agent.silero.VAD, "load", staticmethod(_load))

    vad = voice_agent.load_tuned_vad({"vad_activation_threshold": 0.9}, prewarmed=None)

    assert vad is sentinel
    assert loads[0]["activation_threshold"] == 0.9


# ---------------------------------------------------------------------------
# (4) Fixed JARVIS greeting instead of a generated one
# ---------------------------------------------------------------------------


def test_initial_greeting_is_spoken_verbatim_without_an_llm_round_trip() -> None:
    """The opening line should not wait on a Hermes generation."""
    voice_agent = _load_voice_agent()
    said: list[str] = []

    class _Session:
        async def say(self, text, **_):
            said.append(text)

        async def generate_reply(self, **_):
            raise AssertionError("the initial greeting must not call the LLM")

    asyncio.run(voice_agent.deliver_initial_greeting(_Session()))

    assert said == [voice_agent.INITIAL_GREETING]
    assert "JARVIS" in voice_agent.INITIAL_GREETING


# ---------------------------------------------------------------------------
# (5) scrypt PIN verification runs off the audio event loop
# ---------------------------------------------------------------------------


def test_async_gate_grants_access_for_the_correct_pin() -> None:
    from caal.telephony_auth import CallAccessGate, GateState, hash_pin

    gate = CallAccessGate(hash_pin("2468"), max_attempts=3)

    async def _run():
        for digit in "2468#":
            state = await gate.accept_digit_async(digit)
        return state

    assert asyncio.run(_run()) is GateState.GRANTED
    assert gate.is_granted


def test_async_gate_locks_after_the_attempt_limit() -> None:
    from caal.telephony_auth import CallAccessGate, GateState, hash_pin

    gate = CallAccessGate(hash_pin("2468"), max_attempts=2)

    async def _run():
        for _ in range(2):
            for digit in "1111#":
                state = await gate.accept_digit_async(digit)
        return state

    assert asyncio.run(_run()) is GateState.LOCKED
    assert not gate.is_granted
    assert gate.attempts_remaining == 0


def test_async_gate_verifies_the_pin_off_the_event_loop(monkeypatch) -> None:
    """scrypt is CPU-heavy; running it inline would stall audio for every caller."""
    import caal.telephony_auth as telephony_auth

    real_verify = telephony_auth.verify_pin
    verify_threads: list[int] = []

    def _recording_verify(pin, encoded_hash):
        verify_threads.append(threading.get_ident())
        return real_verify(pin, encoded_hash)

    monkeypatch.setattr(telephony_auth, "verify_pin", _recording_verify)
    gate = telephony_auth.CallAccessGate(telephony_auth.hash_pin("2468"))

    async def _run():
        for digit in "2468#":
            state = await gate.accept_digit_async(digit)
        return state, threading.get_ident()

    state, loop_thread = asyncio.run(_run())

    assert state is telephony_auth.GateState.GRANTED
    assert verify_threads and all(ident != loop_thread for ident in verify_threads)


def test_sip_dtmf_handler_never_verifies_on_the_audio_event_loop(monkeypatch) -> None:
    """The DTMF callback runs on the room's event loop and must stay non-blocking."""
    voice_agent = _load_voice_agent()

    import caal.telephony_auth as telephony_auth

    real_verify = telephony_auth.verify_pin
    verify_threads: list[int] = []

    def _recording_verify(pin, encoded_hash):
        verify_threads.append(threading.get_ident())
        return real_verify(pin, encoded_hash)

    monkeypatch.setattr(telephony_auth, "verify_pin", _recording_verify)
    monkeypatch.setenv("CAAL_CALL_PIN_ENABLED", "true")
    monkeypatch.setenv("CAAL_CALL_PIN_HASH", telephony_auth.hash_pin("2468"))

    registered = asyncio.Event()

    class _Room:
        def __init__(self) -> None:
            self.handlers: dict = {}
            self.remote_participants = {
                "caller": SimpleNamespace(
                    kind=voice_agent.rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                )
            }

        def on(self, event, handler):
            self.handlers[event] = handler
            registered.set()

        def off(self, event, handler):
            self.handlers.pop(event, None)

    class _Session:
        def __init__(self, **_):
            self.spoken: list[str] = []

        async def start(self, *, room, agent):
            return None

        def say(self, text, **_):
            self.spoken.append(text)

        async def aclose(self):
            return None

    monkeypatch.setattr(voice_agent, "AgentSession", _Session)
    monkeypatch.setattr(voice_agent, "Agent", lambda **_: object())

    room = _Room()

    async def _run():
        task = asyncio.create_task(voice_agent.authenticate_sip_call(room, object()))
        await asyncio.wait_for(registered.wait(), timeout=5)
        handler = room.handlers["sip_dtmf_received"]
        for digit in "2468#":
            handler(SimpleNamespace(digit=digit))
        granted = await asyncio.wait_for(task, timeout=10)
        return granted, threading.get_ident()

    granted, loop_thread = asyncio.run(_run())

    assert granted is True
    assert verify_threads and all(ident != loop_thread for ident in verify_threads)


def test_sip_gate_still_rejects_a_wrong_pin(monkeypatch) -> None:
    """Moving scrypt off the loop must not weaken the gate itself."""
    voice_agent = _load_voice_agent()

    from caal.telephony_auth import hash_pin

    monkeypatch.setenv("CAAL_CALL_PIN_ENABLED", "true")
    monkeypatch.setenv("CAAL_CALL_PIN_HASH", hash_pin("2468"))
    monkeypatch.setenv("CAAL_CALL_PIN_MAX_ATTEMPTS", "1")

    registered = asyncio.Event()

    class _Room:
        def __init__(self) -> None:
            self.handlers: dict = {}
            self.remote_participants = {
                "caller": SimpleNamespace(
                    kind=voice_agent.rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                )
            }

        def on(self, event, handler):
            self.handlers[event] = handler
            registered.set()

        def off(self, event, handler):
            self.handlers.pop(event, None)

    class _Session:
        def __init__(self, **_):
            pass

        async def start(self, *, room, agent):
            return None

        def say(self, text, **_):
            return None

        async def aclose(self):
            return None

    monkeypatch.setattr(voice_agent, "AgentSession", _Session)
    monkeypatch.setattr(voice_agent, "Agent", lambda **_: object())

    room = _Room()

    async def _run():
        task = asyncio.create_task(voice_agent.authenticate_sip_call(room, object()))
        await asyncio.wait_for(registered.wait(), timeout=5)
        handler = room.handlers["sip_dtmf_received"]
        for digit in "1111#":
            handler(SimpleNamespace(digit=digit))
        return await asyncio.wait_for(task, timeout=10)

    assert asyncio.run(_run()) is False


# ---------------------------------------------------------------------------
# (6) Tool dispatch uses an explicit allowlist
# ---------------------------------------------------------------------------


class _DispatchAgent:
    def __init__(self) -> None:
        self._hass_tool_callables: dict = {}
        self._friday_tool_callables: dict = {}
        self._n8n_workflow_name_map: dict = {}
        self._n8n_base_url = None
        self._caal_mcp_servers: dict = {}

    async def aclose(self) -> None:
        raise AssertionError("internal agent methods must not be callable as tools")

    async def llm_node(self, *args, **kwargs):
        raise AssertionError("internal agent methods must not be callable as tools")

    async def web_search(self, query: str) -> str:
        return f"results for {query}"


@pytest.mark.parametrize("module_name", ["llm_node", "ollama_node"])
@pytest.mark.parametrize("tool_name", ["aclose", "llm_node", "__init__"])
def test_non_allowlisted_agent_attributes_are_not_dispatchable(module_name, tool_name) -> None:
    """An LLM-supplied name must never reach an arbitrary agent attribute."""
    import importlib

    module = importlib.import_module(f"caal.llm.{module_name}")

    with pytest.raises(ValueError, match="not found"):
        asyncio.run(module._execute_single_tool(_DispatchAgent(), tool_name, {}))


@pytest.mark.parametrize("module_name", ["llm_node", "ollama_node"])
def test_allowlisted_agent_tool_still_executes(module_name) -> None:
    """The allowlist keeps the real agent tool working."""
    import importlib

    module = importlib.import_module(f"caal.llm.{module_name}")

    result = asyncio.run(
        module._execute_single_tool(_DispatchAgent(), "web_search", {"query": "weather"})
    )

    assert result == "results for weather"
