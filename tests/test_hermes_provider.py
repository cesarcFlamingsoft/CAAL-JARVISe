"""Behavioural tests for the Hermes API-backed CAAL LLM provider."""

from __future__ import annotations

import pytest

from caal.llm.providers import create_provider_from_settings
from caal.settings import DEFAULT_SETTINGS


def test_hermes_is_the_default_runtime_llm_provider() -> None:
    assert DEFAULT_SETTINGS["llm_provider"] == "hermes"
    assert DEFAULT_SETTINGS["hermes_api_url"] == "http://host.docker.internal:8642/v1"
    assert DEFAULT_SETTINGS["hermes_model"] == "hermes-agent"


def test_voice_agent_passes_hermes_connection_settings_to_provider_factory(monkeypatch) -> None:
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).parents[1] / "voice_agent.py"
    spec = importlib.util.spec_from_file_location("voice_agent", module_path)
    assert spec and spec.loader
    voice_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(voice_agent)

    runtime_settings = {
        "hermes_api_url": "http://host.docker.internal:8642/v1",
        "hermes_api_key": "test-key",
        "hermes_model": "hermes-agent",
    }
    monkeypatch.setattr(voice_agent.settings_module, "load_settings", lambda: runtime_settings)
    monkeypatch.setattr(voice_agent.settings_module, "load_user_settings", lambda: {})
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    runtime = voice_agent.get_runtime_settings()

    assert runtime["llm_provider"] == "hermes"
    assert runtime["hermes_api_url"] == "http://host.docker.internal:8642/v1"
    assert runtime["hermes_api_key"] == "test-key"
    assert runtime["hermes_model"] == "hermes-agent"


def test_factory_creates_hermes_provider_from_runtime_settings() -> None:
    provider = create_provider_from_settings(
        {
            "llm_provider": "hermes",
            "hermes_api_url": "http://host.docker.internal:8642/v1",
            "hermes_api_key": "test-key",
            "hermes_model": "hermes-agent",
        }
    )

    assert provider.provider_name == "hermes"
    assert provider.model == "hermes-agent"


@pytest.mark.asyncio
async def test_hermes_provider_sends_openai_compatible_chat_request(monkeypatch) -> None:
    """Hermes receives the assembled conversation with a bearer token."""
    from caal.llm.providers.hermes_provider import HermesProvider

    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {"message": {"content": "Ready, Cesar.", "tool_calls": []}}
                ]
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def post(self, url, *, json, headers):
            captured.update(url=url, json=json, headers=headers)
            return FakeResponse()

    monkeypatch.setattr(
        "caal.llm.providers.hermes_provider.httpx.AsyncClient",
        lambda **_: FakeClient(),
    )

    provider = HermesProvider(
        base_url="http://host.docker.internal:8642/v1/",
        api_key="test-key",
        model="hermes-agent",
    )
    response = await provider.chat([{"role": "user", "content": "Hello"}])

    assert response.content == "Ready, Cesar."
    assert response.tool_calls == []
    assert captured == {
        "url": "http://host.docker.internal:8642/v1/chat/completions",
        "json": {
            "model": "hermes-agent",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
        "headers": {"Authorization": "Bearer test-key", "Content-Type": "application/json"},
    }


@pytest.mark.asyncio
async def test_hermes_provider_adds_a_user_turn_for_instruction_only_context(monkeypatch) -> None:
    """A LiveKit initial greeting has only instructions, but Hermes needs a user turn."""
    from caal.llm.providers.hermes_provider import HermesProvider

    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "Hello.", "tool_calls": []}}]}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def post(self, url, *, json, headers):
            captured.update(json=json)
            return FakeResponse()

    monkeypatch.setattr(
        "caal.llm.providers.hermes_provider.httpx.AsyncClient",
        lambda **_: FakeClient(),
    )

    provider = HermesProvider(api_key="test-key")
    await provider.chat([{"role": "system", "content": "Greet the caller briefly."}])

    assert captured["json"]["messages"] == [
        {"role": "system", "content": "Greet the caller briefly."},
        {"role": "user", "content": "Follow the system instruction above."},
    ]


def test_hermes_provider_rejects_missing_api_key() -> None:
    from caal.llm.providers.hermes_provider import HermesProvider

    with pytest.raises(ValueError, match="Hermes API key required"):
        HermesProvider(base_url="http://host.docker.internal:8642/v1", api_key="")


# ---------------------------------------------------------------------------
# SSE streaming
#
# The local Hermes endpoint returns standard OpenAI SSE frames when
# stream=true: `data: {"choices":[{"delta":{"content":"..."}}]}` lines
# terminated by `data: [DONE]`.
# ---------------------------------------------------------------------------


def _sse(*payloads: str) -> list[str]:
    """Render SSE `data:` lines, with the blank separators a real server sends."""
    lines: list[str] = []
    for payload in payloads:
        lines.extend([f"data: {payload}", ""])
    return lines


class _FakeStreamResponse:
    def __init__(self, lines: list[str], *, error: Exception | None = None) -> None:
        self._lines = lines
        self._error = error
        self.consumed: list[str] = []

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error

    async def aiter_lines(self):
        for line in self._lines:
            self.consumed.append(line)
            yield line


class _FakeStreamContext:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeStreamResponse:
        return self._response

    async def __aexit__(self, *args) -> bool:
        return False


class _FakeStreamClient:
    """Stand-in for httpx.AsyncClient exposing only the streaming surface used."""

    def __init__(self, response: _FakeStreamResponse, captured: dict) -> None:
        self._response = response
        self._captured = captured
        self.stream_calls = 0

    def stream(self, method, url, *, json, headers):
        self.stream_calls += 1
        self._captured.update(method=method, url=url, json=json, headers=headers)
        return _FakeStreamContext(self._response)

    async def post(self, url, *, json, headers):
        raise AssertionError("chat_stream must use the streaming request, not post()")

    async def aclose(self) -> None:
        return None


def _stream_provider(monkeypatch, lines, *, error=None):
    """Build a HermesProvider whose shared client returns the given SSE lines."""
    from caal.llm.providers.hermes_provider import HermesProvider

    captured: dict = {}
    response = _FakeStreamResponse(lines, error=error)
    client = _FakeStreamClient(response, captured)
    monkeypatch.setattr(
        "caal.llm.providers.hermes_provider.httpx.AsyncClient",
        lambda **_: client,
    )
    provider = HermesProvider(
        base_url="http://host.docker.internal:8642/v1/",
        api_key="test-key",
        model="hermes-agent",
    )
    return provider, client, captured, response


async def _collect(provider, messages, **kwargs) -> list[str]:
    return [chunk async for chunk in provider.chat_stream(messages, **kwargs)]


@pytest.mark.asyncio
async def test_hermes_chat_stream_requests_a_streaming_completion(monkeypatch) -> None:
    """Streaming must ask Hermes for stream=true on the OpenAI-compatible route."""
    provider, client, captured, _ = _stream_provider(
        monkeypatch,
        _sse('{"choices":[{"delta":{"content":"Hi"}}]}', "[DONE]"),
    )

    await _collect(provider, [{"role": "user", "content": "Hello"}])

    assert client.stream_calls == 1
    assert captured["method"] == "POST"
    assert captured["url"] == "http://host.docker.internal:8642/v1/chat/completions"
    assert captured["json"] == {
        "model": "hermes-agent",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }
    assert captured["headers"] == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }


@pytest.mark.asyncio
async def test_hermes_chat_stream_yields_incremental_content_deltas(monkeypatch) -> None:
    """Each delta is spoken as it arrives instead of waiting for the whole turn."""
    provider, _, _, _ = _stream_provider(
        monkeypatch,
        _sse(
            '{"choices":[{"delta":{"content":"Good "}}]}',
            '{"choices":[{"delta":{"content":"evening, "}}]}',
            '{"choices":[{"delta":{"content":"Cesar."}}]}',
            "[DONE]",
        ),
    )

    chunks = await _collect(provider, [{"role": "user", "content": "Hello"}])

    assert chunks == ["Good ", "evening, ", "Cesar."]


@pytest.mark.asyncio
async def test_hermes_chat_stream_ignores_role_and_empty_chunks(monkeypatch) -> None:
    """Role-only openers, empty deltas and finish frames carry no speakable text."""
    provider, _, _, _ = _stream_provider(
        monkeypatch,
        _sse(
            '{"choices":[{"delta":{"role":"assistant"}}]}',
            '{"choices":[{"delta":{"content":""}}]}',
            '{"choices":[{"delta":{"content":"Ready."}}]}',
            '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
            '{"choices":[]}',
            "[DONE]",
        ),
    )

    chunks = await _collect(provider, [{"role": "user", "content": "Hello"}])

    assert chunks == ["Ready."]


@pytest.mark.asyncio
async def test_hermes_chat_stream_stops_at_the_done_sentinel(monkeypatch) -> None:
    """`data: [DONE]` ends the turn; trailing frames must never be spoken."""
    provider, _, _, _ = _stream_provider(
        monkeypatch,
        _sse(
            '{"choices":[{"delta":{"content":"Done."}}]}',
            "[DONE]",
            '{"choices":[{"delta":{"content":"leaked"}}]}',
        ),
    )

    chunks = await _collect(provider, [{"role": "user", "content": "Hello"}])

    assert chunks == ["Done."]


@pytest.mark.asyncio
async def test_hermes_chat_stream_skips_malformed_sse_lines(monkeypatch) -> None:
    """A truncated or non-JSON frame must not abort an otherwise healthy stream."""
    provider, _, _, _ = _stream_provider(
        monkeypatch,
        [
            ": keep-alive comment",
            "event: message",
            "data: {not json",
            'data: {"choices":[{"delta":{"content":"Still "}}]}',
            "data: []",
            "data: null",
            'data: {"choices":[{"delta":{"content":42}}]}',
            'data: {"choices":[{"delta":{"content":"here."}}]}',
            "data: [DONE]",
        ],
    )

    chunks = await _collect(provider, [{"role": "user", "content": "Hello"}])

    assert chunks == ["Still ", "here."]


@pytest.mark.asyncio
async def test_hermes_chat_stream_propagates_http_failure(monkeypatch) -> None:
    """A rejected stream surfaces as an error rather than a silent empty reply."""
    import httpx

    failure = httpx.HTTPStatusError(
        "401 Unauthorized",
        request=httpx.Request("POST", "http://host.docker.internal:8642/v1/chat/completions"),
        response=httpx.Response(401),
    )
    provider, _, _, response = _stream_provider(
        monkeypatch,
        _sse('{"choices":[{"delta":{"content":"never"}}]}', "[DONE]"),
        error=failure,
    )

    with pytest.raises(httpx.HTTPStatusError):
        await _collect(provider, [{"role": "user", "content": "Hello"}])

    assert response.consumed == []


@pytest.mark.asyncio
async def test_hermes_chat_stream_never_forwards_caal_tools(monkeypatch) -> None:
    """Hermes runs its own tool loop; CAAL schemas must stay out of the request."""
    provider, _, captured, _ = _stream_provider(
        monkeypatch,
        _sse('{"choices":[{"delta":{"content":"Ready."}}]}', "[DONE]"),
    )

    await _collect(
        provider,
        [{"role": "user", "content": "Hello"}],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )

    assert "tools" not in captured["json"]
    assert "tool_choice" not in captured["json"]


@pytest.mark.asyncio
async def test_hermes_chat_stream_adds_a_user_turn_for_instruction_only_context(
    monkeypatch,
) -> None:
    """The streaming path shares the greeting fix-up the completed path applies."""
    provider, _, captured, _ = _stream_provider(
        monkeypatch,
        _sse('{"choices":[{"delta":{"content":"Hello."}}]}', "[DONE]"),
    )

    await _collect(provider, [{"role": "system", "content": "Greet the caller briefly."}])

    assert captured["json"]["messages"] == [
        {"role": "system", "content": "Greet the caller briefly."},
        {"role": "user", "content": "Follow the system instruction above."},
    ]


@pytest.mark.asyncio
async def test_hermes_chat_stream_reuses_the_shared_client(monkeypatch) -> None:
    """Streaming must not open a fresh connection per turn."""
    provider, client, _, _ = _stream_provider(
        monkeypatch,
        _sse('{"choices":[{"delta":{"content":"Ready."}}]}', "[DONE]"),
    )

    await _collect(provider, [{"role": "user", "content": "one"}])
    await _collect(provider, [{"role": "user", "content": "two"}])

    assert client.stream_calls == 2


@pytest.mark.asyncio
async def test_hermes_chat_remains_non_streaming(monkeypatch) -> None:
    """chat() keeps its completed-turn contract and must not use the SSE route."""
    from caal.llm.providers.hermes_provider import HermesProvider

    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "Ready, Cesar."}}]}

    class FakeClient:
        async def post(self, url, *, json, headers):
            captured.update(json=json)
            return FakeResponse()

        def stream(self, *args, **kwargs):
            raise AssertionError("chat() must not open a streaming request")

    monkeypatch.setattr(
        "caal.llm.providers.hermes_provider.httpx.AsyncClient",
        lambda **_: FakeClient(),
    )

    provider = HermesProvider(api_key="test-key")
    response = await provider.chat(
        [{"role": "user", "content": "Hello"}],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )

    assert response.content == "Ready, Cesar."
    assert response.tool_calls == []
    assert captured["json"]["stream"] is False
    assert "tools" not in captured["json"]
