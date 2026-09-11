"""The local model is JARVIS main model; Hermes is the escalation, not the default.

``RoutedProvider`` puts the deterministic reading of :mod:`caal.model_routing`
in front of two providers: the local Ollama model answers ordinary turns, and
only a turn that genuinely needs the agent harness is handed to Hermes. The
fallback between them is bounded and one-way per turn -- at most one other
attempt, never a loop -- and when nothing can answer, the reply says so
instead of leaking an upstream error.

CAAL native tools stay available: the routed provider does not run its own
tool loop, so the local model keeps the user-scoped tool catalog, and the
tools are never forwarded to Hermes, which runs its own.
"""

from __future__ import annotations

import pytest

from caal.llm.providers import LLMProvider, LLMResponse, RoutedProvider, ToolCall

ORDINARY = [{"role": "user", "content": "what time is it"}]
HARNESS = [{"role": "user", "content": "research the alberta grid and write it up"}]
CODING = [{"role": "user", "content": "fix the bug in the checkout script"}]


class FakeProvider(LLMProvider):
    _stream_error_at = 0

    def __init__(
        self,
        name: str,
        *,
        content: str | None = "ok",
        error: Exception | None = None,
        chunks: list[str] | None = None,
        stream_error: Exception | None = None,
    ) -> None:
        self._name = name
        self._content = content
        self._error = error
        self._chunks = chunks if chunks is not None else ["hello ", "there"]
        self._stream_error = stream_error
        self.calls: list[list[dict]] = []
        self.tools_seen: list[object] = []
        self.stream_calls = 0
        self.closed = 0

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return f"{self._name}-model"

    async def chat(self, messages, tools=None, **kwargs):
        self.calls.append(messages)
        self.tools_seen.append(tools)
        if self._error is not None:
            raise self._error
        return LLMResponse(content=self._content, tool_calls=[])

    async def chat_stream(self, messages, tools=None, **kwargs):
        self.stream_calls += 1
        for index, chunk in enumerate(self._chunks):
            if self._stream_error is not None and index == self._stream_error_at:
                raise self._stream_error
            yield chunk
        if self._stream_error is not None and self._stream_error_at >= len(self._chunks):
            raise self._stream_error

    async def aclose(self) -> None:
        self.closed += 1

    @property
    def call_count(self) -> int:
        return len(self.calls)


# --- routing ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_ordinary_turn_is_answered_by_the_local_model() -> None:
    local, harness = FakeProvider("ollama"), FakeProvider("hermes")
    provider = RoutedProvider(primary=local, escalation=harness)

    response = await provider.chat(ORDINARY, tools=[{"type": "function"}])

    assert response.content == "ok"
    assert local.call_count == 1
    assert harness.call_count == 0


@pytest.mark.asyncio
async def test_the_local_model_keeps_the_native_tools() -> None:
    local = FakeProvider("ollama")
    provider = RoutedProvider(primary=local, escalation=FakeProvider("hermes"))

    assert provider.manages_own_tools is False
    await provider.chat(ORDINARY, tools=[{"type": "function", "function": {"name": "x"}}])
    assert local.tools_seen[-1] is not None


@pytest.mark.asyncio
async def test_a_harness_turn_goes_to_hermes_without_caal_tools() -> None:
    local, harness = FakeProvider("ollama"), FakeProvider("hermes", content="researched")
    provider = RoutedProvider(primary=local, escalation=harness)

    response = await provider.chat(HARNESS, tools=[{"type": "function"}])

    assert response.content == "researched"
    assert harness.call_count == 1
    assert harness.tools_seen == [None]
    assert local.call_count == 0


@pytest.mark.asyncio
async def test_a_coding_turn_that_reached_the_llm_is_escalated_not_guessed_at() -> None:
    """The coding bridge claims these upstream; this is the defence in depth."""
    local, harness = FakeProvider("ollama"), FakeProvider("hermes", content="handled")
    provider = RoutedProvider(primary=local, escalation=harness)

    response = await provider.chat(CODING)

    assert response.content == "handled"
    assert local.call_count == 0


@pytest.mark.asyncio
async def test_without_an_escalation_everything_stays_local() -> None:
    local = FakeProvider("ollama")
    provider = RoutedProvider(primary=local, escalation=None)

    await provider.chat(HARNESS)

    assert local.call_count == 1


# --- the bounded fallback -----------------------------------------------------


@pytest.mark.asyncio
async def test_a_failing_local_model_falls_back_once(caplog) -> None:
    local = FakeProvider("ollama", error=RuntimeError("connection refused to 10.0.0.64"))
    harness = FakeProvider("hermes", content="answered")
    provider = RoutedProvider(primary=local, escalation=harness)

    response = await provider.chat(ORDINARY)

    assert response.content == "answered"
    assert local.call_count == 1
    assert harness.call_count == 1
    assert "10.0.0.64" not in " ".join(record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_an_unusable_local_answer_falls_back_once() -> None:
    local = FakeProvider("ollama", content="   ")
    harness = FakeProvider("hermes", content="answered")
    provider = RoutedProvider(primary=local, escalation=harness)

    response = await provider.chat(ORDINARY)

    assert response.content == "answered"
    assert local.call_count == 1
    assert harness.call_count == 1


@pytest.mark.asyncio
async def test_a_local_answer_with_tool_calls_is_never_second_guessed() -> None:
    class ToolCalling(FakeProvider):
        async def chat(self, messages, tools=None, **kwargs):
            self.calls.append(messages)
            self.tools_seen.append(tools)
            return LLMResponse(content=None, tool_calls=[ToolCall(id="1", name="t", arguments={})])

    local, harness = ToolCalling("ollama"), FakeProvider("hermes")
    provider = RoutedProvider(primary=local, escalation=harness)

    response = await provider.chat(ORDINARY, tools=[{"type": "function"}])

    assert response.tool_calls
    assert harness.call_count == 0


@pytest.mark.asyncio
async def test_a_failing_escalation_falls_back_to_the_local_model_once() -> None:
    local = FakeProvider("ollama", content="local answer")
    harness = FakeProvider("hermes", error=RuntimeError("hermes down"))
    provider = RoutedProvider(primary=local, escalation=harness)

    response = await provider.chat(HARNESS)

    assert response.content == "local answer"
    assert harness.call_count == 1
    assert local.call_count == 1


@pytest.mark.asyncio
async def test_when_nothing_can_answer_the_reply_is_truthful() -> None:
    local = FakeProvider("ollama", error=RuntimeError("ollama is gone"))
    harness = FakeProvider("hermes", error=RuntimeError("hermes token expired"))
    provider = RoutedProvider(primary=local, escalation=harness)

    response = await provider.chat(ORDINARY)

    assert response.content
    assert "ollama is gone" not in response.content
    assert "token" not in response.content
    # Bounded: one attempt each, no retry loop.
    assert local.call_count == 1
    assert harness.call_count == 1


# --- streaming ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_ordinary_turn_streams_from_the_local_model() -> None:
    local, harness = FakeProvider("ollama"), FakeProvider("hermes")
    provider = RoutedProvider(primary=local, escalation=harness)

    chunks = [chunk async for chunk in provider.chat_stream(ORDINARY)]

    assert "".join(chunks) == "hello there"
    assert harness.stream_calls == 0


@pytest.mark.asyncio
async def test_a_stream_that_fails_before_a_word_falls_back_once() -> None:
    local = FakeProvider("ollama", chunks=[], stream_error=RuntimeError("no local model"))
    harness = FakeProvider("hermes", chunks=["from ", "hermes"])
    provider = RoutedProvider(primary=local, escalation=harness)

    chunks = [chunk async for chunk in provider.chat_stream(ORDINARY)]

    assert "".join(chunks) == "from hermes"
    assert harness.stream_calls == 1


@pytest.mark.asyncio
async def test_a_stream_that_fails_mid_sentence_is_never_replayed() -> None:
    """Restarting after speech has begun would say the first half twice."""
    local = FakeProvider("ollama", chunks=["half "], stream_error=RuntimeError("dropped"))
    local._stream_error_at = 1
    harness = FakeProvider("hermes", chunks=["whole answer"])
    provider = RoutedProvider(primary=local, escalation=harness)

    chunks = [chunk async for chunk in provider.chat_stream(ORDINARY)]

    assert "".join(chunks) == "half "
    assert harness.stream_calls == 0


@pytest.mark.asyncio
async def test_an_empty_local_stream_falls_back_once() -> None:
    local = FakeProvider("ollama", chunks=[])
    harness = FakeProvider("hermes", chunks=["from hermes"])
    provider = RoutedProvider(primary=local, escalation=harness)

    chunks = [chunk async for chunk in provider.chat_stream(ORDINARY)]

    assert "".join(chunks) == "from hermes"


# --- plumbing -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_closing_the_router_closes_both_providers() -> None:
    local, harness = FakeProvider("ollama"), FakeProvider("hermes")
    provider = RoutedProvider(primary=local, escalation=harness)

    await provider.aclose()

    assert (local.closed, harness.closed) == (1, 1)


def test_the_router_reports_the_local_model() -> None:
    provider = RoutedProvider(primary=FakeProvider("ollama"), escalation=FakeProvider("hermes"))
    assert provider.provider_name == "routed"
    assert provider.model == "ollama-model"


def test_settings_make_ollama_primary_and_hermes_the_escalation() -> None:
    from caal.llm.providers import create_provider_from_settings

    provider = create_provider_from_settings(
        {
            "llm_provider": "routed",
            "ollama_model": "qwen3:8b",
            "ollama_host": "http://host.docker.internal:11434",
            "hermes_api_key": "test-key",
            "hermes_api_url": "http://host.docker.internal:8642/v1",
        }
    )

    assert isinstance(provider, RoutedProvider)
    assert provider.model == "qwen3:8b"
    assert provider.escalation_available is True


def test_a_deployment_without_hermes_credentials_still_routes_locally(monkeypatch) -> None:
    from caal.llm.providers import create_provider_from_settings

    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    provider = create_provider_from_settings(
        {"llm_provider": "routed", "ollama_model": "qwen3:8b", "hermes_api_key": ""}
    )

    assert isinstance(provider, RoutedProvider)
    assert provider.escalation_available is False
