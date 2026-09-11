"""Connected-account knowledge never crosses into the Hermes agent runtime.

The local Ollama model reads the user own connected email and calendar itself:
it is offered the user-scoped knowledge tools and composes its spoken answer
from the bounded, speech-safe result of the one tool call it made. Hermes is a
different runtime, in a different process, with its own tool loop and its own
model -- it is never given those schemas, and it must never be given their
output either.

Three things carried that output across the boundary, and each is pinned here:

* the shared ``ToolDataCache``, which injected recent tool data into the
  context of *every* later turn, including one routed to Hermes;
* the tool and function messages of the turn itself, which the bounded local
  fallback would hand to Hermes verbatim when the local model failed
  mid-workflow;
* the answer already spoken from that data, which comes back on later turns as
  ordinary assistant transcript.

The barrier is one-way and structural. Ordinary conversation, the user own
words, and non-knowledge tool data are deliberately left alone: this must
redact private results, not amputate the context.
"""

from __future__ import annotations

import importlib
import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from caal.llm.context_barrier import (
    REDACTED_ANSWER,
    default_ledger,
    forget_private_answers,
    is_knowledge_tool,
    record_private_answer,
    sanitize_for_escalation,
)
from caal.llm.llm_node import ToolDataCache
from caal.llm.providers import LLMProvider, LLMResponse, RoutedProvider

llm_node_module = importlib.import_module("caal.llm.llm_node")

# Contents that only ever exist inside the accounts of the signed-in user.
SECRET_SUBJECT = "Zebulon Quixote merger review"
SECRET_PREVIEW = "the signed term sheet is attached"
ALIAS = "Vertex"
SPOKEN_ANSWER = "You have one unread email from Dana about " + SECRET_SUBJECT + "."

ORDINARY = [dict(role="user", content="what time is it")]
HARNESS = [dict(role="user", content="research the alberta grid and write it up")]


@pytest.fixture(autouse=True)
def clean_ledger():
    forget_private_answers()
    yield
    forget_private_answers()


def _knowledge_result() -> dict[str, Any]:
    return dict(
        status="ok",
        message=SPOKEN_ANSWER,
        data=dict(
            account=ALIAS,
            emails=[dict(sender="Dana", subject=SECRET_SUBJECT, preview=SECRET_PREVIEW)],
        ),
    )


def _call_message(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return dict(
        role="assistant",
        content="",
        tool_calls=[
            dict(
                id=call_id,
                type="function",
                function=dict(name=name, arguments=json.dumps(arguments)),
            )
        ],
    )


def _turn_after_a_knowledge_tool_call() -> list[dict[str, Any]]:
    """The message list llm_node holds while the local model composes an answer."""
    return [
        dict(role="system", content="You are JARVIS."),
        dict(role="user", content="any new email"),
        _call_message("call-1", "inbox.recent", dict(limit=5, account=ALIAS)),
        dict(role="tool", content=json.dumps(_knowledge_result()), tool_call_id="call-1"),
    ]


def _ordinary_tool_turn() -> list[dict[str, Any]]:
    """The same shape, for a tool that touches nothing private."""
    return [
        dict(role="system", content="You are JARVIS."),
        dict(role="user", content="is the office light on"),
        _call_message("call-9", "hass_get_state", dict(entity="light")),
        dict(role="tool", content=json.dumps(dict(state="on")), tool_call_id="call-9"),
    ]


def _text(messages: list[dict[str, Any]]) -> str:
    return json.dumps(messages)


class FakeProvider(LLMProvider):
    """Records exactly what a provider was handed, for both call shapes."""

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
        self._chunks = chunks if chunks is not None else ["answered"]
        self._stream_error = stream_error
        self.seen: list[list[dict]] = []

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._name + "-model"

    async def chat(self, messages, tools=None, **kwargs):
        self.seen.append(messages)
        if self._error is not None:
            raise self._error
        return LLMResponse(content=self._content, tool_calls=[])

    async def chat_stream(self, messages, tools=None, **kwargs):
        self.seen.append(messages)
        if self._stream_error is not None:
            raise self._stream_error
        for chunk in self._chunks:
            yield chunk


# --- the cache keeps nothing private past the turn it was read in ----------------------------


def test_the_cache_refuses_connected_account_tool_data() -> None:
    cache = ToolDataCache()

    cache.add("inbox.recent", _knowledge_result()["data"])

    assert cache.get_context_message() is None


def test_the_cache_still_carries_ordinary_tool_data() -> None:
    """The control: nothing about this change may quietly disable the cache."""
    cache = ToolDataCache()

    cache.add("hass_get_state", dict(state="on"))

    context = cache.get_context_message()
    assert context is not None
    assert "hass_get_state" in context and "on" in context


def test_the_cache_keeps_ordinary_data_next_to_a_refused_knowledge_read() -> None:
    cache = ToolDataCache()

    cache.add("hass_get_state", dict(state="on"))
    cache.add("schedule.next", dict(events=[dict(title=SECRET_SUBJECT)]))

    context = cache.get_context_message() or ""
    assert "hass_get_state" in context
    assert SECRET_SUBJECT not in context


def test_every_connected_account_tool_is_known_to_the_barrier() -> None:
    for name in ("inbox.recent", "inbox.search", "inbox.read_summary", "schedule.next"):
        assert is_knowledge_tool(name), name
    for name in ("hass_get_state", "memory.recall", "alarms.set", ""):
        assert not is_knowledge_tool(name), name


# --- the sanitization barrier ----------------------------------------------------------------


def test_the_barrier_drops_the_knowledge_call_and_its_result() -> None:
    sanitized = sanitize_for_escalation(_turn_after_a_knowledge_tool_call())

    body = _text(sanitized)
    assert SECRET_SUBJECT not in body
    assert SECRET_PREVIEW not in body
    assert ALIAS not in body
    assert "inbox.recent" not in body
    # The user own words and the agent prompt are not the leak; they stay.
    assert any(m["role"] == "user" and m["content"] == "any new email" for m in sanitized)
    assert sanitized[0]["role"] == "system"


def test_the_barrier_leaves_an_ordinary_tool_workflow_alone() -> None:
    """The control: a non-knowledge tool call is context Hermes may keep."""
    sanitized = sanitize_for_escalation(_ordinary_tool_turn())

    body = _text(sanitized)
    assert "hass_get_state" in body
    assert "light" in body
    assert len(sanitized) == 4


def test_the_barrier_redacts_an_answer_already_spoken_from_private_data() -> None:
    record_private_answer(SPOKEN_ANSWER)
    messages = [
        dict(role="user", content="any new email"),
        dict(role="assistant", content=SPOKEN_ANSWER),
        dict(role="user", content="now research the alberta grid"),
    ]

    sanitized = sanitize_for_escalation(messages)

    body = _text(sanitized)
    assert SECRET_SUBJECT not in body
    assert REDACTED_ANSWER in body
    assert "now research the alberta grid" in body


def test_the_barrier_drops_a_cached_knowledge_injection_it_finds_anyway() -> None:
    """Defence in depth: the cache refuses this, and the barrier would not pass it."""
    cache = ToolDataCache()
    cache.add("hass_get_state", dict(state="on"))
    injected = (cache.get_context_message() or "") + "\ninbox.recent: " + json.dumps(dict(s=1))
    messages = [dict(role="system", content=injected), *ORDINARY]

    sanitized = sanitize_for_escalation(messages)

    body = _text(sanitized)
    assert "inbox.recent" not in body
    assert "hass_get_state" in body


def test_the_ledger_never_holds_the_words_it_redacts() -> None:
    record_private_answer(SPOKEN_ANSWER)

    stored = json.dumps(list(default_ledger().fingerprints))
    assert SECRET_SUBJECT not in stored
    assert SPOKEN_ANSWER not in stored


# --- the routed provider: both ways into Hermes ----------------------------------------------


@pytest.mark.asyncio
async def test_a_local_failure_mid_tool_workflow_does_not_hand_hermes_the_result() -> None:
    local = FakeProvider("ollama", error=RuntimeError("ollama is gone"))
    harness = FakeProvider("hermes", content="answered")
    provider = RoutedProvider(primary=local, escalation=harness)

    await provider.chat(_turn_after_a_knowledge_tool_call())

    assert harness.seen, "the fallback did happen"
    body = _text(harness.seen[0])
    assert SECRET_SUBJECT not in body and SECRET_PREVIEW not in body and ALIAS not in body


@pytest.mark.asyncio
async def test_the_local_model_still_receives_the_result_it_must_answer_from() -> None:
    local = FakeProvider("ollama", content="answered")
    provider = RoutedProvider(primary=local, escalation=FakeProvider("hermes"))

    await provider.chat(_turn_after_a_knowledge_tool_call())

    assert SECRET_SUBJECT in _text(local.seen[0])


@pytest.mark.asyncio
async def test_a_later_harness_turn_starts_behind_the_barrier() -> None:
    record_private_answer(SPOKEN_ANSWER)
    local, harness = FakeProvider("ollama"), FakeProvider("hermes", content="researched")
    provider = RoutedProvider(primary=local, escalation=harness)
    messages = [
        *_turn_after_a_knowledge_tool_call(),
        dict(role="assistant", content=SPOKEN_ANSWER),
        *HARNESS,
    ]

    await provider.chat(messages)

    assert not local.seen, "a harness turn is Hermes own by routing"
    body = _text(harness.seen[0])
    assert SECRET_SUBJECT not in body and ALIAS not in body
    assert "research the alberta grid" in body


@pytest.mark.asyncio
async def test_the_stream_fallback_sanitizes_too() -> None:
    local = FakeProvider("ollama", chunks=[], stream_error=RuntimeError("no local model"))
    harness = FakeProvider("hermes", chunks=["from hermes"])
    provider = RoutedProvider(primary=local, escalation=harness)

    spoken = "".join(
        [chunk async for chunk in provider.chat_stream(_turn_after_a_knowledge_tool_call())]
    )

    assert spoken == "from hermes"
    assert SECRET_SUBJECT not in _text(harness.seen[-1])
    assert SECRET_SUBJECT in _text(local.seen[0]), "the local model keeps its own tool result"


@pytest.mark.asyncio
async def test_the_barrier_costs_an_ordinary_escalation_nothing() -> None:
    """The control: normal chat context reaches Hermes intact."""
    local = FakeProvider("ollama", error=RuntimeError("down"))
    harness = FakeProvider("hermes", content="answered")
    provider = RoutedProvider(primary=local, escalation=harness)
    messages = [
        dict(role="system", content="You are JARVIS."),
        dict(role="user", content="what did we decide about the trip"),
        dict(role="assistant", content="You decided on the Tuesday flight."),
        dict(role="user", content="remind me why"),
    ]

    await provider.chat(messages)

    assert harness.seen[0] == messages


# --- the llm node ------------------------------------------------------------------------------


class ChatMessage:
    """The shape llm_node reads out of a LiveKit chat context."""

    def __init__(self, role: str, text: str) -> None:
        self.role = role
        self.text_content = text


class SelfToolingProvider(FakeProvider):
    """A provider that runs its own tool loop, as Hermes does."""

    @property
    def manages_own_tools(self) -> bool:
        return True


class ToolCallingProvider(LLMProvider):
    """A local model that calls one knowledge tool, then speaks from its result."""

    def __init__(self) -> None:
        self.followup: list[dict] = []

    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def model(self) -> str:
        return "stub"

    async def chat(self, messages, tools=None, **_: Any):
        call = SimpleNamespace(id="call-1", name="inbox.recent", arguments=dict(limit=5))
        return LLMResponse(content=None, tool_calls=[call])

    async def chat_stream(self, messages, tools=None, **_: Any):
        self.followup = list(messages)
        yield SPOKEN_ANSWER


def _agent_with_knowledge_tool(result: dict[str, Any]):
    registry = importlib.import_module("caal.tools.registry")
    catalog = registry.ToolRegistry()
    for tool in registry.create_default_registry().list():
        if tool.name == "inbox.recent":
            tool = registry.ToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                category=tool.category,
                handler=lambda **_: result,
                user_scoped=False,
            )
        catalog.register(tool)
    return SimpleNamespace(
        _native_tool_registry=catalog,
        _llm_tools_cache=[dict(type="function", function=dict(name="inbox.recent"))],
    )


@pytest.mark.asyncio
async def test_a_local_knowledge_answer_is_redacted_on_the_next_hermes_turn(caplog) -> None:
    """The whole path: local tool call, spoken answer, then a turn Hermes gets."""
    provider = ToolCallingProvider()
    agent = _agent_with_knowledge_tool(_knowledge_result())
    cache = ToolDataCache()
    chat_ctx = SimpleNamespace(items=[])

    with caplog.at_level(logging.DEBUG):
        spoken = "".join(
            [
                chunk
                async for chunk in llm_node_module.llm_node(
                    agent, chat_ctx, provider=provider, tool_data_cache=cache
                )
            ]
        )

    assert SECRET_SUBJECT in spoken, "the local model answered from its own tool result"
    assert SECRET_SUBJECT in _text(provider.followup), "and was given the result to answer from"
    assert SECRET_SUBJECT not in caplog.text
    # Nothing of it survives into the context of a later turn...
    assert cache.get_context_message() is None
    # ...and the answer itself is redacted if the transcript reaches Hermes.
    later = sanitize_for_escalation(
        [dict(role="assistant", content=spoken), dict(role="user", content="research that")]
    )
    assert SECRET_SUBJECT not in _text(later)
    assert REDACTED_ANSWER in _text(later)


@pytest.mark.asyncio
async def test_the_llm_node_sanitizes_for_a_provider_that_runs_its_own_tools() -> None:
    """A deployment pointed straight at Hermes gets the same barrier."""
    record_private_answer(SPOKEN_ANSWER)
    harness = SelfToolingProvider("hermes", chunks=["ok"])
    chat_ctx = SimpleNamespace(
        items=[
            ChatMessage("system", "You are JARVIS."),
            ChatMessage("assistant", SPOKEN_ANSWER),
            ChatMessage("user", "research the alberta grid"),
        ]
    )

    async for _ in llm_node_module.llm_node(SimpleNamespace(), chat_ctx, provider=harness):
        pass

    body = _text(harness.seen[0])
    assert SECRET_SUBJECT not in body
    assert "research the alberta grid" in body


# --- the deterministic route -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_deterministic_route_marks_what_it_speaks_as_private() -> None:
    from caal.knowledge_router import KnowledgeTurnHandler

    spoken: list[str] = []

    class Session:
        async def say(self, text: str) -> None:
            spoken.append(text)

    handler = KnowledgeTurnHandler(scope=SimpleNamespace(identity_configured=False))
    await handler._speak(Session(), SPOKEN_ANSWER, "inbox.recent")

    assert spoken == [SPOKEN_ANSWER]
    later = sanitize_for_escalation([dict(role="assistant", content=SPOKEN_ANSWER)])
    assert SECRET_SUBJECT not in _text(later)
    assert REDACTED_ANSWER in _text(later)
