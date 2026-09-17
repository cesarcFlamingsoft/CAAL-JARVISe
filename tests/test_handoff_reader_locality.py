"""The handoff reader's local-only promise, enforced rather than documented.

`build_handoff_reader`'s docstring says "always the local model, for the same
reason the work router is". An independent probe passed it a spy declaring
`provider_name="hermes"` and `base_url="https://example.invalid"`, and the
builder wired it up: the spy received a classifier call and the reader answered
`direct`. `work_router_provider` only *unwraps* wrappers; it does not check
what it unwrapped to.

So this file pins the enforcement, with the same resolver the cross-language
translator already uses (`expansion_runtime.local_provider` plus
`validated_endpoint`, which is `local_ollama`'s narrow local-only allowance):

* a non-local provider, or a local one on an endpoint that is not an approved
  local address, builds **no reader at all** -- and is refused *before* any
  classifier call, so the turn never leaves this machine;
* the routed provider is unwrapped to its local primary and the router itself
  is never called;
* a directly configured local provider is **not** turned off: the real Spanish
  path keeps working, with `tools=None`, `think=False`, a bounded
  `num_predict`, and the runtime's existing timeout;
* a company-private session is never handed off at all, whatever the reader says.

Nothing here reaches a network or a model.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from caal import company_privacy  # noqa: E402
from caal.handoff_semantics import MAX_READER_REPLY_TOKENS  # noqa: E402
from caal.handoff_intent import HandoffIntent  # noqa: E402

SPANISH_REQUEST = "sigamos por teléfono"
LABEL = '{"handoff":"direct","reply_language":"es","language_switch":false}'


def _voice_agent():
    return importlib.import_module("voice_agent")


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content


class _LocalSpy:
    """The shape of ``OllamaProvider``: local name, approved endpoint, declared options."""

    provider_name = "ollama"
    supported_chat_options = frozenset({"think", "num_predict"})

    def __init__(self, base_url: str | None = "http://127.0.0.1:11434") -> None:
        self.base_url = base_url
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools, "kwargs": dict(kwargs)})
        return _Response(LABEL)


class _CloudSpy:
    """What the probe used: Hermes, by its own declaration. Never called."""

    provider_name = "hermes"
    base_url = "https://example.invalid"
    supported_chat_options = frozenset({"think"})

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools, "kwargs": dict(kwargs)})
        return _Response(LABEL)


class _UnnamedSpy:
    """A provider that says nothing about itself. Not evidence of locality."""

    base_url = "http://127.0.0.1:11434"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools, "kwargs": dict(kwargs)})
        return _Response(LABEL)


class _Routed:
    """``RoutedProvider``: may escalate, so it is never the thing that is called."""

    provider_name = "routed"

    def __init__(self, primary: Any) -> None:
        self.primary = primary
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools, "kwargs": dict(kwargs)})
        raise AssertionError("a handoff reading was sent through the routed provider")


class _CAALLLM:
    """The LiveKit wrapper production actually passes (`caal_llm.provider_instance`)."""

    def __init__(self, provider_instance: Any) -> None:
        self.provider_instance = provider_instance

    async def chat(self, *args, **kwargs):
        raise AssertionError("a handoff reading was sent through the LiveKit wrapper")


# --- refusal, before any classifier call --------------------------------------------------


@pytest.mark.asyncio
async def test_a_hermes_provider_builds_no_reader_and_is_never_called():
    spy = _CloudSpy()
    reader = _voice_agent().build_handoff_reader({}, provider=spy)
    assert reader is None
    assert spy.calls == [], "a turn was sent to a non-local provider"


@pytest.mark.asyncio
async def test_a_hermes_provider_behind_the_livekit_wrapper_is_refused_too():
    spy = _CloudSpy()
    assert _voice_agent().build_handoff_reader({}, provider=_CAALLLM(spy)) is None
    assert spy.calls == []


@pytest.mark.asyncio
async def test_a_provider_that_only_forgot_to_name_itself_is_refused():
    """A missing ``provider_name`` is not a local provider; it is an unknown one."""
    spy = _UnnamedSpy()
    assert _voice_agent().build_handoff_reader({}, provider=spy) is None
    assert spy.calls == []


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://example.invalid",
        "http://8.8.8.8:11434",
        "http://user:pw@127.0.0.1:11434",
        "http://127.0.0.1:11434/v1/chat",
        "http://169.254.169.254:80",
        "not a url",
    ],
)
def test_a_local_name_on_an_unapproved_endpoint_builds_no_reader(endpoint):
    spy = _LocalSpy(base_url=endpoint)
    assert _voice_agent().build_handoff_reader({}, provider=spy) is None
    assert spy.calls == []


@pytest.mark.asyncio
async def test_a_refused_provider_leaves_the_controller_with_no_semantic_reader():
    """The offline English net is what remains; nothing is sent anywhere."""
    voice_agent = _voice_agent()
    spy = _CloudSpy()
    reader = voice_agent.build_handoff_reader({}, provider=spy)
    assert reader is None
    assert spy.calls == []


# --- the local path is not turned off ------------------------------------------------------


@pytest.mark.asyncio
async def test_a_direct_local_provider_is_accepted_and_reads_the_spanish_request():
    spy = _LocalSpy()
    reader = _voice_agent().build_handoff_reader({}, provider=spy)
    assert reader is not None and reader.enabled is True
    reading = await reader.read(SPANISH_REQUEST)
    assert reading.intent is HandoffIntent.DIRECT
    assert len(spy.calls) == 1


@pytest.mark.asyncio
async def test_the_routed_provider_is_unwrapped_to_its_local_primary():
    local = _LocalSpy()
    routed = _Routed(primary=local)
    reader = _voice_agent().build_handoff_reader({}, provider=_CAALLLM(routed))
    assert reader is not None and reader.enabled is True
    await reader.read(SPANISH_REQUEST)
    assert routed.calls == []
    assert len(local.calls) == 1


@pytest.mark.asyncio
async def test_the_reading_is_bounded_at_the_api_not_only_by_the_clock():
    spy = _LocalSpy()
    reader = _voice_agent().build_handoff_reader({}, provider=spy)
    assert reader is not None
    await reader.read(SPANISH_REQUEST)
    call = spy.calls[0]
    assert call["tools"] is None, "the reader must never be offered tools"
    assert call["kwargs"]["think"] is False
    assert call["kwargs"]["num_predict"] == MAX_READER_REPLY_TOKENS


@pytest.mark.asyncio
async def test_the_runtime_timeout_is_still_the_one_that_applies():
    class _Slow(_LocalSpy):
        async def chat(self, messages, tools=None, **kwargs):
            await asyncio.sleep(5)
            raise AssertionError("unreachable")

    reader = _voice_agent().build_handoff_reader(
        {"handoff_reader_timeout_seconds": 0.05}, provider=_Slow()
    )
    assert reader is not None
    reading = await reader.read(SPANISH_REQUEST)
    assert reading.intent is HandoffIntent.NONE


def test_the_operator_switch_still_turns_the_reader_off():
    spy = _LocalSpy()
    assert (
        _voice_agent().build_handoff_reader({"handoff_reader_enabled": False}, provider=spy) is None
    )
    assert spy.calls == []


def test_no_provider_at_all_is_still_no_reader():
    assert _voice_agent().build_handoff_reader({}, provider=None) is None


# --- the private session is never handed off, reader or no reader -------------------------


def test_the_handoff_route_is_blocked_in_a_company_private_session():
    assert "phone_handoff" in company_privacy.PRIVATE_SESSION_BLOCKED_ROUTES


def test_the_real_local_provider_declares_the_options_the_reader_asks_for():
    from caal.company.query_expansion import supported_chat_options
    from caal.llm.providers.ollama_provider import OllamaProvider

    assert {"think", "num_predict"} <= set(supported_chat_options(OllamaProvider))
