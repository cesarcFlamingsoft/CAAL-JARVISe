"""A company-private session does not leave this machine. Proved at the exits.

The barrier that redacts a *tool result* on the way to the escalation is not a
confidentiality control for this library, because the sensitive thing is the
question. "What does our contract with FIXTURE Corp say about severance for
FIXTURE Person" is already the leak, and the router reads that question and
picks a destination before any company tool has run. The review reproduced
exactly that:

    initial_sensitive_routing Destination.HARNESS
    initial_sensitive_words_retained True

So the tests here are about the exits, not the redaction. Every one of them
puts a **capturing fake external runtime** on the other side -- a provider that
records anything it is handed and fails the test if it is handed anything at
all -- and then drives the real :class:`RoutedProvider`, the real
:func:`classify_request`, the real durable-work admission and the real tool
dispatcher. A test that only inspects a sanitized message list would pass
whether or not the message list was ever sent.

The fixtures are unmistakably synthetic: FIXTURE people, FIXTURE companies and
a canary string that exists nowhere else.
"""

from __future__ import annotations

import asyncio

import pytest

from caal import company_privacy
from caal.company_privacy import LOCAL_ONLY_NO_ANSWER
from caal.llm.providers.base import LLMResponse
from caal.llm.providers.routed_provider import NO_MODEL_AVAILABLE_REPLY, RoutedProvider
from caal.model_routing import Destination, classify_request

OWNER = "usr_" + "a1" * 12
STRANGER = "usr_" + "d4" * 12
CANARY = "FIXTURE_SECRET_zzqx7731"

# The review's own reproduction, verbatim in shape: a first turn that the
# harness classifier claims, carrying a person and a salary.
SENSITIVE_FIRST_TURN = (
    "Research and analyze the confidential company contract for FIXTURE Person "
    f"with salary {CANARY}"
)


# --- capturing fakes ------------------------------------------------------------------------------


class CapturingProvider:
    """An external runtime that refuses to be used and remembers being asked."""

    def __init__(self, name: str = "capturing-escalation") -> None:
        self.name = name
        self.payloads: list[object] = []

    @property
    def provider_name(self) -> str:
        return self.name

    @property
    def model(self) -> str:
        return "fixture-external-model"

    @property
    def supports_think(self) -> bool:
        return False

    @property
    def manages_own_tools(self) -> bool:
        return True

    async def reachable(self) -> bool:
        return True

    async def chat(self, messages, tools=None, **kwargs):
        self.payloads.append(messages)
        raise AssertionError("the external runtime was handed a company-private turn")

    async def chat_stream(self, messages, tools=None, **kwargs):
        self.payloads.append(messages)
        raise AssertionError("the external runtime was handed a company-private turn")
        yield ""  # pragma: no cover

    def parse_tool_arguments(self, arguments):  # pragma: no cover - never reached
        return {}

    def format_tool_result(self, content, tool_call_id, tool_name):  # pragma: no cover
        return {}

    def format_tool_call_message(self, content, tool_calls):  # pragma: no cover
        return {}

    async def aclose(self) -> None:
        return None


class LocalProvider:
    """A stand-in for the local Ollama model. Optionally broken."""

    def __init__(self, *, working: bool = True) -> None:
        self.working = working
        self.seen: list[object] = []

    @property
    def provider_name(self) -> str:
        return "fixture-local"

    @property
    def model(self) -> str:
        return "fixture-local-model"

    @property
    def supports_think(self) -> bool:
        return True

    @property
    def manages_own_tools(self) -> bool:
        return False

    async def reachable(self) -> bool:
        return self.working

    async def chat(self, messages, tools=None, **kwargs):
        self.seen.append(messages)
        if not self.working:
            raise RuntimeError("the local model is unreachable")
        return LLMResponse(content="FIXTURE local answer.", tool_calls=[])

    async def chat_stream(self, messages, tools=None, **kwargs):
        self.seen.append(messages)
        if not self.working:
            raise RuntimeError("the local model is unreachable")
        yield "FIXTURE local answer."

    def parse_tool_arguments(self, arguments):
        return dict(arguments or {})

    def format_tool_result(self, content, tool_call_id, tool_name):
        return {"role": "tool", "content": content, "tool_call_id": tool_call_id}

    def format_tool_call_message(self, content, tool_calls):
        return {"role": "assistant", "content": content, "tool_calls": tool_calls}

    async def aclose(self) -> None:
        return None


class FakeScope:
    def __init__(self, user_id: str | None) -> None:
        self.user_id = user_id
        self.identity_configured = True
        self.memory_available = user_id is not None


class FakeAgent:
    """Just enough agent for the privacy decision and the tool policy."""

    def __init__(self, user_id: str | None = OWNER) -> None:
        self._user_scope = FakeScope(user_id)


def _messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are JARVIS."},
        {"role": "user", "content": text},
    ]


@pytest.fixture(autouse=True)
def clean_privacy_state(monkeypatch):
    company_privacy.reset()
    monkeypatch.delenv("CAAL_COMPANY_PRIVATE_MODE", raising=False)
    yield
    company_privacy.reset()


@pytest.fixture()
def owner_library(monkeypatch):
    """A configured company library owned by OWNER, without opening one."""
    from caal.company import runtime

    class _Config:
        owner_user_id = OWNER

    monkeypatch.setattr(runtime, "get_config", lambda: _Config())
    return _Config()


@pytest.fixture()
def private_session(owner_library):
    """An engaged company-private session, entered the way the UI enters one."""
    agent = FakeAgent(OWNER)
    company_privacy.bind_session(agent, requested=True)
    company_privacy.begin_turn(agent, "hello")
    assert company_privacy.is_local_only() is True
    return agent


# --- the session decision -------------------------------------------------------------------------


def test_an_entered_company_session_is_local_only_from_its_first_word(owner_library):
    agent = FakeAgent(OWNER)
    company_privacy.bind_session(agent, requested=True)
    state = company_privacy.begin_turn(agent, "what is the weather")
    assert state.engaged is True
    assert state.reason == "explicit_entry"


def test_a_session_that_is_not_the_owners_keeps_its_ordinary_capabilities(owner_library):
    agent = FakeAgent(STRANGER)
    company_privacy.begin_turn(agent, SENSITIVE_FIRST_TURN)
    assert company_privacy.is_local_only() is False
    assert classify_request(SENSITIVE_FIRST_TURN).destination is Destination.HARNESS


def test_engagement_is_sticky_for_the_rest_of_the_session(owner_library):
    agent = FakeAgent(OWNER)
    company_privacy.bind_session(agent, requested=True)

    company_privacy.begin_turn(agent, "what does our contract say about notice periods")
    assert company_privacy.is_local_only() is True

    # A follow-up with nothing company-shaped in it stays private.
    company_privacy.begin_turn(agent, "and the second one")
    assert company_privacy.is_local_only() is True
    # ...and so does one that is small talk.
    company_privacy.begin_turn(agent, "thanks")
    assert company_privacy.is_local_only() is True


def test_using_a_company_tool_engages_the_session_whatever_the_wording_was(owner_library):
    """A backstop. The tools are unreachable outside a private session anyway."""
    agent = FakeAgent(OWNER)
    company_privacy.begin_turn(agent, "look up FIXTURE Person")
    assert company_privacy.is_local_only() is False

    company_privacy.note_company_tool_use()
    assert company_privacy.is_local_only() is True

    company_privacy.begin_turn(agent, "what about the other one")
    assert company_privacy.is_local_only() is True


def test_with_the_mode_off_the_company_tools_are_not_offered_at_all(owner_library, monkeypatch):
    monkeypatch.setenv("CAAL_COMPANY_PRIVATE_MODE", "off")
    assert company_privacy.company_tools_offered(OWNER) is False
    agent = FakeAgent(OWNER)
    company_privacy.begin_turn(agent, SENSITIVE_FIRST_TURN)
    assert company_privacy.is_local_only() is False


def test_the_company_tools_are_only_offered_to_the_provisioned_owner(private_session):
    assert company_privacy.company_tools_offered(OWNER) is True
    assert company_privacy.company_tools_offered(STRANGER) is False
    assert company_privacy.company_tools_offered(None) is False


# --- the routing exit ------------------------------------------------------------------------------


def test_the_sensitive_first_turn_never_reaches_the_external_runtime(private_session):
    """The review's reproduction. It used to be Destination.HARNESS."""
    assert classify_request(SENSITIVE_FIRST_TURN).destination is Destination.LOCAL

    local, external = LocalProvider(), CapturingProvider()
    routed = RoutedProvider(primary=local, escalation=external)

    response = asyncio.run(routed.chat(_messages(SENSITIVE_FIRST_TURN)))
    assert response.content == "FIXTURE local answer."
    assert external.payloads == []
    assert local.seen


def test_a_local_model_failure_is_a_refusal_not_an_escalation(private_session):
    local, external = LocalProvider(working=False), CapturingProvider()
    routed = RoutedProvider(primary=local, escalation=external)

    response = asyncio.run(routed.chat(_messages(SENSITIVE_FIRST_TURN)))
    assert response.content == LOCAL_ONLY_NO_ANSWER
    assert response.content != NO_MODEL_AVAILABLE_REPLY
    assert external.payloads == []


def test_a_local_stream_failure_is_a_refusal_not_an_escalation(private_session):
    local, external = LocalProvider(working=False), CapturingProvider()
    routed = RoutedProvider(primary=local, escalation=external)

    async def _drain():
        return [chunk async for chunk in routed.chat_stream(_messages(SENSITIVE_FIRST_TURN))]

    assert asyncio.run(_drain()) == [LOCAL_ONLY_NO_ANSWER]
    assert external.payloads == []


@pytest.mark.parametrize(
    "turn",
    [
        SENSITIVE_FIRST_TURN,
        # A follow-up that quotes the clause the library just returned.
        'Look into "termination requires thirty days written notice" and cross-reference it',
        # An employee-only initial query, with an alias rather than a full name.
        "Research everything we have on file for FIXPERSON-A",
        "Dig into the FIXTURE Corp master services agreement step by step",
    ],
)
def test_no_wording_of_a_turn_can_find_the_external_runtime(private_session, turn):
    assert classify_request(turn).destination is Destination.LOCAL

    local, external = LocalProvider(), CapturingProvider()
    routed = RoutedProvider(primary=local, escalation=external)
    asyncio.run(routed.chat(_messages(turn)))
    assert external.payloads == []


def test_the_same_turns_do_escalate_in_an_ordinary_session(owner_library):
    """The capability is not removed from the deployment, only from this session."""
    company_privacy.begin_turn(FakeAgent(STRANGER), SENSITIVE_FIRST_TURN)
    local, external = LocalProvider(), CapturingProvider()
    routed = RoutedProvider(primary=local, escalation=external)

    # The capturing fake refuses, so the bounded fallback lands on the local
    # model -- but it was *asked*, which is the whole point of this test.
    asyncio.run(routed.chat(_messages(SENSITIVE_FIRST_TURN)))
    assert external.payloads, "an ordinary session still reaches the agent harness"
    carried = str(external.payloads[0])
    assert CANARY in carried


# --- the durable-work exit -------------------------------------------------------------------------


def test_a_private_turn_is_never_admitted_to_durable_work(private_session):
    """The durable worker is another process and composes on the escalation provider."""
    from caal.background_task_session import BackgroundTaskBridge

    enqueued: list[str] = []

    async def _execute(task):  # pragma: no cover - never reached
        enqueued.append(task.request)
        return ""

    bridge = BackgroundTaskBridge(execute=_execute, session_key="fixture-room")
    outcome = asyncio.run(bridge.process_turn(SENSITIVE_FIRST_TURN, session=None))
    assert outcome.consumed is False
    assert outcome.scheduled is False
    assert enqueued == []


def test_coding_delegation_never_claims_a_private_turn(private_session):
    # The coding admission in both the session bridge and the durable worker is
    # `classify_request(...).destination is Destination.CODING`.
    coding_shaped = "Refactor the python module that parses our employment contracts"
    assert classify_request(coding_shaped).destination is Destination.LOCAL


# --- the tool exit ----------------------------------------------------------------------------------


def _available(agent, name: str) -> bool:
    from caal.llm.llm_node import _tool_available

    return _tool_available(agent, name)


def _bind_registry(agent):
    from caal.tools import create_default_registry

    agent._native_tool_registry = create_default_registry()
    return agent


@pytest.mark.parametrize(
    "name",
    [
        "email.send",
        "email.search",
        "calendar.create_event",
        "inbox.search",
        "schedule.next",
        "network.lookup",
        "network.speedtest",
        "hass_assist",
        "some_external_mcp__fetch_url",
        "n8n_send_report",
    ],
)
def test_a_private_session_is_offered_no_tool_that_can_send_anything(private_session, name):
    agent = _bind_registry(private_session)
    assert _available(agent, name) is False


@pytest.mark.parametrize("name", ["company.search", "company.read"])
def test_a_private_session_keeps_the_two_company_reads(private_session, name):
    agent = _bind_registry(private_session)
    assert _available(agent, name) is True


@pytest.mark.parametrize("name", ["memory.remember", "reminders.create", "alarms.set"])
def test_a_private_session_no_longer_keeps_a_tool_that_persists_or_defers(
    private_session, name
):
    """These used to be allowed on the reasoning that they stay on this machine.

    They do not. A reminder with ``delivery: ["telegram"]`` is a **deferred
    send**, made later by the worker process from text the model chose, and a
    probe watched a canary reach a captured Telegram sink that way. Memory and
    reminders are also **persistence**, and the confidentiality boundary is per
    session: a clause written to memory in a private session is read back in an
    ordinary one, where the escalation path is open. See
    ``tests/test_company_private_surface.py`` and
    ``caal.company_privacy.PRIVATE_SESSION_TOOLS``.
    """
    agent = _bind_registry(private_session)
    assert _available(agent, name) is False


def test_an_injected_instruction_cannot_reach_a_real_sending_tool(private_session):
    """A passage that says "email this to FIXTURE" names a tool that is not there."""
    import importlib

    node_module = importlib.import_module("caal.llm.llm_node")
    agent = _bind_registry(private_session)
    result = asyncio.run(
        node_module._execute_single_tool(
            agent, "email.send", {"to": "fixture@example.invalid", "body": CANARY}
        )
    )
    assert result["status"] == "unsupported_tool"


def test_an_ordinary_session_keeps_its_ordinary_tools(owner_library):
    company_privacy.begin_turn(FakeAgent(STRANGER), "what is in my inbox")
    agent = _bind_registry(FakeAgent(STRANGER))
    assert _available(agent, "inbox.search") is True
    # ...and does not get the company tools.
    assert _available(agent, "company.search") is False
