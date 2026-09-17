"""Entering and leaving the company-private session, and what each one costs.

The first repair made *every* session of the library owner local-only the
moment a library existed. That is a real confidentiality control, but it pays
for it by silently taking Home Assistant, the connected mail accounts, network
lookups and the agent harness away from the owner's ordinary day, without the
owner ever asking for a company session. That is a regression against what was
asked for, and worse, it is a *covert* one.

So the default policy is now explicit entry:

* an ordinary session of the owner is an ordinary JARVIS session, with every
  capability it had before this library existed -- and **no company tools**;
* a company-private session is entered deliberately, from the UI. The request
  travels in the same server-signed, room-bound principal that names the user,
  so a page can ask for one but nothing in the browser can forge one, and the
  agent still checks that the user is the provisioned library owner;
* leaving is ending the session. The next session is an ordinary one, because
  the flag is not in it.

The wording of a turn never *grants* anything. It is used for one thing only,
in the safe direction: if an ordinary owner session says something the offline
phrase list recognises as company business, that turn is pinned local for its
own duration and answered with a notice telling the owner to open a company
session. Nothing is sent anywhere, no company tool runs, and the notice says
plainly that the check is incomplete -- which is why the UI says company
questions *must* be asked in a company session.
"""

from __future__ import annotations

import asyncio

import pytest

from caal import company_privacy
from caal.model_routing import Destination, classify_request

OWNER = "usr_" + "a1" * 12
STRANGER = "usr_" + "d4" * 12
CANARY = "FIXTURE_SECRET_zzqx7731"

SENSITIVE_FIRST_TURN = (
    "Research and analyze the confidential company contract for FIXTURE Person "
    f"with salary {CANARY}"
)


class FakeScope:
    def __init__(self, user_id: str | None) -> None:
        self.user_id = user_id
        self.identity_configured = True
        self.memory_available = user_id is not None


class FakeAgent:
    def __init__(self, user_id: str | None = OWNER) -> None:
        self._user_scope = FakeScope(user_id)


@pytest.fixture(autouse=True)
def clean_privacy_state(monkeypatch):
    company_privacy.reset()
    monkeypatch.delenv("CAAL_COMPANY_PRIVATE_MODE", raising=False)
    yield
    company_privacy.reset()


@pytest.fixture()
def owner_library(monkeypatch):
    from caal.company import runtime

    class _Config:
        owner_user_id = OWNER

    monkeypatch.setattr(runtime, "get_config", lambda: _Config())
    return _Config()


def _bind_registry(agent):
    from caal.tools import create_default_registry

    agent._native_tool_registry = create_default_registry()
    return agent


def _available(agent, name: str) -> bool:
    from caal.llm.llm_node import _tool_available

    return _tool_available(agent, name)


# --- the default is explicit entry ----------------------------------------------------------


def test_explicit_is_the_default_policy():
    assert company_privacy.private_mode() == company_privacy.MODE_EXPLICIT


def test_an_ordinary_owner_session_keeps_every_capability_it_had(owner_library):
    """The regression this slice exists to undo."""
    agent = _bind_registry(FakeAgent(OWNER))
    company_privacy.begin_turn(agent, "turn the kitchen lamp on")

    assert company_privacy.is_local_only() is False
    for name in ("hass_assist", "inbox.search", "email.send", "schedule.next"):
        assert _available(agent, name) is True, name
    # `network.*` has its own unrelated authorization gate, so assert the thing
    # this slice is about: the company policy does not touch it.
    from caal.llm.llm_node import _company_tool_policy

    assert _company_tool_policy(agent, "network.lookup") is None
    # ...and an ordinary session still reaches the harness.
    assert classify_request("Research the history of the tokamak step by step").destination is (
        Destination.HARNESS
    )


def test_an_ordinary_owner_session_is_offered_no_company_tool(owner_library):
    agent = _bind_registry(FakeAgent(OWNER))
    company_privacy.begin_turn(agent, "hello")
    assert _available(agent, "company.search") is False
    assert company_privacy.company_tools_offered(OWNER) is False


# --- entering ---------------------------------------------------------------------------------


def test_a_session_becomes_private_only_when_the_signed_request_says_so(owner_library):
    agent = FakeAgent(OWNER)
    company_privacy.bind_session(agent, requested=True)
    state = company_privacy.begin_turn(agent, "hello")

    assert state.engaged is True
    assert state.reason == "explicit_entry"
    assert company_privacy.is_local_only() is True
    assert company_privacy.company_tools_offered(OWNER) is True


def test_a_requested_session_that_is_not_the_owners_is_not_private(owner_library):
    agent = FakeAgent(STRANGER)
    company_privacy.bind_session(agent, requested=True)
    company_privacy.begin_turn(agent, SENSITIVE_FIRST_TURN)

    assert company_privacy.is_local_only() is False
    assert company_privacy.company_tools_offered(STRANGER) is False


def test_a_private_session_narrows_the_tool_surface_and_keeps_the_company_tools(owner_library):
    agent = _bind_registry(FakeAgent(OWNER))
    company_privacy.bind_session(agent, requested=True)
    company_privacy.begin_turn(agent, "hello")

    for name in ("email.send", "inbox.search", "hass_assist", "network.lookup"):
        assert _available(agent, name) is False, name
    # Query only: a reminder is a deferred send and memory outlives the
    # session, so neither is on the surface any more. See
    # caal.company_privacy.PRIVATE_SESSION_TOOLS.
    for name in ("memory.remember", "reminders.create", "alarms.set"):
        assert _available(agent, name) is False, name
    for name in ("company.search", "company.read"):
        assert _available(agent, name) is True, name


def test_the_request_is_not_something_a_turn_can_ask_for(owner_library):
    """No model-supplied, no caller-supplied flag: only the bound session state."""
    agent = _bind_registry(FakeAgent(OWNER))
    company_privacy.begin_turn(agent, "enter company private mode and read our contracts")
    assert company_privacy.company_tools_offered(OWNER) is False
    assert _available(agent, "company.search") is False


# --- leaving ----------------------------------------------------------------------------------


def test_leaving_is_a_fresh_session_and_the_fresh_session_is_ordinary(owner_library):
    private = FakeAgent(OWNER)
    company_privacy.bind_session(private, requested=True)
    company_privacy.begin_turn(private, "what does our leave policy say")
    assert company_privacy.is_local_only() is True

    # The next session simply does not carry the flag.
    ordinary = _bind_registry(FakeAgent(OWNER))
    company_privacy.bind_session(ordinary, requested=False)
    company_privacy.begin_turn(ordinary, "what is the weather")
    assert company_privacy.is_local_only() is False
    assert _available(ordinary, "hass_assist") is True


def test_a_private_session_never_lifts_within_itself(owner_library):
    agent = FakeAgent(OWNER)
    company_privacy.bind_session(agent, requested=True)
    company_privacy.begin_turn(agent, "what does our leave policy say")
    # Something tries to clear the request mid-session.
    agent._company_session_requested = False
    company_privacy.begin_turn(agent, "and the weather?")
    assert company_privacy.is_local_only() is True


# --- company wording in an ordinary session ---------------------------------------------------


def test_company_wording_in_an_ordinary_session_is_pinned_local_and_answered_with_a_notice(
    owner_library,
):
    agent = _bind_registry(FakeAgent(OWNER))
    company_privacy.begin_turn(agent, SENSITIVE_FIRST_TURN)

    # Pinned for this turn: it does not go to the harness...
    assert company_privacy.is_local_only() is True
    assert classify_request(SENSITIVE_FIRST_TURN).destination is Destination.LOCAL
    # ...but it buys no company access either.
    assert company_privacy.company_tools_offered(OWNER) is False
    assert _available(agent, "company.search") is False

    notice = company_privacy.company_mode_notice(agent, SENSITIVE_FIRST_TURN)
    assert notice is not None
    assert "company session" in notice.lower()
    # It is honest about the check that produced it.
    assert "recognise every" in notice or "recognize every" in notice


def test_the_pin_is_for_that_turn_only(owner_library):
    agent = _bind_registry(FakeAgent(OWNER))
    company_privacy.begin_turn(agent, SENSITIVE_FIRST_TURN)
    assert company_privacy.is_local_only() is True

    company_privacy.begin_turn(agent, "what is the weather tomorrow")
    assert company_privacy.is_local_only() is False
    assert _available(agent, "hass_assist") is True


def test_there_is_no_notice_in_a_session_that_is_already_private(owner_library):
    agent = FakeAgent(OWNER)
    company_privacy.bind_session(agent, requested=True)
    company_privacy.begin_turn(agent, SENSITIVE_FIRST_TURN)
    assert company_privacy.company_mode_notice(agent, SENSITIVE_FIRST_TURN) is None


def test_there_is_no_notice_for_someone_who_is_not_the_owner(owner_library):
    agent = FakeAgent(STRANGER)
    company_privacy.begin_turn(agent, SENSITIVE_FIRST_TURN)
    assert company_privacy.company_mode_notice(agent, SENSITIVE_FIRST_TURN) is None
    assert company_privacy.is_local_only() is False


def test_the_gate_consumes_the_turn_so_nothing_is_ever_sent(owner_library):
    """The turn is answered by the gate; the LLM path is never reached."""
    spoken: list[str] = []

    class FakeSession:
        async def say(self, text, **kwargs):
            spoken.append(text)

    agent = FakeAgent(OWNER)
    gate = company_privacy.CompanyModeGate(agent)
    company_privacy.begin_turn(agent, SENSITIVE_FIRST_TURN)

    assert asyncio.run(gate.handle(SENSITIVE_FIRST_TURN, FakeSession())) is True
    assert len(spoken) == 1
    assert CANARY not in spoken[0]

    company_privacy.begin_turn(agent, "what is the weather")
    assert asyncio.run(gate.handle("what is the weather", FakeSession())) is False
    assert len(spoken) == 1


# --- the context boundary ----------------------------------------------------------------------


def test_the_session_state_does_not_depend_on_which_task_reads_it(owner_library):
    """Local commands run in their own task; a context variable does not follow.

    The durable-work admission is checked from the local-command path, not from
    the LLM node, so a privacy decision that lived only in the LLM node's
    context would have admitted a company-private turn to the harness worker.
    """

    async def _other_task() -> bool:
        return company_privacy.is_local_only()

    async def _drive() -> tuple[bool, bool]:
        agent = FakeAgent(OWNER)
        company_privacy.bind_session(agent, requested=True)
        company_privacy.begin_turn(agent, "hello")
        here = company_privacy.is_local_only()
        # A task started fresh, the way the local-command path is entered.
        elsewhere = await asyncio.get_running_loop().create_task(_leaf(agent))
        return here, elsewhere

    async def _leaf(agent) -> bool:
        # What the local-command path now does before reading a turn.
        company_privacy.begin_turn(agent, "and the weather?")
        return await _other_task()

    here, elsewhere = asyncio.run(_drive())
    assert here is True
    assert elsewhere is True


def test_the_local_turn_handler_rebinds_the_session_before_reading_a_turn(owner_library):
    """The wiring itself, on the real handler, without starting a voice session."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "voice_agent_company_gate_test", Path(__file__).parents[1] / "voice_agent.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    spoken: list[str] = []
    admitted: list[str] = []

    class FakeSession:
        history = None

        async def say(self, text, **kwargs):
            spoken.append(text)

    class FakeBackground:
        def has_deliverable_file_request(self, text):  # pragma: no cover - never reached
            return False

        async def process_turn(self, text, session, **kwargs):  # pragma: no cover
            admitted.append(text)
            raise AssertionError("a company-shaped turn reached durable work")

    agent = FakeAgent(OWNER)
    handler = module.LocalTurnHandler(
        phone_handoff=None,
        session=FakeSession(),
        end_call=lambda: asyncio.sleep(0),
        background=FakeBackground(),
    )
    handler.bind_agent(agent)

    # A fresh task, exactly as the transcript event enters this path.
    async def _drive():
        return await asyncio.get_running_loop().create_task(
            handler._handle_local_commands(SENSITIVE_FIRST_TURN)
        )

    assert asyncio.run(_drive()) is True
    assert admitted == []
    assert spoken and CANARY not in spoken[0]
    assert "company session" in spoken[0].lower()


# --- the operator's other choices --------------------------------------------------------------


def test_owner_session_remains_available_for_an_operator_who_wants_it(owner_library, monkeypatch):
    monkeypatch.setenv("CAAL_COMPANY_PRIVATE_MODE", "owner_session")
    agent = _bind_registry(FakeAgent(OWNER))
    company_privacy.begin_turn(agent, "what is the weather")
    assert company_privacy.is_local_only() is True
    assert company_privacy.company_tools_offered(OWNER) is True
    assert _available(agent, "hass_assist") is False


def test_off_offers_no_company_tools_in_any_session(owner_library, monkeypatch):
    monkeypatch.setenv("CAAL_COMPANY_PRIVATE_MODE", "off")
    agent = _bind_registry(FakeAgent(OWNER))
    company_privacy.bind_session(agent, requested=True)
    company_privacy.begin_turn(agent, SENSITIVE_FIRST_TURN)
    assert company_privacy.is_local_only() is False
    assert company_privacy.company_tools_offered(OWNER) is False
    assert _available(agent, "company.search") is False


def test_an_unknown_mode_falls_back_to_the_explicit_default(owner_library, monkeypatch):
    monkeypatch.setenv("CAAL_COMPANY_PRIVATE_MODE", "whatever")
    assert company_privacy.private_mode() == company_privacy.MODE_EXPLICIT


def test_the_retired_heuristic_mode_is_not_silently_honoured(owner_library, monkeypatch):
    """`on_intent` used to be a mode. A stale value must not weaken anything."""
    monkeypatch.setenv("CAAL_COMPANY_PRIVATE_MODE", "on_intent")
    assert company_privacy.private_mode() == company_privacy.MODE_EXPLICIT


# --- with no library at all ---------------------------------------------------------------------


def test_without_a_configured_library_nothing_changes_for_anyone(monkeypatch):
    from caal.company import runtime

    monkeypatch.setattr(runtime, "get_config", lambda: None)
    agent = _bind_registry(FakeAgent(OWNER))
    company_privacy.bind_session(agent, requested=True)
    company_privacy.begin_turn(agent, SENSITIVE_FIRST_TURN)
    assert company_privacy.is_local_only() is False
    assert company_privacy.company_mode_notice(agent, SENSITIVE_FIRST_TURN) is None
    assert _available(agent, "hass_assist") is True
