"""What a company-private session can actually do, enforced rather than described.

The first repair narrowed a private session's tool surface to four native
categories: ``company``, ``memory``, ``reminders`` and ``alarms``. The
reasoning was that each of those keeps its bytes on this machine. It was wrong
twice over, and an independent probe (`reports/company-knowledge/`
``verify_private_delivery.py``) demonstrated both:

* ``reminders.create`` takes a ``delivery`` list, and ``telegram`` on that list
  means a Telegram bot API call from the durable worker. The probe created a
  reminder whose *title* was a canary inside an engaged private session and
  watched the canary arrive at a captured Telegram sink through the real
  handler and the real dispatcher. "No tool in the session that sends" was not
  true: a reminder is a deferred send, and the deferral is what hid it;
* ``memory`` and ``reminders`` are *persistence*. A passage quoted out of an
  HR document and written to memory in a private session is read back in an
  ordinary session, where the cloud runtime is in the path. The confidentiality
  boundary is per session, so anything that survives the session crosses it.

So the initial private surface is **query only**: the two read-only company
tools and nothing else. No writes, no persistence, no scheduling, no sending,
and no local-command route that schedules, sends or hands the call off. An
ordinary session keeps every one of these capabilities untouched -- that is
checked here too, because a confidentiality control paid for with a silent
capability removal is the defect this whole phase exists to avoid.

Every fixture is synthetic. Nothing here sends anything anywhere.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from caal import company_privacy
from caal.company import runtime as company_runtime
from caal.tools import create_default_registry
from caal.user_scope import UserScope

OWNER = "usr_" + "a1" * 12
STRANGER = "usr_" + "b2" * 12


@pytest.fixture(autouse=True)
def _clean():
    company_privacy.reset()
    yield
    company_privacy.reset()
    company_runtime.reset()


@pytest.fixture()
def owner_configured(monkeypatch):
    """A provisioned library owner, without a library on disk."""
    monkeypatch.setattr(
        company_privacy,
        "_configured_owner",
        lambda: OWNER,
    )


def _agent(user_id: str = OWNER):
    return SimpleNamespace(
        _user_scope=UserScope(user_id=user_id, identity_configured=True, role="admin"),
        _native_tool_registry=create_default_registry(),
    )


def _private(user_id: str = OWNER):
    agent = _agent(user_id)
    company_privacy.bind_session(agent, requested=True)
    company_privacy.begin_turn(agent, "hello")
    return agent


def _ordinary(user_id: str = OWNER):
    agent = _agent(user_id)
    company_privacy.bind_session(agent, requested=False)
    company_privacy.begin_turn(agent, "what is the weather")
    return agent


# --- the native tool surface -----------------------------------------------------------------


def test_the_private_surface_is_exactly_the_two_read_only_company_tools(owner_configured):
    assert company_privacy.PRIVATE_SESSION_TOOLS == frozenset(
        {"company.search", "company.read"}
    )
    assert company_privacy.ALLOWED_TOOL_CATEGORIES == frozenset({"company"})


def test_every_private_surface_tool_exists_and_is_a_company_read(owner_configured):
    registry = create_default_registry()
    for name in company_privacy.PRIVATE_SESSION_TOOLS:
        tool = registry.get(name)
        assert tool.category == "company"


def test_a_private_session_is_not_offered_a_reminder_tool(owner_configured):
    """The probe's finding: a reminder is a deferred send, armed by the model."""
    agent = _private()
    node = importlib.import_module("caal.llm.llm_node")
    assert company_privacy.is_local_only() is True
    assert node._tool_available(agent, "reminders.create") is False
    assert node._tool_available(agent, "reminders.set_delivery") is False


def test_a_private_session_is_not_offered_an_alarm_or_a_memory_write(owner_configured):
    """Persistence outlives the session, and the boundary is per session."""
    agent = _private()
    node = importlib.import_module("caal.llm.llm_node")
    registry = create_default_registry()
    for name in registry.names():
        category = registry.get(name).category
        if category in {"memory", "reminders", "alarms"}:
            assert node._tool_available(agent, name) is False, name


def test_a_private_session_is_offered_the_company_reads(owner_configured):
    agent = _private()
    node = importlib.import_module("caal.llm.llm_node")
    assert node._tool_available(agent, "company.search") is True
    assert node._tool_available(agent, "company.read") is True


def test_no_tool_outside_the_company_reads_survives_a_private_session(owner_configured):
    """The whole registry, not a hand-picked list: the allowlist is the surface."""
    agent = _private()
    node = importlib.import_module("caal.llm.llm_node")
    registry = create_default_registry()
    offered = {name for name in registry.names() if node._tool_available(agent, name)}
    assert offered == {"company.search", "company.read"}


def test_a_private_session_refuses_an_unknown_company_tool_name(owner_configured):
    """An allowlist of names, not of categories: a future company *write* is refused."""
    assert company_privacy.tool_allowed_in_private_session("company", "company.upload") is False
    assert company_privacy.tool_allowed_in_private_session("company", "company.delete") is False
    assert company_privacy.tool_allowed_in_private_session("memory", "memory.write") is False


# --- the ordinary session keeps everything ----------------------------------------------------


def test_an_ordinary_owner_session_keeps_reminders_memory_and_alarms(owner_configured):
    agent = _ordinary()
    node = importlib.import_module("caal.llm.llm_node")
    assert company_privacy.is_local_only() is False
    assert node._tool_available(agent, "reminders.create") is True
    assert node._tool_available(agent, "reminders.set_delivery") is True


def test_an_ordinary_owner_session_is_offered_no_company_tool(owner_configured):
    agent = _ordinary()
    node = importlib.import_module("caal.llm.llm_node")
    assert node._tool_available(agent, "company.search") is False


def test_a_session_that_is_not_the_owners_is_untouched(owner_configured):
    agent = _ordinary(STRANGER)
    node = importlib.import_module("caal.llm.llm_node")
    assert node._tool_available(agent, "reminders.create") is True
    assert node._tool_available(agent, "company.search") is False


# --- the local-command routes ------------------------------------------------------------------


def test_the_blocked_local_routes_are_the_ones_that_schedule_send_or_hand_off():
    assert company_privacy.PRIVATE_SESSION_BLOCKED_ROUTES == frozenset(
        {
            "delivery_answer",
            "scheduled_change",
            "phone_handoff",
            "connected_knowledge",
            "durable_work",
            "end_call_callback",
        }
    )


@pytest.mark.parametrize("route", sorted(company_privacy.PRIVATE_SESSION_BLOCKED_ROUTES))
def test_a_blocked_route_is_refused_in_a_private_session(owner_configured, route):
    _private()
    assert company_privacy.local_route_allowed(route) is False


@pytest.mark.parametrize("route", sorted(company_privacy.PRIVATE_SESSION_BLOCKED_ROUTES))
def test_the_same_route_is_allowed_in_an_ordinary_session(owner_configured, route):
    _ordinary()
    assert company_privacy.local_route_allowed(route) is True


def test_a_route_nobody_named_is_allowed_by_default_outside_a_private_session():
    """This is not a general kill switch: it names exactly what it blocks."""
    assert company_privacy.local_route_allowed("company_mode_gate") is True


def test_the_company_mode_gate_route_is_not_blocked(owner_configured):
    """It answers and consumes a turn locally; blocking it would route the turn on."""
    _private()
    assert company_privacy.local_route_allowed("company_mode_gate") is True


# --- the handler that reads them ---------------------------------------------------------------


class _Spy:
    """Stands in for one local-command route and records whether it was asked."""

    def __init__(self, claims: bool = True) -> None:
        self.calls = 0
        self._claims = claims
        self.language_binder = None

    def set_language_binder(self, binder) -> None:
        """The handoff route's own hook, implemented rather than worked around.

        ``LocalTurnHandler.bind_agent`` hands the phone-handoff route the
        session's language binder, so a spy standing in for that route has to
        accept it. Recording it keeps this a stand-in for the real protocol;
        the privacy assertions below are unchanged and still the point of the
        test -- a blocked route is never *asked*, whatever it can do.
        """
        self.language_binder = binder

    async def handle(self, text, session):
        self.calls += 1
        return self._claims

    # The background route has its own shape.
    async def process_turn(self, text, session, **kwargs):
        self.calls += 1
        return SimpleNamespace(consumed=self._claims, callback_offered=False)

    def has_deliverable_file_request(self, text):
        return False

    @property
    def can_arm_callback(self):
        return True

    async def handle_final_transcript(self, text, session):
        self.calls += 1
        return self._claims


async def _noop_end_call() -> None:
    """A termination that does nothing: these tests never end a call."""


def _handler(agent, **routes):
    voice_agent = importlib.import_module("voice_agent")
    routes.setdefault("phone_handoff", None)
    handler = voice_agent.LocalTurnHandler(
        session=None,
        end_call=_noop_end_call,
        **routes,
    )
    handler.bind_agent(agent)
    return handler


@pytest.mark.asyncio
async def test_a_private_session_never_reaches_the_scheduling_and_sending_routes(
    owner_configured,
):
    agent = _agent()
    company_privacy.bind_session(agent, requested=True)
    delivery, schedule, knowledge, background, handoff = (
        _Spy(),
        _Spy(),
        _Spy(),
        _Spy(),
        _Spy(),
    )
    handler = _handler(
        agent,
        delivery=delivery,
        schedule=schedule,
        knowledge=knowledge,
        background=background,
        phone_handoff=handoff,
    )
    consumed = await handler._handle_local_commands("remind me about the severance clause")
    assert consumed is False
    assert (delivery.calls, schedule.calls, knowledge.calls) == (0, 0, 0)
    assert (background.calls, handoff.calls) == (0, 0)


@pytest.mark.asyncio
async def test_an_ordinary_session_still_reaches_every_one_of_them(owner_configured):
    agent = _agent()
    company_privacy.bind_session(agent, requested=False)
    delivery = _Spy(claims=False)
    schedule = _Spy(claims=True)
    handler = _handler(agent, delivery=delivery, schedule=schedule)
    consumed = await handler._handle_local_commands("cancel my 4pm reminder")
    assert consumed is True
    assert delivery.calls == 1
    assert schedule.calls == 1


@pytest.mark.asyncio
async def test_a_private_session_does_not_arm_a_callback_on_the_way_out(owner_configured):
    """Hanging up is fine. Arming an outbound call back to the user is a send."""
    agent = _agent()
    company_privacy.bind_session(agent, requested=True)
    armed: list[str] = []

    async def _arm():
        armed.append("armed")

    async def _end():
        pass

    voice_agent = importlib.import_module("voice_agent")
    handler = voice_agent.LocalTurnHandler(
        phone_handoff=None,
        session=None,
        end_call=_end,
        arm_callback_and_end_call=_arm,
        background=_Spy(),
    )
    handler.bind_agent(agent)
    company_privacy.begin_turn(agent, "hello")
    assert handler._start_end_call_if_requested("hang up and call me back when you're done") is False
    assert armed == []


@pytest.mark.asyncio
async def test_an_ordinary_session_still_arms_that_callback(owner_configured):
    agent = _agent()
    company_privacy.bind_session(agent, requested=False)
    armed: list[str] = []

    async def _arm():
        armed.append("armed")

    async def _end():
        pass

    voice_agent = importlib.import_module("voice_agent")
    handler = voice_agent.LocalTurnHandler(
        phone_handoff=None,
        session=None,
        end_call=_end,
        arm_callback_and_end_call=_arm,
        background=_Spy(),
    )
    handler.bind_agent(agent)
    company_privacy.begin_turn(agent, "hello")
    assert handler._start_end_call_if_requested("hang up and call me back when you're done") is True
    await handler._end_call_task
    assert armed == ["armed"]


@pytest.mark.asyncio
async def test_a_private_session_can_still_hang_up(owner_configured):
    agent = _agent()
    company_privacy.bind_session(agent, requested=True)
    ended: list[str] = []

    async def _end():
        ended.append("end")

    voice_agent = importlib.import_module("voice_agent")
    handler = voice_agent.LocalTurnHandler(phone_handoff=None, session=None, end_call=_end)
    handler.bind_agent(agent)
    company_privacy.begin_turn(agent, "hello")
    assert handler._start_end_call_if_requested("hang up") is True
    await handler._end_call_task
    assert ended == ["end"]


# --- the deferred worker -----------------------------------------------------------------------


def test_the_durable_worker_admission_refuses_a_private_turn(owner_configured):
    """Already true, pinned here so the two enforcement points stay together."""
    import inspect

    from caal import background_task_session

    source = inspect.getsource(background_task_session)
    assert source.count("if is_local_only():") >= 2


@pytest.mark.asyncio
async def test_no_reminder_can_be_created_by_dispatching_the_tool_directly(owner_configured):
    """The probe's exact path: private session, real handler, real dispatch.

    ``_tool_available`` is what the model is offered; this is what happens if
    something bypasses the offer and dispatches anyway. It must still refuse,
    so the surface is a boundary rather than a prompt.
    """
    agent = _private()
    node = importlib.import_module("caal.llm.llm_node")
    result = await node._execute_single_tool(
        agent,
        "reminders.create",
        {"title": "FIXTURE canary", "due": "in 1 minute", "delivery": ["telegram"]},
    )
    assert result.get("status") != "ok"
