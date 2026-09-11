"""The LLM-first route to the connected email and calendar accounts.

JARVIS main model is the local Ollama one, and it is the thing that should
understand a spoken request about the connected accounts of the user: which
tool to call, over what window, for which account, and how many results to
speak. The deterministic reading in :mod:`caal.knowledge_router` stays, but
only as the safe fallback for the two cases where no local tool-capable model
can answer -- a turn routed to Hermes, which never receives CAAL tool schemas,
and a local model that is not reachable.

Pinned properties:

* nearest/upcoming semantics are a real capability (``schedule.next``): future
  events only, soonest first, exactly one unless the user asked for more;
* the local model is offered those schemas and picks the tool itself, and a
  reachable local model is never preempted by the deterministic reading;
* the deterministic reading understands next/nearest/upcoming as a bounded
  future calendar read, and is used only for Hermes or a local outage;
* during a local outage a request the reading cannot represent is answered
  truthfully, never with a broad all-account query;
* every read stays inside the session own verified scope: an unknown account
  name never widens back out, an ambiguous one asks, and one user never hears
  another;
* nothing logs the utterance, the tool arguments, the tool result, or the
  answer the model composes from it.
"""

from __future__ import annotations

import importlib
import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from test_knowledge import GOOGLE_CALENDAR, NOW, Harness, gevent, run

from caal.knowledge_router import (
    COULD_NOT_INTERPRET,
    KnowledgeTurnHandler,
    LocalToolPath,
    is_interpretable,
    plan_knowledge_turn,
)
from caal.tools import knowledge_tools
from caal.tools.registry import ToolDefinition, ToolRegistry, create_default_registry
from caal.user_scope import UserScope, scoped_tool_arguments

llm_node_module = importlib.import_module("caal.llm.llm_node")

ANA = "usr_" + "a" * 24
# A title that must never reach a log line.
SECRET_TITLE = "Zebulon Quixote merger review"


# --- helpers ----------------------------------------------------------------------------


def _user(user_id: str = ANA) -> UserScope:
    return UserScope.for_user(SimpleNamespace(user_id=user_id, display_name="Ana", role="member"))


class FakeSession:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def say(self, text: str, **_: Any) -> None:
        self.spoken.append(text)


class RecordingTool:
    def __init__(self, message: str = "Your next event is Standup, tomorrow at 1:00 AM.") -> None:
        self.calls: list[dict[str, Any]] = []
        self._message = message

    async def __call__(self, **arguments: Any) -> dict[str, Any]:
        self.calls.append(arguments)
        return dict(status="ok", message=self._message, data=dict(events=[]))


def _tools(**by_short_name: RecordingTool) -> ToolRegistry:
    """The real catalog with the named knowledge tools swapped for recorders."""
    swapped = dict((k.replace("_", ".", 1), v) for k, v in by_short_name.items())
    registry = ToolRegistry()
    for tool in create_default_registry().list():
        if tool.name in swapped:
            tool = ToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                category=tool.category,
                handler=swapped[tool.name],
                user_scoped=tool.user_scoped,
            )
        registry.register(tool)
    return registry


class StubProvider:
    """A provider stand-in: does it offer CAAL tools, and does it answer at all."""

    def __init__(
        self,
        *,
        manages_own_tools: bool = False,
        escalation_available: bool = True,
        reachable: bool = True,
    ) -> None:
        self.manages_own_tools = manages_own_tools
        self.escalation_available = escalation_available
        self._reachable = reachable
        self.probes = 0

    async def reachable(self) -> bool:
        self.probes += 1
        return self._reachable


@pytest.fixture
def h(tmp_path):
    return Harness(tmp_path)


@pytest.fixture(autouse=True)
def capture_every_log(caplog):
    """Every record these modules emit, whatever another module did to them.

    A leak assertion is only worth making if the record would have been seen,
    so the loggers that could carry a subject, a title or an utterance are
    attached explicitly rather than trusted to propagate.
    """
    names = (
        "",
        "caal.knowledge_router",
        "caal.knowledge",
        "caal.knowledge_store",
        "caal.tools",
        "caal.user_scope",
        "caal.llm.llm_node",
        "caal.provider_data",
    )
    targets = [logging.getLogger(name) for name in names]
    previous = [(target.level, target.propagate) for target in targets]
    for target in targets:
        target.setLevel(logging.DEBUG)
        target.propagate = True
        target.addHandler(caplog.handler)
    try:
        yield
    finally:
        for target, (level, propagate) in zip(targets, previous):
            target.removeHandler(caplog.handler)
            target.setLevel(level)
            target.propagate = propagate


@pytest.fixture(autouse=True)
def unbind_tools():
    knowledge_tools.reset()
    yield
    knowledge_tools.reset()


def _events(h: Harness) -> None:
    """A past event, one in progress, and three still ahead, out of order."""
    h.google_events(
        gevent("g-far", "2023-11-20T09:00:00Z", "2023-11-20T10:00:00Z", "Quarterly review"),
        gevent("g-past", "2023-11-14T09:00:00Z", "2023-11-14T10:00:00Z", "Breakfast"),
        gevent("g-now", "2023-11-14T22:00:00Z", "2023-11-14T23:00:00Z", "In progress"),
        gevent("g-mid", "2023-11-16T09:00:00Z", "2023-11-16T10:00:00Z", "Dentist"),
        gevent("g-soon", "2023-11-15T01:00:00Z", "2023-11-15T02:00:00Z", "Standup"),
    )


# --- the reading of nearest / upcoming --------------------------------------------------


@pytest.mark.parametrize(
    "utterance, expected",
    [
        ("what is my next upcoming event", dict(limit=1)),
        ("what is my next upcoming event?", dict(limit=1)),
        ("Jarvis, what is coming up next?", dict(limit=1)),
        ("next upcoming meeting", dict(limit=1)),
        ("what is the nearest event", dict(limit=1)),
        ("what is my soonest meeting", dict(limit=1)),
        ("what is my next meeting", dict(limit=1)),
        ("when is my next appointment", dict(limit=1)),
        ("what is next on my calendar", dict(limit=1)),
        ("what is my next event on my Google calendar", dict(limit=1, account="google")),
        ("what is my next event on the Vertex calendar", dict(limit=1, account="vertex")),
        ("what are my next three meetings", dict(limit=3)),
        ("what are my next two events", dict(limit=2)),
        ("what are my next few appointments", dict(limit=3)),
    ],
)
def test_nearest_questions_plan_a_bounded_future_calendar_read(utterance, expected) -> None:
    plan = plan_knowledge_turn(utterance)

    assert plan is not None, utterance
    assert plan.tool == "schedule.next", utterance
    assert plan.arguments == expected, utterance
    assert "user_id" not in plan.arguments


@pytest.mark.parametrize(
    "utterance, tool, expected",
    [
        # An explicit window is a window, not a nearest question.
        ("what is on my calendar today", "schedule.upcoming", dict(day="today")),
        ("what do I have tomorrow", "schedule.upcoming", dict(day="tomorrow")),
        ("what does next week look like", "schedule.upcoming", dict(days=14)),
        ("what is my schedule for this week", "schedule.upcoming", dict(days=7)),
    ],
)
def test_a_named_window_still_reads_the_window(utterance, tool, expected) -> None:
    plan = plan_knowledge_turn(utterance)

    assert plan is not None and plan.tool == tool, utterance
    assert plan.arguments == expected, utterance


@pytest.mark.parametrize(
    "utterance",
    [
        "schedule a meeting for tomorrow",
        "cancel my next meeting",
        "send Bob an email about my next event",
        "move my nearest appointment to Friday",
    ],
)
def test_an_action_is_never_claimed_as_a_nearest_question(utterance) -> None:
    assert plan_knowledge_turn(utterance) is None, utterance


# --- the capability itself --------------------------------------------------------------


def test_the_next_event_tool_answers_with_the_single_nearest_future_event(h) -> None:
    h.connect(h.ana)
    _events(h)
    knowledge_tools.configure(lambda: h.service)

    result = run(knowledge_tools.next_events(user_id=h.ana))

    assert result["status"] == "ok"
    assert [event["title"] for event in result["data"]["events"]] == ["Standup"]
    assert "Standup" in result["message"]
    for other in ("Breakfast", "In progress", "Dentist", "Quarterly review"):
        assert other not in result["message"], other


def test_the_next_event_tool_orders_and_bounds_what_it_returns(h) -> None:
    h.connect(h.ana)
    _events(h)
    knowledge_tools.configure(lambda: h.service)

    result = run(knowledge_tools.next_events(limit=3, user_id=h.ana))

    assert [event["title"] for event in result["data"]["events"]] == [
        "Standup",
        "Dentist",
        "Quarterly review",
    ]
    starts = [event["start"] for event in result["data"]["events"]]
    assert starts == sorted(starts)
    assert "http" not in result["message"] and "g-soon" not in result["message"]


def test_the_next_event_tool_is_honest_when_nothing_is_coming_up(h) -> None:
    h.connect(h.ana)
    h.google_events(gevent("g-past", "2023-11-14T09:00:00Z", "2023-11-14T10:00:00Z", "Breakfast"))
    knowledge_tools.configure(lambda: h.service)

    result = run(knowledge_tools.next_events(user_id=h.ana))

    assert result["data"]["events"] == []
    assert "nothing" in result["message"].lower()


def test_the_upcoming_tool_can_be_asked_for_what_is_left_of_a_day(h) -> None:
    h.connect(h.ana)
    _events(h)
    knowledge_tools.configure(lambda: h.service)

    whole = run(knowledge_tools.upcoming_schedule(day="today", user_id=h.ana))
    ahead = run(knowledge_tools.upcoming_schedule(day="today", only_future=True, user_id=h.ana))

    assert "Breakfast" in whole["message"]
    assert "Breakfast" not in ahead["message"]
    assert [event["title"] for event in ahead["data"]["events"]] == ["In progress"]


def test_a_named_account_scopes_the_next_event_and_an_unknown_one_never_widens(h) -> None:
    h.connect(h.ana, provider="google", account_label="ana@vertex.example")
    h.connect(
        h.ana,
        provider="microsoft",
        account_label="ana@acme.example",
        provider_account_id="microsoft-account-1",
    )
    _events(h)
    knowledge_tools.configure(lambda: h.service)

    named = run(knowledge_tools.next_events(account="vertex", user_id=h.ana))
    assert named["status"] == "ok"
    assert [a["account"] for a in named["data"]["accounts"]] == ["ana@vertex.example"]

    unknown = run(knowledge_tools.next_events(account="nowhere", user_id=h.ana))
    assert unknown["status"] == "not_found"
    assert unknown["data"].get("events") in (None, [])
    assert "Standup" not in unknown["message"]


def test_an_ambiguous_account_name_asks_which_one_instead_of_reading_both(h) -> None:
    h.connect(h.ana, provider="google", account_label="ana@vertex.example")
    h.connect(
        h.ana,
        provider="microsoft",
        account_label="ops@vertex.example",
        provider_account_id="microsoft-account-1",
    )
    _events(h)
    knowledge_tools.configure(lambda: h.service)

    answer = run(knowledge_tools.next_events(account="vertex", user_id=h.ana))

    assert answer["status"] == "ambiguous"
    assert "which one" in answer["message"].lower()
    assert "Standup" not in answer["message"]


def test_one_user_never_hears_the_next_event_of_another(h) -> None:
    h.connect(h.bo)
    _events(h)
    knowledge_tools.configure(lambda: h.service)

    mine = run(knowledge_tools.next_events(user_id=h.ana))
    theirs = run(knowledge_tools.next_events(user_id=h.bo))

    assert mine["status"] == "no_accounts"
    assert "Standup" not in mine["message"]
    assert "Standup" in theirs["message"]


# --- what the local model is given ------------------------------------------------------


def test_the_connected_account_tools_describe_their_time_and_account_semantics() -> None:
    registry = create_default_registry()

    assert "schedule.next" in registry.names()
    tool = registry.get("schedule.next")
    assert tool.category == "knowledge"
    assert tool.user_scoped and not tool.requires_confirmation
    assert set(tool.parameters["properties"]) >= set(["limit", "account"])
    assert tool.parameters["additionalProperties"] is False
    assert "user_id" not in tool.parameters["properties"]
    assert "connection_id" not in tool.parameters["properties"]

    upcoming = registry.get("schedule.upcoming")
    assert "only_future" in upcoming.parameters["properties"]

    described = " ".join(
        registry.get(name).description.lower()
        for name in ("schedule.next", "schedule.upcoming", "inbox.recent")
    )
    for word in ("next", "nearest", "today", "tomorrow", "account", "soonest", "sorted"):
        assert word in described, word


def test_the_llm_node_offers_the_knowledge_schemas_to_a_tool_capable_model() -> None:
    agent = SimpleNamespace()

    tools = run(llm_node_module._discover_tools(agent))

    names = [tool["function"]["name"] for tool in tools or []]
    for expected in ("schedule.next", "schedule.upcoming", "inbox.recent", "inbox.search"):
        assert expected in names, expected


def test_a_model_chosen_tool_call_is_bound_to_the_session_scope_and_stays_quiet(h, caplog) -> None:
    """The local model picks the tool; the runtime binds the scope and logs nothing private."""
    h.connect(h.ana)
    h.google_events(gevent("g-soon", "2023-11-15T01:00:00Z", "2023-11-15T02:00:00Z", SECRET_TITLE))
    knowledge_tools.configure(lambda: h.service)

    seen: list[dict[str, Any]] = []

    class ToolCallingProvider:
        provider_name = "stub"
        model = "stub"
        manages_own_tools = False
        supports_think = False

        async def chat(self, messages, tools=None, **_: Any):
            assert tools is not None, "the local model must be offered the tool schemas"
            names = [tool["function"]["name"] for tool in tools]
            assert "schedule.next" in names
            call = SimpleNamespace(id="call-1", name="schedule.next", arguments=dict(limit=1))
            return SimpleNamespace(content=None, tool_calls=[call])

        async def chat_stream(self, messages, tools=None, **_: Any):
            seen.extend(m for m in messages if m.get("role") == "tool")
            yield "Your next event is " + SECRET_TITLE + ", tomorrow at one AM."

        def format_tool_call_message(self, content, tool_calls):
            return dict(role="assistant", content=content or "")

        def format_tool_result(self, content, tool_call_id, tool_name):
            return dict(role="tool", content=content, tool_call_id=tool_call_id)

    agent = SimpleNamespace(_user_scope=_user(h.ana))
    chat_ctx = SimpleNamespace(items=[])

    async def drain() -> list[str]:
        return [
            chunk
            async for chunk in llm_node_module.llm_node(
                agent, chat_ctx, provider=ToolCallingProvider()
            )
        ]

    with caplog.at_level(logging.DEBUG):
        spoken = "".join(run(drain()))

    assert SECRET_TITLE in spoken, "the model composed its answer from the tool result"
    assert seen and SECRET_TITLE in seen[0]["content"]
    assert SECRET_TITLE not in caplog.text, "a title must never reach a log line"
    assert "schedule.next" in caplog.text


def test_a_model_cannot_choose_whose_data_a_tool_reads() -> None:
    tool = create_default_registry().get("schedule.next")

    bound = scoped_tool_arguments(
        tool, dict(limit=1, user_id="usr_" + "b" * 24, connection_id="conn_1"), _user(ANA)
    )

    assert bound is not None
    assert bound["user_id"] == ANA
    assert "connection_id" not in bound
    assert scoped_tool_arguments(tool, dict(limit=1), UserScope.anonymous()) is None


# --- when the deterministic fallback is allowed to speak --------------------------------


def test_a_reachable_local_model_keeps_the_question_and_its_tools() -> None:
    tool = RecordingTool()
    session = FakeSession()
    handler = KnowledgeTurnHandler(
        scope=_user(),
        registry=_tools(schedule_next=tool),
        local_tools=LocalToolPath(StubProvider()),
    )

    claimed = run(handler.handle("what is my next upcoming event", session))

    assert claimed is False, "the local model answers this turn itself"
    assert tool.calls == [] and session.spoken == []


def test_hermes_still_gets_the_deterministic_fallback() -> None:
    tool = RecordingTool()
    session = FakeSession()
    handler = KnowledgeTurnHandler(
        scope=_user(),
        registry=_tools(schedule_next=tool),
        local_tools=LocalToolPath(StubProvider(manages_own_tools=True)),
    )

    claimed = run(handler.handle("what is my next upcoming event", session))

    assert claimed is True
    assert tool.calls == [dict(limit=1, user_id=ANA)]
    assert session.spoken == ["Your next event is Standup, tomorrow at 1:00 AM."]


def test_a_local_model_outage_answers_a_recognised_read_from_the_tools() -> None:
    tool = RecordingTool()
    session = FakeSession()
    handler = KnowledgeTurnHandler(
        scope=_user(),
        registry=_tools(schedule_next=tool),
        local_tools=LocalToolPath(StubProvider(reachable=False)),
    )

    claimed = run(handler.handle("what is my next upcoming event", session))

    assert claimed is True
    assert tool.calls == [dict(limit=1, user_id=ANA)]


def test_a_local_model_outage_refuses_semantics_it_cannot_read() -> None:
    tool = RecordingTool()
    session = FakeSession()
    handler = KnowledgeTurnHandler(
        scope=_user(),
        registry=_tools(schedule_upcoming=tool),
        local_tools=LocalToolPath(StubProvider(reachable=False)),
    )

    claimed = run(
        handler.handle(
            "which of my meetings this week overlap with each other and how many hours are they",
            session,
        )
    )

    assert claimed is True
    assert tool.calls == [], "no broad all-account query stands in for what was asked"
    assert session.spoken == [COULD_NOT_INTERPRET]


def test_the_deterministic_fallback_still_runs_when_no_local_path_is_configured() -> None:
    tool = RecordingTool()
    session = FakeSession()
    handler = KnowledgeTurnHandler(scope=_user(), registry=_tools(schedule_next=tool))

    assert run(handler.handle("what is my next upcoming event", session)) is True
    assert tool.calls == [dict(limit=1, user_id=ANA)]


def test_a_turn_routed_to_the_agent_harness_uses_the_deterministic_fallback() -> None:
    path = LocalToolPath(StubProvider())

    assert run(path.status("what is my next upcoming event")) == "local"
    assert run(path.status("research every meeting I have and write it up")) == "escalated"
    assert run(LocalToolPath(StubProvider(reachable=False)).status("what is next")) == "unavailable"
    assert run(LocalToolPath(None).status("what is next")) == "absent"
    assert run(LocalToolPath(StubProvider(manages_own_tools=True)).status("x")) == "absent"


def test_the_reachability_probe_is_not_repeated_for_every_turn() -> None:
    provider = StubProvider()
    path = LocalToolPath(provider)

    for _ in range(4):
        assert run(path.status("what is my next event")) == "local"

    assert provider.probes == 1


@pytest.mark.parametrize(
    "utterance, readable",
    [
        ("what is my next upcoming event", True),
        ("any unread email", True),
        ("what is on my calendar tomorrow", True),
        ("which meetings this week overlap with each other", False),
        ("how many hours of meetings do I have on Thursday", False),
        ("compare my work calendar with my personal one", False),
    ],
)
def test_the_outage_fallback_knows_what_it_cannot_read(utterance, readable) -> None:
    assert is_interpretable(utterance) is readable, utterance


def test_the_route_logs_no_utterance_arguments_or_answer(caplog) -> None:
    tool = RecordingTool(message="Your next event is " + SECRET_TITLE + ", tomorrow.")
    session = FakeSession()
    handler = KnowledgeTurnHandler(
        scope=_user(),
        registry=_tools(schedule_next=tool),
        local_tools=LocalToolPath(StubProvider(manages_own_tools=True)),
    )

    with caplog.at_level(logging.DEBUG, logger="caal.knowledge_router"):
        assert run(handler.handle("what is my next upcoming event on the Vertex calendar", session))

    assert SECRET_TITLE not in caplog.text
    assert "vertex" not in caplog.text.lower()
    assert "upcoming event" not in caplog.text
    assert "schedule.next" in caplog.text


# --- provider data ordering -------------------------------------------------------------


def test_calendar_provider_data_is_sorted_by_event_time(h) -> None:
    connection = h.connect(h.ana)
    _events(h)

    events = run(
        h.data.calendar_events(
            h.ana,
            connection,
            start=datetime.fromtimestamp(NOW - 86400, tz=timezone.utc),
            end=datetime.fromtimestamp(NOW + 30 * 86400, tz=timezone.utc),
            limit=10,
        )
    )

    assert [event.start for event in events] == sorted(event.start for event in events)
    assert [event.title for event in events][0] == "Breakfast"
    assert h.provider.calls(GOOGLE_CALENDAR)


# --- an account the user named themselves, through the model tool loop ------------------


UNIVERSITY = "University"
WORK = "Work"
UNIVERSITY_SUBJECT = "Registrar deadline for Zebulon"
WORK_SUBJECT = "Quixote payroll batch"


def _two_named_accounts(h: Harness) -> tuple[Any, Any]:
    """Two live connections of one user, each with the name their owner gave it."""
    from test_knowledge import GRAPH_MESSAGES, gmail_message

    import httpx

    university = h.connect(h.ana, provider="google", account_label="ana@ualberta.example")
    work = h.connect(
        h.ana,
        provider="microsoft",
        account_label="ana@acme.example",
        provider_account_id="microsoft-account-1",
    )
    h.connections.set_labels(h.ana, university.connection_id, user_label=UNIVERSITY)
    h.connections.set_labels(h.ana, work.connection_id, user_label=WORK)
    h.gmail(gmail_message("g-1", UNIVERSITY_SUBJECT, "registrar@ualberta.example", "2023-11-14T20:00:00Z"))
    h.provider.add(
        "GET",
        GRAPH_MESSAGES,
        httpx.Response(
            200,
            json={
                "value": [
                    {
                        "id": "m-1",
                        "subject": WORK_SUBJECT,
                        "from": {"emailAddress": {"address": "payroll@acme.example"}},
                        "receivedDateTime": "2023-11-14T21:00:00Z",
                        "isRead": False,
                        "bodyPreview": "preview",
                    }
                ]
            },
        ),
    )
    return university, work


class _AliasToolLoop:
    """A local model that picks inbox.recent for a named account, and invents a user."""

    provider_name = "stub"
    model = "stub"
    manages_own_tools = False
    supports_think = False

    def __init__(self, tool: str, arguments: dict[str, Any]) -> None:
        self._tool = tool
        self._arguments = arguments
        self.tool_messages: list[dict[str, Any]] = []
        self.offered: list[str] = []

    async def chat(self, messages, tools=None, **_: Any):
        assert tools is not None, "the local model must be offered the tool schemas"
        self.offered = [tool["function"]["name"] for tool in tools]
        call = SimpleNamespace(id="call-1", name=self._tool, arguments=dict(self._arguments))
        return SimpleNamespace(content=None, tool_calls=[call])

    async def chat_stream(self, messages, tools=None, **_: Any):
        self.tool_messages.extend(m for m in messages if m.get("role") == "tool")
        yield "Here is what I found."

    def format_tool_call_message(self, content, tool_calls):
        return dict(role="assistant", content=content or "")

    def format_tool_result(self, content, tool_call_id, tool_name):
        return dict(role="tool", content=content, tool_call_id=tool_call_id)


def _tool_loop(h: Harness, provider: _AliasToolLoop) -> str:
    """Drive the real llm_node tool loop for the verified user of the harness."""
    agent = SimpleNamespace(
        _user_scope=_user(h.ana), _native_tool_registry=create_default_registry()
    )
    chat_ctx = SimpleNamespace(items=[])

    async def drain() -> list[str]:
        return [
            chunk
            async for chunk in llm_node_module.llm_node(agent, chat_ctx, provider=provider)
        ]

    return "".join(run(drain()))


def test_a_model_named_alias_reads_only_that_account_under_the_verified_user(h, caplog) -> None:
    """"Unread email for my University account", answered the LLM-first way."""
    _two_named_accounts(h)
    knowledge_tools.configure(lambda: h.service)
    provider = _AliasToolLoop(
        "inbox.recent",
        dict(unread_only=True, account="University", user_id="usr_" + "b" * 24),
    )

    with caplog.at_level(logging.DEBUG):
        spoken = _tool_loop(h, provider)

    assert "inbox.recent" in provider.offered and "schedule.upcoming" in provider.offered
    assert provider.tool_messages, "the tool result went back to the model"
    answer = json.loads(provider.tool_messages[0]["content"])
    assert answer["status"] == "ok"
    labels = [account["account"] for account in answer["data"]["accounts"]]
    assert labels == ["ana@ualberta.example"], "only the named connection is read"
    assert UNIVERSITY_SUBJECT in json.dumps(answer)
    assert WORK_SUBJECT not in json.dumps(answer), "the other account is never widened back in"
    assert spoken
    assert UNIVERSITY_SUBJECT not in caplog.text and "University" not in caplog.text


def test_a_model_named_alias_scopes_the_calendar_the_same_way(h) -> None:
    university, _ = _two_named_accounts(h)
    _events(h)
    knowledge_tools.configure(lambda: h.service)
    provider = _AliasToolLoop(
        "schedule.upcoming", dict(day="today", account="university", user_id="usr_" + "c" * 24)
    )

    _tool_loop(h, provider)

    answer = json.loads(provider.tool_messages[0]["content"])
    assert answer["status"] == "ok"
    assert [a["account"] for a in answer["data"]["accounts"]] == ["ana@ualberta.example"]
    assert [a["user_label"] for a in answer["data"]["accounts"]] == [UNIVERSITY]


def test_an_alias_no_account_answers_to_fails_closed_through_the_model_loop(h) -> None:
    _two_named_accounts(h)
    knowledge_tools.configure(lambda: h.service)
    provider = _AliasToolLoop("inbox.recent", dict(unread_only=True, account="Hogwarts"))

    _tool_loop(h, provider)

    answer = json.loads(provider.tool_messages[0]["content"])
    assert answer["status"] == "not_found"
    assert answer["data"] in ({}, None) or not answer["data"].get("messages")
    assert UNIVERSITY_SUBJECT not in json.dumps(answer)
    assert WORK_SUBJECT not in json.dumps(answer)


def test_an_ambiguous_alias_asks_which_one_through_the_model_loop(h) -> None:
    first = h.connect(h.ana, provider="google", account_label="ana@vertex.example")
    second = h.connect(
        h.ana,
        provider="microsoft",
        account_label="ops@vertex.example",
        provider_account_id="microsoft-account-1",
    )
    h.connections.set_labels(h.ana, first.connection_id, user_label="Vertex mail")
    h.connections.set_labels(h.ana, second.connection_id, user_label="Vertex ops")
    knowledge_tools.configure(lambda: h.service)
    provider = _AliasToolLoop("inbox.recent", dict(account="vertex"))

    _tool_loop(h, provider)

    answer = json.loads(provider.tool_messages[0]["content"])
    assert answer["status"] == "ambiguous"
    assert "which one" in answer["message"].lower()


# --- picking recent mail over a search --------------------------------------------------


@pytest.mark.parametrize(
    "utterance",
    [
        "show unread email for my University account",
        "any unread mail in my University account",
        "check the University inbox",
    ],
)
def test_a_request_that_names_only_an_account_reads_recent_mail(utterance) -> None:
    """No search phrase was asked for, so nothing is searched."""
    plan = plan_knowledge_turn(utterance)

    assert plan is not None, utterance
    assert plan.tool == "inbox.recent", utterance
    assert plan.arguments.get("account") == "university", utterance
    assert "query" not in plan.arguments, utterance


def test_a_request_with_a_search_phrase_still_searches() -> None:
    plan = plan_knowledge_turn("any email from the registrar in my University account")

    assert plan is not None and plan.tool == "inbox.search"
    assert plan.arguments.get("account") == "university"
    assert "registrar" in plan.arguments["query"]


def test_a_search_with_no_search_phrase_answers_with_recent_mail_instead(h) -> None:
    """A model that reaches for inbox.search anyway still answers what was asked."""
    _two_named_accounts(h)
    knowledge_tools.configure(lambda: h.service)

    searched = run(knowledge_tools.search_email(query="unread", account="University", user_id=h.ana))
    recent = run(
        knowledge_tools.recent_email(unread_only=True, account="University", user_id=h.ana)
    )

    assert searched["status"] == "ok"
    assert searched["message"] == recent["message"]
    assert [a["account"] for a in searched["data"]["accounts"]] == ["ana@ualberta.example"]


def test_a_real_search_phrase_is_never_turned_into_a_recent_read(h) -> None:
    _two_named_accounts(h)
    knowledge_tools.configure(lambda: h.service)

    found = run(knowledge_tools.search_email(query="registrar", account="University", user_id=h.ana))
    missing = run(knowledge_tools.search_email(query="nothing here", account="University", user_id=h.ana))

    assert found["status"] == "ok" and UNIVERSITY_SUBJECT in found["message"]
    assert missing["status"] == "not_found"


def test_the_search_tool_tells_the_model_when_to_use_recent_mail_instead() -> None:
    registry = create_default_registry()

    described = registry.get("inbox.search").description.lower()
    assert "inbox.recent" in described
