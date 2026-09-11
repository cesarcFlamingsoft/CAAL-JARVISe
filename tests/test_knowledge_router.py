"""The local route that takes connected-account email and calendar questions to the index.

Production runs the Hermes provider, which manages its own tool loop in its own
process: the LLM node never offers it CAAL native tools, so the user-scoped
inbox.* / schedule.* knowledge tools were unreachable from a voice turn and
Hermes answered that it could not access the calendar or email. These tests
pin the durable fix: a deterministic, code-level route that recognises such a
question, executes the right knowledge tool under the session verified scope,
speaks the tool own truthful answer, and never lets the LLM guess.

Pinned properties:

* recognition is deterministic and bounded, matches questions about the user
  own mail and calendar, and declines actions (send, schedule, cancel), other
  channels (SMS), control commands, and unrelated turns;
* the scope is the session one, never the utterance one: a verified user is
  bound, an anonymous session under multi-user is refused truthfully without
  opening anything, and a legacy single-user deployment is left to the LLM;
* failures and slow answers become honest spoken replies, never an exception
  into the session and never a fabricated answer;
* nothing here logs the utterance, the arguments, or the spoken answer.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from caal.knowledge import MessagesAnswer, query_terms
from caal.knowledge_router import (
    TIMEOUT_REPLY,
    KnowledgePlan,
    KnowledgeTurnHandler,
    plan_knowledge_turn,
    plan_knowledge_turns,
)
from caal.tools import knowledge_tools
from caal.tools.knowledge_tools import BACKEND_UNAVAILABLE, SESSION_UNAVAILABLE
from caal.tools.registry import ToolDefinition, ToolRegistry, create_default_registry
from caal.user_scope import UserScope

llm_node_module = importlib.import_module("caal.llm.llm_node")

ANA = "usr_" + "a" * 24
BO = "usr_" + "b" * 24
SECRET_UTTERANCE = "did I get an email from Zebulon Quixote about the merger"
SECRET_ANSWER = "Zebulon Quixote about Merger terms, 5 minutes ago."


@pytest.fixture
def router_log() -> Any:
    """Every record the router logger emits, whatever the root logger is doing."""

    class Collect(logging.Handler):
        def __init__(self) -> None:
            super().__init__(level=logging.DEBUG)
            self.lines: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.lines.append(record.getMessage())

    collector = Collect()
    target = logging.getLogger("caal.knowledge_router")
    previous = target.level
    target.addHandler(collector)
    target.setLevel(logging.DEBUG)
    try:
        yield collector.lines
    finally:
        target.removeHandler(collector)
        target.setLevel(previous)


def _user(user_id: str = ANA) -> UserScope:
    return UserScope.for_user(SimpleNamespace(user_id=user_id, display_name="Ana", role="member"))


class FakeSession:
    """Records what the agent would speak."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def say(self, text: str, **_: Any) -> None:
        self.spoken.append(text)


class RecordingTool:
    """A stand-in knowledge tool that records how it was bound and answers canned."""

    def __init__(
        self,
        result: dict[str, Any] | None = None,
        *,
        fail: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._result = result or dict(status="ok", message=SECRET_ANSWER, data=dict(items=[1]))
        self._fail = fail
        self._delay = delay

    async def __call__(self, **arguments: Any) -> dict[str, Any]:
        self.calls.append(arguments)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._fail is not None:
            raise self._fail
        return self._result


def _registry(**tools: RecordingTool) -> ToolRegistry:
    """The real catalog with the named knowledge tools swapped for recorders."""
    real = create_default_registry()
    registry = ToolRegistry()
    for tool in real.list():
        if tool.name in tools:
            tool = ToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                category=tool.category,
                handler=tools[tool.name],
                user_scoped=tool.user_scoped,
            )
        registry.register(tool)
    return registry


def _tools(**by_short_name: RecordingTool) -> ToolRegistry:
    """_tools(inbox_recent=tool) is _registry with dotted names."""
    return _registry(**dict((k.replace("_", ".", 1), v) for k, v in by_short_name.items()))


class FakeCache:
    def __init__(self) -> None:
        self.added: list[tuple[str, Any]] = []

    def add(self, tool_name: str, data: Any) -> None:
        self.added.append((tool_name, data))


# --- recognition ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "utterance, tool, expected",
    [
        ("do I have any new email", "inbox.recent", dict()),
        ("check my inbox", "inbox.recent", dict()),
        ("Jarvis, what's in my inbox?", "inbox.recent", dict()),
        ("how many unread emails do I have", "inbox.recent", dict(unread_only=True)),
        ("any unread mail?", "inbox.recent", dict(unread_only=True)),
        ("what are my last three emails", "inbox.recent", dict(limit=3)),
        ("what's in my Gmail inbox", "inbox.recent", dict(account="google")),
        ("read me the latest email", "inbox.read_summary", dict()),
        ("read the newest message in my Outlook", "inbox.read_summary", dict(account="microsoft")),
        ("what's on my calendar today", "schedule.upcoming", dict(day="today")),
        ("what's on my calendar tonight", "schedule.upcoming", dict(day="today")),
        ("what do I have tomorrow", "schedule.upcoming", dict(day="tomorrow")),
        ("am I free tomorrow afternoon?", "schedule.upcoming", dict(day="tomorrow")),
        ("what's my schedule for this week", "schedule.upcoming", dict(days=7)),
        ("what does next week look like", "schedule.upcoming", dict(days=14)),
        ("any meetings on Friday", "schedule.upcoming", dict(day="friday")),
        ("what's my next meeting", "schedule.next", dict(limit=1)),
        ("how does my day look", "schedule.upcoming", dict(day="today")),
        (
            "what's on my Outlook calendar tomorrow",
            "schedule.upcoming",
            dict(day="tomorrow", account="microsoft"),
        ),
        ("what's on my zoho calendar", "schedule.upcoming", dict(account="zoho")),
        ("when is my next appointment", "schedule.next", dict(limit=1)),
        ("do I have anything today", "schedule.upcoming", dict(day="today")),
        ("what was that email about", "inbox.read_summary", dict()),
    ],
)
def test_planner_recognises_questions_about_the_users_own_mail_and_calendar(
    utterance, tool, expected
) -> None:
    plan = plan_knowledge_turn(utterance)

    assert plan is not None, utterance
    assert plan.tool == tool
    assert plan.arguments == expected
    assert "user_id" not in plan.arguments


@pytest.mark.parametrize(
    "utterance, tool, terms, extra",
    [
        (
            "did I get an email from Bob about the invoice",
            "inbox.search",
            ["bob", "invoice"],
            dict(),
        ),
        ("anything from Sarah?", "inbox.search", ["sarah"], dict()),
        ("did Sarah email me", "inbox.search", ["sarah"], dict()),
        ("has Acme Billing sent me anything", "inbox.search", ["acme", "billing"], dict()),
        (
            "is there an email about the quarterly numbers",
            "inbox.search",
            ["quarterly", "numbers"],
            dict(),
        ),
        ("read the email from Bob", "inbox.read_summary", ["bob"], dict()),
        (
            "do I have a dentist appointment this week",
            "schedule.find_event",
            ["dentist"],
            dict(days=7),
        ),
        ("when is my meeting with Bob", "schedule.find_event", ["bob"], dict()),
        (
            "is there anything called standup tomorrow",
            "schedule.find_event",
            ["standup"],
            dict(day="tomorrow"),
        ),
        ("when's my dentist appointment", "schedule.find_event", ["dentist"], dict()),
        ("what did Bob say in his email", "inbox.read_summary", ["bob"], dict()),
        ("what's the latest from Acme", "inbox.search", ["acme"], dict()),
    ],
)
def test_planner_extracts_a_search_target_from_the_question(utterance, tool, terms, extra) -> None:
    plan = plan_knowledge_turn(utterance)

    assert plan is not None, utterance
    assert plan.tool == tool
    assert query_terms(plan.arguments["query"]) == terms
    for key, value in extra.items():
        assert plan.arguments.get(key) == value, key
    assert "user_id" not in plan.arguments


@pytest.mark.parametrize(
    "utterance",
    [
        "send an email to Bob saying I'm running late",
        "reply to Sarah's email",
        "schedule a meeting with Bob tomorrow at 3",
        "cancel my dentist appointment",
        "move my meeting with Bob to Friday",
        "make me a PDF summary of my inbox",
        "did I get a text message from Bob",
        "how do I schedule a meeting in Outlook",
        "what is an inbox",
        "did you get my email",
        "can you access my calendar",
        "what's the weather today",
        "what time is it",
        "I hate my inbox",
        "hello",
        "yes",
        "",
        "   ",
        None,
        42,
    ],
)
def test_planner_declines_actions_other_channels_and_unrelated_turns(utterance) -> None:
    assert plan_knowledge_turn(utterance) is None


def test_planner_defers_to_the_work_router_for_control_and_work_turns(monkeypatch) -> None:
    """Cancel, status, background and long-work readings keep their own authority."""
    from caal.work_router import Route, RouteDecision, RouteSource

    router = importlib.import_module("caal.knowledge_router")
    monkeypatch.setattr(
        router,
        "deterministic_route",
        lambda text: RouteDecision(Route.WORK, RouteSource.DETERMINISTIC),
    )

    assert plan_knowledge_turn("what's on my calendar today") is None


def test_planner_is_bounded_and_never_emits_a_scope() -> None:
    plan = plan_knowledge_turn("what's on my calendar " + "x" * 10_000)
    assert plan is None or "user_id" not in plan.arguments

    plan = plan_knowledge_turn("did I get an email from " + "y" * 5_000)
    assert plan is not None and plan.tool == "inbox.search"
    assert len(plan.arguments["query"]) <= 200


def test_plan_is_frozen_and_names_its_kind() -> None:
    plan = KnowledgePlan(tool="inbox.recent", arguments=dict(), kind="email")
    with pytest.raises(Exception):
        plan.tool = "other"  # type: ignore[misc]
    assert plan_knowledge_turn("what's on my calendar").kind == "calendar"
    assert plan_knowledge_turn("any new email").kind == "email"


# --- naming one of the user own connected accounts ----------------------------------------


@pytest.mark.parametrize(
    "utterance, tool, expected",
    [
        # The provider forms keep working exactly as they did.
        ("my Google calendar", "schedule.upcoming", dict(account="google")),
        ("unread Outlook mail", "inbox.recent", dict(unread_only=True, account="microsoft")),
        ("Zoho events today", "schedule.upcoming", dict(day="today", account="zoho")),
        ("what's in my Gmail inbox", "inbox.recent", dict(account="google")),
        # A label the user gave one of their own accounts.
        ("give me the calendar of Vertex account", "schedule.upcoming", dict(account="vertex")),
        ("give me the email of Vertex account", "inbox.recent", dict(account="vertex")),
        ("what's on my Vertex account calendar", "schedule.upcoming", dict(account="vertex")),
        ("what's on the Flaming Soft calendar", "schedule.upcoming", dict(account="flaming soft")),
        ("show me the calendar for Vertex", "schedule.upcoming", dict(account="vertex")),
        (
            "what's on the Vertex calendar tomorrow",
            "schedule.upcoming",
            dict(day="tomorrow", account="vertex"),
        ),
        (
            "any unread email for my work account",
            "inbox.recent",
            dict(unread_only=True, account="work"),
        ),
        ("what's in the inbox for Vertex", "inbox.recent", dict(account="vertex")),
        (
            "give me the email of ana@vertex.example",
            "inbox.recent",
            dict(account="ana@vertex.example"),
        ),
        (
            "what's on the calendar of ana@vertex.example today",
            "schedule.upcoming",
            dict(day="today", account="ana@vertex.example"),
        ),
    ],
)
def test_planner_names_one_connected_account_without_guessing(utterance, tool, expected) -> None:
    plan = plan_knowledge_turn(utterance)

    assert plan is not None, utterance
    assert plan.tool == tool
    assert plan.arguments == expected
    assert "user_id" not in plan.arguments


@pytest.mark.parametrize(
    "utterance, tool, terms",
    [
        ("read the email from Bob", "inbox.read_summary", ["bob"]),
        ("did I get an email from Bob about the invoice", "inbox.search", ["bob", "invoice"]),
        ("what's the latest from Acme", "inbox.search", ["acme"]),
        ("when is my meeting with Bob", "schedule.find_event", ["bob"]),
        ("anything from Sarah?", "inbox.search", ["sarah"]),
    ],
)
def test_a_sender_or_a_title_is_never_read_as_an_account(utterance, tool, terms) -> None:
    plan = plan_knowledge_turn(utterance)

    assert plan is not None and plan.tool == tool
    assert query_terms(plan.arguments["query"]) == terms
    assert "account" not in plan.arguments


@pytest.mark.parametrize(
    "utterance",
    [
        "send an email to my Vertex account",
        "schedule a meeting on my Vertex calendar",
        "what's on my calendar today and send an email to Bob",
        "what's on my calendar today and email Bob the address",
        "check my calendar and text Bob the address",
        "what's on my calendar today and remind me to call mom",
        "any unread email and set a reminder for the dentist",
        "do I have a text message from my Vertex account",
        "how do I add an account in Outlook",
    ],
)
def test_planner_never_absorbs_an_action_or_another_channel(utterance) -> None:
    assert plan_knowledge_turns(utterance) == []
    assert plan_knowledge_turn(utterance) is None


def test_a_name_the_user_chose_is_carried_as_the_account_hint() -> None:
    """The words a person actually uses for an account: university, work, wife."""
    cases = [
        ("what's on my university calendar", "schedule.upcoming", dict(account="university")),
        (
            "any unread email in my personal account",
            "inbox.recent",
            dict(unread_only=True, account="personal"),
        ),
        ("what's in my wife's inbox", "inbox.recent", dict(account="wife")),
        ("show me the calendar for university", "schedule.upcoming", dict(account="university")),
    ]
    for utterance, tool, expected in cases:
        plan = plan_knowledge_turn(utterance)
        assert plan is not None, utterance
        assert (plan.tool, plan.arguments) == (tool, expected), utterance


@pytest.mark.parametrize(
    "utterance, tool, expected",
    [
        # The turn as it was reported: the whole of "u of a" is the account, not
        # its first letter, which used to resolve far too broadly.
        ("what are my events for my u of a calendar", "schedule.upcoming", dict(account="u of a")),
        ("my u of a calendar today", "schedule.upcoming", dict(day="today", account="u of a")),
        (
            "unread emails for my U of A account",
            "inbox.recent",
            dict(unread_only=True, account="u of a"),
        ),
        ("what's on my U of A calendar", "schedule.upcoming", dict(account="u of a")),
        # The same name said in full, and other names built the same way.
        (
            "what's on my University of Alberta calendar",
            "schedule.upcoming",
            dict(account="university of alberta"),
        ),
        (
            "any unread email in my University of Alberta account",
            "inbox.recent",
            dict(unread_only=True, account="university of alberta"),
        ),
        (
            "show me the calendar for the city of edmonton",
            "schedule.upcoming",
            dict(account="city of edmonton"),
        ),
        (
            "what's in the inbox for Smith and Sons",
            "inbox.recent",
            dict(account="smith and sons"),
        ),
    ],
)
def test_a_label_keeps_the_connector_words_inside_it(utterance, tool, expected) -> None:
    """A name with "of" or "and" in it is carried whole, never cut at the connector."""
    plan = plan_knowledge_turn(utterance)

    assert plan is not None, utterance
    assert (plan.tool, plan.arguments) == (tool, expected), utterance
    assert "user_id" not in plan.arguments


def test_a_connector_name_qualifies_both_halves_of_a_combined_question() -> None:
    plans = plan_knowledge_turns(
        "what are my events for my u of a calendar and unread emails for my u of a account"
    )

    assert [p.tool for p in plans] == ["schedule.upcoming", "inbox.recent"]
    assert plans[0].arguments == dict(account="u of a")
    assert plans[1].arguments == dict(unread_only=True, account="u of a")
    assert all("user_id" not in p.arguments for p in plans)


@pytest.mark.parametrize(
    "utterance, tool, terms",
    [
        # A connector after a sender or a subject is still not part of a label.
        ("read the email from Bob of Acme", "inbox.read_summary", ["bob", "acme"]),
        ("did I get an email about the city of edmonton", "inbox.search", ["city", "edmonton"]),
        ("when is my meeting with Bob and Sarah", "schedule.find_event", ["bob", "sarah"]),
        ("anything from the University of Alberta", "inbox.search", ["university", "alberta"]),
    ],
)
def test_a_connector_does_not_turn_a_sender_or_a_title_into_an_account(
    utterance, tool, terms
) -> None:
    plan = plan_knowledge_turn(utterance)

    assert plan is not None and plan.tool == tool, utterance
    assert query_terms(plan.arguments["query"]) == terms
    assert "account" not in plan.arguments, utterance


def test_a_user_name_qualifies_both_halves_of_a_combined_question() -> None:
    shared = plan_knowledge_turns(
        "give me the events for today and unread emails for my university account"
    )
    assert [p.tool for p in shared] == ["schedule.upcoming", "inbox.recent"]
    assert shared[0].arguments == dict(day="today", account="university")
    assert shared[1].arguments == dict(unread_only=True, account="university")

    separate = plan_knowledge_turns(
        "what's on my university calendar and any unread email in my work account"
    )
    assert [p.tool for p in separate] == ["schedule.upcoming", "inbox.recent"]
    assert separate[0].arguments == dict(account="university")
    assert separate[1].arguments == dict(unread_only=True, account="work")
    assert all("user_id" not in p.arguments for p in separate)


def test_one_turn_can_ask_about_the_calendar_and_the_inbox_of_one_named_account() -> None:
    """A qualifier after the conjunction belongs to both halves of the question."""
    plans = plan_knowledge_turns(
        "give me the events for today and unread emails for Vertex account"
    )

    assert [p.tool for p in plans] == ["schedule.upcoming", "inbox.recent"]
    assert [p.kind for p in plans] == ["calendar", "email"]
    assert plans[0].arguments == dict(day="today", account="vertex")
    assert plans[1].arguments == dict(unread_only=True, account="vertex")
    assert all("user_id" not in p.arguments for p in plans)


def test_a_shared_provider_after_the_conjunction_applies_to_both_halves() -> None:
    plans = plan_knowledge_turns("any unread email and what's on my Zoho calendar today")

    assert [p.tool for p in plans] == ["inbox.recent", "schedule.upcoming"]
    assert plans[0].arguments == dict(unread_only=True, account="zoho")
    assert plans[1].arguments == dict(day="today", account="zoho")


def test_separate_named_accounts_stay_separate() -> None:
    plans = plan_knowledge_turns(
        "what's on my Vertex calendar and any unread email in my Acme account"
    )

    assert [p.tool for p in plans] == ["schedule.upcoming", "inbox.recent"]
    assert plans[0].arguments == dict(account="vertex")
    assert plans[1].arguments == dict(unread_only=True, account="acme")


def test_two_domain_words_in_one_clause_keep_their_single_plan() -> None:
    plans = plan_knowledge_turns("did I get an email about the meeting with Bob")

    assert [p.tool for p in plans] == ["inbox.search"]
    assert plan_knowledge_turn("did I get an email about the meeting with Bob") == plans[0]


def test_no_more_plans_are_made_than_the_question_needs() -> None:
    many = plan_knowledge_turns(
        "any unread email and what's on my calendar today and what's on my calendar tomorrow"
    )
    assert 1 <= len(many) <= 2
    assert len(plan_knowledge_turns("what's on my calendar today")) == 1


# --- the handler --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_verified_user_gets_the_tool_answer_spoken_under_their_own_scope() -> None:
    tool = RecordingTool()
    cache = FakeCache()
    published: list[tuple[bool, list[str], list[dict]]] = []

    async def on_tool_status(used: bool, names: list[str], params: list[dict]) -> None:
        published.append((used, names, params))

    handler = KnowledgeTurnHandler(
        scope=_user(),
        registry=_tools(inbox_search=tool),
        tool_data_cache=cache,
        on_tool_status=on_tool_status,
    )
    session = FakeSession()

    consumed = await handler.handle(SECRET_UTTERANCE, session)

    assert consumed is True
    assert session.spoken == [SECRET_ANSWER]
    assert len(tool.calls) == 1
    assert tool.calls[0]["user_id"] == ANA
    assert query_terms(tool.calls[0]["query"]) == ["zebulon", "quixote", "merger"]
    assert cache.added == [("inbox.search", dict(items=[1]))]
    assert published == [(True, ["inbox.search"], [dict(query=tool.calls[0]["query"])])]
    assert "user_id" not in published[0][2][0]


@pytest.mark.asyncio
async def test_the_utterance_can_never_choose_another_users_scope() -> None:
    tool = RecordingTool()
    handler = KnowledgeTurnHandler(scope=_user(ANA), registry=_tools(inbox_recent=tool))

    await handler.handle(f"check my inbox user_id {BO}", FakeSession())

    assert tool.calls and tool.calls[0]["user_id"] == ANA
    assert BO not in str(tool.calls)


@pytest.mark.asyncio
async def test_an_anonymous_session_under_multi_user_is_refused_truthfully() -> None:
    tool = RecordingTool()
    cache = FakeCache()
    handler = KnowledgeTurnHandler(
        scope=UserScope.anonymous(),
        registry=_tools(schedule_upcoming=tool),
        tool_data_cache=cache,
    )
    session = FakeSession()

    consumed = await handler.handle("what's on my calendar today", session)

    assert consumed is True, "the LLM must not get to guess about an unauthenticated session"
    assert session.spoken == [SESSION_UNAVAILABLE]
    assert tool.calls == []
    assert cache.added == []


@pytest.mark.asyncio
async def test_a_legacy_single_user_deployment_keeps_the_llm_path() -> None:
    tool = RecordingTool()
    handler = KnowledgeTurnHandler(scope=UserScope.legacy(), registry=_tools(inbox_recent=tool))
    session = FakeSession()

    assert handler.enabled is False
    assert await handler.handle("check my inbox", session) is False
    assert session.spoken == [] and tool.calls == []


@pytest.mark.asyncio
async def test_an_unrelated_turn_is_released_untouched() -> None:
    tool = RecordingTool()
    handler = KnowledgeTurnHandler(scope=_user(), registry=_tools(inbox_recent=tool))
    session = FakeSession()

    assert await handler.handle("what's the weather like", session) is False
    assert await handler.handle("", session) is False
    assert session.spoken == [] and tool.calls == []


@pytest.mark.asyncio
async def test_a_failing_tool_becomes_an_honest_reply_not_an_exception(router_log) -> None:
    tool = RecordingTool(fail=RuntimeError(SECRET_ANSWER))
    handler = KnowledgeTurnHandler(scope=_user(), registry=_tools(inbox_recent=tool))
    session = FakeSession()

    consumed = await handler.handle("any new email", session)

    assert consumed is True
    assert session.spoken == [BACKEND_UNAVAILABLE]
    assert any("inbox.recent" in line and "failed" in line for line in router_log)
    assert SECRET_ANSWER not in "\n".join(router_log)


@pytest.mark.asyncio
async def test_a_slow_tool_is_bounded_and_reported() -> None:
    tool = RecordingTool(delay=0.5)
    handler = KnowledgeTurnHandler(
        scope=_user(), registry=_tools(inbox_recent=tool), timeout_seconds=0.01
    )
    session = FakeSession()

    assert await handler.handle("any new email", session) is True
    assert session.spoken == [TIMEOUT_REPLY]


@pytest.mark.asyncio
async def test_a_tool_that_answers_off_contract_is_reported_not_spoken() -> None:
    tool = RecordingTool(result=dict(status="ok"))
    handler = KnowledgeTurnHandler(scope=_user(), registry=_tools(inbox_recent=tool))
    session = FakeSession()

    assert await handler.handle("any new email", session) is True
    assert session.spoken == [BACKEND_UNAVAILABLE]


@pytest.mark.asyncio
async def test_a_speech_failure_never_raises_into_the_session(router_log) -> None:
    class MuteSession:
        async def say(self, text: str, **_: Any) -> None:
            raise RuntimeError("tts down " + SECRET_ANSWER)

    handler = KnowledgeTurnHandler(scope=_user(), registry=_tools(inbox_recent=RecordingTool()))

    assert await handler.handle("any new email", MuteSession()) is True
    assert any("Could not speak" in line for line in router_log)
    assert SECRET_ANSWER not in "\n".join(router_log)


@pytest.mark.asyncio
async def test_nothing_about_the_question_or_the_answer_is_logged(router_log) -> None:
    handler = KnowledgeTurnHandler(scope=_user(), registry=_tools(inbox_search=RecordingTool()))

    await handler.handle(SECRET_UTTERANCE, FakeSession())

    joined = "\n".join(router_log)
    assert "inbox.search" in joined, "the call itself is recorded"
    for hidden in ("Zebulon", "Quixote", "merger", "Merger", SECRET_ANSWER, ANA):
        assert hidden not in joined, hidden


@pytest.mark.asyncio
async def test_the_route_reaches_the_real_knowledge_service() -> None:
    """End to end through the real registry and tool into a (fake) KnowledgeService."""
    from zoneinfo import ZoneInfo

    asked: list[dict[str, Any]] = []

    class FakeService:
        zone = ZoneInfo("UTC")

        def now(self) -> int:
            return 1_700_000_000

        def start_of_day(self, ts: int, *, days_ahead: int = 0) -> int:
            return ts - ts % 86400 + days_ahead * 86400

        async def recent_messages(self, user_id, **kwargs):
            asked.append(dict(user_id=user_id, **kwargs))
            return MessagesAnswer(
                messages=[], accounts=[], generated_at=self.now(), connected=False
            )

    knowledge_tools.configure(lambda: FakeService())
    try:
        handler = KnowledgeTurnHandler(scope=_user())
        session = FakeSession()

        assert await handler.handle("do I have any new email", session) is True
    finally:
        knowledge_tools.reset()

    assert asked and asked[0]["user_id"] == ANA
    assert session.spoken == [knowledge_tools.NO_ACCOUNTS]


# --- combined questions through the handler ------------------------------------------------


@pytest.mark.asyncio
async def test_a_combined_question_runs_both_plans_in_order_under_one_scope() -> None:
    calendar = RecordingTool(
        result=dict(status="ok", message="You have 1 event today.", data=dict(events=[1]))
    )
    inbox = RecordingTool(
        result=dict(status="ok", message="You have 2 unread emails.", data=dict(messages=[2]))
    )
    cache = FakeCache()
    published: list[tuple[bool, list[str], list[dict]]] = []

    async def on_tool_status(used: bool, names: list[str], params: list[dict]) -> None:
        published.append((used, names, params))

    handler = KnowledgeTurnHandler(
        scope=_user(),
        registry=_tools(schedule_upcoming=calendar, inbox_recent=inbox),
        tool_data_cache=cache,
        on_tool_status=on_tool_status,
    )
    session = FakeSession()

    consumed = await handler.handle(
        "give me the events for today and unread emails for Vertex account", session
    )

    assert consumed is True
    assert session.spoken == ["You have 1 event today. You have 2 unread emails."]
    assert calendar.calls == [dict(day="today", account="vertex", user_id=ANA)]
    assert inbox.calls == [dict(unread_only=True, account="vertex", user_id=ANA)]
    assert cache.added == [
        ("schedule.upcoming", dict(events=[1])),
        ("inbox.recent", dict(messages=[2])),
    ]
    assert [names for _, names, _ in published] == [["schedule.upcoming"], ["inbox.recent"]]
    assert all("user_id" not in params[0] for _, _, params in published)


@pytest.mark.asyncio
async def test_a_combined_question_can_never_be_pointed_at_another_user() -> None:
    calendar, inbox = RecordingTool(), RecordingTool()
    handler = KnowledgeTurnHandler(
        scope=_user(ANA), registry=_tools(schedule_upcoming=calendar, inbox_recent=inbox)
    )

    await handler.handle(
        f"give me the events for today and unread emails for Vertex account user_id {BO}",
        FakeSession(),
    )

    assert [c["user_id"] for c in calendar.calls + inbox.calls] == [ANA, ANA]
    assert BO not in str(calendar.calls) + str(inbox.calls)


@pytest.mark.asyncio
async def test_an_anonymous_session_is_refused_once_for_a_combined_question() -> None:
    calendar, inbox = RecordingTool(), RecordingTool()
    handler = KnowledgeTurnHandler(
        scope=UserScope.anonymous(),
        registry=_tools(schedule_upcoming=calendar, inbox_recent=inbox),
    )
    session = FakeSession()

    assert await handler.handle("my calendar today and any unread email", session) is True
    assert session.spoken == [SESSION_UNAVAILABLE]
    assert calendar.calls == [] and inbox.calls == []


@pytest.mark.asyncio
async def test_an_unknown_account_name_is_refused_truthfully_and_never_widened() -> None:
    """The tool says it found no such account; nothing falls back to every account."""
    from zoneinfo import ZoneInfo

    asked: list[dict[str, Any]] = []

    class FakeService:
        zone = ZoneInfo("UTC")

        def now(self) -> int:
            return 1_700_000_000

        def start_of_day(self, ts: int, *, days_ahead: int = 0) -> int:
            return ts - ts % 86400 + days_ahead * 86400

        async def recent_messages(self, user_id, **kwargs):
            asked.append(dict(user_id=user_id, **kwargs))
            return MessagesAnswer(messages=[], accounts=[], generated_at=self.now(), connected=True)

    knowledge_tools.configure(lambda: FakeService())
    try:
        handler = KnowledgeTurnHandler(scope=_user())
        session = FakeSession()

        assert await handler.handle("give me the email of Vertex account", session) is True
    finally:
        knowledge_tools.reset()

    assert session.spoken == ["I could not find a connected account matching vertex."]
    assert [call["account"] for call in asked] == ["vertex"], "asked once, for that account only"
    assert asked[0]["user_id"] == ANA


@pytest.mark.asyncio
async def test_a_named_account_is_never_logged(router_log) -> None:
    handler = KnowledgeTurnHandler(
        scope=_user(),
        registry=_tools(schedule_upcoming=RecordingTool(), inbox_recent=RecordingTool()),
    )

    await handler.handle(
        "give me the events for today and unread emails for Zebulon Quixote account", FakeSession()
    )

    joined = "\n".join(router_log)
    assert "schedule.upcoming" in joined and "inbox.recent" in joined
    for hidden in ("Zebulon", "zebulon", "Quixote", "quixote", SECRET_ANSWER, ANA):
        assert hidden not in joined, hidden


# --- the LLM path keeps refusing truthfully (Ollama / Groq) ----------------------------------


@pytest.mark.asyncio
async def test_llm_dispatch_refuses_knowledge_tools_for_unidentified_sessions_in_their_own_words(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        llm_node_module.settings_module, "get_setting", lambda key, default=None: True
    )
    agent = SimpleNamespace(_user_scope=UserScope.anonymous())

    result = await llm_node_module._execute_single_tool(agent, "inbox.recent", dict())

    assert result["status"] == "unauthorized"
    assert result["message"] == SESSION_UNAVAILABLE
    assert result["data"] == dict()


# --- wiring into the voice agent --------------------------------------------------------------


@pytest.fixture(scope="module")
def voice_agent():
    module_path = Path(__file__).parents[1] / "voice_agent.py"
    spec = importlib.util.spec_from_file_location("voice_agent_for_knowledge", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_voice_agent_builds_the_route_only_for_multi_user_sessions(voice_agent) -> None:
    verified = voice_agent.build_knowledge_turn_handler(_user())
    anonymous = voice_agent.build_knowledge_turn_handler(UserScope.anonymous())
    legacy = voice_agent.build_knowledge_turn_handler(UserScope.legacy())

    assert isinstance(verified, KnowledgeTurnHandler) and verified.enabled
    assert isinstance(anonymous, KnowledgeTurnHandler) and anonymous.enabled
    assert legacy is None


def test_voice_agent_gives_the_route_the_local_model_path(voice_agent) -> None:
    """The route must know whether the local model is going to answer instead."""
    local = SimpleNamespace(manages_own_tools=False, escalation_available=True)

    route = voice_agent.build_knowledge_turn_handler(_user(), provider=local)

    assert route is not None
    path = route._local
    assert path is not None and path.offers_tools is True
    assert asyncio.run(path.status("what is my next upcoming event")) == "local"

    hermes = voice_agent.build_knowledge_turn_handler(
        _user(), provider=SimpleNamespace(manages_own_tools=True)
    )
    assert hermes is not None and hermes._local.offers_tools is False


async def _never_end() -> None:
    raise AssertionError("a knowledge question must not end the call")


@pytest.mark.asyncio
async def test_a_spoken_calendar_question_is_answered_locally_and_never_reaches_the_llm(
    voice_agent,
) -> None:
    tool = RecordingTool(result=dict(status="ok", message="You have 1 event today.", data=dict()))
    handler = KnowledgeTurnHandler(scope=_user(), registry=_tools(schedule_upcoming=tool))
    session = FakeSession()
    local = voice_agent.LocalTurnHandler(
        phone_handoff=None, session=session, end_call=_never_end, knowledge=handler
    )

    local.on_final_transcript("what's on my calendar today")
    consumed = await local.turn_consumed("what's on my calendar today")

    assert consumed is True, "consumed turns raise StopResponse, so Hermes never sees them"
    assert session.spoken == ["You have 1 event today."]
    assert tool.calls[0]["user_id"] == ANA
    assert tool.calls[0]["day"] == "today"


@pytest.mark.asyncio
async def test_typed_chat_gets_the_same_route(voice_agent) -> None:
    tool = RecordingTool(result=dict(status="ok", message="No unread email.", data=dict()))
    handler = KnowledgeTurnHandler(scope=_user(), registry=_tools(inbox_recent=tool))
    session = FakeSession()
    local = voice_agent.LocalTurnHandler(
        phone_handoff=None, session=session, end_call=_never_end, knowledge=handler
    )

    assert await local.turn_consumed("any unread email?") is True
    assert session.spoken == ["No unread email."]
    assert tool.calls[0] == dict(unread_only=True, user_id=ANA)


@pytest.mark.asyncio
async def test_an_ordinary_turn_still_flows_past_the_route(voice_agent) -> None:
    handler = KnowledgeTurnHandler(scope=_user(), registry=_tools(inbox_recent=RecordingTool()))
    session = FakeSession()
    local = voice_agent.LocalTurnHandler(
        phone_handoff=None, session=session, end_call=_never_end, knowledge=handler
    )

    assert await local.turn_consumed("tell me a joke") is False
    assert session.spoken == []


def test_voice_assistant_shares_its_tool_data_cache_with_the_route(voice_agent) -> None:
    parameters = inspect.signature(voice_agent.VoiceAssistant.__init__).parameters
    assert "tool_data_cache" in parameters
