"""Semantic routing of a user turn into conversation or long-running work.

The reported failure: on a phone handoff the user said "Actually, can you
create a PDF explaining what you are? It can be short." The hard-coded regex
missed the conversational prefix, the turn went to Hermes synchronously, no
task was queued, no acknowledgement was spoken, and JARVIS went silent.

Widening the regex only moves the wall. This module tests a router that asks
the configured LLM whether a turn is ordinary conversation or a piece of work
that produces an artifact, needs research, or takes several steps, while
keeping every safety-bearing decision deterministic:

* cancel, status, and explicit "in the background" keep their offline
  whole-utterance matches and never reach the model;
* the deterministic long-work net is a floor, not a ceiling: what it already
  recognises is scheduled without an LLM round trip, and the model is asked
  only about turns it does not recognise;
* a closed set of conversational particles ("yes", "no thanks") is answered
  offline, so a one-word reply never pays for a model call;
* the model call is bounded, redacted, and fails safe to the deterministic
  net, so an unreachable model is never worse than today;
* nothing the router logs or speaks carries the request text or a task id.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from caal import background_tasks
from caal.background_task_session import (
    BACKGROUND_ACK_REPLY,
    BACKGROUND_BUSY_REPLY,
    BACKGROUND_CANCELLED_REPLY,
    BACKGROUND_STATUS_IDLE_REPLY,
    LONG_WORK_ACK_REPLY,
    LONG_WORK_OFFER_CALLBACK_REPLY,
    BackgroundTaskBridge,
    BackgroundTurnOutcome,
)
from caal.background_tasks import list_tasks
from caal.work_router import (
    MAX_ROUTER_INPUT_CHARS,
    WORK_ROUTER_SYSTEM_PROMPT,
    Route,
    RouteDecision,
    RouteSource,
    SemanticWorkRouter,
    deterministic_route,
    parse_route_label,
    provider_classifier,
)

# The turn that actually failed on the phone.
REAL_REQUEST = "Actually, can you create a PDF explaining what you are? It can be short."

# Work requests phrased the way people really speak. None of these are matched
# by the deterministic net, and no list of prefixes or artifact nouns would
# cover them; they are what the semantic stage exists for.
NATURAL_WORK_REQUESTS = [
    "While I grab a coffee, see if you can pull together everything we know "
    "about the Alberta grid and have it ready for me",
    "I was thinking it would be handy to have all of that in one place I could print out",
    "Go through the last quarter of invoices and figure out where the money actually went",
    "Something short I can hand to the board about what you do would be great",
    "Can you check the market and come up with a few options for a new laptop under two grand",
    "Take a pass at the onboarding doc and tighten it up",
    "Work out what it would cost to move all of this to Postgres and lay out the tradeoffs",
    "See what you can dig up about that company before my meeting",
    "Turn the notes from today into something I can send out",
]

NATURAL_CONVERSATION = [
    "what time is it",
    "turn off the kitchen lights",
    "what's the weather looking like tomorrow",
    "remind me to call my sister at six",
    "how are you feeling today",
    "who won the game last night",
]

WEB_ROOM = "room-web-router"


# --- fakes -------------------------------------------------------------------


class FakeClassifier:
    """Injected transport: records the messages it was asked to send."""

    def __init__(self, reply: str = '{"route": "work"}', *, delay: float = 0.0) -> None:
        self.reply = reply
        self.delay = delay
        self.calls: list[list[dict[str, str]]] = []
        self.cancelled = False

    async def __call__(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if self.delay:
            try:
                await asyncio.sleep(self.delay)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return self.reply

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_user_text(self) -> str:
        return self.calls[-1][-1]["content"]


class ExplodingClassifier:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error or RuntimeError("hermes is down")
        self.call_count = 0

    async def __call__(self, messages: list[dict[str, str]]) -> str:
        self.call_count += 1
        raise self.error


class FakeSession:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def say(self, text: str, **_: object) -> None:
        self.spoken.append(text)


class FakeExecute:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.gate = asyncio.Event()

    async def __call__(self, request: str, context: str) -> str:
        self.calls.append((request, context))
        await self.gate.wait()
        return "done"


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setattr(background_tasks, "STORE_PATH", tmp_path / "assistant.sqlite3")
    yield


async def _teardown(bridge: BackgroundTaskBridge) -> None:
    await bridge.abandon()
    await bridge.close()


# --- deterministic controls are never delegated to a model -------------------


@pytest.mark.parametrize(
    ("text", "route"),
    [
        ("cancel the background task", Route.CANCEL),
        ("jarvis, stop the background task please", Route.CANCEL),
        ("never mind about the background task", Route.CANCEL),
        ("how's the background task going", Route.STATUS),
        ("is the pdf ready yet", Route.STATUS),
        ("what's the status of the research", Route.STATUS),
        ("look into this in the background and get back to me", Route.BACKGROUND),
        ("work on that in the background", Route.BACKGROUND),
    ],
)
def test_control_commands_stay_deterministic(text: str, route: Route) -> None:
    decision = deterministic_route(text)
    assert decision.route is route
    assert decision.source in (RouteSource.CONTROL, RouteSource.EXPLICIT)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "cancel the background task",
        "how's the background task going",
        "is the pdf ready yet",
        "look into this in the background and get back to me",
    ],
)
async def test_the_router_never_asks_a_model_about_a_control_command(text: str) -> None:
    classifier = FakeClassifier()
    router = SemanticWorkRouter(classify=classifier)
    decision = await router.route(text)
    assert decision.route in (Route.CANCEL, Route.STATUS, Route.BACKGROUND)
    assert classifier.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    ["yes", "no", "yeah sure", "no thanks", "ok thanks", "hello", "goodbye", "jarvis?", ""],
)
async def test_conversational_particles_are_answered_offline(text: str) -> None:
    """A one-word reply must never cost a model round trip."""
    classifier = FakeClassifier()
    router = SemanticWorkRouter(classify=classifier)
    decision = await router.route(text)
    assert decision.route is Route.CONVERSATION
    assert classifier.call_count == 0


@pytest.mark.asyncio
async def test_work_the_deterministic_net_already_knows_costs_no_round_trip() -> None:
    """The offline net is a floor: what it recognises is scheduled immediately."""
    classifier = FakeClassifier()
    router = SemanticWorkRouter(classify=classifier)
    decision = await router.route(REAL_REQUEST)
    assert decision.route is Route.WORK
    assert decision.source is RouteSource.DETERMINISTIC
    assert decision.is_work is True
    assert classifier.call_count == 0


# --- the semantic stage ------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("text", NATURAL_WORK_REQUESTS)
async def test_natural_work_phrasings_reach_the_model_and_are_scheduled(text: str) -> None:
    assert background_tasks.long_running_work_inferred(text) is False  # regex cannot see it
    classifier = FakeClassifier('{"route": "work"}')
    router = SemanticWorkRouter(classify=classifier)
    decision = await router.route(text)
    assert decision == RouteDecision(route=Route.WORK, source=RouteSource.SEMANTIC)
    assert classifier.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("text", NATURAL_CONVERSATION)
async def test_ordinary_turns_stay_conversation_when_the_model_says_so(text: str) -> None:
    classifier = FakeClassifier('{"route": "conversation"}')
    router = SemanticWorkRouter(classify=classifier)
    decision = await router.route(text)
    assert decision == RouteDecision(route=Route.CONVERSATION, source=RouteSource.SEMANTIC)
    assert classifier.call_count == 1


@pytest.mark.asyncio
async def test_the_model_is_asked_with_a_bounded_redacted_two_message_prompt() -> None:
    classifier = FakeClassifier()
    router = SemanticWorkRouter(classify=classifier)
    await router.route("dig through the logs and my api_key = sk-abcdefgh12345678 " + "x" * 5_000)
    messages = classifier.calls[0]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == WORK_ROUTER_SYSTEM_PROMPT
    user = messages[1]["content"]
    assert len(user) <= MAX_ROUTER_INPUT_CHARS
    assert "sk-abcdefgh12345678" not in user
    assert "[REDACTED]" in user


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"route": "work"}', Route.WORK),
        ('{"route":"conversation"}', Route.CONVERSATION),
        ('```json\n{"route": "work"}\n```', Route.WORK),
        ('Sure! Here is the answer: {"route": "WORK"}', Route.WORK),
        ("work", Route.WORK),
        ("  Conversation  ", Route.CONVERSATION),
        ('{"route": "work", "confidence": 0.9}', Route.WORK),
        ('{"route": "maybe"}', None),
        ("", None),
        ("I am not sure what you mean.", None),
        ('{"route": "cancel"}', None),  # the model may not reach a control route
        ('{"route": "status"}', None),
        (None, None),
    ],
)
def test_label_parsing_is_strict_about_what_it_accepts(raw, expected) -> None:
    assert parse_route_label(raw) is expected


# --- fail-safe ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_slow_model_is_abandoned_within_the_timeout() -> None:
    classifier = FakeClassifier(delay=30.0)
    router = SemanticWorkRouter(classify=classifier, timeout_seconds=0.05)
    started = asyncio.get_running_loop().time()
    decision = await router.route("see what you can dig up about that company")
    elapsed = asyncio.get_running_loop().time() - started
    assert elapsed < 5.0
    assert decision.route is Route.CONVERSATION
    assert decision.source is RouteSource.FALLBACK
    assert classifier.cancelled is True


@pytest.mark.asyncio
async def test_an_unreachable_model_falls_back_to_the_deterministic_net() -> None:
    classifier = ExplodingClassifier()
    router = SemanticWorkRouter(classify=classifier)
    # The net still recognises what it always recognised.
    assert (await router.route(REAL_REQUEST)).route is Route.WORK
    # And an unrecognised turn degrades to conversation rather than raising.
    decision = await router.route("take a pass at the onboarding doc")
    assert decision == RouteDecision(route=Route.CONVERSATION, source=RouteSource.FALLBACK)
    assert (await router.route("cancel the background task")).route is Route.CANCEL


@pytest.mark.asyncio
async def test_an_undecidable_answer_falls_back_rather_than_guessing() -> None:
    classifier = FakeClassifier("I think maybe you should try it yourself")
    router = SemanticWorkRouter(classify=classifier)
    decision = await router.route("take a pass at the onboarding doc")
    assert decision == RouteDecision(route=Route.CONVERSATION, source=RouteSource.FALLBACK)


@pytest.mark.asyncio
async def test_a_router_with_no_classifier_is_the_deterministic_net() -> None:
    router = SemanticWorkRouter()
    assert router.semantic_enabled is False
    assert (await router.route(REAL_REQUEST)).route is Route.WORK
    decision = await router.route("take a pass at the onboarding doc")
    assert decision == RouteDecision(route=Route.CONVERSATION, source=RouteSource.DISABLED)


@pytest.mark.asyncio
async def test_the_semantic_stage_can_be_switched_off_without_removing_the_classifier() -> None:
    classifier = FakeClassifier()
    router = SemanticWorkRouter(classify=classifier, enabled=False)
    assert router.semantic_enabled is False
    assert (await router.route("take a pass at the onboarding doc")).route is Route.CONVERSATION
    assert classifier.call_count == 0


@pytest.mark.asyncio
async def test_routing_is_bounded_for_hostile_input() -> None:
    classifier = FakeClassifier('{"route": "conversation"}')
    router = SemanticWorkRouter(classify=classifier)
    for value in (None, 12, b"bytes", "   "):
        decision = await router.route(value)  # type: ignore[arg-type]
        assert decision.route is Route.CONVERSATION
    assert classifier.call_count == 0


# --- privacy -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_router_never_logs_the_request_text() -> None:
    """Observed on the router's own logger.

    ``voice_agent`` makes the ``caal`` logger non-propagating at import time,
    so a root-handler capture would silently see nothing here and pass for the
    wrong reason. Attaching to the module's logger tests what it actually emits.
    """
    secret_phrase = "the Alberta grid interconnection filing"
    request = f"pull together everything about {secret_phrase}"
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    router_logger = logging.getLogger("caal.work_router")
    handler = Capture()
    previous_level = router_logger.level
    router_logger.addHandler(handler)
    router_logger.setLevel(logging.DEBUG)
    try:
        await SemanticWorkRouter(classify=FakeClassifier('{"route": "work"}')).route(request)
        await SemanticWorkRouter(classify=ExplodingClassifier()).route(request)
        await SemanticWorkRouter(classify=FakeClassifier("no idea")).route(request)
        await SemanticWorkRouter(classify=FakeClassifier(delay=30.0), timeout_seconds=0.05).route(
            request
        )
    finally:
        router_logger.removeHandler(handler)
        router_logger.setLevel(previous_level)

    logged = " ".join(record.getMessage() for record in records)
    assert secret_phrase not in logged
    assert "Alberta" not in logged
    assert "hermes is down" not in logged  # nor an upstream error's own text
    # Every path is observable, and every path says only what it decided.
    assert len(records) == 4
    assert "semantically" in logged and logged.count("offline net") == 3


def test_a_decision_carries_no_request_text() -> None:
    decision = RouteDecision(route=Route.WORK, source=RouteSource.SEMANTIC)
    assert "Alberta" not in repr(decision)
    assert set(vars(decision)) == {"route", "source"}


# --- provider transport ------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_classifier_sends_a_plain_completion_and_returns_its_text() -> None:
    class FakeProvider:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        async def chat(self, messages, tools=None, **kwargs):
            self.calls.append((messages, tools, kwargs))
            return type("R", (), {"content": '{"route": "work"}'})()

    provider = FakeProvider()
    classify = provider_classifier(provider)
    assert await classify([{"role": "user", "content": "hi"}]) == '{"route": "work"}'
    messages, tools, _ = provider.calls[0]
    assert messages == [{"role": "user", "content": "hi"}]
    assert tools is None  # a router turn must not invite tool use


@pytest.mark.asyncio
async def test_provider_classifier_treats_an_empty_completion_as_undecided() -> None:
    class EmptyProvider:
        async def chat(self, messages, tools=None, **kwargs):
            return type("R", (), {"content": None})()

    assert await provider_classifier(EmptyProvider())([]) == ""


# --- the bridge, end to end --------------------------------------------------


@pytest.mark.asyncio
async def test_a_semantically_routed_request_is_scheduled_and_acknowledged(store) -> None:
    """No silent failure: the turn is queued and the acknowledgement is spoken."""
    execute = FakeExecute()
    classifier = FakeClassifier('{"route": "work"}')
    bridge = BackgroundTaskBridge(
        execute=execute,
        session_key=WEB_ROOM,
        work_router=SemanticWorkRouter(classify=classifier),
    )
    session = FakeSession()
    await bridge.start()
    try:
        request = "Take a pass at the onboarding doc and tighten it up"
        outcome = await bridge.process_turn(request, session)
        assert outcome == BackgroundTurnOutcome(consumed=True, scheduled=True)
        assert session.spoken == [LONG_WORK_ACK_REPLY]
        assert [task.request for task in list_tasks()] == [request]
    finally:
        execute.gate.set()
        await _teardown(bridge)


@pytest.mark.asyncio
async def test_the_phone_offers_the_callback_only_after_the_work_is_queued(store) -> None:
    execute = FakeExecute()
    bridge = BackgroundTaskBridge(
        execute=execute,
        session_key=WEB_ROOM,
        work_router=SemanticWorkRouter(classify=FakeClassifier('{"route": "work"}')),
    )
    session = FakeSession()
    await bridge.start()
    try:
        outcome = await bridge.process_turn(
            "See what you can dig up about that company before my meeting",
            session,
            offer_callback=True,
        )
        assert outcome == BackgroundTurnOutcome(
            consumed=True, scheduled=True, callback_offered=True
        )
        assert session.spoken == [LONG_WORK_OFFER_CALLBACK_REPLY]
    finally:
        execute.gate.set()
        await _teardown(bridge)


@pytest.mark.asyncio
async def test_work_that_cannot_be_queued_says_so_instead_of_going_silent(
    store, monkeypatch
) -> None:
    """The failure mode that started this: a work turn must never end in silence."""
    import caal.background_task_session as session_module

    def _boom(*_args, **_kwargs):
        raise RuntimeError("the queue is on fire")

    monkeypatch.setattr(session_module, "enqueue", _boom)
    bridge = BackgroundTaskBridge(
        execute=FakeExecute(),
        session_key=WEB_ROOM,
        work_router=SemanticWorkRouter(classify=FakeClassifier('{"route": "work"}')),
    )
    session = FakeSession()
    await bridge.start()
    try:
        outcome = await bridge.process_turn(
            "Turn the notes from today into something I can send out",
            session,
            offer_callback=True,
        )
        # Consumed with a spoken failure, and no callback offered for work
        # that was never scheduled.
        assert outcome == BackgroundTurnOutcome(
            consumed=True, scheduled=False, callback_offered=False
        )
        assert session.spoken == [BACKGROUND_BUSY_REPLY]
        assert list_tasks() == []
    finally:
        await _teardown(bridge)


@pytest.mark.asyncio
async def test_the_bridge_keeps_the_deterministic_controls_under_a_semantic_router(store) -> None:
    """Cancel and status are answered offline even when the model would disagree."""
    execute = FakeExecute()
    classifier = FakeClassifier('{"route": "work"}')
    bridge = BackgroundTaskBridge(
        execute=execute,
        session_key=WEB_ROOM,
        work_router=SemanticWorkRouter(classify=classifier),
    )
    session = FakeSession()
    await bridge.start()
    try:
        assert (await bridge.process_turn("look into this in the background", session)).scheduled
        before = classifier.call_count
        assert (await bridge.process_turn("cancel the background task", session)).consumed
        assert (await bridge.process_turn("how's the background task going", session)).consumed
        assert classifier.call_count == before
        assert session.spoken == [
            BACKGROUND_ACK_REPLY,
            BACKGROUND_CANCELLED_REPLY,
            BACKGROUND_STATUS_IDLE_REPLY,
        ]
    finally:
        execute.gate.set()
        await _teardown(bridge)


# --- runtime wiring ----------------------------------------------------------


def _load_voice_agent():
    """Load voice_agent.py once; it is a script, not an importable package."""
    import importlib.util
    from pathlib import Path

    global _VOICE_AGENT
    try:
        return _VOICE_AGENT
    except NameError:
        pass
    module_path = Path(__file__).parents[1] / "voice_agent.py"
    spec = importlib.util.spec_from_file_location("voice_agent_router_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _VOICE_AGENT = module
    return module


class RecordingProvider:
    def __init__(self, content: str = '{"route": "work"}') -> None:
        self.content = content
        self.calls: list[list[dict[str, str]]] = []

    async def chat(self, messages, tools=None, **_):
        self.calls.append(messages)
        return type("R", (), {"content": self.content})()


def test_router_settings_have_conservative_defaults() -> None:
    from caal import settings as settings_module

    defaults = settings_module.DEFAULT_SETTINGS
    assert defaults["work_router_enabled"] is True
    assert 0.5 <= defaults["work_router_timeout_seconds"] <= 10.0


@pytest.mark.asyncio
async def test_the_runtime_builds_a_router_backed_by_the_configured_llm(store) -> None:
    from caal import settings as settings_module

    voice_agent = _load_voice_agent()
    provider = RecordingProvider()
    runtime = {
        **settings_module.DEFAULT_SETTINGS,
        "background_tasks_enabled": True,
        "telegram_bot_token": "",
        "telegram_chat_id": "",
    }
    bridge = voice_agent.build_background_task_bridge(
        runtime, provider=provider, session_key=WEB_ROOM
    )
    session = FakeSession()
    await bridge.start()
    try:
        outcome = await bridge.process_turn(
            "Take a pass at the onboarding doc and tighten it up", session
        )
        assert outcome.scheduled is True
        assert session.spoken == [LONG_WORK_ACK_REPLY]
        # The router asked the configured provider, with the router prompt.
        assert provider.calls[0][0]["content"] == WORK_ROUTER_SYSTEM_PROMPT
    finally:
        await _teardown(bridge)


@pytest.mark.asyncio
async def test_runtime_unwraps_caal_llm_to_use_its_provider_for_semantic_routing(store) -> None:
    from caal import settings as settings_module

    voice_agent = _load_voice_agent()
    inner = RecordingProvider()

    class CAALLLMWrapper:
        provider_instance = inner

    runtime = {
        **settings_module.DEFAULT_SETTINGS,
        "background_tasks_enabled": True,
        "telegram_bot_token": "",
        "telegram_chat_id": "",
    }
    bridge = voice_agent.build_background_task_bridge(
        runtime, provider=CAALLLMWrapper(), session_key=WEB_ROOM
    )
    session = FakeSession()
    await bridge.start()
    try:
        outcome = await bridge.process_turn("Take a pass at the onboarding doc and tighten it up", session)
        assert outcome.scheduled is True
        assert inner.calls[0][0]["content"] == WORK_ROUTER_SYSTEM_PROMPT
    finally:
        await _teardown(bridge)


@pytest.mark.asyncio
async def test_the_runtime_can_switch_the_semantic_stage_off(store) -> None:
    from caal import settings as settings_module

    voice_agent = _load_voice_agent()
    provider = RecordingProvider()
    runtime = {
        **settings_module.DEFAULT_SETTINGS,
        "background_tasks_enabled": True,
        "work_router_enabled": False,
        "telegram_bot_token": "",
        "telegram_chat_id": "",
    }
    bridge = voice_agent.build_background_task_bridge(
        runtime, provider=provider, session_key=WEB_ROOM
    )
    session = FakeSession()
    await bridge.start()
    try:
        assert (
            await bridge.process_turn("Take a pass at the onboarding doc", session)
        ).consumed is False
        assert provider.calls == []
        # The offline net is untouched by the switch.
        assert (await bridge.process_turn(REAL_REQUEST, session)).scheduled is True
    finally:
        await _teardown(bridge)


@pytest.mark.asyncio
async def test_the_default_bridge_still_uses_the_offline_net(store) -> None:
    """Existing deployments keep working with no router injected at all."""
    execute = FakeExecute()
    bridge = BackgroundTaskBridge(execute=execute, session_key=WEB_ROOM)
    session = FakeSession()
    await bridge.start()
    try:
        assert (await bridge.process_turn(REAL_REQUEST, session)).scheduled is True
        assert (await bridge.process_turn("what time is it", session)).consumed is False
    finally:
        execute.gate.set()
        await _teardown(bridge)
