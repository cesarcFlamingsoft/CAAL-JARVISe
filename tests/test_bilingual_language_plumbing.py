"""Stage 3: the local model's language reading actually reaches the session.

A reading nobody acts on is not a feature. This pins the path the reading takes
from the one model call that already happens per turn to the ``LanguageSession``
the LLM node reads its directive from:

    work router reply -> RouteDecision.language -> BackgroundTurnOutcome.language
    -> LocalTurnHandler -> agent._language_session

and the two properties that path has to have: it must cross the asyncio task
boundary the speech path puts in the middle of it, and it must never reach a
model that is not the local one.

No network call and no model call is made here.
"""

from __future__ import annotations

import asyncio
import contextvars

import pytest

from caal.background_task_session import BackgroundTaskBridge
from caal.language_policy import EN, ES, LanguageReading, LanguageSession
from caal.work_router import SemanticWorkRouter, request_reasoning


# --- 1. The bridge reports what the router read ------------------------------


def _bridge(reply: str) -> BackgroundTaskBridge:
    async def classify(messages):
        return reply

    async def execute(request):  # pragma: no cover - never reached by these turns
        raise AssertionError("no work should be scheduled by these turns")

    return BackgroundTaskBridge(
        execute=execute,
        session_key="room-1",
        work_router=SemanticWorkRouter(classify=classify, timeout_seconds=1.0),
    )


class _Session:
    def __init__(self) -> None:
        self.said: list[str] = []

    async def say(self, text, **kwargs):
        self.said.append(text)


def test_an_unconsumed_conversational_turn_still_reports_the_language():
    """The common case: the turn is ordinary conversation and goes to the LLM."""
    raw = '{"route": "conversation", "reply_language": "es", "language_switch": false}'
    outcome = asyncio.run(_bridge(raw).process_turn("ponme un recordatorio", _Session()))
    assert outcome.consumed is False
    assert outcome.language == LanguageReading(ES, False)


def test_a_turn_the_router_could_not_read_reports_no_language():
    outcome = asyncio.run(_bridge("conversation").process_turn("hello there", _Session()))
    assert outcome.language is None


def test_a_company_private_turn_reports_no_language_and_asks_no_model():
    """A private session queues nothing and consults nothing. Unchanged."""
    from caal import company_privacy

    asked = []

    async def classify(messages):
        asked.append(messages)
        return '{"route": "conversation", "reply_language": "es"}'

    async def execute(request):  # pragma: no cover
        raise AssertionError("private turns are never scheduled")

    bridge = BackgroundTaskBridge(
        execute=execute,
        session_key="room-1",
        work_router=SemanticWorkRouter(classify=classify, timeout_seconds=1.0),
    )
    token = company_privacy._turn_pin.set(True)
    try:
        assert company_privacy.is_local_only() is True
        outcome = asyncio.run(bridge.process_turn("cuáles son las políticas", _Session()))
    finally:
        company_privacy._turn_pin.reset(token)
    assert outcome.consumed is False
    assert outcome.language is None
    assert asked == []


# --- 2. Applying a reading to a session --------------------------------------


class _Agent:
    def __init__(self, preference: str = "auto") -> None:
        self._language_session = LanguageSession(preference)


def test_applying_a_reading_moves_the_session():
    from voice_agent import apply_language_reading

    agent = _Agent()
    apply_language_reading(agent, LanguageReading(ES, False), "ponme un recordatorio")
    assert agent._language_session.current == ES


def test_applying_no_reading_leaves_the_session_alone():
    from voice_agent import apply_language_reading

    agent = _Agent()
    agent._language_session.current = ES
    apply_language_reading(agent, None, "okay")
    assert agent._language_session.current == ES


def test_applying_a_reading_to_an_agent_without_a_session_is_harmless():
    from voice_agent import apply_language_reading

    apply_language_reading(object(), LanguageReading(ES, True), "en español")


def test_a_switch_reading_beats_a_pinned_preference_through_the_helper():
    from voice_agent import apply_language_reading

    agent = _Agent("en")
    apply_language_reading(agent, LanguageReading(ES, True), "contéstame en español")
    assert agent._language_session.current == ES
    # The switch pins the session exactly as the account preference did, so one
    # English turn does not quietly undo what the user asked for...
    apply_language_reading(agent, LanguageReading(EN, False), "what time is it")
    assert agent._language_session.current == ES
    # ...and only another explicit switch takes it back.
    apply_language_reading(agent, LanguageReading(EN, True), "english please")
    assert agent._language_session.current == EN


# --- 3. It has to cross the task boundary ------------------------------------


def test_the_reading_survives_the_task_boundary_a_context_variable_does_not():
    """The speech path reads the turn inside ``asyncio.create_task``.

    A ``ContextVar`` set in that task is invisible to the caller -- which is why
    ``request_reasoning`` is already returned by value there. The language
    reading is carried the same way, on the outcome, and this test demonstrates
    the difference rather than asserting it.
    """
    from voice_agent import apply_language_reading

    agent = _Agent()
    raw = '{"route": "conversation", "reply_language": "es", "language_switch": false}'
    bridge = _bridge(raw)

    async def scenario():
        request_reasoning.set(None)

        async def child():
            # Stand-in for LocalTurnHandler._handle_speech_commands.
            request_reasoning.set(True)
            outcome = await bridge.process_turn("ponme un recordatorio", _Session())
            apply_language_reading(agent, outcome.language, "ponme un recordatorio")
            return outcome.language

        carried = await asyncio.create_task(child(), context=contextvars.copy_context())
        return carried, request_reasoning.get()

    carried, leaked = asyncio.run(scenario())
    # The value carried out of the task arrived; the context variable did not.
    assert carried == LanguageReading(ES, False)
    assert leaked is None
    assert agent._language_session.current == ES


# --- 4. The reading can only ever come from the local model ------------------


def test_the_router_classifier_always_runs_on_the_local_model():
    """No cloud, by construction: the router is handed the primary provider."""
    from voice_agent import work_router_provider

    class Primary:
        name = "local"

    class Escalation:
        name = "hermes"

    class Routed:
        primary = Primary()
        escalation = Escalation()

    class CAALLLM:
        provider_instance = Routed()

    assert work_router_provider(CAALLLM()) is Routed.primary
    assert work_router_provider(Routed()) is Routed.primary


@pytest.mark.parametrize("field", ["reply_language", "language_switch"])
def test_the_language_question_is_asked_in_the_one_request_already_made(field):
    """One request per turn, before and after. The language rides along."""
    sent: list[list[dict[str, str]]] = []

    async def classify(messages):
        sent.append(messages)
        return '{"route": "conversation", "reply_language": "es"}'

    decision = asyncio.run(
        SemanticWorkRouter(classify=classify, timeout_seconds=1.0).route("hola qué tal")
    )
    assert decision.language == LanguageReading(ES, False)
    assert len(sent) == 1
    assert field in sent[0][0]["content"]
