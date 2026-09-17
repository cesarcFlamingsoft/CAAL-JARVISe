"""Stage 3 §5.4: the same turn's language binds *before* that turn's own fixed reply.

The lag this closes is real and was pinned as a gap: the durable-work route
reads the turn on the local model, that same reply says which language to
answer in, and the acknowledgement was spoken *before* the reading was applied.
So the first Spanish turn was acknowledged in English and only the next one was
right.

The fix is ordering, not a new signal: the reading is bound at the action
boundary -- after the router answered, before anything is said -- and it is
bound by *value*, on the shared session object and as an explicit language, not
by mutating a context variable inside whichever task happened to do the reading.

Two turns running concurrently in their own tasks must not see each other's
language, which is what a context variable gives when it is set per task and
never leaked through a shared default.

No test in this file makes a network call or reaches a model.
"""

from __future__ import annotations

import asyncio

import pytest

# Queuing durable work crosses the administrator delegation boundary, which is
# tested on its own in test_delegation_boundaries and test_durable_work. These
# turns are about which *language* the acknowledgement is spoken in, so they use
# the established component fixture rather than a second identity runtime; the
# real authority check is untouched and still runs everywhere else.
pytestmark = pytest.mark.usefixtures("isolated_harness_components")

from caal import background_tasks, reply_localization  # noqa: E402
from caal.background_task_session import (  # noqa: E402
    LONG_WORK_ACK_REPLY,
    BackgroundTaskBridge,
)
from caal.language_policy import (  # noqa: E402
    EN,
    ES,
    LanguageReading,
    LanguageSession,
)
from caal.reply_localization import reply_language, spanish_pairs  # noqa: E402
from caal.work_router import SemanticWorkRouter  # noqa: E402


@pytest.fixture(autouse=True)
def throwaway_queue(monkeypatch, tmp_path):
    """Queue these turns into a scratch database, never the deployment's own."""
    monkeypatch.setattr(background_tasks, "STORE_PATH", tmp_path / "assistant.sqlite3")


class _Session:
    def __init__(self) -> None:
        self.said: list[str] = []

    async def say(self, text, **kwargs):
        self.said.append(text)


class _Agent:
    def __init__(self, preference: str = "auto") -> None:
        self._language_session = LanguageSession(preference)


def _bridge(reply: str, calls: list | None = None) -> BackgroundTaskBridge:
    async def classify(messages):
        if calls is not None:
            calls.append(messages)
        return reply

    async def execute(request):
        return "done"

    return BackgroundTaskBridge(
        execute=execute,
        session_key="room-1",
        work_router=SemanticWorkRouter(classify=classify, timeout_seconds=1.0),
    )


WORK_ES = '{"route": "work", "reply_language": "es", "language_switch": false}'
SWITCH_ES = '{"route": "work", "reply_language": "es", "language_switch": true}'
WORK_EN = '{"route": "work", "reply_language": "en", "language_switch": false}'


def _binder(agent, spoken_language: list[str]):
    """The production binding, in miniature: shared session first, then this task."""
    from voice_agent import apply_language_reading

    def bind(reading, text):
        language = apply_language_reading(agent, reading, text)
        spoken_language.append(reply_localization.begin_turn(agent, language=language))

    return bind


# --- 1. The binding happens before the acknowledgement is spoken --------------


def test_the_first_spanish_work_turn_is_acknowledged_in_spanish():
    agent, session, calls = _Agent(), _Session(), []
    bridge, spoken = _bridge(WORK_ES, calls), []

    outcome = asyncio.run(
        bridge.process_turn(
            "hazme un documento con todo lo que encuentres",
            session,
            bind_language=_binder(agent, spoken),
        )
    )

    assert outcome.consumed is True
    assert agent._language_session.current == ES
    assert session.said == [spanish_pairs()[LONG_WORK_ACK_REPLY]]
    # One turn, one classification. The language costs no second request.
    assert len(calls) == 1


def test_an_english_turn_still_says_the_english_constant():
    """A reading of "en" moves a Spanish session back, on that same turn."""
    agent, session = _Agent(), _Session()
    agent._language_session.current = ES
    token = reply_language.set(ES)
    try:
        asyncio.run(
            _bridge(WORK_EN).process_turn(
                "tell me everything you can find about the Lisbon office",
                session,
                bind_language=_binder(agent, []),
            )
        )
    finally:
        reply_language.reset(token)
    assert agent._language_session.current == EN
    assert session.said == [LONG_WORK_ACK_REPLY]


def test_work_the_offline_net_recognised_carries_no_reading_and_no_model_call():
    """Unchanged and deliberate: the deterministic net consults nothing, so a
    turn it claims is acknowledged in the language the session already had."""
    agent, session, calls = _Agent(), _Session(), []
    outcome = asyncio.run(
        _bridge(WORK_EN, calls).process_turn(
            "write me a document about all of it",
            session,
            bind_language=_binder(agent, []),
        )
    )
    assert calls == []
    assert outcome.language is None
    assert session.said == [LONG_WORK_ACK_REPLY]


def test_a_turn_with_no_binder_behaves_exactly_as_before():
    session = _Session()
    outcome = asyncio.run(
        _bridge(WORK_ES).process_turn("hazme un documento con todo", session)
    )
    assert outcome.language == LanguageReading(ES, False)
    assert session.said == [LONG_WORK_ACK_REPLY]


def test_a_broken_binder_never_breaks_the_turn():
    def bind(reading, text):
        raise RuntimeError("binding failed")

    session = _Session()
    outcome = asyncio.run(
        _bridge(WORK_ES).process_turn("hazme un documento", session, bind_language=bind)
    )
    assert outcome.consumed is True


# --- 2. The value crosses the task boundary, the context variable does not ----


def test_the_language_is_bound_by_value_not_only_in_the_reading_task():
    """A context variable set inside the reading's task is invisible outside it."""
    agent, session, spoken = _Agent(), _Session(), []
    bridge = _bridge(WORK_ES)

    async def run():
        # The reading happens in its own task, exactly as the speech path does.
        task = asyncio.create_task(
            bridge.process_turn(
                "hazme un documento", session, bind_language=_binder(agent, spoken)
            )
        )
        await task
        # Outside that task the context variable is untouched...
        assert reply_localization.current_language() == EN
        # ...but the session object carries the decision, so the next reader of
        # session state -- the LLM node, the next local route -- sees Spanish.
        assert agent._language_session.current == ES

    asyncio.run(run())
    assert spoken == [ES]


def test_an_explicit_language_outranks_a_stale_session_read():
    agent = _Agent()
    agent._language_session.current = EN
    assert reply_localization.begin_turn(agent, language=ES) == ES
    assert reply_localization.current_language() == ES
    reply_language.set(EN)


def test_an_unusable_explicit_language_falls_back_to_the_session():
    agent = _Agent()
    agent._language_session.current = ES
    assert reply_localization.begin_turn(agent, language="fr") == ES
    reply_language.set(EN)


# --- 3. Interleaved sessions do not see each other's language ----------------


def test_concurrent_english_and_spanish_turns_stay_isolated():
    spanish_agent, english_agent = _Agent(), _Agent()
    spanish_session, english_session = _Session(), _Session()

    async def run():
        async def spanish():
            await _bridge(WORK_ES).process_turn(
                "hazme un documento con todo lo que encuentres",
                spanish_session,
                bind_language=_binder(spanish_agent, []),
            )

        async def english():
            await asyncio.sleep(0)
            await _bridge(WORK_EN).process_turn(
                "tell me everything you can find about the Lisbon office",
                english_session,
                bind_language=_binder(english_agent, []),
            )

        await asyncio.gather(
            asyncio.create_task(spanish()), asyncio.create_task(english())
        )

    asyncio.run(run())

    assert spanish_session.said == [spanish_pairs()[LONG_WORK_ACK_REPLY]]
    assert english_session.said == [LONG_WORK_ACK_REPLY]
    assert spanish_agent._language_session.current == ES
    assert english_agent._language_session.current == EN


def test_a_switch_turn_is_answered_in_the_new_language_on_that_same_turn():
    """"From now on answer me in Spanish" is honoured now, not from the next turn."""
    agent, session = _Agent("en"), _Session()
    asyncio.run(
        _bridge(SWITCH_ES).process_turn(
            "de ahora en adelante contéstame en español y hazme un documento con todo",
            session,
            bind_language=_binder(agent, []),
        )
    )
    assert agent._language_session.override == ES
    assert session.said == [spanish_pairs()[LONG_WORK_ACK_REPLY]]
