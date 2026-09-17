"""Stage 3: the language decision is read semantically, not spelled out in word lists.

Stage 2 ended with the language chosen by function-word lists, character classes
and five regexes (reports/bilingual/STAGE2.md §3.3). This stage moves the primary
reading onto the inference the local model **already makes for every turn** -- the
work router's classification call -- by widening the JSON it is already asked for
with two more fields, ``reply_language`` and ``language_switch``. No second model
call is added: the router's request per turn goes from one to one.

Everything the reading says is then validated deterministically here, in code:
only ``en`` and ``es`` exist, anything else is *no reading at all*, and a turn
with no usable reading falls back to the stage-2 speech-server evidence path
rather than to a guess.

No test in this file makes a network call or reaches a model.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from caal.language_policy import (
    EN,
    ES,
    LanguageReading,
    LanguageSession,
    read_semantic,
    resolve,
)
from caal.work_router import (
    WORK_ROUTER_SYSTEM_PROMPT,
    Route,
    RouteSource,
    SemanticWorkRouter,
    parse_language_reading,
)


# --- 1. The reading is validated in code, not trusted from the model ----------


def test_only_en_and_es_are_readable_languages():
    assert read_semantic("es", switch=False) == LanguageReading(ES, False)
    assert read_semantic("en", switch=False) == LanguageReading(EN, False)


@pytest.mark.parametrize(
    "value",
    ["fr", "pt", "EN", "ES", "spanish", "english", "", "  ", "auto", "unknown", None, 7, ["es"]],
)
def test_anything_that_is_not_exactly_en_or_es_is_not_a_reading(value):
    """A model may answer anything. Only two strings mean something here."""
    assert read_semantic(value, switch=False) is None


def test_a_switch_with_no_language_is_not_a_reading():
    assert read_semantic(None, switch=True) is None


def test_a_switch_intent_is_carried_separately_from_the_language():
    reading = read_semantic("es", switch=True)
    assert reading == LanguageReading(ES, True)
    assert reading.switch_requested is True


# --- 2. Parsing the router's existing reply ----------------------------------


def test_the_router_prompt_asks_for_the_language_fields():
    assert "reply_language" in WORK_ROUTER_SYSTEM_PROMPT
    assert "language_switch" in WORK_ROUTER_SYSTEM_PROMPT


def test_a_full_router_reply_yields_route_reasoning_and_language():
    raw = json.dumps(
        {
            "route": "conversation",
            "reasoning": False,
            "reply_language": "es",
            "language_switch": False,
        }
    )
    assert parse_language_reading(raw) == LanguageReading(ES, False)


def test_an_explicit_switch_is_read_from_the_same_reply():
    raw = '{"route": "conversation", "reply_language": "es", "language_switch": true}'
    assert parse_language_reading(raw) == LanguageReading(ES, True)


@pytest.mark.parametrize(
    "raw",
    [
        "conversation",
        '{"route": "conversation"}',
        '{"route": "conversation", "reply_language": "fr"}',
        '{"route": "conversation", "reply_language": "spanish"}',
        '{"route": "conversation", "reply_language": null}',
        "I'm sorry, I can't help with that.",
        "",
        None,
        17,
    ],
)
def test_a_reply_without_a_usable_language_reads_as_no_reading(raw):
    assert parse_language_reading(raw) is None


def test_a_non_boolean_switch_is_not_a_switch():
    raw = '{"reply_language": "es", "language_switch": "yes"}'
    assert parse_language_reading(raw) == LanguageReading(ES, False)


# --- 3. The reading rides the existing routing decision ----------------------


def _router(reply: str) -> SemanticWorkRouter:
    async def classify(messages):
        return reply

    return SemanticWorkRouter(classify=classify, timeout_seconds=1.0)


def test_the_routing_decision_carries_the_language_reading():
    raw = '{"route": "conversation", "reply_language": "es", "language_switch": false}'
    decision = asyncio.run(_router(raw).route("cuál es mi próxima reunión"))
    assert decision.route is Route.CONVERSATION
    assert decision.source is RouteSource.SEMANTIC
    assert decision.language == LanguageReading(ES, False)


def test_a_deterministic_route_carries_no_reading():
    """The offline net never consults a model, so it has nothing to report."""
    decision = asyncio.run(
        _router('{"reply_language": "es"}').route("cancel the background task")
    )
    assert decision.route is Route.CANCEL
    assert decision.language is None


def test_an_unreachable_model_yields_no_reading_rather_than_a_guess():
    async def classify(messages):
        raise ConnectionError("no local model")

    decision = asyncio.run(SemanticWorkRouter(classify=classify).route("apaga la luz"))
    assert decision.source is RouteSource.FALLBACK
    assert decision.language is None


def test_a_timed_out_model_yields_no_reading():
    async def classify(messages):
        await asyncio.sleep(5)
        return '{"route": "conversation", "reply_language": "es"}'

    decision = asyncio.run(
        SemanticWorkRouter(classify=classify, timeout_seconds=0.1).route("apaga la luz")
    )
    assert decision.language is None


def test_a_reply_with_a_language_but_no_route_still_yields_the_reading():
    """The route falls back; the language reading is independent of it."""
    decision = asyncio.run(_router('{"route": "banana", "reply_language": "es"}').route("hola qué tal"))
    assert decision.source is RouteSource.FALLBACK
    assert decision.language == LanguageReading(ES, False)


# --- 4. What the reading is allowed to decide --------------------------------


def test_the_semantic_reading_outranks_the_speech_server_detection():
    """This is the point of the stage: meaning decides, not orthography."""
    # No Spanish function word, no diacritic, no inverted punctuation: the
    # stage-2 grammar cannot see this, and the speech server misheard it.
    language = resolve(
        "auto",
        detected="en",
        transcript="ponme un recordatorio",
        current=EN,
        semantic=LanguageReading(ES, False),
    )
    assert language == ES


def test_an_english_turn_with_spanish_names_stays_english():
    assert (
        resolve(
            "auto",
            detected="es",
            transcript="call José about the piñata order",
            current=EN,
            semantic=LanguageReading(EN, False),
        )
        == EN
    )


def test_a_short_acknowledgement_never_moves_the_session():
    """``sí``/``okay`` are contextual: they keep whatever is being spoken."""
    for transcript in ("sí", "okay", "see", "vale"):
        assert (
            resolve(
                "auto",
                detected="en",
                transcript=transcript,
                current=ES,
                semantic=LanguageReading(EN, False),
            )
            == ES
        )


def test_a_pinned_preference_beats_an_ordinary_reading():
    assert (
        resolve(
            "en",
            detected="es",
            transcript="apaga la luz",
            current=EN,
            semantic=LanguageReading(ES, False),
        )
        == EN
    )


def test_an_explicit_switch_beats_a_pinned_preference():
    assert (
        resolve(
            "en",
            detected="es",
            transcript="a partir de ahora contéstame en español",
            current=EN,
            semantic=LanguageReading(ES, True),
        )
        == ES
    )


def test_an_explicit_switch_is_honoured_even_on_a_short_turn():
    assert (
        resolve(
            "auto",
            detected="en",
            transcript="español",
            current=EN,
            semantic=LanguageReading(ES, True),
        )
        == ES
    )


def test_no_reading_falls_back_to_the_speech_server_evidence():
    """A failed local inference costs the semantic reading and nothing else."""
    assert resolve("auto", detected="es", transcript="apaga la luz", current=EN, semantic=None) == ES
    assert (
        resolve(
            "auto",
            detected="en",
            transcript="call José about the piñata order",
            current=EN,
            semantic=None,
        )
        == EN
    )


def test_no_reading_and_no_evidence_retains_the_current_language():
    assert resolve("auto", detected=None, transcript="mm hm", current=ES, semantic=None) == ES


# --- 5. The session folds the reading in -------------------------------------


def test_a_session_follows_the_reading_and_then_stays_there():
    session = LanguageSession("auto")
    assert session.observe("ponme un recordatorio", "en", semantic=LanguageReading(ES, False)) == ES
    # The next turn is a bare acknowledgement with no reading at all.
    assert session.observe("okay", "en", semantic=None) == ES


def test_a_session_switch_sets_the_override_for_this_session_only():
    session = LanguageSession("en")
    assert session.observe("español por favor", "es", semantic=LanguageReading(ES, True)) == ES
    assert session.override == ES
    # The stored account preference is never written from a conversational turn.
    assert session.preference == "en"


def test_two_sessions_do_not_share_a_reading():
    mine, theirs = LanguageSession("auto"), LanguageSession("auto")
    mine.observe("quiero ver mis correos", "en", semantic=LanguageReading(ES, False))
    assert mine.current == ES
    assert theirs.current == EN


def test_a_session_with_no_reading_behaves_exactly_as_stage_two_did():
    session = LanguageSession("auto")
    assert session.observe("apaga la luz", "es") == ES
    assert session.observe("what is on my calendar today", "en") == EN


# --- 6. The reading can authorize nothing but a language ---------------------


def test_the_reading_never_confirms_anything():
    """A language reading is not consent. ``see``/``sí`` still authorize nothing."""
    from caal import handoff_intent

    assert "see" not in handoff_intent._CONFIRMATIONS
    reading = read_semantic("es", switch=True)
    assert set(vars(reading)) == {"language", "switch_requested"}
