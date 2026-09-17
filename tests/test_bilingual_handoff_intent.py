"""Stage 3 §5.1: a Spanish request to keep talking by phone reaches the same
bounded confirmation controller an English one does.

The reading is semantic -- the local model says whether the turn asks for a
handoff -- because a list of Spanish magic phrases is always one phrasing
behind the speaker, exactly as it was in English. What the model says is then
*narrowed* by deterministic guards that run on the words themselves and can
only ever subtract:

* a hypothetical, an explanation or a quoted phrase is never a request,
* a call aimed at somebody else is never a handoff,
* a turn carrying a number proposes a destination, which this feature refuses,
* a turn that names no phone and no reason to leave the keyboard is not a
  handoff however the model read it.

And the whole point: **the model never authorizes a dial.** A positive reading
can only ask a question. The call still needs the fixed, exact confirmation
vocabulary answered in the turn right after that question, with the pending
scope open and the destination resolved server-side.

No test in this file makes a network call or reaches a model.
"""

from __future__ import annotations

import asyncio

import pytest
from livekit.agents import llm

from caal.handoff_intent import (
    ASK_CLARIFICATION_REPLY,
    ASK_CONFIRMATION_REPLY,
    STARTING_REPLY,
    HandoffIntent,
    PhoneHandoffController,
    classify_handoff_intent,
    handoff_surface_present,
    semantic_handoff_intent,
)
from caal.handoff_semantics import SemanticHandoffReader
from caal.language_policy import ES, LanguageReading
from caal.reply_localization import reply_language, spanish_pairs

APPROVED = "+17805558345"

SPANISH_REQUESTS = (
    "sigamos por teléfono",
    "llámame al teléfono",
    "¿podemos seguir esta conversación por teléfono?",
    "mejor continuemos esto en el celular, ya voy saliendo",
)


class FakeSession:
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.history = llm.ChatContext.empty()

    async def say(self, text: str, **kwargs) -> None:
        self.spoken.append(text)
        self.history.add_message(role="assistant", content=text)


class FakeCall:
    def __init__(self) -> None:
        self.destinations: list[str] = []

    async def __call__(self, destination: str, **kwargs) -> object:
        self.destinations.append(destination)
        return object()


def _reader(label: str, *, language: str = "es", calls: list | None = None):
    async def classify(messages):
        if calls is not None:
            calls.append(messages)
        return (
            '{"handoff": "%s", "reply_language": "%s", "language_switch": false}'
            % (label, language)
        )

    return SemanticHandoffReader(classify=classify, timeout_seconds=1.0)


# --- 1. The guards subtract; they never add ----------------------------------


@pytest.mark.parametrize("text", SPANISH_REQUESTS)
def test_a_spanish_request_has_a_handoff_surface(text):
    assert handoff_surface_present(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "¿qué hora es?",
        "sí",
        "ponme un recordatorio para mañana",
        "what's the weather like",
    ],
)
def test_a_turn_about_anything_else_has_no_handoff_surface(text):
    assert handoff_surface_present(text) is False


@pytest.mark.parametrize(
    "text",
    [
        # Hypothetical and instructional.
        "si te digo sigamos por teléfono, ¿qué haces?",
        "por ejemplo, podría decir llámame al teléfono",
        "¿cómo funciona lo de llamarme al teléfono?",
        '¿reconoces la frase "sigamos por teléfono"?',
        # Somebody else's call.
        "llama a mi esposa al teléfono",
        "llámalo por teléfono cuando puedas",
        # A destination proposed by the caller.
        "llámame al teléfono 780 555 8345",
        # A refusal.
        "no me llames al teléfono",
        # Not a handoff at all, whatever the model says.
        "ponme un recordatorio para mañana",
    ],
)
def test_a_guard_overrides_a_positive_model_reading(text):
    """The model may say "direct" about any of these. None of them may pass."""
    assert semantic_handoff_intent(text, "direct") is HandoffIntent.NONE
    assert semantic_handoff_intent(text, "clarify") is HandoffIntent.NONE


@pytest.mark.parametrize("label", ["yes", "DIRECT ", "", "si", None, 1, ["direct"], "none"])
def test_only_the_three_exact_labels_are_readings(label):
    assert semantic_handoff_intent("sigamos por teléfono", label) is HandoffIntent.NONE


@pytest.mark.parametrize("text", SPANISH_REQUESTS)
def test_a_guarded_positive_reading_survives(text):
    assert semantic_handoff_intent(text, "direct") is HandoffIntent.DIRECT
    assert semantic_handoff_intent(text, "clarify") is HandoffIntent.CLARIFY


# --- 2. The reader is bounded and fails closed -------------------------------


def test_the_reader_reads_the_intent_and_the_language_from_one_reply():
    reading = asyncio.run(_reader("direct").read("sigamos por teléfono"))
    assert reading.intent is HandoffIntent.DIRECT
    assert reading.language == LanguageReading(ES, False)


def test_a_turn_with_no_handoff_surface_costs_no_model_call():
    calls: list = []
    reading = asyncio.run(_reader("direct", calls=calls).read("¿qué hora es?"))
    assert calls == []
    assert reading.intent is HandoffIntent.NONE


def test_an_unreachable_model_reads_nothing_and_does_not_raise():
    async def classify(messages):
        raise RuntimeError("no model here")

    reader = SemanticHandoffReader(classify=classify, timeout_seconds=1.0)
    reading = asyncio.run(reader.read("sigamos por teléfono"))
    assert reading.intent is HandoffIntent.NONE
    assert reading.language is None


def test_a_slow_model_is_abandoned_rather_than_waited_on():
    async def classify(messages):
        await asyncio.sleep(5)
        return '{"handoff": "direct"}'

    reader = SemanticHandoffReader(classify=classify, timeout_seconds=0.1)
    reading = asyncio.run(reader.read("sigamos por teléfono"))
    assert reading.intent is HandoffIntent.NONE


def test_the_reader_never_sends_a_number_it_was_given():
    calls: list = []
    asyncio.run(_reader("direct", calls=calls).read("llámame al 780 555 8345"))
    assert calls == []


# --- 3. The Spanish path reaches the same bounded controller -----------------


def _controller(call, reader, *, bind=None) -> PhoneHandoffController:
    return PhoneHandoffController(
        start_call=call,
        allowed_destinations=APPROVED,
        semantic=reader,
        bind_language=bind,
    )


@pytest.mark.parametrize("text", SPANISH_REQUESTS)
def test_a_spanish_request_then_si_places_exactly_one_approved_call(text):
    session, call = FakeSession(), FakeCall()
    controller = _controller(call, _reader("direct"))

    async def run():
        first = await controller.handle_final_transcript(text, session)
        assert first is True
        assert call.destinations == []
        assert controller.awaiting_confirmation is True
        return await controller.handle_final_transcript("sí", session)

    token = reply_language.set(ES)
    try:
        assert asyncio.run(run()) is True
    finally:
        reply_language.reset(token)

    assert call.destinations == [APPROVED]
    assert session.spoken == [
        spanish_pairs()[ASK_CONFIRMATION_REPLY],
        spanish_pairs()[STARTING_REPLY],
    ]


def test_si_with_nothing_pending_calls_nobody():
    session, call = FakeSession(), FakeCall()
    # Even a model that answers "direct" to a bare "sí" cannot make it one.
    controller = _controller(call, _reader("direct"))

    consumed = asyncio.run(controller.handle_final_transcript("sí", session))

    assert consumed is False
    assert call.destinations == []
    assert session.spoken == []


@pytest.mark.parametrize(
    "text",
    [
        "si te digo sigamos por teléfono, ¿qué haces?",
        "llama a mi esposa al teléfono",
        "llámame al teléfono 780 555 8345",
        "no me llames al teléfono",
    ],
)
def test_a_counterexample_never_opens_the_pending_scope(text):
    session, call = FakeSession(), FakeCall()
    controller = _controller(call, _reader("direct"))

    async def run():
        await controller.handle_final_transcript(text, session)
        return await controller.handle_final_transcript("sí", session)

    asyncio.run(run())
    assert controller.awaiting_confirmation is False
    assert call.destinations == []


def test_a_clarify_reading_only_asks_and_a_yes_to_it_still_needs_a_confirmation():
    session, call = FakeSession(), FakeCall()
    controller = _controller(call, _reader("clarify"))

    async def run():
        await controller.handle_final_transcript("ya me voy, tengo el celular aquí", session)
        await controller.handle_final_transcript("sí", session)
        assert call.destinations == []
        assert controller.awaiting_confirmation is True
        await controller.handle_final_transcript("sí", session)

    token = reply_language.set(ES)
    try:
        asyncio.run(run())
    finally:
        reply_language.reset(token)
    assert call.destinations == [APPROVED]
    assert session.spoken[0] == spanish_pairs()[ASK_CLARIFICATION_REPLY]


def test_the_model_is_not_consulted_while_a_confirmation_is_pending():
    """The pending answer is exact vocabulary. No reading can widen it."""
    session, call, calls = FakeSession(), FakeCall(), []
    controller = _controller(call, _reader("direct", calls=calls))

    async def run():
        await controller.handle_final_transcript("sigamos por teléfono", session)
        assert len(calls) == 1
        await controller.handle_final_transcript("cuéntame un chiste", session)

    asyncio.run(run())
    assert len(calls) == 1
    assert call.destinations == []


def test_see_is_not_a_confirmation_in_either_language():
    """Whisper renders Spanish "Sí." as " See". That must never dial."""
    session, call = FakeSession(), FakeCall()
    controller = _controller(call, _reader("direct"))

    async def run():
        await controller.handle_final_transcript("sigamos por teléfono", session)
        await controller.handle_final_transcript("See", session)

    asyncio.run(run())
    assert call.destinations == []


# --- 4. English is untouched --------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "continue this conversation on my phone",
        "I have to get going, can we keep talking on my phone",
        "what's the weather like",
        "if I say call me on my phone, what happens",
    ],
)
def test_english_classification_is_unchanged_by_the_spanish_work(text):
    before = {
        "continue this conversation on my phone": HandoffIntent.DIRECT,
        "I have to get going, can we keep talking on my phone": HandoffIntent.DIRECT,
        "what's the weather like": HandoffIntent.NONE,
        "if I say call me on my phone, what happens": HandoffIntent.NONE,
    }
    assert classify_handoff_intent(text) is before[text]


def test_an_english_request_needs_no_model_at_all():
    session, call, calls = FakeSession(), FakeCall(), []
    controller = _controller(call, _reader("direct", calls=calls))

    async def run():
        await controller.handle_final_transcript("continue this conversation on my phone", session)
        await controller.handle_final_transcript("yes", session)

    asyncio.run(run())
    assert calls == []
    assert call.destinations == [APPROVED]
    assert session.spoken == [ASK_CONFIRMATION_REPLY, STARTING_REPLY]


# --- 5. The language of the question is this turn's, not the previous turn's ---


def test_the_spanish_question_binds_the_language_before_it_is_asked():
    """§5.4 at the handoff action boundary: the first Spanish turn is Spanish."""
    session, call = FakeSession(), FakeCall()
    bound: list[tuple] = []

    def bind(reading, text):
        bound.append((reading, text))
        reply_language.set(reading.language)

    controller = _controller(call, _reader("direct"), bind=bind)
    asyncio.run(controller.handle_final_transcript("sigamos por teléfono", session))

    assert bound and bound[0][0] == LanguageReading(ES, False)
    assert session.spoken == [spanish_pairs()[ASK_CONFIRMATION_REPLY]]
