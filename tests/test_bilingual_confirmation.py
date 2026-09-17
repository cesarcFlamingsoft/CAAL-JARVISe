"""`sí` confirms exactly where `yes` does -- and nowhere else.

The risk this pins down is the one the bilingual work creates: `sí` is an
extremely common ordinary word, so accepting it as a confirmation must not
widen the window in which *any* word authorizes an action. It does not: the
confirmation readers are consulted only while a question is pending, a fresh
intent replaces a pending question rather than answering it, and the phrase
list stays exact rather than fuzzy.
"""

import pytest

from caal.end_call_intent import EndCallAction, EndCallIntentMachine
from caal.handoff_intent import confirmation_given, denial_given


@pytest.mark.parametrize(
    "spoken",
    ["sí", "si", "Sí.", "sí por favor", "claro", "adelante", "hazlo", "confirmo", "correcto"],
)
def test_spanish_affirmatives_confirm_just_like_their_english_twins(spoken):
    assert confirmation_given(spoken) is True


@pytest.mark.parametrize(
    "spoken", ["no", "no gracias", "cancela", "cancelar", "olvídalo", "ahora no", "detente"]
)
def test_spanish_denials_refuse_just_like_their_english_twins(spoken):
    assert denial_given(spoken) is True


@pytest.mark.parametrize(
    "spoken",
    [
        "sí, pero primero dime qué hora es",
        "sí claro que el catálogo tiene tres productos",
        "creo que sí",
        "no sé si sí o no",
    ],
)
def test_a_si_inside_an_ordinary_sentence_is_not_a_confirmation(spoken):
    assert confirmation_given(spoken) is False


def test_english_confirmation_and_denial_behaviour_is_unchanged():
    for spoken in ("yes", "yes please", "go ahead", "sure", "confirm"):
        assert confirmation_given(spoken) is True
    for spoken in ("no", "no thanks", "cancel", "never mind"):
        assert denial_given(spoken) is True
    for spoken in ("yes the catalogue has three products", "maybe", ""):
        assert confirmation_given(spoken) is False


def test_si_cannot_arm_a_callback_when_nothing_was_asked():
    machine = EndCallIntentMachine()
    assert machine.awaiting_confirmation is False
    decision = machine.observe("sí", callback_available=True)
    assert decision.action is EndCallAction.NONE


def test_si_answers_only_the_question_that_is_actually_open():
    machine = EndCallIntentMachine()
    machine.offer_callback()
    assert machine.awaiting_confirmation is True
    assert machine.observe("sí", callback_available=True).action is EndCallAction.ARM_CALLBACK
    # The question is consumed: a second "sí" arms nothing.
    assert machine.observe("sí", callback_available=True).action is EndCallAction.NONE


def test_a_stale_si_past_the_confirmation_window_does_not_act():
    now = [1000.0]
    machine = EndCallIntentMachine(clock=lambda: now[0])
    machine.offer_callback()
    now[0] += EndCallIntentMachine.CONFIRMATION_TIMEOUT_SECONDS + 1
    assert machine.observe("sí", callback_available=True).action is EndCallAction.NONE


def test_a_spanish_denial_keeps_the_line_exactly_as_the_english_one_does():
    machine = EndCallIntentMachine()
    machine.offer_callback()
    assert machine.observe("no gracias", callback_available=True).action is EndCallAction.CANCEL
