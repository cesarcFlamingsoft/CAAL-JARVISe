"""Inference of "we can wrap this up" and "call me when you finish" on an outbound call.

The exact hang-up and callback commands in ``caal.call_termination`` stay the
only phrases that act without a question. These tests pin the conservative
natural-language reading layered under them: what ends a call outright, what
merely earns "Would you like me to end the call?", what earns the callback
question, and what must stay ordinary conversation for Hermes.
"""

from __future__ import annotations

import pytest

from caal.call_termination import callback_requested, end_call_requested
from caal.end_call_intent import (
    ASK_CALLBACK_REPLY,
    ASK_END_CALL_REPLY,
    END_CALL_CONTROL_REPLIES,
    NO_TASK_ASK_END_CALL_REPLY,
    NO_TASK_REPLY,
    STAY_ON_LINE_REPLY,
    CallbackIntent,
    EndCallAction,
    EndCallIntent,
    EndCallIntentMachine,
    classify_end_call_intent,
)

# --- strong, first-person, imperative: ends the call ----------------------------

DIRECT_END_PHRASES = [
    "I'm good for now",
    "you can let me go",
    "we can wrap this up",
    "I need to get off the phone",
    "that's all for now",
    "That'll be all, thanks.",
    "Okay, I have to go now.",
    "I gotta run",
    "let's wrap up",
    "Alright, that's all I needed. Thank you JARVIS.",
    "bye for now",
    "talk to you later",
    "No, that's everything for today.",
    "I'll let you go",
    "we're done here",
    "go ahead and hang up",
    "JARVIS, I need to go.",
    "Perfect, that's it for now, thanks for the help.",
]

# --- exit-ish, but hedged or partial: only a question -----------------------------

CLARIFY_END_PHRASES = [
    "I'm good",
    "I think we're done",
    "I should probably go",
    "I'm done",
    "that's it",
    "let me go",
    "okay I'm wrapping up here",
    "we should probably call it a day",
    "hang up please",
    "I think that's all for now",
    "maybe we can wrap this up",
]

# --- ordinary conversation: never a call decision --------------------------------

NONE_PHRASES = [
    "",
    "   ",
    "what time is it",
    "tell me about the history of the telephone",
    "How do I end a phone call on my iPhone?",
    "what happens if I say I'm good for now",
    "my brother said he needs to get off the phone",
    "I need to call my manager later",
    "I'm good at chess",
    "the call keeps dropping",
    "if we wrap this up now we'll miss something",
    "I'm not done yet",
    "before we wrap up, what's the weather",
    "can you end the call?",
    "don't hang up",
    "I need to go to the store tomorrow",
    '"that\'s all for now" is what she said',
    "I have a call with the bank at three",
    "when I say I'm good for now, does the call end",
    "I was about to hang up earlier",
    "can we wrap this up?",
    "is that all for now?",
    "I'm good for now but I have one more question",
    "explain what wrap this up means",
]

# --- exit plus a completion-conditioned callback: the callback question ----------

DIRECT_CALLBACK_PHRASES = [
    "I'm good for now, call me when you finish that",
    "let's wrap up; ring me when the research is ready",
    "I need to go, call me with the results",
    "That's all for now. Call me back once the background task is done.",
    "you can let me go and give me a call when it's finished",
    "we can wrap this up, just call me back as soon as you have the results",
    "I have to run, call me when you're done",
]

CLARIFY_CALLBACK_PHRASES = [
    "call me when you're done",
    "I'm good for now, call me later",
    "give me a call back with whatever you find",
    "ring me back",
]

NONE_CALLBACK_PHRASES = [
    "can you call me back when you're done?",
    "should I have you call me when it's done?",
    "call my brother when you're done",
    "call me back at 780 555 1234 when you're done",
    "my wife said call me when you finish",
    "let me know when it's done",
    "I'm good for now, don't call me back",
    "what does call me back when you're done do",
]


@pytest.mark.parametrize("text", DIRECT_END_PHRASES)
def test_strong_first_person_exit_reads_as_direct_end_call(text: str) -> None:
    reading = classify_end_call_intent(text)
    assert reading.end_call is EndCallIntent.DIRECT
    assert reading.callback is CallbackIntent.NONE
    # Inference never widens the literal commands themselves.
    assert callback_requested(text) is False


@pytest.mark.parametrize("text", CLARIFY_END_PHRASES)
def test_hedged_or_partial_exit_only_earns_a_question(text: str) -> None:
    reading = classify_end_call_intent(text)
    assert reading.end_call is EndCallIntent.CLARIFY
    assert reading.callback is CallbackIntent.NONE


@pytest.mark.parametrize("text", NONE_PHRASES)
def test_questions_hypotheticals_third_parties_and_incidental_words_read_as_nothing(
    text: str,
) -> None:
    reading = classify_end_call_intent(text)
    assert reading.end_call is EndCallIntent.NONE
    assert reading.callback is CallbackIntent.NONE
    assert end_call_requested(text) is False


@pytest.mark.parametrize("text", DIRECT_CALLBACK_PHRASES)
def test_exit_with_completion_conditioned_callback_reads_as_direct_callback(text: str) -> None:
    reading = classify_end_call_intent(text)
    assert reading.end_call is EndCallIntent.DIRECT
    assert reading.callback is CallbackIntent.DIRECT
    # These are the natural forms, not the literal command; that stays separate.
    assert callback_requested(text) is False


@pytest.mark.parametrize("text", CLARIFY_CALLBACK_PHRASES)
def test_callback_without_a_clear_exit_or_condition_only_earns_a_question(text: str) -> None:
    reading = classify_end_call_intent(text)
    assert reading.callback is CallbackIntent.CLARIFY


@pytest.mark.parametrize("text", NONE_CALLBACK_PHRASES)
def test_questions_third_parties_numbers_and_refusals_never_read_as_a_callback(
    text: str,
) -> None:
    reading = classify_end_call_intent(text)
    assert reading.callback is CallbackIntent.NONE
    assert reading.end_call is EndCallIntent.NONE


def test_classification_is_bounded_and_deterministic() -> None:
    long_text = "I'm good for now " * 500
    assert classify_end_call_intent(long_text).end_call is EndCallIntent.NONE
    first = classify_end_call_intent("I'm good for now")
    assert classify_end_call_intent("I'm good for now") == first


def test_control_replies_are_all_listed_for_ledger_exclusion() -> None:
    for reply in (
        ASK_END_CALL_REPLY,
        ASK_CALLBACK_REPLY,
        NO_TASK_ASK_END_CALL_REPLY,
        NO_TASK_REPLY,
        STAY_ON_LINE_REPLY,
    ):
        assert reply in END_CALL_CONTROL_REPLIES


# --- the state machine -----------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _machine(clock: Clock | None = None) -> EndCallIntentMachine:
    return EndCallIntentMachine(clock=clock or Clock())


def test_direct_exit_ends_the_call_without_a_question() -> None:
    machine = _machine()
    decision = machine.observe("I'm good for now", callback_available=False)
    assert decision.action is EndCallAction.END_CALL
    assert decision.consumed is True
    assert decision.reply is None
    assert machine.awaiting_confirmation is False


@pytest.mark.parametrize("text", DIRECT_END_PHRASES)
def test_every_direct_exit_ends_without_leaving_a_pending_question(text: str) -> None:
    machine = _machine()
    assert machine.observe(text, callback_available=True).action is EndCallAction.END_CALL
    assert machine.awaiting_confirmation is False


def test_uncertain_exit_asks_and_a_fresh_yes_ends_the_call() -> None:
    machine = _machine()
    asked = machine.observe("I think we're done", callback_available=False)
    assert asked.action is EndCallAction.ASK_END_CALL
    assert asked.reply == ASK_END_CALL_REPLY
    assert asked.consumed is True
    assert machine.awaiting_confirmation is True

    confirmed = machine.observe("yes", callback_available=False)
    assert confirmed.action is EndCallAction.END_CALL
    assert confirmed.consumed is True
    assert machine.awaiting_confirmation is False


@pytest.mark.parametrize("answer", ["yes", "Yes please.", "yeah", "go ahead", "JARVIS, yes"])
def test_only_explicit_affirmatives_confirm(answer: str) -> None:
    machine = _machine()
    machine.observe("I'm done", callback_available=False)
    assert machine.observe(answer, callback_available=False).action is EndCallAction.END_CALL


@pytest.mark.parametrize("answer", ["no", "No thanks.", "not now", "never mind"])
def test_a_denial_cancels_the_pending_question_and_keeps_the_line(answer: str) -> None:
    machine = _machine()
    machine.observe("I'm done", callback_available=False)
    decision = machine.observe(answer, callback_available=False)
    assert decision.action is EndCallAction.CANCEL
    assert decision.reply == STAY_ON_LINE_REPLY
    assert decision.consumed is True
    assert machine.awaiting_confirmation is False
    assert machine.observe("yes", callback_available=False).action is EndCallAction.NONE


def test_blank_final_input_cancels_the_pending_question_silently() -> None:
    machine = _machine()
    machine.observe("I'm done", callback_available=False)
    decision = machine.observe("   ", callback_available=False)
    assert decision.action is EndCallAction.CANCEL
    assert decision.reply is None
    assert decision.consumed is False
    assert machine.awaiting_confirmation is False
    assert machine.observe("yes", callback_available=False).action is EndCallAction.NONE


def test_an_unrelated_turn_cancels_and_is_handed_back_untouched() -> None:
    machine = _machine()
    machine.observe("I'm done", callback_available=False)
    decision = machine.observe("what's the weather tomorrow", callback_available=False)
    assert decision.action is EndCallAction.CANCEL
    assert decision.consumed is False
    assert decision.reply is None
    assert machine.awaiting_confirmation is False
    assert machine.observe("yes", callback_available=False).action is EndCallAction.NONE


def test_a_yes_that_is_not_only_a_yes_does_not_confirm() -> None:
    machine = _machine()
    machine.observe("I'm done", callback_available=False)
    decision = machine.observe("yes but first what's the weather", callback_available=False)
    assert decision.action is EndCallAction.CANCEL
    assert decision.consumed is False


def test_a_stale_yes_after_the_confirmation_window_does_nothing() -> None:
    clock = Clock()
    machine = _machine(clock)
    machine.observe("I'm done", callback_available=False)
    clock.now += EndCallIntentMachine.CONFIRMATION_TIMEOUT_SECONDS + 1
    assert machine.observe("yes", callback_available=False).action is EndCallAction.NONE
    assert machine.awaiting_confirmation is False


def test_a_yes_inside_the_confirmation_window_still_counts() -> None:
    clock = Clock()
    machine = _machine(clock)
    machine.observe("I'm done", callback_available=False)
    clock.now += EndCallIntentMachine.CONFIRMATION_TIMEOUT_SECONDS - 1
    assert machine.observe("yes", callback_available=False).action is EndCallAction.END_CALL


def test_yes_with_nothing_pending_is_ordinary_conversation() -> None:
    machine = _machine()
    decision = machine.observe("yes", callback_available=True)
    assert decision.action is EndCallAction.NONE
    assert decision.consumed is False


def test_reset_drops_a_pending_question_as_a_session_close_would() -> None:
    machine = _machine()
    machine.observe("I'm done", callback_available=False)
    machine.reset()
    assert machine.awaiting_confirmation is False
    assert machine.observe("yes", callback_available=False).action is EndCallAction.NONE


def test_direct_callback_only_asks_and_a_fresh_yes_arms_it() -> None:
    machine = _machine()
    asked = machine.observe(
        "I'm good for now, call me when you finish that", callback_available=True
    )
    assert asked.action is EndCallAction.ASK_CALLBACK
    assert asked.reply == ASK_CALLBACK_REPLY
    assert asked.consumed is True
    assert machine.awaiting_confirmation is True

    confirmed = machine.observe("yes", callback_available=True)
    assert confirmed.action is EndCallAction.ARM_CALLBACK
    assert confirmed.consumed is True
    assert machine.awaiting_confirmation is False


@pytest.mark.parametrize("text", DIRECT_CALLBACK_PHRASES + CLARIFY_CALLBACK_PHRASES)
def test_no_callback_phrase_ever_arms_or_ends_without_a_yes(text: str) -> None:
    machine = _machine()
    decision = machine.observe(text, callback_available=True)
    assert decision.action is EndCallAction.ASK_CALLBACK
    assert machine.awaiting_confirmation is True


def test_callback_question_declined_keeps_the_line_and_arms_nothing() -> None:
    machine = _machine()
    machine.observe("I need to go, call me with the results", callback_available=True)
    decision = machine.observe("no", callback_available=True)
    assert decision.action is EndCallAction.CANCEL
    assert decision.reply == STAY_ON_LINE_REPLY
    assert machine.observe("yes", callback_available=True).action is EndCallAction.NONE


def test_callback_question_is_cancelled_by_blank_unrelated_and_stale_answers() -> None:
    clock = Clock()
    machine = _machine(clock)
    request = "I need to go, call me with the results"

    machine.observe(request, callback_available=True)
    assert machine.observe("", callback_available=True).action is EndCallAction.CANCEL
    assert machine.observe("yes", callback_available=True).action is EndCallAction.NONE

    machine.observe(request, callback_available=True)
    assert machine.observe("book a flight", callback_available=True).consumed is False
    assert machine.observe("yes", callback_available=True).action is EndCallAction.NONE

    machine.observe(request, callback_available=True)
    clock.now += EndCallIntentMachine.CONFIRMATION_TIMEOUT_SECONDS + 1
    assert machine.observe("yes", callback_available=True).action is EndCallAction.NONE


def test_without_an_eligible_task_a_callback_exit_gets_the_plain_end_call_question() -> None:
    machine = _machine()
    asked = machine.observe(
        "I'm good for now, call me when you finish that", callback_available=False
    )
    assert asked.action is EndCallAction.ASK_END_CALL
    assert asked.reply == NO_TASK_ASK_END_CALL_REPLY
    assert asked.consumed is True

    confirmed = machine.observe("yes", callback_available=False)
    assert confirmed.action is EndCallAction.END_CALL


def test_without_an_eligible_task_a_bare_callback_ask_is_explained_and_nothing_pends() -> None:
    machine = _machine()
    decision = machine.observe("call me when you're done", callback_available=False)
    assert decision.action is EndCallAction.DECLINE_CALLBACK
    assert decision.reply == NO_TASK_REPLY
    assert decision.consumed is True
    assert machine.awaiting_confirmation is False
    assert machine.observe("yes", callback_available=False).action is EndCallAction.NONE


def test_a_new_strong_intent_replaces_a_pending_question_instead_of_confirming_it() -> None:
    machine = _machine()
    machine.observe("I'm done", callback_available=True)
    decision = machine.observe("I need to go, call me with the results", callback_available=True)
    assert decision.action is EndCallAction.ASK_CALLBACK
    assert machine.awaiting_confirmation is True
    assert machine.observe("yes", callback_available=True).action is EndCallAction.ARM_CALLBACK


@pytest.mark.parametrize("text", NONE_PHRASES + NONE_CALLBACK_PHRASES)
def test_ordinary_conversation_never_moves_the_machine(text: str) -> None:
    machine = _machine()
    decision = machine.observe(text, callback_available=True)
    assert decision.action is EndCallAction.NONE
    assert decision.consumed is False
    assert decision.reply is None
    assert machine.awaiting_confirmation is False
