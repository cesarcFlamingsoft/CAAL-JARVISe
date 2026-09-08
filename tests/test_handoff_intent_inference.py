"""Coverage for inferring a phone handoff from natural, non-command speech.

A user should be able to imply "let's keep talking by phone" the way people
actually say it, so a final STT transcript such as "I need to take this on the
road" must reach CAAL's local handler instead of Hermes, which has no outbound
call tool. Inference is deterministic and offline: it reads cues in the text
only, and it still never dials. A DIRECT reading only asks for confirmation, a
CLARIFY reading only asks a question, and everything else stays ordinary
conversation.
"""

from __future__ import annotations

import pytest

from caal.handoff_intent import (
    ASK_CLARIFICATION_REPLY,
    ASK_CONFIRMATION_REPLY,
    HandoffAction,
    HandoffIntent,
    HandoffIntentMachine,
    classify_handoff_intent,
    handoff_requested,
)

DIRECT_PHRASES = (
    "I need to take this on the road",
    "Can we pick this up on a phone call?",
    "I want to continue talking while I drive",
    "Call me so we can keep talking.",
    "JARVIS, I have to head out, can we keep talking on my phone?",
    "let's continue this conversation in the car",
    "I'm about to drive, can you call me so we can finish this?",
    # The original explicit commands must keep working unchanged.
    "Continue this conversation on my phone",
    "call me on my phone",
)

CLARIFY_PHRASES = (
    "I'm leaving soon",
    "I might need to use my phone",
    "Can you call?",
    "I have to get going in a minute",
    "I may have to switch to my cell",
)

NONE_PHRASES = (
    "",
    "   ",
    # Explanations and capability questions.
    "How do I continue this conversation on my phone?",
    "How does the phone handoff work?",
    "Are you able to call me on my phone?",
    "Explain how you would continue this on my phone",
    # Hypothetical or instructional mentions.
    'If I say "call me on my phone", what happens?',
    "What would you do if I asked you to call me while I drive?",
    "For example, I could ask you to continue this on my phone",
    # Calls aimed at somebody other than the user.
    "call my mother",
    "Can you call people on the phone?",
    "call me on 555 123 4567",
    "call the office and keep me posted",
    # A mobility or call word alone must not turn an unrelated errand into a handoff.
    "I need to take this to the phone store",
    "I need to call my phone repair shop",
    "I need to take this call while driving",
    "I need to call the phone company",
    "Can you call the phone company?",
    # Device troubleshooting.
    "My phone is broken so I am using the browser",
    "my cell battery is dead and the screen is cracked",
    # Ordinary phone talk.
    "I was reading this on my phone earlier",
    "I have a call at three so let's be quick",
    "remind me to take the trash out at eight",
    "what is the weather tomorrow",
    "please don't call my phone",
)


@pytest.mark.parametrize("phrase", DIRECT_PHRASES)
def test_natural_continuation_requests_are_direct(phrase: str) -> None:
    assert classify_handoff_intent(phrase) is HandoffIntent.DIRECT
    assert handoff_requested(phrase) is True


@pytest.mark.parametrize("phrase", CLARIFY_PHRASES)
def test_partial_cues_only_ask_for_clarification(phrase: str) -> None:
    assert classify_handoff_intent(phrase) is HandoffIntent.CLARIFY
    # A clarification is not a request, so it must never satisfy the command path.
    assert handoff_requested(phrase) is False


@pytest.mark.parametrize("phrase", NONE_PHRASES)
def test_unrelated_or_unsafe_phrases_are_never_handoff_intent(phrase: str) -> None:
    assert classify_handoff_intent(phrase) is HandoffIntent.NONE
    assert handoff_requested(phrase) is False


def test_inference_ignores_casing_punctuation_and_spacing() -> None:
    assert classify_handoff_intent("  I NEED to take this,  on the road!! ") is HandoffIntent.DIRECT


def test_clarification_consumes_the_turn_without_arming_a_confirmation() -> None:
    machine = HandoffIntentMachine()

    decision = machine.observe("I'm leaving soon")

    assert decision.action is HandoffAction.ASK_CLARIFICATION
    assert decision.consumed is True
    assert decision.reply == ASK_CLARIFICATION_REPLY
    # A clarification must not be answerable with "yes" straight into a call.
    assert machine.awaiting_confirmation is False


def test_yes_to_a_clarification_only_asks_for_the_real_confirmation() -> None:
    machine = HandoffIntentMachine()
    machine.observe("I'm leaving soon")

    decision = machine.observe("yes")

    assert decision.action is HandoffAction.ASK_CONFIRMATION
    assert decision.reply == ASK_CONFIRMATION_REPLY
    assert decision.consumed is True
    assert machine.awaiting_confirmation is True


def test_clarification_then_two_yeses_are_required_before_a_handoff_starts() -> None:
    machine = HandoffIntentMachine()
    machine.observe("I might need to use my phone")

    assert machine.observe("yes").action is HandoffAction.ASK_CONFIRMATION
    assert machine.observe("yes").action is HandoffAction.START_HANDOFF


def test_denying_a_clarification_cancels_and_consumes_the_turn() -> None:
    machine = HandoffIntentMachine()
    machine.observe("Can you call?")

    decision = machine.observe("no thanks")

    assert decision.action is HandoffAction.CANCEL
    assert decision.consumed is True
    assert machine.awaiting_confirmation is False


def test_unrelated_answer_to_a_clarification_returns_to_normal_conversation() -> None:
    machine = HandoffIntentMachine()
    machine.observe("I'm leaving soon")

    decision = machine.observe("what is the weather tomorrow")

    assert decision.action is HandoffAction.NONE
    assert decision.consumed is False
    assert machine.observe("yes").action is HandoffAction.NONE


def test_direct_intent_during_a_clarification_skips_straight_to_confirmation() -> None:
    machine = HandoffIntentMachine()
    machine.observe("I'm leaving soon")

    decision = machine.observe("I need to take this on the road")

    assert decision.action is HandoffAction.ASK_CONFIRMATION
    assert machine.awaiting_confirmation is True


def test_silence_keeps_a_pending_clarification_alive() -> None:
    machine = HandoffIntentMachine()
    machine.observe("I'm leaving soon")

    decision = machine.observe("  ")

    assert decision.action is HandoffAction.NONE
    assert machine.observe("yes").action is HandoffAction.ASK_CONFIRMATION


def test_clarifying_cues_while_awaiting_confirmation_re_ask_instead_of_reaching_hermes() -> None:
    machine = HandoffIntentMachine()
    machine.observe("I need to take this on the road")

    decision = machine.observe("I might need to use my phone")

    assert decision.action is HandoffAction.ASK_CONFIRMATION
    assert decision.consumed is True
    assert machine.awaiting_confirmation is True


def test_reset_clears_a_pending_clarification() -> None:
    machine = HandoffIntentMachine()
    machine.observe("I'm leaving soon")

    machine.reset()

    assert machine.observe("yes").action is HandoffAction.NONE


def test_inferred_intent_never_starts_a_handoff_on_its_own() -> None:
    """The whole point of the safety rule: inference asks, it never dials."""
    for phrase in DIRECT_PHRASES + CLARIFY_PHRASES:
        machine = HandoffIntentMachine()
        assert machine.observe(phrase).action is not HandoffAction.START_HANDOFF, phrase
