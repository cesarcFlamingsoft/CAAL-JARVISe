"""Unit coverage for the local phone-handoff intent state machine.

The machine must recognize the handoff request itself so the command never
reaches Hermes as a missing tool, and it must require an explicit confirmation
in the very next turn before any call is considered.
"""

from __future__ import annotations

from caal.handoff_intent import (
    HandoffAction,
    HandoffIntentMachine,
    handoff_requested,
    sole_handoff_destination,
)


def test_handoff_requested_accepts_explicit_phone_handoff_phrases() -> None:
    assert handoff_requested("Continue this conversation on my phone")
    assert handoff_requested("JARVIS, continue this conversation on my phone.")
    assert handoff_requested("please continue on my cell")
    assert handoff_requested("Move this conversation to my phone")
    assert handoff_requested("move conversation to my phone")
    assert handoff_requested("transfer this call to my mobile")
    assert handoff_requested("switch this chat over to my phone")
    assert handoff_requested("call me on my phone")
    assert handoff_requested("Can you call me on my cell phone?".rstrip("?"))


def test_handoff_requested_rejects_ambiguous_or_unrelated_text() -> None:
    assert not handoff_requested("")
    assert not handoff_requested("How do I continue this conversation on my phone?")
    assert not handoff_requested("Can you call people on the phone?")
    assert not handoff_requested("My phone is broken so I am using the browser")
    assert not handoff_requested("I was reading this on my phone earlier")
    assert not handoff_requested("call my mother")
    assert not handoff_requested("call me on 555 123 4567")


def test_request_asks_for_confirmation_and_consumes_the_turn() -> None:
    machine = HandoffIntentMachine()

    decision = machine.observe("Continue this conversation on my phone")

    assert decision.action is HandoffAction.ASK_CONFIRMATION
    assert decision.consumed is True
    assert decision.reply
    assert machine.awaiting_confirmation is True


def test_explicit_yes_in_the_next_turn_starts_the_handoff() -> None:
    machine = HandoffIntentMachine()
    machine.observe("Continue this conversation on my phone")

    decision = machine.observe("yes")

    assert decision.action is HandoffAction.START_HANDOFF
    assert decision.consumed is True
    assert machine.awaiting_confirmation is False


def test_other_affirmative_confirmations_are_accepted() -> None:
    for answer in ("Yes, please.", "yeah", "go ahead", "do it", "confirm", "JARVIS, yes"):
        machine = HandoffIntentMachine()
        machine.observe("call me on my phone")
        assert machine.observe(answer).action is HandoffAction.START_HANDOFF, answer


def test_explicit_denial_cancels_and_consumes_the_turn() -> None:
    machine = HandoffIntentMachine()
    machine.observe("Continue this conversation on my phone")

    decision = machine.observe("no")

    assert decision.action is HandoffAction.CANCEL
    assert decision.consumed is True
    assert decision.reply
    assert machine.awaiting_confirmation is False


def test_other_denials_cancel_without_starting_a_handoff() -> None:
    for answer in ("no thanks", "nope", "cancel", "never mind", "stop", "don't"):
        machine = HandoffIntentMachine()
        machine.observe("Continue this conversation on my phone")
        assert machine.observe(answer).action is HandoffAction.CANCEL, answer


def test_unrelated_reply_cancels_silently_and_is_passed_through() -> None:
    machine = HandoffIntentMachine()
    machine.observe("Continue this conversation on my phone")

    decision = machine.observe("what is the weather tomorrow")

    assert decision.action is HandoffAction.CANCEL
    assert decision.consumed is False
    assert decision.reply is None
    assert machine.awaiting_confirmation is False


def test_blank_transcript_cancels_a_pending_confirmation() -> None:
    machine = HandoffIntentMachine()
    machine.observe("Continue this conversation on my phone")

    decision = machine.observe("   ")

    assert decision.action is HandoffAction.CANCEL
    assert decision.consumed is False
    assert machine.awaiting_confirmation is False
    assert machine.observe("yes").action is HandoffAction.NONE


def test_repeating_the_request_re_asks_instead_of_confirming() -> None:
    machine = HandoffIntentMachine()
    machine.observe("Continue this conversation on my phone")

    decision = machine.observe("continue this conversation on my phone")

    assert decision.action is HandoffAction.ASK_CONFIRMATION
    assert machine.awaiting_confirmation is True


def test_yes_without_a_pending_request_never_starts_a_handoff() -> None:
    machine = HandoffIntentMachine()

    decision = machine.observe("yes")

    assert decision.action is HandoffAction.NONE
    assert decision.consumed is False


def test_ordinary_conversation_is_never_consumed() -> None:
    machine = HandoffIntentMachine()

    decision = machine.observe("remind me to take the trash out at eight")

    assert decision.action is HandoffAction.NONE
    assert decision.consumed is False
    assert machine.awaiting_confirmation is False


def test_reset_clears_a_pending_confirmation() -> None:
    machine = HandoffIntentMachine()
    machine.observe("Continue this conversation on my phone")

    machine.reset()

    assert machine.awaiting_confirmation is False
    assert machine.observe("yes").action is HandoffAction.NONE


def test_sole_handoff_destination_requires_exactly_one_approved_number() -> None:
    assert sole_handoff_destination("+17805558345") == "+17805558345"
    assert sole_handoff_destination(" +17805558345 , +17805558345 ") == "+17805558345"
    assert sole_handoff_destination("") is None
    assert sole_handoff_destination("   ") is None
    assert sole_handoff_destination("+17805558345,+17805550000") is None
