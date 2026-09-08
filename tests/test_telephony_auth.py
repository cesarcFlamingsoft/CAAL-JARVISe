"""Behavioral tests for phone-call keypad authentication."""

from caal.telephony_auth import CallAccessGate, GateState, hash_pin, verify_pin


def test_correct_pin_followed_by_hash_grants_access() -> None:
    gate = CallAccessGate(hash_pin("2468"), max_attempts=3)

    for digit in "2468#":
        state = gate.accept_digit(digit)

    assert state is GateState.GRANTED
    assert gate.is_granted


def test_wrong_pin_keeps_gate_locked_until_attempt_limit() -> None:
    gate = CallAccessGate(hash_pin("2468"), max_attempts=3)

    for digit in "1111#":
        state = gate.accept_digit(digit)

    assert state is GateState.RETRY
    assert not gate.is_granted
    assert gate.attempts_remaining == 2


def test_three_invalid_submissions_lock_the_gate() -> None:
    gate = CallAccessGate(hash_pin("2468"), max_attempts=3)

    for _ in range(3):
        for digit in "1111#":
            state = gate.accept_digit(digit)

    assert state is GateState.LOCKED
    assert gate.attempts_remaining == 0


def test_star_clears_unsubmitted_pin_digits() -> None:
    gate = CallAccessGate(hash_pin("2468"))

    for digit in "24*2468#":
        state = gate.accept_digit(digit)

    assert state is GateState.GRANTED



def test_pin_hash_is_safe_to_store_in_a_docker_compose_env_file() -> None:
    encoded = hash_pin("2468")

    assert "$" not in encoded
    assert verify_pin("2468", encoded)


def test_non_dtmf_input_is_rejected_without_consuming_an_attempt() -> None:
    gate = CallAccessGate(hash_pin("2468"))

    state = gate.accept_digit("x")

    assert state is GateState.INVALID_INPUT
    assert gate.attempts_remaining == 3
