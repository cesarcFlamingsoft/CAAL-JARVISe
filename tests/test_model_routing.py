"""Where one user turn is answered: the local model, or the Hermes agent runtime.

JARVIS main model is the local Ollama one. Two kinds of turn are not its
work: a coding request, which is delegated to Hermes to carry out with its own
Claude Code capability, and a genuinely long or multi-step request, which needs
the Hermes agent harness. Deciding which is which is an authorization, not a
guess, so it is made here: offline, deterministic, bounded, and in a fixed
order -- coding first, harness second, local otherwise.

The reading is privacy-safe by construction. It sees a bounded prefix of the
turn, keeps no copy of it, and nothing it logs or returns carries the words,
a credential read aloud, a file path, or code.
"""

from __future__ import annotations

import logging

import pytest

from caal.model_routing import (
    MAX_ROUTING_INPUT_CHARS,
    Destination,
    RoutingDecision,
    RoutingSource,
    classify_request,
)

CODING_REQUESTS = [
    "write a python function that reverses a linked list",
    "fix the bug in the checkout script",
    "refactor the user store module",
    "there is a traceback in the dashboard api, can you find it",
    "add a unit test for the knowledge router",
    "review the code in the connections module",
    "implement a retry in the ollama provider",
    "the test suite is failing on main",
    "clean up the regex in the work router",
    "rewrite this javascript component in typescript",
    "have claude code look at the background task queue",
    "debug why the api endpoint returns a 500",
]

HARNESS_REQUESTS = [
    "create a pdf explaining what you are",
    "research the alberta grid and put together a summary",
    "look into this in the background and get back to me",
    "write up a report comparing the two vendors",
    "do a deep dive into the invoices from last quarter",
]

LOCAL_REQUESTS = [
    "what time is it",
    "turn off the kitchen lights",
    "what is the weather looking like tomorrow",
    "remind me to call my sister at six",
    "how are you feeling today",
    "who won the game last night",
    "explain the function of the heart",
    "what is on my calendar tomorrow",
    "did I get any email from Bo",
    "hey jarvis",
    "",
]

# Talking *about* coding is not a coding request.
CODING_META = [
    "can you write code",
    "do you know python",
    "are you able to program",
    "what is a function",
]


# --- the three destinations --------------------------------------------------


@pytest.mark.parametrize("text", CODING_REQUESTS)
def test_coding_requests_are_delegated_for_claude_code(text: str) -> None:
    decision = classify_request(text)
    assert decision.destination is Destination.CODING
    assert decision.source is RoutingSource.CODING


@pytest.mark.parametrize("text", HARNESS_REQUESTS)
def test_hard_or_multi_step_requests_go_to_the_harness(text: str) -> None:
    decision = classify_request(text)
    assert decision.destination is Destination.HARNESS


@pytest.mark.parametrize("text", LOCAL_REQUESTS)
def test_ordinary_requests_stay_on_the_local_model(text: str) -> None:
    assert classify_request(text).destination is Destination.LOCAL


@pytest.mark.parametrize("text", CODING_META)
def test_questions_about_coding_are_not_coding_requests(text: str) -> None:
    assert classify_request(text).destination is not Destination.CODING


def test_coding_outranks_the_harness() -> None:
    """A coding request phrased as long work is still a coding request."""
    decision = classify_request(
        "research how our python retry logic works and refactor it in the background"
    )
    assert decision.destination is Destination.CODING


@pytest.mark.parametrize(
    "text",
    [
        "cancel the background task",
        "how is the background task going",
        "is the pdf ready yet",
    ],
)
def test_background_control_commands_are_never_claimed(text: str) -> None:
    """Cancel and status belong to the queue; routing must not steal them."""
    decision = classify_request(text)
    assert decision.destination is Destination.LOCAL
    assert decision.source is RoutingSource.CONTROL


# --- bounded and total -------------------------------------------------------


@pytest.mark.parametrize("value", [None, 42, b"bytes", object(), ["fix the bug"]])
def test_anything_that_is_not_text_is_local(value: object) -> None:
    assert classify_request(value).destination is Destination.LOCAL


def test_only_a_bounded_prefix_is_read() -> None:
    """An unbounded turn cannot become an unbounded reading."""
    padding = "la " * MAX_ROUTING_INPUT_CHARS
    assert classify_request(padding + "fix the bug in the script").destination is Destination.LOCAL
    assert (
        classify_request("fix the bug in the script " + padding).destination is Destination.CODING
    )


def test_the_decision_carries_no_request_text() -> None:
    decision = classify_request("fix the bug in checkout.py, the password is hunter2")
    assert isinstance(decision, RoutingDecision)
    assert "hunter2" not in repr(decision)
    assert "checkout" not in repr(decision)


def test_nothing_is_logged_about_the_request(caplog: pytest.LogCaptureFixture) -> None:
    secret = "sk-abcdefghijklmnop"
    with caplog.at_level(logging.DEBUG, logger="caal.model_routing"):
        classify_request(f"refactor the auth module, my api key is {secret}")
    combined = " ".join(record.getMessage() for record in caplog.records)
    assert secret not in combined
    assert "auth module" not in combined
