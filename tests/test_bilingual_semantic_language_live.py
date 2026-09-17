"""Stage 3: the REAL local model, asked to read the language of a turn.

The semantic language decision is only as good as what the local model actually
answers, so this asks the real model the real router question, through the real
``SemanticWorkRouter``, and validates its reply with the real
``parse_language_reading``. No assertion here is about wording.

What this does and does not do:

* It talks to the real configured Ollama endpoint with the real model, the real
  ``WORK_ROUTER_SYSTEM_PROMPT``, and the real provider transport
  (``provider_classifier``) that production uses for this call.
* It asks about **language only**. No tool is offered, no tool handler runs, no
  work is scheduled, and no connected account, document, mailbox or device is
  touched. The utterances are invented.
* It skips, rather than fails, when the endpoint is unreachable, and it makes
  **no network call at all** -- not even a reachability probe -- unless
  ``CAAL_TEST_LOCAL_MODEL=1`` is set.

Language models are not deterministic, so each case is decided by a majority of
a small fixed number of real attempts, and the required outcome is the *decision
the session would actually take* -- not "the model said something".
"""

import asyncio
import collections
import json
import os
import urllib.request

import pytest

from caal.language_policy import EN, ES, LanguageSession
from caal.work_router import (
    SemanticWorkRouter,
    provider_classifier,
)

HOST = os.environ.get("CAAL_TEST_OLLAMA_HOST", "http://10.0.0.64:11434")
MODEL = os.environ.get("CAAL_TEST_OLLAMA_MODEL", "gemma4:e4b")
#: Each case is judged on a majority of this many real calls.
ATTEMPTS = 5

OPT_IN = os.environ.get("CAAL_TEST_LOCAL_MODEL") == "1"


def _reachable():
    if not OPT_IN:
        return False
    try:
        with urllib.request.urlopen(f"{HOST}/api/tags", timeout=5) as response:
            names = {m["name"] for m in json.loads(response.read())["models"]}
        return MODEL in names
    except Exception:  # noqa: BLE001 - an unreachable LAN host is a skip, not a failure
        return False


pytestmark = pytest.mark.skipif(
    not _reachable(),
    reason=(
        f"set CAAL_TEST_LOCAL_MODEL=1 and make {MODEL} reachable at {HOST}"
        if not OPT_IN
        else f"local model {MODEL} not reachable at {HOST}"
    ),
)


def _router():
    from caal.llm.providers.ollama_provider import OllamaProvider

    provider = OllamaProvider(
        model=MODEL, base_url=HOST, think=False, temperature=0.0, num_ctx=32768
    )
    return SemanticWorkRouter(classify=provider_classifier(provider), timeout_seconds=30.0)


def _readings(utterance, attempts=ATTEMPTS):
    """The real router's reading of one turn, ``attempts`` times."""
    router = _router()
    return [asyncio.run(router.route(utterance)).language for _ in range(attempts)]


def _decisions(utterance, *, preference="auto", current=EN, attempts=ATTEMPTS):
    """What the session would actually answer in, per attempt.

    This is the assertion surface that matters: a reading the policy then
    discards is not a decision, and a reading the policy uses is.
    """
    out = []
    for reading in _readings(utterance, attempts):
        session = LanguageSession(preference)
        session.current = current
        out.append(session.observe(utterance, None, semantic=reading))
    return out


def _majority(values):
    return collections.Counter(values).most_common(1)[0][0]


# --- 1. The model reads the language of an ordinary turn ---------------------


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("apaga la luz de la oficina", ES),
        ("ponme un recordatorio", ES),
        ("¿cuál es mi próxima reunión?", ES),
        ("turn off the office light", EN),
        ("what is my next meeting?", EN),
        ("remind me to call the bank", EN),
    ],
)
def test_the_real_model_reads_the_spoken_language(utterance, expected):
    decisions = _decisions(utterance)
    assert _majority(decisions) == expected, decisions


def test_an_english_turn_naming_a_spanish_person_and_dish_stays_english():
    """The stage-1 defect, now asked of the model rather than of a word list."""
    decisions = _decisions("call José about the piñata order", current=EN)
    assert _majority(decisions) == EN, decisions


# --- 2. A short acknowledgement is contextual, not evidence ------------------


@pytest.mark.parametrize("utterance", ["sí", "okay", "vale"])
def test_a_short_acknowledgement_keeps_a_spanish_session_spanish(utterance):
    decisions = _decisions(utterance, current=ES)
    assert set(decisions) == {ES}, decisions


@pytest.mark.parametrize("utterance", ["yes", "okay"])
def test_a_short_acknowledgement_keeps_an_english_session_english(utterance):
    decisions = _decisions(utterance, current=EN)
    assert set(decisions) == {EN}, decisions


# --- 3. An explicit switch, over a pinned account default --------------------


@pytest.mark.parametrize(
    "utterance",
    [
        "from now on answer me in Spanish",
        "contéstame en español a partir de ahora",
        "háblame en español por favor",
    ],
)
def test_an_explicit_switch_to_spanish_wins_over_an_account_pinned_to_english(utterance):
    decisions = _decisions(utterance, preference=EN, current=EN)
    assert _majority(decisions) == ES, decisions


def test_switching_back_to_english_works_the_same_way():
    decisions = _decisions("speak English from now on", preference=ES, current=ES)
    assert _majority(decisions) == EN, decisions


def test_saying_you_do_not_speak_spanish_is_not_a_switch():
    """A statement about a language is not a request to change to it."""
    decisions = _decisions("I do not speak Spanish, please keep going", current=EN)
    assert _majority(decisions) == EN, decisions


# --- 4. The reading does not disturb what the router was already for ---------


@pytest.mark.parametrize(
    ("utterance", "is_work"),
    [
        ("write me a full report comparing our suppliers and save it as a PDF", True),
        ("what time is it", False),
    ],
)
def test_the_added_language_question_leaves_the_work_reading_intact(utterance, is_work):
    router = _router()
    decisions = [asyncio.run(router.route(utterance)) for _ in range(3)]
    assert _majority([d.is_work for d in decisions]) is is_work, [d.route for d in decisions]
