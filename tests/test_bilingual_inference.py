"""Stage 2: the language decision must follow the speech server and real intent.

Stage 1 decided with a diacritic check and a four-word floor. Both were wrong in
measured ways (reports/bilingual/FINAL-STAGE1.md §4): an English sentence with a
Spanish name flipped the session to Spanish, and a genuine three-word Spanish
command did not. These tests pin the corrected contract.

Everything here is pure and local: no network, no model, no classifier call.
"""

import time

import pytest

from caal.language_policy import (
    EN,
    ES,
    LanguageSession,
    explicit_request,
    is_evidential,
    resolve,
)


# --- 1. Diacritics in an English sentence are not Spanish evidence -----------


@pytest.mark.parametrize(
    "transcript",
    [
        "call José about the piñata order",
        "email María the invoice for the jalapeño shipment",
        "put the la croix in the fridge",
        "add a piñata to the party list and tell André",
    ],
)
def test_english_sentences_with_spanish_names_stay_english(transcript):
    assert resolve("auto", detected="en", transcript=transcript, current=EN) == EN


def test_a_spanish_name_does_not_flip_a_session_that_is_speaking_english():
    session = LanguageSession("auto")
    assert session.observe("call José about the piñata order", "en") == EN
    assert session.current == EN


# --- 2. Short but unambiguous Spanish commands are honoured ------------------


@pytest.mark.parametrize(
    "transcript",
    ["apaga la luz", "enciende la lámpara", "cancela la alarma", "lee mis correos"],
)
def test_short_spanish_commands_switch(transcript):
    assert resolve("auto", detected="es", transcript=transcript, current=EN) == ES


def test_short_english_commands_stay_english():
    assert resolve("auto", detected="en", transcript="turn the light off", current=EN) == EN
    assert resolve("auto", detected="en", transcript="read my email", current=ES) == EN


# --- 3. Short ambiguous acknowledgements keep the session where it is --------


@pytest.mark.parametrize("transcript", ["yes", "okay", "sí", "si", "See", "ok", "sure", "claro"])
def test_acknowledgements_never_move_the_language(transcript):
    assert resolve("auto", detected="en", transcript=transcript, current=ES) == ES
    assert resolve("auto", detected="es", transcript=transcript, current=EN) == EN


def test_a_mislabelled_acknowledgement_is_not_evidence():
    # Measured: the live Whisper server returns " See" tagged en for Spanish "Sí."
    assert is_evidential(" See") is False
    assert is_evidential("sí") is False
    assert is_evidential("apaga la luz") is True


# --- 4. Typed / spoken natural switches ------------------------------------


@pytest.mark.parametrize(
    "transcript",
    [
        "Spanish please",
        "español por favor",
        "en español por favor",
        "castellano",
        "español",
        "habla en español",
        "switch to Spanish",
    ],
)
def test_natural_spanish_switch_requests(transcript):
    assert explicit_request(transcript) == ES


@pytest.mark.parametrize(
    "transcript",
    ["English please", "inglés por favor", "speak English", "back to English please"],
)
def test_natural_english_switch_requests(transcript):
    assert explicit_request(transcript) == EN


@pytest.mark.parametrize(
    "transcript",
    ["I do not speak Spanish", "no hablo español", "she does not speak English"],
)
def test_a_denial_is_not_a_switch_request(transcript):
    assert explicit_request(transcript) is None


def test_ordinary_mentions_are_not_switch_requests():
    assert explicit_request("the Spanish invoice arrived") is None
    assert explicit_request("send Maria the English contract") is None


# --- 5. A conversational override outranks a pinned account preference ------


def test_a_spoken_override_beats_a_pinned_english_account():
    session = LanguageSession("en")
    assert session.current == EN
    assert session.observe("español por favor", "en") == ES
    assert session.observe("¿me lees el correo de la oficina?", "es") == ES


def test_a_spoken_override_beats_a_pinned_spanish_account():
    session = LanguageSession("es")
    assert session.observe("English please", "es") == EN
    assert session.observe("read me the office email", "en") == EN


def test_an_override_does_not_write_the_stored_preference():
    session = LanguageSession("en")
    session.observe("español por favor", "en")
    assert session.preference == "en"
    assert session.override == ES


# --- 6. English backwards compatibility ------------------------------------


def test_an_english_session_with_no_signal_answers_english():
    assert resolve(None, detected=None, transcript=None, current=None) == EN
    assert LanguageSession(None).current == EN
    assert LanguageSession(None).directive() == ""


def test_a_spanish_detection_on_plainly_english_words_is_not_trusted():
    # Whisper mislabels happen in both directions; English text stays English.
    assert resolve("auto", detected="es", transcript="what is on my calendar today", current=EN) == EN


# --- 7. "see" is a language hint only, never an authorization ---------------


@pytest.mark.parametrize("transcript", ["see", "See.", "I see", "sea", "c"])
def test_see_never_authorizes_anything(transcript):
    from caal.handoff_intent import confirmation_given

    assert confirmation_given(transcript) is False


def test_the_acknowledgement_list_is_not_a_confirmation_list():
    from caal import handoff_intent, language_policy

    assert "see" in language_policy._ACKNOWLEDGEMENTS
    assert "see" not in handoff_intent._CONFIRMATIONS
    assert "sea" not in handoff_intent._CONFIRMATIONS


# --- 8. Cost: the decision must stay free -----------------------------------


def test_the_decision_makes_no_call_and_is_microseconds(monkeypatch):
    import socket

    def forbidden(*args, **kwargs):  # pragma: no cover - only runs on a defect
        raise AssertionError("the language decision must not open a socket")

    monkeypatch.setattr(socket, "socket", forbidden)
    samples = [
        "apaga la luz",
        "call José about the piñata order",
        "what is on my calendar today",
        "¿me puedes leer el correo de la oficina, por favor?",
    ]
    start = time.perf_counter()
    for _ in range(200):
        for text in samples:
            resolve("auto", detected="es", transcript=text, current=EN)
    elapsed = time.perf_counter() - start
    assert elapsed / (200 * len(samples)) < 0.001
