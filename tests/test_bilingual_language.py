"""Per-user reply-language preference and the auto-switch policy.

The policy is pure: it takes the STT signal the server already returns and the
current session language, and never calls a model or the network.
"""

import pytest
from fastapi.testclient import TestClient
from test_local_model_api import Harness

from caal import language_api, user_api, webhooks
from caal.language_policy import (
    AUTO,
    EN,
    ES,
    explicit_request,
    resolve,
)
from caal.language_store import LanguageStore

# --- policy -------------------------------------------------------------------------------


def test_explicit_preference_pins_the_reply_language_against_any_detection():
    assert resolve(EN, detected="es", transcript="enciende la lámpara", current=ES) == EN
    assert resolve(ES, detected="en", transcript="turn on the lamp", current=EN) == ES


def test_auto_switches_on_a_confident_spanish_utterance():
    assert resolve(AUTO, detected="es", transcript="enciende la lámpara de la oficina",
                   current=EN) == ES


def test_auto_switches_back_on_a_confident_english_utterance():
    assert resolve(AUTO, detected="en", transcript="turn on the office lamp", current=ES) == EN


def test_auto_keeps_the_current_language_on_a_short_acknowledgement():
    # Measured: the STT server returns language "en" for the Spanish "Sí."
    # (reports/bilingual/evidence/stt-probe-baseline.json). A short turn must
    # never move the session.
    assert resolve(AUTO, detected="en", transcript="See", current=ES) == ES
    assert resolve(AUTO, detected="en", transcript="Okay.", current=ES) == ES
    assert resolve(AUTO, detected="es", transcript="sí", current=EN) == EN


def test_auto_keeps_the_current_language_when_there_is_no_usable_signal():
    # Stage 2 change, deliberate: when detection is missing, orthography is now
    # read in *both* directions, not only the Spanish one. Plain English words
    # therefore answer in English even from a Spanish session, mirroring what
    # Spanish orthography has always done in reverse. A turn with no orthographic
    # evidence either way still keeps the session where it is.
    assert resolve(AUTO, detected=None, transcript="turn on the office lamp", current=ES) == EN
    assert resolve(AUTO, detected="", transcript="turn on the office lamp", current=ES) == EN
    assert resolve(AUTO, detected=None, transcript="lamp office now", current=ES) == ES
    assert resolve(AUTO, detected="fr", transcript="allume la lampe du bureau", current=EN) == EN


def test_auto_trusts_spanish_orthography_when_detection_says_english():
    # Long Spanish turns misdetected as English still read as Spanish.
    assert resolve(
        AUTO, detected="en", transcript="¿Puedes revisar el catálogo de la empresa?", current=EN
    ) == ES


def test_a_spoken_instruction_overrides_detection_in_both_directions():
    assert explicit_request("habla en inglés por favor") == EN
    assert explicit_request("responde en español") == ES
    assert explicit_request("speak spanish from now on") == ES
    assert explicit_request("switch to english") == EN
    assert explicit_request("enciende la lámpara de la oficina") is None
    assert explicit_request("turn on the office lamp") is None


def test_an_instruction_beats_the_short_utterance_guard():
    assert resolve(AUTO, detected="en", transcript="en español", current=EN) == ES


# --- store --------------------------------------------------------------------------------


def test_store_defaults_to_auto_and_keeps_users_separate(tmp_path):
    h = Harness(tmp_path)
    store = LanguageStore(h.identity)
    assert store.preference(h.member) == AUTO
    store.save_preference(h.member, ES)
    assert store.preference(h.member) == ES
    assert store.preference(h.admin) == AUTO


def test_store_rejects_an_unsupported_language(tmp_path):
    store = LanguageStore(Harness(tmp_path).identity)
    with pytest.raises(ValueError):
        store.save_preference("usr_x", "fr")


# --- api ----------------------------------------------------------------------------------


@pytest.fixture
def harness(tmp_path):
    h = Harness(tmp_path)
    webhooks.app.dependency_overrides[user_api.get_runtime] = lambda: h.identity
    yield h
    webhooks.app.dependency_overrides.pop(user_api.get_runtime, None)


def test_anonymous_cannot_read_or_set_the_language(harness):
    with TestClient(webhooks.app) as c:
        assert c.get("/users/me/language").status_code == 401
        assert c.put("/users/me/language", json={"language": "es"}).status_code == 401


def test_member_saves_a_personal_language_that_no_other_user_sees(harness):
    # Every signed principal is single-use, so each call mints its own.
    member = lambda: harness.bearer(harness.member)  # noqa: E731
    with TestClient(webhooks.app) as c:
        assert c.get("/users/me/language", headers=member()).json() == {
            "language": "auto",
            "source": "default",
            "applies_to": "new_sessions",
        }
        assert (
            c.put("/users/me/language", json={"language": "es"}, headers=member()).status_code
            == 200
        )
        assert c.get("/users/me/language", headers=member()).json()["language"] == "es"
        assert c.get("/users/me/language", headers=member()).json()["source"] == "personal"
        # A different signed-in user is untouched.
        assert (
            c.get("/users/me/language", headers=harness.bearer(harness.admin)).json()["language"]
            == "auto"
        )


def test_the_language_route_rejects_anything_outside_auto_en_es(harness):
    with TestClient(webhooks.app) as c:
        for bad in ({"language": "fr"}, {"language": "EN"}, {"language": ""}, {"lang": "es"}):
            assert (
                c.put(
                    "/users/me/language", json=bad, headers=harness.bearer(harness.member)
                ).status_code
                == 422
            )


def test_the_language_route_never_writes_global_settings(harness):
    with TestClient(webhooks.app) as c:
        c.put(
            "/users/me/language",
            json={"language": "es"},
            headers=harness.bearer(harness.member),
        )
    assert harness.saved == []
    assert language_api.router is not None


# --- session ------------------------------------------------------------------------------


def test_an_auto_session_starts_in_english_and_follows_the_speaker():
    from caal.language_policy import LanguageSession

    session = LanguageSession(AUTO)
    assert session.current == EN
    assert session.observe("Turn on the office lamp.", detected="en") == EN
    assert session.observe("enciende la lámpara de la oficina", detected="es") == ES
    # Sticky: a short acknowledgement does not drag it back to English.
    assert session.observe("See", detected="en") == ES
    assert session.observe("Okay.", detected="en") == ES
    assert session.observe("could you check that again please", detected="en") == EN


def test_a_pinned_account_preference_never_moves_with_the_speaker():
    from caal.language_policy import LanguageSession

    session = LanguageSession(EN)
    assert session.current == EN
    assert session.observe("enciende la lámpara de la oficina", detected="es") == EN
    assert session.observe("¿Puedes revisar el catálogo?", detected="es") == EN


def test_a_spoken_instruction_overrides_a_pinned_preference_for_the_session():
    from caal.language_policy import LanguageSession

    session = LanguageSession(EN)
    assert session.observe("habla en español por favor", detected="es") == ES
    # It sticks for later turns, including ones detected as English.
    assert session.observe("Turn on the office lamp.", detected="en") == ES
    assert session.observe("switch back to english", detected="en") == EN
    assert session.observe("enciende la lámpara de la oficina", detected="es") == EN
    # The stored account preference is untouched by the conversational override.
    assert session.preference == EN


def test_a_session_for_an_unidentified_satellite_uses_the_safe_default():
    from caal.language_policy import LanguageSession

    session = LanguageSession(None)
    assert session.preference == AUTO
    assert session.current == EN


def test_the_english_directive_is_empty_so_english_behaviour_is_unchanged():
    from caal.language_policy import reply_directive

    assert reply_directive(EN) == ""
    spanish = reply_directive(ES)
    assert "español" in spanish.lower() or "spanish" in spanish.lower()
    # It must not invite the model to translate identifiers.
    assert "email" in spanish.lower() or "correo" in spanish.lower()
