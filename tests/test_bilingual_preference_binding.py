"""Stage 2: the reply language a session starts in comes from the verified principal.

FINAL-STAGE1 §3.7 listed this path as read-but-untested. These tests drive the
real `voice_agent.load_language_preference` against a real `LanguageStore`, and
assert that a second user's saved choice can never leak into the first's session.
"""

import pytest

import voice_agent
from caal.language_policy import AUTO, EN, ES, LanguageSession
from caal.language_store import LanguageStore


class _Store:
    def __init__(self, path):
        self.path = path

    def connect(self):
        import sqlite3

        connection = sqlite3.connect(self.path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        return connection


class _Identity:
    """The minimal shape `LanguageStore` needs: an identity runtime with a store."""

    def __init__(self, path):
        self.store = _Store(path)


@pytest.fixture
def identity(tmp_path):
    return _Identity(str(tmp_path / "identity.db"))


def test_a_saved_choice_is_read_back_for_that_user_only(identity):
    store = LanguageStore(identity)
    store.save_preference("user-a", ES)
    store.save_preference("user-b", EN)
    assert voice_agent.load_language_preference("user-a", identity=identity) == ES
    assert voice_agent.load_language_preference("user-b", identity=identity) == EN
    # Never saved, so never pinned.
    assert voice_agent.load_language_preference("user-c", identity=identity) == AUTO


def test_a_session_starts_in_the_verified_users_language(identity):
    LanguageStore(identity).save_preference("user-a", ES)
    session = LanguageSession(voice_agent.load_language_preference("user-a", identity=identity))
    assert session.preference == ES
    assert session.current == ES
    # And a different principal on the same store starts English-by-default.
    other = LanguageSession(voice_agent.load_language_preference("user-b", identity=identity))
    assert other.preference == AUTO
    assert other.current == EN


@pytest.mark.parametrize("user_id", [None, "", 0])
def test_without_a_signed_principal_there_is_nothing_to_look_up(identity, user_id):
    LanguageStore(identity).save_preference("user-a", ES)
    assert voice_agent.load_language_preference(user_id, identity=identity) == AUTO
    assert LanguageSession(AUTO).current == EN


def test_with_no_identity_runtime_the_session_falls_back_to_auto():
    assert voice_agent.load_language_preference("user-a", identity=None) == AUTO


def test_a_store_failure_degrades_to_auto_instead_of_raising(tmp_path):
    class BrokenStore:
        def connect(self):
            raise RuntimeError("identity database unavailable")

    class Broken:
        store = BrokenStore()

    assert voice_agent.load_language_preference("user-a", identity=Broken()) == AUTO


def test_a_saved_choice_is_never_overwritten_by_a_conversational_override(identity):
    store = LanguageStore(identity)
    store.save_preference("user-a", EN)
    session = LanguageSession(voice_agent.load_language_preference("user-a", identity=identity))
    assert session.observe("español por favor", "en") == ES
    # The override lives in the session; the stored row is untouched.
    assert store.preference("user-a") == EN
    assert voice_agent.load_language_preference("user-a", identity=identity) == EN
