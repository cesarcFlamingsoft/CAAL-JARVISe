"""The durable, per-user provider connection model and its OAuth state.

Properties pinned here, in the store rather than at the HTTP edge:

* the callback ``state`` is opaque, signed, single use, expiry bound and bound
  to the user who started the authorization;
* provider tokens are encrypted at rest under the profile key ring, bound to
  their own row, and are never part of any view;
* one user can never list, read, or revoke another user's connections;
* nothing about the schema or the API can hold a person's password.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from caal import profile_crypto
from caal.profile_crypto import KeyRing
from caal.provider_connections import (
    MAX_ALIASES,
    STATE_TTL_SECONDS,
    ConnectionStore,
    StateError,
    is_valid_connection_id,
    label_key,
    normalize_aliases,
    normalize_user_label,
)
from caal.user_store import MEMBER, Actor, UserStore

SECRET = "s" * 48
OTHER_SECRET = "o" * 48
NOW = 1_700_000_000
ACCESS = "ya29.ACCESS-TOKEN-PLAINTEXT"
REFRESH = "1//REFRESH-TOKEN-PLAINTEXT"


class Harness:
    def __init__(self, tmp_path) -> None:
        self.now = NOW
        self.ring = KeyRing.from_env(profile_crypto.generate_key_material(version=1))
        self.users = UserStore(tmp_path / "assistant.sqlite3", keyring=self.ring)
        self.store = ConnectionStore(self.users, keyring=self.ring, state_secret=SECRET)
        self.ana = self._user("ana@example.com")
        self.bo = self._user("bo@example.com")

    def _user(self, email: str) -> str:
        return self.users.create_user(
            email=email,
            display_name=email.split("@")[0],
            role=MEMBER,
            actor=Actor.system(),
            now=self.now,
        ).user_id

    def rows(self, table: str) -> list[sqlite3.Row]:
        with closing(self.users.connect()) as connection:
            return connection.execute(f"SELECT * FROM {table}").fetchall()

    def connect(self, user_id: str, provider: str = "google", **overrides):
        params = {
            "access_token": ACCESS,
            "refresh_token": REFRESH,
            "expires_in": 3600,
            "scopes": ("openid", "email"),
            "account_label": "ana@gmail.com",
            "now": self.now,
        }
        params.update(overrides)
        return self.store.complete_authorization(user_id, provider, **params)


@pytest.fixture
def h(tmp_path):
    return Harness(tmp_path)


# --- schema -----------------------------------------------------------------------


def test_migration_adds_the_connection_tables_to_the_shared_ledger(h, tmp_path) -> None:
    assert h.users.schema_version() >= 4
    tables = {row["name"] for row in h.rows("sqlite_master")}
    assert {"provider_connections", "oauth_states"} <= tables
    with closing(h.users.connect()) as connection:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(provider_connections)")
        }
    assert "provider_account_id" in columns
    # Re-opening the same file is idempotent.
    again = UserStore(tmp_path / "assistant.sqlite3", keyring=h.ring)
    assert again.schema_version() >= 4


def test_schema_cannot_hold_a_user_password(h) -> None:
    with closing(h.users.connect()) as connection:
        for table in ("provider_connections", "oauth_states"):
            columns = {
                row["name"].lower() for row in connection.execute(f"PRAGMA table_info({table})")
            }
            assert not any("password" in column for column in columns), (table, columns)
            assert not any("secret" in column for column in columns), (table, columns)


# --- state ------------------------------------------------------------------------


def test_begin_authorization_issues_an_opaque_state_the_owner_can_redeem_once(h) -> None:
    pending = h.store.begin_authorization(h.ana, "google", pkce=True, now=h.now)

    assert pending.expires_at == h.now + STATE_TTL_SECONDS
    assert 40 <= len(pending.state) <= 256 and pending.state.isprintable()
    assert " " not in pending.state and "\n" not in pending.state
    assert h.ana not in pending.state and "google" not in pending.state
    assert pending.code_challenge and len(pending.code_challenge) == 43
    assert "code_verifier" not in dir(pending)

    consumed = h.store.consume_state(pending.state, user_id=h.ana, now=h.now + 30)
    assert consumed.provider == "google"
    assert consumed.user_id == h.ana
    assert consumed.code_verifier and 43 <= len(consumed.code_verifier) <= 128
    assert pending.code_challenge == h.store.code_challenge_for(consumed.code_verifier)

    with pytest.raises(StateError):
        h.store.consume_state(pending.state, user_id=h.ana, now=h.now + 31)


def test_state_without_pkce_carries_no_verifier(h) -> None:
    pending = h.store.begin_authorization(h.ana, "zoho", pkce=False, now=h.now)
    assert pending.code_challenge is None
    consumed = h.store.consume_state(pending.state, user_id=h.ana, now=h.now)
    assert consumed.code_verifier is None


def test_state_expires(h) -> None:
    pending = h.store.begin_authorization(h.ana, "google", pkce=True, now=h.now)
    with pytest.raises(StateError):
        h.store.consume_state(pending.state, user_id=h.ana, now=pending.expires_at)
    # The expired row is gone, not merely refused.
    assert h.rows("oauth_states") == []


def test_state_is_bound_to_the_user_who_started_it_and_burns_on_misuse(h) -> None:
    pending = h.store.begin_authorization(h.ana, "google", pkce=True, now=h.now)
    with pytest.raises(StateError):
        h.store.consume_state(pending.state, user_id=h.bo, now=h.now)
    # A state presented by the wrong person is compromised: nobody may use it now.
    with pytest.raises(StateError):
        h.store.consume_state(pending.state, user_id=h.ana, now=h.now)


def test_tampered_forged_and_malformed_states_are_refused(h) -> None:
    pending = h.store.begin_authorization(h.ana, "google", pkce=True, now=h.now)
    state_id, signature = pending.state.split(".")

    forged_signature = state_id + "." + ("A" if signature[0] != "A" else "B") + signature[1:]
    other_secret = ConnectionStore(h.users, keyring=h.ring, state_secret=OTHER_SECRET)
    foreign = other_secret.begin_authorization(h.ana, "google", pkce=True, now=h.now).state

    for bad in (
        forged_signature,
        foreign,
        state_id,
        signature,
        "",
        "   ",
        "a.b",
        "x" * 300,
        "not\nprintable." + signature,
        None,
        42,
    ):
        with pytest.raises(StateError):
            h.store.consume_state(bad, user_id=h.ana, now=h.now)

    # The forgery above that named the genuine row is treated as a compromise
    # of that row, so the genuine state is burned too.
    with pytest.raises(StateError):
        h.store.consume_state(pending.state, user_id=h.ana, now=h.now)


def test_live_states_are_bounded_per_user(h) -> None:
    for _ in range(12):
        h.store.begin_authorization(h.ana, "google", pkce=True, now=h.now)
    h.store.begin_authorization(h.bo, "google", pkce=True, now=h.now)
    mine = [row for row in h.rows("oauth_states") if row["user_id"] == h.ana]
    assert len(mine) <= h.store.max_live_states_per_user
    assert len([row for row in h.rows("oauth_states") if row["user_id"] == h.bo]) == 1


# --- connections -------------------------------------------------------------------


def test_complete_authorization_encrypts_tokens_bound_to_their_row(h) -> None:
    connection = h.connect(h.ana)

    assert is_valid_connection_id(connection.connection_id)
    assert connection.provider == "google" and connection.status == "connected"
    assert connection.scopes == ("openid", "email")
    assert connection.account_label == "ana@gmail.com"
    assert connection.token_expires_at == h.now + 3600
    assert connection.has_refresh_token is True
    assert connection.connected_at == h.now

    view = connection.view()
    assert set(view) == {
        "connection_id",
        "provider",
        "status",
        "account_label",
        "user_label",
        "aliases",
        "scopes",
        "token_expires_at",
        "has_refresh_token",
        "connected_at",
        "updated_at",
    }
    assert ACCESS not in str(view) and REFRESH not in str(view)
    assert ACCESS not in repr(connection) and REFRESH not in repr(connection)

    (row,) = h.rows("provider_connections")
    for column in row.keys():
        assert ACCESS not in str(row[column]) and REFRESH not in str(row[column]), column
    assert row["access_token_enc"].startswith("enc:v1:")
    assert row["refresh_token_enc"].startswith("enc:v1:")

    credentials = h.store.credentials(h.ana, connection.connection_id)
    assert credentials is not None
    assert credentials.access_token == ACCESS and credentials.refresh_token == REFRESH
    assert credentials.token_expires_at == h.now + 3600
    assert ACCESS not in repr(credentials) and REFRESH not in repr(credentials)

    # Ciphertext moved to another row fails to authenticate rather than decrypting.
    other = h.connect(h.bo, account_label="bo@gmail.com")
    with closing(h.users.connect()) as db:
        db.execute(
            "UPDATE provider_connections SET access_token_enc = ? WHERE connection_id = ?",
            (row["access_token_enc"], other.connection_id),
        )
    assert h.store.credentials(h.bo, other.connection_id) is None


def test_reconnecting_replaces_tokens_and_keeps_the_connection_id(h) -> None:
    first = h.connect(h.ana)
    h.now += 100
    second = h.connect(
        h.ana,
        access_token="ya29.SECOND",
        refresh_token=None,
        expires_in=None,
        scopes=("openid",),
        account_label="ana.second@gmail.com",
    )

    assert second.connection_id == first.connection_id
    assert second.has_refresh_token is False
    assert second.token_expires_at is None
    assert second.account_label == "ana.second@gmail.com"
    assert second.updated_at == h.now
    credentials = h.store.credentials(h.ana, first.connection_id)
    assert credentials.access_token == "ya29.SECOND" and credentials.refresh_token is None
    assert len(h.rows("provider_connections")) == 1


def test_refresh_credentials_replaces_the_access_token_and_keeps_or_rotates_refresh(h) -> None:
    mine = h.connect(h.ana)
    theirs = h.connect(h.bo, access_token="ya29.BO", refresh_token="1//BO")
    h.now += 3600

    # Google-style: a new access token and no new refresh token -> the old one stays.
    refreshed = h.store.refresh_credentials(
        h.ana, mine.connection_id, access_token="ya29.FRESH", expires_in=1800, now=h.now
    )
    assert refreshed is not None
    assert refreshed.connection_id == mine.connection_id
    assert refreshed.token_expires_at == h.now + 1800
    assert refreshed.updated_at == h.now
    assert refreshed.has_refresh_token is True
    credentials = h.store.credentials(h.ana, mine.connection_id)
    assert credentials.access_token == "ya29.FRESH" and credentials.refresh_token == REFRESH
    assert credentials.token_expires_at == h.now + 1800

    # Microsoft-style rotation: a new refresh token replaces the old one.
    rotated = h.store.refresh_credentials(
        h.ana,
        mine.connection_id,
        access_token="ya29.ROTATED",
        refresh_token="1//ROTATED",
        expires_in=None,
        now=h.now + 1,
    )
    assert rotated is not None and rotated.token_expires_at is None
    credentials = h.store.credentials(h.ana, mine.connection_id)
    assert credentials.access_token == "ya29.ROTATED" and credentials.refresh_token == "1//ROTATED"

    # Ciphertext at rest, nothing in the clear.
    rows = h.rows("provider_connections")
    row = next(r for r in rows if r["connection_id"] == mine.connection_id)
    for column in row.keys():
        for secret in ("ya29.FRESH", "ya29.ROTATED", "1//ROTATED", REFRESH):
            assert secret not in str(row[column]), column

    # Nobody else's connection, and no revoked or unknown one, can be refreshed.
    assert (
        h.store.refresh_credentials(
            h.bo, mine.connection_id, access_token="ya29.STOLEN", expires_in=10, now=h.now
        )
        is None
    )
    assert h.store.credentials(h.ana, mine.connection_id).access_token == "ya29.ROTATED"
    assert h.store.credentials(h.bo, theirs.connection_id).access_token == "ya29.BO"
    assert h.store.revoke_connection(h.bo, theirs.connection_id, now=h.now) is True
    assert (
        h.store.refresh_credentials(
            h.bo, theirs.connection_id, access_token="ya29.LATE", expires_in=10, now=h.now
        )
        is None
    )
    assert h.store.credentials(h.bo, theirs.connection_id) is None
    for bad in ("con_" + "z" * 24, "", None):
        assert (
            h.store.refresh_credentials(
                h.ana, bad, access_token="ya29.X", expires_in=10, now=h.now
            )
            is None
        )

    # Tokens are validated exactly as at connect time.
    with pytest.raises(ValueError):
        h.store.refresh_credentials(
            h.ana, mine.connection_id, access_token="", expires_in=10, now=h.now
        )
    with pytest.raises(ValueError):
        h.store.refresh_credentials(
            h.ana, mine.connection_id, access_token="ya29.X", expires_in=-1, now=h.now
        )
    with pytest.raises(ValueError):
        h.store.refresh_credentials(
            h.ana, mine.connection_id, access_token="ya29.X", refresh_token="", now=h.now
        )
    assert h.store.credentials(h.ana, mine.connection_id).access_token == "ya29.ROTATED"


def test_listing_and_reading_are_scoped_to_the_owner(h) -> None:
    mine = h.connect(h.ana)
    theirs = h.connect(h.bo, provider="microsoft", account_label="bo@outlook.com")

    assert [c.connection_id for c in h.store.list_connections(h.ana)] == [mine.connection_id]
    assert [c.connection_id for c in h.store.list_connections(h.bo)] == [theirs.connection_id]
    assert h.store.get_connection(h.ana, theirs.connection_id) is None
    assert h.store.credentials(h.ana, theirs.connection_id) is None
    assert h.store.get_connection(h.bo, theirs.connection_id).account_label == "bo@outlook.com"
    for bad in ("con_" + "z" * 24, "usr_" + "0" * 24, "", None):
        assert h.store.get_connection(h.ana, bad) is None
        assert h.store.credentials(h.ana, bad) is None
        assert h.store.revoke_connection(h.ana, bad, now=h.now) is False
    assert h.store.list_connections("not-a-user") == []


def test_revoke_wipes_tokens_and_hides_the_connection(h) -> None:
    mine = h.connect(h.ana)
    theirs = h.connect(h.bo)

    assert h.store.revoke_connection(h.bo, mine.connection_id, now=h.now) is False
    assert h.store.credentials(h.ana, mine.connection_id) is not None

    assert h.store.revoke_connection(h.ana, mine.connection_id, now=h.now + 5) is True
    assert h.store.list_connections(h.ana) == []
    assert h.store.get_connection(h.ana, mine.connection_id) is None
    assert h.store.credentials(h.ana, mine.connection_id) is None
    assert h.store.revoke_connection(h.ana, mine.connection_id, now=h.now + 6) is False

    row = next(
        r for r in h.rows("provider_connections") if r["connection_id"] == mine.connection_id
    )
    assert row["status"] == "revoked"
    assert row["access_token_enc"] is None and row["refresh_token_enc"] is None
    assert row["revoked_at"] == h.now + 5
    # The other user's connection is untouched.
    assert h.store.credentials(h.bo, theirs.connection_id).access_token == ACCESS

    # Reconnecting after a revoke reuses the row and brings it back.
    again = h.connect(h.ana, access_token="ya29.AGAIN")
    assert again.connection_id == mine.connection_id and again.status == "connected"
    assert h.store.credentials(h.ana, mine.connection_id).access_token == "ya29.AGAIN"


def test_inputs_are_validated_and_never_trusted(h) -> None:
    with pytest.raises(ValueError):
        h.store.begin_authorization(h.ana, "apple", pkce=True, now=h.now)
    with pytest.raises(ValueError):
        h.store.begin_authorization("not-a-user", "google", pkce=True, now=h.now)
    with pytest.raises(ValueError):
        h.connect(h.ana, provider="Google")
    with pytest.raises(ValueError):
        h.connect(h.ana, access_token="")
    with pytest.raises(ValueError):
        h.connect(h.ana, access_token="has\nnewline")
    with pytest.raises(ValueError):
        h.connect(h.ana, account_label="x" * 300)
    with pytest.raises(ValueError):
        h.connect(h.ana, scopes=("ok", "bad scope with space"))
    assert h.rows("provider_connections") == []


# --- several accounts from one provider ------------------------------------------------


def test_two_google_accounts_for_one_user_coexist_and_reconnect_independently(h) -> None:
    personal = h.connect(
        h.ana, provider_account_id="sub-personal", account_label="cesar.personal@gmail.com"
    )
    h.now += 10
    work = h.connect(
        h.ana,
        provider_account_id="sub-work",
        account_label="cesar.work@gmail.com",
        access_token="ya29.WORK",
        refresh_token="1//WORK",
    )

    assert personal.connection_id != work.connection_id
    assert personal.provider_account_id == "sub-personal"
    assert work.provider_account_id == "sub-work"
    assert "provider_account_id" not in personal.view()
    listed = h.store.list_connections(h.ana)
    assert [c.connection_id for c in listed] == [personal.connection_id, work.connection_id]
    assert [c.account_label for c in listed] == ["cesar.personal@gmail.com", "cesar.work@gmail.com"]
    assert h.store.credentials(h.ana, personal.connection_id).access_token == ACCESS
    assert h.store.credentials(h.ana, work.connection_id).access_token == "ya29.WORK"

    # Reconnecting the personal account updates that connection only.
    h.now += 10
    again = h.connect(
        h.ana,
        provider_account_id="sub-personal",
        account_label="cesar.personal@gmail.com",
        access_token="ya29.PERSONAL-AGAIN",
        refresh_token=None,
    )
    assert again.connection_id == personal.connection_id
    assert again.updated_at == h.now and again.has_refresh_token is False
    assert h.store.credentials(h.ana, personal.connection_id).access_token == "ya29.PERSONAL-AGAIN"
    assert h.store.credentials(h.ana, work.connection_id).access_token == "ya29.WORK"
    assert h.store.credentials(h.ana, work.connection_id).refresh_token == "1//WORK"
    assert h.store.get_connection(h.ana, work.connection_id).updated_at == h.now - 10
    assert len(h.rows("provider_connections")) == 2

    # Revoking one leaves the other live; reconnecting the revoked one revives its row.
    assert h.store.revoke_connection(h.ana, work.connection_id, now=h.now) is True
    assert [c.connection_id for c in h.store.list_connections(h.ana)] == [personal.connection_id]
    revived = h.connect(h.ana, provider_account_id="sub-work", account_label="cesar.work@gmail.com")
    assert revived.connection_id == work.connection_id and revived.status == "connected"
    assert len(h.rows("provider_connections")) == 2


def test_the_same_provider_account_linked_by_two_users_stays_separate(h) -> None:
    mine = h.connect(h.ana, provider_account_id="sub-shared", account_label="shared@gmail.com")
    theirs = h.connect(
        h.bo,
        provider_account_id="sub-shared",
        account_label="shared@gmail.com",
        access_token="ya29.BO",
    )
    assert mine.connection_id != theirs.connection_id
    assert h.store.credentials(h.ana, mine.connection_id).access_token == ACCESS
    assert h.store.credentials(h.bo, theirs.connection_id).access_token == "ya29.BO"
    assert h.store.get_connection(h.ana, theirs.connection_id) is None
    assert h.store.get_connection(h.bo, mine.connection_id) is None

    h.connect(h.ana, provider_account_id="sub-shared", access_token="ya29.ANA-AGAIN")
    assert h.store.credentials(h.bo, theirs.connection_id).access_token == "ya29.BO"
    assert h.store.credentials(h.ana, mine.connection_id).access_token == "ya29.ANA-AGAIN"
    assert len(h.rows("provider_connections")) == 2


def test_provider_account_ids_are_validated(h) -> None:
    for bad in ("", "has space", "x" * 300, "new\nline", 42):
        with pytest.raises(ValueError):
            h.connect(h.ana, provider_account_id=bad)
    assert h.rows("provider_connections") == []




# --- the names a user gives their own accounts -----------------------------------


def test_user_names_are_trimmed_bounded_and_deduplicated() -> None:
    assert normalize_user_label("  Work   laptop ") == "Work laptop"
    assert normalize_user_label("") is None
    assert normalize_user_label("   ") is None
    assert normalize_user_label(None) is None
    # Unicode is welcome; control characters and punctuation-only names are not.
    assert normalize_user_label("Universit\u00e9") == "Universit\u00e9"
    for bad in ("x" * 49, "line\nbreak", "\u2013", 7, ["work"]):
        with pytest.raises(ValueError):
            normalize_user_label(bad)

    assert normalize_aliases(None) == ()
    assert normalize_aliases([]) == ()
    # Case, spacing and accents do not make a second name; the first spelling wins.
    assert normalize_aliases(["Work", " work ", "WORK", "", "  "]) == ("Work",)
    assert normalize_aliases(["wife\u2019s", "wifes"]) == ("wife\u2019s",)
    assert normalize_aliases(["office", "day job"]) == ("office", "day job")
    for bad in ("work", ["x" * 49], [42], [None], ["#"], [["work"]]):
        with pytest.raises(ValueError):
            normalize_aliases(bad)
    with pytest.raises(ValueError):
        normalize_aliases([f"name-{index}" for index in range(MAX_ALIASES + 1)])
    assert label_key("  W\u00f6rk\u2019s  Mail ") == "works mail"
    assert label_key("###") == ""


def test_naming_a_connection_leaves_the_grant_and_the_provider_label_alone(h) -> None:
    connection = h.connect(h.ana)

    named = h.store.set_labels(
        h.ana,
        connection.connection_id,
        user_label="  work  ",
        aliases=["Office", "office", "day job"],
        now=NOW + 5,
    )
    assert named.user_label == "work"
    assert named.aliases == ("Office", "day job")
    # Everything the provider decided is exactly as it was.
    assert named.account_label == "ana@gmail.com"
    assert named.scopes == ("openid", "email")
    assert named.token_expires_at == connection.token_expires_at
    assert named.updated_at == NOW + 5
    assert h.store.credentials(h.ana, connection.connection_id).access_token == ACCESS
    assert h.store.credentials(h.ana, connection.connection_id).refresh_token == REFRESH
    assert named.view()["user_label"] == "work" and named.view()["aliases"] == ["Office", "day job"]

    # One field at a time: the other is left as it was.
    only_label = h.store.set_labels(h.ana, connection.connection_id, user_label="university")
    assert only_label.user_label == "university" and only_label.aliases == ("Office", "day job")
    only_aliases = h.store.set_labels(h.ana, connection.connection_id, aliases=["school"])
    assert only_aliases.user_label == "university" and only_aliases.aliases == ("school",)
    # Nothing at all is a no-op that still returns the row.
    assert h.store.set_labels(h.ana, connection.connection_id).aliases == ("school",)

    # Clearing is explicit, and reconnecting the account keeps the names.
    cleared = h.store.set_labels(h.ana, connection.connection_id, user_label=None, aliases=[])
    assert cleared.user_label is None and cleared.aliases == ()
    h.store.set_labels(h.ana, connection.connection_id, user_label="work")
    again = h.connect(h.ana, access_token="ya29.FRESH", now=NOW + 10)
    assert again.connection_id == connection.connection_id
    assert again.user_label == "work"


def test_only_the_owner_of_a_live_connection_can_name_it(h) -> None:
    mine = h.connect(h.ana)
    theirs = h.connect(h.bo, provider_account_id="sub-bo", account_label="bo@gmail.com")

    assert h.store.set_labels(h.bo, mine.connection_id, user_label="stolen") is None
    assert h.store.set_labels(h.ana, theirs.connection_id, user_label="stolen") is None
    assert h.store.set_labels(h.ana, "con_" + "f" * 24, user_label="ghost") is None
    assert h.store.set_labels(h.ana, "not-an-id", user_label="ghost") is None
    assert h.store.set_labels("usr_" + "9" * 24, mine.connection_id, user_label="ghost") is None
    assert h.store.get_connection(h.ana, mine.connection_id).user_label is None
    assert h.store.get_connection(h.bo, theirs.connection_id).user_label is None

    # A refused name is refused before anything is written.
    with pytest.raises(ValueError):
        h.store.set_labels(h.ana, mine.connection_id, user_label="x" * 200)
    assert h.store.get_connection(h.ana, mine.connection_id).user_label is None

    # A revoked connection is no longer nameable.
    h.store.revoke_connection(h.ana, mine.connection_id, now=NOW + 1)
    assert h.store.set_labels(h.ana, mine.connection_id, user_label="work") is None


def _build_version_3_database(path, ring, *, user_id: str, connection_id: str) -> None:
    """A database exactly as the previous slice left it: one row per (user, provider)."""
    from caal import user_store

    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version INTEGER PRIMARY KEY, applied_at INTEGER NOT NULL)"
        )
        for version, statements in user_store._MIGRATIONS:
            if version > 3:
                continue
            for statement in statements:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)", (version, NOW)
            )
        connection.execute(
            "INSERT INTO users (user_id, email, display_name, role, status, created_at, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, "legacy@example.com", "legacy", "member", "active", NOW, NOW),
        )
        connection.execute(
            "INSERT INTO provider_connections (connection_id, user_id, provider, status, "
            "account_label, scopes, access_token_enc, refresh_token_enc, token_expires_at, "
            "created_at, updated_at, connected_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                connection_id,
                user_id,
                "google",
                "connected",
                "legacy@gmail.com",
                '["openid","email"]',
                ring.encrypt(ACCESS, aad=f"caal.provider_connections.access_token:{connection_id}"),
                None,
                NOW + 3600,
                NOW,
                NOW,
                NOW,
            ),
        )
        connection.commit()


def test_migration_to_version_7_leaves_old_rows_readable_and_nameable(tmp_path) -> None:
    """A database written before user names keeps everything and simply has none yet."""
    ring = KeyRing.from_env(profile_crypto.generate_key_material(version=1))
    path = tmp_path / "assistant.sqlite3"
    legacy_user = "usr_" + "3" * 24
    legacy_id = "con_" + "c" * 24
    _build_version_3_database(path, ring, user_id=legacy_user, connection_id=legacy_id)

    users = UserStore(path, keyring=ring)
    assert users.schema_version() >= 7
    store = ConnectionStore(users, keyring=ring, state_secret=SECRET)

    (legacy,) = store.list_connections(legacy_user)
    assert legacy.account_label == "legacy@gmail.com"
    assert legacy.user_label is None and legacy.aliases == ()
    assert store.credentials(legacy_user, legacy_id).access_token == ACCESS

    named = store.set_labels(legacy_user, legacy_id, user_label="work", aliases=["office"])
    assert named.user_label == "work" and named.aliases == ("office",)
    # The provider's own label is untouched by naming, and so are the tokens.
    assert named.account_label == "legacy@gmail.com"
    assert store.credentials(legacy_user, legacy_id).access_token == ACCESS
    # Migrating again changes nothing.
    assert UserStore(path, keyring=ring).schema_version() == users.schema_version()
    assert store.get_connection(legacy_user, legacy_id).aliases == ("office",)


def test_migration_from_version_3_keeps_legacy_connections_and_lets_them_be_claimed(
    tmp_path,
) -> None:
    ring = KeyRing.from_env(profile_crypto.generate_key_material(version=1))
    path = tmp_path / "assistant.sqlite3"
    legacy_user = "usr_" + "1" * 24
    legacy_id = "con_" + "a" * 24
    _build_version_3_database(path, ring, user_id=legacy_user, connection_id=legacy_id)

    users = UserStore(path, keyring=ring)
    assert users.schema_version() >= 4
    store = ConnectionStore(users, keyring=ring, state_secret=SECRET)

    (legacy,) = store.list_connections(legacy_user)
    assert legacy.connection_id == legacy_id
    assert legacy.provider_account_id is None
    assert legacy.account_label == "legacy@gmail.com"
    assert legacy.scopes == ("openid", "email")
    assert store.credentials(legacy_user, legacy_id).access_token == ACCESS
    assert store.list_connections("usr_" + "2" * 24) == []

    # Reconnecting the same account (same label) claims the legacy row and
    # gives it its identity; a different account gets its own row.
    claimed = store.complete_authorization(
        legacy_user,
        "google",
        access_token="ya29.CLAIMED",
        provider_account_id="sub-legacy",
        account_label="Legacy@gmail.com",
        now=NOW + 5,
    )
    assert claimed.connection_id == legacy_id
    assert claimed.provider_account_id == "sub-legacy"
    other = store.complete_authorization(
        legacy_user,
        "google",
        access_token="ya29.OTHER",
        provider_account_id="sub-other",
        account_label="other@gmail.com",
        now=NOW + 6,
    )
    assert other.connection_id != legacy_id
    assert [c.connection_id for c in store.list_connections(legacy_user)] == [
        legacy_id,
        other.connection_id,
    ]
    assert store.credentials(legacy_user, legacy_id).access_token == "ya29.CLAIMED"

    # Once identified, a legacy row is never claimed by a different account.
    third = store.complete_authorization(
        legacy_user,
        "google",
        access_token="ya29.THIRD",
        provider_account_id="sub-third",
        account_label="legacy@gmail.com",
        now=NOW + 7,
    )
    assert third.connection_id not in (legacy_id, other.connection_id)
    assert store.credentials(legacy_user, legacy_id).access_token == "ya29.CLAIMED"

    # Re-opening is idempotent and the migration ledger is complete.
    assert UserStore(path, keyring=ring).schema_version() >= 4
