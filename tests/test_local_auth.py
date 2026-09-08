"""Standalone password sign-in: credentials, lockout, and server-side sessions.

This is the identity path that works with no Cloudflare Access in front of the
deployment. The properties pinned here are the ones an attacker probes first:
whether a response distinguishes an unknown account from a wrong password,
whether brute force is bounded, whether a session can outlive its revocation,
and whether anything sensitive reaches the audit trail.

Hashing cost is turned down to keep the suite fast; production cost comes from
``caal.password_hash.DEFAULT_PARAMS``.
"""

from __future__ import annotations

import pytest

from caal.local_auth import (
    InvalidCredentialsError,
    LocalAuth,
    SessionPolicy,
)
from caal.password_hash import Argon2Params, PasswordPolicyError, hash_password
from caal.profile_crypto import KeyRing, generate_key_material
from caal.user_store import ACTIVE, ADMIN, MEMBER, SUSPENDED, Actor, UserStore

CHEAP = Argon2Params(memory_kib=64, iterations=1, lanes=1)
ADMIN_EMAIL = "cesarc@mexcantech.com"
MEMBER_EMAIL = "someone@example.com"
GOOD_PASSWORD = "a perfectly fine passphrase"
OTHER_PASSWORD = "an entirely different one"
NOW = 1_700_000_000


class Harness:
    def __init__(self, tmp_path, *, policy: SessionPolicy | None = None) -> None:
        self.now = NOW
        self.store = UserStore(
            tmp_path / "assistant.sqlite3",
            keyring=KeyRing.from_env(generate_key_material(version=1)),
        )
        self.auth = LocalAuth(
            self.store,
            params=CHEAP,
            policy=policy or SessionPolicy(),
            clock=lambda: self.now,
        )

    def make_user(self, email: str, *, role: str = MEMBER, password: str | None = GOOD_PASSWORD):
        profile = self.store.create_user(
            email=email, display_name=email.split("@")[0], role=role, actor=Actor.system()
        )
        if password is not None:
            self.auth.set_password(profile.user_id, password, actor=Actor.system())
        return profile

    def advance(self, seconds: int) -> None:
        self.now += seconds


@pytest.fixture
def h(tmp_path):
    return Harness(tmp_path)


# --- sign in ---------------------------------------------------------------------


def test_correct_password_issues_a_session(h) -> None:
    profile = h.make_user(MEMBER_EMAIL)

    result = h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD)

    assert result.ok
    assert result.user.user_id == profile.user_id
    assert result.token and len(result.token) >= 32
    assert result.expires_at > h.now
    assert result.must_change_password is False


def test_email_is_matched_case_insensitively_and_trimmed(h) -> None:
    h.make_user(MEMBER_EMAIL)

    assert h.auth.authenticate("  SomeOne@Example.COM  ", GOOD_PASSWORD).ok


def test_wrong_password_is_refused(h) -> None:
    h.make_user(MEMBER_EMAIL)

    result = h.auth.authenticate(MEMBER_EMAIL, OTHER_PASSWORD)

    assert not result.ok
    assert result.reason == "invalid_credentials"
    assert result.token is None


def test_unknown_account_is_indistinguishable_from_a_wrong_password(h) -> None:
    h.make_user(MEMBER_EMAIL)

    unknown = h.auth.authenticate("nobody@example.com", GOOD_PASSWORD)
    wrong = h.auth.authenticate(MEMBER_EMAIL, OTHER_PASSWORD)
    malformed = h.auth.authenticate("not-an-email", GOOD_PASSWORD)

    assert unknown.reason == wrong.reason == malformed.reason == "invalid_credentials"
    assert unknown.retry_after is wrong.retry_after is None


def test_account_without_a_local_password_cannot_be_signed_into(h) -> None:
    h.make_user(MEMBER_EMAIL, password=None)

    result = h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD)

    assert result.reason == "invalid_credentials"


def test_suspended_account_is_refused_without_admitting_it_exists(h) -> None:
    profile = h.make_user(MEMBER_EMAIL)
    h.store.admin_update(profile.user_id, status=SUSPENDED, actor=Actor.system())

    result = h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD)

    assert result.reason == "invalid_credentials"
    assert result.token is None


# --- brute force -----------------------------------------------------------------


def test_repeated_failures_lock_the_account(h) -> None:
    h.make_user(MEMBER_EMAIL)

    for _ in range(SessionPolicy().max_failed_attempts):
        assert h.auth.authenticate(MEMBER_EMAIL, OTHER_PASSWORD).reason == "invalid_credentials"

    locked = h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD)
    assert locked.reason == "locked"
    assert locked.retry_after and locked.retry_after > 0
    assert locked.token is None


def test_a_lockout_does_not_reveal_itself_to_someone_guessing(h) -> None:
    h.make_user(MEMBER_EMAIL)
    for _ in range(SessionPolicy().max_failed_attempts):
        h.auth.authenticate(MEMBER_EMAIL, OTHER_PASSWORD)

    # The attacker still does not have the password, so they learn nothing new.
    still_guessing = h.auth.authenticate(MEMBER_EMAIL, "yet another wrong guess")

    assert still_guessing.reason == "invalid_credentials"
    assert still_guessing.retry_after is None


def test_lockout_expires_and_the_right_password_then_works(h) -> None:
    h.make_user(MEMBER_EMAIL)
    for _ in range(SessionPolicy().max_failed_attempts):
        h.auth.authenticate(MEMBER_EMAIL, OTHER_PASSWORD)

    h.advance(SessionPolicy().base_lockout_seconds + 1)

    assert h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).ok


def test_lockout_backs_off_further_on_each_new_round_of_failures(h) -> None:
    h.make_user(MEMBER_EMAIL)
    policy = SessionPolicy()
    for _ in range(policy.max_failed_attempts):
        h.auth.authenticate(MEMBER_EMAIL, OTHER_PASSWORD)
    first = h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).retry_after

    h.advance(first + 1)
    h.auth.authenticate(MEMBER_EMAIL, OTHER_PASSWORD)
    second = h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).retry_after

    assert first and second and second > first
    assert second <= policy.max_lockout_seconds


def test_lockout_backoff_is_capped(h) -> None:
    h.make_user(MEMBER_EMAIL)
    policy = SessionPolicy()
    for _ in range(policy.max_failed_attempts + 30):
        h.auth.authenticate(MEMBER_EMAIL, OTHER_PASSWORD)
        h.advance(policy.max_lockout_seconds + 1)
    h.auth.authenticate(MEMBER_EMAIL, OTHER_PASSWORD)

    retry_after = h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).retry_after
    assert retry_after == policy.max_lockout_seconds


def test_a_successful_sign_in_clears_the_failure_count(h) -> None:
    h.make_user(MEMBER_EMAIL)
    policy = SessionPolicy()
    for _ in range(policy.max_failed_attempts - 1):
        h.auth.authenticate(MEMBER_EMAIL, OTHER_PASSWORD)

    assert h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).ok

    for _ in range(policy.max_failed_attempts - 1):
        h.auth.authenticate(MEMBER_EMAIL, OTHER_PASSWORD)
    assert h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).ok


def test_locking_one_account_does_not_lock_another(h) -> None:
    h.make_user(MEMBER_EMAIL)
    h.make_user(ADMIN_EMAIL, role=ADMIN)
    for _ in range(SessionPolicy().max_failed_attempts):
        h.auth.authenticate(MEMBER_EMAIL, OTHER_PASSWORD)

    assert h.auth.authenticate(ADMIN_EMAIL, GOOD_PASSWORD).ok


# --- sessions --------------------------------------------------------------------


def test_a_session_token_resolves_to_its_user(h) -> None:
    profile = h.make_user(MEMBER_EMAIL)
    token = h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).token

    session = h.auth.verify_session(token)

    assert session is not None
    assert session.user.user_id == profile.user_id
    assert session.must_change_password is False


def test_the_raw_token_is_never_stored(h, tmp_path) -> None:
    h.make_user(MEMBER_EMAIL)
    token = h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).token

    blob = (tmp_path / "assistant.sqlite3").read_bytes()

    assert token.encode() not in blob


@pytest.mark.parametrize("bogus", ["", "x", "not-a-token", None, 12345, "a" * 500])
def test_a_bogus_token_resolves_to_nothing(h, bogus) -> None:
    h.make_user(MEMBER_EMAIL)
    h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD)

    assert h.auth.verify_session(bogus) is None


def test_a_session_expires_when_left_idle(h) -> None:
    h.make_user(MEMBER_EMAIL)
    token = h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).token

    h.advance(SessionPolicy().idle_seconds + 1)

    assert h.auth.verify_session(token) is None


def test_use_slides_the_idle_window_but_not_past_the_absolute_cap(h) -> None:
    policy = SessionPolicy(idle_seconds=100, absolute_seconds=250)
    h.auth = LocalAuth(h.store, params=CHEAP, policy=policy, clock=lambda: h.now)
    h.make_user(MEMBER_EMAIL)
    token = h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).token

    for _ in range(3):
        h.advance(80)
        assert h.auth.verify_session(token) is not None

    h.advance(80)  # now past the 250s absolute cap
    assert h.auth.verify_session(token) is None


def test_logout_revokes_only_that_session(h) -> None:
    h.make_user(MEMBER_EMAIL)
    first = h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).token
    second = h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).token

    assert h.auth.revoke_session(first) is True

    assert h.auth.verify_session(first) is None
    assert h.auth.verify_session(second) is not None


def test_revoking_an_unknown_token_is_a_no_op_not_an_error(h) -> None:
    assert h.auth.revoke_session("nope") is False
    assert h.auth.revoke_session(None) is False


def test_suspending_a_user_kills_their_live_sessions(h) -> None:
    profile = h.make_user(MEMBER_EMAIL)
    token = h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).token

    h.store.admin_update(profile.user_id, status=SUSPENDED, actor=Actor.system())

    assert h.auth.verify_session(token) is None


def test_sessions_per_user_are_bounded(h) -> None:
    policy = SessionPolicy(max_sessions_per_user=3)
    h.auth = LocalAuth(h.store, params=CHEAP, policy=policy, clock=lambda: h.now)
    h.make_user(MEMBER_EMAIL)

    tokens = []
    for _ in range(5):
        h.advance(1)
        tokens.append(h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).token)

    alive = [t for t in tokens if h.auth.verify_session(t) is not None]
    assert len(alive) == 3
    assert alive == tokens[-3:]


# --- changing a password ---------------------------------------------------------


def test_changing_a_password_requires_the_current_one(h) -> None:
    profile = h.make_user(MEMBER_EMAIL)

    with pytest.raises(InvalidCredentialsError):
        h.auth.change_password(profile.user_id, "wrong current", "a brand new passphrase")

    assert h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).ok


def test_changing_a_password_replaces_it_and_enforces_policy(h) -> None:
    profile = h.make_user(MEMBER_EMAIL)

    with pytest.raises(PasswordPolicyError):
        h.auth.change_password(profile.user_id, GOOD_PASSWORD, "short")

    h.auth.change_password(profile.user_id, GOOD_PASSWORD, "a brand new passphrase")

    assert not h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).ok
    assert h.auth.authenticate(MEMBER_EMAIL, "a brand new passphrase").ok


def test_reusing_the_current_password_is_refused(h) -> None:
    profile = h.make_user(MEMBER_EMAIL)

    with pytest.raises(PasswordPolicyError):
        h.auth.change_password(profile.user_id, GOOD_PASSWORD, GOOD_PASSWORD)


def test_changing_a_password_revokes_every_other_session(h) -> None:
    profile = h.make_user(MEMBER_EMAIL)
    keep = h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).token
    stolen = h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).token

    h.auth.change_password(
        profile.user_id, GOOD_PASSWORD, "a brand new passphrase", keep_token=keep
    )

    assert h.auth.verify_session(stolen) is None
    assert h.auth.verify_session(keep) is not None


def test_changing_a_password_without_keeping_one_revokes_all(h) -> None:
    profile = h.make_user(MEMBER_EMAIL)
    token = h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).token

    h.auth.change_password(profile.user_id, GOOD_PASSWORD, "a brand new passphrase")

    assert h.auth.verify_session(token) is None


# --- forced change ---------------------------------------------------------------


def test_a_forced_change_flag_travels_from_credential_to_session(h) -> None:
    profile = h.make_user(MEMBER_EMAIL, password=None)
    h.auth.set_password(profile.user_id, GOOD_PASSWORD, actor=Actor.system(), must_change=True)

    result = h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD)

    assert result.ok
    assert result.must_change_password is True
    assert h.auth.verify_session(result.token).must_change_password is True


def test_changing_the_password_clears_the_forced_change_flag(h) -> None:
    profile = h.make_user(MEMBER_EMAIL, password=None)
    h.auth.set_password(profile.user_id, GOOD_PASSWORD, actor=Actor.system(), must_change=True)
    token = h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).token

    h.auth.change_password(
        profile.user_id, GOOD_PASSWORD, "a brand new passphrase", keep_token=token
    )

    assert h.auth.verify_session(token).must_change_password is False
    assert h.auth.authenticate(MEMBER_EMAIL, "a brand new passphrase").must_change_password is False


# --- administrative reset --------------------------------------------------------


def test_an_admin_reset_issues_a_random_one_time_password(h) -> None:
    admin = h.make_user(ADMIN_EMAIL, role=ADMIN)
    target = h.make_user(MEMBER_EMAIL)
    live = h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).token

    issued = h.auth.admin_reset_password(target.user_id, actor=Actor.for_user(admin))

    assert len(issued) >= 20
    assert h.auth.verify_session(live) is None, "reset must kill existing sessions"
    assert not h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).ok
    result = h.auth.authenticate(MEMBER_EMAIL, issued)
    assert result.ok and result.must_change_password is True


def test_two_resets_never_produce_the_same_password(h) -> None:
    admin = h.make_user(ADMIN_EMAIL, role=ADMIN)
    target = h.make_user(MEMBER_EMAIL)
    actor = Actor.for_user(admin)

    assert h.auth.admin_reset_password(target.user_id, actor=actor) != h.auth.admin_reset_password(
        target.user_id, actor=actor
    )


def test_a_member_may_not_reset_anyone(h) -> None:
    member = h.make_user(MEMBER_EMAIL)
    other = h.make_user("third@example.com")

    with pytest.raises(PermissionError):
        h.auth.admin_reset_password(other.user_id, actor=Actor.for_user(member))
    with pytest.raises(PermissionError):
        h.auth.set_password(other.user_id, "another passphrase", actor=Actor.for_user(member))


def test_resetting_an_unknown_user_is_refused(h) -> None:
    admin = h.make_user(ADMIN_EMAIL, role=ADMIN)

    from caal.user_store import UnknownUserError

    with pytest.raises(UnknownUserError):
        h.auth.admin_reset_password("usr_" + "0" * 24, actor=Actor.for_user(admin))


# --- bootstrap -------------------------------------------------------------------


class TestBootstrap:
    def test_seeds_the_configured_admin_into_an_empty_store(self, h) -> None:
        encoded = hash_password(GOOD_PASSWORD, params=CHEAP)

        outcome = h.auth.ensure_bootstrap_admin(ADMIN_EMAIL, encoded)

        assert outcome.created is True
        assert outcome.credential_installed is True
        profile = h.store.get_user(outcome.user_id)
        assert profile.role == ADMIN and profile.status == ACTIVE
        result = h.auth.authenticate(ADMIN_EMAIL, GOOD_PASSWORD)
        assert result.ok and result.must_change_password is True

    def test_is_idempotent_across_restarts(self, h) -> None:
        encoded = hash_password(GOOD_PASSWORD, params=CHEAP)
        first = h.auth.ensure_bootstrap_admin(ADMIN_EMAIL, encoded)

        second = h.auth.ensure_bootstrap_admin(ADMIN_EMAIL, encoded)

        assert second.created is False
        assert second.credential_installed is False
        assert second.user_id == first.user_id
        assert h.store.count_users() == 1

    def test_never_reinstates_the_bootstrap_password_after_a_change(self, h) -> None:
        encoded = hash_password(GOOD_PASSWORD, params=CHEAP)
        outcome = h.auth.ensure_bootstrap_admin(ADMIN_EMAIL, encoded)
        h.auth.change_password(outcome.user_id, GOOD_PASSWORD, "the real admin passphrase")

        h.auth.ensure_bootstrap_admin(ADMIN_EMAIL, encoded)

        assert not h.auth.authenticate(ADMIN_EMAIL, GOOD_PASSWORD).ok
        assert h.auth.authenticate(ADMIN_EMAIL, "the real admin passphrase").ok

    def test_attaches_a_password_to_an_admin_created_by_another_identity_path(self, h) -> None:
        existing = h.make_user(ADMIN_EMAIL, role=ADMIN, password=None)
        encoded = hash_password(GOOD_PASSWORD, params=CHEAP)

        outcome = h.auth.ensure_bootstrap_admin(ADMIN_EMAIL, encoded)

        assert outcome.created is False
        assert outcome.credential_installed is True
        assert outcome.user_id == existing.user_id
        assert h.auth.authenticate(ADMIN_EMAIL, GOOD_PASSWORD).ok

    def test_refuses_to_add_an_admin_to_a_populated_deployment(self, h) -> None:
        h.make_user(MEMBER_EMAIL)
        encoded = hash_password(GOOD_PASSWORD, params=CHEAP)

        outcome = h.auth.ensure_bootstrap_admin(ADMIN_EMAIL, encoded)

        assert outcome.created is False
        assert outcome.credential_installed is False
        assert outcome.refused is True
        assert h.store.count_users() == 1

    def test_refuses_a_hash_that_is_not_one(self, h) -> None:
        for bogus in ["", "hunter2", None, GOOD_PASSWORD]:
            with pytest.raises(ValueError):
                h.auth.ensure_bootstrap_admin(ADMIN_EMAIL, bogus)

    def test_refuses_a_malformed_admin_email(self, h) -> None:
        encoded = hash_password(GOOD_PASSWORD, params=CHEAP)
        with pytest.raises(ValueError):
            h.auth.ensure_bootstrap_admin("not-an-email", encoded)


# --- audit -----------------------------------------------------------------------


class TestAudit:
    def _actions(self, h) -> list[str]:
        return [event.action for event in h.store.list_audit_events(limit=100)]

    def _details(self, h) -> str:
        return repr([event.view() for event in h.store.list_audit_events(limit=100)])

    def test_sign_in_success_and_failure_are_both_recorded(self, h) -> None:
        h.make_user(MEMBER_EMAIL)
        h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD)
        h.auth.authenticate(MEMBER_EMAIL, OTHER_PASSWORD)

        outcomes = {
            (e.action, e.outcome) for e in h.store.list_audit_events(limit=100)
        }
        assert ("auth.login", "ok") in outcomes
        assert ("auth.login", "invalid") in outcomes

    def test_a_lockout_is_recorded(self, h) -> None:
        h.make_user(MEMBER_EMAIL)
        for _ in range(SessionPolicy().max_failed_attempts):
            h.auth.authenticate(MEMBER_EMAIL, OTHER_PASSWORD)

        assert "auth.lockout" in self._actions(h)

    def test_password_changes_and_resets_are_recorded(self, h) -> None:
        admin = h.make_user(ADMIN_EMAIL, role=ADMIN)
        target = h.make_user(MEMBER_EMAIL)
        h.auth.change_password(target.user_id, GOOD_PASSWORD, "a brand new passphrase")
        h.auth.admin_reset_password(target.user_id, actor=Actor.for_user(admin))

        actions = self._actions(h)
        assert "auth.password.change" in actions
        assert "auth.password.reset" in actions

    def test_the_audit_trail_never_holds_a_password_email_or_token(self, h) -> None:
        h.make_user(MEMBER_EMAIL)
        result = h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD)
        h.auth.authenticate(MEMBER_EMAIL, OTHER_PASSWORD)
        h.auth.authenticate("stranger@example.com", GOOD_PASSWORD)

        blob = self._details(h)

        assert GOOD_PASSWORD not in blob
        assert OTHER_PASSWORD not in blob
        assert MEMBER_EMAIL not in blob
        assert "stranger" not in blob
        assert result.token not in blob

    def test_a_failed_sign_in_for_an_unknown_email_names_no_target(self, h) -> None:
        h.auth.authenticate("stranger@example.com", GOOD_PASSWORD)

        events = [e for e in h.store.list_audit_events(limit=10) if e.action == "auth.login"]
        assert events and all(e.target_id is None for e in events)


# --- privilege boundaries --------------------------------------------------------


def test_setting_a_password_outright_is_administrative_even_on_yourself(h) -> None:
    """A stolen session must not be enough to lock the real owner out.

    Changing your own password goes through ``change_password``, which demands
    the current one; ``set_password`` replaces it without proof and is
    therefore an administrator's operation only.
    """
    member = h.make_user(MEMBER_EMAIL)

    with pytest.raises(PermissionError):
        h.auth.set_password(member.user_id, "a brand new passphrase", actor=Actor.for_user(member))

    assert h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).ok


def test_concurrent_failures_cannot_slip_past_the_allowance(h, tmp_path) -> None:
    """Counting failures must be atomic, not read-then-write.

    Two guesses landing together used to be able to read the same count and
    write back the same incremented value, buying an extra attempt each round.
    """
    import threading

    h.make_user(MEMBER_EMAIL)
    policy = SessionPolicy()
    barrier = threading.Barrier(policy.max_failed_attempts)

    def guess() -> None:
        barrier.wait()
        h.auth.authenticate(MEMBER_EMAIL, OTHER_PASSWORD)

    threads = [threading.Thread(target=guess) for _ in range(policy.max_failed_attempts)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert h.auth.authenticate(MEMBER_EMAIL, GOOD_PASSWORD).reason == "locked"
