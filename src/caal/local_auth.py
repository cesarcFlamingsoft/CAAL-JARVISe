"""Standalone password sign-in: credentials, lockout, and server-side sessions.

This is the identity path that needs no Cloudflare Access, no identity
provider, and no outbound network: a person proves who they are with a
password, and the deployment issues an opaque session token that the Next.js
BFF puts in an HttpOnly cookie.

Everything an attacker probes first is decided here rather than in the routes
above it:

* **Anti-enumeration.** An unknown email, an account with no local password, a
  suspended account, and a wrong password all return the same
  ``invalid_credentials`` refusal, and the unknown-account path burns the same
  Argon2 work a real check would (:func:`caal.password_hash.dummy_verify`) so
  the three cannot be told apart by response time either.
* **Bounded guessing.** Consecutive failures lock the account for an
  exponentially growing window, capped, and per account rather than per
  process. A lockout is only *disclosed* to a caller who already supplied the
  right password, so probing cannot map which accounts exist.
* **Real sessions.** The database stores only the SHA-256 digest of a session
  token, so a database leak hands out no live sessions. Expiry, revocation,
  suspension and the forced-password-change flag are re-read from the database
  on every single request; nothing is trusted from inside the cookie.

Passwords never reach a log or the audit trail: audit rows carry opaque user
ids, an action and an outcome, and nothing else.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field

from .access_jwt import normalize_email
from .password_hash import (
    Argon2Params,
    PasswordPolicyError,
    dummy_verify,
    generate_password,
    hash_password,
    is_password_hash,
    validate_password,
    verify_password,
)
from .user_scope import is_valid_user_id
from .user_store import (
    ACTIVE,
    ADMIN,
    Actor,
    UnknownUserError,
    UserProfile,
    UserStore,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AccountLockedError",
    "BootstrapOutcome",
    "InvalidCredentialsError",
    "LocalAuth",
    "LocalAuthError",
    "LoginResult",
    "SessionInfo",
    "SessionPolicy",
    "TOKEN_BYTES",
]

TOKEN_BYTES = 32
MAX_TOKEN_LENGTH = 256
REASON_INVALID = "invalid_credentials"
REASON_LOCKED = "locked"


class LocalAuthError(Exception):
    """Base class; messages never carry a password, a token, or an email."""


class InvalidCredentialsError(LocalAuthError):
    """The supplied password did not match. Deliberately says nothing more."""


class AccountLockedError(LocalAuthError):
    """Too many consecutive failures; the account is temporarily locked."""

    def __init__(self, retry_after: int) -> None:
        super().__init__("Too many attempts")
        self.retry_after = int(retry_after)


@dataclass(frozen=True)
class SessionPolicy:
    """Session lifetimes and brute-force limits."""

    # Sliding: a session dies this long after its last use.
    idle_seconds: int = 8 * 3600
    # Hard ceiling: a session dies this long after sign-in however active it is,
    # so a stolen cookie cannot be renewed forever.
    absolute_seconds: int = 7 * 86400
    max_failed_attempts: int = 5
    base_lockout_seconds: int = 60
    max_lockout_seconds: int = 900
    max_sessions_per_user: int = 20

    def __post_init__(self) -> None:
        if not 60 <= self.idle_seconds <= 30 * 86400:
            raise ValueError("Session idle timeout is out of bounds")
        if not self.idle_seconds <= self.absolute_seconds <= 90 * 86400:
            raise ValueError("Session absolute lifetime is out of bounds")
        if not 1 <= self.max_failed_attempts <= 100:
            raise ValueError("Failed-attempt allowance is out of bounds")
        if not 1 <= self.base_lockout_seconds <= self.max_lockout_seconds <= 86400:
            raise ValueError("Lockout window is out of bounds")
        if not 1 <= self.max_sessions_per_user <= 1000:
            raise ValueError("Session-per-user cap is out of bounds")


@dataclass(frozen=True)
class LoginResult:
    """The outcome of a sign-in attempt. ``token`` never prints."""

    ok: bool
    reason: str | None = None
    token: str | None = field(default=None, repr=False)
    user: UserProfile | None = None
    must_change_password: bool = False
    expires_at: int | None = None
    retry_after: int | None = None


@dataclass(frozen=True)
class SessionInfo:
    """A live session, re-read from the database."""

    user: UserProfile
    must_change_password: bool
    issued_at: int
    expires_at: int


@dataclass(frozen=True)
class BootstrapOutcome:
    """What the one-time administrator seed actually did."""

    user_id: str | None
    created: bool
    credential_installed: bool
    refused: bool = False


def _digest(token: str) -> str:
    """Session tokens are high-entropy already, so a plain SHA-256 is enough."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class LocalAuth:
    """Password credentials and sessions over the shared CAAL SQLite store."""

    def __init__(
        self,
        store: UserStore,
        *,
        params: Argon2Params | None = None,
        policy: SessionPolicy | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store = store
        self._params = params
        self._policy = policy or SessionPolicy()
        self._clock = clock

    @property
    def policy(self) -> SessionPolicy:
        return self._policy

    @property
    def store(self) -> UserStore:
        return self._store

    def _now(self, now: int | None = None) -> int:
        return int(self._clock()) if now is None else int(now)

    def _connect(self) -> sqlite3.Connection:
        return self._store.connect()

    # --- credentials --------------------------------------------------------------

    @staticmethod
    def _require_admin(actor: object) -> None:
        """Setting a password outright is administrative.

        A user changing their own password goes through :meth:`change_password`,
        which demands the current one. Without that split, a stolen session
        would be enough to lock the real owner out of their own account.
        """
        if not isinstance(actor, Actor) or not actor.is_admin:
            raise PermissionError("Administrator role required")

    def has_password(self, user_id: object) -> bool:
        if not is_valid_user_id(user_id):
            return False
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT 1 FROM user_credentials WHERE user_id = ?", (user_id,)
            ).fetchone()
        return row is not None

    def set_password(
        self,
        user_id: object,
        password: object,
        *,
        actor: Actor,
        must_change: bool = False,
        now: int | None = None,
        _action: str = "auth.password.set",
    ) -> None:
        """Replace a user's password and revoke every session they hold."""
        if not is_valid_user_id(user_id):
            raise UnknownUserError("No such user")
        self._require_admin(actor)
        validate_password(password)
        if self._store.get_user(user_id) is None:
            raise UnknownUserError("No such user")
        encoded = hash_password(password, params=self._params)
        moment = self._now(now)
        self._install_credential(user_id, encoded, must_change=must_change, now=moment)  # type: ignore[arg-type]
        self._store.record_audit(
            _action,
            actor=actor,
            target_id=user_id,  # type: ignore[arg-type]
            detail={"must_change": bool(must_change)},
            now=moment,
        )

    def _install_credential(
        self, user_id: str, encoded: str, *, must_change: bool, now: int
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "INSERT INTO user_credentials "
                    "(user_id, password_hash, must_change, updated_at, failed_count, "
                    "locked_until, last_login_at) VALUES (?, ?, ?, ?, 0, NULL, NULL) "
                    "ON CONFLICT(user_id) DO UPDATE SET password_hash = excluded.password_hash, "
                    "must_change = excluded.must_change, updated_at = excluded.updated_at, "
                    "failed_count = 0, locked_until = NULL",
                    (user_id, encoded, 1 if must_change else 0, now),
                )
                # A new password invalidates everything issued under the old one.
                connection.execute(
                    "UPDATE auth_sessions SET revoked_at = ? "
                    "WHERE user_id = ? AND revoked_at IS NULL",
                    (now, user_id),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def change_password(
        self,
        user_id: object,
        current_password: object,
        new_password: object,
        *,
        keep_token: str | None = None,
        now: int | None = None,
    ) -> None:
        """Self-service change. Requires the current password; ends other sessions."""
        if not is_valid_user_id(user_id):
            raise UnknownUserError("No such user")
        moment = self._now(now)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM user_credentials WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            dummy_verify(current_password, params=self._params)
            raise InvalidCredentialsError("Current password is not correct")

        locked_until = row["locked_until"]
        if locked_until is not None and int(locked_until) > moment:
            raise AccountLockedError(int(locked_until) - moment)

        if not verify_password(current_password, row["password_hash"], params=self._params).ok:
            self._record_failure(user_id, now=moment)  # type: ignore[arg-type]
            raise InvalidCredentialsError("Current password is not correct")

        validate_password(new_password)
        if verify_password(new_password, row["password_hash"], params=self._params).ok:
            raise PasswordPolicyError("New password must be different from the current one")

        encoded = hash_password(new_password, params=self._params)
        keep = _digest(keep_token) if isinstance(keep_token, str) and keep_token else None
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "UPDATE user_credentials SET password_hash = ?, must_change = 0, "
                    "updated_at = ?, failed_count = 0, locked_until = NULL WHERE user_id = ?",
                    (encoded, moment, user_id),
                )
                if keep is None:
                    connection.execute(
                        "UPDATE auth_sessions SET revoked_at = ? "
                        "WHERE user_id = ? AND revoked_at IS NULL",
                        (moment, user_id),
                    )
                else:
                    connection.execute(
                        "UPDATE auth_sessions SET revoked_at = ? "
                        "WHERE user_id = ? AND revoked_at IS NULL AND session_id != ?",
                        (moment, user_id, keep),
                    )
                    connection.execute(
                        "UPDATE auth_sessions SET must_change = 0 WHERE session_id = ?", (keep,)
                    )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        profile = self._store.get_user(user_id)
        self._store.record_audit(
            "auth.password.change",
            actor=Actor.for_user(profile) if profile else Actor.system(),
            target_id=user_id,  # type: ignore[arg-type]
            detail={"sessions_kept": 1 if keep else 0},
            now=moment,
        )

    def admin_reset_password(self, user_id: object, *, actor: Actor, now: int | None = None) -> str:
        """Issue a random one-time password for a user. Administrators only.

        The result is returned to the calling administrator exactly once and is
        never stored, logged, or audited in the clear. The user must change it
        on their next sign-in.
        """
        if not isinstance(actor, Actor) or not actor.is_admin:
            raise PermissionError("Administrator role required")
        if not is_valid_user_id(user_id) or self._store.get_user(user_id) is None:
            raise UnknownUserError("No such user")
        issued = generate_password()
        self.set_password(
            user_id,
            issued,
            actor=actor,
            must_change=True,
            now=now,
            _action="auth.password.reset",
        )
        return issued

    # --- sign in ------------------------------------------------------------------

    def _lockout_for(self, failed_count: int) -> int:
        """Exponential backoff past the allowance, capped."""
        over = max(0, failed_count - self._policy.max_failed_attempts)
        window = self._policy.base_lockout_seconds * (2 ** min(over, 20))
        return min(int(window), self._policy.max_lockout_seconds)

    def _record_failure(self, user_id: str, *, now: int) -> int | None:
        """Count a failure and lock the account if it crossed the allowance.

        The increment happens inside one immediate transaction and reads the
        *new* value back, so two simultaneous guesses cannot both observe the
        old count and slip an extra attempt past the allowance.
        """
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "UPDATE user_credentials SET failed_count = failed_count + 1 "
                    "WHERE user_id = ?",
                    (user_id,),
                )
                row = connection.execute(
                    "SELECT failed_count FROM user_credentials WHERE user_id = ?", (user_id,)
                ).fetchone()
                if row is None:
                    connection.execute("COMMIT")
                    return None
                count = int(row["failed_count"])
                locked_until: int | None = None
                if count >= self._policy.max_failed_attempts:
                    locked_until = now + self._lockout_for(count)
                    connection.execute(
                        "UPDATE user_credentials SET locked_until = ? WHERE user_id = ?",
                        (locked_until, user_id),
                    )
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        if locked_until is not None:
            self._store.record_audit(
                "auth.lockout",
                actor=Actor.system(),
                target_id=user_id,
                detail={"until": locked_until},
                now=now,
            )
        return locked_until

    def authenticate(
        self, email: object, password: object, *, now: int | None = None
    ) -> LoginResult:
        """Check a password and, on success, issue a session token.

        Every refusal returns ``invalid_credentials`` with no detail, except a
        lockout reported to a caller who already proved they know the password.
        """
        moment = self._now(now)
        profile = self._store.get_user_by_email(email)
        if profile is None:
            # Unknown or malformed: pay the same cost a real check would.
            dummy_verify(password, params=self._params)
            self._audit_login(None, "invalid", moment)
            return LoginResult(ok=False, reason=REASON_INVALID)

        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM user_credentials WHERE user_id = ?", (profile.user_id,)
            ).fetchone()
        if row is None:
            dummy_verify(password, params=self._params)
            self._audit_login(profile.user_id, "no_password", moment)
            return LoginResult(ok=False, reason=REASON_INVALID)

        # Verify before consulting lock or status, so that "locked" and
        # "suspended" cost the same as "wrong password" and cannot be probed.
        verified = verify_password(password, row["password_hash"], params=self._params)

        locked_until = row["locked_until"]
        if locked_until is not None and int(locked_until) > moment:
            self._audit_login(profile.user_id, "locked", moment)
            if not verified.ok:
                return LoginResult(ok=False, reason=REASON_INVALID)
            return LoginResult(
                ok=False, reason=REASON_LOCKED, retry_after=int(locked_until) - moment
            )

        if not verified.ok:
            self._record_failure(profile.user_id, now=moment)
            self._audit_login(profile.user_id, "invalid", moment)
            return LoginResult(ok=False, reason=REASON_INVALID)

        if profile.status != ACTIVE:
            self._audit_login(profile.user_id, "suspended", moment)
            return LoginResult(ok=False, reason=REASON_INVALID)

        if verified.needs_rehash:
            # The cost policy moved on; upgrade this hash while we hold the
            # only plaintext we will ever see.
            try:
                with closing(self._connect()) as connection:
                    connection.execute(
                        "UPDATE user_credentials SET password_hash = ? WHERE user_id = ?",
                        (hash_password(password, params=self._params), profile.user_id),
                    )
            except Exception:
                logger.warning("Could not upgrade a stored password hash")

        must_change = bool(row["must_change"])
        token, expires_at = self._create_session(
            profile.user_id, must_change=must_change, now=moment
        )
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE user_credentials SET failed_count = 0, locked_until = NULL, "
                "last_login_at = ? WHERE user_id = ?",
                (moment, profile.user_id),
            )
        self._audit_login(profile.user_id, "ok", moment)
        return LoginResult(
            ok=True,
            token=token,
            user=profile,
            must_change_password=must_change,
            expires_at=expires_at,
        )

    def _audit_login(self, user_id: str | None, outcome: str, now: int) -> None:
        self._store.record_audit(
            "auth.login",
            actor=Actor.system(),
            target_id=user_id,
            detail={},
            outcome=outcome,
            now=now,
        )

    # --- sessions -----------------------------------------------------------------

    def _create_session(self, user_id: str, *, must_change: bool, now: int) -> tuple[str, int]:
        token = secrets.token_urlsafe(TOKEN_BYTES)
        absolute = now + self._policy.absolute_seconds
        idle = min(now + self._policy.idle_seconds, absolute)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                # Keep the table from growing without bound: drop what has
                # expired, and revoked rows once they are past any use.
                connection.execute(
                    "DELETE FROM auth_sessions WHERE absolute_expires_at <= ? "
                    "OR (revoked_at IS NOT NULL AND revoked_at <= ?)",
                    (now, now - self._policy.idle_seconds),
                )
                connection.execute(
                    "INSERT INTO auth_sessions (session_id, user_id, created_at, last_seen_at, "
                    "idle_expires_at, absolute_expires_at, revoked_at, must_change) "
                    "VALUES (?, ?, ?, ?, ?, ?, NULL, ?)",
                    (_digest(token), user_id, now, now, idle, absolute, 1 if must_change else 0),
                )
                # Cap concurrent sessions per user: the oldest fall off.
                connection.execute(
                    "DELETE FROM auth_sessions WHERE user_id = ? AND session_id NOT IN ("
                    "SELECT session_id FROM auth_sessions WHERE user_id = ? "
                    "ORDER BY created_at DESC, rowid DESC LIMIT ?)",
                    (user_id, user_id, self._policy.max_sessions_per_user),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return token, idle

    def verify_session(self, token: object, *, now: int | None = None) -> SessionInfo | None:
        """Resolve a session token to its live session, or ``None``.

        Expiry, revocation and the user's current status are read from the
        database on every call; nothing is trusted from the token itself.
        """
        if not isinstance(token, str) or not token or len(token) > MAX_TOKEN_LENGTH:
            return None
        moment = self._now(now)
        key = _digest(token)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM auth_sessions WHERE session_id = ?", (key,)
            ).fetchone()
        if row is None or row["revoked_at"] is not None:
            return None
        if int(row["idle_expires_at"]) <= moment or int(row["absolute_expires_at"]) <= moment:
            return None
        profile = self._store.get_user(row["user_id"])
        if profile is None or profile.status != ACTIVE:
            return None
        absolute = int(row["absolute_expires_at"])
        idle = min(moment + self._policy.idle_seconds, absolute)
        with closing(self._connect()) as connection:
            connection.execute(
                "UPDATE auth_sessions SET last_seen_at = ?, idle_expires_at = ? "
                "WHERE session_id = ? AND revoked_at IS NULL",
                (moment, idle, key),
            )
        return SessionInfo(
            user=profile,
            must_change_password=bool(row["must_change"]),
            issued_at=int(row["created_at"]),
            expires_at=idle,
        )

    def revoke_session(self, token: object, *, now: int | None = None) -> bool:
        """Revoke one session. Unknown tokens are a no-op, never an error."""
        if not isinstance(token, str) or not token or len(token) > MAX_TOKEN_LENGTH:
            return False
        moment = self._now(now)
        with closing(self._connect()) as connection:
            changed = connection.execute(
                "UPDATE auth_sessions SET revoked_at = ? "
                "WHERE session_id = ? AND revoked_at IS NULL",
                (moment, _digest(token)),
            )
        return changed.rowcount == 1

    def revoke_user_sessions(self, user_id: object, *, now: int | None = None) -> int:
        """Revoke every session a user holds, e.g. when they are suspended."""
        if not is_valid_user_id(user_id):
            return 0
        moment = self._now(now)
        with closing(self._connect()) as connection:
            changed = connection.execute(
                "UPDATE auth_sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL",
                (moment, user_id),
            )
        return int(changed.rowcount)

    # --- bootstrap ----------------------------------------------------------------

    def ensure_bootstrap_admin(
        self, email: object, password_hash: object, *, now: int | None = None
    ) -> BootstrapOutcome:
        """Seed the configured administrator from an already-hashed password.

        The operator supplies only a hash, so no plaintext password ever exists
        in the environment, the image, or the source tree. Idempotent: once the
        account has any credential of its own the hash is never re-applied, so
        restarting the deployment cannot silently reinstate the bootstrap
        password after the administrator has changed it.

        Deny-by-default: an account is only *created* when the store is empty.
        In a populated deployment a new administrator must be created through
        the admin panel, so a leaked environment variable cannot mint one.
        """
        normalized = normalize_email(email)
        if not is_password_hash(password_hash):
            raise ValueError("Bootstrap admin password hash is missing or malformed")
        moment = self._now(now)

        profile = self._store.get_user_by_email(normalized)
        if profile is not None:
            if self.has_password(profile.user_id):
                return BootstrapOutcome(profile.user_id, created=False, credential_installed=False)
            if not profile.is_admin:
                logger.warning(
                    "Bootstrap account already exists without administrator role; "
                    "installing its password but leaving the role unchanged"
                )
            self._install_credential(
                profile.user_id, str(password_hash), must_change=True, now=moment
            )
            self._store.record_audit(
                "auth.bootstrap",
                actor=Actor.system(),
                target_id=profile.user_id,
                detail={"created": False, "must_change": True},
                now=moment,
            )
            logger.warning("Installed the bootstrap password for the administrator account")
            return BootstrapOutcome(profile.user_id, created=False, credential_installed=True)

        if self._store.count_users() != 0:
            logger.error(
                "SECURITY: refusing to seed the bootstrap administrator into a deployment "
                "that already has users; create the account from the admin panel instead"
            )
            return BootstrapOutcome(None, created=False, credential_installed=False, refused=True)

        created = self._store.create_user(
            email=normalized,
            display_name=normalized.split("@", 1)[0],
            role=ADMIN,
            actor=Actor.system(),
            now=moment,
        )
        self._install_credential(created.user_id, str(password_hash), must_change=True, now=moment)
        self._store.record_audit(
            "auth.bootstrap",
            actor=Actor.system(),
            target_id=created.user_id,
            detail={"created": True, "must_change": True},
            now=moment,
        )
        logger.warning(
            "Seeded the first administrator account; its one-time password must be "
            "changed at first sign-in"
        )
        return BootstrapOutcome(created.user_id, created=True, credential_installed=True)
