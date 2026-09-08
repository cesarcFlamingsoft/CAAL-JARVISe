"""User profiles, authorization data, and the audit trail for multi-user JARVIS.

One SQLite store (the shared ``assistant.sqlite3`` on ``CAAL_DATA_DIR``)
holds every user JARVIS may act for:

* an opaque id (``usr_<24 hex>``) that is the only identifier allowed to
  travel through sessions, devices, tasks, and phone callbacks;
* the normalized, verified email the Cloudflare Access assertion resolved;
* a display name, a role (``admin``/``member``) and a status
  (``active``/``suspended``);
* the single approved E.164 callback number, encrypted at rest with
  :mod:`caal.profile_crypto` and bound to the row, plus a keyed blind index so
  caller-id can be matched without plaintext in the database.

Resolution is deny-by-default: an unknown verified email is refused, except
for the one-time bootstrap of the configured administrator email into an
otherwise empty store. Every administrative mutation and every sensitive
profile change is written to a bounded audit table that records opaque ids,
actions, and field names only, never emails, phone numbers, or free text.

Nothing in this module logs profile contents.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .access_jwt import normalize_email
from .profile_crypto import DecryptionError, KeyRing
from .user_scope import is_valid_user_id

logger = logging.getLogger(__name__)

__all__ = [
    "ACTIVE",
    "ADMIN",
    "MEMBER",
    "SUSPENDED",
    "Actor",
    "AuditEvent",
    "CallbackNumberError",
    "DuplicateUserError",
    "LastAdminError",
    "NotConfiguredError",
    "UnknownUserError",
    "UserProfile",
    "UserStore",
    "UserStoreError",
    "UserSuspendedError",
    "is_valid_user_id",
    "new_user_id",
    "normalize_e164",
]

ADMIN = "admin"
MEMBER = "member"
ROLES = frozenset({ADMIN, MEMBER})
ACTIVE = "active"
SUSPENDED = "suspended"
STATUSES = frozenset({ACTIVE, SUSPENDED})

MAX_DISPLAY_NAME_LENGTH = 80
MAX_USERS_LIST = 500
MAX_AUDIT_LIST = 500
MAX_AUDIT_ROWS = 10_000
MAX_AUDIT_DETAIL_CHARS = 512
MAX_AUDIT_ACTION_LENGTH = 64
_AUDIT_ACTION = re.compile(r"^[a-z][a-z0-9_.]*$")
_BUSY_TIMEOUT_SECONDS = 5.0
_E164 = re.compile(r"^\+[1-9]\d{7,14}$")
_CALLBACK_PURPOSE = "callback-number"
SYSTEM_ACTOR_ID = "system"


class UserStoreError(Exception):
    """Base class; messages never carry profile contents."""


class NotConfiguredError(UserStoreError):
    """A required piece of multi-user configuration is absent."""


class UnknownUserError(UserStoreError):
    """No active or suspended user matches; the caller is told nothing more."""


class UserSuspendedError(UserStoreError):
    """The user exists but may not act."""


class DuplicateUserError(UserStoreError):
    """The email already belongs to a user."""


class LastAdminError(UserStoreError):
    """The change would leave no active administrator."""


class CallbackNumberError(UserStoreError, ValueError):
    """The number is not a usable E.164 destination or is already approved elsewhere."""


def new_user_id() -> str:
    return "usr_" + secrets.token_hex(12)


def normalize_e164(raw: object) -> str:
    """Strict E.164 with common punctuation removed; never guesses a country code."""
    if not isinstance(raw, str):
        raise CallbackNumberError("Callback number must be text")
    compact = re.sub(r"[\s().-]", "", raw.strip())
    if _E164.fullmatch(compact) is None:
        raise CallbackNumberError("Callback number must be in E.164 format, e.g. +17805551234")
    return compact


def _normalize_display_name(raw: object) -> str:
    if not isinstance(raw, str) or not raw.isprintable():
        raise ValueError("Display name must be plain text without control characters")
    name = " ".join(raw.split())
    if not name or len(name) > MAX_DISPLAY_NAME_LENGTH:
        raise ValueError(f"Display name must be 1 to {MAX_DISPLAY_NAME_LENGTH} characters")
    return name


def _now(now: int | None) -> int:
    return int(time.time()) if now is None else int(now)


@dataclass(frozen=True)
class Actor:
    """Who is performing a change. ``user_id`` is opaque; ``system`` has none."""

    user_id: str | None
    role: str

    @classmethod
    def system(cls) -> Actor:
        return cls(user_id=None, role=ADMIN)

    @classmethod
    def for_user(cls, profile: UserProfile) -> Actor:
        return cls(user_id=profile.user_id, role=profile.role)

    @property
    def is_admin(self) -> bool:
        return self.role == ADMIN

    @property
    def audit_id(self) -> str:
        return self.user_id or SYSTEM_ACTOR_ID


@dataclass(frozen=True)
class UserProfile:
    """One user. The email never prints; the callback number is never held here."""

    user_id: str
    email: str = field(repr=False)
    display_name: str = field(repr=False)
    role: str = MEMBER
    status: str = ACTIVE
    has_callback_number: bool = False
    callback_number_updated_at: int | None = None
    created_at: int = 0
    updated_at: int = 0
    created_by: str | None = None
    last_seen_at: int | None = None

    @property
    def is_active(self) -> bool:
        return self.status == ACTIVE

    @property
    def is_admin(self) -> bool:
        return self.role == ADMIN

    def public_view(self) -> dict[str, Any]:
        """What a user may see about themselves."""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "display_name": self.display_name,
            "role": self.role,
            "status": self.status,
            "has_callback_number": self.has_callback_number,
            "callback_number_updated_at": self.callback_number_updated_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_seen_at": self.last_seen_at,
        }

    def admin_view(self) -> dict[str, Any]:
        """What an administrator may see. Still never the callback number."""
        return {**self.public_view(), "created_by": self.created_by}


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    occurred_at: int
    actor_id: str
    actor_role: str
    action: str
    target_id: str | None
    outcome: str
    detail: dict[str, Any]

    def view(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "occurred_at": self.occurred_at,
            "actor_id": self.actor_id,
            "actor_role": self.actor_role,
            "action": self.action,
            "target_id": self.target_id,
            "outcome": self.outcome,
            "detail": dict(self.detail),
        }


# --- schema --------------------------------------------------------------------

_MIGRATIONS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (
        1,
        (
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('admin', 'member')),
                status TEXT NOT NULL CHECK (status IN ('active', 'suspended')),
                callback_number_enc TEXT,
                callback_number_index TEXT,
                callback_number_updated_at INTEGER,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                created_by TEXT,
                last_seen_at INTEGER
            )
            """,
            "CREATE INDEX IF NOT EXISTS users_callback_index ON users (callback_number_index)",
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                occurred_at INTEGER NOT NULL,
                actor_id TEXT NOT NULL,
                actor_role TEXT NOT NULL,
                action TEXT NOT NULL,
                target_id TEXT,
                outcome TEXT NOT NULL,
                detail TEXT NOT NULL DEFAULT '{}'
            )
            """,
            "CREATE INDEX IF NOT EXISTS audit_events_time ON audit_events (occurred_at)",
            "CREATE TABLE IF NOT EXISTS auth_nonces "
            "(jti TEXT PRIMARY KEY, expires_at INTEGER NOT NULL)",
        ),
    ),
    (
        2,
        (
            # Standalone password sign-in. Credentials live in their own table
            # rather than on `users`, so an identity provider (Cloudflare
            # Access) and a local password can coexist for the same account and
            # a user simply has no row here when they have no local password.
            """
            CREATE TABLE IF NOT EXISTS user_credentials (
                user_id TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                must_change INTEGER NOT NULL DEFAULT 0,
                updated_at INTEGER NOT NULL,
                failed_count INTEGER NOT NULL DEFAULT 0,
                locked_until INTEGER,
                last_login_at INTEGER
            )
            """,
            # Server-side sessions: the browser holds an opaque bearer token,
            # this table holds only its SHA-256 digest, so a database leak does
            # not hand out live sessions. Revocation and expiry are decided
            # here on every request, never from a claim inside a cookie.
            """
            CREATE TABLE IF NOT EXISTS auth_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                last_seen_at INTEGER NOT NULL,
                idle_expires_at INTEGER NOT NULL,
                absolute_expires_at INTEGER NOT NULL,
                revoked_at INTEGER,
                must_change INTEGER NOT NULL DEFAULT 0
            )
            """,
            "CREATE INDEX IF NOT EXISTS auth_sessions_user ON auth_sessions (user_id)",
            "CREATE INDEX IF NOT EXISTS auth_sessions_expiry "
            "ON auth_sessions (absolute_expires_at)",
        ),
    ),
    (
        3,
        (
            # Per-user provider connections (Google, Microsoft, Zoho) linked by
            # OAuth authorization, never by a password. The tokens are encrypted
            # with the profile key ring and bound to their row; the columns are
            # owned by caal.provider_connections.
            """
            CREATE TABLE IF NOT EXISTS provider_connections (
                connection_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                provider TEXT NOT NULL CHECK (provider IN ('google', 'microsoft', 'zoho')),
                status TEXT NOT NULL CHECK (status IN ('connected', 'revoked')),
                account_label TEXT,
                scopes TEXT NOT NULL DEFAULT '[]',
                access_token_enc TEXT,
                refresh_token_enc TEXT,
                token_expires_at INTEGER,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                connected_at INTEGER,
                revoked_at INTEGER,
                UNIQUE (user_id, provider)
            )
            """,
            "CREATE INDEX IF NOT EXISTS provider_connections_user "
            "ON provider_connections (user_id)",
            # Pending OAuth authorizations: who started one, for which provider,
            # until when, and whether it has been redeemed. The state handed to
            # the provider is signed over these columns and is never stored.
            """
            CREATE TABLE IF NOT EXISTS oauth_states (
                state_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                code_verifier_enc TEXT,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL,
                consumed_at INTEGER
            )
            """,
            "CREATE INDEX IF NOT EXISTS oauth_states_user ON oauth_states (user_id)",
            "CREATE INDEX IF NOT EXISTS oauth_states_expiry ON oauth_states (expires_at)",
        ),
    ),
    (
        4,
        (
            # Several accounts from one provider per user. The provider's own
            # stable account id (Google `sub`, Microsoft `oid`, Zoho `ZUID`)
            # joins the uniqueness key, so a second Google account gets its own
            # row and reconnecting one account touches only that row. SQLite
            # cannot relax a UNIQUE constraint in place, so the table is
            # rebuilt; every version-3 row is carried over with a NULL id and
            # stays readable (NULLs are distinct under UNIQUE).
            """
            CREATE TABLE IF NOT EXISTS provider_connections_v4 (
                connection_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                provider TEXT NOT NULL CHECK (provider IN ('google', 'microsoft', 'zoho')),
                status TEXT NOT NULL CHECK (status IN ('connected', 'revoked')),
                provider_account_id TEXT,
                account_label TEXT,
                scopes TEXT NOT NULL DEFAULT '[]',
                access_token_enc TEXT,
                refresh_token_enc TEXT,
                token_expires_at INTEGER,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                connected_at INTEGER,
                revoked_at INTEGER,
                UNIQUE (user_id, provider, provider_account_id)
            )
            """,
            """
            INSERT INTO provider_connections_v4 (connection_id, user_id, provider, status,
                provider_account_id, account_label, scopes, access_token_enc,
                refresh_token_enc, token_expires_at, created_at, updated_at, connected_at,
                revoked_at)
            SELECT connection_id, user_id, provider, status, NULL, account_label, scopes,
                access_token_enc, refresh_token_enc, token_expires_at, created_at,
                updated_at, connected_at, revoked_at
            FROM provider_connections ORDER BY rowid
            """,
            "DROP TABLE provider_connections",
            "ALTER TABLE provider_connections_v4 RENAME TO provider_connections",
            "CREATE INDEX IF NOT EXISTS provider_connections_user "
            "ON provider_connections (user_id)",
        ),
    ),
    (
        5,
        (
            # Per-user weather: where a user's forecast is read from, and what
            # the upstream last answered for that place. The columns are owned
            # by caal.weather_store. A hand-picked city is durable; a browser
            # position carries its own expiry and is erased the first time it
            # is read past it. Both are stored rounded to about a kilometre --
            # a browser's precise coordinates are never written down.
            """
            CREATE TABLE IF NOT EXISTS weather_preferences (
                user_id TEXT PRIMARY KEY,
                city_name TEXT,
                city_latitude REAL,
                city_longitude REAL,
                city_timezone TEXT,
                city_country TEXT,
                city_admin1 TEXT,
                city_updated_at INTEGER,
                browser_latitude REAL,
                browser_longitude REAL,
                browser_updated_at INTEGER,
                browser_expires_at INTEGER,
                updated_at INTEGER NOT NULL
            )
            """,
            # One row per (user, resolved place): the cap that keeps JARVIS to
            # a single upstream forecast call per place per rolling hour.
            """
            CREATE TABLE IF NOT EXISTS weather_forecasts (
                cache_key TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                fetched_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS weather_forecasts_user ON weather_forecasts (user_id)",
            # City lookups are public reference data about place names: keyed
            # by the query alone, holding nothing about who asked.
            """
            CREATE TABLE IF NOT EXISTS weather_geocodes (
                query TEXT PRIMARY KEY,
                results TEXT NOT NULL,
                fetched_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL
            )
            """,
        ),
    ),
)


class UserStore:
    """SQLite-backed users, callback numbers, and audit trail."""

    def __init__(self, path: Path | str, *, keyring: KeyRing | None) -> None:
        self._path = Path(path)
        self._keyring = keyring
        self.migrate()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def encryption_available(self) -> bool:
        return self._keyring is not None

    # --- connections and schema ----------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self._path, timeout=_BUSY_TIMEOUT_SECONDS, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {int(_BUSY_TIMEOUT_SECONDS * 1000)}")
        return connection

    def connect(self) -> sqlite3.Connection:
        """A configured connection to the same file, for sibling modules.

        :mod:`caal.local_auth` owns the credential and session tables, which
        share this database and this migration ledger.
        """
        return self._connect()

    def migrate(self) -> int:
        """Apply pending schema versions in order; return the current version."""
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS schema_migrations "
                    "(version INTEGER PRIMARY KEY, applied_at INTEGER NOT NULL)"
                )
                current = self._schema_version(connection)
                for version, statements in _MIGRATIONS:
                    if version <= current:
                        continue
                    for statement in statements:
                        connection.execute(statement)
                    connection.execute(
                        "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                        (version, int(time.time())),
                    )
                    current = version
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return current

    @staticmethod
    def _schema_version(connection: sqlite3.Connection) -> int:
        row = connection.execute("SELECT MAX(version) AS v FROM schema_migrations").fetchone()
        return int(row["v"] or 0)

    def schema_version(self) -> int:
        with closing(self._connect()) as connection:
            return self._schema_version(connection)

    # --- rows -------------------------------------------------------------------

    @staticmethod
    def _profile(row: sqlite3.Row) -> UserProfile:
        return UserProfile(
            user_id=row["user_id"],
            email=row["email"],
            display_name=row["display_name"],
            role=row["role"],
            status=row["status"],
            has_callback_number=row["callback_number_enc"] is not None,
            callback_number_updated_at=row["callback_number_updated_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            created_by=row["created_by"],
            last_seen_at=row["last_seen_at"],
        )

    @staticmethod
    def _fetch(connection: sqlite3.Connection, user_id: str) -> sqlite3.Row | None:
        return connection.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()

    def _require_row(self, connection: sqlite3.Connection, user_id: object) -> sqlite3.Row:
        if not is_valid_user_id(user_id):
            raise UnknownUserError("No such user")
        row = self._fetch(connection, user_id)  # type: ignore[arg-type]
        if row is None:
            raise UnknownUserError("No such user")
        return row

    @staticmethod
    def _require_admin(actor: Actor) -> None:
        if not isinstance(actor, Actor) or not actor.is_admin:
            raise PermissionError("Administrator role required")

    def _write_audit(
        self,
        connection: sqlite3.Connection,
        *,
        actor: Actor,
        action: str,
        target_id: str | None,
        detail: dict[str, Any] | None,
        outcome: str = "ok",
        now: int,
    ) -> None:
        rendered = json.dumps(detail or {}, sort_keys=True, separators=(",", ":"))
        if len(rendered) > MAX_AUDIT_DETAIL_CHARS:
            rendered = json.dumps({"truncated": True})
        connection.execute(
            "INSERT INTO audit_events "
            "(event_id, occurred_at, actor_id, actor_role, action, target_id, outcome, detail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "evt_" + secrets.token_hex(12),
                now,
                actor.audit_id,
                actor.role,
                action,
                target_id,
                outcome,
                rendered,
            ),
        )
        # Keep the table bounded: drop the oldest rows past the cap.
        connection.execute(
            "DELETE FROM audit_events WHERE rowid IN ("
            "SELECT rowid FROM audit_events ORDER BY occurred_at DESC, rowid DESC "
            "LIMIT -1 OFFSET ?)",
            (MAX_AUDIT_ROWS,),
        )

    # --- identity resolution ----------------------------------------------------

    def resolve_identity(
        self,
        email: object,
        *,
        bootstrap_admin_email: object,
        now: int | None = None,
        on_bootstrap: Callable[[UserProfile], None] | None = None,
    ) -> UserProfile:
        """Map a *verified* email to its active user, bootstrapping the first admin.

        Raises ``ValueError`` for a malformed email, :class:`NotConfiguredError`
        when no bootstrap email is configured, :class:`UnknownUserError` for an
        email that is not a user (deny by default), and
        :class:`UserSuspendedError` for a suspended one. ``on_bootstrap`` runs
        exactly once, after the first administrator has been committed; its
        failure is logged and never undoes the bootstrap.
        """
        try:
            bootstrap = normalize_email(bootstrap_admin_email)
        except ValueError as exc:
            raise NotConfiguredError("CAAL_BOOTSTRAP_ADMIN_EMAIL is not configured") from exc
        normalized = normalize_email(email)
        moment = _now(now)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM users WHERE email = ?", (normalized,)
                ).fetchone()
                if row is None:
                    (count,) = connection.execute("SELECT COUNT(*) FROM users").fetchone()
                    if count != 0 or normalized != bootstrap:
                        connection.execute("COMMIT")
                        raise UnknownUserError("No account for this identity")
                    user_id = new_user_id()
                    connection.execute(
                        "INSERT INTO users (user_id, email, display_name, role, status, "
                        "created_at, updated_at, created_by, last_seen_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            user_id,
                            normalized,
                            _normalize_display_name(normalized.split("@", 1)[0]),
                            ADMIN,
                            ACTIVE,
                            moment,
                            moment,
                            None,
                            moment,
                        ),
                    )
                    self._write_audit(
                        connection,
                        actor=Actor.system(),
                        action="user.bootstrap",
                        target_id=user_id,
                        detail={"role": ADMIN},
                        now=moment,
                    )
                    row = self._fetch(connection, user_id)
                    connection.execute("COMMIT")
                    logger.warning("Bootstrapped the first administrator account")
                    profile = self._profile(row)
                    if on_bootstrap is not None:
                        try:
                            on_bootstrap(profile)
                        except Exception:
                            logger.warning("Bootstrap hook failed", exc_info=False)
                    return profile
                if row["status"] != ACTIVE:
                    connection.execute("COMMIT")
                    raise UserSuspendedError("Account is suspended")
                connection.execute(
                    "UPDATE users SET last_seen_at = ? WHERE user_id = ?",
                    (moment, row["user_id"]),
                )
                row = self._fetch(connection, row["user_id"])
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        return self._profile(row)

    def record_audit(
        self,
        action: str,
        *,
        actor: Actor,
        target_id: str | None = None,
        detail: dict[str, Any] | None = None,
        outcome: str = "ok",
        now: int | None = None,
    ) -> None:
        """Record a system-level event (never with emails, numbers, or free text)."""
        if (
            not isinstance(action, str)
            or not action
            or len(action) > MAX_AUDIT_ACTION_LENGTH
            or _AUDIT_ACTION.fullmatch(action) is None
        ):
            raise ValueError("Audit action must be a short dotted lower-case identifier")
        if target_id is not None and not is_valid_user_id(target_id):
            raise ValueError("Audit target must be an opaque user id")
        moment = _now(now)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._write_audit(
                    connection,
                    actor=actor,
                    action=action,
                    target_id=target_id,
                    detail=detail,
                    outcome=outcome,
                    now=moment,
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    # --- reads --------------------------------------------------------------------

    def get_user(self, user_id: object) -> UserProfile | None:
        if not is_valid_user_id(user_id):
            return None
        with closing(self._connect()) as connection:
            row = self._fetch(connection, user_id)  # type: ignore[arg-type]
        return self._profile(row) if row else None

    def get_user_by_email(self, email: object) -> UserProfile | None:
        """The user with this normalized email, or None. Never raises for bad input."""
        try:
            normalized = normalize_email(email)
        except ValueError:
            return None
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE email = ?", (normalized,)
            ).fetchone()
        return self._profile(row) if row else None

    def list_users(self, *, limit: int = MAX_USERS_LIST) -> list[UserProfile]:
        bounded = max(1, min(int(limit), MAX_USERS_LIST))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM users ORDER BY created_at, rowid LIMIT ?", (bounded,)
            ).fetchall()
        return [self._profile(row) for row in rows]

    def count_users(self) -> int:
        with closing(self._connect()) as connection:
            (count,) = connection.execute("SELECT COUNT(*) FROM users").fetchone()
        return int(count)

    def list_audit_events(
        self, *, limit: int = 100, oldest_first: bool = False
    ) -> list[AuditEvent]:
        bounded = max(1, min(int(limit), MAX_AUDIT_LIST))
        order = "ASC" if oldest_first else "DESC"
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT * FROM audit_events ORDER BY occurred_at {order}, rowid {order} LIMIT ?",
                (bounded,),
            ).fetchall()
        events = []
        for row in rows:
            try:
                detail = json.loads(row["detail"])
            except ValueError:
                detail = {}
            events.append(
                AuditEvent(
                    event_id=row["event_id"],
                    occurred_at=row["occurred_at"],
                    actor_id=row["actor_id"],
                    actor_role=row["actor_role"],
                    action=row["action"],
                    target_id=row["target_id"],
                    outcome=row["outcome"],
                    detail=detail if isinstance(detail, dict) else {},
                )
            )
        return events

    # --- administration -----------------------------------------------------------

    def create_user(
        self,
        *,
        email: object,
        display_name: object,
        role: str,
        actor: Actor,
        now: int | None = None,
    ) -> UserProfile:
        self._require_admin(actor)
        normalized = normalize_email(email)
        name = _normalize_display_name(display_name)
        if role not in ROLES:
            raise ValueError("Role must be admin or member")
        moment = _now(now)
        user_id = new_user_id()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                try:
                    connection.execute(
                        "INSERT INTO users (user_id, email, display_name, role, status, "
                        "created_at, updated_at, created_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (user_id, normalized, name, role, ACTIVE, moment, moment, actor.user_id),
                    )
                except sqlite3.IntegrityError as exc:
                    raise DuplicateUserError("A user with this email already exists") from exc
                self._write_audit(
                    connection,
                    actor=actor,
                    action="user.create",
                    target_id=user_id,
                    detail={"role": role},
                    now=moment,
                )
                row = self._fetch(connection, user_id)
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        return self._profile(row)

    def update_display_name(
        self, user_id: object, display_name: object, *, actor: Actor, now: int | None = None
    ) -> UserProfile:
        """Self-service rename; administrators may rename anyone."""
        if not isinstance(actor, Actor) or (not actor.is_admin and actor.user_id != user_id):
            raise PermissionError("You may only edit your own profile")
        name = _normalize_display_name(display_name)
        moment = _now(now)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._require_row(connection, user_id)
                if row["display_name"] != name:
                    connection.execute(
                        "UPDATE users SET display_name = ?, updated_at = ? WHERE user_id = ?",
                        (name, moment, row["user_id"]),
                    )
                    self._write_audit(
                        connection,
                        actor=actor,
                        action="user.update",
                        target_id=row["user_id"],
                        detail={"fields": ["display_name"]},
                        now=moment,
                    )
                row = self._fetch(connection, row["user_id"])
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        return self._profile(row)

    def admin_update(
        self,
        user_id: object,
        *,
        display_name: object | None = None,
        role: str | None = None,
        status: str | None = None,
        actor: Actor,
        now: int | None = None,
    ) -> UserProfile:
        self._require_admin(actor)
        name = _normalize_display_name(display_name) if display_name is not None else None
        if role is not None and role not in ROLES:
            raise ValueError("Role must be admin or member")
        if status is not None and status not in STATUSES:
            raise ValueError("Status must be active or suspended")
        moment = _now(now)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._require_row(connection, user_id)
                target = row["user_id"]
                losing_admin = (
                    row["role"] == ADMIN
                    and row["status"] == ACTIVE
                    and (
                        (role is not None and role != ADMIN)
                        or (status is not None and status != ACTIVE)
                    )
                )
                if losing_admin:
                    (others,) = connection.execute(
                        "SELECT COUNT(*) FROM users WHERE role = ? AND status = ? AND user_id != ?",
                        (ADMIN, ACTIVE, target),
                    ).fetchone()
                    if others == 0:
                        raise LastAdminError("At least one active administrator must remain")
                if name is not None and name != row["display_name"]:
                    connection.execute(
                        "UPDATE users SET display_name = ?, updated_at = ? WHERE user_id = ?",
                        (name, moment, target),
                    )
                    self._write_audit(
                        connection,
                        actor=actor,
                        action="user.update",
                        target_id=target,
                        detail={"fields": ["display_name"]},
                        now=moment,
                    )
                if role is not None and role != row["role"]:
                    connection.execute(
                        "UPDATE users SET role = ?, updated_at = ? WHERE user_id = ?",
                        (role, moment, target),
                    )
                    self._write_audit(
                        connection,
                        actor=actor,
                        action="user.role",
                        target_id=target,
                        detail={"role": role},
                        now=moment,
                    )
                if status is not None and status != row["status"]:
                    connection.execute(
                        "UPDATE users SET status = ?, updated_at = ? WHERE user_id = ?",
                        (status, moment, target),
                    )
                    self._write_audit(
                        connection,
                        actor=actor,
                        action="user.status",
                        target_id=target,
                        detail={"status": status},
                        now=moment,
                    )
                row = self._fetch(connection, target)
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        return self._profile(row)

    # --- callback numbers ---------------------------------------------------------

    def _require_keyring(self) -> KeyRing:
        if self._keyring is None:
            raise NotConfiguredError("CAAL_PROFILE_ENCRYPTION_KEYS is not configured")
        return self._keyring

    @staticmethod
    def _aad(user_id: str) -> str:
        return f"caal.users.callback_number:{user_id}"

    def set_callback_number(
        self, user_id: object, number: object, *, actor: Actor, now: int | None = None
    ) -> UserProfile:
        """Approve exactly one E.164 number for a user. Administrators only."""
        self._require_admin(actor)
        ring = self._require_keyring()
        moment = _now(now)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._require_row(connection, user_id)
                target = row["user_id"]
                normalized = normalize_e164(number)
                candidates = ring.blind_index_candidates(normalized, purpose=_CALLBACK_PURPOSE)
                placeholders = ", ".join("?" for _ in candidates)
                clash = connection.execute(
                    f"SELECT 1 FROM users WHERE callback_number_index IN ({placeholders}) "
                    "AND user_id != ? LIMIT 1",
                    (*candidates, target),
                ).fetchone()
                if clash is not None:
                    raise CallbackNumberError("That number is already approved for another user")
                connection.execute(
                    "UPDATE users SET callback_number_enc = ?, callback_number_index = ?, "
                    "callback_number_updated_at = ?, updated_at = ? WHERE user_id = ?",
                    (
                        ring.encrypt(normalized, aad=self._aad(target)),
                        ring.blind_index(normalized, purpose=_CALLBACK_PURPOSE),
                        moment,
                        moment,
                        target,
                    ),
                )
                self._write_audit(
                    connection,
                    actor=actor,
                    action="user.callback.set",
                    target_id=target,
                    detail={"fields": ["callback_number"]},
                    now=moment,
                )
                row = self._fetch(connection, target)
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        return self._profile(row)

    def clear_callback_number(
        self, user_id: object, *, actor: Actor, now: int | None = None
    ) -> UserProfile:
        self._require_admin(actor)
        moment = _now(now)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._require_row(connection, user_id)
                target = row["user_id"]
                if row["callback_number_enc"] is not None:
                    connection.execute(
                        "UPDATE users SET callback_number_enc = NULL, "
                        "callback_number_index = NULL, callback_number_updated_at = ?, "
                        "updated_at = ? WHERE user_id = ?",
                        (moment, moment, target),
                    )
                    self._write_audit(
                        connection,
                        actor=actor,
                        action="user.callback.clear",
                        target_id=target,
                        detail={"fields": ["callback_number"]},
                        now=moment,
                    )
                row = self._fetch(connection, target)
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        return self._profile(row)

    def _decrypt_number(self, row: sqlite3.Row) -> str | None:
        """The row's approved number if it decrypts and still validates; else None."""
        if self._keyring is None or row["callback_number_enc"] is None:
            return None
        try:
            plaintext = self._keyring.decrypt(
                row["callback_number_enc"], aad=self._aad(row["user_id"])
            )
            return normalize_e164(plaintext)
        except (DecryptionError, CallbackNumberError):
            logger.warning("Stored callback number could not be decrypted or validated")
            return None

    def approved_callback_number(self, user_id: object) -> str | None:
        """The one number JARVIS may dial for an *active* user, revalidated on every read.

        The caller must never log, speak, or render the result.
        """
        if self._keyring is None or not is_valid_user_id(user_id):
            return None
        with closing(self._connect()) as connection:
            row = self._fetch(connection, user_id)  # type: ignore[arg-type]
        if row is None or row["status"] != ACTIVE:
            return None
        return self._decrypt_number(row)

    def find_user_by_callback_number(self, number: object) -> UserProfile | None:
        """Resolve caller-id to the active user whose approved number it is."""
        if self._keyring is None:
            return None
        try:
            normalized = normalize_e164(number)
        except CallbackNumberError:
            return None
        candidates = self._keyring.blind_index_candidates(normalized, purpose=_CALLBACK_PURPOSE)
        placeholders = ", ".join("?" for _ in candidates)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT * FROM users WHERE callback_number_index IN ({placeholders}) "
                "AND status = ?",
                (*candidates, ACTIVE),
            ).fetchall()
        for row in rows:
            # The index is keyed, but confirm against the authenticated ciphertext
            # anyway so a collision can never map a caller to the wrong user.
            if self._decrypt_number(row) == normalized:
                return self._profile(row)
        return None

    def rotate_encryption(self, *, now: int | None = None) -> int:
        """Re-encrypt every number written under a retired key; return how many."""
        ring = self._require_keyring()
        moment = _now(now)
        rotated = 0
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                rows = connection.execute(
                    "SELECT * FROM users WHERE callback_number_enc IS NOT NULL"
                ).fetchall()
                for row in rows:
                    if not ring.needs_rotation(row["callback_number_enc"]):
                        continue
                    number = self._decrypt_number(row)
                    if number is None:
                        continue
                    connection.execute(
                        "UPDATE users SET callback_number_enc = ?, callback_number_index = ?, "
                        "updated_at = ? WHERE user_id = ?",
                        (
                            ring.encrypt(number, aad=self._aad(row["user_id"])),
                            ring.blind_index(number, purpose=_CALLBACK_PURPOSE),
                            moment,
                            row["user_id"],
                        ),
                    )
                    rotated += 1
                if rotated:
                    self._write_audit(
                        connection,
                        actor=Actor.system(),
                        action="user.callback.rotate",
                        target_id=None,
                        detail={"count": rotated},
                        now=moment,
                    )
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        return rotated

    # --- nonces -------------------------------------------------------------------

    def consume_nonce(self, jti: str, *, expires_at: int, now: int | None = None) -> bool:
        """Single-use registry for internal principals (see :mod:`caal.internal_auth`)."""
        if not isinstance(jti, str) or not jti or len(jti) > 256:
            raise ValueError("jti must be non-empty text")
        moment = _now(now)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("DELETE FROM auth_nonces WHERE expires_at <= ?", (moment,))
                inserted = connection.execute(
                    "INSERT OR IGNORE INTO auth_nonces (jti, expires_at) VALUES (?, ?)",
                    (jti, int(expires_at)),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return inserted.rowcount == 1
