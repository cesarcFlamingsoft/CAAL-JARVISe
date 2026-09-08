"""Durable, per-user provider connections and the OAuth state that creates them.

Two tables in the shared ``assistant.sqlite3`` (schema version 4 in
:mod:`caal.user_store`), owned by this module through :meth:`UserStore.connect`:

* ``provider_connections``  one row per (user, provider, provider account):
  status, granted scopes, the provider's own stable id for the account, an
  optional account label, and the provider's access and refresh tokens
  encrypted with :mod:`caal.profile_crypto` (AES-256-GCM under the profile
  key ring) and bound to their own row, so a ciphertext copied to another row
  fails to authenticate instead of changing whose account it is. A user may
  link several accounts from one provider; completing an authorization for
  an account that is already linked updates that row and no other. Rows
  written before schema version 4 carry no account id; the first identified
  reconnect whose label matches such a row claims it, so a legacy connection
  keeps its id rather than being duplicated.
* ``oauth_states``  pending authorizations: who started one, for which
  provider, until when, whether it has been redeemed, and the PKCE verifier
  (encrypted, bound to the row).

The ``state`` handed to the provider is ``<id>.<signature>``: a random id and
an HMAC-SHA256 over the row's user, provider and expiry under a key derived
from the internal auth secret. It reveals nothing, cannot be forged or
re-bound, dies with its expiry, and is consumed atomically the first time it
is presented -- by anyone. A state presented by the wrong user or with a bad
signature is treated as compromised and is burned along with the row it named.

Nothing here models a password. Nothing here logs a token, a state, a code,
a label, or an email.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import re
import secrets
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any

from .internal_auth import MIN_SECRET_LENGTH
from .oauth_providers import MAX_SCOPES, PROVIDERS
from .profile_crypto import DecryptionError, KeyRing
from .user_scope import is_valid_user_id
from .user_store import UserStore

logger = logging.getLogger(__name__)

__all__ = [
    "CONNECTED",
    "MAX_LIVE_STATES_PER_USER",
    "REVOKED",
    "STATE_TTL_SECONDS",
    "ConnectionCredentials",
    "ConnectionStore",
    "ConnectionStoreError",
    "ConsumedState",
    "PendingAuthorization",
    "ProviderConnection",
    "StateError",
    "is_valid_connection_id",
    "new_connection_id",
]

CONNECTED = "connected"
REVOKED = "revoked"
STATE_TTL_SECONDS = 600
MAX_LIVE_STATES_PER_USER = 10
MAX_STATE_LENGTH = 256
MAX_TOKEN_LENGTH = 8192
MAX_ACCOUNT_LABEL_LENGTH = 254
MAX_PROVIDER_ACCOUNT_ID_LENGTH = 256
MAX_TOKEN_LIFETIME_SECONDS = 10 * 365 * 86400

CONNECTION_ID_PATTERN = re.compile(r"^con_[0-9a-f]{24}$")
_STATE_ID_BYTES = 24  # token_urlsafe(24) -> 32 characters
_STATE_ID = re.compile(r"^[A-Za-z0-9_-]{32}$")
_SIGNATURE = re.compile(r"^[A-Za-z0-9_-]{43}$")
_SCOPE = re.compile(r"^[A-Za-z0-9_.:/\-]{1,256}$")
_STATE_KEY_DOMAIN = b"caal.provider_connections.oauth_state.v1"
_VERIFIER_BYTES = 64  # token_urlsafe(64) -> 86 characters, within RFC 7636's 43..128


class ConnectionStoreError(Exception):
    """Base class; messages never carry a token, a state, or an email."""


class StateError(ConnectionStoreError, ValueError):
    """The state could not be redeemed. One message for every cause, on purpose."""

    def __init__(self) -> None:
        super().__init__("The authorization state is not valid")


def new_connection_id() -> str:
    return "con_" + secrets.token_hex(12)


def is_valid_connection_id(value: object) -> bool:
    return isinstance(value, str) and CONNECTION_ID_PATTERN.fullmatch(value) is not None


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _now(now: int | None) -> int:
    return int(time.time()) if now is None else int(now)


def _require_provider(provider: object) -> str:
    if not isinstance(provider, str) or provider not in PROVIDERS:
        raise ValueError("Unknown provider")
    return provider


def _require_user(user_id: object) -> str:
    if not is_valid_user_id(user_id):
        raise ValueError("User id has an invalid shape")
    return user_id  # type: ignore[return-value]


def _require_token(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_TOKEN_LENGTH
        or not value.isprintable()
    ):
        raise ValueError(f"{name} must be non-empty printable text")
    return value


def _normalize_label(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.isprintable():
        raise ValueError("Account label must be plain text")
    label = " ".join(value.split())
    if not label:
        return None
    if len(label) > MAX_ACCOUNT_LABEL_LENGTH:
        raise ValueError("Account label is too long")
    return label


def _normalize_provider_account_id(value: object) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_PROVIDER_ACCOUNT_ID_LENGTH
        or not value.isprintable()
        or any(ch.isspace() for ch in value)
    ):
        raise ValueError("Provider account id must be a single printable token")
    return value


def _label_key(label: str | None) -> str | None:
    return None if label is None else label.casefold()


def _normalize_scopes(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        raise ValueError("Scopes must be a sequence of text")
    scopes = tuple(dict.fromkeys(value))
    if len(scopes) > MAX_SCOPES:
        raise ValueError("Too many scopes")
    for scope in scopes:
        if not isinstance(scope, str) or _SCOPE.fullmatch(scope) is None:
            raise ValueError("A scope has unexpected characters")
    return scopes


def _normalize_expires_in(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("expires_in must be a whole number of seconds")
    if not 0 <= value <= MAX_TOKEN_LIFETIME_SECONDS:
        raise ValueError("expires_in is out of bounds")
    return value


# --- models -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PendingAuthorization:
    """What the authorize route needs. The verifier stays in the database."""

    state: str = field(repr=False)
    code_challenge: str | None
    expires_at: int


@dataclass(frozen=True)
class ConsumedState:
    """A redeemed state. The verifier is for the token exchange only."""

    user_id: str
    provider: str
    code_verifier: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class ProviderConnection:
    """One linked account. Never holds a token; the id and label never print."""

    connection_id: str
    user_id: str
    provider: str
    status: str
    account_label: str | None = field(repr=False)
    provider_account_id: str | None = field(default=None, repr=False)
    scopes: tuple[str, ...] = ()
    token_expires_at: int | None = None
    has_refresh_token: bool = False
    created_at: int = 0
    updated_at: int = 0
    connected_at: int | None = None
    revoked_at: int | None = None

    def view(self) -> dict[str, Any]:
        """What the owner may see. Never a token, never the opaque user id."""
        return {
            "connection_id": self.connection_id,
            "provider": self.provider,
            "status": self.status,
            "account_label": self.account_label,
            "scopes": list(self.scopes),
            "token_expires_at": self.token_expires_at,
            "has_refresh_token": self.has_refresh_token,
            "connected_at": self.connected_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ConnectionCredentials:
    """Decrypted tokens for server-side use only. Never render, log, or return them."""

    access_token: str = field(repr=False)
    refresh_token: str | None = field(default=None, repr=False)
    token_expires_at: int | None = None


# --- store --------------------------------------------------------------------------------


class ConnectionStore:
    """Per-user provider connections and OAuth states over the shared CAAL store."""

    def __init__(
        self,
        users: UserStore,
        *,
        keyring: KeyRing,
        state_secret: str,
        state_ttl_seconds: int = STATE_TTL_SECONDS,
        max_live_states_per_user: int = MAX_LIVE_STATES_PER_USER,
    ) -> None:
        if not isinstance(keyring, KeyRing):
            raise ValueError("A profile key ring is required to store provider tokens")
        if not isinstance(state_secret, str) or len(state_secret) < MIN_SECRET_LENGTH:
            raise ValueError("The internal auth secret is missing or too short")
        if not 30 <= int(state_ttl_seconds) <= 3600:
            raise ValueError("State lifetime must be between 30 seconds and one hour")
        self._users = users
        self._keyring = keyring
        # Domain-separated from every other use of the secret.
        self._state_key = hmac.new(
            state_secret.encode("utf-8"), _STATE_KEY_DOMAIN, hashlib.sha256
        ).digest()
        self._state_ttl = int(state_ttl_seconds)
        self._max_live_states = max(1, int(max_live_states_per_user))

    @property
    def state_ttl_seconds(self) -> int:
        return self._state_ttl

    @property
    def max_live_states_per_user(self) -> int:
        return self._max_live_states

    def _connect(self) -> sqlite3.Connection:
        return self._users.connect()

    # --- state -------------------------------------------------------------------

    def _sign(self, state_id: str, user_id: str, provider: str, expires_at: int) -> str:
        message = f"{state_id}\n{user_id}\n{provider}\n{int(expires_at)}".encode("utf-8")
        return _b64(hmac.new(self._state_key, message, hashlib.sha256).digest())

    @staticmethod
    def code_challenge_for(code_verifier: str) -> str:
        """RFC 7636 ``S256`` challenge for a verifier."""
        digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
        return _b64(digest)

    @staticmethod
    def _verifier_aad(state_id: str) -> str:
        return f"caal.oauth_states.code_verifier:{state_id}"

    def begin_authorization(
        self, user_id: object, provider: object, *, pkce: bool, now: int | None = None
    ) -> PendingAuthorization:
        """Record a pending authorization and return its opaque, signed state."""
        owner = _require_user(user_id)
        name = _require_provider(provider)
        moment = _now(now)
        expires_at = moment + self._state_ttl
        state_id = secrets.token_urlsafe(_STATE_ID_BYTES)
        verifier: str | None = None
        verifier_enc: str | None = None
        challenge: str | None = None
        if pkce:
            verifier = secrets.token_urlsafe(_VERIFIER_BYTES)
            verifier_enc = self._keyring.encrypt(verifier, aad=self._verifier_aad(state_id))
            challenge = self.code_challenge_for(verifier)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("DELETE FROM oauth_states WHERE expires_at <= ?", (moment,))
                connection.execute(
                    "INSERT INTO oauth_states (state_id, user_id, provider, code_verifier_enc, "
                    "created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (state_id, owner, name, verifier_enc, moment, expires_at),
                )
                # Keep the number of live states per user bounded: the newest win.
                connection.execute(
                    "DELETE FROM oauth_states WHERE user_id = ? AND consumed_at IS NULL "
                    "AND state_id NOT IN (SELECT state_id FROM oauth_states "
                    "WHERE user_id = ? AND consumed_at IS NULL "
                    "ORDER BY created_at DESC, rowid DESC LIMIT ?)",
                    (owner, owner, self._max_live_states),
                )
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        signature = self._sign(state_id, owner, name, expires_at)
        return PendingAuthorization(
            state=f"{state_id}.{signature}", code_challenge=challenge, expires_at=expires_at
        )

    @staticmethod
    def _split_state(state: object) -> tuple[str, str]:
        if (
            not isinstance(state, str)
            or not 1 <= len(state) <= MAX_STATE_LENGTH
            or not state.isprintable()
            or state.count(".") != 1
        ):
            raise StateError()
        state_id, signature = state.split(".")
        if _STATE_ID.fullmatch(state_id) is None or _SIGNATURE.fullmatch(signature) is None:
            raise StateError()
        return state_id, signature

    def consume_state(
        self, state: object, *, user_id: object, now: int | None = None
    ) -> ConsumedState:
        """Redeem a state exactly once for the user who started it.

        The row is marked consumed *before* the signature and the user binding
        are checked, so any presentation of a state -- genuine, forged with a
        real id, or stolen -- retires it. Every failure raises the same
        :class:`StateError`.
        """
        state_id, signature = self._split_state(state)
        if not is_valid_user_id(user_id):
            raise StateError()
        moment = _now(now)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("DELETE FROM oauth_states WHERE expires_at <= ?", (moment,))
                claimed = connection.execute(
                    "UPDATE oauth_states SET consumed_at = ? "
                    "WHERE state_id = ? AND consumed_at IS NULL",
                    (moment, state_id),
                )
                row = (
                    connection.execute(
                        "SELECT * FROM oauth_states WHERE state_id = ?", (state_id,)
                    ).fetchone()
                    if claimed.rowcount == 1
                    else None
                )
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        if row is None:
            raise StateError()
        expected = self._sign(state_id, row["user_id"], row["provider"], row["expires_at"])
        if not hmac.compare_digest(expected.encode("ascii"), signature.encode("ascii")):
            raise StateError()
        if not hmac.compare_digest(
            str(row["user_id"]).encode("utf-8"), str(user_id).encode("utf-8")
        ):
            raise StateError()
        verifier: str | None = None
        if row["code_verifier_enc"] is not None:
            try:
                verifier = self._keyring.decrypt(
                    row["code_verifier_enc"], aad=self._verifier_aad(state_id)
                )
            except DecryptionError as exc:
                raise StateError() from exc
        return ConsumedState(
            user_id=row["user_id"], provider=row["provider"], code_verifier=verifier
        )

    # --- connections ------------------------------------------------------------------

    @staticmethod
    def _token_aad(field_name: str, connection_id: str) -> str:
        return f"caal.provider_connections.{field_name}:{connection_id}"

    @staticmethod
    def _connection(row: sqlite3.Row) -> ProviderConnection:
        try:
            loaded = json.loads(row["scopes"])
            scopes = tuple(str(item) for item in loaded) if isinstance(loaded, list) else ()
        except ValueError:
            scopes = ()
        return ProviderConnection(
            connection_id=row["connection_id"],
            user_id=row["user_id"],
            provider=row["provider"],
            status=row["status"],
            account_label=row["account_label"],
            provider_account_id=row["provider_account_id"],
            scopes=scopes,
            token_expires_at=row["token_expires_at"],
            has_refresh_token=row["refresh_token_enc"] is not None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            connected_at=row["connected_at"],
            revoked_at=row["revoked_at"],
        )

    @staticmethod
    def _fetch_owned(
        connection: sqlite3.Connection, user_id: object, connection_id: object
    ) -> sqlite3.Row | None:
        if not is_valid_user_id(user_id) or not is_valid_connection_id(connection_id):
            return None
        return connection.execute(
            "SELECT * FROM provider_connections WHERE connection_id = ? AND user_id = ? "
            "AND status = ?",
            (connection_id, user_id, CONNECTED),
        ).fetchone()

    def complete_authorization(
        self,
        user_id: object,
        provider: object,
        *,
        access_token: object,
        refresh_token: object = None,
        expires_in: object = None,
        scopes: object = (),
        provider_account_id: object = None,
        account_label: object = None,
        now: int | None = None,
    ) -> ProviderConnection:
        """Store fresh tokens for one of the user's accounts at ``provider``.

        With a ``provider_account_id`` the row for exactly that account is
        updated, or created if the account is new to this user; other accounts
        from the same provider are untouched. A legacy row that carries no id
        is claimed once, when its label matches. Without an id (callers that
        predate identity resolution) the single unidentified row for the
        provider is updated or created, as before.
        """
        owner = _require_user(user_id)
        name = _require_provider(provider)
        access = _require_token(access_token, name="access_token")
        refresh = (
            None if refresh_token is None else _require_token(refresh_token, name="refresh_token")
        )
        lifetime = _normalize_expires_in(expires_in)
        granted = _normalize_scopes(scopes)
        account_id = _normalize_provider_account_id(provider_account_id)
        label = _normalize_label(account_label)
        moment = _now(now)
        token_expires_at = None if lifetime is None else moment + lifetime
        rendered_scopes = json.dumps(list(granted), separators=(",", ":"))
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = self._find_row_to_update(connection, owner, name, account_id, label)
                connection_id = existing["connection_id"] if existing else new_connection_id()
                access_enc = self._keyring.encrypt(
                    access, aad=self._token_aad("access_token", connection_id)
                )
                refresh_enc = (
                    None
                    if refresh is None
                    else self._keyring.encrypt(
                        refresh, aad=self._token_aad("refresh_token", connection_id)
                    )
                )
                if existing:
                    connection.execute(
                        "UPDATE provider_connections SET status = ?, provider_account_id = ?, "
                        "account_label = ?, scopes = ?, access_token_enc = ?, "
                        "refresh_token_enc = ?, token_expires_at = ?, updated_at = ?, "
                        "connected_at = ?, revoked_at = NULL "
                        "WHERE connection_id = ? AND user_id = ?",
                        (
                            CONNECTED,
                            (
                                account_id
                                if account_id is not None
                                else existing["provider_account_id"]
                            ),
                            label,
                            rendered_scopes,
                            access_enc,
                            refresh_enc,
                            token_expires_at,
                            moment,
                            moment,
                            connection_id,
                            owner,
                        ),
                    )
                else:
                    connection.execute(
                        "INSERT INTO provider_connections (connection_id, user_id, provider, "
                        "status, provider_account_id, account_label, scopes, access_token_enc, "
                        "refresh_token_enc, token_expires_at, created_at, updated_at, "
                        "connected_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            connection_id,
                            owner,
                            name,
                            CONNECTED,
                            account_id,
                            label,
                            rendered_scopes,
                            access_enc,
                            refresh_enc,
                            token_expires_at,
                            moment,
                            moment,
                            moment,
                        ),
                    )
                row = connection.execute(
                    "SELECT * FROM provider_connections WHERE connection_id = ?",
                    (connection_id,),
                ).fetchone()
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        return self._connection(row)

    @staticmethod
    def _find_row_to_update(
        connection: sqlite3.Connection,
        owner: str,
        provider: str,
        account_id: str | None,
        label: str | None,
    ) -> sqlite3.Row | None:
        """The owner's row that this authorization refreshes, if any.

        Identified: the row for that exact account, else a legacy row without
        an id whose label matches (claimed once; it then carries the id).
        Unidentified: the single row without an id, as the previous schema had.
        """
        base = "SELECT * FROM provider_connections WHERE user_id = ? AND provider = ? "
        if account_id is not None:
            exact = connection.execute(
                base + "AND provider_account_id = ?", (owner, provider, account_id)
            ).fetchone()
            if exact is not None:
                return exact
            wanted = _label_key(label)
            if wanted is None:
                return None
            for row in connection.execute(
                base + "AND provider_account_id IS NULL ORDER BY created_at, rowid",
                (owner, provider),
            ):
                if _label_key(row["account_label"]) == wanted:
                    return row
            return None
        return connection.execute(
            base + "AND provider_account_id IS NULL ORDER BY created_at, rowid LIMIT 1",
            (owner, provider),
        ).fetchone()

    def list_connections(self, user_id: object) -> list[ProviderConnection]:
        """The user's live connections, oldest first. Nothing for an invalid id."""
        if not is_valid_user_id(user_id):
            return []
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM provider_connections WHERE user_id = ? AND status = ? "
                "ORDER BY created_at, rowid",
                (user_id, CONNECTED),
            ).fetchall()
        return [self._connection(row) for row in rows]

    def get_connection(self, user_id: object, connection_id: object) -> ProviderConnection | None:
        """A live connection, only if ``user_id`` owns it."""
        with closing(self._connect()) as connection:
            row = self._fetch_owned(connection, user_id, connection_id)
        return self._connection(row) if row else None

    def credentials(self, user_id: object, connection_id: object) -> ConnectionCredentials | None:
        """Decrypt the owner's tokens for server-side use. Never render the result."""
        with closing(self._connect()) as connection:
            row = self._fetch_owned(connection, user_id, connection_id)
        if row is None or row["access_token_enc"] is None:
            return None
        try:
            access = self._keyring.decrypt(
                row["access_token_enc"], aad=self._token_aad("access_token", row["connection_id"])
            )
            refresh = (
                None
                if row["refresh_token_enc"] is None
                else self._keyring.decrypt(
                    row["refresh_token_enc"],
                    aad=self._token_aad("refresh_token", row["connection_id"]),
                )
            )
        except DecryptionError:
            logger.warning("A stored provider token could not be decrypted")
            return None
        return ConnectionCredentials(
            access_token=access, refresh_token=refresh, token_expires_at=row["token_expires_at"]
        )

    def refresh_credentials(
        self,
        user_id: object,
        connection_id: object,
        *,
        access_token: object,
        refresh_token: object = None,
        expires_in: object = None,
        now: int | None = None,
    ) -> ProviderConnection | None:
        """Store a renewed access token for the owner's live connection.

        A new refresh token replaces the stored one when the provider rotated
        it; ``None`` keeps the existing one. Nothing is written, and ``None``
        is returned, unless ``user_id`` owns a live ``connection_id``. The
        tokens are validated exactly as at connect time.
        """
        access = _require_token(access_token, name="access_token")
        refresh = (
            None if refresh_token is None else _require_token(refresh_token, name="refresh_token")
        )
        lifetime = _normalize_expires_in(expires_in)
        if not is_valid_user_id(user_id) or not is_valid_connection_id(connection_id):
            return None
        moment = _now(now)
        token_expires_at = None if lifetime is None else moment + lifetime
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._fetch_owned(connection, user_id, connection_id)
                if row is None:
                    connection.execute("COMMIT")
                    return None
                owned_id = row["connection_id"]
                access_enc = self._keyring.encrypt(
                    access, aad=self._token_aad("access_token", owned_id)
                )
                if refresh is None:
                    connection.execute(
                        "UPDATE provider_connections SET access_token_enc = ?, "
                        "token_expires_at = ?, updated_at = ? "
                        "WHERE connection_id = ? AND user_id = ? AND status = ?",
                        (access_enc, token_expires_at, moment, owned_id, user_id, CONNECTED),
                    )
                else:
                    refresh_enc = self._keyring.encrypt(
                        refresh, aad=self._token_aad("refresh_token", owned_id)
                    )
                    connection.execute(
                        "UPDATE provider_connections SET access_token_enc = ?, "
                        "refresh_token_enc = ?, token_expires_at = ?, updated_at = ? "
                        "WHERE connection_id = ? AND user_id = ? AND status = ?",
                        (
                            access_enc,
                            refresh_enc,
                            token_expires_at,
                            moment,
                            owned_id,
                            user_id,
                            CONNECTED,
                        ),
                    )
                updated = connection.execute(
                    "SELECT * FROM provider_connections WHERE connection_id = ?", (owned_id,)
                ).fetchone()
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        return self._connection(updated)

    def revoke_connection(
        self, user_id: object, connection_id: object, *, now: int | None = None
    ) -> bool:
        """Wipe the tokens and retire the connection. False unless the caller owns a live one."""
        if not is_valid_user_id(user_id) or not is_valid_connection_id(connection_id):
            return False
        moment = _now(now)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                updated = connection.execute(
                    "UPDATE provider_connections SET status = ?, access_token_enc = NULL, "
                    "refresh_token_enc = NULL, token_expires_at = NULL, revoked_at = ?, "
                    "updated_at = ? WHERE connection_id = ? AND user_id = ? AND status = ?",
                    (REVOKED, moment, moment, connection_id, user_id, CONNECTED),
                )
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        return updated.rowcount == 1
