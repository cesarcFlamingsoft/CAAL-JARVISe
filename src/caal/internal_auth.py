"""Short-lived signed principals for CAAL's internal trust boundary.

The Next.js BFF verifies the Cloudflare Access assertion and then speaks to
the CAAL backend (and, through LiveKit dispatch metadata, to the voice agent)
using a compact HS256 JWT signed with a shared secret from the environment.
The token names an opaque subject and an audience, nothing more:

* ``caal-identity``  BFF -> backend, once per resolution: carries the verified
  email so the backend can map it to a user. Single use.
* ``caal-backend``   BFF -> backend API calls: subject is the opaque user id.
  Single use.
* ``caal-agent``     BFF -> voice agent via signed LiveKit room configuration:
  subject is the opaque user id, bound to the room it was minted for.

Verification pins the algorithm, requires every claim, enforces issuer and
audience, bounds the lifetime regardless of what the issuer asked for, and
refuses weak secrets outright. Nothing here logs token contents.
"""

from __future__ import annotations

import hmac
import secrets
import sqlite3
import time
from collections import OrderedDict, deque
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import jwt

__all__ = [
    "AUDIENCE_AGENT",
    "AUDIENCE_BACKEND",
    "AUDIENCE_IDENTITY",
    "DEFAULT_TTL_SECONDS",
    "ISSUER_BFF",
    "MAX_TTL_SECONDS",
    "MIN_SECRET_LENGTH",
    "InMemoryNonceStore",
    "NonceStore",
    "Principal",
    "PrincipalError",
    "RateLimiter",
    "SqliteNonceStore",
    "mint_principal",
    "verify_principal",
]

ISSUER_BFF = "caal-bff"
AUDIENCE_BACKEND = "caal-backend"
AUDIENCE_AGENT = "caal-agent"
AUDIENCE_IDENTITY = "caal-identity"

DEFAULT_TTL_SECONDS = 60
MAX_TTL_SECONDS = 300
MIN_SECRET_LENGTH = 32
LEEWAY_SECONDS = 5
MAX_TOKEN_LENGTH = 4096
MAX_SUBJECT_LENGTH = 254
MAX_AUDIENCE_LENGTH = 64
MAX_JTI_LENGTH = 128
_ALGORITHM = "HS256"
_RESERVED_CLAIMS = frozenset({"iss", "aud", "sub", "iat", "nbf", "exp", "jti"})
_BUSY_TIMEOUT_SECONDS = 5.0


class PrincipalError(ValueError):
    """The token could not be minted or verified. Messages never echo it."""


@dataclass(frozen=True)
class Principal:
    """A verified internal principal. Extra claims never print."""

    subject: str
    audience: str
    issuer: str
    jti: str = field(repr=False)
    issued_at: int
    expires_at: int
    claims: dict[str, Any] = field(default_factory=dict, repr=False)


class NonceStore(Protocol):
    """Single-use registry for token ids."""

    def consume(self, jti: str, *, expires_at: int, now: int | None = None) -> bool:
        """Record ``jti``; return False if it was already used and is still live."""


def _now(now: int | float | None) -> int:
    return int(time.time()) if now is None else int(now)


def _require_secret(secret: object) -> str:
    if not isinstance(secret, str) or len(secret) < MIN_SECRET_LENGTH:
        raise PrincipalError("Internal auth secret is missing or too short")
    return secret


def _require_text(value: object, *, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value or len(value) > limit or not value.isprintable():
        raise PrincipalError(f"{name} is missing or malformed")
    return value


def mint_principal(
    *,
    secret: str,
    subject: str,
    audience: str,
    issuer: str = ISSUER_BFF,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    claims: dict[str, Any] | None = None,
    now: int | float | None = None,
) -> str:
    """Sign a principal. Used by tests and by trusted server-side callers only."""
    key = _require_secret(secret)
    sub = _require_text(subject, name="subject", limit=MAX_SUBJECT_LENGTH)
    aud = _require_text(audience, name="audience", limit=MAX_AUDIENCE_LENGTH)
    iss = _require_text(issuer, name="issuer", limit=MAX_AUDIENCE_LENGTH)
    try:
        ttl = int(ttl_seconds)
    except (TypeError, ValueError) as exc:
        raise PrincipalError("ttl must be an integer") from exc
    if not 1 <= ttl <= MAX_TTL_SECONDS:
        raise PrincipalError(f"ttl must be between 1 and {MAX_TTL_SECONDS} seconds")
    extra: dict[str, Any] = {}
    if claims:
        if not isinstance(claims, dict):
            raise PrincipalError("claims must be a mapping")
        for name, value in claims.items():
            if not isinstance(name, str) or name in _RESERVED_CLAIMS:
                raise PrincipalError("claims may not override reserved names")
            extra[name] = value
    issued = _now(now)
    payload = {
        **extra,
        "iss": iss,
        "aud": aud,
        "sub": sub,
        "iat": issued,
        "nbf": issued,
        "exp": issued + ttl,
        "jti": secrets.token_urlsafe(16),
    }
    return jwt.encode(payload, key, algorithm=_ALGORITHM)


def verify_principal(
    token: object,
    *,
    secret: str,
    audience: str,
    issuer: str = ISSUER_BFF,
    now: int | float | None = None,
    nonce_store: NonceStore | None = None,
    room: str | None = None,
    max_ttl_seconds: int = MAX_TTL_SECONDS,
) -> Principal:
    """Verify a principal or raise :class:`PrincipalError`.

    ``room`` binds ``caal-agent`` tokens to the LiveKit room they were minted
    for. ``nonce_store`` makes the token single use.
    """
    key = _require_secret(secret)
    if not isinstance(token, str) or not token or len(token) > MAX_TOKEN_LENGTH:
        raise PrincipalError("Token is missing or malformed")
    aud = _require_text(audience, name="audience", limit=MAX_AUDIENCE_LENGTH)
    moment = _now(now)
    try:
        header = jwt.get_unverified_header(token)
        if header.get("alg") != _ALGORITHM:
            raise PrincipalError("Unexpected token algorithm")
        payload = jwt.decode(
            token,
            key,
            algorithms=[_ALGORITHM],
            audience=aud,
            issuer=issuer,
            options={
                "require": ["exp", "iat", "jti", "iss", "aud", "sub"],
                # Time claims are checked below against the caller's clock.
                "verify_exp": False,
                "verify_iat": False,
                "verify_nbf": False,
            },
        )
    except jwt.PyJWTError as exc:
        raise PrincipalError("Token failed verification") from exc

    try:
        issued_at = int(payload["iat"])
        expires_at = int(payload["exp"])
        not_before = int(payload.get("nbf", issued_at))
    except (TypeError, ValueError) as exc:
        raise PrincipalError("Token time claims are malformed") from exc
    if expires_at <= issued_at or expires_at - issued_at > max_ttl_seconds:
        raise PrincipalError("Token lifetime is out of bounds")
    if issued_at > moment + LEEWAY_SECONDS or not_before > moment + LEEWAY_SECONDS:
        raise PrincipalError("Token is not yet valid")
    if expires_at <= moment - LEEWAY_SECONDS:
        raise PrincipalError("Token has expired")
    if not isinstance(payload.get("aud"), str):
        raise PrincipalError("Token audience must be a single value")

    subject = payload.get("sub")
    jti = payload.get("jti")
    if not isinstance(subject, str) or not subject or len(subject) > MAX_SUBJECT_LENGTH:
        raise PrincipalError("Token subject is malformed")
    if not isinstance(jti, str) or not 8 <= len(jti) <= MAX_JTI_LENGTH:
        raise PrincipalError("Token id is malformed")

    extra = {name: value for name, value in payload.items() if name not in _RESERVED_CLAIMS}
    if room is not None:
        bound = extra.get("room")
        if not isinstance(bound, str) or not hmac.compare_digest(
            bound.encode("utf-8"), room.encode("utf-8")
        ):
            raise PrincipalError("Token is bound to a different room")
    if nonce_store is not None and not nonce_store.consume(jti, expires_at=expires_at, now=moment):
        raise PrincipalError("Token has already been used")

    return Principal(
        subject=subject,
        audience=aud,
        issuer=payload["iss"],
        jti=jti,
        issued_at=issued_at,
        expires_at=expires_at,
        claims=extra,
    )


# --- nonce stores ------------------------------------------------------------


def _require_jti(jti: object) -> str:
    if not isinstance(jti, str) or not jti or len(jti) > 256:
        raise ValueError("jti must be non-empty text")
    return jti


class InMemoryNonceStore:
    """Process-local single-use registry, bounded and self-pruning."""

    def __init__(self, *, max_entries: int = 50_000) -> None:
        self._seen: dict[str, int] = {}
        self._max_entries = max(1, int(max_entries))

    def consume(self, jti: str, *, expires_at: int, now: int | None = None) -> bool:
        key = _require_jti(jti)
        moment = _now(now)
        self._prune(moment)
        if key in self._seen:
            return False
        if len(self._seen) >= self._max_entries:
            # Refuse rather than forget: forgetting would re-admit a replay.
            oldest = min(self._seen, key=self._seen.__getitem__)
            if self._seen[oldest] > moment:
                return False
            del self._seen[oldest]
        self._seen[key] = int(expires_at)
        return True

    def _prune(self, moment: int) -> None:
        expired = [key for key, until in self._seen.items() if until <= moment]
        for key in expired:
            del self._seen[key]


class SqliteNonceStore:
    """Single-use registry shared across processes through the CAAL SQLite file."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self._path, timeout=_BUSY_TIMEOUT_SECONDS, isolation_level=None
        )
        connection.execute(f"PRAGMA busy_timeout = {int(_BUSY_TIMEOUT_SECONDS * 1000)}")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS auth_nonces "
            "(jti TEXT PRIMARY KEY, expires_at INTEGER NOT NULL)"
        )
        return connection

    def consume(self, jti: str, *, expires_at: int, now: int | None = None) -> bool:
        key = _require_jti(jti)
        moment = _now(now)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("DELETE FROM auth_nonces WHERE expires_at <= ?", (moment,))
                inserted = connection.execute(
                    "INSERT OR IGNORE INTO auth_nonces (jti, expires_at) VALUES (?, ?)",
                    (key, int(expires_at)),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return inserted.rowcount == 1


# --- rate limiting -----------------------------------------------------------


class RateLimiter:
    """Sliding-window limiter keyed by caller, bounded in memory."""

    def __init__(self, *, limit: int, window_seconds: float, max_keys: int = 10_000) -> None:
        if int(limit) < 1 or float(window_seconds) <= 0:
            raise ValueError("limit and window must be positive")
        self._limit = int(limit)
        self._window = float(window_seconds)
        self._max_keys = max(1, int(max_keys))
        self._events: OrderedDict[str, deque[float]] = OrderedDict()

    @property
    def tracked_keys(self) -> int:
        return len(self._events)

    def allow(self, key: str, *, now: float | None = None) -> bool:
        moment = time.time() if now is None else float(now)
        events = self._touch(key, moment)
        if len(events) >= self._limit:
            return False
        events.append(moment)
        return True

    def retry_after(self, key: str, *, now: float | None = None) -> float:
        moment = time.time() if now is None else float(now)
        events = self._touch(key, moment)
        if len(events) < self._limit:
            return 0.0
        return max(0.0, events[0] + self._window - moment)

    def _touch(self, key: str, moment: float) -> deque[float]:
        events = self._events.get(key)
        if events is None:
            events = deque()
            self._events[key] = events
            while len(self._events) > self._max_keys:
                self._events.popitem(last=False)
        else:
            self._events.move_to_end(key)
        cutoff = moment - self._window
        while events and events[0] <= cutoff:
            events.popleft()
        return events
