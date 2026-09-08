"""Cloudflare Access application-token verification.

Cloudflare Access fronts the public JARVIS portal and attaches a signed JWT
to every request it lets through (``Cf-Access-Jwt-Assertion`` header and the
``CF_Authorization`` cookie). Only that signature is identity: a bare
``Cf-Access-Authenticated-User-Email`` header is never trusted.

The verifier checks the token against the team's published JWKS with a fixed
algorithm, the configured issuer (the team domain) and application audience
tag, lifetime claims, and requires a verified ``email`` claim, which service
tokens and login "meta" tokens do not carry. Keys are cached by key id with a
TTL, refreshed once on an unknown key id under a cooldown so an attacker
cannot turn bad tokens into origin traffic, and never trusted when the fetch
fails. Nothing here logs token contents or claims.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import jwt
from jwt.algorithms import RSAAlgorithm

__all__ = [
    "AccessConfig",
    "AccessConfigError",
    "AccessIdentity",
    "AccessTokenError",
    "AccessVerifier",
    "normalize_email",
]

_TEAM_DOMAIN = re.compile(r"^https://[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.cloudflareaccess\.com$")
_AUDIENCE = re.compile(r"^[0-9a-f]{64}$")
_EMAIL = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_ALGORITHM = "RS256"
MAX_EMAIL_LENGTH = 254
MAX_TOKEN_LENGTH = 8192
MAX_KID_LENGTH = 256
MAX_JWKS_BYTES = 256 * 1024
DEFAULT_JWKS_TTL_SECONDS = 3600
MAX_JWKS_STALE_SECONDS = 24 * 3600
DEFAULT_REFRESH_COOLDOWN_SECONDS = 60
DEFAULT_LEEWAY_SECONDS = 30
_FETCH_TIMEOUT_SECONDS = 10.0


class AccessConfigError(ValueError):
    """The Cloudflare Access configuration is unusable."""


class AccessTokenError(ValueError):
    """The assertion did not verify. The message never echoes the token."""


def normalize_email(raw: object) -> str:
    """Lower-case, trimmed email or ``ValueError``. Never logged by callers."""
    if not isinstance(raw, str):
        raise ValueError("email must be text")
    value = raw.strip().lower()
    if not value or len(value) > MAX_EMAIL_LENGTH or not value.isprintable():
        raise ValueError("email is empty, too long, or contains control characters")
    if _EMAIL.fullmatch(value) is None or value.count("@") != 1:
        raise ValueError("email is not well formed")
    return value


@dataclass(frozen=True)
class AccessConfig:
    """Non-secret Cloudflare Access settings for one application."""

    team_domain: str
    audience: str
    jwks_ttl_seconds: int = DEFAULT_JWKS_TTL_SECONDS
    refresh_cooldown_seconds: int = DEFAULT_REFRESH_COOLDOWN_SECONDS
    leeway_seconds: int = DEFAULT_LEEWAY_SECONDS

    def __post_init__(self) -> None:
        if (
            not isinstance(self.team_domain, str)
            or _TEAM_DOMAIN.fullmatch(self.team_domain) is None
        ):
            raise AccessConfigError(
                "CF_ACCESS_TEAM_DOMAIN must look like https://<team>.cloudflareaccess.com"
            )
        if not isinstance(self.audience, str) or _AUDIENCE.fullmatch(self.audience) is None:
            raise AccessConfigError("CF_ACCESS_AUD must be the 64-character hex application tag")
        if int(self.jwks_ttl_seconds) < 1 or int(self.leeway_seconds) < 0:
            raise AccessConfigError("JWKS TTL and leeway must be positive")
        if int(self.refresh_cooldown_seconds) < 0:
            raise AccessConfigError("Refresh cooldown must not be negative")

    @property
    def issuer(self) -> str:
        return self.team_domain

    @property
    def jwks_url(self) -> str:
        return f"{self.team_domain}/cdn-cgi/access/certs"


@dataclass(frozen=True)
class AccessIdentity:
    """A verified user identity. The email never prints."""

    email: str = field(repr=False)
    subject: str = ""
    issued_at: int = 0
    expires_at: int = 0


def fetch_jwks_over_https(url: str) -> dict[str, Any]:
    """Default JWKS fetcher: bounded, TLS-only, no redirects."""
    if not url.startswith("https://"):
        raise AccessTokenError("JWKS URL must use https")
    with httpx.Client(timeout=_FETCH_TIMEOUT_SECONDS, follow_redirects=False) as client:
        response = client.get(url, headers={"Accept": "application/json"})
        response.raise_for_status()
        if len(response.content) > MAX_JWKS_BYTES:
            raise AccessTokenError("JWKS document is too large")
        document = response.json()
    if not isinstance(document, dict):
        raise AccessTokenError("JWKS document is not an object")
    return document


class AccessVerifier:
    """Verify Cloudflare Access tokens against a cached, rotating key set."""

    def __init__(
        self,
        config: AccessConfig,
        *,
        fetch_jwks: Callable[[str], dict[str, Any]] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self._fetch = fetch_jwks or fetch_jwks_over_https
        self._clock = clock
        self._keys: dict[str, Any] = {}
        self._fetched_at: float | None = None
        # Refreshes provoked by an unknown key id are what an attacker can
        # trigger at will, so they get their own cooldown clock.
        self._last_unknown_kid_refresh: float | None = None

    @property
    def cached_key_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._keys))

    def verify(self, token: object) -> AccessIdentity:
        """Return the verified identity or raise :class:`AccessTokenError`."""
        if not isinstance(token, str) or not token or len(token) > MAX_TOKEN_LENGTH:
            raise AccessTokenError("Access token is missing or malformed")
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise AccessTokenError("Access token header is malformed") from exc
        if header.get("alg") != _ALGORITHM:
            raise AccessTokenError("Access token uses an unexpected algorithm")
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid or len(kid) > MAX_KID_LENGTH:
            raise AccessTokenError("Access token names no signing key")

        now = self._clock()
        key = self._key_for(kid, now)
        try:
            payload = jwt.decode(
                token,
                key,
                algorithms=[_ALGORITHM],
                audience=self.config.audience,
                issuer=self.config.issuer,
                options={
                    "require": ["exp", "iat", "iss", "aud", "sub"],
                    # Time claims are checked against the injected clock below.
                    "verify_exp": False,
                    "verify_iat": False,
                    "verify_nbf": False,
                },
            )
        except jwt.PyJWTError as exc:
            raise AccessTokenError("Access token failed verification") from exc

        leeway = int(self.config.leeway_seconds)
        try:
            issued_at = int(payload["iat"])
            expires_at = int(payload["exp"])
            not_before = int(payload.get("nbf", issued_at))
        except (TypeError, ValueError) as exc:
            raise AccessTokenError("Access token time claims are malformed") from exc
        if expires_at <= now - leeway:
            raise AccessTokenError("Access token has expired")
        if issued_at > now + leeway or not_before > now + leeway:
            raise AccessTokenError("Access token is not yet valid")
        token_type = payload.get("type")
        if token_type is not None and token_type != "app":
            raise AccessTokenError("Access token is not an application token")
        subject = payload.get("sub")
        if not isinstance(subject, str) or not subject:
            raise AccessTokenError("Access token has no subject")
        try:
            email = normalize_email(payload.get("email"))
        except ValueError as exc:
            raise AccessTokenError("Access token carries no verified email") from exc
        return AccessIdentity(
            email=email, subject=subject, issued_at=issued_at, expires_at=expires_at
        )

    # --- key cache -------------------------------------------------------------

    def _key_for(self, kid: str, now: float) -> Any:
        if self._fetched_at is None:
            self._refresh(now, required=True)
        elif now - self._fetched_at > int(self.config.jwks_ttl_seconds):
            self._refresh(now, required=now - self._fetched_at > MAX_JWKS_STALE_SECONDS)
        key = self._keys.get(kid)
        if key is not None:
            return key
        cooldown = int(self.config.refresh_cooldown_seconds)
        last = self._last_unknown_kid_refresh
        if last is None or now - last >= cooldown:
            self._last_unknown_kid_refresh = now
            self._refresh(now, required=False)
            key = self._keys.get(kid)
            if key is not None:
                return key
        raise AccessTokenError("Access token is signed by an unknown key")

    def _refresh(self, now: float, *, required: bool) -> None:
        try:
            document = self._fetch(self.config.jwks_url)
            keys = _parse_jwks(document)
        except Exception as exc:
            if required or not self._keys:
                raise AccessTokenError("Access signing keys are unavailable") from exc
            return
        self._keys = keys
        self._fetched_at = now


def _parse_jwks(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict) or not isinstance(document.get("keys"), list):
        raise AccessTokenError("JWKS document has no keys")
    parsed: dict[str, Any] = {}
    for entry in document["keys"]:
        if not isinstance(entry, dict) or entry.get("kty") != "RSA":
            continue
        kid = entry.get("kid")
        if not isinstance(kid, str) or not kid or len(kid) > MAX_KID_LENGTH:
            continue
        if entry.get("use") not in (None, "sig") or entry.get("alg") not in (None, _ALGORITHM):
            continue
        try:
            parsed[kid] = RSAAlgorithm.from_jwk(json.dumps(entry))
        except (jwt.PyJWTError, ValueError, TypeError):
            continue
    return parsed
