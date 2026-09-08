"""Startup validation for multi-user identity, authorization, and encryption.

Multi-user JARVIS is only ever *on* when every required setting is present and
valid:

``CAAL_INTERNAL_AUTH_SECRET``     shared secret for BFF -> backend principals
``CAAL_PROFILE_ENCRYPTION_KEYS``  versioned AES-256-GCM key ring
``CAAL_BOOTSTRAP_ADMIN_EMAIL``    the one email allowed to bootstrap as admin

Sign-in itself is **standalone by default**: a password checked against an
Argon2id hash in the deployment's own database, with server-side sessions. No
identity provider, no outbound network, and nothing about Cloudflare is needed
to run multi-user JARVIS.

Cloudflare Access is supported as an *optional additional* identity provider.
Setting ``CF_ACCESS_TEAM_DOMAIN`` and ``CF_ACCESS_AUD`` turns it on; leaving
both unset simply means it is not used, and removes no local control. Setting
only one of the pair is an error rather than a silent downgrade.

Optional local-auth settings:

``CAAL_BOOTSTRAP_ADMIN_PASSWORD_HASH``  Argon2id/scrypt hash (never plaintext)
                                        used once to seed the administrator
``CAAL_PASSWORD_LOGIN``                 ``false`` disables password sign-in,
                                        allowed only when Access is configured
``CAAL_SESSION_IDLE_MINUTES``           sliding session timeout (default 480)
``CAAL_SESSION_ABSOLUTE_HOURS``         hard session lifetime (default 168)

Anything less than a valid configuration leaves every user-scoped endpoint
failing closed. Problems are reported by variable name only; values, secrets,
password hashes, and the admin email never appear in a message or a log line.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .access_jwt import AccessConfig, AccessConfigError, normalize_email
from .internal_auth import MIN_SECRET_LENGTH
from .local_auth import SessionPolicy
from .password_hash import is_password_hash
from .profile_crypto import KeyRing, KeyRingError

__all__ = [
    "ENV_ACCESS_AUD",
    "ENV_ACCESS_TEAM_DOMAIN",
    "ENV_BOOTSTRAP_ADMIN_EMAIL",
    "ENV_BOOTSTRAP_ADMIN_PASSWORD_HASH",
    "ENV_DATA_DIR",
    "ENV_INTERNAL_AUTH_SECRET",
    "ENV_PASSWORD_LOGIN",
    "ENV_PROFILE_ENCRYPTION_KEYS",
    "ENV_SESSION_ABSOLUTE_HOURS",
    "ENV_SESSION_IDLE_MINUTES",
    "OPTIONAL_ENV",
    "REQUIRED_ENV",
    "ConfigProblem",
    "MultiUserConfig",
    "MultiUserStatus",
    "load_multi_user_config",
    "log_startup_status",
]

ENV_INTERNAL_AUTH_SECRET = "CAAL_INTERNAL_AUTH_SECRET"
ENV_PROFILE_ENCRYPTION_KEYS = "CAAL_PROFILE_ENCRYPTION_KEYS"
ENV_BOOTSTRAP_ADMIN_EMAIL = "CAAL_BOOTSTRAP_ADMIN_EMAIL"
ENV_BOOTSTRAP_ADMIN_PASSWORD_HASH = "CAAL_BOOTSTRAP_ADMIN_PASSWORD_HASH"
ENV_PASSWORD_LOGIN = "CAAL_PASSWORD_LOGIN"
ENV_SESSION_IDLE_MINUTES = "CAAL_SESSION_IDLE_MINUTES"
ENV_SESSION_ABSOLUTE_HOURS = "CAAL_SESSION_ABSOLUTE_HOURS"
ENV_ACCESS_TEAM_DOMAIN = "CF_ACCESS_TEAM_DOMAIN"
ENV_ACCESS_AUD = "CF_ACCESS_AUD"
ENV_DATA_DIR = "CAAL_DATA_DIR"

REQUIRED_ENV: tuple[str, ...] = (
    ENV_INTERNAL_AUTH_SECRET,
    ENV_PROFILE_ENCRYPTION_KEYS,
    ENV_BOOTSTRAP_ADMIN_EMAIL,
)
OPTIONAL_ENV: tuple[str, ...] = (
    ENV_BOOTSTRAP_ADMIN_PASSWORD_HASH,
    ENV_PASSWORD_LOGIN,
    ENV_SESSION_IDLE_MINUTES,
    ENV_SESSION_ABSOLUTE_HOURS,
    ENV_ACCESS_TEAM_DOMAIN,
    ENV_ACCESS_AUD,
)
DEFAULT_DATA_DIR = "/app/data"
STORE_FILENAME = "assistant.sqlite3"

_DEFAULT_IDLE_MINUTES = 480
_DEFAULT_ABSOLUTE_HOURS = 168
_TRUE = ("true", "1", "yes", "on")
_FALSE = ("false", "0", "no", "off")


@dataclass(frozen=True)
class ConfigProblem:
    """One unusable setting, by name. ``problem`` is ``missing`` or ``invalid: ...``."""

    name: str
    problem: str


@dataclass(frozen=True)
class MultiUserConfig:
    """Validated multi-user settings. Secrets and the admin email never print."""

    internal_auth_secret: str = field(repr=False)
    keyring: KeyRing = field(repr=False)
    bootstrap_admin_email: str = field(repr=False)
    access: AccessConfig | None = field(default=None, repr=False)
    store_path: Path = field(default=Path(DEFAULT_DATA_DIR) / STORE_FILENAME)
    password_login: bool = True
    bootstrap_admin_password_hash: str | None = field(default=None, repr=False)
    session_policy: SessionPolicy = field(default_factory=SessionPolicy)

    @property
    def access_enabled(self) -> bool:
        return self.access is not None

    def __repr__(self) -> str:
        providers = ["password"] if self.password_login else []
        if self.access is not None:
            providers.append("cloudflare-access")
        return (
            f"MultiUserConfig(keyring={self.keyring!r}, providers={providers}, "
            f"store_path={str(self.store_path)!r})"
        )

    __str__ = __repr__


@dataclass(frozen=True)
class MultiUserStatus:
    """Outcome of validation: a config, or the list of what stands in the way."""

    config: MultiUserConfig | None
    problems: tuple[ConfigProblem, ...]
    attempted: bool

    @property
    def enabled(self) -> bool:
        return self.config is not None

    def describe(self) -> str:
        """Operator-facing summary that never includes a value."""
        if self.enabled:
            assert self.config is not None
            providers = []
            if self.config.password_login:
                providers.append("local password sign-in")
            if self.config.access is not None:
                providers.append("Cloudflare Access")
            return (
                "Multi-user identity is configured ("
                + " and ".join(providers)
                + "): admin panel, per-user profiles and session management are enabled."
            )
        if not self.attempted:
            return (
                "Multi-user identity is not configured; running in legacy single-user mode. "
                "User-scoped endpoints (admin panel, profiles, per-user memory and callbacks) "
                "are disabled and fail closed. To enable them, set: "
                + ", ".join(REQUIRED_ENV)
                + "."
            )
        lines = ", ".join(f"{problem.name} ({problem.problem})" for problem in self.problems)
        return (
            "Multi-user identity configuration is incomplete or invalid, so every user-scoped "
            f"endpoint is disabled and will fail closed until it is fixed: {lines}."
        )


def _present(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is None or value == "":
        return None
    return value


def load_multi_user_config(env: Mapping[str, str] | None = None) -> MultiUserStatus:
    """Validate every setting from ``env`` (default: the process environment)."""
    source: Mapping[str, str] = os.environ if env is None else env
    problems: list[ConfigProblem] = []
    attempted = False

    def missing(name: str) -> None:
        problems.append(ConfigProblem(name=name, problem="missing"))

    def invalid(name: str, reason: str) -> None:
        problems.append(ConfigProblem(name=name, problem=f"invalid: {reason}"))

    secret = _present(source, ENV_INTERNAL_AUTH_SECRET)
    if secret is None:
        missing(ENV_INTERNAL_AUTH_SECRET)
    else:
        attempted = True
        if len(secret.strip()) < MIN_SECRET_LENGTH or secret != secret.strip():
            invalid(
                ENV_INTERNAL_AUTH_SECRET,
                f"must be at least {MIN_SECRET_LENGTH} characters with no surrounding whitespace",
            )

    keyring: KeyRing | None = None
    keys = _present(source, ENV_PROFILE_ENCRYPTION_KEYS)
    if keys is None:
        missing(ENV_PROFILE_ENCRYPTION_KEYS)
    else:
        attempted = True
        try:
            keyring = KeyRing.from_env(keys)
        except KeyRingError as exc:
            invalid(ENV_PROFILE_ENCRYPTION_KEYS, str(exc))

    bootstrap: str | None = None
    raw_bootstrap = _present(source, ENV_BOOTSTRAP_ADMIN_EMAIL)
    if raw_bootstrap is None:
        missing(ENV_BOOTSTRAP_ADMIN_EMAIL)
    else:
        attempted = True
        try:
            bootstrap = normalize_email(raw_bootstrap)
        except ValueError:
            invalid(ENV_BOOTSTRAP_ADMIN_EMAIL, "must be a single well-formed email address")

    # --- Cloudflare Access: optional, but both halves or neither ------------------
    team_domain = _present(source, ENV_ACCESS_TEAM_DOMAIN)
    audience = _present(source, ENV_ACCESS_AUD)
    access: AccessConfig | None = None
    if team_domain is not None or audience is not None:
        attempted = True
        if team_domain is None:
            missing(ENV_ACCESS_TEAM_DOMAIN)
        if audience is None:
            missing(ENV_ACCESS_AUD)
        if team_domain is not None and audience is not None:
            try:
                access = AccessConfig(team_domain=team_domain.strip(), audience=audience.strip())
            except AccessConfigError as exc:
                message = str(exc)
                name = ENV_ACCESS_AUD if ENV_ACCESS_AUD in message else ENV_ACCESS_TEAM_DOMAIN
                invalid(name, message.replace(f"{name} ", ""))

    # --- local password sign-in ---------------------------------------------------
    password_login = True
    raw_password_login = _present(source, ENV_PASSWORD_LOGIN)
    if raw_password_login is not None:
        attempted = True
        flag = raw_password_login.strip().lower()
        if flag in _TRUE:
            password_login = True
        elif flag in _FALSE:
            password_login = False
        else:
            invalid(ENV_PASSWORD_LOGIN, "expected true or false")
    if not password_login and access is None:
        invalid(
            ENV_PASSWORD_LOGIN,
            "cannot be false with no identity provider configured: that leaves no way to sign in",
        )

    bootstrap_hash = _present(source, ENV_BOOTSTRAP_ADMIN_PASSWORD_HASH)
    if bootstrap_hash is not None:
        attempted = True
        if not is_password_hash(bootstrap_hash.strip()):
            invalid(
                ENV_BOOTSTRAP_ADMIN_PASSWORD_HASH,
                "must be an Argon2id or scrypt hash, never a plaintext password",
            )
        else:
            bootstrap_hash = bootstrap_hash.strip()

    idle_seconds = _DEFAULT_IDLE_MINUTES * 60
    raw_idle = _present(source, ENV_SESSION_IDLE_MINUTES)
    if raw_idle is not None:
        attempted = True
        try:
            minutes = int(raw_idle.strip())
            if not 1 <= minutes <= 30 * 24 * 60:
                raise ValueError
            idle_seconds = minutes * 60
        except ValueError:
            invalid(ENV_SESSION_IDLE_MINUTES, "expected a whole number of minutes from 1 to 43200")

    absolute_seconds = _DEFAULT_ABSOLUTE_HOURS * 3600
    raw_absolute = _present(source, ENV_SESSION_ABSOLUTE_HOURS)
    if raw_absolute is not None:
        attempted = True
        try:
            hours = int(raw_absolute.strip())
            if not 1 <= hours <= 90 * 24:
                raise ValueError
            absolute_seconds = hours * 3600
        except ValueError:
            invalid(ENV_SESSION_ABSOLUTE_HOURS, "expected a whole number of hours from 1 to 2160")

    session_policy = SessionPolicy()
    if not any(
        problem.name in (ENV_SESSION_IDLE_MINUTES, ENV_SESSION_ABSOLUTE_HOURS)
        for problem in problems
    ):
        try:
            session_policy = SessionPolicy(
                idle_seconds=idle_seconds, absolute_seconds=absolute_seconds
            )
        except ValueError:
            invalid(
                ENV_SESSION_IDLE_MINUTES,
                "must not exceed " + ENV_SESSION_ABSOLUTE_HOURS,
            )

    data_dir = _present(source, ENV_DATA_DIR) or DEFAULT_DATA_DIR
    store_path = Path(data_dir) / STORE_FILENAME

    if problems:
        return MultiUserStatus(config=None, problems=tuple(problems), attempted=attempted)
    assert secret is not None and keyring is not None and bootstrap is not None
    return MultiUserStatus(
        config=MultiUserConfig(
            internal_auth_secret=secret,
            keyring=keyring,
            bootstrap_admin_email=bootstrap,
            access=access,
            store_path=store_path,
            password_login=password_login,
            bootstrap_admin_password_hash=bootstrap_hash,
            session_policy=session_policy,
        ),
        problems=(),
        attempted=True,
    )


def log_startup_status(status: MultiUserStatus, *, logger: logging.Logger | None = None) -> None:
    """Announce the outcome loudly enough that a half-configured deployment is noticed."""
    target = logger or logging.getLogger(__name__)
    message = status.describe()
    if status.enabled:
        target.info(message)
    elif not status.attempted:
        target.warning(message)
    else:
        target.error("SECURITY CONFIGURATION ERROR: %s", message)
