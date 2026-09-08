"""Catalogue and configuration of the OAuth providers a user may connect.

Three providers are supported -- ``google``, ``microsoft`` and ``zoho``. Each
has a fixed specification (endpoints, default scopes, whether PKCE is used)
and an operator-supplied configuration read from the environment **by name**:

``CAAL_OAUTH_<PROVIDER>_CLIENT_ID``      the application's OAuth client id
``CAAL_OAUTH_<PROVIDER>_CLIENT_SECRET``  the application's OAuth client secret
``CAAL_OAUTH_<PROVIDER>_SCOPES``         optional space-separated scope override
``CAAL_OAUTH_MICROSOFT_TENANT``          optional tenant (default ``common``)
``CAAL_OAUTH_ZOHO_ACCOUNTS_DOMAIN``      optional accounts host, ``https://`` only
                                        (default ``https://accounts.zoho.com``)
``CAAL_PUBLIC_ORIGIN``                   the portal origin the provider sends the
                                        browser back to; the agent reads it from
                                        the same ``.env`` the frontend uses

These are the *application's* credentials, never a person's. Nothing in this
module asks for, models, or stores a user password or app password; a user
links an account only by authorizing at the provider. This is deliberately
separate from the operator-wide ``settings.json`` credentials that the legacy
email and calendar tools read.

The default scopes are read-only mail and calendar access plus whatever the
provider needs to say *which* account was linked (its identity scopes). A
scope override replaces the product scopes but the identity scopes are always
kept: without them two accounts from one provider could not be told apart.
Scopes are given space-separated in the environment; each provider's own
delimiter (Zoho uses commas) is applied when the authorization URL is built.

Missing or unusable settings are reported by variable name only. Values -- the
client secret above all -- never render, log, or travel in a URL.

No network call lives here. Redeeming an authorization code is the job of a
:class:`TokenExchanger`, an injectable collaborator; this module ships the
protocol, the result type and the bounded error vocabulary, not a transport
(see :mod:`caal.oauth_exchange` for the production one).
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlencode, urlsplit

from .security_config import ConfigProblem

__all__ = [
    "DEFAULT_MICROSOFT_TENANT",
    "DEFAULT_ZOHO_ACCOUNTS_DOMAIN",
    "ENV_MICROSOFT_TENANT",
    "ENV_PUBLIC_ORIGIN",
    "ENV_ZOHO_ACCOUNTS_DOMAIN",
    "EXCHANGE_REASONS",
    "MAX_SCOPES",
    "OAUTH_CALLBACK_PATH",
    "PROVIDERS",
    "ZOHO_ACCOUNTS_ORIGINS",
    "ProviderConfig",
    "ProviderRegistry",
    "ProviderSpec",
    "TokenExchangeError",
    "TokenExchanger",
    "TokenGrant",
    "all_env_names",
    "load_provider_registry",
    "log_provider_status",
    "origin_of",
    "spec_for",
    "zoho_accounts_origin",
]

PROVIDERS: tuple[str, ...] = ("google", "microsoft", "zoho")

ENV_PUBLIC_ORIGIN = "CAAL_PUBLIC_ORIGIN"
ENV_MICROSOFT_TENANT = "CAAL_OAUTH_MICROSOFT_TENANT"
ENV_ZOHO_ACCOUNTS_DOMAIN = "CAAL_OAUTH_ZOHO_ACCOUNTS_DOMAIN"

# The BFF route that receives the provider redirect. It authenticates the
# browser's session and forwards ``state`` and ``code`` to the backend.
OAUTH_CALLBACK_PATH = "/api/connections/callback"

DEFAULT_MICROSOFT_TENANT = "common"
DEFAULT_ZOHO_ACCOUNTS_DOMAIN = "https://accounts.zoho.com"
MAX_SCOPES = 50
_MAX_CLIENT_VALUE_LENGTH = 512

# Zoho runs one accounts server per data center. A code issued by one cannot
# be redeemed at another, and the provider tells us which one issued it with
# a two-letter ``location`` on the redirect. Only these are ever trusted.
ZOHO_ACCOUNTS_ORIGINS: dict[str, str] = {
    "us": "https://accounts.zoho.com",
    "eu": "https://accounts.zoho.eu",
    "in": "https://accounts.zoho.in",
    "au": "https://accounts.zoho.com.au",
    "jp": "https://accounts.zoho.jp",
    "ca": "https://accounts.zohocloud.ca",
    "sa": "https://accounts.zoho.sa",
    "uk": "https://accounts.zoho.uk",
}

# Why an exchange failed, for callers that must answer without quoting the
# provider. ``settings`` on the error names which variables an operator would
# look at; nothing else about the failure travels.
EXCHANGE_REASONS: tuple[str, ...] = (
    "provider_refused",
    "transport",
    "malformed_response",
    "identity_unavailable",
    "insufficient_scope",
    "datacenter_mismatch",
)

_ORIGIN = re.compile(
    r"^(?P<scheme>https?)://(?P<host>[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?)"
    r"(?P<port>:\d{1,5})?/?$"
)
_TENANT = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.-]{0,126}[A-Za-z0-9])?$")
_SCOPE = re.compile(r"^[A-Za-z0-9_.:/\-]{1,256}$")
_ZOHO_LOCATION = re.compile(r"^[a-z]{2}$")


# --- catalogue ---------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderSpec:
    """What is fixed about a provider. Holds no credential of any kind."""

    provider: str
    display_name: str
    default_scopes: tuple[str, ...]
    identity_scopes: tuple[str, ...]
    pkce: bool
    scope_delimiter: str = " "
    extra_authorize_params: tuple[tuple[str, str], ...] = ()

    def _env(self, suffix: str) -> str:
        return f"CAAL_OAUTH_{self.provider.upper()}_{suffix}"

    @property
    def env_client_id(self) -> str:
        return self._env("CLIENT_ID")

    @property
    def env_client_secret(self) -> str:
        return self._env("CLIENT_SECRET")

    @property
    def env_scopes(self) -> str:
        return self._env("SCOPES")


_SPECS: dict[str, ProviderSpec] = {
    "google": ProviderSpec(
        provider="google",
        display_name="Google",
        # Read-only Gmail and Calendar. ``openid email`` yields the ID token
        # that names the account (``sub``) so two Google accounts can coexist.
        default_scopes=(
            "openid",
            "email",
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/calendar.readonly",
        ),
        identity_scopes=("openid", "email"),
        pkce=True,
        extra_authorize_params=(
            ("access_type", "offline"),
            # ``select_account`` so a person with several Google accounts can
            # pick the one to link; ``consent`` so a refresh token is issued.
            ("prompt", "select_account consent"),
            ("include_granted_scopes", "true"),
        ),
    ),
    "microsoft": ProviderSpec(
        provider="microsoft",
        display_name="Microsoft",
        # Read-only Outlook mail and calendar; ``offline_access`` for a
        # refresh token; ``openid email`` for the ID token naming the account.
        default_scopes=("openid", "email", "offline_access", "Mail.Read", "Calendars.Read"),
        identity_scopes=("openid", "email"),
        pkce=True,
        extra_authorize_params=(("response_mode", "query"), ("prompt", "select_account")),
    ),
    "zoho": ProviderSpec(
        provider="zoho",
        display_name="Zoho",
        # Read-only Zoho Mail and Calendar. ``AaaServer.profile.READ`` lets the
        # accounts server say which Zoho account (ZUID) was linked.
        default_scopes=(
            "AaaServer.profile.READ",
            "ZohoMail.accounts.READ",
            "ZohoMail.messages.READ",
            "ZohoCalendar.calendar.READ",
            "ZohoCalendar.event.READ",
        ),
        identity_scopes=("AaaServer.profile.READ",),
        # Zoho's server-side web flow is documented without PKCE; the client
        # secret plus the signed, user-bound state protect the exchange.
        pkce=False,
        # Zoho wants scopes comma-separated in the authorization request.
        scope_delimiter=",",
        extra_authorize_params=(("access_type", "offline"), ("prompt", "consent")),
    ),
}


def spec_for(provider: object) -> ProviderSpec | None:
    """The fixed specification for an exact provider name, else ``None``."""
    if not isinstance(provider, str):
        return None
    return _SPECS.get(provider)


def all_env_names() -> tuple[str, ...]:
    """Every environment variable this module reads, for documentation and tests."""
    names: list[str] = []
    for spec in _SPECS.values():
        names.extend((spec.env_client_id, spec.env_client_secret, spec.env_scopes))
    names.extend((ENV_MICROSOFT_TENANT, ENV_ZOHO_ACCOUNTS_DOMAIN, ENV_PUBLIC_ORIGIN))
    return tuple(names)


def _endpoints(spec: ProviderSpec, *, tenant: str, zoho_domain: str) -> tuple[str, str, str]:
    """(authorization, token, identity) endpoints. All https, all fixed per provider."""
    if spec.provider == "google":
        return (
            "https://accounts.google.com/o/oauth2/v2/auth",
            "https://oauth2.googleapis.com/token",
            "https://openidconnect.googleapis.com/v1/userinfo",
        )
    if spec.provider == "microsoft":
        base = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0"
        return (f"{base}/authorize", f"{base}/token", "https://graph.microsoft.com/v1.0/me")
    if spec.provider == "zoho":
        return (
            f"{zoho_domain}/oauth/v2/auth",
            f"{zoho_domain}/oauth/v2/token",
            f"{zoho_domain}/oauth/user/info",
        )
    raise ValueError("Unknown provider")


def origin_of(url: str) -> str:
    """``scheme://host[:port]`` of an absolute URL, lower-cased."""
    parts = urlsplit(url)
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}"


def zoho_accounts_origin(location: object) -> str | None:
    """The accounts origin for a Zoho data-center code, or ``None`` for anything else."""
    if not isinstance(location, str) or _ZOHO_LOCATION.fullmatch(location) is None:
        return None
    return ZOHO_ACCOUNTS_ORIGINS.get(location)


# --- configuration -----------------------------------------------------------------


@dataclass(frozen=True)
class ProviderConfig:
    """A provider the operator has configured. The client secret never prints."""

    provider: str
    display_name: str
    client_id: str
    client_secret: str = field(repr=False)
    scopes: tuple[str, ...]
    authorization_endpoint: str
    token_endpoint: str
    redirect_uri: str
    pkce: bool
    identity_scopes: tuple[str, ...] = ()
    identity_endpoint: str = ""
    scope_delimiter: str = " "
    extra_authorize_params: tuple[tuple[str, str], ...] = field(default=(), repr=False)

    def __repr__(self) -> str:
        return (
            f"ProviderConfig(provider={self.provider!r}, redirect_uri={self.redirect_uri!r}, "
            f"pkce={self.pkce}, scopes={len(self.scopes)})"
        )

    __str__ = __repr__

    @property
    def env_scopes(self) -> str:
        """The variable an operator edits to change this provider's scopes."""
        return _SPECS[self.provider].env_scopes if self.provider in _SPECS else ""

    def authorization_url(self, *, state: str, code_challenge: str | None) -> str:
        """The URL the browser is sent to. Carries the state, never the secret."""
        if not isinstance(state, str) or not state or not state.isprintable():
            raise ValueError("state is required")
        params: list[tuple[str, str]] = [
            ("client_id", self.client_id),
            ("redirect_uri", self.redirect_uri),
            ("response_type", "code"),
            ("scope", self.scope_delimiter.join(self.scopes)),
            ("state", state),
            *self.extra_authorize_params,
        ]
        if code_challenge is not None:
            if not isinstance(code_challenge, str) or not code_challenge:
                raise ValueError("code_challenge must be text")
            params.append(("code_challenge", code_challenge))
            params.append(("code_challenge_method", "S256"))
        return f"{self.authorization_endpoint}?{urlencode(params)}"


@dataclass(frozen=True)
class ProviderRegistry:
    """Which providers are usable, and by name what stands in the way of the rest."""

    _configs: dict[str, ProviderConfig] = field(repr=False)
    _missing: dict[str, tuple[str, ...]] = field(repr=False)
    problems: tuple[ConfigProblem, ...] = ()

    def __repr__(self) -> str:
        return (
            f"ProviderRegistry(configured={list(self.configured_providers)}, "
            f"missing={dict(self._missing)})"
        )

    __str__ = __repr__

    @property
    def configured_providers(self) -> tuple[str, ...]:
        return tuple(name for name in PROVIDERS if name in self._configs)

    def get(self, provider: object) -> ProviderConfig | None:
        if not isinstance(provider, str):
            return None
        return self._configs.get(provider)

    def configured(self, provider: object) -> bool:
        return self.get(provider) is not None

    def missing(self, provider: object) -> tuple[str, ...]:
        """Names of the settings that are absent or unusable for ``provider``.

        Empty for a configured provider, and for a name that is not a provider.
        """
        if not isinstance(provider, str):
            return ()
        return self._missing.get(provider, ())

    def availability(self) -> list[dict[str, Any]]:
        """What a signed-in user may see: names and a boolean, nothing else."""
        return [
            {
                "provider": name,
                "display_name": _SPECS[name].display_name,
                "configured": name in self._configs,
            }
            for name in PROVIDERS
        ]

    def describe(self) -> str:
        """Operator-facing summary that never includes a value."""
        parts = []
        for name in PROVIDERS:
            if name in self._configs:
                parts.append(f"{name} configured")
            else:
                parts.append(f"{name} needs " + ", ".join(self._missing.get(name, ())))
        text = "Provider connections: " + "; ".join(parts) + "."
        if self.problems:
            text += " Unusable settings: " + ", ".join(
                f"{problem.name} ({problem.problem})" for problem in self.problems
            )
            text += "."
        return text


def _present(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _normalize_origin(value: str, *, require_https: bool = False) -> str:
    match = _ORIGIN.fullmatch(value)
    if match is None:
        raise ValueError("must be an http(s) origin with no path, query or credentials")
    if require_https and match.group("scheme") != "https":
        raise ValueError("must use https")
    port = match.group("port") or ""
    return f"{match.group('scheme')}://{match.group('host')}{port}"


def _parse_scopes(value: str) -> tuple[str, ...]:
    scopes = tuple(dict.fromkeys(value.split()))
    if not scopes or len(scopes) > MAX_SCOPES:
        raise ValueError(f"expected 1 to {MAX_SCOPES} space-separated scopes")
    for scope in scopes:
        if _SCOPE.fullmatch(scope) is None:
            raise ValueError("contains a scope with unexpected characters")
    return scopes


def _client_value(value: str) -> bool:
    return 0 < len(value) <= _MAX_CLIENT_VALUE_LENGTH and value.isprintable() and " " not in value


def load_provider_registry(env: Mapping[str, str] | None = None) -> ProviderRegistry:
    """Validate every provider setting from ``env`` (default: the process environment)."""
    source: Mapping[str, str] = os.environ if env is None else env
    problems: list[ConfigProblem] = []

    def invalid(name: str, reason: str) -> None:
        problems.append(ConfigProblem(name=name, problem=f"invalid: {reason}"))

    origin: str | None = None
    origin_unusable = True
    raw_origin = _present(source, ENV_PUBLIC_ORIGIN)
    if raw_origin is not None:
        try:
            origin = _normalize_origin(raw_origin)
            origin_unusable = False
        except ValueError as exc:
            invalid(ENV_PUBLIC_ORIGIN, str(exc))

    tenant = DEFAULT_MICROSOFT_TENANT
    tenant_unusable = False
    raw_tenant = _present(source, ENV_MICROSOFT_TENANT)
    if raw_tenant is not None:
        if _TENANT.fullmatch(raw_tenant) is None:
            invalid(ENV_MICROSOFT_TENANT, "must be a tenant id or domain")
            tenant_unusable = True
        else:
            tenant = raw_tenant

    zoho_domain = DEFAULT_ZOHO_ACCOUNTS_DOMAIN
    zoho_unusable = False
    raw_zoho = _present(source, ENV_ZOHO_ACCOUNTS_DOMAIN)
    if raw_zoho is not None:
        try:
            zoho_domain = _normalize_origin(raw_zoho, require_https=True)
        except ValueError as exc:
            invalid(ENV_ZOHO_ACCOUNTS_DOMAIN, str(exc))
            zoho_unusable = True

    configs: dict[str, ProviderConfig] = {}
    missing: dict[str, tuple[str, ...]] = {}
    for name in PROVIDERS:
        spec = _SPECS[name]
        unusable: list[str] = []

        client_id = _present(source, spec.env_client_id)
        if client_id is None:
            unusable.append(spec.env_client_id)
        elif not _client_value(client_id):
            invalid(spec.env_client_id, "must be a single printable token")
            unusable.append(spec.env_client_id)

        client_secret = _present(source, spec.env_client_secret)
        if client_secret is None:
            unusable.append(spec.env_client_secret)
        elif not _client_value(client_secret):
            invalid(spec.env_client_secret, "must be a single printable token")
            unusable.append(spec.env_client_secret)

        scopes = spec.default_scopes
        raw_scopes = _present(source, spec.env_scopes)
        if raw_scopes is not None:
            try:
                scopes = _parse_scopes(raw_scopes)
            except ValueError as exc:
                invalid(spec.env_scopes, str(exc))
                unusable.append(spec.env_scopes)

        if name == "microsoft" and tenant_unusable:
            unusable.append(ENV_MICROSOFT_TENANT)
        if name == "zoho" and zoho_unusable:
            unusable.append(ENV_ZOHO_ACCOUNTS_DOMAIN)
        if origin_unusable:
            unusable.append(ENV_PUBLIC_ORIGIN)

        missing[name] = tuple(unusable)
        if unusable:
            continue
        assert client_id is not None and client_secret is not None and origin is not None
        authorization_endpoint, token_endpoint, identity_endpoint = _endpoints(
            spec, tenant=tenant, zoho_domain=zoho_domain
        )
        # The identity scopes are never optional: they are how the linked
        # account is told apart from the user's other accounts at the provider.
        scopes = tuple(dict.fromkeys((*spec.identity_scopes, *scopes)))
        configs[name] = ProviderConfig(
            provider=name,
            display_name=spec.display_name,
            client_id=client_id,
            client_secret=client_secret,
            scopes=scopes,
            authorization_endpoint=authorization_endpoint,
            token_endpoint=token_endpoint,
            redirect_uri=origin + OAUTH_CALLBACK_PATH,
            pkce=spec.pkce,
            identity_scopes=spec.identity_scopes,
            identity_endpoint=identity_endpoint,
            scope_delimiter=spec.scope_delimiter,
            extra_authorize_params=spec.extra_authorize_params,
        )

    return ProviderRegistry(_configs=configs, _missing=missing, problems=tuple(problems))


def log_provider_status(
    registry: ProviderRegistry, *, logger: logging.Logger | None = None
) -> None:
    """Announce which providers are usable, by name only."""
    target = logger or logging.getLogger(__name__)
    if registry.problems:
        target.error("OAUTH CONFIGURATION ERROR: %s", registry.describe())
    else:
        target.info(registry.describe())


# --- token exchange --------------------------------------------------------------------


class TokenExchangeError(Exception):
    """The provider refused or failed the code exchange.

    ``reason`` is one of :data:`EXCHANGE_REASONS` and ``settings`` names the
    environment variables an operator would look at. Both are safe to answer
    and to log. The message is for the exception chain only: callers must
    never surface it to a browser or a log line.
    """

    def __init__(
        self,
        message: str = "The token exchange failed",
        *,
        reason: str = "provider_refused",
        settings: tuple[str, ...] = (),
        provider_code: str | None = None,
    ) -> None:
        if reason not in EXCHANGE_REASONS:
            raise ValueError("Unknown token exchange reason")
        if provider_code is not None and (
            not isinstance(provider_code, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", provider_code) is None
        ):
            raise ValueError("Unsafe provider error code")
        super().__init__(message)
        self.reason = reason
        self.settings = tuple(settings)
        # A short OAuth error *code* (for example ``invalid_client``) is safe
        # for an operator diagnostic. Provider descriptions are never retained.
        self.provider_code = provider_code


@dataclass(frozen=True)
class TokenGrant:
    """What a successful exchange yields.

    ``provider_account_id`` is the provider's own stable identifier for the
    linked account (Google ``sub``, Microsoft ``oid``, Zoho ``ZUID``); it is
    what keeps two accounts from one provider apart and lets a reconnect find
    its own row. Tokens, the id and the label never print.
    """

    access_token: str = field(repr=False)
    refresh_token: str | None = field(default=None, repr=False)
    expires_in: int | None = None
    scopes: tuple[str, ...] = ()
    provider_account_id: str | None = field(default=None, repr=False)
    account_label: str | None = field(default=None, repr=False)


class TokenExchanger(Protocol):
    """Redeems an authorization code at the provider's token endpoint.

    The production runtime wires :class:`caal.oauth_exchange.HttpTokenExchanger`
    once at least one provider is configured; with none, the callback route
    answers ``token_exchange_unavailable`` rather than pretending. Tests
    inject a fake.
    """

    async def exchange(
        self,
        config: ProviderConfig,
        *,
        code: str,
        redirect_uri: str,
        code_verifier: str | None,
    ) -> TokenGrant: ...
