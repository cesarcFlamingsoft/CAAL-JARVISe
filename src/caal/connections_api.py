"""A user's own provider connections: list, start authorization, finish it, revoke.

The routes live under ``/users/me/connections`` so they inherit the identity
boundary of :mod:`caal.user_api`: a single-use ``caal-backend`` principal from
the BFF names the user, the user is loaded from the database on every call,
mutations share the per-user budget, and the identity middleware makes every
response uncacheable and free of the app-wide CORS policy.

``GET    /users/me/connections``
    the caller's live connections (several per provider are possible), which
    providers the operator has configured (names and a boolean, nothing
    else), and whether this backend can really finish an exchange.
``POST   /users/me/connections/{provider}/authorize``
    an authorization URL carrying an opaque, signed, one-time, expiring,
    user-bound ``state`` -- or, when the provider is not configured, an
    explicit ``configuration_needed`` answer naming the settings by variable
    name so nothing is attempted half-set-up.
``POST   /users/me/connections/callback``
    the BFF forwards the provider redirect's ``state`` and ``code`` (and, for
    Zoho, the two-letter data-center ``location``). The state is redeemed
    first (any failure is the same ``invalid_state``), the code is exchanged
    through the runtime's :class:`TokenExchanger` -- the bounded HTTPS
    transport in :mod:`caal.oauth_exchange` in production -- the provider
    names the account, and the tokens are stored encrypted against that
    account so a second account from the same provider never overwrites the
    first. A failed exchange is ``token_exchange_failed`` with a bounded
    ``reason`` and the setting names an operator would look at, never the
    provider's wording. With no exchanger (no provider configured) the answer
    is an explicit ``token_exchange_unavailable``; the state is spent either
    way.
``DELETE /users/me/connections/{connection_id}``
    revoke: tokens wiped, row retired. Someone else's id is ``not_found``.

Tokens, codes, account ids and emails never appear in an error, an audit
row, or a log line. No route accepts a password or an app password.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from . import user_api
from .oauth_exchange import HttpTokenExchanger
from .oauth_providers import (
    ENV_ZOHO_ACCOUNTS_DOMAIN,
    ProviderRegistry,
    TokenExchangeError,
    TokenExchanger,
    load_provider_registry,
    log_provider_status,
    origin_of,
    spec_for,
    zoho_accounts_origin,
)
from .provider_connections import (
    MAX_STATE_LENGTH,
    STATE_TTL_SECONDS,
    ConnectionStore,
    StateError,
    is_valid_connection_id,
)
from .provider_data import ProviderDataClient
from .user_api import CurrentUser, IdentityRuntime, require_user, throttle_mutation

logger = logging.getLogger(__name__)

__all__ = [
    "STATE_TTL_SECONDS",
    "ConnectionsRuntime",
    "get_connections_runtime",
    "reset_connections_runtime",
    "router",
]

MAX_CODE_LENGTH = 4096
_NOT_CONFIGURED = "Multi-user identity is not configured on this CAAL backend."


# --- runtime ---------------------------------------------------------------------------


class ConnectionsRuntime:
    """The identity runtime plus the provider registry, store, exchange and data transports.

    Tests build one directly; production builds one lazily from the environment
    through :func:`get_connections_runtime`, which attaches the real exchange
    transport only once at least one provider is configured. With no exchanger
    the callback answers ``token_exchange_unavailable`` and the list says so.
    The bounded reader behind the dashboard feeds (:mod:`caal.dashboard_api`)
    is built on first use; tests inject one with a scripted transport.
    """

    def __init__(
        self,
        identity: IdentityRuntime,
        *,
        providers: ProviderRegistry,
        exchanger: TokenExchanger | None = None,
        store: ConnectionStore | None = None,
        data_client: ProviderDataClient | None = None,
    ) -> None:
        self.identity = identity
        self.providers = providers
        self.exchanger = exchanger
        self.store = store or ConnectionStore(
            identity.store,
            keyring=identity.config.keyring,
            state_secret=identity.config.internal_auth_secret,
        )
        self.data_client = data_client

    @property
    def data(self) -> ProviderDataClient:
        """The bounded reader for a user's calendar and inbox data, built on first use."""
        if self.data_client is None:
            self.data_client = ProviderDataClient(self.store, self.providers)
        return self.data_client

    @property
    def token_exchange_available(self) -> bool:
        """Whether an approval at a provider can really become a connection here."""
        return self.exchanger is not None

    def now(self) -> int:
        return self.identity.now()


_runtime_lock = threading.Lock()
_cached: tuple[IdentityRuntime, ConnectionsRuntime] | None = None


def get_connections_runtime(
    identity: IdentityRuntime | None = Depends(user_api.get_runtime),
) -> ConnectionsRuntime | None:
    """The process-wide runtime, or ``None`` while multi-user identity is unconfigured.

    Built once per identity runtime. The bounded HTTPS exchanger is attached
    only when a provider is configured, so an unconfigured backend never has
    a transport to misuse and reports the exchange as unavailable. Tests
    override this dependency rather than touching the environment.
    """
    global _cached
    if identity is None:
        return None
    with _runtime_lock:
        if _cached is None or _cached[0] is not identity:
            registry = load_provider_registry()
            log_provider_status(registry, logger=logger)
            exchanger = HttpTokenExchanger() if registry.configured_providers else None
            if exchanger is None:
                logger.info("Provider token exchange is unavailable: no provider is configured")
            _cached = (
                identity,
                ConnectionsRuntime(identity, providers=registry, exchanger=exchanger),
            )
        return _cached[1]


def reset_connections_runtime() -> None:
    global _cached
    with _runtime_lock:
        _cached = None


def require_connections(
    runtime: ConnectionsRuntime | None = Depends(get_connections_runtime),
) -> ConnectionsRuntime:
    if runtime is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_NOT_CONFIGURED)
    return runtime


# --- schemas -----------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class ProviderAvailability(_Strict):
    provider: str
    display_name: str
    configured: bool


class ConnectionResponse(_Strict):
    connection_id: str
    provider: str
    status: str
    account_label: str | None
    scopes: list[str]
    token_expires_at: int | None
    has_refresh_token: bool
    connected_at: int | None
    updated_at: int


class ConnectionsListResponse(_Strict):
    connections: list[ConnectionResponse]
    providers: list[ProviderAvailability]
    token_exchange_available: bool


class AuthorizeRequest(_Strict):
    """Deliberately empty: any field -- a password above all -- is refused."""


class AuthorizeResponse(_Strict):
    provider: str
    authorization_url: str
    expires_at: int


class CallbackRequest(_Strict):
    state: str = Field(min_length=1, max_length=MAX_STATE_LENGTH)
    code: str = Field(min_length=1, max_length=MAX_CODE_LENGTH)
    # Zoho's data-center code from the redirect (``us``, ``eu``, ...). It
    # decides nothing by itself: it is only checked against the configured
    # accounts server so a code is never sent where it cannot be redeemed.
    location: str | None = Field(default=None, pattern=r"^[a-z]{2}$")


# --- helpers --------------------------------------------------------------------------------


def _configuration_needed(runtime: ConnectionsRuntime, provider: str) -> JSONResponse:
    """The explicit answer for a provider the operator has not set up."""
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": "provider_not_configured",
            "status": "configuration_needed",
            "provider": provider,
            "missing": list(runtime.providers.missing(provider)),
        },
    )


def _exchange_failed(
    provider: str,
    reason: str,
    settings: tuple[str, ...] = (),
    *,
    provider_code: str | None = None,
) -> JSONResponse:
    """The bounded answer for a failed exchange: a reason code and setting names only."""
    if provider_code is None:
        logger.warning("Token exchange with %s failed: %s", provider, reason)
    else:
        logger.warning(
            "Token exchange with %s failed: %s (provider_code=%s)",
            provider,
            reason,
            provider_code,
        )
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={
            "detail": "token_exchange_failed",
            "reason": reason,
            "provider": provider,
            "settings": list(settings),
        },
    )


def _zoho_datacenter_matches(config_token_endpoint: str, location: str | None) -> bool:
    """Whether a Zoho redirect's data center is the one the operator configured.

    No location means the provider did not say; the configured server is used.
    A location that is unknown, or names another data center, fails closed:
    the code would be refused there and must not travel anywhere else.
    """
    if location is None:
        return True
    expected = zoho_accounts_origin(location)
    return expected is not None and expected == origin_of(config_token_endpoint)


def _audit(runtime: ConnectionsRuntime, action: str, user: CurrentUser, provider: str) -> None:
    runtime.identity.store.record_audit(
        action,
        actor=user.actor,
        target_id=user.profile.user_id,
        detail={"provider": provider},
        now=runtime.now(),
    )


# --- routes ---------------------------------------------------------------------------------

router = APIRouter(tags=["connections"])


@router.get("/users/me/connections", response_model=ConnectionsListResponse)
async def list_connections(
    user: CurrentUser = Depends(require_user),
    runtime: ConnectionsRuntime = Depends(require_connections),
) -> ConnectionsListResponse:
    connections = runtime.store.list_connections(user.profile.user_id)
    return ConnectionsListResponse(
        connections=[ConnectionResponse(**item.view()) for item in connections],
        providers=[ProviderAvailability(**entry) for entry in runtime.providers.availability()],
        token_exchange_available=runtime.token_exchange_available,
    )


@router.post("/users/me/connections/callback", response_model=ConnectionResponse)
async def complete_connection(
    body: CallbackRequest,
    user: CurrentUser = Depends(throttle_mutation),
    runtime: ConnectionsRuntime = Depends(require_connections),
) -> Any:
    user_id = user.profile.user_id
    now = runtime.now()
    try:
        consumed = runtime.store.consume_state(body.state, user_id=user_id, now=now)
    except StateError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_state"
        ) from exc

    provider = consumed.provider
    config = runtime.providers.get(provider)
    if config is None:
        return _configuration_needed(runtime, provider)
    if runtime.exchanger is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="token_exchange_unavailable"
        )
    if provider == "zoho" and not _zoho_datacenter_matches(config.token_endpoint, body.location):
        return _exchange_failed(provider, "datacenter_mismatch", (ENV_ZOHO_ACCOUNTS_DOMAIN,))
    try:
        grant = await runtime.exchanger.exchange(
            config,
            code=body.code,
            redirect_uri=config.redirect_uri,
            code_verifier=consumed.code_verifier,
        )
    except TokenExchangeError as exc:
        # The provider's wording may quote the request; only the bounded error
        # code is retained in operator logs for connection troubleshooting.
        return _exchange_failed(
            provider, exc.reason, exc.settings, provider_code=exc.provider_code
        )
    except Exception as exc:  # noqa: BLE001 - transport failure, never echoed
        logger.error("Token exchange with %s raised %s", provider, type(exc).__name__)
        return _exchange_failed(provider, "transport")
    if grant.provider_account_id is None:
        # Without the provider naming the account, a second account from the
        # same provider could overwrite the first. Refuse rather than guess.
        return _exchange_failed(provider, "identity_unavailable", (config.env_scopes,))

    try:
        connection = runtime.store.complete_authorization(
            user_id,
            provider,
            access_token=grant.access_token,
            refresh_token=grant.refresh_token,
            expires_in=grant.expires_in,
            scopes=tuple(grant.scopes),
            provider_account_id=grant.provider_account_id,
            account_label=grant.account_label,
            now=now,
        )
    except ValueError:
        return _exchange_failed(provider, "malformed_response")
    _audit(runtime, "connection.connect", user, provider)
    logger.info("Provider connection completed for %s", provider)
    return ConnectionResponse(**connection.view())


@router.post("/users/me/connections/{provider}/authorize", response_model=AuthorizeResponse)
async def start_authorization(
    provider: str,
    body: AuthorizeRequest | None = None,
    user: CurrentUser = Depends(throttle_mutation),
    runtime: ConnectionsRuntime = Depends(require_connections),
) -> Any:
    if spec_for(provider) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown_provider")
    config = runtime.providers.get(provider)
    if config is None:
        return _configuration_needed(runtime, provider)
    pending = runtime.store.begin_authorization(
        user.profile.user_id, provider, pkce=config.pkce, now=runtime.now()
    )
    url = config.authorization_url(state=pending.state, code_challenge=pending.code_challenge)
    _audit(runtime, "connection.authorize", user, provider)
    logger.info("Provider connection authorization started for %s", provider)
    return AuthorizeResponse(
        provider=provider, authorization_url=url, expires_at=pending.expires_at
    )


@router.delete("/users/me/connections/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def disconnect(
    connection_id: str,
    user: CurrentUser = Depends(throttle_mutation),
    runtime: ConnectionsRuntime = Depends(require_connections),
) -> Response:
    user_id = user.profile.user_id
    if not is_valid_connection_id(connection_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")
    existing = runtime.store.get_connection(user_id, connection_id)
    if existing is None or not runtime.store.revoke_connection(
        user_id, connection_id, now=runtime.now()
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")
    _audit(runtime, "connection.revoke", user, existing.provider)
    logger.info("Provider connection revoked for %s", existing.provider)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
