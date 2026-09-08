"""Identity, profile, and administration API for multi-user JARVIS.

Every route here sits behind CAAL's internal trust boundary. The only
legitimate caller is the Next.js BFF, which proves itself with a short-lived,
single-use signed principal (:mod:`caal.internal_auth`) on each request.

Sign-in is **standalone**: ``POST /auth/login`` checks a password against the
Argon2id hash in this deployment's own database (:mod:`caal.local_auth`) and
returns an opaque server-side session token that the BFF stores in an HttpOnly
cookie. ``POST /auth/session`` re-resolves that token, from the database, on
every request. Nothing about this path needs Cloudflare, an identity provider,
or any outbound network.

Cloudflare Access remains available as an *optional additional* provider: when
it is configured, ``POST /auth/resolve`` maps a verified Access assertion to
the same opaque users. It is never required, and when it is absent no local
control is weakened -- the route simply does not exist.

Authorization decisions are made here and in :mod:`caal.user_store` from the
database, never from token claims: a principal only names an opaque user id.
Responses carry ``Cache-Control: no-store`` and never include an approved
callback number. The whole router fails closed with ``503`` until the
multi-user configuration (:mod:`caal.security_config`) validates.
"""

from __future__ import annotations

import hmac
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from .access_jwt import AccessTokenError, AccessVerifier, normalize_email
from .internal_auth import (
    AUDIENCE_BACKEND,
    AUDIENCE_IDENTITY,
    InMemoryNonceStore,
    NonceStore,
    Principal,
    PrincipalError,
    RateLimiter,
    verify_principal,
)
from .local_auth import (
    REASON_LOCKED,
    AccountLockedError,
    InvalidCredentialsError,
    LocalAuth,
)
from .password_hash import MAX_PASSWORD_LENGTH, PasswordPolicyError
from .security_config import MultiUserConfig, load_multi_user_config, log_startup_status
from .tools import memory_tools
from .user_store import (
    ACTIVE,
    ADMIN,
    Actor,
    CallbackNumberError,
    DuplicateUserError,
    LastAdminError,
    NotConfiguredError,
    UnknownUserError,
    UserProfile,
    UserStore,
    UserSuspendedError,
    is_valid_user_id,
    normalize_e164,
)

logger = logging.getLogger(__name__)

__all__ = [
    "IdentityResponseHeadersMiddleware",
    "IdentityRuntime",
    "get_runtime",
    "reset_runtime",
    "router",
]

IDENTITY_PATH_PREFIXES = ("/identity", "/auth", "/users", "/admin")
RESOLVE_LIMIT_PER_MINUTE = 120
MUTATION_LIMIT_PER_MINUTE = 60
# Deliberately tight: every attempt past this costs an attacker a full minute,
# and the per-account lockout in :mod:`caal.local_auth` runs underneath it.
LOGIN_LIMIT_PER_MINUTE = 10
MAX_AUDIT_LIMIT = 500
MAX_SESSION_TOKEN_LENGTH = 256
MIN_SESSION_TOKEN_LENGTH = 16

_NOT_CONFIGURED = "Multi-user identity is not configured on this CAAL backend."
_UNAUTHORIZED = "unauthorized"
_FORBIDDEN = "forbidden"
_INVALID_CREDENTIALS = "invalid_credentials"


# --- runtime ------------------------------------------------------------------


class _StoreNonces:
    """Adapt the user store's nonce table to the :class:`NonceStore` protocol."""

    def __init__(self, store: UserStore) -> None:
        self._store = store

    def consume(self, jti: str, *, expires_at: int, now: int | None = None) -> bool:
        return self._store.consume_nonce(jti, expires_at=expires_at, now=now)


class IdentityRuntime:
    """The validated configuration plus the collaborators the routes need.

    Tests build one directly with fakes; production builds one from the
    environment through :func:`get_runtime`.
    """

    def __init__(
        self,
        config: MultiUserConfig,
        *,
        store: UserStore,
        access_verifier: AccessVerifier | None = None,
        local_auth: LocalAuth | None = None,
        nonce_store: NonceStore | None = None,
        resolve_limiter: RateLimiter | None = None,
        mutation_limiter: RateLimiter | None = None,
        login_limiter: RateLimiter | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.store = store
        self.access_verifier = access_verifier
        self.clock = clock
        self.local_auth = local_auth or LocalAuth(
            store, policy=config.session_policy, clock=clock
        )
        self.nonce_store: NonceStore = nonce_store or _StoreNonces(store)
        self.resolve_limiter = resolve_limiter or RateLimiter(
            limit=RESOLVE_LIMIT_PER_MINUTE, window_seconds=60
        )
        self.mutation_limiter = mutation_limiter or RateLimiter(
            limit=MUTATION_LIMIT_PER_MINUTE, window_seconds=60
        )
        self.login_limiter = login_limiter or RateLimiter(
            limit=LOGIN_LIMIT_PER_MINUTE, window_seconds=60
        )

    @classmethod
    def from_config(cls, config: MultiUserConfig) -> IdentityRuntime:
        store = UserStore(config.store_path, keyring=config.keyring)
        runtime = cls(
            config,
            store=store,
            # Cloudflare Access is optional: no configuration, no verifier, and
            # the route that would use one refuses outright.
            access_verifier=AccessVerifier(config.access) if config.access else None,
        )
        runtime.seed_bootstrap_admin()
        return runtime

    def seed_bootstrap_admin(self) -> None:
        """Install the configured administrator's one-time password, once.

        The operator supplies only a hash, so no plaintext ever exists in the
        environment or the image, and the account is flagged to force a change
        at first sign-in. Idempotent, and it never re-applies the hash once the
        account has a credential of its own.
        """
        encoded = self.config.bootstrap_admin_password_hash
        if not encoded:
            return
        try:
            self.local_auth.ensure_bootstrap_admin(
                self.config.bootstrap_admin_email, encoded, now=self.now()
            )
        except Exception:
            # Never let a seeding problem take the whole backend down; the
            # message is suppressed because it could quote configuration.
            logger.error("SECURITY: the bootstrap administrator could not be seeded")

    def now(self) -> int:
        return int(self.clock())


_runtime_lock = threading.Lock()
_runtime: IdentityRuntime | None = None
_runtime_loaded = False


def get_runtime() -> IdentityRuntime | None:
    """The process-wide runtime, or ``None`` while multi-user is unconfigured.

    Built once from the environment. Tests override this dependency through
    ``app.dependency_overrides`` rather than by touching the environment.
    """
    global _runtime, _runtime_loaded
    with _runtime_lock:
        if not _runtime_loaded:
            status_ = load_multi_user_config()
            log_startup_status(status_, logger=logger)
            _runtime = IdentityRuntime.from_config(status_.config) if status_.config else None
            _runtime_loaded = True
        return _runtime


def reset_runtime() -> None:
    """Forget the cached runtime so the next request re-reads the environment."""
    global _runtime, _runtime_loaded
    with _runtime_lock:
        _runtime = None
        _runtime_loaded = False


def require_runtime(runtime: IdentityRuntime | None = Depends(get_runtime)) -> IdentityRuntime:
    if runtime is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_NOT_CONFIGURED)
    return runtime


@dataclass(frozen=True)
class _SecretOnlyConfig:
    internal_auth_secret: str = field(repr=False)


class PrincipalOnlyRuntime:
    """A runtime that can verify principals but has no user store.

    Used by tests of routes that only need principal *authentication* (the
    device registry). Production always uses :class:`IdentityRuntime`.
    """

    def __init__(
        self,
        *,
        secret: str,
        now: Callable[[], float] = time.time,
        nonce_store: NonceStore | None = None,
    ) -> None:
        self.config = _SecretOnlyConfig(internal_auth_secret=secret)
        self.store: UserStore | None = None
        self.nonce_store: NonceStore = nonce_store or InMemoryNonceStore()
        self.clock = now

    def now(self) -> int:
        return int(self.clock())


class LockedIdentityRuntime:
    """What the voice agent gets when multi-user was attempted but does not validate.

    It has no secret and no store, so every verification fails and every
    session runs anonymous: nothing falls back to shared single-user state.
    """

    def __init__(self) -> None:
        self.config = _SecretOnlyConfig(internal_auth_secret="")
        self.store: UserStore | None = None
        self.nonce_store: NonceStore = InMemoryNonceStore()
        self.clock = time.time

    def now(self) -> int:
        return int(self.clock())


def principal_subject(
    header_value: str | None, runtime: Any, *, audience: str = AUDIENCE_BACKEND
) -> str | None:
    """Verify an optional principal header and return its opaque user id.

    ``None`` when no principal was presented (a legacy, unscoped caller). A
    presented principal that cannot be verified, or that names an unknown or
    suspended user, is an error: the request is never silently downgraded.
    """
    if header_value is None or not header_value.strip():
        return None
    if runtime is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_UNAUTHORIZED)
    try:
        principal = verify_principal(
            header_value.strip(),
            secret=runtime.config.internal_auth_secret,
            audience=audience,
            now=runtime.now(),
            nonce_store=runtime.nonce_store,
        )
    except PrincipalError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_UNAUTHORIZED) from exc
    if not is_valid_user_id(principal.subject):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_UNAUTHORIZED)
    store = getattr(runtime, "store", None)
    if store is not None:
        profile = store.get_user(principal.subject)
        if profile is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_UNAUTHORIZED)
        if profile.status != ACTIVE:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="suspended")
    return principal.subject


# --- response hygiene -------------------------------------------------------------


class IdentityResponseHeadersMiddleware:
    """Pure-ASGI middleware for identity routes.

    Responses (including errors) are never cached, and the app-wide permissive
    CORS policy that the LAN frontend relies on for legacy endpoints is
    stripped here: no browser origin may ever call these routes cross-site.
    """

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or not str(scope.get("path", "")).startswith(
            IDENTITY_PATH_PREFIXES
        ):
            await self._app(scope, receive, send)
            return

        async def send_with_headers(message: dict) -> None:
            if message.get("type") == "http.response.start":
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() not in (b"cache-control", b"x-content-type-options", b"pragma")
                    and not name.lower().startswith(b"access-control-")
                ]
                headers.extend(
                    [
                        (b"cache-control", b"no-store"),
                        (b"pragma", b"no-cache"),
                        (b"x-content-type-options", b"nosniff"),
                    ]
                )
                message = {**message, "headers": headers}
            await send(message)

        await self._app(scope, receive, send_with_headers)


# --- authentication dependencies ------------------------------------------------------


@dataclass(frozen=True)
class CurrentUser:
    profile: UserProfile

    @property
    def actor(self) -> Actor:
        return Actor.for_user(self.profile)


def _bearer(request: Request) -> str | None:
    scheme, _, token = (request.headers.get("authorization") or "").partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def _client_key(request: Request) -> str:
    client = request.client
    return client.host if client is not None else "unknown"


def _throttle(limiter: RateLimiter, key: str, now: float) -> None:
    if not limiter.allow(key, now=now):
        retry_after = max(1, int(limiter.retry_after(key, now=now)) + 1)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate_limited",
            headers={"Retry-After": str(retry_after)},
        )


def _identity_principal(request: Request, runtime: IdentityRuntime) -> Principal:
    """Verify the BFF's single-use ``caal-identity`` assertion, or refuse.

    Used by the routes that run *before* a user is known: sign-in, session
    lookup, and sign-out.
    """
    token = _bearer(request)
    if token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_UNAUTHORIZED)
    try:
        return verify_principal(
            token,
            secret=runtime.config.internal_auth_secret,
            audience=AUDIENCE_IDENTITY,
            now=runtime.now(),
            nonce_store=runtime.nonce_store,
        )
    except PrincipalError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_UNAUTHORIZED) from exc


def _caller_key(principal: Principal, request: Request) -> str:
    """The end user's rate-limit key, not the BFF's.

    Every browser request reaches this backend through the one BFF, so the
    peer address is the same for everybody and useless for limiting sign-in.
    The BFF therefore signs the browser's own key into the assertion; it is
    trustworthy exactly as far as the BFF is, which is the whole trust model
    here. A principal without one falls back to the peer address.
    """
    claimed = principal.claims.get("client")
    if isinstance(claimed, str) and 0 < len(claimed) <= 128 and claimed.isprintable():
        return f"login:{claimed}"
    return f"login:{_client_key(request)}"


def require_user(
    request: Request, runtime: IdentityRuntime = Depends(require_runtime)
) -> CurrentUser:
    """Authenticate a single-use backend principal and load its *active* user."""
    token = _bearer(request)
    if token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_UNAUTHORIZED)
    try:
        principal = verify_principal(
            token,
            secret=runtime.config.internal_auth_secret,
            audience=AUDIENCE_BACKEND,
            now=runtime.now(),
            nonce_store=runtime.nonce_store,
        )
    except PrincipalError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_UNAUTHORIZED) from exc
    profile = runtime.store.get_user(principal.subject)
    if profile is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_UNAUTHORIZED)
    if profile.status != ACTIVE:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="suspended")
    return CurrentUser(profile=profile)


def require_admin(user: CurrentUser = Depends(require_user)) -> CurrentUser:
    if user.profile.role != ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=_FORBIDDEN)
    return user


def throttle_mutation(
    user: CurrentUser = Depends(require_user), runtime: IdentityRuntime = Depends(require_runtime)
) -> CurrentUser:
    _throttle(runtime.mutation_limiter, user.profile.user_id, runtime.clock())
    return user


def throttle_admin_mutation(
    user: CurrentUser = Depends(require_admin), runtime: IdentityRuntime = Depends(require_runtime)
) -> CurrentUser:
    _throttle(runtime.mutation_limiter, user.profile.user_id, runtime.clock())
    return user


# --- schemas ------------------------------------------------------------------------------

Role = Literal["admin", "member"]
Status = Literal["active", "suspended"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class IdentityStatusResponse(_Strict):
    """What sign-in methods this backend offers. Reveals nothing else."""

    configured: bool
    password_login: bool = False
    cloudflare_access: bool = False


class LoginRequest(_Strict):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)


class LoginResponse(_Strict):
    user_id: str
    role: Role
    status: Status
    display_name: str
    must_change_password: bool
    session_token: str
    expires_at: int


class SessionRequest(_Strict):
    session_token: str = Field(
        min_length=MIN_SESSION_TOKEN_LENGTH, max_length=MAX_SESSION_TOKEN_LENGTH
    )


class SessionResponse(_Strict):
    user_id: str
    role: Role
    status: Status
    display_name: str
    must_change_password: bool
    expires_at: int


class LogoutRequest(_Strict):
    session_token: str = Field(min_length=1, max_length=MAX_SESSION_TOKEN_LENGTH)


class ChangePasswordRequest(_Strict):
    current_password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)
    new_password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)
    keep_session_token: str | None = Field(
        default=None, min_length=MIN_SESSION_TOKEN_LENGTH, max_length=MAX_SESSION_TOKEN_LENGTH
    )


class ResolveResponse(_Strict):
    user_id: str
    role: Role
    status: Status
    display_name: str


class ProfileResponse(_Strict):
    user_id: str
    email: str
    display_name: str
    role: Role
    status: Status
    has_callback_number: bool
    callback_number_updated_at: int | None
    created_at: int
    updated_at: int
    last_seen_at: int | None


class AdminUserResponse(ProfileResponse):
    created_by: str | None


class UsersListResponse(_Strict):
    users: list[AdminUserResponse]


class AdminCreatedUserResponse(AdminUserResponse):
    """A newly created user, plus the one-time password if one was issued."""

    one_time_password: str | None = None


class OneTimePasswordResponse(_Strict):
    user_id: str
    one_time_password: str


class ProfileUpdateRequest(_Strict):
    display_name: str = Field(min_length=1, max_length=80)


class AdminCreateUserRequest(_Strict):
    email: str = Field(min_length=3, max_length=254)
    display_name: str = Field(min_length=1, max_length=80)
    role: Role
    # When true the new account gets a random one-time password, returned to
    # the calling administrator exactly once and never stored in the clear.
    with_password: bool = False


class AdminUpdateUserRequest(_Strict):
    display_name: str | None = Field(default=None, min_length=1, max_length=80)
    role: Role | None = None
    status: Status | None = None


class CallbackNumberRequest(_Strict):
    number: str = Field(min_length=1, max_length=32)


class AuditEventResponse(_Strict):
    event_id: str
    occurred_at: int
    actor_id: str
    actor_role: str
    action: str
    target_id: str | None
    outcome: str
    detail: dict[str, Any]


class AuditListResponse(_Strict):
    events: list[AuditEventResponse]


# --- helpers ---------------------------------------------------------------------------------


def _store_errors(operation: Callable[[], UserProfile]) -> UserProfile:
    """Run a store mutation, translating its failures into bounded HTTP errors."""
    try:
        return operation()
    except UnknownUserError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found") from exc
    except DuplicateUserError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="duplicate") from exc
    except LastAdminError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="last_admin") from exc
    except CallbackNumberError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT if "already" in str(exc) else 422,
            detail="callback_number",
        ) from exc
    except NotConfiguredError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_NOT_CONFIGURED
        ) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=_FORBIDDEN) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="invalid") from exc


def _user_or_404(user_id: str) -> str:
    if not is_valid_user_id(user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")
    return user_id


# --- routes ------------------------------------------------------------------------------------

router = APIRouter(tags=["identity"])


@router.get("/identity/status", response_model=IdentityStatusResponse)
async def identity_status(
    runtime: IdentityRuntime | None = Depends(get_runtime),
) -> IdentityStatusResponse:
    """Which sign-in methods this backend offers. Reveals nothing else.

    The sign-in page needs this to know whether to render a password form, so
    it is deliberately readable without any credential -- it exposes only
    booleans an unauthenticated visitor could infer from the page anyway.
    """
    if runtime is None:
        return IdentityStatusResponse(configured=False)
    return IdentityStatusResponse(
        configured=True,
        password_login=runtime.config.password_login,
        cloudflare_access=runtime.config.access is not None,
    )


def _require_password_login(runtime: IdentityRuntime) -> None:
    """Password sign-in must be switched on; otherwise the route does not exist."""
    if not runtime.config.password_login:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")


@router.post("/auth/login", response_model=LoginResponse)
async def login(
    body: LoginRequest, request: Request, runtime: IdentityRuntime = Depends(require_runtime)
) -> LoginResponse:
    """Check a password and issue an opaque server-side session.

    Every refusal is the same ``invalid_credentials``: an unknown email, an
    account with no local password, a suspended account and a wrong password
    are indistinguishable, in body and in timing. The one exception is a
    lockout, and that is only disclosed to a caller who already supplied the
    correct password.
    """
    _require_password_login(runtime)
    principal = _identity_principal(request, runtime)
    # Throttle before any Argon2 work, so guessing cannot also be a way to
    # burn the backend's memory and CPU.
    _throttle(runtime.login_limiter, _caller_key(principal, request), runtime.clock())

    result = runtime.local_auth.authenticate(body.email, body.password, now=runtime.now())
    if not result.ok:
        if result.reason == REASON_LOCKED:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=REASON_LOCKED,
                headers={"Retry-After": str(max(1, int(result.retry_after or 1)))},
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=_INVALID_CREDENTIALS
        )
    assert result.user is not None and result.token is not None
    return LoginResponse(
        user_id=result.user.user_id,
        role=result.user.role,
        status=result.user.status,
        display_name=result.user.display_name,
        must_change_password=result.must_change_password,
        session_token=result.token,
        expires_at=int(result.expires_at or 0),
    )


@router.post("/auth/session", response_model=SessionResponse)
async def read_session(
    body: SessionRequest, request: Request, runtime: IdentityRuntime = Depends(require_runtime)
) -> SessionResponse:
    """Resolve a session token to its user, from the database, every time.

    Expiry, revocation, suspension and the forced-change flag are all re-read
    here, so a cookie can never outlive the decision that revoked it.
    """
    principal = _identity_principal(request, runtime)
    _throttle(runtime.resolve_limiter, _caller_key(principal, request), runtime.clock())

    session = runtime.local_auth.verify_session(body.session_token, now=runtime.now())
    if session is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_UNAUTHORIZED)
    return SessionResponse(
        user_id=session.user.user_id,
        role=session.user.role,
        status=session.user.status,
        display_name=session.user.display_name,
        must_change_password=session.must_change_password,
        expires_at=session.expires_at,
    )


@router.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    body: LogoutRequest, request: Request, runtime: IdentityRuntime = Depends(require_runtime)
) -> Response:
    """Revoke a session. An unknown token succeeds: sign-out is not a probe."""
    _identity_principal(request, runtime)
    runtime.local_auth.revoke_session(body.session_token, now=runtime.now())
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/auth/password", status_code=status.HTTP_204_NO_CONTENT)
async def change_own_password(
    body: ChangePasswordRequest,
    user: CurrentUser = Depends(throttle_mutation),
    runtime: IdentityRuntime = Depends(require_runtime),
) -> Response:
    """Change your own password. The principal decides whose, never the body.

    Succeeding revokes every other session that user holds, so a change made
    because a device was lost actually ends the lost device's access.
    """
    _require_password_login(runtime)
    try:
        runtime.local_auth.change_password(
            user.profile.user_id,
            body.current_password,
            body.new_password,
            keep_token=body.keep_session_token,
            now=runtime.now(),
        )
    except AccountLockedError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=REASON_LOCKED,
            headers={"Retry-After": str(max(1, exc.retry_after))},
        ) from exc
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=_INVALID_CREDENTIALS
        ) from exc
    except PasswordPolicyError as exc:
        raise HTTPException(status_code=422, detail="password_policy") from exc
    except UnknownUserError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found") from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/auth/resolve", response_model=ResolveResponse)
async def resolve_identity(
    request: Request, runtime: IdentityRuntime = Depends(require_runtime)
) -> ResolveResponse:
    """Map a verified Cloudflare Access identity to its opaque user.

    Requires a single-use ``caal-identity`` assertion from the BFF *and* the
    original ``Cf-Access-Jwt-Assertion``; the email in each must agree.

    Cloudflare Access is optional. Where it is not configured this route does
    not exist, so no half-verified Access path can ever be reached.
    """
    if runtime.access_verifier is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")
    now = runtime.clock()
    _throttle(runtime.resolve_limiter, _client_key(request), now)

    token = _bearer(request)
    if token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_UNAUTHORIZED)
    try:
        assertion = verify_principal(
            token,
            secret=runtime.config.internal_auth_secret,
            audience=AUDIENCE_IDENTITY,
            now=int(now),
            nonce_store=runtime.nonce_store,
        )
        asserted_email = normalize_email(assertion.claims.get("email"))
    except (PrincipalError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_UNAUTHORIZED) from exc

    try:
        identity = runtime.access_verifier.verify(request.headers.get("cf-access-jwt-assertion"))
    except AccessTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_UNAUTHORIZED) from exc
    if not hmac.compare_digest(identity.email.encode("utf-8"), asserted_email.encode("utf-8")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_UNAUTHORIZED)

    def adopt_legacy_memories(profile: UserProfile) -> None:
        # The deployment was single-user until this moment; its memories belong
        # to the person whose email was configured to bootstrap as administrator.
        count = memory_tools.adopt_legacy_memories(profile.user_id)
        runtime.store.record_audit(
            "memory.adopt_legacy",
            actor=Actor.system(),
            target_id=profile.user_id,
            detail={"count": count},
            now=int(now),
        )

    try:
        profile = runtime.store.resolve_identity(
            identity.email,
            bootstrap_admin_email=runtime.config.bootstrap_admin_email,
            now=int(now),
            on_bootstrap=adopt_legacy_memories,
        )
    except UnknownUserError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="no_account") from exc
    except UserSuspendedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="suspended") from exc
    except NotConfiguredError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_NOT_CONFIGURED
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_UNAUTHORIZED) from exc
    return ResolveResponse(
        user_id=profile.user_id,
        role=profile.role,
        status=profile.status,
        display_name=profile.display_name,
    )


@router.get("/users/me", response_model=ProfileResponse)
async def read_own_profile(user: CurrentUser = Depends(require_user)) -> ProfileResponse:
    return ProfileResponse(**user.profile.public_view())


@router.patch("/users/me", response_model=ProfileResponse)
async def update_own_profile(
    body: ProfileUpdateRequest,
    user: CurrentUser = Depends(throttle_mutation),
    runtime: IdentityRuntime = Depends(require_runtime),
) -> ProfileResponse:
    profile = _store_errors(
        lambda: runtime.store.update_display_name(
            user.profile.user_id, body.display_name, actor=user.actor, now=runtime.now()
        )
    )
    return ProfileResponse(**profile.public_view())


@router.get("/admin/users", response_model=UsersListResponse)
async def admin_list_users(
    _: CurrentUser = Depends(require_admin), runtime: IdentityRuntime = Depends(require_runtime)
) -> UsersListResponse:
    return UsersListResponse(
        users=[AdminUserResponse(**profile.admin_view()) for profile in runtime.store.list_users()]
    )


@router.post(
    "/admin/users", response_model=AdminCreatedUserResponse, status_code=status.HTTP_201_CREATED
)
async def admin_create_user(
    body: AdminCreateUserRequest,
    admin: CurrentUser = Depends(throttle_admin_mutation),
    runtime: IdentityRuntime = Depends(require_runtime),
) -> AdminCreatedUserResponse:
    profile = _store_errors(
        lambda: runtime.store.create_user(
            email=body.email,
            display_name=body.display_name,
            role=body.role,
            actor=admin.actor,
            now=runtime.now(),
        )
    )
    issued: str | None = None
    if body.with_password and runtime.config.password_login:
        # Shown to this administrator once, in this response, and nowhere else.
        issued = runtime.local_auth.admin_reset_password(
            profile.user_id, actor=admin.actor, now=runtime.now()
        )
    return AdminCreatedUserResponse(**profile.admin_view(), one_time_password=issued)


@router.post("/admin/users/{user_id}/password", response_model=OneTimePasswordResponse)
async def admin_reset_password(
    user_id: str,
    admin: CurrentUser = Depends(throttle_admin_mutation),
    runtime: IdentityRuntime = Depends(require_runtime),
) -> OneTimePasswordResponse:
    """Issue a random one-time password for a user, shown to this admin once.

    This is the deliberate replacement for self-service password reset: with
    no mail transport in the deployment, an emailed reset link would be a
    channel CAAL cannot secure. Resetting revokes the user's live sessions and
    forces a change at their next sign-in.
    """
    _require_password_login(runtime)
    target = _user_or_404(user_id)
    try:
        issued = runtime.local_auth.admin_reset_password(
            target, actor=admin.actor, now=runtime.now()
        )
    except UnknownUserError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=_FORBIDDEN) from exc
    return OneTimePasswordResponse(user_id=target, one_time_password=issued)


@router.get("/admin/users/{user_id}", response_model=AdminUserResponse)
async def admin_read_user(
    user_id: str,
    _: CurrentUser = Depends(require_admin),
    runtime: IdentityRuntime = Depends(require_runtime),
) -> AdminUserResponse:
    profile = runtime.store.get_user(_user_or_404(user_id))
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")
    return AdminUserResponse(**profile.admin_view())


@router.patch("/admin/users/{user_id}", response_model=AdminUserResponse)
async def admin_update_user(
    user_id: str,
    body: AdminUpdateUserRequest,
    admin: CurrentUser = Depends(throttle_admin_mutation),
    runtime: IdentityRuntime = Depends(require_runtime),
) -> AdminUserResponse:
    target = _user_or_404(user_id)
    profile = _store_errors(
        lambda: runtime.store.admin_update(
            target,
            display_name=body.display_name,
            role=body.role,
            status=body.status,
            actor=admin.actor,
            now=runtime.now(),
        )
    )
    if profile.status != ACTIVE:
        # Every session lookup re-reads status, so this is belt and braces --
        # but it makes "suspend" mean the rows are gone, not merely ignored.
        runtime.local_auth.revoke_user_sessions(target, now=runtime.now())
    return AdminUserResponse(**profile.admin_view())


@router.put("/admin/users/{user_id}/callback-number", response_model=AdminUserResponse)
async def admin_set_callback_number(
    user_id: str,
    body: CallbackNumberRequest,
    admin: CurrentUser = Depends(throttle_admin_mutation),
    runtime: IdentityRuntime = Depends(require_runtime),
) -> AdminUserResponse:
    target = _user_or_404(user_id)
    try:
        number = normalize_e164(body.number)
    except CallbackNumberError as exc:
        raise HTTPException(status_code=422, detail="callback_number") from exc
    profile = _store_errors(
        lambda: runtime.store.set_callback_number(
            target, number, actor=admin.actor, now=runtime.now()
        )
    )
    return AdminUserResponse(**profile.admin_view())


@router.delete("/admin/users/{user_id}/callback-number", response_model=AdminUserResponse)
async def admin_clear_callback_number(
    user_id: str,
    admin: CurrentUser = Depends(throttle_admin_mutation),
    runtime: IdentityRuntime = Depends(require_runtime),
) -> AdminUserResponse:
    target = _user_or_404(user_id)
    profile = _store_errors(
        lambda: runtime.store.clear_callback_number(target, actor=admin.actor, now=runtime.now())
    )
    return AdminUserResponse(**profile.admin_view())


@router.get("/admin/audit", response_model=AuditListResponse)
async def admin_list_audit(
    limit: int = 100,
    _: CurrentUser = Depends(require_admin),
    runtime: IdentityRuntime = Depends(require_runtime),
) -> AuditListResponse:
    bounded = max(1, min(int(limit), MAX_AUDIT_LIMIT))
    return AuditListResponse(
        events=[
            AuditEventResponse(**event.view())
            for event in runtime.store.list_audit_events(limit=bounded)
        ]
    )
