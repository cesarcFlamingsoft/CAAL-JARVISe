"""Choosing the local model JARVIS runs on, from the settings UI.

``GET /users/me/local-model``
    the endpoint and model in force, plus what the routing actually does with
    them, and when a change takes effect.
``POST /users/me/local-model/models``
    the models installed at an endpoint, read from its ``/api/tags``. The
    endpoint is the one being typed into the form, or the saved one when the
    body is empty. Administrators only.
``PUT /users/me/local-model``
    save an endpoint and a model. Administrators only.

These are deployment-wide settings rather than a personal preference, so a
member may see what JARVIS is running on and an administrator may change it.
Both sit under ``/users/me`` and inherit the identity boundary of
:mod:`caal.user_api`: a single-use ``caal-backend`` principal from the BFF
names the user, the user is loaded from the database on every call, and the
identity middleware makes every response uncacheable and free of the app-wide
CORS policy.

The browser never reaches an Ollama itself: it asks this backend, which is
the only thing that opens a socket, and only to an endpoint
:mod:`caal.local_ollama` has already accepted as local. A refusal is a short
code and a sentence -- never an upstream body, an exception string, or
anything else out of the settings file.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict

from . import settings as settings_module
from . import user_api
from .local_ollama import (
    DISCOVERY_TIMEOUT,
    DiscoveryError,
    EndpointError,
    configured_endpoint,
    discover_models,
    is_model_name,
    normalize_endpoint,
)
from .user_api import CurrentUser, IdentityRuntime, require_user, throttle_admin_mutation

logger = logging.getLogger(__name__)

__all__ = [
    "LocalModelRuntime",
    "get_local_model_runtime",
    "require_local_model",
    "router",
]

_NOT_CONFIGURED = "Multi-user identity is not configured on this CAAL backend."
UNPROCESSABLE = 422
BAD_GATEWAY = 502

# What the routed provider actually does with these settings. Fixed strings:
# the UI describes the routing rather than offering a provider to pick.
ROUTING = dict(primary="ollama", escalation="hermes", coding="hermes_delegation")


# --- runtime -----------------------------------------------------------------------------


class LocalModelRuntime:
    """The identity runtime plus the settings file and one bounded HTTP client.

    Tests build one with an in-memory settings pair and a scripted transport;
    production builds one from the real settings module.
    """

    def __init__(
        self,
        identity: IdentityRuntime,
        *,
        load: Callable[[], dict[str, Any]] = settings_module.load_settings,
        save: Callable[[dict[str, Any]], None] = settings_module.save_settings,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.identity = identity
        self._load = load
        self._save = save
        self._transport = transport

    def settings(self) -> dict[str, Any]:
        return self._load() or {}

    def endpoint(self) -> str:
        return configured_endpoint(self.settings())

    def model(self) -> str:
        model = self.settings().get("ollama_model")
        return model if is_model_name(model) else ""

    def save(self, *, endpoint: str, model: str) -> None:
        # This is a narrow settings mutation.  Preserve every unrelated runtime
        # setting (identity, provider connections, audio, routing, etc.) rather
        # than replacing the settings document with these two values.
        updated = self.settings()
        updated["ollama_host"] = endpoint
        updated["ollama_model"] = model
        self._save(updated)

    async def models_at(self, endpoint: str) -> list[str]:
        client = httpx.AsyncClient(
            timeout=DISCOVERY_TIMEOUT, trust_env=False, transport=self._transport
        )
        try:
            return await discover_models(endpoint, client=client)
        finally:
            await client.aclose()


_runtime_lock = threading.Lock()
_cached: tuple[IdentityRuntime, LocalModelRuntime] | None = None


def get_local_model_runtime(
    identity: IdentityRuntime | None = Depends(user_api.get_runtime),
) -> LocalModelRuntime | None:
    """The process-wide runtime, or ``None`` while multi-user identity is unconfigured."""
    global _cached
    if identity is None:
        return None
    with _runtime_lock:
        if _cached is None or _cached[0] is not identity:
            _cached = (identity, LocalModelRuntime(identity))
        return _cached[1]


def reset_local_model_runtime() -> None:
    global _cached
    with _runtime_lock:
        _cached = None


def require_local_model(
    runtime: LocalModelRuntime | None = Depends(get_local_model_runtime),
) -> LocalModelRuntime:
    if runtime is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_NOT_CONFIGURED)
    return runtime


# --- schemas -----------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class RoutingView(_Strict):
    """What each kind of turn is answered by. Describes routing, does not set it."""

    primary: str
    escalation: str
    coding: str


class LocalModelResponse(_Strict):
    endpoint: str
    model: str
    local_only: bool
    applies_to: str
    routing: RoutingView


class DiscoverRequest(_Strict):
    endpoint: str | None = None


class DiscoverResponse(_Strict):
    endpoint: str
    models: list[str]


class SaveRequest(_Strict):
    endpoint: str
    model: str


# --- helpers -----------------------------------------------------------------------------


def _view(runtime: LocalModelRuntime) -> LocalModelResponse:
    return LocalModelResponse(
        endpoint=runtime.endpoint(),
        model=runtime.model(),
        local_only=True,
        # A provider is built when a session starts, so a change is picked up
        # by the next session rather than by a call already in progress.
        applies_to="new_sessions",
        routing=RoutingView(**ROUTING),
    )


def _refused(exc: EndpointError) -> HTTPException:
    return HTTPException(status_code=UNPROCESSABLE, detail=exc.code)


def _upstream(exc: DiscoveryError) -> HTTPException:
    return HTTPException(status_code=BAD_GATEWAY, detail=exc.code)


# --- routes ------------------------------------------------------------------------------

router = APIRouter(tags=["local-model"])


@router.get("/users/me/local-model", response_model=LocalModelResponse)
def read_local_model(
    user: CurrentUser = Depends(require_user),
    runtime: LocalModelRuntime = Depends(require_local_model),
) -> LocalModelResponse:
    return _view(runtime)


@router.post("/users/me/local-model/models", response_model=DiscoverResponse)
async def discover(
    body: DiscoverRequest,
    user: CurrentUser = Depends(throttle_admin_mutation),
    runtime: LocalModelRuntime = Depends(require_local_model),
) -> DiscoverResponse:
    raw = body.endpoint if body.endpoint else runtime.endpoint()
    try:
        endpoint = normalize_endpoint(raw)
    except EndpointError as exc:
        raise _refused(exc) from exc
    try:
        models = await runtime.models_at(endpoint)
    except EndpointError as exc:
        raise _refused(exc) from exc
    except DiscoveryError as exc:
        raise _upstream(exc) from exc
    return DiscoverResponse(endpoint=endpoint, models=models)


@router.put("/users/me/local-model", response_model=LocalModelResponse)
def save_local_model(
    body: SaveRequest,
    user: CurrentUser = Depends(throttle_admin_mutation),
    runtime: LocalModelRuntime = Depends(require_local_model),
) -> LocalModelResponse:
    try:
        endpoint = normalize_endpoint(body.endpoint)
    except EndpointError as exc:
        raise _refused(exc) from exc
    if not is_model_name(body.model):
        raise HTTPException(status_code=UNPROCESSABLE, detail="invalid_model")
    runtime.save(endpoint=endpoint, model=body.model)
    logger.info("Local model settings updated by an administrator")
    return _view(runtime)
