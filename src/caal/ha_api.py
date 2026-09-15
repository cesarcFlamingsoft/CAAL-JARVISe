"""Home Assistant grants and real browser authorization behind the authenticated BFF."""

import os
import time
from urllib.parse import urlencode, urlsplit

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from . import settings
from .ha_access import HAStore
from .ha_client import HAClient
from .user_api import (
    require_admin,
    require_runtime,
    require_user,
    throttle_admin_mutation,
    throttle_mutation,
)
from .user_scope import UserScope

router = APIRouter()
CALLBACK = "/api/home-assistant/callback"


class Grant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: StrictBool
    connection_id: str | None = Field(default=None, pattern=r"^ha_[a-f0-9]{24}$")


class Callback(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: str = Field(min_length=32, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")
    code: str = Field(min_length=1, max_length=4096)


def configured():
    s = settings.load_settings()
    if not s.get("hass_enabled"):
        raise HTTPException(503, "ha_disabled")
    try:
        endpoint = HAClient(s.get("hass_host", "")).endpoint
    except ValueError:
        raise HTTPException(503, "ha_configuration_required") from None
    return endpoint


def view(store, user_id, actor_id):
    user = store.active(user_id)
    data = store.access(UserScope.for_user(user))
    return {
        **data,
        "connections": store.choices(actor_id, user_id),
        "scope": "states_and_lights",
        "can_connect": user_id == actor_id,
    }


@router.get("/users/me/home-assistant")
async def own(user=Depends(require_user), identity=Depends(require_runtime)):
    return view(HAStore(identity), user.profile.user_id, user.profile.user_id)


@router.get("/admin/users/{user_id}/home-assistant")
async def get_grant(user_id: str, user=Depends(require_admin), identity=Depends(require_runtime)):
    try:
        return view(HAStore(identity), user_id, user.profile.user_id)
    except PermissionError:
        raise HTTPException(404, "not_found") from None


@router.put("/admin/users/{user_id}/home-assistant")
async def put_grant(
    user_id: str,
    body: Grant,
    user=Depends(throttle_admin_mutation),
    identity=Depends(require_runtime),
):
    store = HAStore(identity)
    try:
        store.grant(user.profile.user_id, user_id, **body.model_dump())
        return view(store, user_id, user.profile.user_id)
    except (ValueError, PermissionError):
        raise HTTPException(403, "ha_assignment_denied") from None


@router.post("/users/me/home-assistant/authorize")
async def authorize(user=Depends(throttle_mutation), identity=Depends(require_runtime)):
    endpoint = configured()
    origin = os.getenv("CAAL_PUBLIC_ORIGIN", "").rstrip("/")
    parts = urlsplit(origin)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.path
        or parts.query
        or parts.fragment
        or parts.username
    ):
        raise HTTPException(503, "ha_callback_configuration_required")
    store = HAStore(identity)
    state = store.start_state(user.profile.user_id, endpoint, origin)
    return {
        "authorization_url": endpoint
        + "/auth/authorize?"
        + urlencode({"client_id": origin, "redirect_uri": origin + CALLBACK, "state": state}),
        "state_id": state,
        "expires_at": int(time.time()) + 600,
    }


@router.post("/users/me/home-assistant/callback")
async def callback(
    body: Callback, user=Depends(throttle_mutation), identity=Depends(require_runtime)
):
    store = HAStore(identity)
    try:
        state = store.consume_state(user.profile.user_id, body.state)
        if state["endpoint"] != configured():
            raise ValueError("configuration_changed")
        client = HAClient(state["endpoint"])
        tokens = await client.request(
            "POST",
            "/auth/token",
            form={
                "grant_type": "authorization_code",
                "code": body.code,
                "client_id": state["client_id"],
            },
        )
        if not isinstance(tokens, dict) or not isinstance(tokens.get("refresh_token"), str):
            raise ValueError("invalid_tokens")
        account = await client.current_user(tokens["access_token"])
        store.save_connection(
            user.profile.user_id,
            account,
            tokens,
            endpoint=state["endpoint"],
            client_id=state["client_id"],
        )
        identity.store.record_audit("ha.connect", actor=user.actor, target_id=user.profile.user_id)
        return {"status": "connected"}
    except Exception:
        raise HTTPException(400, "ha_authorization_failed") from None


@router.delete("/users/me/home-assistant/connections/{connection_id}")
async def disconnect(
    connection_id: str, user=Depends(throttle_mutation), identity=Depends(require_runtime)
):
    HAStore(identity).disconnect(user.profile.user_id, connection_id)
    identity.store.record_audit("ha.disconnect", actor=user.actor, target_id=user.profile.user_id)
    return {"status": "disconnected"}
