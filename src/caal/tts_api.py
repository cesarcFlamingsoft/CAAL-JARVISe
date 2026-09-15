"""Authenticated, narrow TTS selection using the existing user/admin boundary."""

import uuid
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import settings as settings_module
from .tts_selection import trial_config
from .tts_store import TTSStore
from .user_api import (
    CurrentUser,
    IdentityRuntime,
    get_runtime,
    require_admin,
    throttle_admin_mutation,
    throttle_mutation,
)
from .voicebox import VoiceboxClient, checked_origin, verify_profile

router = APIRouter(tags=["tts"])
HEALTH_TRANSPORT = None
VOICEBOX_TRANSPORT = None


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["kokoro", "piper", "qwen-trial", "voicebox"]
    profile_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]{1,100}$")
    engine: (
        Literal[
            "qwen",
            "qwen_custom_voice",
            "luxtts",
            "chatterbox",
            "chatterbox_turbo",
            "tada",
            "kokoro",
        ]
        | None
    ) = None
    model_size: Literal["1.7B", "0.6B", "1B", "3B"] | None = None

    @model_validator(mode="after")
    def profile_required(self):
        if self.provider == "voicebox" and not all((self.profile_id, self.engine, self.model_size)):
            raise ValueError("Choose a Voicebox profile, engine and model size")
        if self.provider != "voicebox" and any((self.profile_id, self.engine, self.model_size)):
            raise ValueError("Profile options require Voicebox")
        return self


async def view(user=None, identity=None):
    settings = settings_module.load_settings()
    preference = TTSStore(identity).preference(user.profile.user_id) if identity else None
    config = TTSStore(identity).config() if identity else None
    status = "not_configured"
    if config:
        try:
            await verify_voicebox(config)
            status = (
                "configuration_changed"
                if preference
                and preference.get("provider") == "voicebox"
                and preference.get("config_revision") != config["revision"]
                else "api_verified"
            )
        except Exception:
            status = "unavailable"
    return {
        "voicebox_status": status,
        "can_configure": bool(user and user.profile.role == "admin"),
        "profile_id": preference.get("profile_id") if preference else None,
        "engine": preference.get("engine") if preference else None,
        "model_size": preference.get("model_size") if preference else None,
        "provider": preference["provider"]
        if preference
        else settings.get("tts_provider", "kokoro"),
        "source": "personal" if preference else "default",
        "qwen_configured": trial_config() is not None,
        "qwen_voice": "jarvis-designed",
        "applies_to": "new_sessions",
    }


@router.get("/users/me/tts")
async def read_tts(
    user: CurrentUser = Depends(throttle_mutation), identity: IdentityRuntime = Depends(get_runtime)
):
    return await view(user, identity)


@router.put("/users/me/tts")
async def write_tts(
    body: Selection,
    user: CurrentUser = Depends(throttle_mutation),
    identity: IdentityRuntime = Depends(get_runtime),
):
    if body.provider == "qwen-trial":
        config = trial_config()
        if config is None:
            raise HTTPException(503, "qwen_trial_unavailable")
        try:
            async with httpx.AsyncClient(
                timeout=3, trust_env=False, transport=HEALTH_TRANSPORT
            ) as c:
                response = await c.get(
                    config["endpoint"] + "/health",
                    headers={
                        "Authorization": "Bearer " + config["token"],
                    },
                )
                response.raise_for_status()
                if response.json().get("status") != "ok":
                    raise ValueError("Unhealthy")
        except Exception:
            raise HTTPException(503, "qwen_trial_unavailable") from None
    value = body.model_dump()
    if body.provider == "voicebox":
        config = TTSStore(identity).config()
        if not config:
            raise HTTPException(503, "voicebox_unavailable")
        try:
            await verify_voicebox(config)
        except Exception:
            raise HTTPException(503, "voicebox_unavailable") from None
        try:
            async with httpx.AsyncClient(
                transport=VOICEBOX_TRANSPORT, trust_env=False, timeout=3
            ) as http:
                await verify_profile(
                    VoiceboxClient(config["endpoint"], config["credential"], client=http), body
                )
        except Exception:
            raise HTTPException(422, "voicebox_profile_unavailable") from None
        value["config_revision"] = config["revision"]
    TTSStore(identity).save_preference(user.profile.user_id, value)
    return await view(user, identity)


class VoiceboxConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    endpoint: str = Field(max_length=200)
    credential: str | None = Field(None, max_length=4096)


def config_view(config):
    return {
        "endpoint": config["endpoint"] if config else "",
        "credential_configured": bool(config and config["credential"]),
    }


async def verify_voicebox(config):
    async with httpx.AsyncClient(transport=VOICEBOX_TRANSPORT, trust_env=False, timeout=3) as http:
        client = VoiceboxClient(config["endpoint"], config["credential"], client=http)
        health = await client.json("/health")
        schema = await client.json("/openapi.json")
        if health.get("status") != "ok" or not isinstance(health.get("model_loaded"), bool):
            raise ValueError("invalid_voicebox_health")
        ref = schema["paths"]["/generate/stream"]["post"]["requestBody"]["content"][
            "application/json"
        ]["schema"]["$ref"]
        props = schema["components"]["schemas"]["GenerationRequest"]["properties"]
        if ref != "#/components/schemas/GenerationRequest" or not {
            "profile_id",
            "text",
            "engine",
            "model_size",
            "personality",
        }.issubset(props):
            raise ValueError("invalid_voicebox_schema")
        if "/profiles/{profile_id}" not in schema["paths"]:
            raise ValueError("invalid_voicebox_schema")


def candidate_config(body, store):
    try:
        endpoint = checked_origin(body.endpoint)
        previous = store.config()
        if body.credential is None:
            # Never forward old credentials to a changed destination.
            credential = (
                previous["credential"] if previous and previous["endpoint"] == endpoint else ""
            )
        else:
            credential = body.credential
        if any(ord(c) < 33 or ord(c) > 126 for c in credential):
            raise ValueError("invalid credential")
        return {
            "endpoint": endpoint,
            "credential": credential,
            "revision": previous["revision"]
            if previous
            and previous["endpoint"] == endpoint
            and previous["credential"] == credential
            else uuid.uuid4().hex,
        }
    except ValueError:
        raise HTTPException(422, "invalid_voicebox_config") from None


@router.get("/users/me/tts/voicebox")
def read_voicebox(
    user: CurrentUser = Depends(require_admin), identity: IdentityRuntime = Depends(get_runtime)
):
    return config_view(TTSStore(identity).config())


@router.post("/users/me/tts/voicebox/test")
async def test_voicebox(
    body: VoiceboxConfiguration,
    user: CurrentUser = Depends(throttle_admin_mutation),
    identity: IdentityRuntime = Depends(get_runtime),
):
    config = candidate_config(body, TTSStore(identity))
    try:
        await verify_voicebox(config)
    except Exception:
        raise HTTPException(503, "voicebox_unavailable") from None
    return {**config_view(config), "status": "api_verified", "synthesis_tested": False}


@router.put("/users/me/tts/voicebox")
async def save_voicebox(
    body: VoiceboxConfiguration,
    user: CurrentUser = Depends(throttle_admin_mutation),
    identity: IdentityRuntime = Depends(get_runtime),
):
    store = TTSStore(identity)
    config = candidate_config(body, store)
    try:
        await verify_voicebox(config)
    except Exception:
        raise HTTPException(503, "voicebox_unavailable") from None
    store.save_config(config)
    return config_view(store.config())
