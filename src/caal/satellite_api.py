"""Admin enrollment and independent non-person satellite text/audio credentials."""

import asyncio
import json
from contextlib import aclosing
from typing import Literal
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from caal.ha_access import HAStore
from caal.satellite import MAX_AUDIO, MAX_TEXT, SatelliteStore, TurnEngine
from caal.satellite_home import discover
from caal.user_api import require_admin, require_runtime, throttle_admin_mutation

router = APIRouter()
HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


class Enrollment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    satellite_id: str = Field(pattern=r"^assist_satellite\.[a-z0-9_]{1,150}$")
    connection_id: str = Field(pattern=r"^ha_[a-f0-9]{24}$")


class PermissionContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    connection_id: str = Field(pattern=r"^ha_[a-f0-9]{24}$")
    scope: Literal["conversation", "states", "states_and_lights"]


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conversation_id: UUID
    request_id: UUID
    text: str = Field(min_length=1, max_length=MAX_TEXT)
    satellite_id: str = Field(max_length=160)
    device_id: str = Field(max_length=64)


class Speech(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=MAX_TEXT)


def store(identity=Depends(require_runtime)):
    return SatelliteStore(identity)


def device(authorization: str = Header(default=""), storage=Depends(store)):
    try:
        if not authorization.startswith("Bearer "):
            raise PermissionError()
        return storage.authenticate(authorization[7:])
    except PermissionError:
        raise HTTPException(401, "satellite_unauthorized") from None


def engine(storage=Depends(store)):
    identity = storage.identity
    if not hasattr(identity, "_satellite_engine"):
        identity._satellite_engine = TurnEngine(storage)
    return identity._satellite_engine


@router.get("/admin/satellites")
async def status(
    response: Response,
    connection_id: str | None = None,
    admin=Depends(require_admin),
    storage=Depends(store),
):
    from contextlib import closing

    response.headers.update(HEADERS)
    actor_id = admin.profile.user_id
    choices = HAStore(storage.identity).choices(actor_id, actor_id)
    devices, account, error = [], None, None
    if connection_id:
        try:
            devices, account = await discover(storage, connection_id, actor_id=actor_id)
        except Exception:
            error = "The selected HA connection could not verify the voice device registry."
    with closing(storage.identity.store.connect()) as db:
        enrollments = [
            dict(r)
            for r in db.execute(
                "SELECT id,satellite_id,device_id,connection_id,scope,active "
                "FROM satellite_identities "
                "WHERE active=1 ORDER BY satellite_id"
            )
        ]
    by_device = {r["satellite_id"]: r for r in enrollments}
    for row in devices:
        row["enrollment"] = by_device.get(row["satellite_id"])
    return {
        "devices": devices,
        "enrollments": enrollments,
        "connections": choices,
        "connection_id": connection_id,
        "provider_identity": account,
        "personal_data": False,
        "error": error,
    }


@router.post("/admin/satellites")
async def enroll(
    body: Enrollment,
    response: Response,
    admin=Depends(throttle_admin_mutation),
    storage=Depends(store),
):
    response.headers.update(HEADERS)
    try:
        await discover(storage, body.connection_id, actor_id=admin.profile.user_id)
        result = storage.enroll(admin.profile.user_id, **body.model_dump())
    except Exception:
        raise HTTPException(403, "satellite_registry_or_connection_required") from None
    storage.identity.store.record_audit(
        "satellite.enroll", actor=admin.actor, target_id=admin.profile.user_id
    )
    return result


@router.put("/admin/satellites/{sid}")
async def configure(
    sid: str,
    body: PermissionContract,
    admin=Depends(throttle_admin_mutation),
    storage=Depends(store),
    turns=Depends(engine),
):
    try:
        await discover(storage, body.connection_id, actor_id=admin.profile.user_id)
        storage.configure(admin.profile.user_id, sid, **body.model_dump())
    except Exception:
        raise HTTPException(403, "satellite_configuration_denied") from None
    active = turns.active.get(sid)
    if active:
        active[1].cancel()
    for mapping in (turns.histories, turns.receipts):
        for key in list(mapping):
            if key[0] == sid:
                del mapping[key]
    storage.identity.store.record_audit(
        "satellite.permissions", actor=admin.actor, target_id=admin.profile.user_id
    )
    return {"status": "configured"}


@router.delete("/admin/satellites/{sid}")
async def revoke(
    sid: str, admin=Depends(throttle_admin_mutation), storage=Depends(store), turns=Depends(engine)
):
    storage.revoke(admin.profile.user_id, sid)
    active = turns.active.get(sid)
    if active:
        active[1].cancel()
    for mapping in (turns.histories, turns.receipts):
        for key in list(mapping):
            if key[0] == sid:
                del mapping[key]
    storage.identity.store.record_audit(
        "satellite.revoke", actor=admin.actor, target_id=admin.profile.user_id
    )
    return {"status": "revoked"}


@router.get("/satellite/v1/identity")
async def identity(response: Response, principal=Depends(device), storage=Depends(store)):
    response.headers.update(HEADERS)
    return {
        **vars(principal),
        "personal_data": False,
        "device_actions": storage.permissions(principal)["scope"] == "states_and_lights",
        "scope": storage.permissions(principal)["scope"],
        "protocol": 1,
        "audio": "pcm_s16le_24000_mono",
    }


@router.post("/satellite/v1/turn")
async def turn(body: Turn, principal=Depends(device), turns=Depends(engine)):
    args = body.model_dump(mode="json")
    stream = turns.stream(principal, **args)
    # Validate and obtain the first chunk before sending successful headers.
    try:
        first = await anext(stream)
    except PermissionError:
        raise HTTPException(403, "satellite_denied") from None
    except ValueError:
        raise HTTPException(409, "satellite_turn_conflict") from None
    except Exception:
        raise HTTPException(503, "satellite_unavailable") from None

    async def frames():
        async with aclosing(stream):
            yield json.dumps({"text": first}) + "\n"
            try:
                async for text in stream:
                    yield json.dumps({"text": text}) + "\n"
                yield '{"done":true}\n'
            except Exception:
                yield '{"error":"satellite_unavailable"}\n'

    return StreamingResponse(frames(), media_type="application/x-ndjson", headers=HEADERS)


@router.delete("/satellite/v1/turn/{request_id}")
async def cancel(request_id: UUID, principal=Depends(device), turns=Depends(engine)):
    await turns.cancel(principal, str(request_id))
    return {"status": "cancelled"}


async def qwen_audio(text, principal, storage, *, config=None, transport=None):
    from caal.ha_client import HAClient
    from caal.tts_selection import trial_config

    config = config or trial_config()
    if not config:
        raise ValueError("qwen_not_configured")
    target = HAClient(config["endpoint"])
    url, headers = target.target("/v1/audio/speech")
    headers["Authorization"] = "Bearer " + config["token"]
    async with (
        asyncio.timeout(60),
        httpx.AsyncClient(
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(15, connect=2),
            limits=httpx.Limits(max_connections=1),
        ) as client,
    ):
        async with client.stream(
            "POST",
            url,
            headers=headers,
            json={
                "input": text,
                "model": "qwen-trial",
                "voice": "jarvis-designed",
                "response_format": "pcm",
            },
        ) as response:
            if (
                response.status_code != 200
                or response.headers.get("x-audio-sample-rate") != "24000"
                or response.headers.get("x-audio-channels") != "1"
                or response.headers.get("content-type", "").split(";")[0] != "audio/pcm"
            ):
                raise ValueError("qwen_unavailable")
            size = 0
            async for chunk in response.aiter_bytes(4096):
                storage.check(principal)
                size += len(chunk)
                if size > MAX_AUDIO:
                    raise ValueError("audio_too_large")
                yield chunk
            if size == 0 or size % 2:
                raise ValueError("invalid_pcm")


@router.post("/satellite/v1/audio")
async def audio(body: Speech, principal=Depends(device), storage=Depends(store)):
    # No waiting queue: reject parallel synthesis rather than growing buffers.
    identity = storage.identity
    if not hasattr(identity, "_satellite_audio_busy"):
        identity._satellite_audio_busy = set()
    if principal.id in identity._satellite_audio_busy:
        raise HTTPException(429, "satellite_audio_busy")
    identity._satellite_audio_busy.add(principal.id)
    stream = qwen_audio(body.text, principal, storage)
    try:
        first = await anext(stream)
    except BaseException:
        identity._satellite_audio_busy.discard(principal.id)
        await stream.aclose()
        raise HTTPException(503, "satellite_audio_unavailable") from None

    async def data():
        try:
            async with aclosing(stream):
                yield first
                async for chunk in stream:
                    yield chunk
        finally:
            identity._satellite_audio_busy.discard(principal.id)

    return StreamingResponse(
        data(),
        media_type="audio/pcm",
        headers={
            **HEADERS,
            "X-Audio-Sample-Rate": "24000",
            "X-Audio-Channels": "1",
        },
    )


class SatelliteBodyLimit:
    """Bound request bodies before FastAPI parses JSON, including chunked requests."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/satellite/"):
            return await self.app(scope, receive, send)
        data = bytearray()
        async with asyncio.timeout(10):
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                data.extend(message.get("body", b""))
                if len(data) > 16384:
                    return await Response(status_code=413)(scope, receive, send)
                if not message.get("more_body", False):
                    break
        consumed = False

        async def bounded_receive():
            nonlocal consumed
            if consumed:
                return await receive()
            consumed = True
            return {"type": "http.request", "body": bytes(data), "more_body": False}

        return await self.app(scope, bounded_receive, send)
