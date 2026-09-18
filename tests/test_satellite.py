"""Restricted satellite boundaries; synthetic credentials only."""

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient
from test_local_model_api import Harness

from caal import user_api, webhooks
from caal.llm.llm_node import _discover_tools, _execute_single_tool
from caal.satellite import SatelliteStore, TurnEngine, restricted_agent

PILOT = "assist_satellite.home_assistant_voice_0a3d6b_assist_satellite"
DEVICE = "0bf018dfe200b28d8f7cff95e8d2aa75"


def issue(store, actor):
    cid = "ha_" + "a" * 24
    store.update_inventory(cid, [{"satellite_id": PILOT, "device_id": DEVICE, "name": "Bedroom"}])
    return store.enroll(actor, satellite_id=PILOT, connection_id=cid)


@pytest.fixture
def identity(tmp_path):
    return Harness(tmp_path)


def test_enrollment_admin_revocation_and_persistence(identity):
    store = SatelliteStore(identity.identity)
    with pytest.raises(PermissionError):
        issue(store, identity.member)
    issued = issue(store, identity.admin)
    assert len(issued["credential"]) >= 43
    principal = SatelliteStore(identity.identity).authenticate(issued["credential"])
    assert principal.satellite_id == PILOT and principal.device_id == DEVICE
    assert not hasattr(principal, "user_id")
    with pytest.raises(PermissionError):
        store.authenticate("x" * 43)
    store.revoke(identity.admin, principal.id)
    with pytest.raises(PermissionError):
        store.authenticate(issued["credential"])


@pytest.mark.asyncio
async def test_tool_surface_and_execution_cannot_fall_back():
    agent = restricted_agent()
    catalog = await _discover_tools(agent)
    assert [x["function"]["name"] for x in catalog] == ["satellite.capabilities"]
    for name in [
        "email.search",
        "calendar.list",
        "memory.search",
        "hass_assist",
        "delegate_to_hermes",
        "coding.execute",
        "alarms.set",
    ]:
        result = await _execute_single_tool(agent, name, {"user_id": "forged"})
        assert result["status"] == "unauthorized"
    assert (await _execute_single_tool(agent, "satellite.capabilities", {}))["data"][
        "personal_data"
    ] is False


@pytest.mark.asyncio
async def test_turn_isolation_idempotency_cancellation_and_bounds(identity):
    store = SatelliteStore(identity.identity)
    issued = issue(store, identity.admin)
    p = store.authenticate(issued["credential"])
    calls = []

    async def generate(history, text):
        calls.append((list(history), text))
        yield "Synthetic answer."

    engine = TurnEngine(store, generate=generate)
    cid, rid = str(uuid.uuid4()), str(uuid.uuid4())

    async def consume(**changes):
        args = dict(
            conversation_id=cid, request_id=rid, text="Hello", satellite_id=PILOT, device_id=DEVICE
        )
        args.update(changes)
        return "".join([x async for x in engine.stream(p, **args)])

    assert await consume() == await consume() == "Synthetic answer."
    assert len(calls) == 1
    with pytest.raises(ValueError):
        await consume(text="changed")
    with pytest.raises(PermissionError):
        await consume(device_id="livingroom")
    await consume(conversation_id=str(uuid.uuid4()), request_id=str(uuid.uuid4()))
    assert calls[-1][0] == []
    with pytest.raises(ValueError):
        await consume(text="x" * 4097)
    await engine.cancel(p, rid)
    with pytest.raises(ValueError):
        await consume()
    store.revoke(identity.admin, p.id)
    with pytest.raises(PermissionError):
        await consume(request_id=str(uuid.uuid4()))


def test_http_enrollment_requires_real_admin_and_device_auth(identity, monkeypatch):
    from caal import satellite_api

    async def verified(storage, connection_id, *, actor_id):
        storage.update_inventory(
            connection_id, [{"satellite_id": PILOT, "device_id": DEVICE, "name": "Bedroom"}]
        )
        return [], {}

    monkeypatch.setattr(satellite_api, "discover", verified)
    body = {"satellite_id": PILOT, "connection_id": "ha_" + "a" * 24}
    webhooks.app.dependency_overrides[user_api.get_runtime] = lambda: identity.identity
    try:
        with TestClient(webhooks.app) as c:
            path = "/admin/satellites"
            assert c.post(path, json={}).status_code == 401
            assert (
                c.post(path, headers=identity.bearer(identity.member), json={}).status_code == 403
            )
            r = c.post(path, headers=identity.bearer(identity.admin), json=body)
            assert r.status_code == 200
            token = r.json()["credential"]
            assert c.get("/satellite/v1/identity").status_code == 401
            assert (
                c.get(
                    "/satellite/v1/identity", headers={"Authorization": "Bearer " + token}
                ).json()["personal_data"]
                is False
            )
            assert (
                c.post(
                    "/satellite/v1/turn",
                    headers={"Authorization": "Bearer " + token},
                    json={"user_id": "fake"},
                ).status_code
                == 422
            )
            assert (
                c.delete(
                    path + "/" + r.json()["id"], headers=identity.bearer(identity.admin)
                ).status_code
                == 200
            )
            assert (
                c.get(
                    "/satellite/v1/identity", headers={"Authorization": "Bearer " + token}
                ).status_code
                == 401
            )
    finally:
        webhooks.app.dependency_overrides.pop(user_api.get_runtime, None)


@pytest.mark.asyncio
async def test_real_loop_uses_prompt_and_bounded_local_transport(monkeypatch):
    import httpx

    from caal import satellite_model, settings
    from caal.satellite import generate_reply

    requests = []

    def handle(req):
        import json

        body = json.loads(req.content)
        requests.append(body)
        if req.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion", "tools"]})
        assert body["model"] == "fixture-local"
        assert body["options"]["num_predict"] == 1024
        assert body["options"]["temperature"] == 0.0
        assert "Actual fixture prompt" in body["messages"][0]["content"]
        assert (
            "Only use an exact entity_id returned by home.states"
            in body["messages"][0]["content"]
        )
        if len(requests) == 2:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "satellite.capabilities", "arguments": {}}}
                        ],
                    }
                },
            )
        assert body["messages"][-1]["role"] == "tool"
        return httpx.Response(
            200,
            content=(
                '{"message":{"role":"assistant","content":"Personal data is disabled."},'
                '"done":true}\n'
            ),
        )

    monkeypatch.setattr(
        settings,
        "load_settings",
        lambda: {"ollama_model": "fixture-local", "ollama_host": "http://127.0.0.1:11434"},
    )
    monkeypatch.setattr(settings, "load_prompt_with_context", lambda: "Actual fixture prompt")
    original = satellite_model.SatelliteModel
    monkeypatch.setattr(
        satellite_model, "SatelliteModel", lambda: original(transport=httpx.MockTransport(handle))
    )
    result = "".join([s async for s in generate_reply([], "What can this pilot do?")])
    assert result == "Personal data is disabled."
    assert len(requests) == 3


@pytest.mark.asyncio
async def test_audio_auth_format_bound_and_revocation(identity):
    import httpx

    from caal.satellite_api import qwen_audio

    storage = SatelliteStore(identity.identity)
    token = issue(storage, identity.admin)["credential"]
    principal = storage.authenticate(token)

    def handle(req):
        assert req.headers["authorization"] == "Bearer " + "q" * 43
        return httpx.Response(
            200,
            content=b"\1\2" * 100,
            headers={
                "Content-Type": "audio/pcm",
                "X-Audio-Sample-Rate": "24000",
                "X-Audio-Channels": "1",
            },
        )

    args = dict(
        config={"endpoint": "http://127.0.0.1:18003", "token": "q" * 43},
        transport=httpx.MockTransport(handle),
    )
    assert (
        len(b"".join([x async for x in qwen_audio("Synthetic test", principal, storage, **args)]))
        == 200
    )
    storage.revoke(identity.admin, principal.id)
    with pytest.raises(PermissionError):
        _ = [x async for x in qwen_audio("Synthetic test", principal, storage, **args)]


@pytest.mark.asyncio
async def test_active_cancel_stops_generation_and_does_not_commit_history(identity):
    storage = SatelliteStore(identity.identity)
    principal = storage.authenticate(issue(storage, identity.admin)["credential"])
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def generate(history, text):
        try:
            started.set()
            await asyncio.Event().wait()
            yield "never"
        finally:
            stopped.set()

    engine = TurnEngine(storage, generate=generate)
    rid = str(uuid.uuid4())

    async def consume():
        return [
            x
            async for x in engine.stream(
                principal,
                conversation_id=str(uuid.uuid4()),
                request_id=rid,
                text="test",
                satellite_id=PILOT,
                device_id=DEVICE,
            )
        ]

    task = asyncio.create_task(consume())
    await started.wait()
    await engine.cancel(principal, rid)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set() and not engine.histories and not engine.active


def test_admin_status_no_credentials_and_body_bound(identity):
    webhooks.app.dependency_overrides[user_api.get_runtime] = lambda: identity.identity
    try:
        with TestClient(webhooks.app) as c:
            assert c.get("/admin/satellites").status_code == 401
            assert (
                c.get("/admin/satellites", headers=identity.bearer(identity.member)).status_code
                == 403
            )
            assert (
                c.get("/admin/satellites", headers=identity.bearer(identity.admin)).json()[
                    "enrollments"
                ]
                == []
            )
            result = issue(SatelliteStore(identity.identity), identity.admin)
            status = c.get("/admin/satellites", headers=identity.bearer(identity.admin))
            assert (
                status.json()["enrollments"][0]["id"] == result["id"]
                and "credential" not in status.text
            )
            huge = c.post("/satellite/v1/turn", content=b" " * 17000)
            assert huge.status_code == 413
    finally:
        webhooks.app.dependency_overrides.pop(user_api.get_runtime, None)


@pytest.mark.asyncio
async def test_model_metadata_has_separate_bounded_budget(monkeypatch):
    import httpx

    from caal import satellite_model, settings

    monkeypatch.setattr(
        settings,
        "load_settings",
        lambda: {"ollama_host": "http://127.0.0.1:11434", "ollama_model": "fixture-local"},
    )

    def handle(req):
        if req.url.path == "/api/show":
            return httpx.Response(
                200, json={"capabilities": ["completion"], "template": "x" * 100000}
            )
        return httpx.Response(200, json={"message": {"content": "Hello"}})

    provider = satellite_model.SatelliteModel(transport=httpx.MockTransport(handle))
    try:
        assert (await provider.chat([{"role": "user", "content": "Hello"}])).content == "Hello"
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_replay_after_engine_restart_is_rejected_and_idle_history_expires(
    identity, monkeypatch
):
    from caal import satellite

    storage = SatelliteStore(identity.identity)
    principal = storage.authenticate(issue(storage, identity.admin)["credential"])

    async def generate(history, text):
        yield "Synthetic result"

    monkeypatch.setattr(satellite, "RETENTION", 0.02)
    first = TurnEngine(storage, generate=generate)
    args = {
        "conversation_id": str(uuid.uuid4()),
        "request_id": str(uuid.uuid4()),
        "text": "test",
        "satellite_id": PILOT,
        "device_id": DEVICE,
    }
    _ = [x async for x in first.stream(principal, **args)]
    second = TurnEngine(storage, generate=generate)
    with pytest.raises(ValueError):
        _ = [x async for x in second.stream(principal, **args)]
    await asyncio.sleep(0.05)
    assert not first.histories and not first.receipts
