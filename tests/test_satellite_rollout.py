"""Device-specific enrollment and permissions; no production identities."""

import asyncio
import uuid
from dataclasses import replace

import pytest
from test_local_model_api import Harness

from caal.satellite import DEVICE, PILOT, SatelliteStore, TurnEngine

LIVING = "assist_satellite.living"
LIVING_DEVICE = "b" * 32


@pytest.fixture
def setup(tmp_path):
    h = Harness(tmp_path)
    s = SatelliteStore(h.identity)
    return h, s


def catalog(s, connection="ha_" + "a" * 24):
    s.update_inventory(
        connection,
        [
            {"satellite_id": PILOT, "device_id": DEVICE, "name": "Bedroom"},
            {"satellite_id": LIVING, "device_id": LIVING_DEVICE, "name": "Living"},
        ],
    )
    return connection


def test_legacy_migration_keeps_credential_and_identity(setup):
    h, s = setup
    # Reproduce the old three-column DB, including a real digest, then reopen.
    import hashlib

    with h.identity.store.connect() as db:
        db.execute("DROP TABLE satellite_identities")
        db.execute(
            "CREATE TABLE satellite_identities (id TEXT PRIMARY KEY,digest TEXT UNIQUE NOT NULL,"
            "active INTEGER NOT NULL)"
        )
        db.execute(
            "INSERT INTO satellite_identities VALUES (?,?,1)",
            ("sat_old", hashlib.sha256(b"x" * 43).hexdigest()),
        )
    migrated = SatelliteStore(h.identity)
    p = migrated.authenticate("x" * 43)
    assert p.satellite_id == PILOT and p.device_id == DEVICE
    assert migrated.permissions(p)["scope"] == "conversation"


def test_independent_enrollment_rotation_and_forgery(setup):
    h, s = setup
    cid = catalog(s)
    a = s.enroll(h.admin, satellite_id=PILOT, connection_id=cid)
    b = s.enroll(h.admin, satellite_id=LIVING, connection_id=cid)
    p = s.authenticate(a["credential"])
    q = s.authenticate(b["credential"])
    assert p.satellite_id != q.satellite_id and p.id != q.id
    with pytest.raises(PermissionError):
        s.check(replace(p, satellite_id=q.satellite_id, device_id=q.device_id))
    s.enroll(h.admin, satellite_id=LIVING, connection_id=cid)
    s.check(p)
    with pytest.raises(PermissionError):
        s.check(q)
    with pytest.raises(PermissionError):
        s.enroll(h.admin, satellite_id="assist_satellite.fake", connection_id=cid)
    with pytest.raises(PermissionError):
        s.enroll(h.member, satellite_id=PILOT, connection_id=cid)


@pytest.mark.asyncio
async def test_independent_concurrent_turns_and_revocation(setup):
    h, s = setup
    cid = catalog(s)
    ps = [
        s.authenticate(s.enroll(h.admin, satellite_id=e, connection_id=cid)["credential"])
        for e in (PILOT, LIVING)
    ]
    arrived = asyncio.Event()
    release = asyncio.Event()
    count = 0

    async def generate(history, text):
        nonlocal count
        count += 1
        if count == 2:
            arrived.set()
        await release.wait()
        yield text

    engine = TurnEngine(s, generate=generate)

    async def turn(p):
        return "".join(
            [
                x
                async for x in engine.stream(
                    p,
                    conversation_id=str(uuid.uuid4()),
                    request_id=str(uuid.uuid4()),
                    text=p.satellite_id,
                    satellite_id=p.satellite_id,
                    device_id=p.device_id,
                )
            ]
        )

    tasks = [asyncio.create_task(turn(p)) for p in ps]
    await asyncio.wait_for(arrived.wait(), 1)
    s.revoke(h.admin, ps[0].id)
    release.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert isinstance(results[0], PermissionError)
    assert results[1] == LIVING


def test_replay_capacity_is_per_device(setup):
    h, s = setup
    cid = catalog(s)
    a, b = [
        s.authenticate(s.enroll(h.admin, satellite_id=e, connection_id=cid)["credential"])
        for e in (PILOT, LIVING)
    ]
    for _ in range(128):
        s.reserve(a, str(uuid.uuid4()))
    with pytest.raises(ValueError):
        s.reserve(a, str(uuid.uuid4()))
    s.reserve(b, str(uuid.uuid4()))


@pytest.mark.asyncio
async def test_native_home_tools_use_explicit_connection_and_recheck(setup):
    from caal.ha_access import HAStore
    from caal.llm.llm_node import _discover_tools, _execute_single_tool
    from caal.satellite import restricted_agent
    from caal.satellite_home import SatelliteHome

    h, s = setup
    hs = HAStore(h.identity)
    cid = hs.save_connection(
        h.admin,
        {"id": "actual-ha-user", "is_admin": True},
        {"access_token": "fixture-token", "expires_in": 3600},
        endpoint="http://127.0.0.1:8123",
        client_id="https://fixture.invalid",
    )
    catalog(s, cid)
    p = s.authenticate(s.enroll(h.admin, satellite_id=PILOT, connection_id=cid)["credential"])
    s.configure(h.admin, p.id, connection_id=cid, scope="states_and_lights")
    calls = []

    class Client:
        endpoint = "http://127.0.0.1:8123"

        async def current_user(self, token):
            assert token == "fixture-token"
            return {"id": "actual-ha-user", "is_admin": True}

        async def registry(self, token):
            return [
                {
                    "entity_id": PILOT,
                    "device_id": DEVICE,
                    "disabled_by": None,
                    "config_entry_id": "entry",
                }
            ]

        async def request(self, method, path, **kw):
            calls.append((method, path, kw))
            if method == "GET":
                return [
                    {
                        "entity_id": "light.desk",
                        "state": "on",
                        "attributes": {"friendly_name": "Desk"},
                    },
                    {"entity_id": "lock.front", "state": "locked", "attributes": {}},
                    {"entity_id": "sensor.private_calendar", "state": "secret", "attributes": {}},
                ]
            return []

    home = SatelliteHome(
        s,
        p,
        client_factory=lambda _: Client(),
        settings_getter=lambda: {"hass_enabled": True, "hass_host": Client.endpoint},
    )
    agent = restricted_agent(home=home)
    assert {"home.states", "home.light"} <= {
        t["function"]["name"] for t in await _discover_tools(agent)
    }
    read = await _execute_single_tool(agent, "home.states", {})
    assert read["status"] == "ok" and "secret" not in str(read) and "locked" not in str(read)
    for arguments in [
        {"operation": "turn_on", "entity_ids": ["lock.front"]},
        {"operation": "turn_on", "entity_ids": ["light.desk"], "user_id": h.admin},
        {"operation": "toggle", "entity_ids": ["light.desk"]},
        {"operation": "turn_on", "entity_ids": ["light.desk"], "service": "homeassistant.restart"},
    ]:
        bad = await _execute_single_tool(agent, "home.light", arguments)
        assert bad["status"] != "ok"
    assert not any(c[0] == "POST" for c in calls)
    ok = await _execute_single_tool(
        agent, "home.light", {"operation": "turn_off", "entity_ids": ["light.desk"]}
    )
    assert ok["status"] == "ok"
    assert calls[-1][1] == "/api/services/light/turn_off"
    assert calls[-1][2]["body"] == {"entity_id": ["light.desk"]}
    s.configure(h.admin, p.id, connection_id=cid, scope="states")
    count = len(calls)
    assert (
        await _execute_single_tool(
            agent, "home.light", {"operation": "turn_on", "entity_ids": ["light.desk"]}
        )
    )["status"] != "ok"
    assert len(calls) == count
    hs.disconnect(h.admin, cid)
    assert (await _execute_single_tool(agent, "home.states", {}))["status"] != "ok"


def test_unbound_satellite_has_no_home_tools_or_global_fallback(setup):
    from caal.satellite_home import SatelliteHome

    h, s = setup
    cid = catalog(s)
    p = s.authenticate(s.enroll(h.admin, satellite_id=PILOT, connection_id=cid)["credential"])
    home = SatelliteHome(s, p, settings_getter=lambda: {"hass_token": "must-not-use"})
    assert home.tools() == []


@pytest.mark.asyncio
async def test_registry_transport_is_fixed_and_authenticated():
    from aiohttp import web

    from caal.ha_client import HAClient

    seen = []

    async def ws_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "auth_required"})
        seen.append(await ws.receive_json())
        await ws.send_json({"type": "auth_ok"})
        seen.append(await ws.receive_json())
        await ws.send_json({"id": 1, "success": True, "result": []})
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/api/websocket", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        port = site._server.sockets[0].getsockname()[1]
        assert await HAClient(f"http://127.0.0.1:{port}").registry("fixture-token") == []
        assert seen == [
            {"type": "auth", "access_token": "fixture-token"},
            {"id": 1, "type": "config/entity_registry/list"},
        ]
    finally:
        await runner.cleanup()


def test_admin_inventory_and_permission_contract(setup, monkeypatch):
    from fastapi.testclient import TestClient

    from caal import satellite_api, user_api, webhooks
    from caal.ha_access import HAStore

    h, s = setup
    cid = HAStore(h.identity).save_connection(
        h.admin,
        {"id": "actual-ha", "is_admin": True},
        {"access_token": "fixture", "expires_in": 3600},
        endpoint="http://127.0.0.1:8123",
        client_id="https://test.invalid",
    )

    async def discover(storage, connection_id, *, actor_id=None):
        assert connection_id == cid and actor_id == h.admin
        catalog(storage, cid)
        return [
            {"satellite_id": PILOT, "device_id": DEVICE, "name": "Bedroom"},
            {"satellite_id": LIVING, "device_id": LIVING_DEVICE, "name": "Living"},
        ], {"is_admin": True, "is_owner": True}

    monkeypatch.setattr(satellite_api, "discover", discover)
    webhooks.app.dependency_overrides[user_api.get_runtime] = lambda: h.identity
    try:
        with TestClient(webhooks.app) as c:
            assert c.get("/admin/satellites", headers=h.bearer(h.member)).status_code == 403
            status = c.get(
                "/admin/satellites", headers=h.bearer(h.admin), params={"connection_id": cid}
            )
            assert len(status.json()["devices"]) == 2
            assert status.json()["provider_identity"]["is_owner"] is True
            assert (
                c.post("/admin/satellites", headers=h.bearer(h.admin), json={}).status_code == 422
            )
            b = c.post(
                "/admin/satellites",
                headers=h.bearer(h.admin),
                json={"satellite_id": LIVING, "connection_id": cid},
            )
            assert b.status_code == 200
            issued = b.json()
            assert issued["satellite_id"] == LIVING
            assert (
                c.post(
                    "/admin/satellites",
                    headers=h.bearer(h.admin),
                    json={"satellite_id": LIVING, "connection_id": cid, "device_id": DEVICE},
                ).status_code
                == 422
            )
            grant = c.put(
                "/admin/satellites/" + issued["id"],
                headers=h.bearer(h.admin),
                json={"connection_id": cid, "scope": "states_and_lights"},
            )
            assert grant.status_code == 200
            token = {"Authorization": "Bearer " + issued["credential"]}
            info = c.get("/satellite/v1/identity", headers=token).json()
            assert info["scope"] == "states_and_lights" and info["personal_data"] is False
            assert (
                c.put(
                    "/admin/satellites/" + issued["id"],
                    headers=h.bearer(h.admin),
                    json={"connection_id": cid, "scope": "admin"},
                ).status_code
                == 422
            )
            assert (
                c.delete("/admin/satellites/" + issued["id"], headers=h.bearer(h.admin)).status_code
                == 200
            )
            assert c.get("/satellite/v1/identity", headers=token).status_code == 401
    finally:
        webhooks.app.dependency_overrides.pop(user_api.get_runtime, None)


@pytest.mark.asyncio
async def test_revoked_during_provider_read_does_not_return_data_or_control(setup):
    from caal.ha_access import HAStore
    from caal.satellite_home import SatelliteHome

    h, s = setup
    cid = HAStore(h.identity).save_connection(
        h.admin,
        {"id": "ha-user"},
        {"access_token": "fixture", "expires_in": 3600},
        endpoint="http://127.0.0.1:8123",
        client_id="https://test.invalid",
    )
    catalog(s, cid)
    p = s.authenticate(s.enroll(h.admin, satellite_id=PILOT, connection_id=cid)["credential"])
    s.configure(h.admin, p.id, connection_id=cid, scope="states_and_lights")
    calls = []

    class Client:
        async def current_user(self, token):
            return {"id": "ha-user"}

        async def registry(self, token):
            return [{"entity_id": PILOT, "device_id": DEVICE, "config_entry_id": "entry"}]

        async def request(self, method, path, **kw):
            calls.append((method, path))
            s.revoke(h.admin, p.id)
            return [{"entity_id": "light.desk", "state": "on", "attributes": {}}]

    home = SatelliteHome(
        s,
        p,
        client_factory=lambda _: Client(),
        settings_getter=lambda: {"hass_enabled": True, "hass_host": "http://127.0.0.1:8123"},
    )
    with pytest.raises(PermissionError):
        await home.light("turn_off", ["light.desk"])
    assert calls == [("GET", "/api/states")]


def test_device_call_audit_correlates_without_transcripts(setup):
    h, s = setup
    cid = catalog(s)
    p = s.authenticate(s.enroll(h.admin, satellite_id=PILOT, connection_id=cid)["credential"])
    rid = str(uuid.uuid4())
    s.record_call(
        p,
        request_id=rid,
        connection_id=cid,
        provider_id="ha-opaque-user",
        operation="light.turn_on",
        outcome="requested",
    )
    with h.identity.store.connect() as db:
        row = dict(db.execute("SELECT * FROM satellite_calls").fetchone())
    assert (
        row["request_id"] == rid
        and row["satellite"] == p.id
        and row["provider_id"] == "ha-opaque-user"
    )
    assert set(row) == {
        "satellite",
        "request_id",
        "connection_id",
        "provider_id",
        "operation",
        "outcome",
        "created",
    }


@pytest.mark.asyncio
async def test_health_reports_actual_in_process_satellite_work(setup, monkeypatch):
    from types import SimpleNamespace

    from caal import user_api, webhooks

    h, s = setup
    h.identity._satellite_engine = SimpleNamespace(active={"one": object(), "two": object()})
    h.identity._satellite_audio_busy = {"one"}
    monkeypatch.setattr(user_api, "get_runtime", lambda: h.identity)

    class LK:
        room = None

        def __init__(self):
            self.room = self

        async def list_rooms(self, *args):
            return SimpleNamespace(rooms=[])

        async def aclose(self):
            pass

    monkeypatch.setattr(webhooks, "get_livekit_api", LK)
    result = await webhooks.health()
    assert result.satellite_turns == 2 and result.satellite_audio_streams == 1
