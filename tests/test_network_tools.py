import importlib
from types import SimpleNamespace

import pytest

from caal.tools.registry import create_default_registry
from caal.user_scope import UserScope

node = importlib.import_module("caal.llm.llm_node")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope",
    [
        None,
        UserScope.legacy(),
        UserScope.anonymous(),
        UserScope("usr_" + "a" * 24, True, role="user"),
    ],
)
async def test_network_requires_verified_admin(scope):
    agent = SimpleNamespace(_user_scope=scope)
    result = await node._execute_single_tool(agent, "network.status", {"role": "admin"})
    assert result["status"] == "unauthorized"


def test_native_network_catalog():
    registry = create_default_registry()
    assert {
        "network.status",
        "network.addresses",
        "network.lookup",
        "network.speedtest",
        "network.clients",
        "network.target",
    } <= set(registry.names())


@pytest.mark.asyncio
async def test_admin_dispatch_real_handler(monkeypatch):
    from caal import network_diagnostics as net

    async def fake(operation, arguments):
        return {"status": "ok", "data": {"operation": operation}, "message": "Fixture."}

    monkeypatch.setattr(net, "execute", fake)
    from caal.ha_policy import verified_user_scope

    profile = SimpleNamespace(user_id="usr_" + "a" * 24, role="admin", is_active=True)
    identity = SimpleNamespace(store=SimpleNamespace(get_user=lambda uid: profile))
    agent = SimpleNamespace(_user_scope=verified_user_scope(profile, identity))
    result = await node._execute_single_tool(agent, "network.status", {})
    assert result["data"]["operation"] == "status"


@pytest.mark.parametrize(
    "value",
    [
        "127.0.0.1",
        "169.254.169.254",
        "8.8.8.8",
        "::1",
        "192.168.1.1;id",
        "-n",
        "10.0.0.0/8",
        "localhost",
    ],
)
def test_unsafe_target(value):
    from caal.network_diagnostics import private_target

    with pytest.raises(ValueError):
        private_target(value)


@pytest.mark.parametrize("name", ["-x", "a;id", "https://example.com", "a\nwhoami", "a" * 254])
def test_dns_injection(name):
    from caal.network_diagnostics import dns_name

    with pytest.raises(ValueError):
        dns_name(name)


@pytest.mark.asyncio
async def test_container_inventory_is_not_lan(monkeypatch):
    from caal import network_diagnostics as net

    monkeypatch.setattr(net, "vantage", lambda: "docker-container")
    result = await net.local_execute("clients", {})
    assert result["status"] == "unavailable"
    assert result["data"]["clients"] == []


@pytest.mark.asyncio
async def test_busy_and_cancellation_release(monkeypatch):
    import asyncio

    from caal import network_diagnostics as net

    entered = asyncio.Event()

    async def block(*args):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(net, "local_execute", block)
    task = asyncio.create_task(net.execute("status", {}))
    await entered.wait()
    assert (await net.execute("status", {}))["status"] == "busy"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert net._active is False


@pytest.mark.asyncio
async def test_speed_units_and_rate_limit(monkeypatch):
    """The speed tool now reports a real speedtest.net run, in decimal Mbps."""
    import math

    from caal import network_diagnostics as net
    from caal import speedtest_browser as sb

    monkeypatch.setattr(net, "_last_speed", float("-inf"))

    async def fake_run(**kwargs):
        return sb.build_measurement(
            {
                "completed": True,
                "url": "https://www.speedtest.net/result/19675933672",
                "download": {"value": "940.10", "unit": "Mbps"},
                "upload": {"value": "512.00", "unit": "Mbps"},
                "latency": {"idle": "7", "download": "21", "upload": "11", "jitter": None},
                "server": {"name": "Example", "location": "Edmonton, AB", "id": "1"},
                "provider": "Example ISP",
                "connection_mode": "Multi",
                "announcement": "",
            },
            elapsed_seconds=38.0,
        )

    monkeypatch.setattr(net, "run_speedtest", fake_run)
    # local=True is the host-side path; the container routes to the bridge instead.
    response = await net.execute("speedtest", {}, local=True)
    assert response["status"] == "ok"
    assert response["data"]["download_mbps"] == 940.10
    assert response["data"]["upload_mbps"] == 512.00
    assert response["data"]["units"] == "decimal Mbps"
    assert math.isfinite(response["data"]["download_mbps"])
    assert (await net.execute("speedtest", {}, local=True))["status"] == "rate_limited"


@pytest.mark.asyncio
async def test_missing_bridge_is_honest(monkeypatch):
    from caal import network_diagnostics as net

    monkeypatch.delenv("CAAL_NETWORK_BRIDGE_TOKEN", raising=False)
    assert (await net.execute("clients", {}))["status"] == "unavailable"


@pytest.mark.asyncio
async def test_bridge_auth_size_and_operation(monkeypatch):
    import httpx

    from caal.network_bridge import create_app

    app = create_app("x" * 32)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        assert (await c.post("/diagnose", json={"operation": "clients"})).status_code == 401
        headers = {"Authorization": "Bearer " + "x" * 32}
        assert (await c.post("/diagnose", headers=headers, content="x" * 2049)).status_code == 413
        assert (
            await c.post("/diagnose", headers=headers, json={"operation": "shell", "arguments": {}})
        ).status_code == 400
        assert (
            await c.post(
                "/diagnose",
                headers=headers,
                json={
                    "operation": "target",
                    "arguments": {"address": "169.254.169.254", "service": "https"},
                },
            )
        ).status_code == 400


@pytest.mark.asyncio
async def test_forged_admin_role_is_denied():
    agent = SimpleNamespace(_user_scope=UserScope("usr_" + "a" * 24, True, role="admin"))
    response = await node._execute_single_tool(agent, "network.status", {})
    assert response["status"] == "unauthorized"


@pytest.mark.asyncio
async def test_target_must_use_physical_direct_route(monkeypatch):
    from caal import network_diagnostics as net

    async def route(*args, **kwargs):
        return " interface: utun4\n gateway: 10.0.0.1\n flags: <UP,GATEWAY,HOST>"

    monkeypatch.setattr(net, "command", route)
    with pytest.raises(ValueError):
        await net.direct_target("192.168.1.20")


def test_network_privacy_barrier_and_cache():
    from caal.llm.context_barrier import sanitize_for_escalation

    messages = [
        {"role": "tool", "name": "network.clients", "content": "PRIVATE_SENTINEL"},
        {"role": "assistant", "content": "PRIVATE_SENTINEL"},
    ]
    assert "PRIVATE_SENTINEL" not in str(sanitize_for_escalation(messages))
    assert node.ToolDataCache().add("network.clients", {"private": "PRIVATE_SENTINEL"}) is False


@pytest.mark.asyncio
async def test_one_endpoint_failure_is_partial(monkeypatch):
    from caal import network_diagnostics as net

    async def config():
        return {"gateway": None}

    async def lookup(*args):
        return {"answers": ["1.1.1.1"], "status": "ok"}

    async def probe(address, port, hostname=None):
        return {"reachable": hostname != "www.google.com", "latency_ms": 2.0}

    async def http(*args, **kwargs):
        return b"", 0.002

    monkeypatch.setattr(net, "public_request", http)
    monkeypatch.setattr(net, "configuration", config)
    monkeypatch.setattr(net, "lookup", lookup)
    monkeypatch.setattr(net, "probe", probe)
    response = await net.execute("status", {})
    assert response["status"] == "partial"
    assert "Internet connectivity is working" in response["message"]


@pytest.mark.asyncio
async def test_unavailable_speed_reports_no_number_at_all(monkeypatch):
    """A failed run is a stated failure, never a smaller substitute measurement."""
    from caal import network_diagnostics as net
    from caal import speedtest_browser as sb

    async def fail(**kwargs):
        raise sb.SpeedtestUnavailableError("browser_unavailable")

    monkeypatch.setattr(net, "run_speedtest", fail)
    monkeypatch.setattr(net, "_last_speed", float("-inf"))
    response = await net.execute("speedtest", {}, local=True)
    assert response["status"] == "unavailable"
    assert response["data"]["reason"] == "browser_unavailable"
    assert "download_mbps" not in response["data"]


@pytest.mark.asyncio
async def test_public_dns_rebinding_is_blocked(monkeypatch):
    from caal import network_diagnostics as net

    async def lookup(*args):
        return {"answers": ["169.254.169.254"]}

    monkeypatch.setattr(net, "lookup", lookup)
    with pytest.raises(OSError):
        await net.PublicResolver().resolve("example.com", 443)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["NaN", "Infinity", "", "--"])
async def test_nonfinite_measurement_unavailable(monkeypatch, value):
    """A number the site did not really publish produces no measurement."""
    from caal import network_diagnostics as net
    from caal import speedtest_browser as sb

    async def fake_run(**kwargs):
        return sb.build_measurement(
            {
                "completed": True,
                "url": "https://www.speedtest.net/result/1",
                "download": {"value": value, "unit": "Mbps"},
                "upload": {"value": "512.00", "unit": "Mbps"},
                "latency": {},
                "server": {},
                "provider": None,
                "announcement": "",
            },
            elapsed_seconds=38.0,
        )

    monkeypatch.setattr(net, "run_speedtest", fake_run)
    response = await net.local_execute("speedtest", {})
    assert response["status"] == "unavailable"
    assert "download_mbps" not in response["data"]


@pytest.mark.asyncio
async def test_process_lock_busy(monkeypatch, tmp_path):
    import fcntl

    from caal import network_diagnostics as net

    lock = tmp_path / "network.lock"
    monkeypatch.setattr(net, "LOCK_PATH", lock)
    with lock.open("w") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert (await net.execute("status", {}))["status"] == "busy"


@pytest.fixture(autouse=True)
def isolated_network_coordination(monkeypatch, tmp_path):
    from caal import network_diagnostics as net
    from caal import speedtest_browser as sb

    monkeypatch.setattr(net, "LOCK_PATH", tmp_path / "lease")
    monkeypatch.setattr(net, "SPEED_LOCK_PATH", tmp_path / "speed-lease")
    monkeypatch.setattr(net, "_last_speed", float("-inf"))

    async def no_real_browser(**kwargs):
        raise sb.SpeedtestUnavailableError("browser_unavailable")

    # No test starts a real browser or moves real bandwidth. A test that means to
    # exercise a measurement replaces this deliberately.
    monkeypatch.setattr(net, "run_speedtest", no_real_browser)


@pytest.mark.asyncio
async def test_revoked_admin_and_satellite_denied():
    from caal.ha_policy import verified_user_scope

    profile = SimpleNamespace(user_id="usr_" + "a" * 24, role="admin", is_active=True)
    identity = SimpleNamespace(store=SimpleNamespace(get_user=lambda uid: profile))
    scope = verified_user_scope(profile, identity)
    agent = SimpleNamespace(_user_scope=scope, _satellite_restricted=True)
    assert (await node._execute_single_tool(agent, "network.clients", {}))[
        "status"
    ] == "unauthorized"
    agent._satellite_restricted = False
    profile.is_active = False
    assert (await node._execute_single_tool(agent, "network.status", {}))[
        "status"
    ] == "unauthorized"
    assert node._tool_available(agent, "network.status") is False


@pytest.mark.asyncio
async def test_subprocess_timeout_and_output_cap():
    import sys

    from caal.network_diagnostics import command

    with pytest.raises(TimeoutError):
        await command(sys.executable, "-c", "import time; time.sleep(5)", timeout=0.02)
    with pytest.raises(ValueError):
        await command(sys.executable, "-c", 'print("x"*20000)')


@pytest.mark.asyncio
async def test_arp_observation_parsing(monkeypatch):
    from caal import network_diagnostics as net

    async def arp(*args):
        return (
            "? (192.168.1.3) at aa:bb:cc:dd:ee:ff on en0 ifscope\n"
            "? (172.17.0.2) at aa:bb:cc:dd:ee:ff on bridge100 ifscope\n"
        )

    monkeypatch.setattr(net, "vantage", lambda: "darwin-host")
    monkeypatch.setattr(net, "command", arp)
    response = await net.local_execute("clients", {})
    assert len(response["data"]["clients"]) == 1
    assert response["data"]["clients"][0]["reachable"] is None
    assert response["data"]["clients"][0]["client_type"] == "unknown"


@pytest.mark.asyncio
async def test_https_and_dns_independent_tcp(monkeypatch):
    from caal import network_diagnostics as net

    async def config():
        return {"gateway": None}

    async def lookup(*args):
        return {"answers": [], "status": "unavailable"}

    async def probe(*args):
        return {"reachable": True, "latency_ms": 1.0}

    async def http(*args, **kwargs):
        raise OSError

    monkeypatch.setattr(net, "configuration", config)
    monkeypatch.setattr(net, "lookup", lookup)
    monkeypatch.setattr(net, "probe", probe)
    monkeypatch.setattr(net, "public_request", http)
    response = await net.local_execute("status", {})
    assert all(c["direct_tcp"]["reachable"] for c in response["data"]["checks"])
    assert all(c["https"]["reachable"] is False for c in response["data"]["checks"])


@pytest.mark.parametrize("address", [3232235777, True, b"192.168.1.1"])
def test_target_requires_literal_string(address):
    from caal.network_diagnostics import private_target

    with pytest.raises(ValueError):
        private_target(address)


@pytest.mark.asyncio
async def test_broadcast_target_denied(monkeypatch):
    from caal import network_diagnostics as net

    async def command(*args, **kwargs):
        if args[0] == "/sbin/route":
            return "interface: en0\nflags: <UP,HOST,DONE>"
        return "inet 192.168.1.4 netmask 0xffffff00 broadcast 192.168.1.255"

    monkeypatch.setattr(net.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(net, "vantage", lambda: "darwin-host")
    monkeypatch.setattr(net, "command", command)
    with pytest.raises(ValueError):
        await net.direct_target("192.168.1.255")
    assert await net.direct_target("192.168.1.3") == "192.168.1.3"


@pytest.mark.asyncio
async def test_configured_bridge_supplies_host_addresses(monkeypatch):
    from caal import network_bridge
    from caal import network_diagnostics as net

    async def call_host(operation, arguments):
        return {"status": "ok", "data": {"vantage": "darwin-host"}, "bridge_called": True}

    monkeypatch.setenv("CAAL_NETWORK_BRIDGE_TOKEN", "x" * 32)
    monkeypatch.setattr(network_bridge, "call_host", call_host)
    assert (await net.execute("addresses", {}))["bridge_called"] is True


@pytest.mark.asyncio
async def test_authenticated_bridge_dispatch(monkeypatch):
    import httpx

    from caal import network_bridge

    seen = []

    async def execute(operation, arguments, *, local):
        seen.append((operation, arguments, local))
        return {"status": "ok", "message": "Fixture.", "data": {}}

    monkeypatch.setattr(network_bridge, "execute", execute)
    app = network_bridge.create_app("x" * 32)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        response = await c.post(
            "/diagnose",
            headers={"Authorization": "Bearer " + "x" * 32},
            json={"operation": "clients", "arguments": {}},
        )
    assert response.json()["status"] == "ok"
    assert seen == [("clients", {}, True)]


@pytest.mark.asyncio
async def test_bridge_fragmented_response_and_limit():
    from caal.network_bridge import bounded_body

    class Stream:
        async def iter_chunked(self, size):
            yield b'{"sta'
            yield b'tus":"ok"}'

    assert await bounded_body(Stream(), 64) == b'{"status":"ok"}'
    with pytest.raises(ValueError):
        await bounded_body(Stream(), 4)


@pytest.mark.asyncio
async def test_subprocess_fragmented_output():
    import sys

    from caal.network_diagnostics import command

    output = await command(
        sys.executable,
        "-c",
        'import sys,time; print("first",flush=True); time.sleep(.03); print("second",flush=True)',
    )
    assert output == "first\nsecond\n"


@pytest.mark.asyncio
async def test_native_dispatch_does_not_log_private_target(caplog):
    import logging

    from caal.ha_policy import verified_user_scope

    profile = SimpleNamespace(user_id="usr_" + "a" * 24, role="admin", is_active=True)
    identity = SimpleNamespace(store=SimpleNamespace(get_user=lambda uid: profile))
    agent = SimpleNamespace(_user_scope=verified_user_scope(profile, identity))
    sentinel = "PRIVATE_NETWORK_SENTINEL;$(id)"
    with caplog.at_level(logging.DEBUG):
        response = await node._execute_single_tool(
            agent, "network.target", {"address": sentinel, "service": "https"}
        )
    assert response["status"] == "invalid_request"
    assert sentinel not in caplog.text
    assert sentinel not in str(response)
