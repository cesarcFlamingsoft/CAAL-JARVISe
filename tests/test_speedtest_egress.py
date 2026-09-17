"""Fail-closed public-only egress for the real speedtest browser.

Every test here exercises production code. Where a seam is substituted it is the
narrowest one available - the name resolver, or the socket opener - never the
policy under test and never a stand-in for the proxy or the browser itself.
"""

import asyncio
import base64

import pytest
import pytest_asyncio

from caal import speedtest_browser as sb
from caal import speedtest_proxy as sp


class FakeRoute:
    """Playwright hands the guard one of these per request; only the verdict is ours."""

    def __init__(self, url):
        self.request = type("R", (), {"url": url})()
        self.aborted = False
        self.continued = False

    async def abort(self, *args):
        self.aborted = True

    async def continue_(self, **kwargs):
        self.continued = True


def async_resolver(mapping):
    sync = resolver_for(mapping)

    async def resolve(host, port, *args):
        return sync(host, port, *args)

    return resolve


def resolver_for(mapping):
    """A stand-in for getaddrinfo that answers from a fixed table."""

    def resolve(host, port, *args, **kwargs):
        import socket

        answers = mapping.get(host)
        if answers is None:
            raise OSError(f"no answer for {host}")
        entries = []
        for address in answers:
            family = socket.AF_INET6 if ":" in address else socket.AF_INET
            sockaddr = (address, port, 0, 0) if family == socket.AF_INET6 else (address, port)
            entries.append((family, socket.SOCK_STREAM, 6, "", sockaddr))
        return entries

    return resolve


class TestNumericHostAliases:
    """A private address written in an unusual encoding is still a private address."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://0x7f000001:18004/diagnose",
            "http://0x7F.0x0.0x0.0x1/",
            "http://2130706433/",
            "http://017700000001/",
            "http://0177.0.0.01/",
            "http://127.1/",
            "http://127.0.1/",
            "http://0/",
            "http://[::ffff:127.0.0.1]/",
            "http://[0:0:0:0:0:0:0:1]/",
            "http://[fe80::1%25en0]/",
            "http://3232235777/",
            "http://0xa9fea9fe/",
        ],
    )
    def test_alternate_numeric_encodings_of_private_addresses_are_denied(self, url):
        assert sb.allowed_request(url) is False

    @pytest.mark.parametrize(
        "url",
        [
            "http://host.docker.internal:18004/diagnose",
            "http://gateway.docker.internal/",
            "http://router/",
            "http://nas.home.arpa/",
            "http://printer.lan/",
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://speedtest/",
        ],
    )
    def test_container_and_single_label_lan_names_are_denied(self, url):
        assert sb.allowed_request(url) is False

    @pytest.mark.parametrize(
        "host",
        [
            "www.speedtest.net",
            "cdn.speedtest.net",
            "speedtest.telus.com",
            "ookla-speedtest.shawcable.net",
            "xn--bcher-kva.de",
            "8.8.8.8",
            "www.gstatic.com",
        ],
    )
    def test_ordinary_public_names_are_still_allowed(self, host):
        assert sb.allowed_request(f"https://{host}/x") is True


class TestResolvedPrivateAliases:
    """A public-looking name that answers with a private address is still private."""

    @pytest.mark.parametrize(
        "answers",
        [
            ["127.0.0.1"],
            ["192.168.1.10"],
            ["10.1.2.3"],
            ["172.20.0.5"],
            ["169.254.169.254"],
            ["::1"],
            ["fd00::1"],
            ["::ffff:127.0.0.1"],
            ["0.0.0.0"],
        ],
    )
    def test_a_name_resolving_to_a_private_address_is_refused(self, answers):
        with pytest.raises(sp.BlockedDestinationError) as blocked:
            sp.public_endpoints(
                "evil.example.net", 443, resolve=resolver_for({"evil.example.net": answers})
            )
        assert blocked.value.reason == "private_address"

    def test_a_mixed_answer_is_refused_whole(self):
        """Taking the public half would leave the private half one retry away."""
        with pytest.raises(sp.BlockedDestinationError) as blocked:
            sp.public_endpoints(
                "mixed.example.net",
                443,
                resolve=resolver_for({"mixed.example.net": ["93.184.216.34", "127.0.0.1"]}),
            )
        assert blocked.value.reason == "private_address"

    def test_an_unresolvable_name_is_refused_not_admitted(self):
        with pytest.raises(sp.BlockedDestinationError) as blocked:
            sp.public_endpoints("nowhere.example.net", 443, resolve=resolver_for({}))
        assert blocked.value.reason == "unresolvable"

    def test_an_empty_answer_is_refused(self):
        with pytest.raises(sp.BlockedDestinationError) as blocked:
            sp.public_endpoints(
                "empty.example.net", 443, resolve=resolver_for({"empty.example.net": []})
            )
        assert blocked.value.reason == "unresolvable"

    def test_a_public_answer_is_returned_as_a_pinned_address(self):
        endpoints = sp.public_endpoints(
            "ookla.example.net",
            8080,
            resolve=resolver_for({"ookla.example.net": ["93.184.216.34", "2606:2800:220::1"]}),
        )
        assert [address for _, address, _ in endpoints] == ["93.184.216.34", "2606:2800:220::1"]
        assert {port for _, _, port in endpoints} == {8080}


class TestGuardCoversEveryRequestShape:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "url",
        [
            "https://rebind.example.net/asset.js",
            "wss://rebind.example.net/ws",
            "ws://rebind.example.net/ws",
        ],
    )
    async def test_websocket_and_http_requests_to_a_rebound_name_are_denied(self, url):
        denied = await sb.allowed_destination(
            url, resolve=async_resolver({"rebind.example.net": ["127.0.0.1"]})
        )
        assert denied is False

    @pytest.mark.asyncio
    async def test_a_redirect_target_is_screened_as_its_own_request(self):
        """A redirect arrives at the guard as a fresh request with a new host."""
        route = FakeRoute("https://www.speedtest.net/")
        resolve = async_resolver(
            {"www.speedtest.net": ["151.101.2.219"], "internal.example.net": ["10.0.0.9"]}
        )
        await sb.guard_route(route, resolve=resolve)
        assert route.continued is True

        redirected = FakeRoute("https://internal.example.net/")
        await sb.guard_route(redirected, resolve=resolve)
        assert redirected.aborted is True and redirected.continued is False


class EchoServer:
    """A real local TCP peer, so tunnelled bytes are actually tunnelled."""

    def __init__(self):
        self.server = None
        self.port = None
        self.received = bytearray()

    async def start(self):
        async def handle(reader, writer):
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                self.received.extend(chunk)
                writer.write(b"echo:" + chunk)
                await writer.drain()
            writer.close()

        self.server = await asyncio.start_server(handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def close(self):
        self.server.close()
        await self.server.wait_closed()


async def proxy_request(proxy, head, *, credential=True, payload=b"", read_bytes=4096):
    """One real TCP conversation with the real proxy."""
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy.port)
    try:
        auth = ""
        if credential:
            token = base64.b64encode(f"{proxy.username}:{proxy.password}".encode()).decode()
            auth = f"Proxy-Authorization: Basic {token}\r\n"
        writer.write(head.encode() + auth.encode() + b"\r\n" + payload)
        await writer.drain()
        async with asyncio.timeout(10):
            return await reader.read(read_bytes), reader, writer
    except TimeoutError:
        writer.close()
        raise


@pytest_asyncio.fixture
async def proxy_factory():
    started = []

    async def make(**kwargs):
        proxy = await sp.EgressProxy(**kwargs).start()
        started.append(proxy)
        return proxy

    yield make
    for proxy in started:
        await proxy.close()


class TestPerRunProxyRefusesPrivateDestinations:
    @pytest.mark.asyncio
    async def test_a_connect_to_a_name_that_resolves_to_loopback_is_refused(self, proxy_factory):
        opened = []

        async def never(**kwargs):
            opened.append(kwargs)
            raise AssertionError("the proxy must not open a private socket")

        proxy = await proxy_factory(
            resolve=async_resolver({"rebind.example.net": ["127.0.0.1"]}), open_connection=never
        )
        response, _, writer = await proxy_request(
            proxy, "CONNECT rebind.example.net:18004 HTTP/1.1\r\nHost: rebind.example.net\r\n"
        )
        writer.close()
        assert b"403" in response and b"private_address" in response
        assert opened == []
        assert proxy.denied and proxy.denied[0]["reason"] == "private_address"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "authority",
        ["127.0.0.1:18004", "0x7f000001:18004", "[::1]:80", "192.168.1.1:80", "localhost:18004"],
    )
    async def test_a_connect_written_as_a_private_literal_is_refused(
        self, proxy_factory, authority
    ):
        proxy = await proxy_factory(resolve=async_resolver({}))
        response, _, writer = await proxy_request(
            proxy, f"CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n"
        )
        writer.close()
        assert b"403" in response

    @pytest.mark.asyncio
    async def test_a_plain_http_request_at_the_loopback_bridge_is_refused(self, proxy_factory):
        proxy = await proxy_factory(resolve=async_resolver({}))
        response, _, writer = await proxy_request(
            proxy, "GET http://127.0.0.1:18004/diagnose HTTP/1.1\r\nHost: 127.0.0.1:18004\r\n"
        )
        writer.close()
        assert b"403" in response

    @pytest.mark.asyncio
    async def test_a_request_without_the_run_credential_is_refused(self, proxy_factory):
        proxy = await proxy_factory(
            resolve=async_resolver({"www.speedtest.net": ["151.101.2.219"]})
        )
        response, _, writer = await proxy_request(
            proxy,
            "CONNECT www.speedtest.net:443 HTTP/1.1\r\nHost: www.speedtest.net\r\n",
            credential=False,
        )
        writer.close()
        assert b"407" in response and b"Proxy-Authenticate" in response

    @pytest.mark.asyncio
    async def test_it_listens_on_loopback_with_a_fresh_per_run_credential(self, proxy_factory):
        first = await proxy_factory(resolve=async_resolver({}))
        second = await proxy_factory(resolve=async_resolver({}))
        assert first._server.sockets[0].getsockname()[0] == "127.0.0.1"
        assert first.password != second.password and first.username != second.username
        assert len(first.password) >= 32
        settings = first.settings()
        assert settings["server"] == f"http://127.0.0.1:{first.port}"
        assert settings["bypass"] == "<-loopback>"


class TestPerRunProxyCarriesGenuineTraffic:
    @pytest.mark.asyncio
    async def test_it_tunnels_to_the_pinned_public_address_it_screened(self, proxy_factory):
        echo = await EchoServer().start()
        pinned = []

        async def open_pinned(*, host, port, family=None):
            pinned.append((host, port))
            return await asyncio.open_connection("127.0.0.1", echo.port)

        proxy = await proxy_factory(
            resolve=async_resolver({"ookla.example.net": ["93.184.216.34"]}),
            open_connection=open_pinned,
        )
        try:
            response, reader, writer = await proxy_request(
                proxy, "CONNECT ookla.example.net:8080 HTTP/1.1\r\nHost: ookla.example.net\r\n"
            )
            assert b"200 Connection established" in response
            # The socket was opened to the screened address, not to the name.
            assert pinned == [("93.184.216.34", 8080)]
            writer.write(b"payload")
            await writer.drain()
            async with asyncio.timeout(10):
                tunnelled = await reader.read(64)
            assert tunnelled == b"echo:payload"
            writer.close()
        finally:
            await echo.close()

    @pytest.mark.asyncio
    async def test_closing_the_proxy_stops_it_accepting(self, proxy_factory):
        proxy = await proxy_factory(resolve=async_resolver({}))
        port = proxy.port
        await proxy.close()
        with pytest.raises(OSError):
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"CONNECT www.speedtest.net:443 HTTP/1.1\r\n\r\n")
            await writer.drain()
            if not await reader.read(16):
                raise OSError("closed")


class TestBrowserIsConfinedToTheProxy:
    def test_launch_options_route_the_browser_through_the_per_run_loopback_proxy(self):
        settings = {
            "server": "http://127.0.0.1:54321",
            "bypass": "<-loopback>",
            "username": "u",
            "password": "p",
        }
        options = sb.launch_options(proxy=settings)
        assert options["proxy"] == settings
        # Chromium's implicit localhost bypass is what would let the page reach
        # the bridge directly, so it has to be switched off.
        assert options["proxy"]["bypass"] == "<-loopback>"

    def test_launch_options_never_take_a_proxy_from_anywhere_but_this_run(self):
        """No argument path, so nothing outside the process can aim the browser."""
        rendered = " ".join(sb.launch_options().get("args", []))
        assert "--proxy-server" not in rendered and "--proxy-bypass-list" not in rendered
        assert "proxy" not in sb.launch_options()

    def test_service_workers_cannot_open_an_unscreened_path(self):
        assert sb.context_options()["service_workers"] == "block"

    @pytest.mark.asyncio
    async def test_the_guard_is_installed_on_the_whole_context_not_one_page(self):
        installed = {}

        class Context:
            async def route(self, pattern, handler):
                installed["context"] = pattern

            async def new_page(self):
                return Page()

        class Page:
            async def route(self, pattern, handler):
                installed["page"] = pattern

        context = Context()
        page = await sb.prepare_context(context)
        assert installed.get("context") == "**/*"
        assert isinstance(page, Page)


def chromium_pids():
    """Chromium processes belonging to this user's Playwright install, by command line."""
    import subprocess

    listing = subprocess.run(
        ["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=False
    ).stdout
    found = set()
    for line in listing.splitlines():
        pid, _, command = line.strip().partition(" ")
        if pid.isdigit() and "ms-playwright" in command and "chrom" in command.lower():
            found.add(int(pid))
    return found


def playwright_chromium_available():
    try:
        from playwright.async_api import async_playwright  # noqa: F401
    except ImportError:
        return False
    import pathlib

    root = pathlib.Path.home() / "Library" / "Caches" / "ms-playwright"
    return any(root.glob("chromium-*")) if root.exists() else False


requires_chromium = pytest.mark.skipif(
    not playwright_chromium_available(),
    reason="a real Playwright Chromium install is required for owned-process cleanup evidence",
)


@requires_chromium
class TestRealBrowserCleanup:
    """No fake context manager here: a real Chromium is launched and then killed."""

    @pytest.mark.asyncio
    async def test_cancelling_a_run_tears_down_every_owned_browser_process(self):
        before = chromium_pids()
        opened = asyncio.Event()
        seen = {}

        async def run():
            async with sb._playwright_page() as page:
                seen["page"] = page
                opened.set()
                await asyncio.sleep(600)

        task = asyncio.create_task(run())
        async with asyncio.timeout(180):
            await opened.wait()
        spawned = chromium_pids() - before
        assert spawned, "the run must actually have launched a real browser"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Teardown must have finished by the time the cancellation surfaces. A
        # browser that is merely scheduled to close later would keep moving
        # bandwidth, and would be orphaned outright if the loop shut down here.
        assert not (spawned & chromium_pids())

    @pytest.mark.asyncio
    async def test_a_completed_run_leaves_no_browser_and_no_listening_proxy(self):
        before = chromium_pids()
        ports = {}

        async def run():
            async with sb._playwright_page() as page:
                ports["page"] = page
                return chromium_pids() - before

        spawned = await run()
        assert spawned
        surviving = spawned
        async with asyncio.timeout(60):
            while surviving & chromium_pids():
                await asyncio.sleep(0.5)
        assert not (spawned & chromium_pids())


class FakeRequest:
    """The parts of a Starlette request the bridge handler actually touches."""

    def __init__(self, token, payload, *, disconnect_after=None):
        import json

        self.headers = {"authorization": "Bearer " + token}
        self._body = json.dumps(payload).encode()
        self._disconnect_after = disconnect_after
        self.polls = 0

    async def stream(self):
        yield self._body

    async def is_disconnected(self):
        self.polls += 1
        if self._disconnect_after is None:
            return False
        return self.polls > self._disconnect_after


def bridge_endpoint(app):
    for route in app.routes:
        if getattr(route, "path", None) == "/diagnose":
            return route.endpoint
    raise AssertionError("the bridge has no /diagnose route")


TOKEN = "x" * 48


class TestBridgeCancelsOnClientDisconnect:
    """A hung-up caller must not leave a real browser running a full measurement."""

    @pytest.mark.asyncio
    async def test_a_disconnected_client_cancels_the_running_speedtest(self, monkeypatch):
        from caal import network_bridge

        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def never_finishes(operation, arguments, *, local=False):
            entered.set()
            try:
                await asyncio.sleep(600)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return {"status": "ok"}

        monkeypatch.setattr(network_bridge, "execute", never_finishes)
        app = network_bridge.create_app(TOKEN)
        request = FakeRequest(
            TOKEN, {"operation": "speedtest", "arguments": {}}, disconnect_after=1
        )
        async with asyncio.timeout(30):
            response = await bridge_endpoint(app)(request)
            await cancelled.wait()
        assert entered.is_set()
        assert cancelled.is_set()
        assert response.status_code == 499

    @pytest.mark.asyncio
    async def test_a_connected_client_still_gets_its_result(self, monkeypatch):
        from caal import network_bridge

        async def finishes(operation, arguments, *, local=False):
            return {"status": "ok", "message": "done"}

        monkeypatch.setattr(network_bridge, "execute", finishes)
        app = network_bridge.create_app(TOKEN)
        request = FakeRequest(TOKEN, {"operation": "speedtest", "arguments": {}})
        async with asyncio.timeout(30):
            response = await bridge_endpoint(app)(request)
        assert response == {"status": "ok", "message": "done"}


class TestNoTrafficEscapesTheProxy:
    """An HTTP proxy cannot carry UDP, so WebRTC is the one path that could slip."""

    def test_webrtc_may_not_open_a_non_proxied_socket(self):
        rendered = " ".join(sb.launch_options().get("args", []))
        assert "--force-webrtc-ip-handling-policy=disable_non_proxied_udp" in rendered

    def test_the_hardening_flags_are_present_without_weakening_anything(self):
        options = sb.launch_options()
        rendered = " ".join(options["args"])
        assert options["chromium_sandbox"] is True
        for weakening in ("--no-sandbox", "--ignore-certificate-errors", "--disable-web-security"):
            assert weakening not in rendered
