"""Public-only egress for the speedtest browser, enforced at the socket.

The browser resolves its own names, so a hostname allow-list is not a boundary:
a public name that answers with a private address walks straight past it. This
module therefore does the resolving itself and hands the connection a *pinned*
address that has already been screened, which is also what removes the window
between checking a name and connecting to it.

Two pieces:

* a destination policy that understands every way an address can be written,
  including the numeric encodings that are meant to slip past a textual check;
* a per-run loopback proxy that every browser request must traverse, so context
  traffic, service workers, redirects and WebSockets are covered by the same
  decision rather than by a page-level interception that never sees them.

It is fail-closed throughout: an unresolvable name, a partly private answer, an
unparsable request or an unauthenticated client all end in a refusal, never in a
connection. Genuine public Ookla hosts and CDNs are unaffected.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hmac
import ipaddress
import re
import secrets
import socket

# A name the browser may look up. Public sites are always dotted; a single label
# is a search-domain or container name, so it never reaches a resolver here.
DNS_NAME = re.compile(
    r"\A(?=.{4,253}\Z)(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
    r"[a-zA-Z]{2}[a-zA-Z0-9-]{0,61}\Z"
)

# Names that never denote a public destination, whatever a resolver would say.
PRIVATE_SUFFIXES = (
    ".local",
    ".localhost",
    ".internal",
    ".lan",
    ".home",
    ".home.arpa",
    ".arpa",
    ".intranet",
    ".corp",
    ".private",
    ".test",
    ".invalid",
    ".example",
    ".onion",
)

MAX_HEAD = 16384
HEAD_TIMEOUT = 15
CONNECT_TIMEOUT = 15
IDLE_TIMEOUT = 300


class BlockedDestinationError(Exception):
    """A destination this run may not reach. Carries why, for honest reporting."""

    def __init__(self, reason, host=None):
        super().__init__(reason)
        self.reason = reason
        self.host = host


def _int_in_base(part):
    """One dotted component as inet_aton reads it: hex, octal or decimal."""
    if not part:
        return None
    lowered = part.lower()
    try:
        if lowered.startswith("0x"):
            return int(lowered[2:], 16) if lowered[2:] else None
        if lowered.startswith("0") and len(lowered) > 1:
            return int(lowered[1:], 8)
        return int(lowered, 10)
    except ValueError:
        return None


def numeric_host(host):
    """Canonicalize any numeric spelling of an address, or return None for a name.

    ``0x7f000001``, ``2130706433``, ``017700000001`` and ``127.1`` are all the
    loopback address as far as a connect(2) is concerned, so they are all the
    loopback address here too.
    """
    if not isinstance(host, str) or not host:
        return None
    candidate = host.strip().strip("[]").split("%")[0]
    if not candidate:
        return None
    with contextlib.suppress(ValueError):
        return str(ipaddress.ip_address(candidate))
    parts = candidate.split(".")
    if len(parts) > 4 or any(not part for part in parts):
        return None
    values = [_int_in_base(part) for part in parts]
    if any(value is None or value < 0 for value in values):
        return None
    if any(value > 255 for value in values[:-1]):
        return None
    tail_width = 4 - (len(values) - 1)
    if values[-1] >= 256**tail_width:
        return None
    packed = 0
    for value in values[:-1]:
        packed = (packed << 8) | value
    packed = (packed << (8 * tail_width)) | values[-1]
    return str(ipaddress.IPv4Address(packed))


def address_is_public(address):
    """Globally routable, with IPv4-mapped IPv6 judged as the address it carries."""
    try:
        parsed = ipaddress.ip_address(str(address).split("%")[0])
    except ValueError:
        return False
    mapped = getattr(parsed, "ipv4_mapped", None)
    if mapped is not None:
        parsed = mapped
    return bool(parsed.is_global) and not parsed.is_multicast and not parsed.is_reserved


def resolvable_name(host):
    """A dotted public DNS name, and not one of the reserved private suffixes."""
    if not isinstance(host, str) or not host:
        return False
    name = host.strip().rstrip(".").lower()
    if not DNS_NAME.match(name) or name == "localhost":
        return False
    return not name.endswith(PRIVATE_SUFFIXES)


def screen_host(host):
    """Classify a request host before any lookup happens.

    Returns the canonical literal address for a numeric host, or ``None`` when
    the host is a public name that still has to be resolved. Anything else is a
    refusal.
    """
    literal = numeric_host(host)
    if literal is not None:
        if not address_is_public(literal):
            return _blocked("private_address", host)
        return literal
    if not resolvable_name(host):
        return _blocked("non_public_name", host)
    return None


def _blocked(reason, host):
    raise BlockedDestinationError(reason, host)


def screen_answers(addresses, host=None):
    """Every answer must be public. One private answer condemns the whole name.

    Accepting the public half of a mixed answer would leave the browser free to
    retry onto the private half, so a mixed answer is treated as an attempt.
    """
    screened = [str(address) for address in addresses if address]
    if not screened:
        raise BlockedDestinationError("unresolvable", host)
    if not all(address_is_public(address) for address in screened):
        raise BlockedDestinationError("private_address", host)
    return screened


def public_endpoints(host, port, *, resolve=socket.getaddrinfo):
    """Screened, pinned socket destinations for one connection.

    The addresses returned are the ones the caller must connect to. Nothing
    re-resolves the name afterwards, so there is no second lookup to poison.
    """
    literal = screen_host(host)
    if literal is not None:
        family = socket.AF_INET6 if ":" in literal else socket.AF_INET
        return [(family, literal, port)]
    try:
        answers = resolve(host, port, 0, socket.SOCK_STREAM)
    except (OSError, ValueError) as error:
        raise BlockedDestinationError("unresolvable", host) from error
    endpoints = [(entry[0], entry[4][0], entry[4][1]) for entry in answers if entry[4]]
    screen_answers([address for _, address, _ in endpoints], host)
    return endpoints


async def public_endpoints_async(host, port, *, resolve=None):
    """The same decision, off the event loop thread."""
    if resolve is None:
        loop = asyncio.get_running_loop()

        async def resolve(name, service, *args):  # noqa: A001 - local rebinding is the seam
            return await loop.getaddrinfo(name, service, type=socket.SOCK_STREAM)

    literal = screen_host(host)
    if literal is not None:
        family = socket.AF_INET6 if ":" in literal else socket.AF_INET
        return [(family, literal, port)]
    try:
        answers = await resolve(host, port, 0, socket.SOCK_STREAM)
    except (OSError, ValueError) as error:
        raise BlockedDestinationError("unresolvable", host) from error
    endpoints = [(entry[0], entry[4][0], entry[4][1]) for entry in answers if entry[4]]
    screen_answers([address for _, address, _ in endpoints], host)
    return endpoints


def split_authority(authority, default_port):
    """``host:port`` as a proxy writes it, including a bracketed IPv6 literal."""
    if not isinstance(authority, str) or not authority:
        raise BlockedDestinationError("malformed_request", authority)
    text = authority.strip()
    if text.startswith("["):
        closing = text.find("]")
        if closing < 0:
            raise BlockedDestinationError("malformed_request", authority)
        host, rest = text[1:closing], text[closing + 1 :]
        port = rest[1:] if rest.startswith(":") else ""
    elif text.count(":") == 1:
        host, _, port = text.partition(":")
    else:
        host, port = text, ""
    if not host:
        raise BlockedDestinationError("malformed_request", authority)
    if not port:
        return host, default_port
    if not port.isdigit() or not 1 <= int(port) <= 65535:
        raise BlockedDestinationError("malformed_request", authority)
    return host, int(port)


HOP_BY_HOP = (
    "proxy-authorization",
    "proxy-connection",
    "connection",
    "keep-alive",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
)


class EgressProxy:
    """One throwaway loopback proxy, owned by a single speedtest run.

    It listens on an ephemeral port on 127.0.0.1 and requires a credential that
    exists only for this run, so it is not a relay for anything else on the host,
    and it will only ever open a socket to an address it screened itself.
    """

    def __init__(self, *, resolve=None, open_connection=None, idle_timeout=IDLE_TIMEOUT):
        self._resolve = resolve
        self._open_connection = open_connection or asyncio.open_connection
        self._idle_timeout = idle_timeout
        self._server = None
        self._tasks = set()
        self.username = "caal-" + secrets.token_hex(8)
        self.password = secrets.token_urlsafe(32)
        self.port = None
        # Evidence for the run report. Hosts only, never bytes of traffic.
        self.allowed = []
        self.denied = []

    async def start(self):
        self._server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def close(self):
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
            self._tasks.clear()

    def settings(self):
        """Playwright proxy settings for this run. Loopback only, never model-supplied."""
        return {
            "server": f"http://127.0.0.1:{self.port}",
            # Chromium bypasses loopback by default; that would be a hole straight
            # back to the bridge, so the implicit bypass is switched off.
            "bypass": "<-loopback>",
            "username": self.username,
            "password": self.password,
        }

    def _authorized(self, headers):
        supplied = headers.get("proxy-authorization", "")
        scheme, _, encoded = supplied.partition(" ")
        if scheme.lower() != "basic":
            return False
        expected = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
        return hmac.compare_digest(encoded.strip(), expected)

    async def _serve(self, reader, writer):
        task = asyncio.current_task()
        self._tasks.add(task)
        try:
            await self._handle(reader, writer)
        except (asyncio.CancelledError, ConnectionError, TimeoutError, BlockedDestinationError):
            pass
        except Exception:  # noqa: BLE001 - a proxy failure is a refusal, never a passthrough
            pass
        finally:
            self._tasks.discard(task)
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    async def _handle(self, reader, writer):
        try:
            async with asyncio.timeout(HEAD_TIMEOUT):
                head = await reader.readuntil(b"\r\n\r\n")
        except (asyncio.IncompleteReadError, TimeoutError, asyncio.LimitOverrunError):
            return
        if len(head) > MAX_HEAD:
            return await self._refuse(writer, 431, "request_too_large")
        lines = head.decode("latin-1").split("\r\n")
        parts = lines[0].split(" ")
        headers = {}
        for line in lines[1:]:
            name, _, value = line.partition(":")
            if name:
                headers[name.strip().lower()] = value.strip()
        if len(parts) != 3:
            return await self._refuse(writer, 400, "malformed_request")
        if not self._authorized(headers):
            return await self._refuse(writer, 407, "unauthenticated")
        method, target = parts[0].upper(), parts[1]
        try:
            if method == "CONNECT":
                host, port = split_authority(target, 443)
                return await self._tunnel(reader, writer, host, port)
            return await self._forward(reader, writer, method, target, lines, headers)
        except BlockedDestinationError as blocked:
            return await self._refuse(writer, 403, blocked.reason, host=blocked.host)

    async def _refuse(self, writer, status, reason, host=None):
        if status == 403:
            self.denied.append({"host": host, "reason": reason})
        challenge = 'Proxy-Authenticate: Basic realm="caal-speedtest"\r\n' if status == 407 else ""
        body = reason.encode()
        with contextlib.suppress(Exception):
            writer.write(
                f"HTTP/1.1 {status} {reason}\r\n{challenge}"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body
            )
            await writer.drain()

    async def _connect_pinned(self, host, port):
        """Connect only to an address this proxy screened, and remember which."""
        endpoints = await public_endpoints_async(host, port, resolve=self._resolve)
        last = None
        for family, address, resolved_port in endpoints:
            try:
                async with asyncio.timeout(CONNECT_TIMEOUT):
                    upstream = await self._open_connection(
                        host=address, port=resolved_port, family=family
                    )
            except (OSError, TimeoutError) as error:
                last = error
                continue
            self.allowed.append({"host": host, "port": resolved_port})
            return upstream
        raise BlockedDestinationError("upstream_unreachable", host) from last

    async def _tunnel(self, reader, writer, host, port):
        upstream_reader, upstream_writer = await self._connect_pinned(host, port)
        writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        await writer.drain()
        await self._pump(reader, writer, upstream_reader, upstream_writer)

    async def _forward(self, reader, writer, method, target, lines, headers):
        from urllib.parse import urlsplit

        parsed = urlsplit(target)
        if parsed.scheme != "http" or not parsed.netloc:
            raise BlockedDestinationError("non_web_scheme", target)
        host, port = split_authority(parsed.netloc, 80)
        upstream_reader, upstream_writer = await self._connect_pinned(host, port)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        rebuilt = [f"{method} {path} HTTP/1.1"]
        for line in lines[1:]:
            name = line.partition(":")[0].strip().lower()
            if name and name not in HOP_BY_HOP:
                rebuilt.append(line)
        # One request per upstream connection: a pooled connection must not be
        # reusable for a second, unscreened host.
        rebuilt.append("Connection: close")
        upstream_writer.write(("\r\n".join(rebuilt) + "\r\n\r\n").encode("latin-1"))
        await upstream_writer.drain()
        await self._pump(reader, writer, upstream_reader, upstream_writer)

    async def _pump(self, reader, writer, upstream_reader, upstream_writer):
        async def copy(source, sink):
            with contextlib.suppress(Exception):
                while True:
                    chunk = await source.read(65536)
                    if not chunk:
                        break
                    sink.write(chunk)
                    await sink.drain()
            with contextlib.suppress(Exception):
                sink.close()

        try:
            async with asyncio.timeout(self._idle_timeout):
                await asyncio.gather(
                    copy(reader, upstream_writer),
                    copy(upstream_reader, writer),
                )
        finally:
            with contextlib.suppress(Exception):
                upstream_writer.close()
                await upstream_writer.wait_closed()
