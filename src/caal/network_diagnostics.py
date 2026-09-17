"""Bounded network measurements; no shell, scans, content fetching or implicit bandwidth jobs."""

from __future__ import annotations

import asyncio
import fcntl
import ipaddress
import json
import os
import platform
import re
import ssl
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from caal.speedtest_browser import DEADLINE_SECONDS as SPEED_DEADLINE
from caal.speedtest_browser import SpeedtestUnavailableError, run_speedtest

PRIVATE = tuple(ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))
SERVICES = {"https": 443, "ssh": 22, "smb": 445, "rdp": 3389}
LOCK_PATH = Path(tempfile.gettempdir()) / f"caal-network-{os.getuid()}.lock"

# A real speed test runs for a minute or more and moves real traffic, so it gets
# its own lease. Sharing the diagnostics lease would freeze every unrelated DNS
# or connectivity check for the length of the measurement.
SPEED_LOCK_PATH = Path(tempfile.gettempdir()) / f"caal-speedtest-{os.getuid()}.lock"
SPEED_COOLDOWN = 120
DIAGNOSTIC_DEADLINE = 25

# The browser runs on the macOS host, so the container always routes there.
BRIDGE_OPERATIONS = ("clients", "target", "speedtest")

_active = False
_speed_active = False
_last_speed = float("-inf")


def deadline_for(operation):
    """A genuine measurement needs room; every other diagnostic stays tight."""
    return SPEED_DEADLINE + 20 if operation == "speedtest" else DIAGNOSTIC_DEADLINE


def vantage():
    return (
        "docker-container" if Path("/.dockerenv").exists() else platform.system().lower() + "-host"
    )


def private_target(value):
    if not isinstance(value, str):
        raise ValueError("private IPv4 string required")
    address = ipaddress.IPv4Address(value)
    if not any(address in network for network in PRIVATE):
        raise ValueError("private IPv4 required")
    return str(address)


def dns_name(value):
    if (
        not isinstance(value, str)
        or len(value) > 253
        or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", value)
        or any(
            not part or len(part) > 63 or part.startswith("-") or part.endswith("-")
            for part in value.split(".")
        )
    ):
        raise ValueError("invalid DNS name")
    return value


def result(status, message, **data):
    return {
        "status": status,
        "message": message,
        "data": {
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "vantage": vantage(),
            **data,
        },
    }


async def command(*argv, timeout=3):
    """Fixed callers only. Kill and reap on timeout/cancellation; cap captured output."""
    process = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )
    try:
        async with asyncio.timeout(timeout):
            output = bytearray()
            while chunk := await process.stdout.read(4096):
                if len(output) + len(chunk) > 16384:
                    raise ValueError("output limit")
                output.extend(chunk)
            await process.wait()
            if process.returncode:
                raise OSError("command failed")
            return output.decode("utf-8", errors="replace")
    finally:
        if process.returncode is None:
            process.kill()
        await process.wait()


_RESOLVE = """import socket,json,sys
name,record=sys.argv[1:]
if record == 'PTR':
 result=[socket.gethostbyaddr(name)[0]]
else:
 family=socket.AF_INET if record=='A' else socket.AF_INET6
 rows=socket.getaddrinfo(name,None,family,socket.SOCK_STREAM)
 result=list(dict.fromkeys(x[4][0] for x in rows))[:16]
print(json.dumps(result))
"""


async def lookup(name, record):
    if record not in ("A", "AAAA", "PTR"):
        raise ValueError("invalid record")
    name = str(ipaddress.ip_address(name)) if record == "PTR" else dns_name(name)
    start = time.monotonic()
    try:
        values = json.loads(await command(sys.executable, "-c", _RESOLVE, name, record))
        return {
            "status": "ok",
            "answers": values,
            "latency_ms": round((time.monotonic() - start) * 1000, 2),
            "record": record,
        }
    except (OSError, TimeoutError):
        return {
            "status": "unavailable",
            "answers": [],
            "latency_ms": None,
            "record": record,
            "error": "resolution_failed_or_timed_out",
        }


async def probe(address, port, hostname=None):
    start = time.monotonic()
    writer = None
    try:
        async with asyncio.timeout(2):
            _, writer = await asyncio.open_connection(
                address,
                port,
                ssl=ssl.create_default_context() if hostname else None,
                server_hostname=hostname,
            )
            data = {"reachable": True, "latency_ms": round((time.monotonic() - start) * 1000, 2)}
            if hostname:
                cert = writer.get_extra_info("ssl_object").getpeercert()
                data["tls_expires_at"] = cert.get("notAfter")
                data["tls_verified"] = True
            return data
    except (OSError, TimeoutError):
        return {"reachable": False, "latency_ms": None, "error": "connection_failed_or_timed_out"}
    finally:
        if writer:
            writer.close()
            try:
                async with asyncio.timeout(0.3):
                    await writer.wait_closed()
            except (OSError, TimeoutError):
                pass


async def configuration():
    commands = (
        {
            "routes": ("/sbin/route", "-n", "get", "default"),
            "interfaces": ("/sbin/ifconfig",),
            "dns": ("/usr/sbin/scutil", "--dns"),
        }
        if platform.system() == "Darwin"
        else {"routes": ("ip", "-j", "route"), "interfaces": ("ip", "-j", "address")}
    )
    data = {}
    for name, argv in commands.items():
        try:
            data[name] = (await command(*argv))[:8192]
        except (OSError, TimeoutError, ValueError):
            data[name] = None
    if platform.system() != "Darwin":
        data["dns"] = Path("/etc/resolv.conf").read_text()[:4096]
    if data.get("routes") is None and Path("/proc/net/route").exists():
        data["routes"] = Path("/proc/net/route").read_text()[:8192]
    routes = data.get("routes") or ""
    match = re.search(r"gateway:\s*([0-9.]+)", routes)
    if not match:
        match = re.search(r'"gateway":\s*"([0-9.]+)"', routes)
    data["gateway"] = match[1] if match else None
    if not data["gateway"] and Path("/proc/net/route").exists():
        for line in routes.splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 3 and fields[1] == "00000000":
                data["gateway"] = str(ipaddress.IPv4Address(bytes.fromhex(fields[2])[::-1]))
                break
    return data


async def direct_target(address):
    address = private_target(address)
    if platform.system() != "Darwin" or vantage() == "docker-container":
        raise ValueError("physical host required")
    route = await command("/sbin/route", "-n", "get", address)
    if not re.search(r"interface:\s*en[0-9]+\b", route) or "GATEWAY" in route:
        raise ValueError("directly connected physical LAN target required")
    interface = re.search(r"interface:\s*(en[0-9]+)\b", route)[1]
    config = await command("/sbin/ifconfig", interface)
    target = ipaddress.IPv4Address(address)
    for own, mask in re.findall(r"inet ([0-9.]+) netmask (0x[0-9a-f]+)", config):
        network = ipaddress.IPv4Network(
            (own, str(ipaddress.IPv4Address(int(mask, 16)))), strict=False
        )
        if target in network and target not in (
            network.network_address,
            network.broadcast_address,
            ipaddress.IPv4Address(own),
        ):
            return address
    raise ValueError("individual directly connected LAN target required")


async def ping(address):
    start = time.monotonic()
    try:
        await command(
            "/sbin/ping" if platform.system() == "Darwin" else "ping",
            "-n",
            "-c",
            "1",
            private_target(address),
            timeout=2,
        )
        return {
            "reachable": True,
            "elapsed_ms": round((time.monotonic() - start) * 1000, 2),
            "method": "one ICMP echo; elapsed includes process startup",
        }
    except (OSError, ValueError, TimeoutError):
        return {"reachable": None, "elapsed_ms": None, "error": "no_echo_or_ping_unavailable"}


class PublicResolver(aiohttp.abc.AbstractResolver):
    async def resolve(self, host, port=0, family=0):
        answers = await lookup(host, "A")
        addresses = answers["answers"]
        if not addresses or any(not ipaddress.ip_address(a).is_global for a in addresses):
            raise OSError("public resolution required")
        return [
            {"hostname": host, "host": a, "port": port, "family": 2, "proto": 0, "flags": 0}
            for a in addresses
        ]

    async def close(self):
        pass


async def public_request(url, *, upload=None, cap=4096, head=False):
    """Only hardcoded caller URLs. Public DNS pinned to connector; no redirects/proxies."""
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(resolver=PublicResolver()),
        trust_env=False,
        timeout=aiohttp.ClientTimeout(total=6),
        auto_decompress=False,
        headers={"Accept-Encoding": "identity"},
        max_line_size=4096,
        max_field_size=4096,
    ) as session:
        start = time.monotonic()
        async with session.request(
            "HEAD" if head else "POST" if upload is not None else "GET",
            url,
            data=upload,
            allow_redirects=False,
        ) as response:
            if response.status != 200:
                raise OSError("endpoint unavailable")
            body = bytearray()
            async for chunk in response.content.iter_chunked(16384):
                body.extend(chunk[: cap - len(body)])
                if len(body) >= cap:
                    break
            return bytes(body), time.monotonic() - start


SPEED_SOURCE = "speedtest.net (Speedtest by Ookla)"

# Every outcome that is not a completed measurement says so plainly. None of
# these fall back to an estimate, a cached figure or a partial number.
SPEED_FAILURE = {
    "timeout": (
        "timeout",
        f"speedtest.net did not finish within {SPEED_DEADLINE} seconds, "
        "so no speed was measured.",
    ),
    "bot_check": (
        "blocked",
        "speedtest.net presented an automated-access check, so the test was not run. "
        "That check was not bypassed.",
    ),
    "consent_required": (
        "blocked",
        "speedtest.net is showing a consent dialog that a person needs to answer. "
        "No agreement was accepted on your behalf, so no speed was measured.",
    ),
    "start_control_unavailable": (
        "blocked",
        "speedtest.net loaded but its start control could not be used, "
        "so no speed was measured.",
    ),
    "browser_unavailable": (
        "unavailable",
        "The measurement browser could not be started on the macOS host, "
        "so no speed was measured.",
    ),
    "incomplete": (
        "unavailable",
        "speedtest.net never published a completed result, so no speed was measured. "
        "No in-progress reading was used.",
    ),
    "incomplete_result": (
        "unavailable",
        "speedtest.net finished but did not publish a usable download and upload figure, "
        "so no speed was reported.",
    ),
    "inconsistent_result": (
        "unavailable",
        "The figures on speedtest.net did not agree with the result it announced, "
        "so nothing was reported rather than risk a wrong number.",
    ),
}


def speed_speech(data):
    """What actually happened, in the words a person asked for."""
    sentence = (
        f"speedtest.net measured {data['download_mbps']:,.0f} megabits per second down "
        f"and {data['upload_mbps']:,.0f} up"
    )
    idle = data["latency"]["idle_ms"]
    if idle is not None:
        sentence += f", with {idle:.0f} milliseconds idle latency"
    if data.get("provider"):
        sentence += f", through {data['provider']}"
    server = data.get("server") or {}
    if server.get("name"):
        sentence += f" to the {server['name']} server"
        if server.get("location"):
            sentence += f" in {server['location']}"
    return (
        sentence + ". That is what this connection delivered during the test just now, "
        "not a guaranteed ISP rate."
    )


async def local_execute(operation, arguments):
    if operation == "lookup":
        data = await lookup(arguments["name"], arguments["record"])
        return result(
            data["status"],
            "DNS lookup completed."
            if data["answers"]
            else "DNS did not return an answer within the limit.",
            lookup=data,
        )
    if operation == "clients":
        if vantage() == "docker-container":
            return result(
                "unavailable",
                "Physical LAN inventory needs the macOS host bridge. "
                "Docker neighbors are not physical LAN clients.",
                clients=[],
            )
        raw = await command("/usr/sbin/arp", "-an")
        clients = []
        for address, mac, interface in re.findall(
            r"\(([0-9.]+)\) at ([0-9a-f:]+) on ([a-zA-Z0-9]+)", raw
        ):
            try:
                private_target(address)
            except ValueError:
                continue
            if not re.fullmatch(r"en\d+", interface):
                continue
            clients.append(
                {
                    "address": address,
                    "mac": mac,
                    "interface": interface,
                    "source": "host ARP cache",
                    "observed": True,
                    "reachable": None,
                    "client_type": "unknown",
                    "vendor": "unknown",
                    "os": "unknown",
                    "confidence": "insufficient evidence",
                }
            )
        return result(
            "ok",
            f"Observed {min(len(clients), 64)} LAN neighbors in the host ARP cache. "
            "Reachability and device types are unverified.",
            clients=clients[:64],
        )
    if operation == "target":
        address = await direct_target(arguments["address"])
        service = arguments["service"]
        if service not in SERVICES:
            raise ValueError("invalid service")
        echo, tcp = await asyncio.gather(ping(address), probe(address, SERVICES[service]))
        return result(
            "ok" if tcp["reachable"] else "partial",
            "The service accepted a TCP connection."
            if tcp["reachable"]
            else "The service did not accept a connection; this does not prove the device is off.",
            ping=echo,
            service=service,
            tcp=tcp,
        )
    if operation == "addresses":
        config = await configuration()
        try:
            body, _ = await public_request("https://api.ipify.org")
            config["public_ip"] = str(ipaddress.ip_address(body.decode().strip()))
        except (OSError, ValueError, aiohttp.ClientError, TimeoutError):
            config["public_ip"] = None
        return result(
            "partial" if any(v is None for v in config.values()) else "ok",
            "Address configuration read from " + vantage() + ".",
            **config,
        )
    if operation == "status":
        config = await configuration()
        gateway = await ping(config["gateway"]) if config["gateway"] else {"reachable": None}

        async def endpoint(hostname, direct_ip):
            dns = await lookup(hostname, "A")
            addresses = [a for a in dns["answers"] if ipaddress.ip_address(a).is_global]

            async def https():
                try:
                    _, elapsed = await public_request("https://" + hostname + "/", head=True)
                    return {"reachable": True, "latency_ms": round(elapsed * 1000, 2)}
                except (OSError, TimeoutError, aiohttp.ClientError):
                    return {
                        "reachable": False,
                        "latency_ms": None,
                        "error": "https_failed_or_timed_out",
                    }

            async def resolved_probe(tls=False):
                return (
                    await probe(addresses[0], 443, hostname if tls else None)
                    if addresses
                    else {"reachable": None, "latency_ms": None}
                )

            tcp, tls, direct, http = await asyncio.gather(
                resolved_probe(), resolved_probe(True), probe(direct_ip, 443), https()
            )
            return {
                "endpoint": hostname,
                "dns": dns,
                "tcp": tcp,
                "tls": tls,
                "direct_tcp": direct,
                "https": http,
            }

        checks = await asyncio.gather(
            endpoint("www.cloudflare.com", "1.1.1.1"), endpoint("www.google.com", "8.8.8.8")
        )
        successes = sum(c["tls"]["reachable"] is True for c in checks)
        return result(
            "ok" if successes == 2 and all(c["https"]["reachable"] for c in checks) else "partial",
            f"{successes} of 2 independent Internet TLS endpoints responded from "
            f"{vantage()}. "
            + (
                "Internet connectivity is working."
                if successes
                else "Connectivity is inconclusive; inspect DNS, gateway and TCP results."
            ),
            gateway=gateway,
            checks=checks,
            method="DNS, independent numeric TCP 443, verified TLS and fixed HTTPS HEAD",
        )
    if operation == "speedtest":
        # speedtest.net runs its own full measurement in a real browser and this
        # only reports what that site published. There is no second, cheaper
        # measurement anywhere in this path: when the run cannot finish, the
        # caller is told why and gets no number at all.
        try:
            data = await run_speedtest(deadline_seconds=SPEED_DEADLINE)
        except SpeedtestUnavailableError as error:
            status, message = SPEED_FAILURE.get(
                error.reason,
                ("unavailable", "The speed test could not be completed. No measurement was taken."),
            )
            return result(status, message, reason=error.reason, source=SPEED_SOURCE)
        return result("ok", speed_speech(data), **data)
    raise ValueError("unknown operation")


async def _execute_locked(operation, arguments, *, local=False):
    """In-process guard. A speed test and an ordinary diagnostic do not exclude
    each other, so a long measurement never starves connectivity or DNS checks."""
    global _active, _speed_active, _last_speed
    speed = operation == "speedtest"
    if _speed_active if speed else _active:
        return result("busy", "A network diagnostic is already running. Please try again shortly.")
    if speed and time.monotonic() - _last_speed < SPEED_COOLDOWN:
        return result(
            "rate_limited",
            f"A full speed test moves real traffic. Wait {SPEED_COOLDOWN} seconds between runs.",
        )
    if speed:
        _speed_active, _last_speed = True, time.monotonic()
    else:
        _active = True
    try:
        async with asyncio.timeout(deadline_for(operation)):
            if not local and (
                operation in BRIDGE_OPERATIONS
                or operation in ("status", "addresses")
                and len(os.environ.get("CAAL_NETWORK_BRIDGE_TOKEN", "")) >= 32
            ):
                from caal import network_bridge

                return await network_bridge.call_host(operation, arguments)
            return await local_execute(operation, arguments)
    except (ValueError, KeyError, TypeError):
        return result(
            "invalid_request", "Use a valid DNS name, record type or private LAN address."
        )
    except (OSError, TimeoutError, aiohttp.ClientError):
        if speed:
            return result(*SPEED_FAILURE["timeout"], reason="timeout", source=SPEED_SOURCE)
        return result("unavailable", "The diagnostic could not finish within its limits.")
    finally:
        if speed:
            _speed_active = False
        else:
            _active = False


async def execute(operation, arguments, *, local=False):
    """A process-shared lease and cooldown survive voice session worker replacement."""
    global _last_speed
    speed = operation == "speedtest"
    # Separate leases: a measurement in progress must not make an unrelated
    # connectivity or DNS check report "busy" for the next three minutes.
    path = SPEED_LOCK_PATH if speed else LOCK_PATH
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    except OSError:
        return result("unavailable", "Network diagnostic coordination is unavailable.")
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return result(
                "busy",
                "A speed test is already running. Please try again shortly."
                if speed
                else "A network diagnostic is already running.",
            )
        if speed:
            raw = os.read(fd, 128)
            try:
                previous = float(raw) if raw else float("-inf")
            except ValueError:
                return result("unavailable", "Speed test coordination is unavailable.")
            now = time.monotonic()
            if 0 <= now - previous < SPEED_COOLDOWN or now - _last_speed < SPEED_COOLDOWN:
                return result(
                    "rate_limited",
                    "A full speed test moves real traffic. "
                    f"Wait {SPEED_COOLDOWN} seconds between runs.",
                )
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, str(now).encode())
            os.ftruncate(fd, os.lseek(fd, 0, os.SEEK_CUR))
        return await _execute_locked(operation, arguments, local=local)
    finally:
        os.close(fd)
