"""Where JARVIS local model lives, and which models are installed there.

An operator types this endpoint into the settings UI, so the backend ends up
making a request on someone elses say-so. That is the shape of a server-side
request forgery, and the answer here is a narrow allowance rather than a
blocklist:

* plain ``http`` only -- a local Ollama does not speak TLS, and https would
  only widen what can be reached;
* an explicit port, so nothing is guessed;
* no credentials, no path, no query and no fragment: the only path this
  module ever requests is ``/api/tags``;
* a host that is either one of two named local aliases (``localhost``,
  ``host.docker.internal``) or an IP literal inside loopback, RFC1918 private
  IPv4, or IPv6 loopback / ULA / link-local.

Everything else is refused offline, before a socket is opened. The named
aliases are resolved conservatively when a request is actually made: every
address they resolve to must itself be local, or the request does not happen.

Nothing here logs the endpoint, the payload or an upstream error string: an
address is infrastructure, and a model list is a fact about a private
network. Failures are reduced to a short code the UI can translate.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import socket
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_ENDPOINT",
    "DISCOVERY_TIMEOUT",
    "LOCAL_ALIASES",
    "MAX_ENDPOINT_LENGTH",
    "MAX_MODELS",
    "MAX_MODEL_NAME",
    "DiscoveryError",
    "EndpointError",
    "configured_endpoint",
    "discover_models",
    "is_model_name",
    "normalize_endpoint",
    "resolve_local_alias",
]

DEFAULT_ENDPOINT = "http://localhost:11434"
# The two names a local Ollama is reached by from inside and outside a
# container. Any other name is a name someone else controls.
LOCAL_ALIASES = frozenset({"localhost", "host.docker.internal"})
MAX_ENDPOINT_LENGTH = 200
MAX_MODEL_NAME = 120
MAX_MODELS = 100

# Connecting to a machine on the same network either works at once or is not
# there; the read is the only part worth waiting on, and not for long.
DISCOVERY_TIMEOUT = httpx.Timeout(connect=2.0, read=5.0, write=2.0, pool=2.0)

_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]*$")
_CONTROL = re.compile(r"[\x00-\x20\x7f]")


class EndpointError(ValueError):
    """A refusal to accept an endpoint. Carries a code and a plain sentence."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class DiscoveryError(Exception):
    """Discovery did not produce a model list. Never carries upstream text."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


_REFUSALS = dict(
    invalid_endpoint="That does not look like a URL. Use the form http://192.168.1.50:11434.",
    scheme_not_http="Only plain http endpoints are allowed; a local Ollama does not use https.",
    credentials_not_allowed="Remove the username and password from the URL.",
    path_not_allowed="Give the base URL only, with no path, query or fragment.",
    port_required="Add the port, usually 11434.",
    invalid_port="That port number is not valid.",
    host_not_local="Only localhost, host.docker.internal and private network addresses "
    "are allowed. This setting is local-network-only.",
    unresolvable="That name could not be resolved on this network.",
)


def _refuse(code: str) -> EndpointError:
    return EndpointError(code, _REFUSALS[code])


# --- validation ---------------------------------------------------------------------------


def _is_local_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Loopback, RFC1918 private IPv4, or IPv6 loopback / ULA / link-local.

    Deliberately narrower than :attr:`is_private`, which also covers carrier
    NAT, IPv4 link-local (and with it the cloud metadata address) and several
    reserved ranges that have nothing to do with a home or office network.
    """
    if isinstance(address, ipaddress.IPv4Address):
        if address.is_loopback:
            return True
        return any(
            address in network
            for network in (
                ipaddress.ip_network("10.0.0.0/8"),
                ipaddress.ip_network("172.16.0.0/12"),
                ipaddress.ip_network("192.168.0.0/16"),
            )
        )
    if address.ipv4_mapped is not None:
        # An IPv4 address wearing an IPv6 coat: refuse rather than unwrap.
        return False
    return address.is_loopback or _ula(address)


def _ula(address: ipaddress.IPv6Address) -> bool:
    return address in ipaddress.ip_network("fc00::/7") or address in ipaddress.ip_network(
        "fe80::/10"
    )


def _port_of(parts: Any) -> int:
    try:
        port = parts.port
    except ValueError as exc:
        raise _refuse("invalid_port") from exc
    if port is None:
        raise _refuse("port_required")
    if not 1 <= port <= 65535:
        raise _refuse("invalid_port")
    return port


def normalize_endpoint(raw: object) -> str:
    """The one stored form of an acceptable endpoint, or :class:`EndpointError`."""
    if isinstance(raw, bool) or not isinstance(raw, str):
        raise _refuse("invalid_endpoint")
    text = raw.strip()
    if not text or len(text) > MAX_ENDPOINT_LENGTH or not text.isascii():
        raise _refuse("invalid_endpoint")
    if _CONTROL.search(text):
        raise _refuse("invalid_endpoint")
    try:
        parts = urlsplit(text)
    except ValueError as exc:
        raise _refuse("invalid_endpoint") from exc
    if parts.scheme.lower() != "http":
        raise _refuse("scheme_not_http")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise _refuse("path_not_allowed")
    if "@" in parts.netloc:
        raise _refuse("credentials_not_allowed")
    host = (parts.hostname or "").lower()
    if not host:
        raise _refuse("invalid_endpoint")
    port = _port_of(parts)
    if host in LOCAL_ALIASES:
        return f"http://{host}:{port}"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        raise _refuse("host_not_local") from None
    if not _is_local_ip(address):
        raise _refuse("host_not_local")
    shown = f"[{address.compressed}]" if address.version == 6 else address.compressed
    return f"http://{shown}:{port}"


def is_model_name(value: object) -> bool:
    """Whether this is plausibly an Ollama model name, e.g. ``qwen3:8b``."""
    return (
        isinstance(value, str)
        and 0 < len(value) <= MAX_MODEL_NAME
        and bool(_MODEL_NAME.match(value))
    )


def resolve_local_alias(
    host: str,
    *,
    getaddrinfo: Callable[..., Any] = socket.getaddrinfo,
) -> list[str]:
    """The local addresses a host stands for, refusing anything that is not local.

    An IP literal stands for itself and needs no lookup; ``localhost`` is
    answered without a resolver at all, because a resolver is one more thing
    that could be told to point somewhere else.
    """
    try:
        return [ipaddress.ip_address(host).compressed]
    except ValueError:
        pass
    if host == "localhost":
        return ["127.0.0.1"]
    if host not in LOCAL_ALIASES:
        raise _refuse("host_not_local")
    try:
        infos = getaddrinfo(host, None, 0, socket.SOCK_STREAM)
    except OSError as exc:
        raise _refuse("unresolvable") from exc
    found: list[str] = []
    for info in infos:
        candidate = info[4][0].split("%")[0]
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            raise _refuse("host_not_local") from None
        if not _is_local_ip(address):
            raise _refuse("host_not_local")
        if address.compressed not in found:
            found.append(address.compressed)
    if not found:
        raise _refuse("unresolvable")
    return found


# --- discovery ----------------------------------------------------------------------------


def _names(payload: object) -> list[str]:
    """The model names in an Ollama tags payload: plausible, unique, capped, sorted."""
    if not isinstance(payload, dict):
        raise DiscoveryError("unexpected_response", _DISCOVERY["unexpected_response"])
    rows = payload.get("models")
    if not isinstance(rows, list):
        raise DiscoveryError("unexpected_response", _DISCOVERY["unexpected_response"])
    found: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name") or row.get("model")
        if is_model_name(name):
            found.add(name)
    return sorted(found)[:MAX_MODELS]


_DISCOVERY = dict(
    unreachable="Could not reach Ollama at that address. Check that it is running and "
    "that it listens on the network, not only on its own loopback.",
    timeout="Ollama did not answer in time.",
    upstream_error="Ollama answered, but not with a model list.",
    unexpected_response="That address answered, but it does not look like Ollama.",
)


async def discover_models(
    endpoint: str,
    *,
    client: httpx.AsyncClient | None = None,
    getaddrinfo: Callable[..., Any] = socket.getaddrinfo,
) -> list[str]:
    """The models installed at ``endpoint``.

    Raises :class:`EndpointError` before any request when the endpoint is not
    a local one, and :class:`DiscoveryError` with a short code when the
    request is made and does not produce a model list.
    """
    checked = normalize_endpoint(endpoint)
    host = urlsplit(checked).hostname or ""
    resolve_local_alias(host, getaddrinfo=getaddrinfo)
    owned = client is None
    session = client or httpx.AsyncClient(timeout=DISCOVERY_TIMEOUT, trust_env=False)
    try:
        response = await session.get(
            f"{checked}/api/tags",
            timeout=DISCOVERY_TIMEOUT,
            follow_redirects=False,
            headers={"Accept": "application/json"},
        )
    except httpx.TimeoutException as exc:
        logger.warning("Local model discovery timed out")
        raise DiscoveryError("timeout", _DISCOVERY["timeout"]) from exc
    except httpx.HTTPError as exc:
        logger.warning("Local model discovery could not connect (%s)", type(exc).__name__)
        raise DiscoveryError("unreachable", _DISCOVERY["unreachable"]) from exc
    finally:
        if owned:
            await session.aclose()
    if response.status_code != 200:
        logger.warning("Local model discovery answered %d", response.status_code)
        raise DiscoveryError("upstream_error", _DISCOVERY["upstream_error"])
    try:
        payload = response.json()
    except ValueError as exc:
        raise DiscoveryError("unexpected_response", _DISCOVERY["unexpected_response"]) from exc
    models = _names(payload)
    logger.info("Local model discovery found %d models", len(models))
    return models


# --- what the runtime uses ----------------------------------------------------------------


def configured_endpoint(settings: dict[str, Any] | None) -> str:
    """The endpoint JARVIS should use: the saved one, else the environment, else the default.

    Never raises. An endpoint that is no longer acceptable is passed over
    rather than allowed to stop the assistant from starting.
    """
    saved = (settings or {}).get("ollama_host")
    for candidate in (saved, os.getenv("OLLAMA_HOST"), DEFAULT_ENDPOINT):
        if not candidate:
            continue
        try:
            return normalize_endpoint(candidate)
        except EndpointError as exc:
            logger.warning("Ignoring a configured Ollama endpoint (%s)", exc.code)
    return DEFAULT_ENDPOINT
