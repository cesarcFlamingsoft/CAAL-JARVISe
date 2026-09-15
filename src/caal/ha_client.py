"""Bounded Home Assistant transport using the operator's validated local origin."""

import asyncio
import ipaddress
import json
import re
from urllib.parse import urlsplit

import aiohttp
import httpx

from .local_ollama import normalize_endpoint, resolve_local_alias

MAX_BYTES = 4 * 1024 * 1024


class HAClient:
    def __init__(self, endpoint, *, transport=None):
        self.endpoint = normalize_endpoint(endpoint)
        if any(c in endpoint for c in ["?", "#", "\\"]):
            raise ValueError("invalid_ha_endpoint")
        self.transport = transport

    def target(self, path):
        parts = urlsplit(self.endpoint)
        addresses = resolve_local_alias(parts.hostname)
        if not addresses or any(ipaddress.ip_address(a).is_link_local for a in addresses):
            raise ValueError("invalid_ha_endpoint")
        host = "[" + addresses[0] + "]" if ":" in addresses[0] else addresses[0]
        return f"http://{host}:{parts.port}{path}", {"Host": parts.netloc}

    async def request(self, method, path, *, token=None, body=None, form=None):
        return await asyncio.wait_for(
            self._request(method, path, token=token, body=body, form=form), 12
        )

    async def _request(self, method, path, *, token=None, body=None, form=None):
        allowed = (method == "GET" and path in ("/api/", "/api/config", "/api/states")) or (
            method == "POST" and path == "/auth/token"
        )
        if method == "POST" and path in (
            "/api/services/light/turn_on",
            "/api/services/light/turn_off",
        ):
            allowed = (
                isinstance(body, dict)
                and set(body) == {"entity_id"}
                and isinstance(body["entity_id"], list)
                and 1 <= len(body["entity_id"]) <= 16
                and all(
                    isinstance(e, str) and re.fullmatch(r"light\.[a-z0-9_]{1,150}", e)
                    for e in body["entity_id"]
                )
            )
        if not allowed:
            raise ValueError("unsupported_ha_operation")
        url, headers = self.target(path)
        if token:
            headers["Authorization"] = "Bearer " + token
        async with httpx.AsyncClient(
            transport=self.transport,
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(8, connect=2),
            limits=httpx.Limits(max_connections=2),
        ) as client:
            async with client.stream(
                method, url, headers=headers, json=body, data=form
            ) as response:
                if response.status_code in (401, 403):
                    raise PermissionError("ha_reconnect_required")
                if response.status_code != 200:
                    raise ValueError("ha_unavailable")
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > MAX_BYTES:
                        raise ValueError("ha_response_too_large")
                return json.loads(data)

    async def current_user(self, token):
        async def blocked_redirect(*args):
            raise ValueError("ha_redirect_refused")

        trace = aiohttp.TraceConfig()
        trace.on_request_redirect.append(blocked_redirect)
        url, headers = self.target("/api/websocket")

        async def exchange():
            async with aiohttp.ClientSession(
                trust_env=False, trace_configs=[trace], timeout=aiohttp.ClientTimeout(total=8)
            ) as session:
                async with session.ws_connect(
                    url, headers=headers, max_msg_size=65536, autoping=True
                ) as ws:
                    if (await ws.receive_json()).get("type") != "auth_required":
                        raise ValueError("invalid_ha_handshake")
                    await ws.send_json({"type": "auth", "access_token": token})
                    if (await ws.receive_json()).get("type") != "auth_ok":
                        raise PermissionError("ha_reconnect_required")
                    await ws.send_json({"id": 1, "type": "auth/current_user"})
                    data = await ws.receive_json()
                    user = data.get("result")
                    if (
                        not data.get("success")
                        or not isinstance(user, dict)
                        or not isinstance(user.get("id"), str)
                    ):
                        raise ValueError("ha_identity_unavailable")
                    return {k: user.get(k) for k in ("id", "name", "is_admin", "is_owner")}

        return await asyncio.wait_for(exchange(), 10)

    async def registry(self, token):
        """Fixed read-only registry query; never a caller-selected websocket command."""

        async def blocked_redirect(*args):
            raise ValueError("ha_redirect_refused")

        trace = aiohttp.TraceConfig()
        trace.on_request_redirect.append(blocked_redirect)
        url, headers = self.target("/api/websocket")
        async with (
            asyncio.timeout(10),
            aiohttp.ClientSession(
                trust_env=False, trace_configs=[trace], timeout=aiohttp.ClientTimeout(total=8)
            ) as session,
        ):
            async with session.ws_connect(url, headers=headers, max_msg_size=MAX_BYTES) as ws:
                if (await ws.receive_json()).get("type") != "auth_required":
                    raise ValueError("invalid_ha_handshake")
                await ws.send_json({"type": "auth", "access_token": token})
                if (await ws.receive_json()).get("type") != "auth_ok":
                    raise PermissionError("ha_reconnect_required")
                await ws.send_json({"id": 1, "type": "config/entity_registry/list"})
                data = await ws.receive_json()
                if not data.get("success") or not isinstance(data.get("result"), list):
                    raise PermissionError("ha_registry_unavailable")
                return data["result"]
