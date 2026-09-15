"""Private literal-address transport; bounded streams, no redirects or proxy environment."""

import asyncio
import ipaddress
import json
import re
from urllib.parse import urlsplit

import aiohttp

MAX_TEXT = 4096
MAX_AUDIO = 4 * 1024 * 1024


def validate_endpoint(endpoint):
    parts = urlsplit(endpoint)
    if (
        parts.scheme != "http"
        or not parts.hostname
        or not parts.port
        or parts.username
        or parts.password
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
        or any(c.isspace() for c in endpoint)
        or any(c in endpoint for c in ("?", "#", "\\"))
    ):
        raise ValueError("invalid_backend")
    address = ipaddress.ip_address(parts.hostname)
    networks = (
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "127.0.0.0/8",
        "::1/128",
        "fc00::/7",
    )
    if not any(address in ipaddress.ip_network(n) for n in networks):
        raise ValueError("backend_must_be_private")
    return endpoint.rstrip("/")


class BridgeClient:
    def __init__(self, endpoint, credential):
        self.endpoint = validate_endpoint(endpoint)
        if not isinstance(credential, str) or not 43 <= len(credential) <= 128:
            raise ValueError("invalid_credential")
        if not all(c.isalnum() or c in "_-" for c in credential) or not credential.isascii():
            raise ValueError("invalid_credential")
        self.session = aiohttp.ClientSession(
            headers={"Authorization": "Bearer " + credential},
            trust_env=False,
            timeout=aiohttp.ClientTimeout(total=95, sock_connect=2, sock_read=90),
            connector=aiohttp.TCPConnector(limit=3),
        )
        self.identity_data = None

    async def close(self):
        await self.session.close()

    async def identity(self):
        async with (
            asyncio.timeout(5),
            self.session.get(
                self.endpoint + "/satellite/v1/identity",
                allow_redirects=False,
            ) as response,
        ):
            if response.status != 200:
                raise ValueError("invalid_auth")
            data = await response.content.read(8193)
            if len(data) > 8192:
                raise ValueError("invalid_identity")
            data = json.loads(data)
            if (
                data.get("protocol") != 1
                or data.get("personal_data") is not False
                or type(data.get("device_actions")) is not bool
                or data.get("scope", "conversation")
                not in ("conversation", "states", "states_and_lights")
                or data.get("device_actions") != (data.get("scope") == "states_and_lights")
                or not re.fullmatch(
                    r"assist_satellite\.[a-z0-9_]{1,150}", str(data.get("satellite_id", ""))
                )
                or not re.fullmatch(r"[a-f0-9]{32}", str(data.get("device_id", "")))
                or not isinstance(data.get("id"), str)
            ):
                raise ValueError("invalid_identity")
            self.identity_data = data
            return data

    async def turn(self, body):
        async with self.session.post(
            self.endpoint + "/satellite/v1/turn", json=body, allow_redirects=False
        ) as response:
            if response.status != 200:
                raise ValueError("turn_failed")
            pending = b""
            total = 0
            done = False
            async for chunk in response.content.iter_chunked(1024):
                pending += chunk
                if len(pending) > 32768:
                    raise ValueError("frame_too_large")
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    data = json.loads(line)
                    if done or "error" in data:
                        raise ValueError("turn_failed")
                    if data.get("done") is True:
                        done = True
                    elif isinstance(data.get("text"), str):
                        total += len(data["text"])
                        if total > MAX_TEXT:
                            raise ValueError("text_too_large")
                        yield data["text"]
                    else:
                        raise ValueError("invalid_frame")
            if not done or pending:
                raise ValueError("incomplete_turn")

    async def cancel(self, request_id):
        async with (
            asyncio.timeout(3),
            self.session.delete(
                self.endpoint + "/satellite/v1/turn/" + request_id,
                allow_redirects=False,
            ) as response,
        ):
            if response.status != 200:
                raise ValueError("cancel_failed")

    async def audio(self, text):
        async with self.session.post(
            self.endpoint + "/satellite/v1/audio", json={"text": text}, allow_redirects=False
        ) as response:
            if (
                response.status != 200
                or response.headers.get("X-Audio-Sample-Rate") != "24000"
                or response.headers.get("X-Audio-Channels") != "1"
                or response.content_type != "audio/pcm"
            ):
                raise ValueError("audio_failed")
            total = 0
            async for chunk in response.content.iter_chunked(4096):
                total += len(chunk)
                if total > MAX_AUDIO:
                    raise ValueError("audio_too_large")
                yield chunk
            if total == 0 or total % 2:
                raise ValueError("invalid_audio")
