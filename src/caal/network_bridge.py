"""Optional macOS loopback bridge. Start one worker; never expose on a LAN interface."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import os

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from caal.network_diagnostics import SERVICES, execute, private_target, result


def timeout_for(operation):
    """Outlast the engine's own deadline, so the engine reports why, not the socket."""
    from caal.network_diagnostics import deadline_for

    return deadline_for(operation) + 15 if operation == "speedtest" else 26


def validate(operation, arguments):
    if operation not in (
        "clients",
        "target",
        "addresses",
        "status",
        "speedtest",
    ) or not isinstance(arguments, dict):
        raise ValueError("invalid operation")
    expected = {"address", "service"} if operation == "target" else set()
    if set(arguments) != expected:
        raise ValueError("invalid arguments")
    if operation == "target":
        private_target(arguments["address"])
        if arguments["service"] not in SERVICES:
            raise ValueError("invalid service")


def create_app(token):
    if not isinstance(token, str) or len(token) < 32 or not token.isascii():
        raise ValueError("a strong bridge token is required")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.post("/diagnose")
    async def diagnose(request: Request):
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied.encode(), ("Bearer " + token).encode()):
            return JSONResponse({"status": "unauthorized"}, status_code=401)
        try:
            body = bytearray()
            async with asyncio.timeout(2):
                async for chunk in request.stream():
                    if len(body) + len(chunk) > 2048:
                        return JSONResponse({"status": "too_large"}, status_code=413)
                    body.extend(chunk)
            payload = json.loads(body)
            if not isinstance(payload, dict) or set(payload) != {"operation", "arguments"}:
                raise ValueError("invalid request")
            validate(payload["operation"], payload["arguments"])
        except (ValueError, KeyError, TypeError, TimeoutError):
            return JSONResponse({"status": "invalid_request"}, status_code=400)
        return await run_until_disconnect(request, payload["operation"], payload["arguments"])

    return app


DISCONNECT_POLL_SECONDS = 0.5


async def run_until_disconnect(request, operation, arguments, *, poll=DISCONNECT_POLL_SECONDS):
    """Run the diagnostic, but stop it the moment the caller has hung up.

    A speed test owns a real browser moving real bandwidth for up to three
    minutes. If the voice turn is cancelled or the client goes away, nobody is
    left to read the result, so the work is cancelled rather than left running:
    the cancellation reaches the browser context manager, which tears the whole
    browser down.
    """
    work = asyncio.ensure_future(execute(operation, arguments, local=True))
    try:
        while True:
            done, _ = await asyncio.wait({work}, timeout=poll)
            if done:
                return work.result()
            if await request.is_disconnected():
                work.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await work
                return JSONResponse({"status": "cancelled"}, status_code=499)
    except asyncio.CancelledError:
        work.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await work
        raise


async def bounded_body(stream, cap):
    body = bytearray()
    async for chunk in stream.iter_chunked(4096):
        if len(body) + len(chunk) > cap:
            raise ValueError("bridge response too large")
        body.extend(chunk)
    return bytes(body)


async def call_host(operation, arguments):
    validate(operation, arguments)
    token = os.environ.get("CAAL_NETWORK_BRIDGE_TOKEN", "")
    if len(token) < 32:
        return result("unavailable", "The authenticated macOS network bridge is not configured.")
    # Fixed origin and path; no model-controlled URL, proxy, redirect or local API retrieval.
    host = "host.docker.internal" if os.path.exists("/.dockerenv") else "127.0.0.1"
    async with aiohttp.ClientSession(
        trust_env=False, timeout=aiohttp.ClientTimeout(total=timeout_for(operation))
    ) as session:
        async with session.post(
            f"http://{host}:18004/diagnose",
            headers={"Authorization": "Bearer " + token},
            json={"operation": operation, "arguments": arguments},
            allow_redirects=False,
        ) as response:
            if response.status != 200:
                return result("unavailable", "The host network bridge refused the request.")
            body = await bounded_body(response.content, 65536)
            return json.loads(body)


def main():
    import uvicorn

    token = os.environ.get("CAAL_NETWORK_BRIDGE_TOKEN", "")
    uvicorn.run(
        create_app(token),
        host="127.0.0.1",
        port=18004,
        workers=1,
        access_log=False,
        limit_concurrency=4,
        timeout_keep_alive=2,
    )


if __name__ == "__main__":
    main()
