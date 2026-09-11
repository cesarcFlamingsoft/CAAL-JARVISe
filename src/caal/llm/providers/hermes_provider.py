"""Hermes Agent API provider for CAAL.

Hermes exposes an authenticated, OpenAI-compatible API server.  This adapter
lets the LiveKit voice agent use that full agent runtime as its LLM backend
instead of relying on a local chat model.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .base import LLMProvider, LLMResponse

__all__ = ["HermesProvider"]

_SSE_DATA_PREFIX = "data:"
_SSE_DONE = "[DONE]"


class HermesProvider(LLMProvider):
    """Call a local Hermes API server through its OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        base_url: str = "http://host.docker.internal:8642/v1",
        api_key: str | None = None,
        model: str = "hermes-agent",
        timeout: float = 180.0,
    ) -> None:
        if not api_key:
            raise ValueError("Hermes API key required. Configure hermes_api_key.")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        # One client per provider keeps the Hermes connection (and its TLS/TCP
        # handshake) warm across turns instead of paying setup on every reply.
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    @property
    def provider_name(self) -> str:
        return "hermes"

    @property
    def model(self) -> str:
        return self._model

    @property
    def manages_own_tools(self) -> bool:
        """Hermes executes its own tool loop inside its own process."""
        return True

    # A caller that runs work far longer than a voice turn -- a delegated
    # coding job, for one -- can raise the bound for its own request without
    # letting an ordinary turn hang for that long.
    accepts_request_timeout = True

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        """Close the shared client; safe to call more than once."""
        async with self._client_lock:
            client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    def _request_timeout(self, requested: float | None) -> float:
        """The bound for one request: never shorter than the configured default."""
        if requested is None:
            return self._timeout
        return max(self._timeout, float(requested))

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, messages: list[dict[str, Any]], *, stream: bool) -> dict[str, Any]:
        """Assemble the OpenAI-compatible request body Hermes expects.

        Hermes owns its own tool loop, so CAAL tool schemas are deliberately not
        forwarded; they would otherwise suggest tools the Hermes runtime cannot
        execute inside its independently configured process.
        """
        # LiveKit calls generate_reply for the initial greeting before any caller
        # speech exists. Hermes' API correctly rejects a system-only request, so
        # add a neutral user turn that asks it to carry out the greeting.
        request_messages = list(messages)
        has_user_turn = any(
            message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and message["content"].strip()
            for message in request_messages
        )
        if not has_user_turn:
            request_messages.append(
                {"role": "user", "content": "Follow the system instruction above."}
            )

        return {
            "model": self._model,
            "messages": request_messages,
            "stream": stream,
        }

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        request_timeout: float | None = None,
        **_: Any,
    ) -> LLMResponse:
        """Run a completed Hermes turn and return its spoken response.

        ``request_timeout`` raises the bound for this one request only. Work
        that legitimately outlives a voice turn needs it; the shared client
        keeps its short default so an ordinary turn still fails fast.
        """
        payload = self._build_payload(messages, stream=False)
        client = await self._get_client()
        response = await client.post(
            f"{self._base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
            timeout=self._request_timeout(request_timeout),
        )
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Hermes returned no completion choices.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Hermes returned an empty completion.")
        return LLMResponse(content=content, tool_calls=[])

    @staticmethod
    def _delta_content(data: str) -> str | None:
        """Extract the speakable text from one SSE `data:` payload.

        Returns None for anything that carries no text: role-only openers, empty
        deltas, finish frames, and malformed or truncated JSON. A dropped frame
        must degrade the reply, never abort the stream mid-sentence.
        """
        try:
            chunk = json.loads(data)
        except ValueError:
            return None
        if not isinstance(chunk, dict):
            return None
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None
        delta = first.get("delta")
        if not isinstance(delta, dict):
            return None
        content = delta.get("content")
        if not isinstance(content, str) or not content:
            return None
        return content

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Yield Hermes' reply incrementally from its SSE stream.

        Speaking each delta as it arrives removes the whole-turn wait before the
        first word reaches the caller.
        """
        payload = self._build_payload(messages, stream=True)
        client = await self._get_client()
        async with client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith(_SSE_DATA_PREFIX):
                    # Blank separators, comments and `event:` lines carry no text.
                    continue
                data = line[len(_SSE_DATA_PREFIX) :].strip()
                if data == _SSE_DONE:
                    return
                content = self._delta_content(data)
                if content:
                    yield content
