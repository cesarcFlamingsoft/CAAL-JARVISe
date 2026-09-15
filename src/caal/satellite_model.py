"""Cancellable bounded local transport for the existing Jarvis conversation/tool loop."""

import json

import httpx

from caal import settings
from caal.ha_client import HAClient
from caal.llm.providers.base import LLMProvider, LLMResponse, ToolCall
from caal.llm.providers.ollama_provider import OllamaProvider
from caal.local_ollama import configured_endpoint


class SatelliteModel(LLMProvider):
    provider_name = "ollama"
    format_tool_call_message = OllamaProvider.format_tool_call_message

    def __init__(self, *, transport=None):
        config = settings.load_settings()
        self._model = config["ollama_model"]
        self.target = HAClient(configured_endpoint(config))
        self.client = httpx.AsyncClient(
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(30, connect=2),
            limits=httpx.Limits(max_connections=1),
        )
        self.think_fields = None

    @property
    def model(self):
        return self._model

    async def aclose(self):
        await self.client.aclose()

    async def _read(self, path, body):
        url, headers = self.target.target(path)
        data = bytearray()
        async with self.client.stream("POST", url, headers=headers, json=body) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(4096):
                data.extend(chunk)
                if len(data) > (2 * 1024 * 1024 if path == "/api/show" else 65536):
                    raise ValueError("model_response_too_large")
        return json.loads(data)

    async def _body(self, messages, tools, stream):
        if self.think_fields is None:
            info = await self._read("/api/show", {"model": self.model})
            self.think_fields = (
                {"think": False} if "thinking" in info.get("capabilities", []) else {}
            )
        return dict(
            model=self.model,
            messages=messages,
            tools=tools,
            stream=stream,
            options={"num_predict": 1024, "num_ctx": 8192, "temperature": 0.7},
            **self.think_fields,
        )

    async def chat(self, messages, tools=None, **kwargs):
        result = await self._read("/api/chat", await self._body(messages, tools, False))
        message = result["message"]
        calls = message.get("tool_calls", [])
        if len(calls) > 4:
            raise ValueError("too_many_tools")
        return LLMResponse(
            message.get("content"),
            [
                ToolCall(
                    str(i),
                    call["function"]["name"],
                    call["function"].get("arguments", {}),
                )
                for i, call in enumerate(calls)
            ],
        )

    async def chat_stream(self, messages, tools=None, **kwargs):
        body = await self._body(messages, tools, True)
        url, headers = self.target.target("/api/chat")
        pending = b""
        total = 0
        async with self.client.stream("POST", url, headers=headers, json=body) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(1024):
                total += len(chunk)
                if total > 262144:
                    raise ValueError("model_response_too_large")
                pending += chunk
                if len(pending) > 65536:
                    raise ValueError("model_frame_too_large")
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if line.strip():
                        data = json.loads(line)
                        if data.get("error") or data.get("message", {}).get("tool_calls"):
                            raise ValueError("unexpected_model_stream")
                        if text := data.get("message", {}).get("content"):
                            yield text
            if pending.strip():
                raise ValueError("incomplete_model_stream")
