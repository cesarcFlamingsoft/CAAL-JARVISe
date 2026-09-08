"""Ollama LLM provider implementation.

Provides native Ollama integration with think parameter support for Qwen3 models.
Uses the ollama Python library for direct API calls.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import ollama

from .base import LLMProvider, LLMResponse, ToolCall

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["OllamaProvider"]

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama LLM provider with think parameter support.

    Features:
        - Supports Qwen3's think parameter for low-latency responses
        - Arguments returned as dict (no parsing needed)
        - Tool results don't require name field

    Args:
        model: Ollama model name (e.g., "qwen3:8b", "llama3.2:3b")
        think: Enable Qwen3 thinking mode. False for lower latency.
        temperature: Sampling temperature (0.0-2.0)
        top_p: Nucleus sampling threshold (0.0-1.0)
        top_k: Top-k sampling limit
        num_ctx: Context window size
        base_url: Ollama server URL (read from OLLAMA_HOST env if not set)
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
        think: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        num_ctx: int = 8192,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._think = think
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._num_ctx = num_ctx
        self._base_url = base_url

        # Create client with custom host if provided
        self._client = ollama.Client(host=base_url) if base_url else ollama.Client()

        logger.debug(
            f"OllamaProvider initialized: {model} "
            f"(think={think}, num_ctx={num_ctx}, host={base_url or 'default'})"
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    @property
    def supports_think(self) -> bool:
        return True

    @property
    def think(self) -> bool:
        """Whether to use Qwen3 thinking mode."""
        return self._think

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def num_ctx(self) -> int:
        return self._num_ctx

    def _get_options(self) -> dict[str, Any]:
        """Get Ollama options dict."""
        return {
            "temperature": self._temperature,
            "top_p": self._top_p,
            "top_k": self._top_k,
            "num_ctx": self._num_ctx,
        }

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute non-streaming Ollama chat completion.

        Args:
            messages: List of message dicts
            tools: Optional tool definitions
            **kwargs: Additional options (think override, etc.)

        Returns:
            Normalized LLMResponse
        """
        think = kwargs.get("think", self._think)
        options = self._get_options()

        # Run sync client.chat in thread pool
        response = await asyncio.to_thread(
            self._client.chat,
            model=self._model,
            messages=messages,
            tools=tools,
            think=think,
            stream=False,
            options=options,
        )

        # Extract tool calls if present
        tool_calls: list[ToolCall] = []
        if hasattr(response, "message"):
            msg = response.message
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(
                        ToolCall(
                            id=getattr(tc, "id", "") or "",
                            name=tc.function.name,
                            arguments=tc.function.arguments or {},
                        )
                    )

        # Extract content
        content = None
        if hasattr(response, "message") and hasattr(response.message, "content"):
            content = response.message.content

        return LLMResponse(content=content, tool_calls=tool_calls)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Execute streaming Ollama chat completion.

        Args:
            messages: List of message dicts
            tools: Optional list of tool definitions (for validation of tool_calls in history)
            **kwargs: Additional options

        Yields:
            String chunks of response content
        """
        think = kwargs.get("think", self._think)
        options = self._get_options()

        # Run sync client.chat with streaming in thread pool
        # We need to iterate in a thread since client.chat returns a sync iterator
        def _stream():
            return self._client.chat(
                model=self._model,
                messages=messages,
                tools=tools,
                think=think,
                stream=True,
                options=options,
            )

        response = await asyncio.to_thread(_stream)

        # Iterate over chunks (sync iterator from thread)
        for chunk in response:
            if hasattr(chunk, "message") and hasattr(chunk.message, "content"):
                if chunk.message.content:
                    yield chunk.message.content

    # parse_tool_arguments: Use default (Ollama returns dict)
    # format_tool_result: Use default (no name field needed)

    def format_tool_call_message(
        self,
        content: str | None,
        tool_calls: list[ToolCall],
    ) -> dict[str, Any]:
        """Format assistant message containing tool calls for Ollama.

        Ollama requires arguments as dict, not JSON string.

        Args:
            content: Optional assistant content
            tool_calls: List of tool calls to include

        Returns:
            Message dict for assistant with tool calls
        """
        return {
            "role": "assistant",
            "content": content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,  # Ollama needs dict, not JSON string
                    },
                }
                for tc in tool_calls
            ],
        }
