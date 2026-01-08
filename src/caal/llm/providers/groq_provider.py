"""Groq LLM provider implementation.

Provides Groq cloud LLM integration with tool calling support.
Uses the groq Python library for async API calls.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

from groq import AsyncGroq

from .base import LLMProvider, LLMResponse, ToolCall

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["GroqProvider"]

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    """Groq cloud LLM provider.

    Features:
        - Async API calls via groq.AsyncGroq
        - Arguments returned as JSON string (parsed automatically)
        - Tool results require name field

    Args:
        model: Groq model name (e.g., "llama-3.3-70b-versatile")
        api_key: Groq API key (reads from GROQ_API_KEY env if not set)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        self._temperature = temperature
        self._max_tokens = max_tokens

        if not self._api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self._client = AsyncGroq(api_key=self._api_key)

        logger.debug(f"GroqProvider initialized: {model}")

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model(self) -> str:
        return self._model

    @property
    def temperature(self) -> float:
        return self._temperature

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute non-streaming Groq chat completion.

        Args:
            messages: List of message dicts
            tools: Optional tool definitions
            **kwargs: Additional options

        Returns:
            Normalized LLMResponse
        """
        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
        }

        # Disable thinking for Qwen3 models (reduces latency)
        if "qwen" in self._model.lower():
            request_kwargs["reasoning_effort"] = "none"

        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**request_kwargs)

        # Extract from Groq response format
        message = response.choices[0].message

        # Extract tool calls if present
        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                # Groq returns arguments as JSON string
                args = self.parse_tool_arguments(tc.function.arguments)
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        return LLMResponse(content=message.content, tool_calls=tool_calls)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Execute streaming Groq chat completion.

        Args:
            messages: List of message dicts
            tools: Optional tool definitions (for validation of tool_calls in history)
            **kwargs: Additional options

        Yields:
            String chunks of response content
        """
        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": True,
        }

        # Disable thinking for Qwen3 models (reduces latency)
        if "qwen" in self._model.lower():
            request_kwargs["reasoning_effort"] = "none"

        # Include tools if provided (for validation of tool_calls in message history)
        if tools:
            request_kwargs["tools"] = tools

        stream = await self._client.chat.completions.create(**request_kwargs)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def parse_tool_arguments(self, arguments: Any) -> dict[str, Any]:
        """Parse Groq tool arguments from JSON string.

        Groq returns tool call arguments as a JSON string, unlike Ollama
        which returns them as a dict.

        Args:
            arguments: JSON string or dict of arguments

        Returns:
            Parsed arguments dict
        """
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool arguments: {arguments}")
                return {}
        if isinstance(arguments, dict):
            return arguments
        return {}

    def format_tool_result(
        self,
        content: str,
        tool_call_id: str | None,
        tool_name: str,
    ) -> dict[str, Any]:
        """Format tool result message for Groq.

        Groq requires the tool name in addition to tool_call_id.

        Args:
            content: Tool execution result as string
            tool_call_id: ID of the tool call being responded to
            tool_name: Name of the tool that was called

        Returns:
            Message dict for Groq API
        """
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call_id,
            "name": tool_name,  # Required by Groq
        }
