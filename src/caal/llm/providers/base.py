"""Abstract base class for LLM providers.

This module defines the interface that all LLM providers must implement,
enabling CAAL to work with different LLM backends (Ollama, Groq, etc.)
while sharing common tool orchestration logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["LLMProvider", "LLMResponse", "ToolCall"]


@dataclass
class ToolCall:
    """Normalized tool call representation."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Normalized LLM response representation."""

    content: str | None
    tool_calls: list[ToolCall]


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Providers handle the differences between LLM APIs:
    - Ollama: Uses ollama.chat(), arguments as dict, no name in tool results
    - Groq: Uses AsyncGroq, arguments as JSON string, requires name in tool results

    The llm_node uses this interface to remain provider-agnostic while
    maintaining full tool calling support.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider identifier (e.g., 'ollama', 'groq')."""
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        """Return the model name."""
        ...

    @property
    def supports_think(self) -> bool:
        """Whether provider supports think parameter (Qwen3 specific)."""
        return False

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute non-streaming chat completion.

        Args:
            messages: List of message dicts with role/content
            tools: List of tool definitions in OpenAI format
            **kwargs: Provider-specific options

        Returns:
            Normalized LLMResponse with content and/or tool_calls
        """
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Execute streaming chat completion.

        Args:
            messages: List of message dicts with role/content
            tools: Optional tool definitions (for validation of tool_calls in history)
            **kwargs: Provider-specific options

        Yields:
            String chunks of the response content
        """
        ...

    def parse_tool_arguments(self, arguments: Any) -> dict[str, Any]:
        """Parse tool call arguments to dict.

        Override for providers that return arguments as JSON string (e.g., Groq).
        Default implementation assumes arguments are already a dict (e.g., Ollama).

        Args:
            arguments: Raw arguments from tool call

        Returns:
            Parsed arguments dict
        """
        if isinstance(arguments, dict):
            return arguments
        return {}

    def format_tool_result(
        self,
        content: str,
        tool_call_id: str | None,
        tool_name: str,
    ) -> dict[str, Any]:
        """Format tool result message for this provider.

        Override for providers that require additional fields (e.g., Groq needs name).
        Default implementation uses Ollama's format.

        Args:
            content: Tool execution result as string
            tool_call_id: ID of the tool call being responded to
            tool_name: Name of the tool that was called

        Returns:
            Message dict to append to conversation
        """
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call_id,
        }

    def format_tool_call_message(
        self,
        content: str | None,
        tool_calls: list[ToolCall],
    ) -> dict[str, Any]:
        """Format assistant message containing tool calls.

        Args:
            content: Optional assistant content
            tool_calls: List of tool calls to include

        Returns:
            Message dict for assistant with tool calls
        """
        import json

        return {
            "role": "assistant",
            "content": content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        # Arguments must be JSON string for Groq compatibility
                        "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else str(tc.arguments),
                    },
                }
                for tc in tool_calls
            ],
        }
