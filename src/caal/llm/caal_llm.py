"""CAALLLM - Provider-agnostic LLM wrapper for LiveKit Agents.

This module provides a LiveKit LLM interface that wraps different LLM providers
(Ollama, Groq), allowing seamless switching between backends while maintaining
full tool calling support.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from livekit.agents import llm
from livekit.agents.llm import ChatChunk, ChatContext, ChoiceDelta
from livekit.agents.llm.tool_context import FunctionTool, RawFunctionTool
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from .providers import LLMProvider, create_provider_from_settings

__all__ = ["CAALLLM"]

logger = logging.getLogger(__name__)


class CAALLLM(llm.LLM):
    """Provider-agnostic LLM wrapper for LiveKit Agents.

    This class wraps an LLMProvider instance and implements LiveKit's llm.LLM
    interface. It's designed to be used with a VoiceAssistant that overrides
    the llm_node method - the actual LLM calls are handled by llm_node(),
    not by this class's chat() method.

    The CAALLLM class:
    1. Satisfies LiveKit's llm.LLM interface (prevents "no LLM" errors)
    2. Stores the provider instance and exposes its configuration
    3. Provides model/provider info for logging and metrics

    Args:
        provider: LLMProvider instance (OllamaProvider, GroqProvider, etc.)

    Example:
        >>> from caal.llm.providers import create_provider
        >>> provider = create_provider("ollama", model="qwen3:8b")
        >>> llm = CAALLLM(provider=provider)
        >>> session = AgentSession(llm=llm, ...)
    """

    def __init__(self, provider: LLMProvider) -> None:
        super().__init__()
        self._provider = provider

        logger.debug(
            f"CAALLLM initialized with {provider.provider_name}: {provider.model}"
        )

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> "CAALLLM":
        """Create CAALLLM from settings dict.

        Factory method that creates the appropriate provider based on
        the llm_provider setting.

        Args:
            settings: Runtime settings dict with provider configuration

        Returns:
            Configured CAALLLM instance

        Example:
            >>> from caal.settings import load_settings
            >>> settings = load_settings()
            >>> llm = CAALLLM.from_settings(settings)
        """
        provider = create_provider_from_settings(settings)
        return cls(provider=provider)

    # === Provider access ===

    @property
    def provider_instance(self) -> LLMProvider:
        """Get the underlying LLM provider instance."""
        return self._provider

    # === Required LLM interface properties ===

    @property
    def model(self) -> str:
        """Model name for logging and metrics."""
        return self._provider.model

    @property
    def provider(self) -> str:
        """Provider name for logging and metrics."""
        return self._provider.provider_name

    # === Configuration accessors for llm_node ===

    @property
    def think(self) -> bool:
        """Whether to use thinking mode (Ollama/Qwen3 only)."""
        if hasattr(self._provider, "think"):
            return self._provider.think
        return False

    @property
    def temperature(self) -> float:
        """Sampling temperature."""
        if hasattr(self._provider, "temperature"):
            return self._provider.temperature
        return 0.7

    @property
    def num_ctx(self) -> int:
        """Context window size (Ollama only)."""
        if hasattr(self._provider, "num_ctx"):
            return self._provider.num_ctx
        return 8192

    # === Required LLM interface method ===

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> llm.LLMStream:
        """Create an LLM stream for chat completion.

        Note: When using VoiceAssistant with llm_node override, this method
        is bypassed. The llm_node override calls llm_node() directly.

        This implementation exists for interface compliance and fallback.
        """
        return _CAALLLMStream(
            llm=self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        """Cleanup resources."""
        # Providers don't currently need cleanup, but this is here for future use
        pass


class _CAALLLMStream(llm.LLMStream):
    """Minimal LLMStream implementation for interface compliance.

    In practice, VoiceAssistant's llm_node override bypasses this entirely.
    This exists to satisfy the type system and handle edge cases.
    """

    def __init__(
        self,
        llm: CAALLLM,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._caal_llm = llm

    async def _run(self) -> None:
        """Minimal implementation that emits a fallback response.

        This method is typically never called because VoiceAssistant's
        llm_node override handles all LLM interactions via llm_node().

        If this is called unexpectedly, it emits a placeholder response
        to prevent crashes.
        """
        request_id = str(uuid.uuid4())

        logger.warning(
            "CAALLLM._run() called directly - this usually means llm_node "
            "override is not active. Using fallback response."
        )

        chunk = ChatChunk(
            id=request_id,
            delta=ChoiceDelta(
                role="assistant",
                content=(
                    "I'm configured to use a custom LLM node. "
                    "Please ensure the agent's llm_node override is active."
                ),
            ),
        )
        self._event_ch.send_nowait(chunk)
