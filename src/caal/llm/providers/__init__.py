"""LLM provider implementations for CAAL.

This package provides a unified interface for different LLM backends,
enabling CAAL to work with Ollama, Groq, and potentially other providers
while sharing common tool orchestration logic.

Providers:
    - OllamaProvider: Local Ollama with think parameter support
    - GroqProvider: Groq cloud API

Example:
    >>> from caal.llm.providers import create_provider
    >>>
    >>> # Create Ollama provider
    >>> provider = create_provider("ollama", model="qwen3:8b", think=False)
    >>>
    >>> # Create Groq provider
    >>> provider = create_provider("groq", model="llama-3.3-70b-versatile")
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .base import LLMProvider, LLMResponse, ToolCall
from .groq_provider import GroqProvider
from .ollama_provider import OllamaProvider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "ToolCall",
    "OllamaProvider",
    "GroqProvider",
    "create_provider",
]

logger = logging.getLogger(__name__)


def create_provider(
    provider_name: str,
    **kwargs: Any,
) -> LLMProvider:
    """Factory function to create an LLM provider by name.

    Args:
        provider_name: Provider identifier ("ollama" or "groq")
        **kwargs: Provider-specific configuration options

    Returns:
        Configured LLMProvider instance

    Raises:
        ValueError: If provider_name is not recognized

    Example:
        >>> provider = create_provider(
        ...     "ollama",
        ...     model="qwen3:8b",
        ...     think=False,
        ...     temperature=0.7,
        ... )
    """
    provider_name = provider_name.lower()

    if provider_name == "ollama":
        return OllamaProvider(**kwargs)
    elif provider_name == "groq":
        return GroqProvider(**kwargs)
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider_name}. "
            f"Supported providers: ollama, groq"
        )


def create_provider_from_settings(settings: dict[str, Any]) -> LLMProvider:
    """Create an LLM provider from CAAL settings dict.

    This function reads the provider type and model settings from the
    runtime settings dictionary and creates the appropriate provider.

    Args:
        settings: Runtime settings dict with keys like:
            - llm_provider: "ollama" or "groq"
            - model: Ollama model name
            - groq_model: Groq model name
            - temperature: Sampling temperature
            - num_ctx: Context window size (Ollama only)

    Returns:
        Configured LLMProvider instance

    Example:
        >>> from caal.settings import load_settings
        >>> settings = load_settings()
        >>> provider = create_provider_from_settings(settings)
    """
    provider_name = settings.get("llm_provider", "ollama").lower()

    if provider_name == "ollama":
        return OllamaProvider(
            model=settings.get("ollama_model", "qwen3:8b"),
            base_url=settings.get("ollama_host"),
            think=settings.get("think", False),
            temperature=settings.get("temperature", 0.7),
            num_ctx=settings.get("num_ctx", 8192),
        )
    elif provider_name == "groq":
        # API key from settings, fallback to environment variable
        api_key = settings.get("groq_api_key") or os.environ.get("GROQ_API_KEY")
        return GroqProvider(
            model=settings.get("groq_model", "llama-3.3-70b-versatile"),
            api_key=api_key,
            temperature=settings.get("temperature", 0.7),
        )
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider_name}. "
            f"Supported providers: ollama, groq"
        )
