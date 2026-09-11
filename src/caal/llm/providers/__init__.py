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

from caal.local_ollama import configured_endpoint

from .base import LLMProvider, LLMResponse, ToolCall
from .groq_provider import GroqProvider
from .hermes_provider import HermesProvider
from .ollama_provider import OllamaProvider
from .routed_provider import RoutedProvider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "ToolCall",
    "OllamaProvider",
    "GroqProvider",
    "HermesProvider",
    "RoutedProvider",
    "create_provider",
    "create_provider_from_settings",
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
    elif provider_name == "hermes":
        return HermesProvider(**kwargs)
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider_name}. "
            "Supported providers: routed, ollama, groq, hermes"
        )


def _ollama_from_settings(settings: dict[str, Any]) -> OllamaProvider:
    # The endpoint an operator saved in the settings UI wins over OLLAMA_HOST,
    # and one that is no longer acceptable falls back rather than being used.
    return OllamaProvider(
        model=settings.get("ollama_model", "qwen3:8b"),
        base_url=configured_endpoint(settings),
        think=settings.get("think", False),
        temperature=settings.get("temperature", 0.7),
        num_ctx=settings.get("num_ctx", 8192),
    )


def _hermes_from_settings(settings: dict[str, Any]) -> HermesProvider | None:
    """The Hermes escalation, or ``None`` when this deployment has no credentials.

    A missing key is not an error: the local model is the main model, and a
    deployment that has not configured Hermes simply keeps every turn local.
    """
    api_key = settings.get("hermes_api_key") or os.environ.get("HERMES_API_KEY")
    if not api_key:
        logger.info("No Hermes credentials configured; every turn stays on the local model")
        return None
    return HermesProvider(
        base_url=settings.get("hermes_api_url", "http://host.docker.internal:8642/v1"),
        api_key=api_key,
        model=settings.get("hermes_model", "hermes-agent"),
    )


def create_provider_from_settings(settings: dict[str, Any]) -> LLMProvider:
    """Create an LLM provider from CAAL settings dict.

    This function reads the provider type and model settings from the
    runtime settings dictionary and creates the appropriate provider.

    Args:
        settings: Runtime settings dict with keys like:
            - llm_provider: "routed" (default), "ollama", "hermes" or "groq"
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
    provider_name = settings.get("llm_provider", "routed").lower()

    if provider_name == "routed":
        # The default: the local model is JARVIS' main model, and Hermes is
        # the escalation for work that needs an agent harness.
        return RoutedProvider(
            primary=_ollama_from_settings(settings),
            escalation=_hermes_from_settings(settings),
        )
    elif provider_name == "ollama":
        return _ollama_from_settings(settings)
    elif provider_name == "groq":
        # API key from settings, fallback to environment variable
        api_key = settings.get("groq_api_key") or os.environ.get("GROQ_API_KEY")
        return GroqProvider(
            model=settings.get("groq_model", "llama-3.3-70b-versatile"),
            api_key=api_key,
            temperature=settings.get("temperature", 0.7),
        )
    elif provider_name == "hermes":
        return HermesProvider(
            base_url=settings.get("hermes_api_url", "http://host.docker.internal:8642/v1"),
            api_key=settings.get("hermes_api_key") or os.environ.get("HERMES_API_KEY"),
            model=settings.get("hermes_model", "hermes-agent"),
        )
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider_name}. "
            "Supported providers: routed, ollama, groq, hermes"
        )
