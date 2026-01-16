"""
CAAL - Voice Assistant
======================

A modular voice assistant with n8n workflow integrations and multi-provider LLM support.

Core Components:
    CAALLLM: Provider-agnostic LLM wrapper (supports Ollama, Groq)
    OllamaLLM: Native Ollama LLM with think parameter support for Qwen3 (deprecated)

STT/TTS:
    - Speaches container for Faster-Whisper STT
    - Kokoro container for TTS

Integrations:
    n8n: Workflow discovery and execution via n8n MCP

Example:
    >>> from caal import CAALLLM
    >>> from caal.settings import load_settings
    >>>
    >>> settings = load_settings()
    >>> llm = CAALLLM.from_settings(settings)

Repository: https://github.com/CoreWorxLab/caal
License: MIT
"""

__version__ = "0.1.0"
__author__ = "CoreWorxLab"

from .llm import CAALLLM, OllamaLLM

__all__ = [
    "CAALLLM",
    "OllamaLLM",  # Backward compatibility
    "__version__",
]
