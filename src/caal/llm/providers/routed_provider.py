"""Route one turn to the local model or to the Hermes agent harness.

JARVIS main model is the local Ollama one: it answers ordinary requests, with
CAAL native tools available to it. A turn that :mod:`caal.model_routing` reads
as genuinely long or multi-step is handed to Hermes instead, which runs its
own tool loop in its own process and is never given CAAL tool schemas.

The fallback between the two is bounded and unmistakable. Each turn gets at
most one attempt on each side, in a fixed order, with no retry loop: a local
model that is unreachable or answers with nothing costs one escalation, and an
escalation that fails costs one local attempt. When neither can answer, the
reply says so in plain words rather than carrying an upstream error, a host
name, or a credential back to the user.

A coding turn is claimed upstream by coding delegation and should never reach
a provider at all; if one does, it is escalated to Hermes rather than guessed
at by the small local model.

Escalating is also a privacy boundary, and it is enforced in one place: every
message list handed to the escalation goes through
:func:`~caal.llm.context_barrier.sanitize_for_escalation` first. That holds
whichever way the turn arrived there -- routed on purpose, or fallen back after
the local model failed part-way through a tool workflow, still holding the tool
result it was answering from.

Nothing here logs the messages, the reply, or an upstream error string.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from caal.llm.context_barrier import sanitize_for_escalation
from caal.model_routing import Destination, classify_request

from .base import LLMProvider, LLMResponse, ToolCall

logger = logging.getLogger(__name__)

__all__ = ["NO_MODEL_AVAILABLE_REPLY", "RoutedProvider"]

NO_MODEL_AVAILABLE_REPLY = (
    "I'm having trouble reaching my models just now, so I can't answer that. "
    "Please try me again in a moment."
)


class RoutedProvider(LLMProvider):
    """Local model first, agent harness on purpose, one bounded fallback either way."""

    def __init__(
        self,
        *,
        primary: LLMProvider,
        escalation: LLMProvider | None = None,
    ) -> None:
        self._primary = primary
        self._escalation = escalation

    # --- identity -------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "routed"

    @property
    def model(self) -> str:
        """The local model: it is the one that answers ordinary turns."""
        return self._primary.model

    @property
    def primary(self) -> LLMProvider:
        return self._primary

    @property
    def escalation(self) -> LLMProvider | None:
        return self._escalation

    @property
    def escalation_available(self) -> bool:
        return self._escalation is not None

    @property
    def supports_think(self) -> bool:
        return self._primary.supports_think

    @property
    def manages_own_tools(self) -> bool:
        """False: the local model keeps the CAAL native tool catalog."""
        return False

    async def reachable(self) -> bool:
        """Whether the local model can answer; the escalation is not a substitute.

        Hermes runs its own tool loop and never receives CAAL tool schemas, so
        for anything that needs a CAAL tool an unreachable local model is an
        outage even when Hermes is up.
        """
        return await self._primary.reachable()

    # --- routing --------------------------------------------------------------

    @staticmethod
    def _last_user_text(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages or []):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
        return ""

    def _escalates(self, messages: list[dict[str, Any]]) -> bool:
        if self._escalation is None:
            return False
        return classify_request(self._last_user_text(messages)).destination is not Destination.LOCAL

    def _order(self, messages: list[dict[str, Any]]) -> tuple[LLMProvider, LLMProvider | None]:
        """The provider to try first, and the single fallback, if any."""
        if self._escalates(messages):
            assert self._escalation is not None
            return self._escalation, self._primary
        return self._primary, self._escalation

    def _tools_for(
        self, provider: LLMProvider, tools: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        """Only the local model is offered CAAL tools; Hermes runs its own."""
        if provider is self._primary and not provider.manages_own_tools:
            return tools
        return None

    def _messages_for(
        self, provider: LLMProvider, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """The local model sees the turn as it is; the escalation sees it redacted.

        The local model is the one that read the connected accounts, and it
        needs the bounded result it is answering from. The escalation is
        another runtime entirely: connected-account results, the account labels
        the user chose, and answers already composed from either do not cross.
        """
        if provider is self._primary:
            return messages
        return sanitize_for_escalation(messages)

    @staticmethod
    def _usable(response: LLMResponse | None) -> bool:
        if response is None:
            return False
        if response.tool_calls:
            return True
        return isinstance(response.content, str) and bool(response.content.strip())

    # --- calls ----------------------------------------------------------------

    async def _attempt(
        self,
        provider: LLMProvider,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        **kwargs: Any,
    ) -> LLMResponse | None:
        """One bounded attempt. ``None`` means unreachable or unusable."""
        try:
            response = await provider.chat(
                self._messages_for(provider, messages),
                tools=self._tools_for(provider, tools),
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - the caller decides what happens next
            # No exception text: an upstream error can carry a host or a token.
            logger.warning("%s did not answer (%s)", provider.provider_name, type(exc).__name__)
            return None
        if not self._usable(response):
            logger.warning("%s answered with nothing usable", provider.provider_name)
            return None
        return response

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        first, second = self._order(messages)
        response = await self._attempt(first, messages, tools, **kwargs)
        if response is not None:
            return response
        if second is not None:
            logger.info("Falling back from %s once", first.provider_name)
            response = await self._attempt(second, messages, tools, **kwargs)
            if response is not None:
                return response
        logger.error("No model could answer the turn")
        return LLMResponse(content=NO_MODEL_AVAILABLE_REPLY, tool_calls=[])

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        first, second = self._order(messages)
        spoken = False
        try:
            async for chunk in first.chat_stream(
                self._messages_for(first, messages),
                tools=self._tools_for(first, tools),
                **kwargs,
            ):
                spoken = True
                yield chunk
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s stream failed (%s)", first.provider_name, type(exc).__name__)
        if spoken:
            # Restarting after speech has begun would repeat the first half of
            # the answer out loud; a truncated reply is the lesser failure.
            return
        if second is not None:
            logger.info("Falling back from the %s stream once", first.provider_name)
            try:
                async for chunk in second.chat_stream(
                    self._messages_for(second, messages),
                    tools=self._tools_for(second, tools),
                    **kwargs,
                ):
                    spoken = True
                    yield chunk
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s stream failed (%s)", second.provider_name, type(exc).__name__)
        if not spoken:
            logger.error("No model could stream the turn")
            yield NO_MODEL_AVAILABLE_REPLY

    # --- tool plumbing --------------------------------------------------------
    #
    # Tools only ever go to the local model, so its formats are the ones that
    # matter here.

    def parse_tool_arguments(self, arguments: Any) -> dict[str, Any]:
        return self._primary.parse_tool_arguments(arguments)

    def format_tool_result(
        self, content: str, tool_call_id: str | None, tool_name: str
    ) -> dict[str, Any]:
        return self._primary.format_tool_result(content, tool_call_id, tool_name)

    def format_tool_call_message(
        self, content: str | None, tool_calls: list[ToolCall]
    ) -> dict[str, Any]:
        return self._primary.format_tool_call_message(content, tool_calls)

    async def aclose(self) -> None:
        for provider in (self._primary, self._escalation):
            if provider is None:
                continue
            try:
                await provider.aclose()
            except Exception:  # noqa: BLE001
                logger.warning("Could not close %s cleanly", provider.provider_name)
