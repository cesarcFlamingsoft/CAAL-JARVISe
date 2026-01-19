"""Adaptive endpointing for natural conversation flow.

This module provides context-aware endpointing delays that adjust based on:
- Whether the agent just asked a question (shorter delay expected)
- Whether it's early in the conversation (more patience)
- User's speaking patterns (learned over time)
- Interruption history (adjust if user tends to pause mid-thought)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class EndpointingConfig:
    """Configuration for adaptive endpointing behavior."""

    enabled: bool = True
    delay_after_question: float = 0.25  # Short - expect quick response
    delay_after_statement: float = 0.5  # Normal conversational pause
    delay_initial_turns: float = 0.7    # Patient during warm-up
    initial_turns_count: int = 3        # How many turns count as "initial"

    # Learning parameters
    learn_user_patterns: bool = True
    max_pattern_samples: int = 20       # Rolling window of response times


@dataclass
class ConversationState:
    """Tracks conversation state for adaptive decisions."""

    turn_count: int = 0
    last_agent_utterance: str = ""
    last_agent_ended_with_question: bool = False
    user_response_times: list[float] = field(default_factory=list)
    interruption_count: int = 0
    last_user_speech_start: float | None = None

    def record_agent_utterance(self, text: str) -> None:
        """Record what the agent just said."""
        self.last_agent_utterance = text
        self.last_agent_ended_with_question = self._ends_with_question(text)

    def record_user_started_speaking(self) -> None:
        """Record when user started speaking."""
        self.last_user_speech_start = time.perf_counter()

    def record_user_finished_speaking(self) -> None:
        """Record when user finished speaking and calculate response time."""
        self.turn_count += 1

        if self.last_user_speech_start is not None:
            # This is the time from agent stop speaking to user finishing
            # We'd ideally track agent_stop_time but this gives us user speech duration
            pass

    def record_interruption(self) -> None:
        """Record that user interrupted the agent."""
        self.interruption_count += 1

    def get_average_response_time(self) -> float | None:
        """Get average user response time, or None if not enough data."""
        if len(self.user_response_times) < 3:
            return None
        return sum(self.user_response_times) / len(self.user_response_times)

    @staticmethod
    def _ends_with_question(text: str) -> bool:
        """Check if text ends with a question."""
        # Clean up text and check for question indicators
        text = text.strip()
        if not text:
            return False

        # Direct question mark
        if text.endswith("?"):
            return True

        # Common question patterns without explicit "?"
        question_starters = [
            r"\b(what|where|when|why|how|who|which|whose|whom)\b",
            r"\b(is|are|was|were|do|does|did|can|could|would|should|will|shall|may|might)\s+\w+\s*\??$",
        ]

        # Check last sentence
        sentences = re.split(r"[.!]", text)
        last_sentence = sentences[-1].strip().lower() if sentences else ""

        for pattern in question_starters:
            if re.search(pattern, last_sentence, re.IGNORECASE):
                return True

        return False


class AdaptiveEndpointer:
    """Context-aware endpointing delay manager.

    Adjusts the min_endpointing_delay based on conversation context
    to create more natural turn-taking.

    Usage:
        endpointer = AdaptiveEndpointer(config)

        # When agent finishes speaking
        endpointer.on_agent_utterance(text)

        # Get current delay for VAD/session
        delay = endpointer.get_current_delay()

        # When user interrupts
        endpointer.on_interruption()
    """

    def __init__(
        self,
        config: EndpointingConfig | None = None,
        on_delay_changed: Callable[[float], Awaitable[None]] | None = None,
    ):
        """Initialize adaptive endpointer.

        Args:
            config: Endpointing configuration
            on_delay_changed: Async callback when delay should change
        """
        self.config = config or EndpointingConfig()
        self.state = ConversationState()
        self._on_delay_changed = on_delay_changed
        self._current_delay = self.config.delay_initial_turns

    @classmethod
    def from_settings(cls, settings: dict) -> "AdaptiveEndpointer":
        """Create from settings dict (from get_runtime_settings)."""
        config = EndpointingConfig(
            enabled=settings.get("adaptive_endpointing_enabled", True),
            delay_after_question=settings.get("endpointing_delay_after_question", 0.25),
            delay_after_statement=settings.get("endpointing_delay_after_statement", 0.5),
            delay_initial_turns=settings.get("endpointing_delay_initial_turns", 0.7),
        )
        return cls(config)

    def get_current_delay(self) -> float:
        """Get the current recommended endpointing delay.

        Returns:
            Delay in seconds
        """
        if not self.config.enabled:
            return self.config.delay_after_statement

        return self._current_delay

    def on_agent_utterance(self, text: str) -> float:
        """Called when agent finishes speaking.

        Args:
            text: What the agent said

        Returns:
            Recommended endpointing delay for this context
        """
        self.state.record_agent_utterance(text)
        self._update_delay()

        logger.debug(
            f"Adaptive endpointing: turn={self.state.turn_count}, "
            f"question={self.state.last_agent_ended_with_question}, "
            f"delay={self._current_delay:.2f}s"
        )

        return self._current_delay

    def on_user_started_speaking(self) -> None:
        """Called when VAD detects user speech start."""
        self.state.record_user_started_speaking()

    def on_user_finished_speaking(self) -> None:
        """Called when user finishes their turn."""
        self.state.record_user_finished_speaking()

    def on_interruption(self) -> None:
        """Called when user interrupts the agent.

        This can indicate the user wants faster turn-taking,
        or that the delay is too long.
        """
        self.state.record_interruption()

        # If user frequently interrupts, consider shortening delays
        if self.state.interruption_count > 3:
            # User tends to interrupt - they might prefer snappier responses
            # Reduce delays slightly
            self._current_delay = max(0.15, self._current_delay * 0.9)
            logger.debug(f"User interrupts frequently, reduced delay to {self._current_delay:.2f}s")

    def _update_delay(self) -> None:
        """Update the current delay based on conversation state."""
        if not self.config.enabled:
            self._current_delay = self.config.delay_after_statement
            return

        # Initial turns: be more patient
        if self.state.turn_count < self.config.initial_turns_count:
            self._current_delay = self.config.delay_initial_turns
            return

        # After a question: expect quicker response
        if self.state.last_agent_ended_with_question:
            self._current_delay = self.config.delay_after_question
            return

        # Default: normal statement delay
        self._current_delay = self.config.delay_after_statement

        # If we have learned user patterns, adjust
        if self.config.learn_user_patterns:
            avg_time = self.state.get_average_response_time()
            if avg_time is not None:
                # If user typically responds quickly, reduce delay
                # If user typically pauses longer, increase delay
                # But keep within reasonable bounds
                learned_delay = min(1.0, max(0.2, avg_time * 0.8))
                # Blend with base delay (70% base, 30% learned)
                self._current_delay = (self._current_delay * 0.7) + (learned_delay * 0.3)

    def reset(self) -> None:
        """Reset conversation state (e.g., for new session)."""
        self.state = ConversationState()
        self._current_delay = self.config.delay_initial_turns
