"""Protected conversation snapshot carried across a phone handoff.

When a caller moves a web/voice session to their phone, the outbound call
should pick up where the conversation left off. The only conversational state
that crosses that boundary is a small, bounded snapshot of recent user and
assistant text. Everything else (system prompts, tool calls and their outputs,
developer notes, images) stays behind.

Safety envelope, in order of application:

* only ``user``/``assistant`` text is captured, in order;
* credential-like values are redacted before the snapshot leaves the session;
* turn count, characters per turn and total characters are bounded;
* the serialized form is size-checked and re-validated when parsed back;
* the snapshot never prints its own contents;
* it is injected into the outbound session exactly once, as a private
  continuation preamble the agent must not read back.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from livekit.agents import llm

logger = logging.getLogger(__name__)

SNAPSHOT_VERSION = 1
MAX_SNAPSHOT_TURNS = 12
MAX_TURN_CHARS = 600
MAX_SNAPSHOT_CHARS = 4000
MAX_SNAPSHOT_METADATA_BYTES = 8192
REDACTED = "[REDACTED]"
CONTINUATION_MESSAGE_ID = "caal.handoff.continuation"

_ALLOWED_ROLES = frozenset({"user", "assistant"})

# --- redaction ---------------------------------------------------------------

_SECRET_KEYWORDS = (
    r"pass ?words?|pass ?codes?|pass ?phrases?|pins?|pin codes?|codes?|"
    r"api ?keys?|access keys?|secret keys?|secrets?|tokens?|access tokens?|"
    r"auth tokens?|bearer tokens?|otps?|one[- ]time codes?|verification codes?|"
    r"security codes?|cvv|cvc|ssn|social security numbers?|session ids?|"
    r"account numbers?|routing numbers?|private keys?|credentials?"
)

_REDACTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # PEM blocks, terminated or not.
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)",
        re.DOTALL,
    ),
    # "password is X", "api key = X", "the pin: X" — the value after a keyword.
    re.compile(
        rf"\b(?:{_SECRET_KEYWORDS})\b\s*(?:(?:is|was|are|were|be|of)\s*[:=]?\s*|[:=]\s*)(\S+)",
        re.IGNORECASE,
    ),
    # Bearer credentials and JWTs.
    re.compile(r"\bBearer\s+(\S+)", re.IGNORECASE),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
    # Well-known credential prefixes.
    re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"),
    # Long hex digests and UUIDs (session ids, hashes).
    re.compile(r"\b[0-9a-fA-F]{32,}\b"),
    re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
    # Payment cards: 13-19 digits with optional separators.
    re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
    # Phone numbers, national or E.164.
    re.compile(r"\+?\d{1,3}?[ .-]?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b"),
    # Any other long digit run (account numbers, garage codes).
    re.compile(r"\b\d{6,}\b"),
    # Long opaque tokens that survived everything above.
    re.compile(r"\b[A-Za-z0-9_-]{40,}\b"),
)


def _mask(match: re.Match[str]) -> str:
    if match.lastindex:
        start, end = match.span(1)
        prefix = match.group(0)[: start - match.start()]
        suffix = match.group(0)[end - match.start() :]
        return prefix + REDACTED + suffix
    return REDACTED


def redact_sensitive_text(text: str) -> str:
    """Mask credential-like values while leaving ordinary conversation intact."""
    for pattern in _REDACTION_PATTERNS:
        text = pattern.sub(_mask, text)
    return text


# --- snapshot ----------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotTurn:
    """One visible turn of the earlier conversation."""

    role: str
    text: str

    def __repr__(self) -> str:
        return f"SnapshotTurn(role={self.role!r}, chars={len(self.text)})"


@dataclass(frozen=True)
class ConversationSnapshot:
    """Bounded, redacted recent history. Never prints its contents."""

    turns: tuple[SnapshotTurn, ...] = field(default=())

    def __post_init__(self) -> None:
        for turn in self.turns:
            if turn.role not in _ALLOWED_ROLES:
                raise ValueError("Snapshot turns may only be user or assistant text")

    @property
    def total_chars(self) -> int:
        return sum(len(turn.text) for turn in self.turns)

    def __repr__(self) -> str:
        return f"ConversationSnapshot(turns={len(self.turns)}, chars={self.total_chars})"

    __str__ = __repr__

    def to_metadata(self) -> dict[str, Any]:
        """Serialize for private dispatch metadata."""
        return {
            "v": SNAPSHOT_VERSION,
            "turns": [{"role": turn.role, "text": turn.text} for turn in self.turns],
        }

    @classmethod
    def from_metadata(cls, raw: object) -> ConversationSnapshot:
        """Parse dispatch metadata, re-applying every bound and redaction.

        Raises ``ValueError`` for anything malformed, oversized, or carrying a
        role other than user/assistant. Metadata is trusted only as far as the
        dispatch channel; the contents are still treated as untrusted text.
        """
        if not isinstance(raw, dict):
            raise ValueError("Handoff context must be an object")
        try:
            encoded = len(json.dumps(raw).encode())
        except (TypeError, ValueError) as exc:
            raise ValueError("Handoff context is not serializable") from exc
        if encoded > MAX_SNAPSHOT_METADATA_BYTES:
            raise ValueError("Handoff context exceeds the size limit")
        if raw.get("v") != SNAPSHOT_VERSION:
            raise ValueError("Handoff context has an unsupported version")
        turns = raw.get("turns")
        if not isinstance(turns, list):
            raise ValueError("Handoff context turns must be a list")
        if not turns:
            raise ValueError("Handoff context carries no turns")
        if len(turns) > MAX_SNAPSHOT_TURNS:
            raise ValueError("Handoff context carries too many turns")

        parsed: list[SnapshotTurn] = []
        for item in turns:
            if not isinstance(item, dict):
                raise ValueError("Handoff context turn must be an object")
            role = item.get("role")
            text = item.get("text")
            if role not in _ALLOWED_ROLES:
                raise ValueError("Handoff context turn has a disallowed role")
            if not isinstance(text, str):
                raise ValueError("Handoff context turn text must be a string")
            parsed.append(SnapshotTurn(role=role, text=_bound_text(text)))
        bounded = _bound_turns(parsed)
        if not bounded:
            raise ValueError("Handoff context carries no usable turns")
        return cls(turns=tuple(bounded))

    def continuation_preamble(self) -> str:
        """Private system text framing the snapshot for the outbound session."""
        lines = [
            "Continuation context (private, for your reference only).",
            "The user was just talking with you in another session and asked to "
            "continue this conversation on their phone. This phone call is that "
            "continuation, so pick up naturally where things left off.",
            "The most recent turns of the earlier conversation follow. Do not read "
            "this transcript back to the user and do not repeat earlier details "
            "unless the user asks about them.",
            "",
            "[Earlier conversation]",
        ]
        for turn in self.turns:
            label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{label}: {turn.text}")
        lines.append("[End of earlier conversation]")
        return "\n".join(lines)


def _bound_text(text: str) -> str:
    """Redact, then truncate so a cut can never expose part of a secret."""
    text = redact_sensitive_text(" ".join(text.split()))
    if len(text) > MAX_TURN_CHARS:
        text = text[:MAX_TURN_CHARS]
    return text


def _bound_turns(turns: list[SnapshotTurn]) -> list[SnapshotTurn]:
    """Keep the most recent turns within the count and total-size limits."""
    kept = [turn for turn in turns if turn.text][-MAX_SNAPSHOT_TURNS:]
    while kept and sum(len(turn.text) for turn in kept) > MAX_SNAPSHOT_CHARS:
        kept.pop(0)
    return kept


def _message_text(item: Any) -> str | None:
    """Return plain text for a chat message, ignoring non-text content."""
    content = getattr(item, "content", None)
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts = [part for part in content if isinstance(part, str)]
    if not parts:
        return None
    return "\n".join(parts)


def capture_conversation_snapshot(
    items: Iterable[Any],
    *,
    exclude_assistant_texts: Iterable[str] = (),
    exclude_user_texts: Iterable[str] = (),
) -> ConversationSnapshot | None:
    """Capture recent user/assistant text from a LiveKit chat history.

    System and developer messages, tool calls and their outputs, and non-text
    content are dropped. ``exclude_*_texts`` remove the handoff's own control
    turns (the confirmation prompt, the caller's request and "yes") so the
    outbound session does not see the mechanics of getting there. Returns
    ``None`` when nothing worth carrying remains.
    """
    skip_assistant = {" ".join(text.split()) for text in exclude_assistant_texts}
    skip_user = {" ".join(text.split()) for text in exclude_user_texts}

    turns: list[SnapshotTurn] = []
    for item in items:
        if getattr(item, "type", None) != "message":
            continue
        role = getattr(item, "role", None)
        if role not in _ALLOWED_ROLES:
            continue
        text = _message_text(item)
        if text is None:
            continue
        normalized = " ".join(text.split())
        if not normalized:
            continue
        if role == "assistant" and normalized in skip_assistant:
            continue
        if role == "user" and normalized in skip_user:
            continue
        turns.append(SnapshotTurn(role=role, text=_bound_text(normalized)))

    bounded = _bound_turns(turns)
    if not bounded:
        return None
    return ConversationSnapshot(turns=tuple(bounded))


# --- injection ---------------------------------------------------------------


def inject_continuation_preamble(chat_ctx: llm.ChatContext, snapshot: ConversationSnapshot) -> bool:
    """Add the preamble as a system message, or refresh it in place; return whether it changed.

    There is only ever one continuation preamble per context. The same context
    again is a no-op; newer context (a later phone leg) replaces the existing
    message in its original position rather than adding a second one. The
    replacement is a new message object, so a shallow copy's source is never
    mutated.
    """
    preamble = snapshot.continuation_preamble()
    index = chat_ctx.index_by_id(CONTINUATION_MESSAGE_ID)
    if index is None:
        chat_ctx.add_message(role="system", content=preamble, id=CONTINUATION_MESSAGE_ID)
        return True
    existing = chat_ctx.items[index]
    if getattr(existing, "text_content", None) == preamble:
        return False
    chat_ctx.items[index] = llm.ChatMessage(
        role="system",
        content=[preamble],
        id=CONTINUATION_MESSAGE_ID,
        created_at=existing.created_at,
    )
    return True


class _ContextAgent(Protocol):
    @property
    def chat_ctx(self) -> llm.ChatContext: ...

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> Any: ...


async def restore_conversation_context(
    agent: _ContextAgent, snapshot: ConversationSnapshot
) -> bool:
    """Give an agent the snapshot as private context; refresh it when it is newer.

    Returns whether the agent's history changed. Logs only counts, never
    contents.
    """
    chat_ctx = agent.chat_ctx.copy()
    if not inject_continuation_preamble(chat_ctx, snapshot):
        logger.debug("Handoff context already restored; skipping")
        return False
    await agent.update_chat_ctx(chat_ctx)
    logger.info(
        "Restored handoff context turns=%d chars=%d", len(snapshot.turns), snapshot.total_chars
    )
    return True
