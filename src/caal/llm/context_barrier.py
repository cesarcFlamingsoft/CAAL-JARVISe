"""A one-way barrier for connected-account knowledge.

The local Ollama model is JARVIS main model, and it is the one that reads the
connected email and calendar accounts of the signed-in user: it holds the
user-scoped knowledge tools, calls one, and composes its spoken answer from the
bounded, speech-safe result. Hermes is a separate agent runtime in a separate
process, with its own tool loop and its own model. It is never given the CAAL
tool schemas, and this module is what makes sure it is never given their output
either -- not the results, not the account labels the user chose, not a preview,
and not an answer already composed from any of them.

Three carriers exist, and each is closed here:

``ToolDataCache``
    injected recent tool data into the context of every later turn. Connected
    account data is now refused outright: it lives exactly as long as the turn
    that read it, which is all the local model needs to answer.

The tool and function messages of the turn itself
    would be handed to Hermes verbatim by the bounded fallback in
    :class:`~caal.llm.providers.routed_provider.RoutedProvider` when the local
    model failed after a tool call. :func:`sanitize_for_escalation` drops them.

The answer already spoken from that data
    comes back on later turns as ordinary assistant transcript.
    :class:`PrivateAnswerLedger` remembers those answers as salted hashes -- it
    stores no plaintext and writes nothing to a log -- so the barrier can
    recognise and redact them later.

What the barrier deliberately leaves alone: the system prompt, the user own
words, ordinary conversation, and non-knowledge tool workflows. Escalating a
turn must cost the user context, not their conversation.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "REDACTED_ANSWER",
    "TOOL_DATA_HEADER",
    "PrivateAnswerLedger",
    "default_ledger",
    "forget_private_answers",
    "is_knowledge_tool",
    "knowledge_tool_names",
    "record_private_answer",
    "sanitize_for_escalation",
]

# What stands in for a redacted answer. It says that something was withheld, so
# the other runtime is not left to infer a gap, and names nothing that was in it.
REDACTED_ANSWER = (
    "[Withheld: this turn was answered from the connected email and calendar "
    "accounts of the signed-in user. Those contents are private to that "
    "runtime and are not available here.]"
)

# The first line of the context block ToolDataCache injects.
TOOL_DATA_HEADER = "Recent tool response data for reference:"

_KNOWLEDGE_CATEGORY = "knowledge"
# The namespaces of the connected-account tools. Kept as a rule as well as a
# lookup so a catalog that cannot be built, or a tool added later under the same
# namespace, is still treated as private rather than quietly passed along.
_PRIVATE_PREFIXES = ("inbox.", "schedule.")

_names: frozenset[str] | None = None


def knowledge_tool_names() -> frozenset[str]:
    """The registered connected-account tools, read from the catalog once."""
    global _names
    if _names is None:
        try:
            from caal.tools import create_default_registry

            _names = frozenset(
                tool.name
                for tool in create_default_registry().list()
                if tool.category == _KNOWLEDGE_CATEGORY
            )
        except Exception as exc:  # noqa: BLE001 - the prefix rule still holds
            logger.warning("Could not read the tool catalog (%s)", type(exc).__name__)
            return frozenset()
    return _names


def is_knowledge_tool(name: object) -> bool:
    """Whether a tool reads the connected accounts of the signed-in user."""
    if not isinstance(name, str) or not name:
        return False
    cleaned = name.strip()
    if cleaned in knowledge_tool_names():
        return True
    return cleaned.startswith(_PRIVATE_PREFIXES)


# --- what has already been spoken ---------------------------------------------------------------


class PrivateAnswerLedger:
    """Remembers answers composed from connected-account data, as hashes only.

    A spoken answer returns on later turns as ordinary assistant transcript,
    where nothing about its shape says where it came from. The ledger is how
    the barrier recognises one. It holds a keyed hash of the normalised text
    and never the text: the salt is per process and random, so the entries are
    useless to anything that reads them, including a crash dump.

    Bounded on purpose. An answer old enough to have fallen out of the ledger
    has also fallen out of the sliding context window that would carry it.
    """

    def __init__(self, max_entries: int = 64) -> None:
        self._salt = secrets.token_bytes(16)
        self._seen: deque[str] = deque(maxlen=max(1, int(max_entries)))

    @property
    def fingerprints(self) -> tuple[str, ...]:
        """The stored hashes. Exposed so a test can prove there is no plaintext."""
        return tuple(self._seen)

    def _fingerprint(self, text: str) -> str:
        normalised = " ".join(text.split()).casefold().encode("utf-8", "replace")
        return hashlib.blake2b(normalised, key=self._salt, digest_size=16).hexdigest()

    def record(self, text: object) -> None:
        """Mark one spoken answer as private. Never logs or stores its words."""
        if not isinstance(text, str) or not text.strip():
            return
        fingerprint = self._fingerprint(text)
        if fingerprint not in self._seen:
            self._seen.append(fingerprint)

    def matches(self, text: object) -> bool:
        if not isinstance(text, str) or not text.strip() or not self._seen:
            return False
        return self._fingerprint(text) in self._seen

    def clear(self) -> None:
        self._seen.clear()


_ledger = PrivateAnswerLedger()


def default_ledger() -> PrivateAnswerLedger:
    """The process-wide ledger.

    Shared deliberately: a false positive costs one redacted line of context, a
    miss costs a private answer. Nothing user-identifying is stored, so sharing
    it across sessions cannot cross-contaminate anything but that trade.
    """
    return _ledger


def record_private_answer(text: object) -> None:
    """Record an answer that was composed from connected-account data."""
    _ledger.record(text)


def forget_private_answers() -> None:
    _ledger.clear()


# --- the barrier itself ---------------------------------------------------------------------------


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls")
    return [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []


def _call_name(call: dict[str, Any]) -> str:
    function = call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str):
            return name
    name = call.get("name")
    return name if isinstance(name, str) else ""


def _known_and_private_ids(messages: list[Any]) -> tuple[set[str], set[str]]:
    """Every tool call id in the list, and the subset that read private data."""
    known: set[str] = set()
    private: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        for call in _tool_calls(message):
            identifier = call.get("id")
            if not isinstance(identifier, str):
                continue
            known.add(identifier)
            if is_knowledge_tool(_call_name(call)):
                private.add(identifier)
    return known, private


def _filtered_tool_data(content: str) -> str | None:
    """The injected tool-data block with every connected-account line removed."""
    kept = [TOOL_DATA_HEADER]
    for line in content.splitlines()[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        name = stripped.split(":", 1)[0].strip()
        if is_knowledge_tool(name):
            continue
        kept.append(line)
    return "\n".join(kept) if len(kept) > 1 else None


def _is_private_result(message: dict[str, Any], known: set[str], private: set[str]) -> bool:
    """Whether a tool or function result carries connected-account data.

    A result that no call in this list vouches for is dropped rather than
    trusted: the escalation cannot use a dangling tool result anyway, so the
    safe reading costs nothing and the unsafe one costs the user their mail.
    """
    name = message.get("name")
    if isinstance(name, str) and name.strip():
        return is_knowledge_tool(name)
    identifier = message.get("tool_call_id") or message.get("id")
    if isinstance(identifier, str) and identifier:
        if identifier in private:
            return True
        if identifier in known:
            return False
    return True


def _sanitised_assistant(
    message: dict[str, Any],
    private: set[str],
    ledger: PrivateAnswerLedger,
    *,
    after_private: bool,
) -> dict[str, Any] | None:
    """One assistant message, with anything private removed. ``None`` drops it."""
    calls = _tool_calls(message)
    if calls:
        kept = [
            call
            for call in calls
            if not is_knowledge_tool(_call_name(call)) and call.get("id") not in private
        ]
        if len(kept) == len(calls):
            return message
        if not kept:
            # The preamble of a private tool call goes with the call: it names
            # the account and what is about to be read from it.
            return None
        cleaned = dict(message)
        cleaned["tool_calls"] = kept
        return cleaned
    content = message.get("content")
    if isinstance(content, str) and content.strip() and (after_private or ledger.matches(content)):
        redacted = dict(message)
        redacted["content"] = REDACTED_ANSWER
        return redacted
    return message


def sanitize_for_escalation(
    messages: list[Any],
    ledger: PrivateAnswerLedger | None = None,
) -> list[Any]:
    """Return ``messages`` with every trace of the connected accounts removed.

    Called on the way to a runtime that must not see them -- the Hermes
    escalation, whether the turn was routed there or fell back there after the
    local model failed mid-workflow. It is a filter, not a truncation: the
    system prompt, the words the user said, and ordinary conversation come
    through untouched, and so does a tool workflow that read nothing private.

    Nothing here logs what it removed, or how much of it there was.
    """
    if not isinstance(messages, list) or not messages:
        return list(messages or [])
    ledger = default_ledger() if ledger is None else ledger
    known, private = _known_and_private_ids(messages)

    sanitised: list[Any] = []
    after_private = False
    for message in messages:
        if not isinstance(message, dict):
            sanitised.append(message)
            continue
        role = message.get("role")

        if role in ("tool", "function"):
            if _is_private_result(message, known, private):
                # The answer the model composes from a dropped result is the
                # same data in sentences; the next assistant turn goes too.
                after_private = True
                continue
            after_private = False
            sanitised.append(message)
            continue

        if role == "assistant":
            kept = _sanitised_assistant(message, private, ledger, after_private=after_private)
            after_private = False
            if kept is not None:
                sanitised.append(kept)
            continue

        if role == "system":
            content = message.get("content")
            if isinstance(content, str) and content.startswith(TOOL_DATA_HEADER):
                filtered = _filtered_tool_data(content)
                if filtered is None:
                    continue
                if filtered != content:
                    message = dict(message)
                    message["content"] = filtered
            sanitised.append(message)
            continue

        sanitised.append(message)
    return sanitised
