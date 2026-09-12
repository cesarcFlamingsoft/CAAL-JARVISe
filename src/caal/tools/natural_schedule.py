"""The schedule a natural voice request carried, recovered from that one turn.

Cesar says "remind me in one minute to stretch". The local model is meant to
call ``reminders.create`` with a title and a ``due``; in practice it often
writes the title and drops the time. The reminder was then stored correctly and
silently -- a list item nobody asked for, with no delivery question, because an
undated reminder rightly has nothing to deliver.

This closes that gap without inventing anything. The current user turn is read
back, and a time is recovered *only* when the words contain one phrase the
bounded parser in :mod:`caal.tools.scheduled_time` already accepts ("in 1
minute", "in an hour", "in 2 hours"). A vague phrase -- later, in a bit, in a
few minutes, tomorrow -- recovers nothing, and the reminder stays the undated
list item it honestly is. Two different times in one breath recover nothing
either: guessing between them is worse than asking.

Where the turn lives is the point. It is held in a :class:`ContextVar` for the
duration of the tool execution of that one turn and nowhere else: it is never
written to the store, never added to the tool data cache, never put into a
message, never sent to the escalation model, and never logged. Nothing in this
module logs at all, and nothing it returns carries the utterance -- only a
bounded title the user themselves said and an ISO-8601 duration.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from caal.tools.scheduled_time import MAX_SEARCH_LENGTH, find_relative_phrases

__all__ = [
    "SCHEDULE_RECOVERING_TOOLS",
    "current_turn",
    "iso_duration",
    "recover",
    "recovered_arguments",
    "user_turn",
]

#: The native tools whose arguments may be completed from the turn they came
#: from. Deliberately one: it is the one whose omission is silent.
SCHEDULE_RECOVERING_TOOLS = frozenset({"reminders.create"})

MAX_TITLE_LENGTH = 200

#: The words a request is introduced with, so what follows is the request.
_REQUEST_LEAD = re.compile(
    r"\b(?:remind|reminder\s+for)\s+(?:me|us)\b(?:\s+(?:to|that|about)\b)?",
    re.IGNORECASE,
)
#: Connectors left stranded once a time phrase is taken out of the middle.
_STRANDED = re.compile(r"^(?:to|that|about|and|,|:|-)\s+", re.IGNORECASE)
_TRAILING = re.compile(r"[\s,.;:!?-]+$")
_LEADING = re.compile(r"^[\s,.;:!?-]+")

# The turn is per-task and per-turn. A default of "" means "nothing is in
# scope", which is what every caller outside one turn must see.
_TURN: ContextVar[str] = ContextVar("caal_user_turn", default="")


def _flat(value: object, limit: int) -> str:
    """One bounded line of plain text, or nothing at all."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:limit]


@contextmanager
def user_turn(text: object) -> Iterator[None]:
    """Hold the current user turn for the duration of this turn own tool calls.

    The value is restored on the way out, including when the turn raises, so a
    later turn -- or a background task that borrowed this context -- can never
    read words that were said to somebody else.
    """
    token = _TURN.set(_flat(text, MAX_SEARCH_LENGTH))
    try:
        yield
    finally:
        _TURN.reset(token)


def current_turn() -> str:
    """What this user just said, while their turn is executing; otherwise ""."""
    return _TURN.get()


def iso_duration(seconds: int) -> str:
    """A count of seconds as the duration form the time parser reads back."""
    seconds = int(seconds)
    if seconds % 3600 == 0:
        return f"PT{seconds // 3600}H"
    if seconds % 60 == 0:
        return f"PT{seconds // 60}M"
    return f"PT{seconds}S"


def _without(text: str, spans: list[tuple[int, int, int]]) -> str:
    """The words with every time phrase taken back out of them."""
    kept = []
    last = 0
    for start, end, _ in spans:
        kept.append(text[last:start])
        last = end
    kept.append(text[last:])
    return " ".join("".join(kept).split())


def _tidy(text: str) -> str:
    """A stranded connector, and the punctuation around a removed phrase."""
    cleaned = _LEADING.sub("", _TRAILING.sub("", text))
    cleaned = _STRANDED.sub("", cleaned)
    return _LEADING.sub("", _TRAILING.sub("", cleaned))


def _request_of(turn: str, spans: list[tuple[int, int, int]]) -> str:
    """The thing the user asked to be reminded about, in their own words.

    Only the part of their sentence after "remind me" is ever used, and only
    with the time phrase removed. A turn that does not introduce a request
    yields nothing rather than a guess at which words were the subject.
    """
    lead = _REQUEST_LEAD.search(turn)
    if lead is None:
        return ""
    after = lead.end()
    tail = turn[after:]
    shifted = [(start - after, end - after, value) for start, end, value in spans if start >= after]
    return _tidy(_without(tail, shifted))[:MAX_TITLE_LENGTH]


def recover(arguments: dict[str, Any], turn: object) -> dict[str, Any]:
    """Complete a ``reminders.create`` call from the turn it was made in.

    Returns a copy. A ``due`` the model already wrote is never touched: this
    only ever fills in what was dropped. The title is replaced only when the
    model left it out or filled it with nothing but time wording, and then only
    with words the user said themselves.
    """
    recovered = dict(arguments)
    title = _flat(recovered.get("title"), MAX_TITLE_LENGTH)
    said = _flat(turn, MAX_SEARCH_LENGTH)

    if _flat(recovered.get("due"), 200):
        return recovered

    turn_spans = find_relative_phrases(said)
    title_spans = find_relative_phrases(title)
    # Exactly one time, said once. Two is a choice nobody authorised us to make.
    seconds: int | None = None
    if len({value for _, _, value in turn_spans}) == 1:
        seconds = turn_spans[0][2]
    elif not turn_spans and len({value for _, _, value in title_spans}) == 1:
        seconds = title_spans[0][2]

    if seconds is None:
        return recovered

    recovered["due"] = iso_duration(seconds)
    subject = _tidy(_without(title, title_spans))
    recovered["title"] = subject or _request_of(said, turn_spans) or title
    return recovered


def recovered_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """The arguments a native tool is actually called with, for this turn.

    Everything but the one recovering tool passes through untouched, so no
    other tool can ever be completed from words the user said about something
    else.
    """
    if tool_name not in SCHEDULE_RECOVERING_TOOLS or not isinstance(arguments, dict):
        return arguments
    return recover(arguments, current_turn())
