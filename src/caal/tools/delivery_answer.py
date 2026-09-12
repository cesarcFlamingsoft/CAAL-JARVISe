"""The answer to the delivery question, read from the words and nothing else.

CAAL asks one question when a timed reminder is set without a channel: when it
comes due, do you want me to say it here, send it to your Telegram, call you,
or do nothing about it? The reply is usually two words -- "call me" -- and two
words are exactly what a general model reads wrong. It sees a phone request, a
piece of background work, or something for the calendar, and the reminder that
asked the question never gets its answer.

This reads that reply directly. It is a whitelist and only a whitelist: the
utterance is normalised, bounded, and then every one of its words has to be
consumed by a known answer phrase. Anything left over -- a name, a number, a
time, a second request, a hypothetical, a question mark -- means this is not
plainly the answer, and nothing is returned, so the turn goes where it always
went.

What comes back is the bounded channel vocabulary of
:mod:`caal.tools.reminder_delivery` and never a destination: no number, no chat,
no id. Nothing here logs, stores or forwards a single word of the utterance.
"""

from __future__ import annotations

import re

from caal.tools.reminder_delivery import ALL, CALL, CHANNELS, DEFAULT, NONE, SPEAK, TELEGRAM

__all__ = ["MAX_ANSWER_CHARS", "read_delivery_answer"]

#: An answer to a one-line question is a short line. Anything longer is a
#: sentence with something else in it, and is never read here.
MAX_ANSWER_CHARS = 64

#: Said before the answer without changing it.
_LEADING_FILLER = frozenset(
    (
        "please",
        "ok",
        "okay",
        "yeah",
        "yes",
        "yep",
        "yup",
        "sure",
        "just",
        "um",
        "uh",
        "well",
        "alright",
        "jarvis",
    )
)
#: Said after the answer without changing it.
_TRAILING_FILLER = frozenset(("please", "thanks", "thank", "you"))

#: The ways a person points back at the reminder that was just asked about.
#: They add no channel and no target: the target is already fixed by the one
#: reminder awaiting an answer, so these are read as punctuation.
_REFERENCES = (
    "about that reminder",
    "about the reminder",
    "about that one",
    "about the one",
    "about that",
    "about it",
    "for that reminder",
    "for the reminder",
    "for that one",
    "for it",
    "on that one",
    "on it",
    "with that",
)

#: The connectors that join two channels in one answer.
_CONNECTORS = frozenset(("and", "plus", "also", "too"))

_PHRASES: dict[str, str] = {}


def _phrases(channel: str, *spoken: str) -> None:
    for phrase in spoken:
        _PHRASES[phrase] = channel


_phrases(
    CALL,
    "call",
    "call me",
    "a call",
    "give me a call",
    "give me a ring",
    "ring me",
    "phone me",
    "call my phone",
    "by phone",
    "with a call",
)
_phrases(
    TELEGRAM,
    "telegram",
    "telegram me",
    "a telegram",
    "message",
    "a message",
    "message me",
    "text me",
    "send me a message",
    "send me a telegram",
    "send it to telegram",
    "send it to my telegram",
    "on telegram",
    "by telegram",
)
_phrases(
    SPEAK,
    "speak",
    "speak it",
    "say it",
    "say it here",
    "say it to me",
    "say it out loud",
    "out loud",
    "aloud",
    "here",
    "in here",
    "tell me",
    "tell me here",
)
_phrases(
    ALL,
    "all",
    "all of them",
    "all of it",
    "all of the above",
    "all three",
    "all ways",
    "every way",
    "each way",
    "everything",
)
_phrases(
    DEFAULT,
    "default",
    "the default",
    "your default",
    "usual",
    "the usual",
    "as usual",
)
_phrases(
    NONE,
    "none",
    "no alert",
    "no alerts",
    "no notification",
    "no notifications",
    "nothing",
    "nothing at all",
    "do nothing",
    "dont tell me",
    "do not tell me",
    "leave it silent",
    "silent",
)

#: The three answers that describe a whole selection on their own. Combining
#: one of them with a channel is two answers, not one, so it is left alone.
_WHOLE = frozenset((ALL, DEFAULT, NONE))

_WORDS = re.compile(r"[a-z]+")


def _normalised(text: object) -> str:
    """One bounded lower-case line, or "" when this can never be an answer."""
    if not isinstance(text, str):
        return ""
    line = " ".join(text.split())
    if not line or len(line) > MAX_ANSWER_CHARS:
        return ""
    if "?" in line:
        # A question about the choice is not the choice.
        return ""
    line = line.lower().replace("’", "").replace("'", "")
    if any(character.isdigit() for character in line):
        # A number is a time, a count or a destination. Never an answer here.
        return ""
    return " ".join(_WORDS.findall(line))


def _without_reference(line: str) -> str:
    """The answer with a trailing pointer back at the reminder taken off."""
    for reference in _REFERENCES:
        if line == reference:
            return ""
        if line.endswith(" " + reference):
            return line[: -len(reference) - 1].strip()
    return line


def _trimmed(words: list[str]) -> list[str]:
    """The answer with the words that carry no answer taken off either end."""
    start, end = 0, len(words)
    while start < end and words[start] in _LEADING_FILLER:
        start += 1
    while end > start and words[end - 1] in _TRAILING_FILLER:
        end -= 1
    return words[start:end]


def read_delivery_answer(text: object) -> tuple[str, ...] | None:
    """The channels this turn plainly chose, or ``None`` to leave the turn alone.

    ``None`` is the default and the safe answer: it means this route recognised
    nothing and the turn keeps whatever path it already had. A returned tuple
    is the bounded vocabulary of :func:`reminder_delivery.parse_channels` --
    channels in their canonical order, or one of ``all``, ``default``, ``none``
    on its own.
    """
    line = _without_reference(_normalised(text))
    words = _trimmed(line.split()) if line else []
    if not words:
        return None

    segments: list[list[str]] = [[]]
    for word in words:
        if word in _CONNECTORS:
            segments.append([])
            continue
        segments[-1].append(word)

    chosen: list[str] = []
    for segment in segments:
        phrase = " ".join(segment)
        channel = _PHRASES.get(phrase)
        if channel is None:
            # One unread word is enough: this is somebody saying something
            # else, and guessing at it could arm a call nobody asked for.
            return None
        if channel not in chosen:
            chosen.append(channel)

    if any(channel in _WHOLE for channel in chosen):
        # "nothing, and call me" is two answers. Ask rather than pick one.
        return (chosen[0],) if len(chosen) == 1 else None
    return tuple(channel for channel in CHANNELS if channel in chosen)
