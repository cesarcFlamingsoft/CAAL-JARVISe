"""When an alarm, timer or timed reminder is due.

The local model writes this argument, so the accepted forms are the ones a
model reaches for first: an ISO-8601 duration from now (PT2M, PT30S, PT1H30M,
P1D), the spoken shorthand a person uses (10m, 2 hours), or an exact
timezone-aware timestamp. A timestamp without an offset is refused rather than
guessed: local time is ambiguous across the two hours a year that repeat, and a
wrong guess means an alarm that never fires when the user expects it.

Everything is bounded. A duration is capped at a year and its digits are capped
before they are multiplied, so no argument can overflow the stored integer or
push a due time past the end of the epoch.
"""

from __future__ import annotations

import re
from datetime import datetime

from caal.tools.errors import SafeToolError

__all__ = [
    "MAX_HORIZON_SECONDS",
    "WHEN_HINT",
    "parse_duration_seconds",
    "parse_when",
]

# A year ahead. Anything further is a calendar event, not an alarm.
MAX_HORIZON_SECONDS = 366 * 24 * 3600

WHEN_HINT = (
    "Give the time as an ISO-8601 duration from now such as PT2M, PT30S, PT1H30M or P1D, "
    "as a plain duration such as 10m or 2 hours, or as an exact timestamp that includes a "
    "timezone offset such as 2026-09-09T18:30:00-06:00."
)

# P[nW][nD][T[nH][nM][nS]]. The digit caps keep any product far inside a 64-bit
# integer; the horizon check below rejects what is merely large.
_ISO_DURATION = re.compile(
    r"^[+]?P(?!$)"
    r"(?:(?P<weeks>\d{1,4})W)?"
    r"(?:(?P<days>\d{1,5})D)?"
    r"(?:T(?!$)"
    r"(?:(?P<hours>\d{1,6})H)?"
    r"(?:(?P<minutes>\d{1,8})M)?"
    r"(?:(?P<seconds>\d{1,9})(?:[.,]\d{1,6})?S)?"
    r")?$",
    re.IGNORECASE,
)

_PLAIN_DURATION = re.compile(
    r"^(?P<value>\d{1,9})\s*"
    r"(?P<unit>s|sec|secs|second|seconds|m|min|mins|minute|minutes|"
    r"h|hr|hrs|hour|hours|d|day|days|w|week|weeks)$",
    re.IGNORECASE,
)

_UNIT_SECONDS = dict(s=1, m=60, h=3600, d=86400, w=604800)

_ISO_UNIT_SECONDS = dict(weeks=604800, days=86400, hours=3600, minutes=60, seconds=1)


def _unit_seconds(unit: str) -> int:
    """Seconds in a spoken unit, which its first letter already identifies."""
    return _UNIT_SECONDS[unit[0].lower()]


def parse_duration_seconds(when: str) -> int | None:
    """Seconds in an ISO-8601 or shorthand duration, or None when it is neither.

    A duration that parses but is zero, or that reaches past the horizon, is a
    refusal rather than a None: the caller must not fall through and try to read
    it as a timestamp.
    """
    text = " ".join(str(when).split())
    match = _ISO_DURATION.fullmatch(text)
    if match is not None:
        parts = match.groupdict()
        if not any(value is not None for value in parts.values()):
            raise SafeToolError(f"That duration has no length. {WHEN_HINT}")
        total = sum(
            int(parts[name]) * seconds
            for name, seconds in _ISO_UNIT_SECONDS.items()
            if parts[name] is not None
        )
        return _bounded(total)
    plain = _PLAIN_DURATION.fullmatch(text)
    if plain is not None:
        return _bounded(int(plain.group("value")) * _unit_seconds(plain.group("unit")))
    return None


def _bounded(total: int) -> int:
    if total <= 0:
        raise SafeToolError(
            "I can only set that for a time in the future, and that duration is zero. " + WHEN_HINT
        )
    if total > MAX_HORIZON_SECONDS:
        raise SafeToolError("I can only schedule that up to a year ahead.")
    return total


def _absolute_seconds(when: str) -> int:
    """The epoch second of a timezone-aware ISO-8601 timestamp."""
    text = " ".join(str(when).split())
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00").replace("z", "+00:00"))
    except (ValueError, TypeError) as error:
        raise SafeToolError(f"I could not read that as a time. {WHEN_HINT}") from error
    if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
        raise SafeToolError(
            "That time has no timezone, so I cannot tell which moment it means. " + WHEN_HINT
        )
    try:
        return int(moment.timestamp())
    except (OverflowError, OSError, ValueError) as error:
        raise SafeToolError("That date is outside the range I can schedule.") from error


def parse_when(when: object, now: int) -> int:
    """The epoch second something asked for at ``when`` is due, or a safe refusal.

    ``now`` is the epoch second the request is being made at, which is what a
    duration is measured from.
    """
    if not isinstance(when, str) or not when.strip():
        raise SafeToolError(f"I need to know when. {WHEN_HINT}")
    if len(when) > 200:
        raise SafeToolError(f"I could not read that as a time. {WHEN_HINT}")
    duration = parse_duration_seconds(when)
    if duration is not None:
        return int(now) + duration
    due_at = _absolute_seconds(when)
    if due_at <= int(now):
        raise SafeToolError("That time has already passed, so tell me a time still to come.")
    if due_at - int(now) > MAX_HORIZON_SECONDS:
        raise SafeToolError("I can only schedule that up to a year ahead.")
    return due_at
