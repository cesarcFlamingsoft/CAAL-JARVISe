"""Explicit caller-requested termination for a LiveKit room."""

from __future__ import annotations

import re
from typing import Any

from livekit import api

_END_CALL_PATTERNS = (
    re.compile(r"^(?:jarvis[,. ]+)?(?:please )?hang up(?: now)?[.!]*$", re.IGNORECASE),
    re.compile(
        r"^(?:jarvis[,. ]+)?(?:please )?end (?:this |the )?call(?: now)?[.!]*$",
        re.IGNORECASE,
    ),
    re.compile(r"^(?:jarvis[,. ]+)?(?:please )?disconnect(?: now)?[.!]*$", re.IGNORECASE),
    re.compile(r"^goodbye(?:,? jarvis)?[.!]*$", re.IGNORECASE),
)


def end_call_requested(transcript: str) -> bool:
    """Return true only for a short, unambiguous caller hang-up command."""
    normalized = " ".join(transcript.split())
    return bool(normalized) and any(pattern.fullmatch(normalized) for pattern in _END_CALL_PATTERNS)


# "Hang up now and call me back when the background task is done" is its own
# command, distinct from a plain hang-up: it authorizes an outbound call later.
# Only an imperative, self-addressed, complete sentence qualifies. Questions,
# hypotheticals, reported speech, a third-party callee, a dictated number, or
# any extra request in the same breath all fall through to Hermes untouched.
_CALLBACK_MAX_CHARS = 200
_CALLBACK_LEAD_IN = r"(?:(?:okay|ok|alright|jarvis)[,. ]+)*(?:please )?"
_CALLBACK_HANG_UP = r"(?:hang up|end (?:this |the )?call|disconnect)(?: now| for now)?"
_CALLBACK_JOIN = r"[,]? ?(?:and then|and|then) "
_CALLBACK_CALL_ME = r"(?:call me(?: back| again)?|ring me back|give me a call)"
_CALLBACK_WHEN_DONE = (
    r"(?:"
    r"(?:when|once|as soon as|after) "
    r"(?:you(?:'re| are)|it(?:'s| is)"
    r"|(?:the |this |that )?(?:background )?(?:task|job|work) is"
    r"|you have|you've got) "
    r"(?:done|finished|complete|completed|ready|the results|results)"
    r"|with (?:the )?results"
    r")"
)
_CALLBACK_CLAUSE = rf"{_CALLBACK_CALL_ME} {_CALLBACK_WHEN_DONE}"
_CALLBACK_PATTERNS = (
    re.compile(
        rf"^{_CALLBACK_LEAD_IN}{_CALLBACK_HANG_UP}{_CALLBACK_JOIN}{_CALLBACK_CLAUSE}$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^{_CALLBACK_LEAD_IN}{_CALLBACK_CLAUSE}{_CALLBACK_JOIN}{_CALLBACK_HANG_UP}$",
        re.IGNORECASE,
    ),
)


def callback_requested(transcript: str) -> bool:
    """Return true only for an explicit "hang up, then call me back when done" command.

    Deterministic and bounded: the text is whitespace-normalized, capped in
    length, stripped of trailing punctuation, and must fully match one fixed
    sentence shape. A plain hang-up never matches here, and this never matches
    ``end_call_requested``; the two commands are mutually exclusive.
    """
    normalized = " ".join(transcript.split())
    if not normalized or len(normalized) > _CALLBACK_MAX_CHARS:
        return False
    normalized = normalized.rstrip(".!")
    return any(pattern.fullmatch(normalized) for pattern in _CALLBACK_PATTERNS)


async def end_livekit_room(room_service: Any, room_name: str) -> None:
    """Disconnect every participant in exactly one LiveKit room, including SIP."""
    if not room_name:
        raise ValueError("A room name is required to end a call.")
    await room_service.delete_room(api.DeleteRoomRequest(room=room_name))


async def acknowledge_and_end_call(session: Any, room_service: Any, room_name: str) -> None:
    """Give a brief confirmation, then terminate the current call room."""
    await session.say("Ending the call. Goodbye.")
    await end_livekit_room(room_service, room_name)
