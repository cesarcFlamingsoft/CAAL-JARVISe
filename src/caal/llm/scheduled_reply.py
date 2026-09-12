"""What a finished scheduled-item call says, without asking a model to say it.

A reminder or an alarm that was just written is already the whole answer. The
handler composed one concise sentence about it -- and, for a new timed
reminder, the delivery-choice question in the same breath -- so the turn has
nothing left to work out.

Until this existed the runtime asked the local model for a natural-language
follow-up built from the assistant tool-call history plus the tool result. The
effective local model streams ordinary text fine and returns *zero* chunks for
that particular continuation shape; the routed provider then fell through to
the escalation runtime, which failed on its own, and the person heard the
generic "no model available" line about a reminder that was already stored and
already armed.

So a scheduled outcome is spoken as it stands. That removes the failing
generation from the path entirely, removes the escalation behind it, keeps the
reminder of this person out of a second runtime, and says the same thing once
instead of twice.

The mapping is deliberately narrow:

* only the three tools that write a scheduled item (:data:`SCHEDULED_TOOLS`,
  read from :mod:`caal.scheduled_events` so the direct reply and the dashboard
  packet can never disagree about what counts);
* a success speaks the handler message;
* a refusal or an unusable argument speaks the human half of its message --
  the sentences addressed to the model are dropped, since nobody should hear
  "call the tool again";
* a handler that reported an error speaks one fixed internal-failure line,
  which says nothing about the cause, never blames a model, and never claims a
  reminder was set or delivered.

Everything else -- connected accounts, email, calendar, Home Assistant, memory,
every other tool -- returns ``None`` here and keeps the ordinary model
follow-up it always had.
"""

from __future__ import annotations

import re
from typing import Any

from caal.scheduled_events import SCHEDULED_TOOLS

__all__ = [
    "DONE_REPLY",
    "INTERNAL_ERROR_REPLY",
    "SCHEDULED_TOOLS",
    "UNCLEAR_REQUEST_REPLY",
    "is_scheduled_tool",
    "spoken_outcome",
]

#: Said when the handler itself reported a failure. No cause, no provider, no
#: traceback, and no claim that anything was scheduled or sent.
INTERNAL_ERROR_REPLY = (
    "Something went wrong on my end, so that is not set. Ask me again and I will try it."
)

#: Said when a refusal carried nothing a person could act on.
UNCLEAR_REQUEST_REPLY = "I did not catch enough to set that up. Tell me what and when."

#: Said when a success carried no message of its own. It claims the write that
#: actually happened and nothing beyond it: no channel, and no promise to alert.
DONE_REPLY = "That is set."

_OK = "ok"
_ERROR = "error"

# Sentences written for the model, not for the person listening. A refusal
# message is allowed to instruct the model; none of that is ever spoken.
_TO_THE_MODEL = (
    "call the tool again",
    "ask the user",
    "tell the user",
    "say nothing about why",
    "offer to try again",
)

_SENTENCE = re.compile(r"[^.!?]+[.!?]?")
_MAX_SPOKEN = 400


def is_scheduled_tool(tool_name: object) -> bool:
    """Whether this tool writes a scheduled item of the signed-in user."""
    return tool_name in SCHEDULED_TOOLS


def _for_the_person(message: str) -> str:
    """The part of a message a person should hear, in its original order."""
    kept = [
        sentence.strip()
        for sentence in _SENTENCE.findall(message)
        if sentence.strip() and not any(phrase in sentence.lower() for phrase in _TO_THE_MODEL)
    ]
    return " ".join(kept)[:_MAX_SPOKEN]


def spoken_outcome(tool_name: object, result: Any) -> str | None:
    """The sentence to speak for a finished scheduled call, or ``None``.

    ``None`` means this call is not one of ours and must keep the ordinary
    model follow-up. A string means the turn is answered: it is spoken as it
    is, and no provider is asked for anything further.
    """
    if not is_scheduled_tool(tool_name) or not isinstance(result, dict):
        return None
    status = result.get("status")
    if status == _ERROR:
        return INTERNAL_ERROR_REPLY
    message = result.get("message")
    spoken = _for_the_person(message) if isinstance(message, str) else ""
    if spoken:
        return spoken
    # A result with nothing sayable in it still ends the turn honestly: a
    # success that said nothing is a success, and a refusal that said nothing
    # asks for the detail again rather than inventing one.
    return DONE_REPLY if status == _OK else UNCLEAR_REQUEST_REPLY
