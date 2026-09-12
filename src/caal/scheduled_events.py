"""The one packet that tells a dashboard its own scheduled items changed.

A reminder set by voice has to appear on the dashboard now, not at the end of
the next polling interval. The cheapest honest way to do that is to say, on the
data channel of the room the person is already in, that *something* of theirs
changed -- and to say nothing else.

So this publishes a constant. A version and a kind:

    {"v": 1, "kind": "scheduled_changed"}

There is no title in it, no due time, no row id, no owner, no tool arguments,
no model output, no phone number and no chat. It carries no capability either:
a listener learns only that it should re-read its *own* authenticated feed,
which is the surface that already decides what that person may see. Tool status
packets are a different topic with different rules, and none of their parameters
is ever reused here.

It is published after the change is written and never before, so a refused or
failed call announces nothing.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["EVENT", "PAYLOAD", "SCHEDULED_TOOLS", "TOPIC", "announce", "is_scheduled_mutation"]

#: The data-channel topic. A listener validates this exactly.
TOPIC = "scheduled_changed"
KIND = "scheduled_changed"
VERSION = 1

EVENT: dict[str, Any] = {"v": VERSION, "kind": KIND}
#: The literal bytes-to-be. Constant, so there is nothing in it to leak.
PAYLOAD = json.dumps(EVENT, separators=(",", ":"))

#: The native tools that change what a scheduled feed shows.
SCHEDULED_TOOLS = frozenset(
    {"reminders.create", "reminders.set_delivery", "alarms.set", "scheduled.change"}
)

_OK = "ok"


def is_scheduled_mutation(tool_name: object, result: object) -> bool:
    """Whether a finished tool call actually changed this owner scheduled state.

    Both halves matter: the tool has to be one that writes, and the result has
    to say it succeeded. A refusal, a failure and a read all announce nothing.
    """
    if tool_name not in SCHEDULED_TOOLS:
        return False
    return isinstance(result, dict) and result.get("status") == _OK


async def announce(agent: Any) -> bool:
    """Tell this session room that its scheduled items changed, if it can.

    A session with no way to publish, or a room that has gone away, is not a
    reason to lose anything: the write already happened and the dashboard still
    has its polling fallback. The failure is recorded as a fact with nothing in
    it.
    """
    publish = getattr(agent, "_on_scheduled_change", None)
    if publish is None:
        return False
    try:
        outcome = publish()
        if inspect.isawaitable(outcome):
            await outcome
    except Exception:
        logger.info("Could not tell this room that its scheduled items changed")
        return False
    return True
