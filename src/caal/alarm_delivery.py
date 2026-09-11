"""Deliver persisted alarms, timers and timed reminders to their own user.

An alarm is only ever announced in a live voice session of the user who set it.
There is no global fallback: if nobody eligible is listening, the alarm stays
pending, and no reply anywhere claims it was delivered. A claim that cannot be
spoken is handed straight back, so the next eligible session still gets it, and
a claim that is spoken is settled so no session says it twice.

An anonymous session under multi-user identity claims nothing at all: it has no
verified user, and guessing would mean announcing the private alarm of somebody
else in the room.
"""

from __future__ import annotations

import logging
from typing import Any

from caal.tools import reminder_delivery
from caal.tools.alarms_tools import claim_due_alarms, mark_delivered, release_alarms
from caal.user_scope import UserScope

logger = logging.getLogger(__name__)

__all__ = ["alarm_message", "announce_due_alarms", "may_deliver"]


def alarm_message(alarm: dict[str, Any]) -> str:
    """A concise, speakable announcement: the label of the user and nothing else."""
    label = str(alarm.get("label", "")).strip()
    kind = alarm.get("kind")
    if kind == "timer":
        return f"Your timer is finished: {label}." if label else "Your timer is finished."
    if kind == "reminder":
        return f"Reminder: {label}." if label else "Here is your reminder."
    return f"Alarm: {label}." if label else "Your alarm is due."


def may_deliver(scope: UserScope | None) -> bool:
    """Whether this session may claim anything at all.

    A legacy single-user deployment keeps its unowned alarms; a verified user
    gets their own; an anonymous session under multi-user gets nothing.
    """
    effective = scope if scope is not None else UserScope.legacy()
    return effective.memory_available


async def announce_due_alarms(
    session: Any, scope: UserScope | None = None, now: int | None = None
) -> int:
    """Speak every due alarm of this session own user, exactly once.

    Returns how many were actually announced. Anything claimed but not spoken
    is released before the exception leaves this function.

    Speech is also one of the channels a timed reminder can be delivered on,
    so an alarm that was actually said settles that reminder spoken channel
    here -- after the words left the session, never before. The other channels
    of the same reminder are untouched: they belong to the durable worker.
    """
    if not may_deliver(scope):
        return 0
    user_id = (scope or UserScope.legacy()).user_id
    alarms = claim_due_alarms(now=now, user_id=user_id)
    if not alarms:
        return 0
    spoken: list[str] = []
    try:
        for alarm in alarms:
            await session.say(alarm_message(alarm))
            spoken.append(alarm["id"])
    finally:
        pending = [alarm["id"] for alarm in alarms if alarm["id"] not in set(spoken)]
        if spoken:
            mark_delivered(spoken, now=now)
            reminder_delivery.settle_speak(spoken, now=now)
        if pending:
            # Never announced, so never delivered: it waits for the next session.
            released = release_alarms(pending)
            logger.info("Returned %d undelivered alarm(s) to the queue", released)
    return len(spoken)
