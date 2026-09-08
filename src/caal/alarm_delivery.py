"""Deliver persisted alarms to an active CAAL voice session."""

from __future__ import annotations

from typing import Any

from caal.tools.alarms_tools import claim_due_alarms


def alarm_message(alarm: dict[str, Any]) -> str:
    """Return a concise, speakable alarm announcement."""
    label = alarm["label"]
    if alarm["kind"] == "timer":
        return f"Your timer '{label}' is finished."
    return f"Alarm: {label}."


async def announce_due_alarms(session: Any) -> int:
    """Speak every due alarm once while a voice session is connected."""
    alarms = claim_due_alarms()
    for alarm in alarms:
        await session.say(alarm_message(alarm))
    return len(alarms)
