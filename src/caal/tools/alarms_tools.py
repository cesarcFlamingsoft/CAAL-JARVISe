"""Persistent alarms and timers for CAAL."""

from __future__ import annotations

import os
import re
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

STORE_PATH = Path(os.getenv("CAAL_DATA_DIR", "/app/data")) / "assistant.sqlite3"
_DURATION = re.compile(
    r"^(?P<value>\d+)\s*(?P<unit>s|m|h|d|sec(?:onds?)?|min(?:utes?)?|hours?|days?)$", re.I
)


def _connect() -> sqlite3.Connection:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(STORE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE IF NOT EXISTS alarms (
            id TEXT PRIMARY KEY, label TEXT NOT NULL, kind TEXT NOT NULL,
            due_at INTEGER NOT NULL, fired_at INTEGER, created_at INTEGER NOT NULL
        )"""
    )
    return connection


def _due_at(when: str, now: int) -> int:
    match = _DURATION.fullmatch(when.strip())
    if match:
        units = match.group("unit").lower()
        multiplier = (
            1
            if units.startswith("s")
            else 60
            if units.startswith("m")
            else 3600
            if units.startswith("h")
            else 86400
        )
        return now + int(match.group("value")) * multiplier
    try:
        return int(datetime.fromisoformat(when.replace("Z", "+00:00")).timestamp())
    except ValueError as error:
        raise ValueError("Alarm time must be ISO-8601 or a duration such as 10m") from error


def set_alarm(label: str, when: str, kind: str, now: int | None = None) -> dict[str, Any]:
    """Persist an alarm or timer until it can be announced by the agent."""
    if kind not in {"alarm", "timer"}:
        raise ValueError("Alarm kind must be 'alarm' or 'timer'")
    current = int(time.time()) if now is None else now
    due_at = _due_at(when, current)
    if due_at <= current:
        raise ValueError("Alarm time must be in the future")
    alarm = {
        "id": str(uuid.uuid4()),
        "label": label.strip(),
        "kind": kind,
        "due_at": due_at,
        "created_at": current,
    }
    if not alarm["label"]:
        raise ValueError("Alarm label is required")
    with _connect() as connection:
        connection.execute(
            (
                "INSERT INTO alarms (id,label,kind,due_at,created_at) "
                "VALUES (:id,:label,:kind,:due_at,:created_at)"
            ),
            alarm,
        )
    return {"status": "ok", "message": f"Set {kind}: {alarm['label']}.", "data": alarm}


def claim_due_alarms(now: int | None = None) -> list[dict[str, Any]]:
    """Atomically claim due alarms so retries never announce an alarm twice."""
    current = int(time.time()) if now is None else now
    with _connect() as connection:
        rows = connection.execute(
            (
                "SELECT id,label,kind,due_at FROM alarms "
                "WHERE fired_at IS NULL AND due_at <= ? ORDER BY due_at"
            ),
            (current,),
        ).fetchall()
        if rows:
            connection.executemany(
                "UPDATE alarms SET fired_at = ? WHERE id = ? AND fired_at IS NULL",
                [(current, row["id"]) for row in rows],
            )
    return [dict(row) for row in rows]
