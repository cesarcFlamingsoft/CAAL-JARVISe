"""Persistent alarms, timers and timed reminders, owned by one user each.

Every row belongs to exactly one scope: a verified user id, or the empty legacy
scope that a deployment without multi-user identity keeps using. A session only
ever claims rows in its own scope, so a due alarm of one user is never announced
in the room of another, and rows written before ownership existed stay in the
legacy scope: kept, never spoken to a signed-in user.

Nothing here logs a label, a note or a row id. The label is the words of the
user and is only ever handed back to the session that owns it.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from caal.tools.errors import SafeToolError, safe_error_result
from caal.tools.scheduled_time import WHEN_HINT, parse_when
from caal.user_scope import require_user_id

logger = logging.getLogger(__name__)

STORE_PATH = Path(os.getenv("CAAL_DATA_DIR", "/app/data")) / "assistant.sqlite3"

LEGACY_SCOPE = ""
MAX_LABEL_LENGTH = 120
MODEL_KINDS = ("alarm", "timer")
KINDS = ("alarm", "timer", "reminder")
_BUSY_TIMEOUT_SECONDS = 5.0


def _connect() -> sqlite3.Connection:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(STORE_PATH, timeout=_BUSY_TIMEOUT_SECONDS, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout = {int(_BUSY_TIMEOUT_SECONDS * 1000)}")
    _ensure_schema(connection)
    return connection


def _columns(connection: sqlite3.Connection) -> set[str]:
    return {row["name"] for row in connection.execute("PRAGMA table_info(alarms)")}


_CREATE = """
    CREATE TABLE IF NOT EXISTS {name} (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL DEFAULT '',
        label TEXT NOT NULL,
        kind TEXT NOT NULL,
        due_at INTEGER NOT NULL,
        fired_at INTEGER,
        delivered_at INTEGER,
        created_at INTEGER NOT NULL
    )
"""


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """Create the owned table, upgrading an unowned pre-multi-user table in place.

    The rows of the old table are kept under the legacy scope rather than
    handed to whoever happens to sign in first.
    """
    columns = _columns(connection)
    if columns and {"user_id", "delivered_at"} <= columns:
        return
    connection.execute("BEGIN IMMEDIATE")
    try:
        columns = _columns(connection)  # re-check under the write lock
        if not columns:
            connection.execute(_CREATE.format(name="alarms"))
        elif not {"user_id", "delivered_at"} <= columns:
            connection.execute(_CREATE.format(name="alarms_owned"))
            delivered = "fired_at" if "fired_at" in columns else "NULL"
            connection.execute(
                "INSERT INTO alarms_owned (id,user_id,label,kind,due_at,fired_at,"
                f"delivered_at,created_at) SELECT id,'',label,kind,due_at,fired_at,{delivered},"
                "created_at FROM alarms"
            )
            connection.execute("DROP TABLE alarms")
            connection.execute("ALTER TABLE alarms_owned RENAME TO alarms")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS alarms_due ON alarms (user_id, fired_at, due_at)"
        )
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise


def _scope(user_id: object) -> str:
    """The storage scope of a caller: the legacy scope, or a validated user id."""
    if user_id is None:
        return LEGACY_SCOPE
    return require_user_id(user_id)


def _clean_label(label: object) -> str:
    if not isinstance(label, str):
        raise SafeToolError("I need a short name for it, such as laundry or standup.")
    cleaned = " ".join(label.split())
    if not cleaned:
        raise SafeToolError("I need a short name for it, such as laundry or standup.")
    return cleaned[:MAX_LABEL_LENGTH]


def _iso(due_at: int) -> str:
    return datetime.fromtimestamp(due_at, tz=timezone.utc).isoformat()


def describe_delay(seconds: int) -> str:
    """A spoken phrase for how far off something is, without naming a clock."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"in {seconds} seconds" if seconds != 1 else "in 1 second"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"in {minutes} minutes" if minutes != 1 else "in 1 minute"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        if minutes:
            return (
                f"in {hours} hours and {minutes} minutes"
                if hours != 1
                else (f"in 1 hour and {minutes} minutes")
            )
        return f"in {hours} hours" if hours != 1 else "in 1 hour"
    days = hours // 24
    return f"in {days} days" if days != 1 else "in 1 day"


def schedule(
    label: str,
    due_at: int,
    kind: str,
    user_id: str | None = None,
    now: int | None = None,
    alarm_id: str | None = None,
) -> dict[str, Any]:
    """Store one already-parsed due time for the owner, and describe it truthfully.

    ``alarm_id`` lets an internal caller choose the row id it will also record
    elsewhere: a timed reminder links its spoken channel to this row. It is
    never a tool argument, so no model can name one, and it is never returned.
    """
    if kind not in KINDS:
        raise SafeToolError("I can set an alarm, a timer or a timed reminder.")
    current = int(time.time()) if now is None else int(now)
    if due_at <= current:
        raise SafeToolError("That time has already passed, so tell me a time still to come.")
    row = dict(
        id=alarm_id or str(uuid.uuid4()),
        user_id=_scope(user_id),
        label=_clean_label(label),
        kind=kind,
        due_at=int(due_at),
        created_at=current,
    )
    with closing(_connect()) as connection:
        connection.execute(
            "INSERT INTO alarms (id,user_id,label,kind,due_at,created_at) "
            "VALUES (:id,:user_id,:label,:kind,:due_at,:created_at)",
            row,
        )
    logger.info("Stored a %s due in %ds", kind, int(due_at) - current)
    delay = describe_delay(int(due_at) - current)
    return {
        "status": "ok",
        "message": f"{kind.capitalize()} set {delay}: {row['label']}.",
        "data": dict(label=row["label"], kind=kind, due_at=int(due_at), due_iso=_iso(int(due_at))),
    }


def set_alarm(
    label: str,
    when: str,
    kind: str = "alarm",
    now: int | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Set an alarm or timer that this session announces when it comes due.

    Only stored state: nothing is sent anywhere, and the success message is
    returned after the row is written, never before.
    """
    current = int(time.time()) if now is None else int(now)
    try:
        if kind not in MODEL_KINDS:
            raise SafeToolError("I can set an alarm or a timer. Which one did you mean?")
        due_at = parse_when(when, current)
        return schedule(label, due_at, kind, user_id=user_id, now=current)
    except SafeToolError as error:
        logger.info("Refused an alarm request: %s", type(error).__name__)
        return safe_error_result(error, status="invalid_time")


def claim_due_alarms(now: int | None = None, user_id: str | None = None) -> list[dict[str, Any]]:
    """Atomically claim the due alarms of one owner, so none is announced twice.

    A claim is exclusive: a second caller, in this process or another, sees
    nothing. A claim that is never announced can be handed back with
    :func:`release_alarms`, and one that is announced is settled with
    :func:`mark_delivered`.
    """
    current = int(time.time()) if now is None else int(now)
    scope = _scope(user_id)
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            rows = connection.execute(
                "SELECT id,label,kind,due_at FROM alarms "
                "WHERE user_id = ? AND fired_at IS NULL AND due_at <= ? ORDER BY due_at",
                (scope, current),
            ).fetchall()
            if rows:
                connection.executemany(
                    "UPDATE alarms SET fired_at = ? WHERE id = ? AND fired_at IS NULL",
                    [(current, row["id"]) for row in rows],
                )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return [dict(row) for row in rows]


def mark_delivered(alarm_ids: list[str], now: int | None = None) -> int:
    """Settle claimed alarms that were actually announced."""
    if not alarm_ids:
        return 0
    current = int(time.time()) if now is None else int(now)
    with closing(_connect()) as connection:
        cursor = connection.executemany(
            "UPDATE alarms SET delivered_at = ? WHERE id = ? AND delivered_at IS NULL",
            [(current, alarm_id) for alarm_id in alarm_ids],
        )
    return int(cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0)


def release_alarms(alarm_ids: list[str]) -> int:
    """Hand back claimed alarms that were never announced, so they stay pending."""
    if not alarm_ids:
        return 0
    with closing(_connect()) as connection:
        cursor = connection.executemany(
            "UPDATE alarms SET fired_at = NULL WHERE id = ? AND delivered_at IS NULL",
            [(alarm_id,) for alarm_id in alarm_ids],
        )
    return int(cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0)


def cancel_pending(alarm_id: str, user_id: str | None = None) -> int:
    """Drop one owner own alarm that has not been announced yet.

    Scoped to the owner and to an undelivered row, so cancelling a spoken
    reminder channel can never reach anybody else alarm or un-say something
    that was already said.
    """
    if not alarm_id:
        return 0
    with closing(_connect()) as connection:
        cursor = connection.execute(
            "DELETE FROM alarms WHERE id = ? AND user_id = ? AND delivered_at IS NULL",
            (alarm_id, _scope(user_id)),
        )
    return int(cursor.rowcount or 0)


def pending_count(user_id: str | None = None, now: int | None = None) -> int:
    """How many alarms of this owner are still waiting to be announced."""
    current = int(time.time()) if now is None else int(now)
    with closing(_connect()) as connection:
        (count,) = connection.execute(
            "SELECT COUNT(*) FROM alarms WHERE user_id = ? AND delivered_at IS NULL AND due_at > ?",
            (_scope(user_id), current),
        ).fetchone()
    return int(count)


WHEN_DESCRIPTION = WHEN_HINT
