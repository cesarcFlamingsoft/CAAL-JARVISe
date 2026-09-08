"""Durable registry of active JARVIS device sessions and pending handoffs.

Phase 1 of seamless handoff: persistence and safe models only. Sessions are
keyed by an opaque bearer token so a device can prove ownership of its own
session without exposing anything about the other devices it may hand off to.
Nothing here dials, dispatches, or talks to LiveKit.
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .user_scope import require_user_id

STORE_PATH = Path(os.getenv("CAAL_DATA_DIR", "/app/data")) / "assistant.sqlite3"


class _Unscoped:
    """Marker for "every owner": operator tooling and legacy callers."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSCOPED"


UNSCOPED: Any = _Unscoped()

MAX_LABEL_LENGTH = 40
MAX_IDENTIFIER_LENGTH = 128
SESSION_TTL_SECONDS = 300
HANDOFF_TTL_SECONDS = 120
MAX_ACTIVE_SESSIONS = 25
MAX_CONTEXT_KEYS = 16
MAX_CONTEXT_KEY_CHARS = 64
MAX_CONTEXT_VALUE_CHARS = 512
MAX_CONTEXT_CHARS = 4096

_TOKEN_BYTES = 32
_BUSY_TIMEOUT_SECONDS = 5.0


class Transport(str, Enum):
    """How a device is connected to JARVIS."""

    WEB = "web"
    MOBILE = "mobile"
    PHONE = "phone"


@dataclass(frozen=True)
class DeviceSession:
    """One live device session. ``session_id`` is a secret, so it never reprs."""

    session_id: str = field(repr=False)
    device_id: str
    room_name: str
    label: str
    transport: Transport
    last_seen: int
    # Opaque id of the verified user this device belongs to; ``None`` = legacy.
    user_id: str | None = None


@dataclass(frozen=True)
class HandoffRecord:
    """A pending or claimed handoff. ``handoff_id`` is a secret, so it never reprs."""

    handoff_id: str = field(repr=False)
    device_id: str
    room_name: str
    context: dict[str, Any]
    created_at: int
    claimed_at: int | None = None
    claimed_by: str | None = None
    user_id: str | None = None


def _now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _resolve_now(now: int | None) -> int:
    return _now() if now is None else int(now)


def _new_token() -> str:
    return secrets.token_urlsafe(_TOKEN_BYTES)


def normalize_label(label: str) -> str:
    """Return a friendly, display-safe device label, or raise ``ValueError``."""
    if not isinstance(label, str):
        raise ValueError("Device label must be text.")
    normalized = " ".join(label.split())
    if not normalized:
        raise ValueError("Device label is required.")
    if len(normalized) > MAX_LABEL_LENGTH:
        raise ValueError(f"Device label must be at most {MAX_LABEL_LENGTH} characters.")
    if not normalized.isprintable():
        raise ValueError("Device label must not contain control characters.")
    return normalized


def _normalize_identifier(value: str, field_name: str) -> str:
    """Return a bounded, control-character-free identifier, or raise ``ValueError``."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be text.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required.")
    if len(normalized) > MAX_IDENTIFIER_LENGTH:
        raise ValueError(f"{field_name} must be at most {MAX_IDENTIFIER_LENGTH} characters.")
    if not normalized.isprintable():
        raise ValueError(f"{field_name} must not contain control characters.")
    return normalized


def coerce_transport(transport: Transport | str) -> Transport:
    """Return the matching :class:`Transport`, or raise ``ValueError``."""
    if isinstance(transport, Transport):
        return transport
    if not isinstance(transport, str):
        raise ValueError("Transport must be one of: web, mobile, phone.")
    try:
        return Transport(transport.strip().lower())
    except ValueError as error:
        raise ValueError("Transport must be one of: web, mobile, phone.") from error


def bound_context(context: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a flat, size-bounded snapshot safe to persist and replay."""
    if context is None:
        return {}
    if not isinstance(context, Mapping):
        raise ValueError("Handoff context must be a mapping of scalar values.")
    if len(context) > MAX_CONTEXT_KEYS:
        raise ValueError(f"Handoff context may hold at most {MAX_CONTEXT_KEYS} keys.")

    snapshot: dict[str, Any] = {}
    for key, value in context.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Handoff context keys must be non-empty text.")
        if len(key) > MAX_CONTEXT_KEY_CHARS:
            raise ValueError(
                f"Handoff context keys must be at most {MAX_CONTEXT_KEY_CHARS} characters."
            )
        if isinstance(value, str):
            if len(value) > MAX_CONTEXT_VALUE_CHARS:
                value = value[: MAX_CONTEXT_VALUE_CHARS - 1] + "…"
        elif not isinstance(value, (bool, int, float, type(None))):
            raise ValueError("Handoff context values must be text, numbers, booleans, or null.")
        snapshot[key] = value

    if len(json.dumps(snapshot, sort_keys=True)) > MAX_CONTEXT_CHARS:
        raise ValueError(f"Handoff context must serialize to under {MAX_CONTEXT_CHARS} characters.")
    return snapshot


def _connect() -> sqlite3.Connection:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(STORE_PATH, timeout=_BUSY_TIMEOUT_SECONDS, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout = {int(_BUSY_TIMEOUT_SECONDS * 1000)}")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS device_sessions (
            session_id TEXT PRIMARY KEY,
            device_id TEXT NOT NULL UNIQUE,
            room_name TEXT NOT NULL,
            label TEXT NOT NULL,
            transport TEXT NOT NULL,
            last_seen INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS device_handoffs (
            handoff_id TEXT PRIMARY KEY,
            device_id TEXT NOT NULL,
            room_name TEXT NOT NULL,
            context TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            claimed_at INTEGER,
            claimed_by TEXT
        )
        """
    )
    # Multi-user: devices and handoffs belong to an opaque user id.
    _ensure_column(connection, "device_sessions", "user_id", "TEXT")
    _ensure_column(connection, "device_handoffs", "user_id", "TEXT")
    return connection


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    """Upgrade a table written before ``column`` existed."""
    columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _session_from_row(row: sqlite3.Row) -> DeviceSession:
    return DeviceSession(
        session_id=row["session_id"],
        device_id=row["device_id"],
        room_name=row["room_name"],
        label=row["label"],
        transport=Transport(row["transport"]),
        last_seen=row["last_seen"],
        user_id=row["user_id"],
    )


def _handoff_from_row(row: sqlite3.Row) -> HandoffRecord:
    return HandoffRecord(
        handoff_id=row["handoff_id"],
        device_id=row["device_id"],
        room_name=row["room_name"],
        context=json.loads(row["context"]),
        created_at=row["created_at"],
        claimed_at=row["claimed_at"],
        claimed_by=row["claimed_by"],
        user_id=row["user_id"],
    )


def register_device_session(
    *,
    device_id: str,
    room_name: str,
    label: str,
    transport: Transport | str,
    now: int | None = None,
    user_id: str | None = None,
) -> DeviceSession:
    """Start (or restart) the single session for one device and return it.

    Re-registering a device retires its previous session token, so a stolen or
    stale token stops working the moment the device reconnects. ``user_id``
    binds the device to a verified user; it only ever sees that user's devices.
    """
    session = DeviceSession(
        session_id=_new_token(),
        device_id=_normalize_identifier(device_id, "Device id"),
        room_name=_normalize_identifier(room_name, "Room name"),
        label=normalize_label(label),
        transport=coerce_transport(transport),
        last_seen=_resolve_now(now),
        user_id=None if user_id is None else require_user_id(user_id),
    )
    with closing(_connect()) as connection:
        connection.execute(
            """
            INSERT INTO device_sessions
                (session_id, device_id, room_name, label, transport, last_seen, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(device_id) DO UPDATE SET
                session_id = excluded.session_id,
                room_name = excluded.room_name,
                label = excluded.label,
                transport = excluded.transport,
                last_seen = excluded.last_seen,
                user_id = excluded.user_id
            """,
            (
                session.session_id,
                session.device_id,
                session.room_name,
                session.label,
                session.transport.value,
                session.last_seen,
                session.user_id,
            ),
        )
    return session


def get_session(session_id: str, *, now: int | None = None) -> DeviceSession | None:
    """Return the live session for a token, or ``None`` if unknown or expired."""
    if not isinstance(session_id, str) or not session_id:
        return None
    cutoff = _resolve_now(now) - SESSION_TTL_SECONDS
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT * FROM device_sessions WHERE session_id = ? AND last_seen >= ?",
            (session_id, cutoff),
        ).fetchone()
    return _session_from_row(row) if row else None


def heartbeat(session_id: str, *, now: int | None = None) -> DeviceSession | None:
    """Refresh a live session's ``last_seen``, or return ``None`` if it has lapsed."""
    if not isinstance(session_id, str) or not session_id:
        return None
    moment = _resolve_now(now)
    with closing(_connect()) as connection:
        updated = connection.execute(
            "UPDATE device_sessions SET last_seen = ? WHERE session_id = ? AND last_seen >= ?",
            (moment, session_id, moment - SESSION_TTL_SECONDS),
        )
        if updated.rowcount != 1:
            return None
        row = connection.execute(
            "SELECT * FROM device_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    return _session_from_row(row) if row else None


def list_active_sessions(
    *, now: int | None = None, limit: int = MAX_ACTIVE_SESSIONS, user_id: object = UNSCOPED
) -> list[DeviceSession]:
    """List unexpired sessions, most recently seen first, capped for safety.

    ``user_id`` restricts the list to one user's devices (``None`` for legacy,
    unscoped devices); the default lists every owner for operator tooling.
    """
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer.")
    cutoff = _resolve_now(now) - SESSION_TTL_SECONDS
    scope_sql = ""
    params: list[object] = [cutoff]
    if user_id is not UNSCOPED:
        scope_sql = " AND user_id IS ?"
        params.append(None if user_id is None else require_user_id(user_id))
    params.append(min(limit, MAX_ACTIVE_SESSIONS))
    with closing(_connect()) as connection:
        rows = connection.execute(
            f"""
            SELECT * FROM device_sessions
            WHERE last_seen >= ?{scope_sql}
            ORDER BY last_seen DESC, device_id
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [_session_from_row(row) for row in rows]


def purge_expired_sessions(*, now: int | None = None) -> int:
    """Delete every lapsed session and expired handoff; return sessions removed."""
    moment = _resolve_now(now)
    with closing(_connect()) as connection:
        removed = connection.execute(
            "DELETE FROM device_sessions WHERE last_seen < ?", (moment - SESSION_TTL_SECONDS,)
        ).rowcount
        connection.execute(
            "DELETE FROM device_handoffs WHERE created_at < ?", (moment - HANDOFF_TTL_SECONDS,)
        )
    return removed


def create_handoff(
    session_id: str,
    *,
    context: Mapping[str, Any] | None = None,
    now: int | None = None,
) -> HandoffRecord:
    """Snapshot a live session's context behind a fresh, opaque handoff token."""
    moment = _resolve_now(now)
    snapshot = bound_context(context)
    source = get_session(session_id, now=moment)
    if source is None:
        raise LookupError("No live device session for that session id.")

    record = HandoffRecord(
        handoff_id=_new_token(),
        device_id=source.device_id,
        room_name=source.room_name,
        context=snapshot,
        created_at=moment,
        user_id=source.user_id,
    )
    with closing(_connect()) as connection:
        connection.execute(
            """
            INSERT INTO device_handoffs
                (handoff_id, device_id, room_name, context, created_at, claimed_at, claimed_by,
                 user_id)
            VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)
            """,
            (
                record.handoff_id,
                record.device_id,
                record.room_name,
                json.dumps(record.context, sort_keys=True),
                record.created_at,
                record.user_id,
            ),
        )
    return record


def claim_handoff(
    handoff_id: str, *, device_id: str, now: int | None = None, user_id: str | None = None
) -> HandoffRecord | None:
    """Claim a handoff exactly once; return ``None`` if unknown, expired, or taken.

    The conditional ``UPDATE`` inside an immediate transaction is what makes the
    claim single-winner: racing devices serialize on the write lock and every
    loser sees ``claimed_at`` already set. A handoff staged by one user's
    device can only ever be claimed by another device of the same user.
    """
    if not isinstance(handoff_id, str) or not handoff_id:
        return None
    claimant = _normalize_identifier(device_id, "Device id")
    owner = None if user_id is None else require_user_id(user_id)
    moment = _resolve_now(now)

    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            claimed = connection.execute(
                """
                UPDATE device_handoffs SET claimed_at = ?, claimed_by = ?
                WHERE handoff_id = ? AND claimed_at IS NULL AND created_at >= ?
                  AND user_id IS ?
                """,
                (moment, claimant, handoff_id, moment - HANDOFF_TTL_SECONDS, owner),
            )
            if claimed.rowcount != 1:
                connection.execute("ROLLBACK")
                return None
            row = connection.execute(
                "SELECT * FROM device_handoffs WHERE handoff_id = ?", (handoff_id,)
            ).fetchone()
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return _handoff_from_row(row) if row else None
