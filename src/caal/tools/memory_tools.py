"""Durable, explicit user preferences for CAAL, scoped per user.

A memory is only ever written on the user's explicit request and only ever
read back for the same user. The ``user_id`` scope is supplied by the session
(see :mod:`caal.user_scope`), never by the model. Rows written before
multi-user existed live under the empty legacy scope, which is what a
deployment without identity configured keeps using; the bootstrap
administrator may adopt them exactly once.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from caal.user_scope import require_user_id

STORE_PATH = Path(os.getenv("CAAL_DATA_DIR", "/app/data")) / "assistant.sqlite3"

LEGACY_SCOPE = ""
MAX_KEY_LENGTH = 100
MAX_VALUE_LENGTH = 2000
MAX_MEMORIES_LISTED = 200
_BUSY_TIMEOUT_SECONDS = 5.0


def _connect() -> sqlite3.Connection:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(STORE_PATH, timeout=_BUSY_TIMEOUT_SECONDS, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout = {int(_BUSY_TIMEOUT_SECONDS * 1000)}")
    _ensure_schema(connection)
    return connection


def _columns(connection: sqlite3.Connection) -> set[str]:
    return {row["name"] for row in connection.execute("PRAGMA table_info(preferences)")}


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """Create the scoped table, upgrading a pre-multi-user table in place."""
    columns = _columns(connection)
    if columns and "user_id" in columns:
        return
    connection.execute("BEGIN IMMEDIATE")
    try:
        columns = _columns(connection)  # re-check under the write lock
        if not columns:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS preferences (
                    user_id TEXT NOT NULL DEFAULT '',
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at INTEGER NOT NULL DEFAULT (unixepoch()),
                    PRIMARY KEY (user_id, key)
                )
                """
            )
        elif "user_id" not in columns:
            connection.execute(
                """
                CREATE TABLE preferences_scoped (
                    user_id TEXT NOT NULL DEFAULT '',
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at INTEGER NOT NULL DEFAULT (unixepoch()),
                    PRIMARY KEY (user_id, key)
                )
                """
            )
            connection.execute(
                "INSERT INTO preferences_scoped (user_id, key, value, updated_at) "
                "SELECT '', key, value, updated_at FROM preferences"
            )
            connection.execute("DROP TABLE preferences")
            connection.execute("ALTER TABLE preferences_scoped RENAME TO preferences")
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise


def _scope(user_id: object) -> str:
    """The storage scope for a caller: the legacy scope, or a validated user id."""
    if user_id is None:
        return LEGACY_SCOPE
    return require_user_id(user_id)


def _clean(value: object, *, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.isprintable():
        raise ValueError(f"Preference {name} must be plain text")
    cleaned = " ".join(value.split())
    if not cleaned or len(cleaned) > limit:
        raise ValueError(f"Preference {name} must be 1 to {limit} characters")
    return cleaned


def remember(key: str, value: str, *, user_id: str | None = None) -> dict[str, Any]:
    """Save an explicit user preference for future voice sessions of the same user."""
    scope = _scope(user_id)
    normalized_key = _clean(key, name="key", limit=MAX_KEY_LENGTH)
    normalized_value = _clean(value, name="value", limit=MAX_VALUE_LENGTH)
    with closing(_connect()) as connection:
        connection.execute(
            """
            INSERT INTO preferences (user_id, key, value, updated_at)
            VALUES (?, ?, ?, unixepoch())
            ON CONFLICT(user_id, key) DO UPDATE SET
                value = excluded.value, updated_at = excluded.updated_at
            """,
            (scope, normalized_key, normalized_value),
        )
    return {
        "status": "ok",
        "message": f"I'll remember that {normalized_key} is {normalized_value}.",
        "data": {"key": normalized_key, "value": normalized_value},
    }


def recall(key: str = "", *, user_id: str | None = None) -> dict[str, Any]:
    """Recall one preference, or list every preference, for the same user only."""
    scope = _scope(user_id)
    normalized_key = " ".join(key.split()) if isinstance(key, str) else ""
    with closing(_connect()) as connection:
        if normalized_key:
            row = connection.execute(
                "SELECT key, value FROM preferences WHERE user_id = ? AND key = ?",
                (scope, normalized_key),
            ).fetchone()
            if row is None:
                return {
                    "status": "not_found",
                    "message": f"I don't have a saved preference for {normalized_key}.",
                    "data": {},
                }
            preference = {"key": row["key"], "value": row["value"]}
            return {
                "status": "ok",
                "message": f"{preference['key']} is {preference['value']}.",
                "data": preference,
            }
        rows = connection.execute(
            "SELECT key, value FROM preferences WHERE user_id = ? ORDER BY key LIMIT ?",
            (scope, MAX_MEMORIES_LISTED),
        ).fetchall()
    memories = [{"key": row["key"], "value": row["value"]} for row in rows]
    count = len(memories)
    return {
        "status": "ok",
        "message": f"I remember {count} {'preference' if count == 1 else 'preferences'}.",
        "data": {"memories": memories},
    }


def adopt_legacy_memories(user_id: str) -> int:
    """Move every legacy (pre-multi-user) memory into ``user_id``'s scope.

    A key the user already has keeps the user's own value. Returns the number
    of legacy rows that were moved or superseded; a second call finds none.
    """
    scope = require_user_id(user_id)
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            (count,) = connection.execute(
                "SELECT COUNT(*) FROM preferences WHERE user_id = ?", (LEGACY_SCOPE,)
            ).fetchone()
            connection.execute(
                "DELETE FROM preferences WHERE user_id = ? AND key IN "
                "(SELECT key FROM preferences WHERE user_id = ?)",
                (LEGACY_SCOPE, scope),
            )
            connection.execute(
                "UPDATE preferences SET user_id = ? WHERE user_id = ?", (scope, LEGACY_SCOPE)
            )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return int(count)
