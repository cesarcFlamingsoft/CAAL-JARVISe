"""Persistent local reminder tools for CAAL."""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STORE_PATH = Path(os.getenv("CAAL_DATA_DIR", "/app/data")) / "assistant.sqlite3"


def _result(message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"status": "ok", "message": message, "data": data or {}}


def _connect() -> sqlite3.Connection:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(STORE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS reminders (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            due TEXT,
            list_name TEXT NOT NULL,
            notes TEXT NOT NULL,
            completed INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )
    return connection


def create_reminder(
    title: str,
    due: str | None = None,
    list_name: str | None = None,
    list: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Create a reminder in CAAL's persistent local reminder store."""
    reminder = {
        "id": str(uuid.uuid4()),
        "title": title.strip(),
        "due": due or None,
        "list": (list or list_name or "Reminders").strip(),
        "notes": (notes or "").strip(),
        "completed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if not reminder["title"]:
        raise ValueError("Reminder title is required")

    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO reminders (id, title, due, list_name, notes, completed, created_at)
            VALUES (:id, :title, :due, :list, :notes, :completed, :created_at)
            """,
            reminder,
        )
    return _result(f"Created reminder: {reminder['title']}.", reminder)


def list_reminders(include_completed: bool = False) -> dict[str, Any]:
    """List local reminders, ordered by due date then creation time."""
    query = "SELECT id, title, due, list_name, notes, completed, created_at FROM reminders"
    if not include_completed:
        query += " WHERE completed = 0"
    query += " ORDER BY due IS NULL, due, created_at"
    with _connect() as connection:
        rows = connection.execute(query).fetchall()
    reminders = [
        {
            "id": row["id"],
            "title": row["title"],
            "due": row["due"],
            "list": row["list_name"],
            "notes": row["notes"],
            "completed": bool(row["completed"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]
    return _result(
        f"Found {len(reminders)} {'reminder' if len(reminders) == 1 else 'reminders'}.",
        {"reminders": reminders},
    )
