"""Durable background task manager.

A background task is a piece of work the user asked the assistant to carry
on with while the conversation moves on ("look into this and get back to
me"). This module owns the whole lifecycle:

* a persistent queue in the shared CAAL SQLite store, so queued work
  survives a restart and work that was mid-flight is reported as
  interrupted rather than silently lost;
* an async runner that executes queued tasks through an injected worker
  coroutine with a hard concurrency cap;
* an exact-once notification claim, so a finished task is announced to the
  user by one consumer and never twice;
* a one-time callback authorization, so "hang up and call me back when
  you're done" places exactly one outbound call once the task settles, and
  none at all if the task is cancelled or interrupted first.

Task text is bounded and redacted before it is persisted. Nothing in this
module logs task text, task ids, or callback destinations, and task
snapshots never include text in ``repr``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import secrets
import sqlite3
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .user_scope import require_user_id

logger = logging.getLogger(__name__)


class _Unscoped:
    """Marker for "do not filter by user": operator tooling and legacy callers."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSCOPED"


UNSCOPED: Any = _Unscoped()


def _scope_clause(user_id: object) -> tuple[str, list[object]]:
    """SQL fragment binding a query to one user scope (``None`` = unscoped rows)."""
    if user_id is UNSCOPED:
        return "", []
    return " AND user_id IS ?", [None if user_id is None else require_user_id(user_id)]

STORE_PATH = Path(os.getenv("CAAL_DATA_DIR", "/app/data")) / "assistant.sqlite3"
_BUSY_TIMEOUT_SECONDS = 5.0

# Statuses.
QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"
INTERRUPTED = "interrupted"
TERMINAL_STATUSES = frozenset({SUCCEEDED, FAILED, CANCELLED, INTERRUPTED})
ALL_STATUSES = frozenset({QUEUED, RUNNING}) | TERMINAL_STATUSES
# Outcomes that may trigger a callback. Cancelled work was acknowledged when
# the user asked; interrupted work belongs to a process that is gone.
SETTLED_STATUSES = frozenset({SUCCEEDED, FAILED})

# Bounds.
MAX_REQUEST_CHARS = 2_000
MAX_RESULT_CHARS = 4_000
MAX_ERROR_CHARS = 500
MAX_QUEUE_DEPTH = 25
MAX_LIST = 50
DEFAULT_MAX_CONCURRENCY = 2
MAX_CONCURRENCY_LIMIT = 8
MAX_CLASSIFIER_CHARS = 2_000

_TRUNCATION_MARK = "…"
_REDACTED = "[REDACTED]"

# Conservative credential shapes. Values after a credential-ish label, common
# provider token prefixes, and bearer headers. False positives cost a little
# context; false negatives leak a secret into the store, so err toward masking.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?i)\b(api[_ -]?key|secret|token|password|passwd|pwd|authorization|auth)"
        r"\b\s*[:=]\s*(?:bearer\s+)?(?P<value>\S+)"
    ),
    re.compile(r"(?i)\bbearer\s+(?P<value>[A-Za-z0-9._~+/=-]{8,})"),
    re.compile(r"\b(?P<value>sk-[A-Za-z0-9_-]{8,})"),
    re.compile(r"\b(?P<value>(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,})\b"),
    re.compile(r"\b(?P<value>xox[abprs]-[A-Za-z0-9-]{10,})\b"),
    re.compile(r"\b(?P<value>AKIA[0-9A-Z]{16})\b"),
)


class QueueFullError(RuntimeError):
    """Raised when the bounded queue cannot accept more work."""


@dataclass(frozen=True, repr=False)
class BackgroundTask:
    """Snapshot of one task. Text fields are deliberately kept out of ``repr``."""

    task_id: str
    status: str
    session_key: str | None
    created_at: int
    updated_at: int
    started_at: int | None
    finished_at: int | None
    notified_at: int | None
    request: str
    result: str | None
    error: str | None
    # Opaque id of the user the work belongs to; ``None`` for legacy/unscoped work.
    user_id: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def __repr__(self) -> str:
        return (
            f"BackgroundTask(task_id={self.task_id!r}, status={self.status!r}, "
            f"session_key={self.session_key!r}, user_id={self.user_id!r}, "
            f"request_chars={len(self.request)}, "
            f"result_chars={len(self.result or '')}, error_chars={len(self.error or '')}, "
            f"created_at={self.created_at}, finished_at={self.finished_at}, "
            f"notified_at={self.notified_at})"
        )

    __str__ = __repr__


# ---------------------------------------------------------------------------
# Text hygiene
# ---------------------------------------------------------------------------


def redact_secrets(text: str) -> str:
    """Mask credential-shaped substrings. Deterministic and idempotent."""
    if not text:
        return ""
    redacted = text
    for pattern in _SECRET_PATTERNS:

        def _mask(match: re.Match[str]) -> str:
            start, end = match.span("value")
            return match.group(0)[: start - match.start()] + _REDACTED

        redacted = pattern.sub(_mask, redacted)
    return redacted


def _bound(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - len(_TRUNCATION_MARK)] + _TRUNCATION_MARK


def _clean(text: object, limit: int) -> str:
    """Redact first so a truncated secret cannot escape the redaction shape."""
    return _bound(redact_secrets(str(text)).strip(), limit)


_TASK_ID = re.compile(r"^bt_[0-9a-f]{16}$")


def _new_task_id() -> str:
    return "bt_" + secrets.token_hex(8)


def is_valid_task_id(value: object) -> bool:
    """Whether ``value`` has the exact opaque shape this module issues."""
    return isinstance(value, str) and _TASK_ID.fullmatch(value) is not None


def _now() -> int:
    return int(time.time())


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def _connect() -> sqlite3.Connection:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(STORE_PATH, timeout=_BUSY_TIMEOUT_SECONDS, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout = {int(_BUSY_TIMEOUT_SECONDS * 1000)}")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS background_tasks (
            task_id TEXT PRIMARY KEY,
            session_key TEXT,
            status TEXT NOT NULL,
            request TEXT NOT NULL,
            result TEXT,
            error TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            started_at INTEGER,
            finished_at INTEGER,
            notified_at INTEGER,
            notified_by TEXT
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS background_tasks_status_idx "
        "ON background_tasks (status, created_at)"
    )
    # One pending callback per task. The destination is the only copy of the
    # number this feature keeps, and it is re-checked against the allowlist
    # at dial time; it is never logged.
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS background_callbacks (
            task_id TEXT PRIMARY KEY,
            destination TEXT NOT NULL,
            armed_by TEXT,
            armed_at INTEGER NOT NULL,
            claimed_at INTEGER,
            claimed_by TEXT
        )
        """
    )
    # Multi-user: work and callbacks are owned by an opaque user id. A callback
    # bound to a user stores an empty destination; the number is resolved from
    # the user's profile at dial time, never persisted here.
    _ensure_column(connection, "background_tasks", "user_id", "TEXT")
    _ensure_column(connection, "background_callbacks", "user_id", "TEXT")
    return connection


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    """Upgrade a table written before ``column`` existed."""
    columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _from_row(row: sqlite3.Row) -> BackgroundTask:
    return BackgroundTask(
        task_id=row["task_id"],
        status=row["status"],
        session_key=row["session_key"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        notified_at=row["notified_at"],
        request=row["request"],
        result=row["result"],
        error=row["error"],
        user_id=row["user_id"],
    )


def _fetch(connection: sqlite3.Connection, task_id: str) -> BackgroundTask | None:
    row = connection.execute(
        "SELECT * FROM background_tasks WHERE task_id = ?", (task_id,)
    ).fetchone()
    return _from_row(row) if row else None


def enqueue(
    request: str, *, session_key: str | None = None, user_id: str | None = None
) -> BackgroundTask:
    """Persist a new queued task. The stored request is redacted and bounded.

    ``user_id`` files the work under a verified user; only that user's
    sessions can later inspect, cancel, or arm a callback for it.
    """
    cleaned = _clean(request or "", MAX_REQUEST_CHARS)
    if not cleaned:
        raise ValueError("background task request must not be empty")
    owner = None if user_id is None else require_user_id(user_id)
    task_id = _new_task_id()
    moment = _now()
    with _connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            (depth,) = connection.execute(
                "SELECT COUNT(*) FROM background_tasks WHERE status IN (?, ?)",
                (QUEUED, RUNNING),
            ).fetchone()
            if depth >= MAX_QUEUE_DEPTH:
                raise QueueFullError(f"background task queue is full ({MAX_QUEUE_DEPTH})")
            connection.execute(
                """
                INSERT INTO background_tasks
                    (task_id, session_key, status, request, created_at, updated_at, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (task_id, session_key, QUEUED, cleaned, moment, moment, owner),
            )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        task = _fetch(connection, task_id)
    assert task is not None
    logger.info("background task queued (%d chars)", len(cleaned))
    return task


def get_task(task_id: str) -> BackgroundTask | None:
    with _connect() as connection:
        return _fetch(connection, task_id)


def list_tasks(
    *,
    session_key: str | None = None,
    statuses: Iterable[str] | None = None,
    limit: int = MAX_LIST,
    user_id: object = UNSCOPED,
) -> list[BackgroundTask]:
    """Oldest first. Bounded by ``MAX_LIST`` regardless of the requested limit.

    ``user_id`` restricts the listing to one user's work (``None`` for
    unscoped legacy work); the default lists every owner.
    """
    clauses: list[str] = []
    params: list[object] = []
    if session_key is not None:
        clauses.append("session_key = ?")
        params.append(session_key)
    if statuses is not None:
        wanted = [status for status in statuses if status in ALL_STATUSES]
        if not wanted:
            return []
        clauses.append(f"status IN ({', '.join('?' for _ in wanted)})")
        params.extend(wanted)
    scope_sql, scope_params = _scope_clause(user_id)
    if scope_sql:
        clauses.append(scope_sql[len(" AND ") :])
        params.extend(scope_params)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(max(1, min(int(limit), MAX_LIST)))
    with _connect() as connection:
        rows = connection.execute(
            f"SELECT * FROM background_tasks {where} ORDER BY created_at, rowid LIMIT ?",
            params,
        ).fetchall()
    return [_from_row(row) for row in rows]


def _mark_running(task_id: str) -> bool:
    """Move a queued task to running. Returns False if it was no longer queued."""
    moment = _now()
    with _connect() as connection:
        cursor = connection.execute(
            "UPDATE background_tasks SET status = ?, started_at = ?, updated_at = ? "
            "WHERE task_id = ? AND status = ?",
            (RUNNING, moment, moment, task_id, QUEUED),
        )
        return cursor.rowcount == 1


def _claim_next_queued() -> BackgroundTask | None:
    """Atomically move the oldest queued task to running and return it."""
    moment = _now()
    with _connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT task_id FROM background_tasks WHERE status = ? "
                "ORDER BY created_at, rowid LIMIT 1",
                (QUEUED,),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            connection.execute(
                "UPDATE background_tasks SET status = ?, started_at = ?, updated_at = ? "
                "WHERE task_id = ?",
                (RUNNING, moment, moment, row["task_id"]),
            )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        return _fetch(connection, row["task_id"])


def _finish(
    task_id: str,
    status: str,
    *,
    result: object | None = None,
    error: object | None = None,
    from_statuses: Iterable[str] = (RUNNING,),
) -> bool:
    """Move a task into a terminal status.

    Conditional on the task still being in one of ``from_statuses`` so a
    worker that limps in after a cancel cannot overwrite the cancellation.
    """
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"not a terminal status: {status!r}")
    sources = list(from_statuses)
    moment = _now()
    with _connect() as connection:
        cursor = connection.execute(
            "UPDATE background_tasks SET status = ?, result = ?, error = ?, "
            "finished_at = ?, updated_at = ? "
            f"WHERE task_id = ? AND status IN ({', '.join('?' for _ in sources)})",
            (
                status,
                _clean(result, MAX_RESULT_CHARS) if result is not None else None,
                _clean(error, MAX_ERROR_CHARS) if error is not None else None,
                moment,
                moment,
                task_id,
                *sources,
            ),
        )
        changed = cursor.rowcount == 1
        if changed and status not in SETTLED_STATUSES:
            # Only work that actually settled may call the user back.
            _disarm_callback(connection, task_id)
        return changed


def cancel(task_id: str) -> bool:
    """Cancel a queued or running task in the store. Returns True if it changed."""
    changed = _finish(task_id, CANCELLED, from_statuses=(QUEUED, RUNNING))
    if changed:
        logger.info("background task cancelled")
    return changed


def recover_interrupted() -> int:
    """Call once at startup, before any runner: running tasks cannot be running."""
    moment = _now()
    with _connect() as connection:
        cursor = connection.execute(
            "UPDATE background_tasks SET status = ?, error = ?, finished_at = ?, updated_at = ? "
            "WHERE status = ?",
            (INTERRUPTED, "interrupted by restart", moment, moment, RUNNING),
        )
        count = cursor.rowcount
        # Interrupted work never calls anyone back; the fallback reports it.
        connection.execute(
            "DELETE FROM background_callbacks WHERE claimed_at IS NULL AND task_id IN "
            "(SELECT task_id FROM background_tasks WHERE status = ?)",
            (INTERRUPTED,),
        )
    if count:
        logger.warning("recovered %d interrupted background task(s)", count)
    return count


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def _disarm_callback(connection: sqlite3.Connection, task_id: str) -> bool:
    cursor = connection.execute(
        "DELETE FROM background_callbacks WHERE task_id = ? AND claimed_at IS NULL",
        (task_id,),
    )
    return cursor.rowcount == 1


@dataclass(frozen=True)
class CallbackTarget:
    """Who a claimed callback should reach: a stored legacy number, or a user.

    A user-bound target carries no number at all; the dialer resolves the
    user's *current* approved number from their profile at dial time.
    """

    destination: str | None = field(default=None, repr=False)
    user_id: str | None = None


def arm_callback(
    task_id: str,
    destination: str | None,
    *,
    session_key: str,
    user_id: str | None = None,
) -> bool:
    """Authorize one callback for an open task of ``session_key``, replacing any earlier one.

    Refuses silently (returns False) when the task is unknown, already
    terminal, or belongs to another session or another user, so a caller can
    only ever arm the work they scheduled themselves. Idempotent for the same
    task. With ``user_id`` the callback is bound to that user and stores no
    number; otherwise ``destination`` is the legacy allowlisted number.
    """
    owner = None if user_id is None else require_user_id(user_id)
    if destination is None and owner is None:
        raise ValueError("a callback needs either a destination or a user to dial")
    if not is_valid_task_id(task_id) or destination == "" or not session_key:
        return False
    moment = _now()
    with _connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT 1 FROM background_tasks WHERE task_id = ? AND session_key = ? "
                "AND user_id IS ? AND status IN (?, ?)",
                (task_id, session_key, owner, QUEUED, RUNNING),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return False
            connection.execute(
                "INSERT INTO background_callbacks "
                "(task_id, destination, armed_by, armed_at, user_id) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(task_id) DO UPDATE SET destination = excluded.destination, "
                "armed_by = excluded.armed_by, armed_at = excluded.armed_at, "
                "user_id = excluded.user_id "
                "WHERE claimed_at IS NULL",
                (task_id, destination or "", session_key, moment, owner),
            )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    logger.info("background task callback armed")
    return True


def callback_armed(task_id: str) -> bool:
    """Whether an unclaimed callback authorization exists for the task."""
    with _connect() as connection:
        row = connection.execute(
            "SELECT 1 FROM background_callbacks WHERE task_id = ? AND claimed_at IS NULL",
            (task_id,),
        ).fetchone()
    return row is not None


def disarm_callback(task_id: str) -> bool:
    """Withdraw an unclaimed callback authorization. Returns True if one existed."""
    with _connect() as connection:
        return _disarm_callback(connection, task_id)


def claim_callback_target(
    task_id: str, claimant: str, *, include_user_bound: bool = True
) -> CallbackTarget | None:
    """Exactly one caller wins the right to place the callback; returns its target.

    Succeeds only once, and only for a task that succeeded or failed, so a
    cancelled or interrupted task can never dial out even if a claim races
    its cancellation.
    """
    moment = _now()
    with _connect() as connection:
        user_clause = "" if include_user_bound else " AND user_id IS NULL"
        cursor = connection.execute(
            "UPDATE background_callbacks SET claimed_at = ?, claimed_by = ? "
            "WHERE task_id = ? AND claimed_at IS NULL AND task_id IN "
            f"(SELECT task_id FROM background_tasks WHERE status IN (?, ?)){user_clause}",
            (moment, claimant, task_id, *SETTLED_STATUSES),
        )
        if cursor.rowcount != 1:
            return None
        row = connection.execute(
            "SELECT destination, user_id FROM background_callbacks WHERE task_id = ?", (task_id,)
        ).fetchone()
    if row is None:
        return None
    logger.info("background task callback claimed")
    return CallbackTarget(destination=row["destination"] or None, user_id=row["user_id"])


def claim_callback(task_id: str, claimant: str) -> str | None:
    """Legacy claim: the stored destination of a number-bound callback, or ``None``.

    User-bound callbacks are never claimed here; they carry no number.
    """
    target = claim_callback_target(task_id, claimant, include_user_bound=False)
    return target.destination if target is not None else None


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------


def _terminal_placeholders() -> str:
    return ", ".join("?" for _ in TERMINAL_STATUSES)


def pending_notifications(
    *, session_key: str | None = None, limit: int = MAX_LIST, user_id: object = UNSCOPED
) -> list[BackgroundTask]:
    """Finished tasks nobody has announced yet, oldest first, optionally for one user."""
    params: list[object] = list(TERMINAL_STATUSES)
    where = f"status IN ({_terminal_placeholders()}) AND notified_at IS NULL"
    if session_key is not None:
        where += " AND session_key = ?"
        params.append(session_key)
    scope_sql, scope_params = _scope_clause(user_id)
    where += scope_sql
    params.extend(scope_params)
    params.append(max(1, min(int(limit), MAX_LIST)))
    with _connect() as connection:
        rows = connection.execute(
            f"SELECT * FROM background_tasks WHERE {where} ORDER BY finished_at, rowid LIMIT ?",
            params,
        ).fetchall()
    return [_from_row(row) for row in rows]


def claim_notification(task_id: str, claimant: str) -> BackgroundTask | None:
    """Exactly one caller wins the right to announce a finished task.

    The claim is a single conditional UPDATE, so concurrent claimants across
    threads or processes cannot both succeed.
    """
    moment = _now()
    with _connect() as connection:
        cursor = connection.execute(
            "UPDATE background_tasks SET notified_at = ?, notified_by = ?, updated_at = ? "
            f"WHERE task_id = ? AND notified_at IS NULL AND status IN ({_terminal_placeholders()})",
            (moment, claimant, moment, task_id, *TERMINAL_STATUSES),
        )
        if cursor.rowcount != 1:
            return None
        task = _fetch(connection, task_id)
    logger.info("background task notification claimed (%s)", task.status)
    return task


def claim_next_notification(
    session_key: str | None, claimant: str, *, user_id: object = UNSCOPED
) -> BackgroundTask | None:
    """Claim the oldest unannounced finished task for a session (and user), if any."""
    for candidate in pending_notifications(session_key=session_key, user_id=user_id):
        claimed = claim_notification(candidate.task_id, claimant)
        if claimed is not None:
            return claimed
    return None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

Worker = Callable[[BackgroundTask], Awaitable[object]]
Settled = Callable[[str], Awaitable[None]]


class BackgroundTaskRunner:
    """Executes queued tasks through an injected worker coroutine.

    The runner owns no task text: it hands the persisted snapshot to the
    worker and stores whatever comes back, bounded and redacted. Task ids are
    opaque handles and never appear in its logs either.

    ``on_settled`` is awaited with the task id once a task has succeeded or
    failed and its outcome is persisted, so a consumer can announce it
    without polling. Cancellations and shutdown interruptions do not fire it:
    the former is acknowledged when requested, the latter belongs to a dying
    process.
    """

    def __init__(
        self,
        worker: Worker,
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        on_settled: Settled | None = None,
    ):
        if not 1 <= int(max_concurrency) <= MAX_CONCURRENCY_LIMIT:
            raise ValueError(f"max_concurrency must be in 1..{MAX_CONCURRENCY_LIMIT}")
        self._worker = worker
        self._on_settled = on_settled
        self._max_concurrency = int(max_concurrency)
        self._active: dict[str, asyncio.Task[None]] = {}
        self._waiters: dict[str, asyncio.Event] = {}
        self._started = False
        self._stopping = False

    @property
    def running_count(self) -> int:
        return len(self._active)

    async def start(self, *, recover: bool = True) -> None:
        """Begin draining the queue.

        ``recover`` marks tasks left running by a previous process as
        interrupted first. Pass ``False`` when that already happened at process
        boot and other runners in sibling processes may legitimately be mid-task.
        """
        if self._started:
            return
        if recover:
            recover_interrupted()
        self._started = True
        self._stopping = False
        self._pump()

    def poke(self) -> None:
        """Pick up work that was enqueued directly rather than via ``submit``."""
        self._pump()

    async def stop(self) -> None:
        """Interrupt in-flight work and refuse new submissions."""
        if not self._started:
            return
        self._stopping = True
        active = list(self._active.items())
        for task_id, aio_task in active:
            _finish(task_id, INTERRUPTED, error="interrupted by shutdown")
            aio_task.cancel()
        for _, aio_task in active:
            try:
                await aio_task
            except (asyncio.CancelledError, Exception):
                pass
        self._active.clear()
        self._started = False

    async def submit(self, request: str, *, session_key: str | None = None) -> BackgroundTask:
        if not self._started or self._stopping:
            raise RuntimeError("background task runner is not running")
        task = enqueue(request, session_key=session_key)
        self._pump()
        return task

    async def cancel(self, task_id: str) -> bool:
        changed = cancel(task_id)
        aio_task = self._active.get(task_id)
        if aio_task is not None:
            aio_task.cancel()
        elif changed:
            # Cancelled straight out of the queue: nobody will run it, so
            # release anyone waiting on it now.
            self._wake(task_id)
        return changed

    async def wait(self, task_id: str, *, timeout: float | None = None) -> BackgroundTask:
        """Block until the task is terminal and its worker has unwound.

        Raises ``KeyError`` for unknown ids and ``asyncio.TimeoutError`` on timeout.
        """
        task = get_task(task_id)
        if task is None:
            raise KeyError(task_id)
        if task.is_terminal and task_id not in self._active:
            return task
        event = self._waiters.setdefault(task_id, asyncio.Event())
        try:
            await asyncio.wait_for(event.wait(), timeout)
        finally:
            if event.is_set():
                self._waiters.pop(task_id, None)
        task = get_task(task_id)
        assert task is not None
        return task

    def _pump(self) -> None:
        if not self._started or self._stopping:
            return
        while len(self._active) < self._max_concurrency:
            task = _claim_next_queued()
            if task is None:
                return
            aio_task = asyncio.get_running_loop().create_task(self._run(task))
            self._active[task.task_id] = aio_task
            aio_task.add_done_callback(lambda _done, task_id=task.task_id: self._on_done(task_id))

    def _wake(self, task_id: str) -> None:
        event = self._waiters.get(task_id)
        if event is not None:
            event.set()

    def _on_done(self, task_id: str) -> None:
        self._active.pop(task_id, None)
        self._wake(task_id)
        self._pump()

    async def _run(self, task: BackgroundTask) -> None:
        logger.info("background task started (%d running)", len(self._active) + 1)
        try:
            result = await self._worker(task)
        except asyncio.CancelledError:
            _finish(task.task_id, CANCELLED)
            logger.info("background task cancelled while running")
            raise
        except Exception as exc:
            settled = _finish(task.task_id, FAILED, error=f"{type(exc).__name__}: {exc}")
            logger.warning("background task failed (%s)", type(exc).__name__)
        else:
            settled = _finish(task.task_id, SUCCEEDED, result="" if result is None else result)
            if settled:
                logger.info("background task succeeded")
            else:
                logger.info("background task finished after leaving running")
        if settled and self._on_settled is not None:
            try:
                await self._on_settled(task.task_id)
            except Exception:
                logger.warning("background task settlement hook failed", exc_info=True)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

_BACKGROUND_CUES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bin the background\b"),
    re.compile(r"\bbackground (task|job)\b"),
    re.compile(r"\b(get|come|circle) back to me\b"),
    re.compile(
        r"\b(let me know|tell me|notify me|ping me|report back)\b.{0,20}"
        r"\b(when|once)\b.{0,12}\b(done|finished|ready|complete|completed)\b"
    ),
    re.compile(r"\b(keep|carry on|continue) working on\b.*\bwhile i\b"),
)

# Requests that mention "background" but plainly are not about deferred work.
_BACKGROUND_VETOES: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bbackground (music|noise|sound|audio|image|wallpaper|colou?r|picture|photo|video)"
    ),
    re.compile(
        r"\b(music|jazz|song|songs|playlist|audio|noise|sounds?|video)\b.{0,30}"
        r"\bin the background\b"
    ),
    re.compile(r"^\s*(what|what's|whats|why|how|who|where|define|explain)\b"),
)


# Exact local control phrases. These are whole-utterance matches so that a
# sentence which merely mentions a background task ("cancel the background
# task and also book the flight") still goes to the normal conversation.
_CTRL_PREFIX = r"(?:jarvis[,. ]+)?(?:please )?"
_CTRL_SUFFIX = r"(?: please)?[.!?]*"
_CTRL_TASK = (
    r"(?:the |that |my |our |this |all |all the |all my |all of my )?"
    r"background (?:task|job|work)s?"
)
# The work itself, named the way a user names it once it has been scheduled
# ("is the PDF ready yet?"). Deliberately a closed list of work nouns, so
# "is dinner done yet" or "the status of my order" stay ordinary conversation.
_CTRL_WORK = (
    r"(?:the |that |my |our |this )?"
    r"(?:pdf|report|document|presentation|slides|deck|spreadsheet|summary|list|"
    r"research|investigation|write-?up|task|job|work)s?"
)
_STATUS_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(rf"{_CTRL_PREFIX}{body}{_CTRL_SUFFIX}")
    for body in (
        rf"how(?:'s| is| are) {_CTRL_TASK} (?:going|coming along|doing)",
        rf"(?:what's|what is|whats) the status of {_CTRL_TASK}",
        rf"status of {_CTRL_TASK}",
        rf"is {_CTRL_TASK} (?:done|finished|ready|complete)(?: yet)?",
        rf"(?:is there )?any (?:update|news|progress) on {_CTRL_TASK}",
        rf"{_CTRL_TASK} status",
        rf"how(?:'s| is) {_CTRL_WORK} (?:going|coming along|coming|doing)",
        rf"(?:what's|what is|whats) the status of {_CTRL_WORK}",
        rf"is {_CTRL_WORK} (?:done|finished|ready|complete)(?: yet)?",
        rf"(?:is there )?any (?:update|news|progress) on {_CTRL_WORK}",
    )
)
_CANCEL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(rf"{_CTRL_PREFIX}{body}{_CTRL_SUFFIX}")
    for body in (
        rf"(?:cancel|stop|abort|forget|drop) {_CTRL_TASK}",
        rf"never ?mind (?:about )?{_CTRL_TASK}",
    )
)


def _control_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text[:MAX_CLASSIFIER_CHARS].split()).lower()


def background_status_requested(text: str) -> bool:
    """Whole-utterance match for asking how the background work is going."""
    normalized = _control_text(text)
    return bool(normalized) and any(p.fullmatch(normalized) for p in _STATUS_PATTERNS)


def background_cancel_requested(text: str) -> bool:
    """Whole-utterance match for stopping the background work."""
    normalized = _control_text(text)
    return bool(normalized) and any(p.fullmatch(normalized) for p in _CANCEL_PATTERNS)


def background_task_requested(text: str) -> bool:
    """Conservative, deterministic detection of an explicit "do it in the background" ask.

    Only explicit cues trigger. Anything ambiguous ("later", "remind me") is
    left to the normal conversational path, and the exact status and cancel
    phrases are never mistaken for new work. Input is bounded before matching.
    """
    if not isinstance(text, str):
        return False
    lowered = text[:MAX_CLASSIFIER_CHARS].strip().lower()
    if not lowered:
        return False
    if background_status_requested(lowered) or background_cancel_requested(lowered):
        return False
    if any(veto.search(lowered) for veto in _BACKGROUND_VETOES):
        return False
    return any(cue.search(lowered) for cue in _BACKGROUND_CUES)


# ---------------------------------------------------------------------------
# Inferred long-running work
# ---------------------------------------------------------------------------
#
# A request that plainly asks for a deliverable to be produced (a PDF, report,
# document, presentation) or for something to be researched, investigated, or
# compiled is long-running work even without "in the background". It is read
# the same way as every other local command: offline, deterministic, bounded,
# against fixed cue lists, and only when the sentence is an imperative aimed
# at JARVIS. Questions about such work, instructions on how to do it, reported
# speech, hypotheticals, refusals, and bare fragments stay ordinary conversation.

# Politeness and address that may precede the imperative.
_LONG_WORK_LEAD_IN = re.compile(
    r"^(?:(?:hey|hi|ok|okay|alright|so|and|also|now|actually|jarvis|please|um|uh)[,.! ]+)*"
    r"(?:(?:can|could|would|will) you(?: please)?(?: just)? )?"
    r"(?:(?:please|just|go ahead and|go and) )*"
)

# Any hit means the turn is ordinary conversation.
_LONG_WORK_BLOCKERS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        # Questions by form and capability questions. Polite requests ("can
        # you create...") are stripped as a lead-in before these run.
        r"^(?:what|when|where|who|whom|whose|why|how|which)\b",
        r"^(?:did|does|do|is|are|was|were|has|have|had|should|shall|can|could|would|will) "
        r"(?:you|we|i|they|it|he|she|there|the|that|this|my|your|our)\b",
        r"\bhow (?:do|does|did|can|could|would|should|to)\b",
        r"\bwhat (?:is|are|was|were|does|do|did|would|will|about)\b",
        r"\bwhy (?:do|does|did|would|can|is|are)\b",
        r"\b(?:explain|describe|define)\b",
        r"\b(?:tell|show) me (?:about|what|how|why|whether)\b",
        r"\bare you able to\b",
        r"\bis it possible\b",
        # Hypotheticals and mentions of the request itself.
        r"\bif (?:i|you|we|it|that|this)\b",
        r"\b(?:suppose|hypothetically|for example|for instance|imagine|pretend)\b",
        r"\bwhen i say\b",
        r"\bthe (?:phrase|command|word|words)\b",
        # Reported speech and third parties.
        r"\b(?:said|says|saying|told|tells|telling|asked|asking)\b",
        r"\b(?:he|she|they) (?:will|would|should|could|is|are|was|were|has|have|had)\b",
        # Refusals and negations.
        r"\b(?:do not|don't|dont|never|no need to|without|instead of|rather than)\b",
        r"\bjust (?:tell|give|read|say)\b",
        # Plainly past or already-done work.
        r"\b(?:already|yesterday|earlier|last night|last week|this morning|the other day)\b",
        r"\b(?:i|you) (?:sent|made|created|wrote|did|finished)\b",
    )
)

# What gets produced. A closed list: an email, a haiku, or a text message is
# a quick reply, not a job.
_LONG_WORK_DELIVERABLE = (
    r"(?:pdf|report|document|presentation|slide ?deck|slides|slideshow|deck|"
    r"spreadsheet|summary|write-?up|dossier|white ?paper|brief|proposal|"
    r"list|outline|analysis|comparison)s?"
)
_LONG_WORK_OBJECT = (
    r"(?:(?:me|us) )?(?:(?:a|an|another|some|one|new|a new) )?"
    r"(?:[a-z][a-z-]* )?" + _LONG_WORK_DELIVERABLE + r"\b"
)
# Pronouns and fragments that do not name a subject to work on.
_LONG_WORK_NO_SUBJECT = re.compile(
    r"^(?:it|that|this|them|him|her|us|me|there|here|things|stuff|something|"
    r"anything|everything|nothing)?(?: (?:for|please|now|again|too|as well))*$"
)
_LONG_WORK_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        # Produce a deliverable: "create a PDF of ...", "write a report on ...".
        r"^(?:create|generate|prepare|make|write|produce|build|draft|put together|"
        r"compile|assemble|compose|type up|write up|work up|develop) " + _LONG_WORK_OBJECT,
        # Ask for one: "I need a PDF summary of ...".
        r"^(?:i need|i want|i would like|i'd like|i'll need|get me|give me|send me) "
        + _LONG_WORK_OBJECT,
        # Investigate a subject: "research ...", "look into ...", "do a deep dive into ...".
        r"^(?:research|investigate|look into|dig into|explore|"
        r"do (?:some|a bit of|a little|thorough|detailed|extensive)? ?research (?:on|into|about)|"
        r"do (?:a|an) (?:deep dive|analysis|investigation|comparison) (?:into|on|of)|"
        r"deep dive (?:into|on)|find out everything about|compile (?:a list|everything)"
        r"(?: of| about| on)?) (?P<subject>.+)$",
    )
)


def long_running_work_inferred(text: object) -> bool:
    """Whether a turn plainly asks for long-running work without saying "background".

    Deterministic and bounded: the text is capped at ``MAX_CLASSIFIER_CHARS``
    before anything else, so a request buried past the cap is not seen, while
    one at the front still is. The explicit commands (schedule, status,
    cancel) are never inferred work; they keep their own handling.
    """
    if not isinstance(text, str):
        return False
    lowered = " ".join(text[:MAX_CLASSIFIER_CHARS].split()).lower()
    if not lowered:
        return False
    if (
        background_status_requested(lowered)
        or background_cancel_requested(lowered)
        or background_task_requested(lowered)
    ):
        return False
    lowered = lowered.replace("’", "'")
    plain = " ".join(re.sub(r"[^a-z0-9' -]+", " ", lowered).split())
    stripped = _LONG_WORK_LEAD_IN.sub("", plain, count=1)
    if not stripped:
        return False
    if any(blocker.search(stripped) for blocker in _LONG_WORK_BLOCKERS):
        return False
    for pattern in _LONG_WORK_PATTERNS:
        match = pattern.match(stripped)
        if match is None:
            continue
        subject = match.groupdict().get("subject")
        if subject is not None:
            # An investigation needs a real subject, not a pronoun or a fragment.
            words = subject.split()
            if len(words) < 2 or _LONG_WORK_NO_SUBJECT.fullmatch(subject):
                return False
        return True
    return False
