"""The per-user knowledge index behind the email and calendar answers JARVIS gives.

Three tables in the shared assistant.sqlite3 (schema version 6 in
:mod:caal.user_store), owned by this module:

* knowledge_messages and knowledge_events -- one row per item of one
  connection, holding exactly the bounded, dashboard-safe fields that
  :mod:caal.provider_data already reduces a provider answer to (a subject, a
  sender label, a short preview, a title, a place, a link, times) and nothing
  more: never a body, an attendee, a recipient list, HTML, or a raw payload.
  The text is encrypted with the profile key ring and bound to its exact
  (user, connection, item), so a row copied to another user or connection
  fails to authenticate instead of changing whose mail it is. The numeric
  columns (when a message arrived, when an event starts and ends) are plain,
  so recency and time-window questions are answered from indexes over
  (user, connection, time) rather than by scanning or decrypting everything.
* knowledge_sync -- when each connection was last read for each kind of
  data, what window that read covered, and how it went. This is the
  freshness ledger: a question is answered from the index while the ledger
  says it is fresh and covers the window asked about, and after one bounded
  provider read otherwise.

Every write and every read is scoped by the user *and* the connection ids the
caller names; the caller is expected to name only the live connections of
that user. The index is bounded per connection and pruned by age. Nothing
here contacts a network, and nothing here logs a subject, a sender, a title,
or a user id.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .oauth_providers import PROVIDERS
from .profile_crypto import DecryptionError, KeyRing
from .provider_connections import is_valid_connection_id
from .provider_data import CalendarEvent, InboxMessage
from .user_scope import is_valid_user_id
from .user_store import UserStore

logger = logging.getLogger(__name__)

__all__ = [
    "EVENT_RETENTION_SECONDS",
    "KINDS",
    "MAX_INDEXED_EVENTS",
    "MAX_INDEXED_MESSAGES",
    "MAX_QUERY_LIMIT",
    "MAX_SYNC_TTL_SECONDS",
    "MESSAGE_RETENTION_SECONDS",
    "SYNC_STATUSES",
    "IndexedEvent",
    "IndexedMessage",
    "KnowledgeStore",
    "SyncState",
    "event_bounds",
    "parse_instant",
]

KINDS: tuple[str, ...] = ("calendar", "inbox")
# What the ledger may say about the last read of one connection: the states
# the dashboard already shows, plus nothing else.
SYNC_STATUSES: tuple[str, ...] = (
    "ok",
    "reconnect_required",
    "insufficient_scope",
    "not_configured",
    "unsupported",
    "unavailable",
)
# One connection contributes at most this many items; the newest messages,
# the events nearest in time. Enough for any spoken answer.
MAX_INDEXED_MESSAGES = 50
MAX_INDEXED_EVENTS = 100
MAX_QUERY_LIMIT = 500
# A message older than this, or an event finished longer ago than this, is
# only clutter and is dropped on the next prune.
MESSAGE_RETENTION_SECONDS = 30 * 86400
EVENT_RETENTION_SECONDS = 2 * 86400
MAX_SYNC_TTL_SECONDS = 86400
_MAX_REASON_LENGTH = 64
_DEFAULT_EVENT_LENGTH_SECONDS = 3600
_DAY_SECONDS = 86400
_MAX_CONNECTIONS_PER_QUERY = 100
_MAX_PAYLOAD_CHARS = 4096


def _require_user(user_id: object) -> str:
    if not is_valid_user_id(user_id):
        raise ValueError("User id has an invalid shape")
    return user_id  # type: ignore[return-value]


def _require_connection(connection_id: object) -> str:
    if not is_valid_connection_id(connection_id):
        raise ValueError("Connection id has an invalid shape")
    return connection_id  # type: ignore[return-value]


def _require_connections(connection_ids: object) -> list[str]:
    if isinstance(connection_ids, str) or not isinstance(connection_ids, Iterable):
        raise ValueError("Connection ids must be a sequence")
    unique = list(dict.fromkeys(connection_ids))
    if len(unique) > _MAX_CONNECTIONS_PER_QUERY:
        raise ValueError("Too many connections in one query")
    return [_require_connection(item) for item in unique]


def _require_provider(provider: object) -> str:
    if not isinstance(provider, str) or provider not in PROVIDERS:
        raise ValueError("Unknown provider")
    return provider


def _require_kind(kind: object) -> str:
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValueError("Unknown knowledge kind")
    return kind


def _require_limit(limit: object) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_QUERY_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_QUERY_LIMIT}")
    return limit


def _require_ts(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative whole number of seconds")
    return value


def _require_ttl(ttl_seconds: object) -> int:
    if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int):
        raise ValueError("ttl_seconds must be a whole number of seconds")
    if not 0 < ttl_seconds <= MAX_SYNC_TTL_SECONDS:
        raise ValueError("ttl_seconds is out of bounds")
    return ttl_seconds


def _reason_text(reason: object) -> str | None:
    if reason is None:
        return None
    if (
        not isinstance(reason, str)
        or not reason
        or len(reason) > _MAX_REASON_LENGTH
        or not reason.isprintable()
        or any(ch.isspace() for ch in reason)
    ):
        raise ValueError("A sync reason must be a short token")
    return reason


# --- time --------------------------------------------------------------------------------


def parse_instant(value: object) -> int | None:
    """A UTC ISO 8601 instant, or a plain date taken as midnight UTC, as epoch seconds."""
    if not isinstance(value, str) or not 10 <= len(value) <= 40:
        return None
    text = value.strip()
    if len(text) == 10:
        try:
            day = datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
        return int(day.timestamp())
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return int(moment.timestamp())


def event_bounds(event: CalendarEvent) -> tuple[int, int] | None:
    """When an event occupies time, as [start, end) epoch seconds; None if unreadable.

    An all-day event runs from midnight UTC of its first day to midnight UTC
    after its last day (its end date is exclusive, as every provider states
    it). A timed event without an end is taken to last an hour.
    """
    start = parse_instant(event.start)
    if start is None:
        return None
    end = parse_instant(event.end) if event.end is not None else None
    if event.all_day:
        if end is None or end <= start:
            end = start + _DAY_SECONDS
    elif end is None or end < start:
        end = start + _DEFAULT_EVENT_LENGTH_SECONDS
    return start, end


def _start_of_day_utc(ts: int) -> int:
    moment = datetime.fromtimestamp(ts, tz=timezone.utc)
    return int(moment.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())


def _days_after(ts: int, days: int) -> int:
    return int((datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(days=days)).timestamp())


# --- values --------------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexedMessage:
    """One indexed message and the connection it belongs to."""

    connection_id: str
    provider: str
    item: InboxMessage
    indexed_at: int


@dataclass(frozen=True)
class IndexedEvent:
    """One indexed event, the connection it belongs to, and when it occupies time."""

    connection_id: str
    provider: str
    item: CalendarEvent
    indexed_at: int
    start_ts: int = 0
    end_ts: int = 0


@dataclass(frozen=True)
class SyncState:
    """How the last read of one connection for one kind of data went."""

    connection_id: str
    kind: str
    status: str
    reason: str | None
    synced_at: int
    expires_at: int
    item_count: int = 0
    window_start_ts: int | None = None
    window_end_ts: int | None = None
    # When this connection was last read successfully, kept across failures.
    last_ok_at: int | None = None

    def is_fresh(self, now: int) -> bool:
        return int(now) < self.expires_at

    def covers(self, start_ts: int, end_ts: int) -> bool:
        """Whether the last read asked the provider about this whole window."""
        if self.window_start_ts is None or self.window_end_ts is None:
            return False
        return self.window_start_ts <= start_ts and end_ts <= self.window_end_ts


# --- store -------------------------------------------------------------------------------


class KnowledgeStore:
    """Bounded, encrypted, per-user index of connected-account mail and events."""

    def __init__(self, users: UserStore, *, keyring: KeyRing) -> None:
        if not isinstance(keyring, KeyRing):
            raise ValueError("A profile key ring is required to index connected-account data")
        self._users = users
        self._keyring = keyring

    def _connect(self) -> sqlite3.Connection:
        return self._users.connect()

    # --- encryption ----------------------------------------------------------------

    @staticmethod
    def _aad(kind: str, user_id: str, connection_id: str, item_id: str) -> str:
        return f"caal.knowledge.{kind}:{user_id}:{connection_id}:{item_id}"

    def _seal(self, kind: str, user_id: str, connection_id: str, item_id: str, view: dict) -> str:
        body = dict(view)
        body.pop("id", None)
        rendered = json.dumps(body, separators=(",", ":"), sort_keys=True)
        if len(rendered) > _MAX_PAYLOAD_CHARS:
            raise ValueError("An indexed item is larger than the bounded fields allow")
        return self._keyring.encrypt(rendered, aad=self._aad(kind, user_id, connection_id, item_id))

    def _open(self, kind: str, row: sqlite3.Row) -> dict[str, Any] | None:
        aad = self._aad(kind, row["user_id"], row["connection_id"], row["item_id"])
        try:
            loaded = json.loads(self._keyring.decrypt(row["payload_enc"], aad=aad))
        except (DecryptionError, ValueError):
            logger.warning("An indexed %s row could not be decrypted and was skipped", kind)
            return None
        return loaded if isinstance(loaded, dict) else None

    # --- writes ---------------------------------------------------------------------

    def upsert_messages(
        self,
        user_id: object,
        connection_id: object,
        provider: object,
        messages: Sequence[InboxMessage],
        *,
        now: int,
    ) -> int:
        """Merge the newest messages of one connection; keep at most the cap, newest first."""
        owner = _require_user(user_id)
        connection_id = _require_connection(connection_id)
        name = _require_provider(provider)
        moment = int(now)
        rows: list[tuple[Any, ...]] = []
        for message in messages:
            if not isinstance(message, InboxMessage):
                raise ValueError("Only inbox messages can be indexed here")
            received = parse_instant(message.received_at)
            if received is None:
                continue
            sealed = self._seal("inbox", owner, connection_id, message.id, message.view())
            rows.append(
                (
                    owner,
                    connection_id,
                    name,
                    message.id,
                    received,
                    int(message.unread),
                    sealed,
                    moment,
                )
            )
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.executemany(
                    "INSERT INTO knowledge_messages (user_id, connection_id, provider, item_id, "
                    "received_ts, unread, payload_enc, indexed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(connection_id, item_id) DO UPDATE SET user_id = excluded.user_id, "
                    "provider = excluded.provider, received_ts = excluded.received_ts, "
                    "unread = excluded.unread, payload_enc = excluded.payload_enc, "
                    "indexed_at = excluded.indexed_at",
                    rows,
                )
                connection.execute(
                    "DELETE FROM knowledge_messages WHERE user_id = ? AND connection_id = ? "
                    "AND rowid NOT IN (SELECT rowid FROM knowledge_messages "
                    "WHERE user_id = ? AND connection_id = ? "
                    "ORDER BY received_ts DESC, item_id LIMIT ?)",
                    (owner, connection_id, owner, connection_id, MAX_INDEXED_MESSAGES),
                )
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        return len(rows)

    def replace_events(
        self,
        user_id: object,
        connection_id: object,
        provider: object,
        events: Sequence[CalendarEvent],
        *,
        window: tuple[int, int],
        now: int,
    ) -> int:
        """Make the index of one connection match what the provider listed for a window.

        Every indexed event that starts inside the window is dropped first, so
        an event the provider no longer lists (cancelled, moved) disappears
        rather than lingering; events outside the window are untouched.
        """
        owner = _require_user(user_id)
        connection_id = _require_connection(connection_id)
        name = _require_provider(provider)
        if not isinstance(window, tuple) or len(window) != 2:
            raise ValueError("A window is a (start, end) pair of epoch seconds")
        window_start = _require_ts(window[0], name="window start")
        window_end = _require_ts(window[1], name="window end")
        if window_end <= window_start:
            raise ValueError("The window must be positive")
        moment = int(now)
        rows: list[tuple[Any, ...]] = []
        for event in events:
            if not isinstance(event, CalendarEvent):
                raise ValueError("Only calendar events can be indexed here")
            bounds = event_bounds(event)
            if bounds is None:
                continue
            sealed = self._seal("calendar", owner, connection_id, event.id, event.view())
            rows.append(
                (
                    owner,
                    connection_id,
                    name,
                    event.id,
                    bounds[0],
                    bounds[1],
                    int(event.all_day),
                    sealed,
                    moment,
                )
            )
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "DELETE FROM knowledge_events WHERE user_id = ? AND connection_id = ? "
                    "AND start_ts >= ? AND start_ts < ?",
                    (owner, connection_id, window_start, window_end),
                )
                connection.executemany(
                    "INSERT INTO knowledge_events (user_id, connection_id, provider, item_id, "
                    "start_ts, end_ts, all_day, payload_enc, indexed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(connection_id, item_id) DO UPDATE SET user_id = excluded.user_id, "
                    "provider = excluded.provider, start_ts = excluded.start_ts, "
                    "end_ts = excluded.end_ts, all_day = excluded.all_day, "
                    "payload_enc = excluded.payload_enc, indexed_at = excluded.indexed_at",
                    rows,
                )
                # Keep the events nearest in time; anything past the cap is dropped.
                connection.execute(
                    "DELETE FROM knowledge_events WHERE user_id = ? AND connection_id = ? "
                    "AND rowid NOT IN (SELECT rowid FROM knowledge_events "
                    "WHERE user_id = ? AND connection_id = ? "
                    "ORDER BY start_ts, item_id LIMIT ?)",
                    (owner, connection_id, owner, connection_id, MAX_INDEXED_EVENTS),
                )
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        return len(rows)

    # --- reads -----------------------------------------------------------------------

    def messages(
        self,
        user_id: object,
        connection_ids: Sequence[str],
        *,
        limit: int,
        unread_only: bool = False,
        since_ts: int | None = None,
    ) -> list[IndexedMessage]:
        """The newest indexed messages of these connections of this user, newest first."""
        owner = _require_user(user_id)
        ids = _require_connections(connection_ids)
        count = _require_limit(limit)
        if not ids:
            return []
        clauses = ["user_id = ?", f"connection_id IN ({', '.join('?' for _ in ids)})"]
        params: list[Any] = [owner, *ids]
        if unread_only:
            clauses.append("unread = 1")
        if since_ts is not None:
            clauses.append("received_ts >= ?")
            params.append(_require_ts(since_ts, name="since_ts"))
        params.append(count)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM knowledge_messages WHERE "
                + " AND ".join(clauses)
                + " ORDER BY received_ts DESC, item_id LIMIT ?",
                params,
            ).fetchall()
        found = [self._message_of(row) for row in rows]
        return [item for item in found if item is not None]

    def find_message(
        self, user_id: object, connection_ids: Sequence[str], item_id: object
    ) -> IndexedMessage | None:
        """One indexed message by its provider id, if one of these connections holds it."""
        owner = _require_user(user_id)
        ids = _require_connections(connection_ids)
        if not ids or not isinstance(item_id, str) or not 0 < len(item_id) <= 512:
            return None
        if not item_id.isprintable() or any(ch.isspace() for ch in item_id):
            return None
        placeholders = ", ".join("?" for _ in ids)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_messages WHERE user_id = ? AND item_id = ? "
                f"AND connection_id IN ({placeholders}) LIMIT 1",
                (owner, item_id, *ids),
            ).fetchone()
        if row is None:
            return None
        return self._message_of(row)

    def _message_of(self, row: sqlite3.Row) -> IndexedMessage | None:
        view = self._open("inbox", row)
        if view is None:
            return None
        try:
            item = InboxMessage(id=row["item_id"], **view)
        except TypeError:
            return None
        return IndexedMessage(
            connection_id=row["connection_id"],
            provider=row["provider"],
            item=item,
            indexed_at=row["indexed_at"],
        )

    def events(
        self,
        user_id: object,
        connection_ids: Sequence[str],
        *,
        start_ts: int,
        end_ts: int,
        limit: int,
    ) -> list[IndexedEvent]:
        """Indexed events of these connections that overlap [start_ts, end_ts), soonest first."""
        owner = _require_user(user_id)
        ids = _require_connections(connection_ids)
        count = _require_limit(limit)
        begin = _require_ts(start_ts, name="start_ts")
        finish = _require_ts(end_ts, name="end_ts")
        if not ids or finish <= begin:
            return []
        placeholders = ", ".join("?" for _ in ids)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM knowledge_events WHERE user_id = ? "
                f"AND connection_id IN ({placeholders}) AND start_ts < ? AND end_ts > ? "
                "ORDER BY start_ts, item_id LIMIT ?",
                (owner, *ids, finish, begin, count),
            ).fetchall()
        found: list[IndexedEvent] = []
        for row in rows:
            view = self._open("calendar", row)
            if view is None:
                continue
            try:
                item = CalendarEvent(id=row["item_id"], **view)
            except TypeError:
                continue
            found.append(
                IndexedEvent(
                    connection_id=row["connection_id"],
                    provider=row["provider"],
                    item=item,
                    indexed_at=row["indexed_at"],
                    start_ts=row["start_ts"],
                    end_ts=row["end_ts"],
                )
            )
        return found

    # --- freshness ledger ----------------------------------------------------------

    def record_sync(
        self,
        user_id: object,
        connection_id: object,
        kind: object,
        *,
        status: str,
        reason: object = None,
        item_count: int = 0,
        window: tuple[int, int] | None = None,
        now: int,
        ttl_seconds: int,
    ) -> SyncState:
        """Note how the last read of one connection went and how long it stays fresh."""
        owner = _require_user(user_id)
        connection_id = _require_connection(connection_id)
        name = _require_kind(kind)
        if status not in SYNC_STATUSES:
            raise ValueError("Unknown sync status")
        why = _reason_text(reason)
        count = int(item_count)
        if count < 0:
            raise ValueError("item_count cannot be negative")
        ttl = _require_ttl(ttl_seconds)
        moment = int(now)
        window_start = window_end = None
        if window is not None:
            window_start = _require_ts(window[0], name="window start")
            window_end = _require_ts(window[1], name="window end")
        state = SyncState(
            connection_id=connection_id,
            kind=name,
            status=status,
            reason=why,
            synced_at=moment,
            expires_at=moment + ttl,
            item_count=count,
            window_start_ts=window_start,
            window_end_ts=window_end,
            last_ok_at=moment if status == "ok" else None,
        )
        with closing(self._connect()) as connection:
            connection.execute(
                "INSERT INTO knowledge_sync (connection_id, kind, user_id, status, reason, "
                "synced_at, expires_at, item_count, window_start_ts, window_end_ts, last_ok_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(connection_id, kind) DO UPDATE SET user_id = excluded.user_id, "
                "status = excluded.status, reason = excluded.reason, "
                "synced_at = excluded.synced_at, expires_at = excluded.expires_at, "
                "item_count = excluded.item_count, window_start_ts = excluded.window_start_ts, "
                "window_end_ts = excluded.window_end_ts, "
                "last_ok_at = COALESCE(excluded.last_ok_at, knowledge_sync.last_ok_at)",
                (
                    connection_id,
                    name,
                    owner,
                    status,
                    why,
                    moment,
                    state.expires_at,
                    count,
                    window_start,
                    window_end,
                    state.last_ok_at,
                ),
            )
            row = connection.execute(
                "SELECT last_ok_at FROM knowledge_sync WHERE connection_id = ? AND kind = ?",
                (connection_id, name),
            ).fetchone()
        if row is not None and row["last_ok_at"] != state.last_ok_at:
            state = SyncState(**dict(state.__dict__, last_ok_at=row["last_ok_at"]))
        return state

    def sync_states(
        self, user_id: object, connection_ids: Sequence[str], kind: object
    ) -> dict[str, SyncState]:
        """The ledger entries of these connections of this user for one kind, by connection."""
        owner = _require_user(user_id)
        ids = _require_connections(connection_ids)
        name = _require_kind(kind)
        if not ids:
            return {}
        placeholders = ", ".join("?" for _ in ids)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM knowledge_sync WHERE user_id = ? AND kind = ? "
                f"AND connection_id IN ({placeholders})",
                (owner, name, *ids),
            ).fetchall()
        states: dict[str, SyncState] = {}
        for row in rows:
            states[row["connection_id"]] = SyncState(
                connection_id=row["connection_id"],
                kind=row["kind"],
                status=row["status"],
                reason=row["reason"],
                synced_at=row["synced_at"],
                expires_at=row["expires_at"],
                item_count=row["item_count"],
                window_start_ts=row["window_start_ts"],
                window_end_ts=row["window_end_ts"],
                last_ok_at=row["last_ok_at"],
            )
        return states

    # --- housekeeping ----------------------------------------------------------------

    def forget_connection(self, user_id: object, connection_id: object) -> None:
        """Erase everything indexed for one connection, if this user owns the rows."""
        owner = _require_user(user_id)
        connection_id = _require_connection(connection_id)
        with closing(self._connect()) as connection:
            for table in ("knowledge_messages", "knowledge_events", "knowledge_sync"):
                connection.execute(
                    f"DELETE FROM {table} WHERE user_id = ? AND connection_id = ?",
                    (owner, connection_id),
                )

    def forget_user(self, user_id: object) -> None:
        """Erase everything this module holds for one user."""
        owner = _require_user(user_id)
        with closing(self._connect()) as connection:
            for table in ("knowledge_messages", "knowledge_events", "knowledge_sync"):
                connection.execute(f"DELETE FROM {table} WHERE user_id = ?", (owner,))

    def prune(self, *, now: int) -> None:
        """Drop messages past retention and events finished long enough ago."""
        moment = int(now)
        with closing(self._connect()) as connection:
            connection.execute(
                "DELETE FROM knowledge_messages WHERE received_ts < ?",
                (moment - MESSAGE_RETENTION_SECONDS,),
            )
            connection.execute(
                "DELETE FROM knowledge_events WHERE end_ts < ?",
                (moment - EVENT_RETENTION_SECONDS,),
            )
