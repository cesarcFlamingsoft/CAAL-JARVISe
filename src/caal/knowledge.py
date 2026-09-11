"""What JARVIS knows about the connected mail and calendars of a user, and how fresh it is.

The dashboard feeds and the voice tools both go through here. Every answer
is built from the per-user index in :mod:caal.knowledge_store; the only
question is whether the index is refreshed first:

* a **dashboard read** (:meth:KnowledgeService.calendar_feed and
  :meth:KnowledgeService.inbox_feed) always reads the providers, exactly as
  the dashboard did before, and writes what came back into the index, so a
  visible dashboard keeps the index warm for free;
* a **question** (:meth:KnowledgeService.upcoming_events and
  :meth:KnowledgeService.recent_messages) is answered from the index while
  the freshness ledger says every live connection was read within its TTL
  and, for events, that the read covered the window asked about. Otherwise
  the connections that need it are read once -- concurrent askers of the
  same user share that one read -- within a wall-clock budget, and the answer
  is built from the index afterwards. A read that fails or runs out of time
  leaves what was indexed in place and marks the account *stale*, so the
  answer can say "as of" rather than guess or go silent.

Only live connections of the named user are ever consulted, so a revoked
connection cannot surface even if its rows were not yet erased. Nothing here
logs a subject, a title, a sender, a label, or a user id.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol
from zoneinfo import ZoneInfo

from .knowledge_store import (
    MAX_INDEXED_EVENTS,
    MAX_INDEXED_MESSAGES,
    IndexedEvent,
    IndexedMessage,
    KnowledgeStore,
    SyncState,
)
from .provider_connections import ConnectionStore, ProviderConnection, label_key
from .provider_data import MAX_ITEMS, AccountFetch, CalendarEvent, InboxMessage
from .user_scope import is_valid_user_id

logger = logging.getLogger(__name__)

__all__ = [
    "CALENDAR_TTL_SECONDS",
    "FAILURE_TTL_SECONDS",
    "INBOX_TTL_SECONDS",
    "MAX_ANSWER_ITEMS",
    "MAX_QUERY_LENGTH",
    "MAX_REFRESH_DAYS",
    "REFRESH_BUDGET_SECONDS",
    "REFRESH_WINDOW_DAYS",
    "AccountAnswer",
    "AccountSelection",
    "DataReader",
    "EventsAnswer",
    "KnowledgeService",
    "MessagesAnswer",
    "matches",
    "normalize_text",
    "resolve_account",
]

# How long one successful read of a connection answers questions before the
# provider is asked again. A failure is retried sooner.
CALENDAR_TTL_SECONDS = 300
INBOX_TTL_SECONDS = 180
FAILURE_TTL_SECONDS = 60
_MAX_TTL_SECONDS = 3600
# The wall clock a question may spend refreshing before it answers from what
# is indexed. Well under a spoken pause; the provider client has its own
# per-request timeouts inside this.
REFRESH_BUDGET_SECONDS = 8.0
_MAX_REFRESH_BUDGET_SECONDS = 60.0
# A calendar refresh always reads from the start of today to at least this
# far ahead, so the next question about this week is answered from the index.
REFRESH_WINDOW_DAYS = 14
# The farthest a single refresh looks; within the provider limit of 62 days.
MAX_REFRESH_DAYS = 45
MAX_QUERY_LENGTH = 200
MAX_ANSWER_ITEMS = 25
_DEFAULT_TIMEZONE = "America/Los_Angeles"
_TIMEOUT_REASON = "refresh_timeout"

# Function words a spoken question carries that are not worth matching on.
_STOPWORDS = frozenset(
    (
        "a",
        "an",
        "the",
        "my",
        "me",
        "i",
        "we",
        "our",
        "you",
        "your",
        "with",
        "for",
        "of",
        "to",
        "and",
        "or",
        "on",
        "at",
        "in",
        "from",
        "about",
        "regarding",
        "re",
        "is",
        "are",
        "was",
        "do",
        "does",
        "did",
        "any",
        "some",
        "have",
        "has",
        "had",
        "there",
        "that",
        "this",
        "it",
        "email",
        "emails",
        "mail",
        "mails",
        "message",
        "messages",
    )
)
_NON_WORD = re.compile(r"[^0-9a-z]+")


# --- text ---------------------------------------------------------------------------------


def normalize_text(value: object) -> str:
    """Casefolded, accent-stripped words separated by single spaces; empty for nothing."""
    if not isinstance(value, str) or not value:
        return ""
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(_NON_WORD.sub(" ", stripped.casefold()).split())


def query_terms(query: object) -> list[str]:
    """The words of a question worth matching, bounded in length."""
    text = query[:MAX_QUERY_LENGTH] if isinstance(query, str) else ""
    return [term for term in normalize_text(text).split() if term not in _STOPWORDS]


def matches(query: object, *fields: object) -> bool:
    """Whether every meaningful word of the query appears in the joined fields.

    Matching is by substring on normalized text, so "bo" finds "Bo Example"
    and "dentist" finds "Dentist appointment", and a question with no
    meaningful words matches everything. Nothing fuzzier than that: an
    answer is never a guess.
    """
    terms = query_terms(query)
    if not terms:
        return True
    haystack = " " + " ".join(normalize_text(field) for field in fields if field) + " "
    return all(term in haystack for term in terms)


# --- naming one connected account --------------------------------------------------------


@dataclass(frozen=True)
class AccountSelection:
    """Which of a user's live connections a question named, if it named one.

    ``connections`` empty with ``ambiguous`` false means the name matched
    nothing: the answer says so, and never widens back out to every account.
    ``ambiguous`` means the name fits several accounts equally well, which is
    also not an answer -- the user is asked which one they meant.
    """

    connections: list[ProviderConnection]
    ambiguous: bool = False
    hint: str = ""


def _user_keys(connection: ProviderConnection) -> list[str]:
    """The normalized names the *owner* gave this connection."""
    names = [connection.user_label, *connection.aliases]
    return [key for key in (label_key(name) for name in names) if key]


def _resolved(matched: list[ProviderConnection], hint: str) -> AccountSelection:
    """One account is an answer; several equally good ones are a question back."""
    if len(matched) == 1:
        return AccountSelection(matched, hint=hint)
    return AccountSelection([], ambiguous=True, hint=hint)


def resolve_account(connections: Sequence[ProviderConnection], account: object) -> AccountSelection:
    """The live connections a question is about, by a deterministic precedence.

    In order, and never mixing two of them:

    1. no name, or nothing matchable in it -- every live connection;
    2. a provider ("google", "outlook" is normalized upstream) -- every
       account linked at that provider, which is what "my Google calendar"
       asks for even with two Google accounts;
    3. a name the owner gave an account, matched whole ("work", "university",
       "wife");
    4. the provider's own label for the account, matched whole -- normally
       the address it was linked with;
    5. only then, that name occurring inside one of those names, so
       "vertex" still finds ``ana@vertex.example``.

    Every tier past the provider one resolves to exactly one account or to an
    ambiguity; nothing here guesses between two accounts, and nothing here
    reads a connection of another user -- it is given the caller's own live
    connections and matches within them.
    """
    live = list(connections)
    if account is None:
        return AccountSelection(live)
    hint = label_key(account)
    if not hint:
        return AccountSelection(live)
    by_provider = [item for item in live if hint == item.provider]
    if by_provider:
        return AccountSelection(by_provider, hint=hint)
    named = [item for item in live if hint in _user_keys(item)]
    if named:
        return _resolved(named, hint)
    addressed = [item for item in live if hint == label_key(item.account_label)]
    if addressed:
        return _resolved(addressed, hint)
    partial = [
        item
        for item in live
        if any(hint in key for key in (*_user_keys(item), label_key(item.account_label)))
    ]
    return _resolved(partial, hint) if partial else AccountSelection([], hint=hint)


class DataReader(Protocol):
    """What the service needs from :class:caal.provider_data.ProviderDataClient."""

    async def calendar_feed(
        self,
        user_id: str,
        connections: Sequence[ProviderConnection],
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[AccountFetch]: ...

    async def inbox_feed(
        self, user_id: str, connections: Sequence[ProviderConnection], *, limit: int
    ) -> list[AccountFetch]: ...


# --- answers -------------------------------------------------------------------------------


@dataclass(frozen=True)
class AccountAnswer:
    """What one live connection contributed to an answer, and how current that is."""

    connection_id: str
    provider: str
    account_label: str | None
    user_label: str | None
    status: str
    reason: str | None
    stale: bool
    count: int
    synced_at: int | None = None
    indexed_at: int | None = None


@dataclass(frozen=True)
class EventsAnswer:
    events: list[IndexedEvent]
    accounts: list[AccountAnswer]
    window_start_ts: int
    window_end_ts: int
    generated_at: int
    connected: bool
    # The name the question used fits more than one connected account, so
    # nothing was read: the caller must ask the user which one they meant.
    ambiguous: bool = False

    @property
    def stale(self) -> bool:
        return any(account.stale for account in self.accounts)


@dataclass(frozen=True)
class MessagesAnswer:
    messages: list[IndexedMessage]
    accounts: list[AccountAnswer]
    generated_at: int
    connected: bool
    ambiguous: bool = False

    @property
    def unread_count(self) -> int:
        return sum(1 for message in self.messages if message.item.unread)

    @property
    def stale(self) -> bool:
        return any(account.stale for account in self.accounts)


# --- helpers -------------------------------------------------------------------------------


def _require_user(user_id: object) -> str:
    if not is_valid_user_id(user_id):
        raise ValueError("User id has an invalid shape")
    return user_id  # type: ignore[return-value]


def _require_limit(limit: object) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_ANSWER_ITEMS:
        raise ValueError(f"limit must be between 1 and {MAX_ANSWER_ITEMS}")
    return limit


def _require_aware(moment: object, *, name: str) -> datetime:
    if not isinstance(moment, datetime) or moment.tzinfo is None:
        raise ValueError(f"{name} must be an aware datetime")
    return moment


def _ts(moment: datetime) -> int:
    return int(moment.timestamp())


def _at(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _zone(name: object) -> ZoneInfo:
    candidates = [name] if isinstance(name, str) and name else []
    candidates.extend([os.getenv("TIMEZONE") or "", _DEFAULT_TIMEZONE])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ZoneInfo(candidate)
        except (KeyError, ValueError, OSError):
            logger.warning("Ignoring an unusable timezone name for spoken times")
    return ZoneInfo("UTC")


# --- the service ---------------------------------------------------------------------------


class KnowledgeService:
    """Answers about the connected mail and calendars of one user, from a fresh-enough index."""

    def __init__(
        self,
        connections: ConnectionStore,
        data: DataReader,
        index: KnowledgeStore,
        *,
        clock: Callable[[], float] = time.time,
        calendar_ttl_seconds: int = CALENDAR_TTL_SECONDS,
        inbox_ttl_seconds: int = INBOX_TTL_SECONDS,
        refresh_budget_seconds: float = REFRESH_BUDGET_SECONDS,
        timezone: str | None = None,
    ) -> None:
        for name, value in (
            ("calendar_ttl_seconds", calendar_ttl_seconds),
            ("inbox_ttl_seconds", inbox_ttl_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 < value <= _MAX_TTL_SECONDS
            ):
                raise ValueError(f"{name} must be within (0, {_MAX_TTL_SECONDS}] seconds")
        if isinstance(refresh_budget_seconds, bool) or not isinstance(
            refresh_budget_seconds, (int, float)
        ):
            raise ValueError("refresh_budget_seconds must be a number of seconds")
        if not 0 < float(refresh_budget_seconds) <= _MAX_REFRESH_BUDGET_SECONDS:
            raise ValueError(
                f"refresh_budget_seconds must be within (0, {_MAX_REFRESH_BUDGET_SECONDS}]"
            )
        self._connections = connections
        self._data = data
        self._index = index
        self._clock = clock
        self._calendar_ttl = int(calendar_ttl_seconds)
        self._inbox_ttl = int(inbox_ttl_seconds)
        self._budget = float(refresh_budget_seconds)
        self._zone = _zone(timezone)
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

    @property
    def zone(self) -> ZoneInfo:
        """The zone spoken times are given in."""
        return self._zone

    @property
    def index(self) -> KnowledgeStore:
        return self._index

    def now(self) -> int:
        return int(self._clock())

    def _lock_for(self, user_id: str, kind: str) -> asyncio.Lock:
        key = (user_id, kind)
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    def start_of_day(self, ts: int, *, days_ahead: int = 0) -> int:
        """Midnight, in the spoken zone, of the day holding ts plus days_ahead days."""
        local = _at(ts).astimezone(self._zone)
        day = (local + timedelta(days=days_ahead)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return _ts(day)

    # --- write-through for the dashboard ---------------------------------------------

    async def calendar_feed(
        self,
        user_id: object,
        connections: Sequence[ProviderConnection],
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[AccountFetch]:
        """Read every connection like the dashboard always did, and index what came back."""
        owner = _require_user(user_id)
        begin, finish = _require_aware(start, name="start"), _require_aware(end, name="end")
        fetches = await self._data.calendar_feed(
            owner, connections, start=begin, end=finish, limit=limit
        )
        self._index_calendar(owner, fetches, window=(_ts(begin), _ts(finish)))
        return fetches

    async def inbox_feed(
        self, user_id: object, connections: Sequence[ProviderConnection], *, limit: int
    ) -> list[AccountFetch]:
        owner = _require_user(user_id)
        fetches = await self._data.inbox_feed(owner, connections, limit=limit)
        self._index_inbox(owner, fetches)
        return fetches

    def _index_calendar(
        self, owner: str, fetches: Iterable[AccountFetch], *, window: tuple[int, int]
    ) -> None:
        now = self.now()
        for fetch in fetches:
            connection = fetch.connection
            if connection.user_id != owner:
                logger.error("Refusing to index a calendar read for another user")
                continue
            count = 0
            if fetch.error is None:
                count = self._index.replace_events(
                    owner,
                    connection.connection_id,
                    connection.provider,
                    fetch.events,
                    window=window,
                    now=now,
                )
            self._index.record_sync(
                owner,
                connection.connection_id,
                "calendar",
                status=fetch.status,
                reason=fetch.reason,
                item_count=count,
                window=window if fetch.error is None else None,
                now=now,
                ttl_seconds=self._calendar_ttl if fetch.error is None else FAILURE_TTL_SECONDS,
            )
        self._index.prune(now=now)

    def _index_inbox(self, owner: str, fetches: Iterable[AccountFetch]) -> None:
        now = self.now()
        for fetch in fetches:
            connection = fetch.connection
            if connection.user_id != owner:
                logger.error("Refusing to index an inbox read for another user")
                continue
            count = 0
            if fetch.error is None:
                count = self._index.upsert_messages(
                    owner, connection.connection_id, connection.provider, fetch.messages, now=now
                )
            self._index.record_sync(
                owner,
                connection.connection_id,
                "inbox",
                status=fetch.status,
                reason=fetch.reason,
                item_count=count,
                now=now,
                ttl_seconds=self._inbox_ttl if fetch.error is None else FAILURE_TTL_SECONDS,
            )
        self._index.prune(now=now)

    # --- what was last indexed for one account (the dashboard fallback) --------------

    def cached_events(
        self,
        user_id: object,
        connection: ProviderConnection,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> tuple[list[CalendarEvent], SyncState | None]:
        owner = _require_user(user_id)
        if connection.user_id != owner:
            return [], None
        ids = [connection.connection_id]
        rows = self._index.events(
            owner,
            ids,
            start_ts=_ts(_require_aware(start, name="start")),
            end_ts=_ts(_require_aware(end, name="end")),
            limit=max(1, min(int(limit), MAX_INDEXED_EVENTS)),
        )
        return [row.item for row in rows], self._index.sync_states(owner, ids, "calendar").get(
            ids[0]
        )

    def cached_messages(
        self, user_id: object, connection: ProviderConnection, *, limit: int
    ) -> tuple[list[InboxMessage], SyncState | None]:
        owner = _require_user(user_id)
        if connection.user_id != owner:
            return [], None
        ids = [connection.connection_id]
        rows = self._index.messages(owner, ids, limit=max(1, min(int(limit), MAX_INDEXED_MESSAGES)))
        return [row.item for row in rows], self._index.sync_states(owner, ids, "inbox").get(ids[0])

    def forget_connection(self, user_id: object, connection_id: object) -> None:
        """Erase what was indexed for a connection the user revoked."""
        self._index.forget_connection(user_id, connection_id)

    def forget_user(self, user_id: object) -> None:
        self._index.forget_user(user_id)

    # --- questions -------------------------------------------------------------------

    async def upcoming_events(
        self,
        user_id: object,
        *,
        start: datetime,
        end: datetime,
        limit: int,
        query: object = None,
        account: object = None,
    ) -> EventsAnswer:
        """Events overlapping [start, end) on the live calendars of this user, soonest first."""
        owner = _require_user(user_id)
        count = _require_limit(limit)
        begin = _ts(_require_aware(start, name="start"))
        finish = _ts(_require_aware(end, name="end"))
        if finish <= begin:
            raise ValueError("The window must be positive")
        live = self._connections.list_connections(owner)
        selection = resolve_account(live, account)
        if selection.ambiguous:
            return EventsAnswer(
                events=[],
                accounts=[],
                window_start_ts=begin,
                window_end_ts=finish,
                generated_at=self.now(),
                connected=True,
                ambiguous=True,
            )
        chosen = selection.connections
        states = await self._ensure_fresh(owner, chosen, "calendar", window=(begin, finish))
        ids = [item.connection_id for item in chosen]
        rows = self._index.events(
            owner, ids, start_ts=begin, end_ts=finish, limit=max(1, len(ids)) * MAX_INDEXED_EVENTS
        )
        if query is not None:
            rows = [row for row in rows if matches(query, row.item.title, row.item.location)]
        return EventsAnswer(
            events=rows[:count],
            accounts=self._accounts(chosen, states, rows),
            window_start_ts=begin,
            window_end_ts=finish,
            generated_at=self.now(),
            connected=bool(live),
        )

    async def recent_messages(
        self,
        user_id: object,
        *,
        limit: int,
        unread_only: bool = False,
        query: object = None,
        account: object = None,
        message_id: object = None,
    ) -> MessagesAnswer:
        """The newest indexed mail of this user, newest first, optionally narrowed."""
        owner = _require_user(user_id)
        count = _require_limit(limit)
        live = self._connections.list_connections(owner)
        selection = resolve_account(live, account)
        if selection.ambiguous:
            return MessagesAnswer(
                messages=[], accounts=[], generated_at=self.now(), connected=True, ambiguous=True
            )
        chosen = selection.connections
        states = await self._ensure_fresh(owner, chosen, "inbox", window=None)
        ids = [item.connection_id for item in chosen]
        if message_id is not None:
            found = self._index.find_message(owner, ids, message_id)
            rows = [found] if found is not None else []
        else:
            rows = self._index.messages(
                owner,
                ids,
                limit=max(1, len(ids)) * MAX_INDEXED_MESSAGES,
                unread_only=bool(unread_only),
            )
        if query is not None:
            rows = [
                row
                for row in rows
                if matches(query, row.item.subject, row.item.sender, row.item.preview)
            ]
        return MessagesAnswer(
            messages=rows[:count],
            accounts=self._accounts(chosen, states, rows),
            generated_at=self.now(),
            connected=bool(live),
        )

    @staticmethod
    def _accounts(
        connections: Sequence[ProviderConnection],
        states: dict[str, SyncState],
        rows: Sequence[IndexedEvent] | Sequence[IndexedMessage],
    ) -> list[AccountAnswer]:
        contributed: dict[str, tuple[int, int]] = {}
        for row in rows:
            count, latest = contributed.get(row.connection_id, (0, 0))
            contributed[row.connection_id] = (count + 1, max(latest, row.indexed_at))
        answers: list[AccountAnswer] = []
        for item in connections:
            state = states.get(item.connection_id)
            count, latest = contributed.get(item.connection_id, (0, 0))
            answers.append(
                AccountAnswer(
                    connection_id=item.connection_id,
                    provider=item.provider,
                    account_label=item.account_label,
                    user_label=item.user_label,
                    status=state.status if state is not None else "unavailable",
                    reason=state.reason if state is not None else _TIMEOUT_REASON,
                    stale=state is None or state.status != "ok",
                    count=count,
                    synced_at=state.synced_at if state is not None else None,
                    indexed_at=latest or None,
                )
            )
        return answers

    # --- freshness -------------------------------------------------------------------

    @staticmethod
    def _needs_refresh(
        state: SyncState | None, kind: str, now: int, window: tuple[int, int] | None
    ) -> bool:
        if state is None or not state.is_fresh(now):
            return True
        if kind == "calendar" and state.status == "ok" and window is not None:
            return not state.covers(*window)
        return False

    async def _ensure_fresh(
        self,
        owner: str,
        connections: Sequence[ProviderConnection],
        kind: str,
        *,
        window: tuple[int, int] | None,
    ) -> dict[str, SyncState]:
        """Read whichever of these connections the ledger says need it, once, within budget."""
        ids = [item.connection_id for item in connections]
        if not ids:
            return {}
        now = self.now()
        states = self._index.sync_states(owner, ids, kind)
        if not any(self._needs_refresh(states.get(i), kind, now, window) for i in ids):
            return states
        async with self._lock_for(owner, kind):
            now = self.now()
            states = self._index.sync_states(owner, ids, kind)
            needed = [
                item
                for item in connections
                if self._needs_refresh(states.get(item.connection_id), kind, now, window)
            ]
            if needed:
                await self._refresh(owner, needed, kind, window, now)
                states = self._index.sync_states(owner, ids, kind)
        return states

    async def _refresh(
        self,
        owner: str,
        connections: Sequence[ProviderConnection],
        kind: str,
        window: tuple[int, int] | None,
        now: int,
    ) -> None:
        try:
            if kind == "calendar":
                asked = window or (now, now)
                begin = min(asked[0], self.start_of_day(now))
                finish = max(asked[1], now + REFRESH_WINDOW_DAYS * 86400)
                finish = min(finish, begin + MAX_REFRESH_DAYS * 86400)
                fetches = await asyncio.wait_for(
                    self._data.calendar_feed(
                        owner, connections, start=_at(begin), end=_at(finish), limit=MAX_ITEMS
                    ),
                    timeout=self._budget,
                )
                self._index_calendar(owner, fetches, window=(begin, finish))
            else:
                fetches = await asyncio.wait_for(
                    self._data.inbox_feed(owner, connections, limit=MAX_ITEMS),
                    timeout=self._budget,
                )
                self._index_inbox(owner, fetches)
        except asyncio.TimeoutError:
            logger.warning(
                "Refreshing %s knowledge ran out of time for %d account(s)", kind, len(connections)
            )
            self._note_failure(owner, connections, kind, _TIMEOUT_REASON)
        except Exception as exc:  # noqa: BLE001 - an answer must never break on a refresh
            logger.error("Refreshing %s knowledge raised %s", kind, type(exc).__name__)
            self._note_failure(owner, connections, kind, "transport")
        else:
            answered = sum(1 for fetch in fetches if fetch.error is None)
            logger.info(
                "Refreshed %s knowledge: %d of %d accounts answered", kind, answered, len(fetches)
            )

    def _note_failure(
        self, owner: str, connections: Sequence[ProviderConnection], kind: str, reason: str
    ) -> None:
        moment = self.now()
        for item in connections:
            self._index.record_sync(
                owner,
                item.connection_id,
                kind,
                status="unavailable",
                reason=reason,
                now=moment,
                ttl_seconds=FAILURE_TTL_SECONDS,
            )
