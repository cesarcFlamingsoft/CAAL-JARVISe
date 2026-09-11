"""The dashboard feeds from a user's connected accounts: upcoming events and recent mail.

``GET /users/me/dashboard/calendar?days=7&limit=25``
    the next ``days`` days of events from every live connection of the
    caller, in start order and each carrying the connection it came from so a
    reader can group by account, plus one entry per account saying whether it
    answered and, if not, why: ``reconnect_required``, ``insufficient_scope``,
    ``not_configured``, ``unsupported`` or ``unavailable``. The window starts
    now, so an event in progress is included.
``GET /users/me/dashboard/inbox?limit=20``
    the newest inbox messages from every live connection, merged newest
    first, with the same per-account report and an unread count.
``GET /users/me/dashboard/reminders``
    the caller own local reminders, soonest due first and undated last, each
    with the channels it will be delivered on and the state of each one, plus
    the channels this owner is allowed to use at all.
``GET|PUT /users/me/dashboard/reminders/delivery``
    the channels *future* reminders of this owner will use. The minimal edit
    surface: it names no reminder, no number and no chat, and it never changes
    anything that already exists.

Both routes live under ``/users/me`` so they inherit the identity boundary of
:mod:`caal.user_api`: a single-use ``caal-backend`` principal from the BFF
names the user, the user is loaded from the database on every call, and the
identity middleware makes every response uncacheable and free of the
app-wide CORS policy. The reads themselves are :mod:`caal.provider_data`:
bounded, https-only, token-renewing, and reduced to plain summaries. Nothing
here returns or logs a token, a body, or an address beyond an account label.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from .connections_api import ConnectionsRuntime, require_connections
from .provider_data import MAX_ITEMS, AccountFetch, CalendarEvent, InboxMessage
from .tools import reminder_delivery, reminders_tools
from .tools.errors import SafeToolError
from .user_api import CurrentUser, require_user

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CALENDAR_LIMIT",
    "DEFAULT_INBOX_LIMIT",
    "DEFAULT_WINDOW_DAYS",
    "MAX_WINDOW_DAYS",
    "router",
]

DEFAULT_WINDOW_DAYS = 7
MAX_WINDOW_DAYS = 31
DEFAULT_CALENDAR_LIMIT = 25
DEFAULT_INBOX_LIMIT = 20


# --- schemas -----------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class DashboardAccount(_Strict):
    connection_id: str
    provider: str
    account_label: str | None
    status: str
    reason: str | None
    count: int


class CalendarEventResponse(_Strict):
    id: str
    connection_id: str
    provider: str
    title: str | None
    start: str
    end: str | None
    all_day: bool
    location: str | None
    link: str | None
    status: str | None


class CalendarFeedResponse(_Strict):
    generated_at: int
    window_start: str
    window_end: str
    accounts: list[DashboardAccount]
    events: list[CalendarEventResponse]


class InboxMessageResponse(_Strict):
    id: str
    connection_id: str
    provider: str
    subject: str | None
    sender: str | None
    preview: str | None
    received_at: str
    unread: bool
    link: str | None


class InboxFeedResponse(_Strict):
    generated_at: int
    accounts: list[DashboardAccount]
    messages: list[InboxMessageResponse]
    unread_count: int


# --- helpers --------------------------------------------------------------------------------


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _account(fetch: AccountFetch, status: str, count: int) -> DashboardAccount:
    return DashboardAccount(
        connection_id=fetch.connection.connection_id,
        provider=fetch.connection.provider,
        account_label=fetch.connection.account_label,
        status=status,
        reason=fetch.reason,
        count=count,
    )


def _events_of(
    runtime: ConnectionsRuntime,
    user_id: str,
    fetch: AccountFetch,
    start: datetime,
    end: datetime,
    limit: int,
) -> tuple[str, list[CalendarEvent]]:
    """What to show for one account: the live read, or the last indexed read marked stale.

    Only a transient failure (the provider did not answer) falls back, and only
    for an account that has been read successfully before. An account that
    needs reconnecting, lacks a scope, or was never read stays as it is: old
    data is never dressed up as an answer to a permanent problem.
    """
    if fetch.error is None:
        return fetch.status, list(fetch.events)
    if fetch.status == "unavailable":
        cached, state = runtime.knowledge.cached_events(
            user_id, fetch.connection, start=start, end=end, limit=limit
        )
        if state is not None and state.last_ok_at is not None:
            return "stale", cached
    return fetch.status, []


def _messages_of(
    runtime: ConnectionsRuntime, user_id: str, fetch: AccountFetch, limit: int
) -> tuple[str, list[InboxMessage]]:
    if fetch.error is None:
        return fetch.status, list(fetch.messages)
    if fetch.status == "unavailable":
        cached, state = runtime.knowledge.cached_messages(user_id, fetch.connection, limit=limit)
        if state is not None and state.last_ok_at is not None:
            return "stale", cached
    return fetch.status, []


def _note(what: str, results: list[AccountFetch]) -> None:
    answered = sum(1 for fetch in results if fetch.error is None)
    logger.info("Dashboard %s feed: %d of %d accounts answered", what, answered, len(results))


# --- routes ---------------------------------------------------------------------------------

router = APIRouter(tags=["dashboard"])


@router.get("/users/me/dashboard/calendar", response_model=CalendarFeedResponse)
async def calendar_feed(
    days: int = Query(default=DEFAULT_WINDOW_DAYS, ge=1, le=MAX_WINDOW_DAYS),
    limit: int = Query(default=DEFAULT_CALENDAR_LIMIT, ge=1, le=MAX_ITEMS),
    user: CurrentUser = Depends(require_user),
    runtime: ConnectionsRuntime = Depends(require_connections),
) -> CalendarFeedResponse:
    now = runtime.now()
    start = datetime.fromtimestamp(now, tz=timezone.utc)
    end = start + timedelta(days=days)
    user_id = user.profile.user_id
    connections = runtime.store.list_connections(user_id)
    results = await runtime.knowledge.calendar_feed(
        user_id, connections, start=start, end=end, limit=limit
    )
    _note("calendar", results)
    shown = [(fetch, *_events_of(runtime, user_id, fetch, start, end, limit)) for fetch in results]
    events = [
        CalendarEventResponse(
            connection_id=fetch.connection.connection_id,
            provider=fetch.connection.provider,
            **event.view(),
        )
        for fetch, _status, items in shown
        for event in items
    ]
    events.sort(key=lambda event: event.start)
    return CalendarFeedResponse(
        generated_at=now,
        window_start=_iso(start),
        window_end=_iso(end),
        accounts=[_account(fetch, status, len(items)) for fetch, status, items in shown],
        events=events[:limit],
    )


@router.get("/users/me/dashboard/inbox", response_model=InboxFeedResponse)
async def inbox_feed(
    limit: int = Query(default=DEFAULT_INBOX_LIMIT, ge=1, le=MAX_ITEMS),
    user: CurrentUser = Depends(require_user),
    runtime: ConnectionsRuntime = Depends(require_connections),
) -> InboxFeedResponse:
    user_id = user.profile.user_id
    connections = runtime.store.list_connections(user_id)
    results = await runtime.knowledge.inbox_feed(user_id, connections, limit=limit)
    _note("inbox", results)
    per_account = [(fetch, *_messages_of(runtime, user_id, fetch, limit)) for fetch in results]
    messages = [
        InboxMessageResponse(
            connection_id=fetch.connection.connection_id,
            provider=fetch.connection.provider,
            **message.view(),
        )
        for fetch, _status, items in per_account
        for message in items
    ]
    messages.sort(key=lambda message: message.received_at, reverse=True)
    shown = messages[:limit]
    return InboxFeedResponse(
        generated_at=runtime.now(),
        accounts=[_account(fetch, status, len(items)) for fetch, status, items in per_account],
        messages=shown,
        unread_count=sum(1 for message in shown if message.unread),
    )


# --- the caller own local reminders ------------------------------------------------------


class ReminderChannelState(_Strict):
    channel: str
    state: str


class ReminderItem(_Strict):
    id: str
    title: str
    due: str | None
    timed: bool
    list_name: str
    notes: str
    completed: bool
    created_at: str
    delivery: list[ReminderChannelState]


class RemindersFeedResponse(_Strict):
    generated_at: int
    reminders: list[ReminderItem]
    available: list[str]
    defaults: list[str]


class DeliveryDefaults(_Strict):
    delivery: list[str]
    available: list[str]
    saved: bool


class DeliveryChoice(_Strict):
    """What the owner may send. There is no owner field: the caller is the owner."""

    delivery: list[str]


def _due(raw: str | None) -> str | None:
    """A stored UTC instant, as the one instant format the browser parses."""
    if not raw:
        return None
    try:
        return _iso(datetime.fromisoformat(raw))
    except ValueError:
        return None


def _available(user_id: str) -> list[str]:
    availability = reminder_delivery.channel_availability(user_id)
    return [channel for channel in reminder_delivery.CHANNELS if availability.get(channel)]


@router.get("/users/me/dashboard/reminders", response_model=RemindersFeedResponse)
async def reminders_feed(user: CurrentUser = Depends(require_user)) -> RemindersFeedResponse:
    """The caller own reminders, and nobody else own, ever.

    A read: it creates nothing, arms nothing and settles nothing. The scope is
    the verified caller, so there is no parameter that could widen it.
    """
    user_id = user.profile.user_id
    rows = reminders_tools.dashboard_reminders(user_id=user_id)
    logger.info("Dashboard reminders feed: %d reminder(s)", len(rows))
    return RemindersFeedResponse(
        generated_at=int(time.time()),
        reminders=[
            ReminderItem(
                id=row["id"],
                title=row["title"],
                due=_due(row["due"]),
                timed=bool(row["timed"]),
                list_name=row["list"],
                notes=row["notes"],
                completed=bool(row["completed"]),
                created_at=row["created_at"],
                delivery=[ReminderChannelState(**channel) for channel in row["delivery"]],
            )
            for row in rows
        ],
        available=_available(user_id),
        defaults=list(reminder_delivery.default_channels(user_id)),
    )


@router.get("/users/me/dashboard/reminders/delivery", response_model=DeliveryDefaults)
async def read_delivery_defaults(user: CurrentUser = Depends(require_user)) -> DeliveryDefaults:
    """What future reminders of this owner will use, and what they may choose from."""
    user_id = user.profile.user_id
    return DeliveryDefaults(
        delivery=list(reminder_delivery.default_channels(user_id)),
        available=_available(user_id),
        saved=reminder_delivery.has_default(user_id),
    )


@router.put("/users/me/dashboard/reminders/delivery", response_model=DeliveryDefaults)
async def set_delivery_defaults(
    choice: DeliveryChoice, user: CurrentUser = Depends(require_user)
) -> DeliveryDefaults:
    """Choose the channels future reminders of this owner will use.

    Bounded on both sides: the body is a list from a fixed enum and nothing
    else, and a channel this owner is not authorised for is refused rather
    than saved. It changes no reminder that already exists, so nothing that is
    already armed is re-armed, cancelled or delivered twice.
    """
    user_id = user.profile.user_id
    try:
        saved = reminder_delivery.set_default_channels(user_id, choice.delivery)
    except SafeToolError:
        # The reason is the owner own: it names a channel, never a destination.
        logger.info("Refused a reminder delivery default that this owner may not use")
        raise HTTPException(status_code=422, detail="unavailable_channel") from None
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid") from None
    logger.info("Saved a reminder delivery default of %d channel(s)", len(saved))
    return DeliveryDefaults(delivery=list(saved), available=_available(user_id), saved=True)
