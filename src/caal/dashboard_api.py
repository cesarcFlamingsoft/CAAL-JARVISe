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
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict

from .connections_api import ConnectionsRuntime, require_connections
from .provider_data import MAX_ITEMS, AccountFetch
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


def _account(fetch: AccountFetch, count: int) -> DashboardAccount:
    return DashboardAccount(
        connection_id=fetch.connection.connection_id,
        provider=fetch.connection.provider,
        account_label=fetch.connection.account_label,
        status=fetch.status,
        reason=fetch.reason,
        count=count,
    )


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
    results = await runtime.data.calendar_feed(
        user_id, connections, start=start, end=end, limit=limit
    )
    _note("calendar", results)
    events = [
        CalendarEventResponse(
            connection_id=fetch.connection.connection_id,
            provider=fetch.connection.provider,
            **event.view(),
        )
        for fetch in results
        for event in fetch.events
    ]
    events.sort(key=lambda event: event.start)
    return CalendarFeedResponse(
        generated_at=now,
        window_start=_iso(start),
        window_end=_iso(end),
        accounts=[_account(fetch, len(fetch.events)) for fetch in results],
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
    results = await runtime.data.inbox_feed(user_id, connections, limit=limit)
    _note("inbox", results)
    messages = [
        InboxMessageResponse(
            connection_id=fetch.connection.connection_id,
            provider=fetch.connection.provider,
            **message.view(),
        )
        for fetch in results
        for message in fetch.messages
    ]
    messages.sort(key=lambda message: message.received_at, reverse=True)
    shown = messages[:limit]
    return InboxFeedResponse(
        generated_at=runtime.now(),
        accounts=[_account(fetch, len(fetch.messages)) for fetch in results],
        messages=shown,
        unread_count=sum(1 for message in shown if message.unread),
    )
