"""Calendar events and inbox messages from a user's connected provider accounts.

The dashboard asks this module, per connection, for the next few days of
events and the latest inbox messages. Each read is one bounded https
conversation with the provider's own API, authenticated with the access
token that :mod:`caal.provider_connections` holds encrypted for the account:

* Google: Calendar ``events.list`` on the primary calendar, and Gmail
  ``messages.list`` plus metadata-only ``messages.get`` (headers, labels and
  the snippet; never a body);
* Microsoft: Graph ``/me/calendarView`` and ``/me/mailFolders/inbox/messages``
  with an explicit ``$select`` (``bodyPreview`` is Graph's own plain-text
  preview; the body is never requested);
* Zoho: Calendar ``GET /api/v1/calendars`` then ``events`` per calendar within
  a range Zoho documents as at most 31 days, and Mail ``GET /api/accounts``
  for the account this grant covers followed by its documented ``newMails``
  search (sender, subject and Zoho's own ``summary``; never a body). Its API
  host is derived from the operator's configured accounts host, and it is
  authenticated with Zoho's own ``Zoho-oauthtoken`` scheme.

An access token at or past its expiry is renewed with the stored refresh
token (the ``refresh_token`` grant, with the application's client
credentials) and the renewed token is stored; a provider that answers ``401``
is retried once after such a renewal. Without a refresh token, or when the
provider refuses the renewal, the account is ``reconnect_required``: the user
re-approves it under Settings, and nothing here guesses around that.

What comes back is reduced to bounded plain text -- a title, a place, a
sender, a subject, a short preview, times in UTC, the provider's own https
link to the item -- and nothing else: no bodies, no attendees, no address
lists, no HTML, no raw payload. Only ``https`` endpoints are contacted,
redirects are never followed, every response is read within a byte budget,
every socket operation has a timeout, and each account's read has a
wall-clock budget on top.

Nothing here logs a token, a secret, a title, a subject, or an address.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import httpx

from .oauth_providers import ProviderConfig, ProviderRegistry
from .provider_connections import ConnectionCredentials, ConnectionStore, ProviderConnection

logger = logging.getLogger(__name__)

__all__ = [
    "DATA_REASONS",
    "FETCH_BUDGET_SECONDS",
    "MAX_ITEMS",
    "MAX_PREVIEW_LENGTH",
    "MAX_RESPONSE_BYTES",
    "MAX_TITLE_LENGTH",
    "MAX_WINDOW_DAYS",
    "TIMEOUT_SECONDS",
    "AccountFetch",
    "CalendarEvent",
    "InboxMessage",
    "ProviderDataClient",
    "ProviderDataError",
    "account_status",
]

# Per-operation socket budget, the connect budget within it, and the wall
# clock for one account's whole read (Gmail needs a list plus one metadata
# call per message).
TIMEOUT_SECONDS = 10.0
CONNECT_TIMEOUT_SECONDS = 5.0
FETCH_BUDGET_SECONDS = 20.0
_MAX_TIMEOUT_SECONDS = 30.0
_MAX_BUDGET_SECONDS = 120.0
# A page of events or previews is tens of kilobytes at most.
MAX_RESPONSE_BYTES = 256 * 1024
_MIN_RESPONSE_BYTES = 1024
_MAX_RESPONSE_BYTES_LIMIT = 1024 * 1024

MAX_ITEMS = 50
MAX_WINDOW_DAYS = 62
MAX_TITLE_LENGTH = 200
MAX_PREVIEW_LENGTH = 160
MAX_SENDER_LENGTH = 120
_MAX_ID_LENGTH = 512
_MAX_LINK_LENGTH = 2048
_MAX_TOKEN_LENGTH = 8192
_MAX_TOKEN_LIFETIME_SECONDS = 10 * 365 * 86400
# A token this close to expiry is renewed first rather than sent.
TOKEN_EXPIRY_SKEW_SECONDS = 60
# Gmail metadata calls, one per message: bounded fan-out and bounded count.
GMAIL_METADATA_CONCURRENCY = 4
MAX_GMAIL_MESSAGES = 15

_USER_AGENT = "CAAL-JARVIS connected-accounts"

DATA_REASONS: tuple[str, ...] = (
    "reconnect_required",
    "insufficient_scope",
    "not_configured",
    "provider_refused",
    "transport",
    "malformed_response",
    "unsupported",
)

# What the dashboard shows per account; every reason maps to exactly one.
_STATUS_BY_REASON = dict(
    reconnect_required="reconnect_required",
    insufficient_scope="insufficient_scope",
    not_configured="not_configured",
    unsupported="unsupported",
    provider_refused="unavailable",
    transport="unavailable",
    malformed_response="unavailable",
)

# Zoho runs one API host per data centre, alongside the accounts host the
# provider is configured with: ``accounts.zoho.com`` -> ``mail.zoho.com`` and
# ``calendar.zoho.com``. The origin is derived, never guessed.
ZOHO_ACCOUNTS_LABEL = "accounts."
ZOHO_CALENDAR_PATH = "/api/v1/calendars"
ZOHO_MAIL_PATH = "/api/accounts"
# Zoho documents a maximum event range of 31 days, and reads at most this many
# of the user's own calendars, two at a time.
ZOHO_MAX_RANGE_DAYS = 31
MAX_ZOHO_CALENDARS = 5
ZOHO_CALENDAR_CONCURRENCY = 2
MAX_ZOHO_ACCOUNTS = 25

_GOOGLE_CALENDAR_EVENTS = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
_GMAIL_MESSAGES = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
_GRAPH_CALENDAR_VIEW = "https://graph.microsoft.com/v1.0/me/calendarView"
_GRAPH_INBOX_MESSAGES = "https://graph.microsoft.com/v1.0/me/mailFolders/inbox/messages"
_GRAPH_SCOPE_PREFIX = "https://graph.microsoft.com/"

# Zoho authenticates with its own scheme, not ``Bearer``.
_AUTH_SCHEMES = dict(zoho="Zoho-oauthtoken")
_DEFAULT_AUTH_SCHEME = "Bearer"

# Any one of these grants is enough for the read this module performs. A
# connection whose granted scopes are known and include none of them is
# reported without a provider call; unknown grants are left to the provider.
_CALENDAR_SCOPES: dict[str, tuple[str, ...]] = dict(
    google=(
        "https://www.googleapis.com/auth/calendar.readonly",
        "https://www.googleapis.com/auth/calendar",
        "https://www.googleapis.com/auth/calendar.events.readonly",
        "https://www.googleapis.com/auth/calendar.events",
    ),
    microsoft=(
        "Calendars.Read",
        "Calendars.ReadWrite",
        "Calendars.Read.Shared",
        "Calendars.ReadWrite.Shared",
    ),
    zoho=("ZohoCalendar.event.READ", "ZohoCalendar.event.ALL"),
)
_MAIL_SCOPES: dict[str, tuple[str, ...]] = dict(
    google=(
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.modify",
        "https://mail.google.com/",
    ),
    microsoft=("Mail.Read", "Mail.ReadWrite", "Mail.Read.Shared", "Mail.ReadWrite.Shared"),
    zoho=("ZohoMail.messages.READ", "ZohoMail.messages.ALL"),
)

# Refresh-grant refusals that mean the stored grant itself is dead.
_DEAD_GRANT_CODES = frozenset(
    ("invalid_grant", "invalid_token", "unauthorized_client", "interaction_required")
)
# Google ``403`` reasons that are about the project or quota, not the scopes.
_GOOGLE_NON_SCOPE_403 = frozenset(
    (
        "accessNotConfigured",
        "dailyLimitExceeded",
        "rateLimitExceeded",
        "userRateLimitExceeded",
        "quotaExceeded",
    )
)

_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FRACTION = re.compile(r"(\.\d{1,6})\d+")
_TAG = re.compile(r"<[^<>]{0,500}>")
_PROVIDER_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
# An identifier safe to place in a URL path without escaping or traversal.
_PATH_TOKEN = re.compile(r"^[A-Za-z0-9_.~=-]{1,128}$")
_ZOHO_DAY = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
_ZOHO_STAMP = re.compile(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})(Z|[+-]\d{4})?$")


# --- errors and results ----------------------------------------------------------------


class ProviderDataError(Exception):
    """A read from one connected account failed.

    ``reason`` is one of :data:`DATA_REASONS`, safe to answer and to log. The
    message is for the exception chain only and must never reach a browser or
    a log line. ``provider_code`` is a short lower-case code the provider used
    (``invalid_grant`` and the like), kept for operator diagnostics.
    """

    def __init__(
        self,
        message: str = "The provider data could not be read",
        *,
        reason: str = "provider_refused",
        provider_code: str | None = None,
        status: int | None = None,
    ) -> None:
        if reason not in DATA_REASONS:
            raise ValueError("Unknown provider data reason")
        if provider_code is not None and _PROVIDER_CODE.fullmatch(provider_code) is None:
            raise ValueError("Unsafe provider error code")
        super().__init__(message)
        self.reason = reason
        self.provider_code = provider_code
        # The HTTP status the provider answered, when one was received.
        self.status = status


def account_status(error: ProviderDataError | None) -> str:
    """The per-account state the dashboard shows for a read outcome."""
    return "ok" if error is None else _STATUS_BY_REASON[error.reason]


@dataclass(frozen=True)
class CalendarEvent:
    """One upcoming event, bounded. Times are UTC ISO 8601; all-day events carry dates."""

    id: str = field(repr=False)
    start: str = ""
    end: str | None = None
    all_day: bool = False
    title: str | None = field(default=None, repr=False)
    location: str | None = field(default=None, repr=False)
    link: str | None = field(default=None, repr=False)
    status: str | None = None

    def view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "start": self.start,
            "end": self.end,
            "all_day": self.all_day,
            "location": self.location,
            "link": self.link,
            "status": self.status,
        }


@dataclass(frozen=True)
class InboxMessage:
    """One recent message, bounded: never a body, never a recipient list."""

    id: str = field(repr=False)
    received_at: str = ""
    unread: bool = False
    subject: str | None = field(default=None, repr=False)
    sender: str | None = field(default=None, repr=False)
    preview: str | None = field(default=None, repr=False)
    link: str | None = field(default=None, repr=False)

    def view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "sender": self.sender,
            "preview": self.preview,
            "received_at": self.received_at,
            "unread": self.unread,
            "link": self.link,
        }


@dataclass(frozen=True)
class AccountFetch:
    """What one connection yielded: items, or the bounded reason it could not."""

    connection: ProviderConnection
    events: tuple[CalendarEvent, ...] = ()
    messages: tuple[InboxMessage, ...] = ()
    error: ProviderDataError | None = None

    @property
    def status(self) -> str:
        return account_status(self.error)

    @property
    def reason(self) -> str | None:
        return None if self.error is None else self.error.reason


# --- bounded values ----------------------------------------------------------------------


def _text(value: object, limit: int) -> str | None:
    """Plain, single-line, bounded text; ``None`` for anything else or for nothing."""
    if not isinstance(value, str) or not value:
        return None
    pieces: list[str] = []
    for ch in value[: limit * 8]:
        category = unicodedata.category(ch)
        if category == "Cc":
            pieces.append(" ")
        elif category[0] != "C":
            pieces.append(ch)
    text = " ".join("".join(pieces).split())
    return text[:limit] or None


def _preview(value: object, limit: int) -> str | None:
    """A snippet with markup removed and entities decoded, then bounded."""
    if not isinstance(value, str):
        return None
    return _text(html.unescape(_TAG.sub(" ", value[: limit * 8])), limit)


def _sender(name: object, address: object) -> str | None:
    label = _text(name, MAX_SENDER_LENGTH)
    return label if label is not None else _text(address, MAX_SENDER_LENGTH)


def _sender_from_header(value: object) -> str | None:
    """``Name <addr>`` -> the name, else the address; bounded."""
    if not isinstance(value, str):
        return None
    name, address = parseaddr(value[: MAX_SENDER_LENGTH * 4])
    if name or address:
        return _sender(name, address)
    return _text(value, MAX_SENDER_LENGTH)


def _printable_token(value: object, limit: int) -> str | None:
    if not isinstance(value, str) or not 0 < len(value) <= limit:
        return None
    if not value.isprintable() or any(ch.isspace() for ch in value):
        return None
    return value


def _id_like(value: object) -> str | None:
    return _printable_token(value, _MAX_ID_LENGTH)


def _token_like(value: object) -> str | None:
    return _printable_token(value, _MAX_TOKEN_LENGTH)


def _https_link(value: object) -> str | None:
    """The provider's own https link to an item, or ``None``. Never anything else."""
    link = _printable_token(value, _MAX_LINK_LENGTH)
    if link is None:
        return None
    parts = urlsplit(link)
    if parts.scheme != "https" or not parts.netloc or "@" in parts.netloc:
        return None
    return link


def _is_https(url: str) -> bool:
    parts = urlsplit(url)
    return parts.scheme == "https" and bool(parts.netloc)


def _expires_in(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if not isinstance(value, int) or not 0 <= value <= _MAX_TOKEN_LIFETIME_SECONDS:
        return None
    return value


def _json_object(body: bytes) -> dict[str, Any] | None:
    try:
        loaded = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _short_code(value: object) -> str | None:
    """A provider's short error code as a safe lower-case token, if it is one."""
    if not isinstance(value, str) or not 0 < len(value) <= 64:
        return None
    code = re.sub(r"[^a-z0-9_]", "_", value.strip().lower())
    return code if _PROVIDER_CODE.fullmatch(code) else None


def _error_code_of(data: dict[str, Any] | None) -> str | None:
    """The provider's own error code from a refusal body, in its own words, or None."""
    if not data:
        return None
    error = data.get("error")
    if isinstance(error, str):
        return error
    if not isinstance(error, dict):
        return None
    details = error.get("errors")
    if isinstance(details, list) and details and isinstance(details[0], dict):
        reason = details[0].get("reason")
        if isinstance(reason, str):
            return reason
    for key in ("code", "status"):
        if isinstance(error.get(key), str):
            return error[key]
    return None


# --- time ----------------------------------------------------------------------------------


def _iso_utc(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_datetime(value: object) -> datetime | None:
    """ISO 8601 with an offset, a ``Z``, or none (taken as UTC); any fraction length."""
    if not isinstance(value, str) or not 10 <= len(value) <= 40:
        return None
    text = value.strip()
    if text[-1:] in ("Z", "z"):
        text = text[:-1] + "+00:00"
    text = _FRACTION.sub(r"\1", text)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _epoch_millis(value: object) -> datetime | None:
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value < 10**14:
        return None
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc)


def _date_only(value: object) -> str | None:
    return value if isinstance(value, str) and _DATE_ONLY.fullmatch(value) else None


def _check_window(start: datetime, end: datetime) -> None:
    for moment in (start, end):
        if not isinstance(moment, datetime) or moment.tzinfo is None:
            raise ValueError("The window must be given as aware datetimes")
    if not start < end or end - start > timedelta(days=MAX_WINDOW_DAYS):
        raise ValueError(f"The window must be positive and at most {MAX_WINDOW_DAYS} days")


def _check_limit(limit: object) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_ITEMS:
        raise ValueError(f"limit must be between 1 and {MAX_ITEMS}")
    return limit


# --- scopes --------------------------------------------------------------------------------


def _granted_scopes(connection: ProviderConnection) -> set[str]:
    granted: set[str] = set()
    for scope in connection.scopes:
        if connection.provider == "microsoft" and scope.startswith(_GRAPH_SCOPE_PREFIX):
            scope = scope[len(_GRAPH_SCOPE_PREFIX) :]
        granted.add(scope)
    return granted


def _require_scope(connection: ProviderConnection, accepted: dict[str, tuple[str, ...]]) -> None:
    if connection.provider not in accepted:
        raise ProviderDataError(f"No data adapter for {connection.provider}", reason="unsupported")
    granted = _granted_scopes(connection)
    if granted and not granted.intersection(accepted[connection.provider]):
        raise ProviderDataError("The read scope was not granted", reason="insufficient_scope")


# --- Google shapes -------------------------------------------------------------------------


def _google_when(value: object) -> tuple[str, bool] | None:
    if not isinstance(value, dict):
        return None
    moment = _parse_datetime(value.get("dateTime"))
    if moment is not None:
        return _iso_utc(moment), False
    day = _date_only(value.get("date"))
    return (day, True) if day is not None else None


def _google_event(item: object) -> CalendarEvent | None:
    if not isinstance(item, dict):
        return None
    event_id = _id_like(item.get("id"))
    status = item.get("status") if item.get("status") in ("confirmed", "tentative") else None
    start = _google_when(item.get("start"))
    if event_id is None or start is None or item.get("status") == "cancelled":
        return None
    end = _google_when(item.get("end"))
    return CalendarEvent(
        id=event_id,
        start=start[0],
        end=None if end is None else end[0],
        all_day=start[1],
        title=_text(item.get("summary"), MAX_TITLE_LENGTH),
        location=_text(item.get("location"), MAX_TITLE_LENGTH),
        link=_https_link(item.get("htmlLink")),
        status=status,
    )


def _gmail_message(data: dict[str, Any]) -> InboxMessage | None:
    message_id = _id_like(data.get("id"))
    received = _epoch_millis(data.get("internalDate"))
    if message_id is None or received is None:
        return None
    labels = data.get("labelIds")
    headers: dict[str, object] = {}
    payload = data.get("payload")
    raw_headers = payload.get("headers") if isinstance(payload, dict) else None
    if isinstance(raw_headers, list):
        for entry in raw_headers[:64]:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                headers.setdefault(entry["name"].lower(), entry.get("value"))
    return InboxMessage(
        id=message_id,
        received_at=_iso_utc(received),
        unread=isinstance(labels, list) and "UNREAD" in labels,
        subject=_text(headers.get("subject"), MAX_TITLE_LENGTH),
        sender=_sender_from_header(headers.get("from")),
        preview=_preview(data.get("snippet"), MAX_PREVIEW_LENGTH),
        link=None,
    )


# --- Microsoft Graph shapes ----------------------------------------------------------------


def _graph_when(value: object, *, all_day: bool) -> str | None:
    if not isinstance(value, dict) or not isinstance(value.get("dateTime"), str):
        return None
    raw = value["dateTime"]
    if all_day:
        return _date_only(raw[:10])
    moment = _parse_datetime(raw)
    if moment is None:
        return None
    zone = value.get("timeZone")
    if isinstance(zone, str) and zone and zone.upper() != "UTC":
        try:
            info = ZoneInfo(zone)
        except (KeyError, ValueError, OSError):
            return None
        moment = moment.replace(tzinfo=info).astimezone(timezone.utc)
    return _iso_utc(moment)


def _graph_event(item: object) -> CalendarEvent | None:
    if not isinstance(item, dict) or item.get("isCancelled") is True:
        return None
    event_id = _id_like(item.get("id"))
    all_day = item.get("isAllDay") is True
    start = _graph_when(item.get("start"), all_day=all_day)
    if event_id is None or start is None:
        return None
    location = item.get("location")
    place = location.get("displayName") if isinstance(location, dict) else None
    return CalendarEvent(
        id=event_id,
        start=start,
        end=_graph_when(item.get("end"), all_day=all_day),
        all_day=all_day,
        title=_text(item.get("subject"), MAX_TITLE_LENGTH),
        location=_text(place, MAX_TITLE_LENGTH),
        link=_https_link(item.get("webLink")),
        status="tentative" if item.get("showAs") == "tentative" else "confirmed",
    )


def _graph_message(item: object) -> InboxMessage | None:
    if not isinstance(item, dict):
        return None
    message_id = _id_like(item.get("id"))
    received = _parse_datetime(item.get("receivedDateTime"))
    if message_id is None or received is None:
        return None
    sender = item.get("from")
    address = sender.get("emailAddress") if isinstance(sender, dict) else None
    mailbox = address if isinstance(address, dict) else {}
    return InboxMessage(
        id=message_id,
        received_at=_iso_utc(received),
        unread=item.get("isRead") is False,
        subject=_text(item.get("subject"), MAX_TITLE_LENGTH),
        sender=_sender(mailbox.get("name"), mailbox.get("address")),
        preview=_preview(item.get("bodyPreview"), MAX_PREVIEW_LENGTH),
        link=_https_link(item.get("webLink")),
    )


# --- Zoho shapes -----------------------------------------------------------------------------


def _zoho_stamp(moment: datetime) -> str:
    """Zoho's basic-format instant, ``yyyyMMdd'T'HHmmss'Z'``."""
    return moment.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y%m%dT%H%M%SZ")


def _zoho_day(value: object) -> str | None:
    """``20231117`` -> ``2023-11-17``; anything else is nothing."""
    if not isinstance(value, str):
        return None
    match = _ZOHO_DAY.fullmatch(value.strip())
    return "-".join(match.groups()) if match is not None else None


def _zoho_moment(value: object, zone: object) -> str | None:
    """A Zoho basic-format time as UTC, anchored by its own offset or its calendar's zone.

    A time with neither is not guessed at: it is dropped, because a wall clock
    without a zone is not an instant.
    """
    if not isinstance(value, str):
        return None
    match = _ZOHO_STAMP.fullmatch(value.strip())
    if match is None:
        return None
    year, month, day, hour, minute, second, offset = match.groups()
    text = f"{year}-{month}-{day}T{hour}:{minute}:{second}"
    if offset:
        moment = _parse_datetime(text + offset)
        return None if moment is None else _iso_utc(moment)
    if not isinstance(zone, str) or not zone:
        return None
    try:
        info = ZoneInfo(zone)
    except (KeyError, ValueError, OSError):
        return None
    naive = _parse_datetime(text)
    return None if naive is None else _iso_utc(naive.replace(tzinfo=info))


def _zoho_event(item: object) -> CalendarEvent | None:
    """One Zoho event reduced to a bounded summary. Never an attendee or a description."""
    if not isinstance(item, dict):
        return None
    event_id = _id_like(item.get("uid"))
    when = item.get("dateandtime")
    if event_id is None or not isinstance(when, dict):
        return None
    all_day = item.get("isallday") is True
    if all_day:
        start = _zoho_day(when.get("start")) or _zoho_day(item.get("start"))
        end = _zoho_day(when.get("end")) or _zoho_day(item.get("end"))
    else:
        zone = when.get("timezone")
        start = _zoho_moment(when.get("start"), zone)
        end = _zoho_moment(when.get("end"), zone)
    if start is None:
        return None
    return CalendarEvent(
        id=event_id,
        start=start,
        end=end,
        all_day=all_day,
        title=_text(item.get("title"), MAX_TITLE_LENGTH),
        # Zoho documents neither a place nor a browser link on this payload, and
        # neither is invented here.
        location=None,
        link=None,
        status=None,
    )


def _zoho_message(item: object) -> InboxMessage | None:
    """One Zoho mail summary: sender, subject, Zoho's own preview. Never a body."""
    if not isinstance(item, dict):
        return None
    message_id = _id_like(item.get("messageId"))
    stamp = item.get("receivedTime")
    if stamp is None:
        stamp = item.get("receivedtime")
    received = _epoch_millis(stamp)
    if message_id is None or received is None:
        return None
    status = item.get("status")
    return InboxMessage(
        id=message_id,
        received_at=_iso_utc(received),
        # Zoho states the read state in its own words; any other value is not
        # read as a claim either way, and the message is shown as read.
        unread=isinstance(status, str) and status.strip().lower() == "unread",
        subject=_preview(item.get("subject"), MAX_TITLE_LENGTH),
        sender=_sender(item.get("sender"), item.get("fromAddress")),
        preview=_preview(item.get("summary"), MAX_PREVIEW_LENGTH),
        # ``URI`` is Zoho's API address for the message, not a page a person
        # can open, so no link is offered rather than a misleading one.
        link=None,
    )


# --- one account's bearer ------------------------------------------------------------------


class _Bearer:
    """The access token for one connection during one read; renewed at most once."""

    def __init__(
        self,
        owner: ProviderDataClient,
        client: httpx.AsyncClient,
        user_id: str,
        connection: ProviderConnection,
    ) -> None:
        self._owner = owner
        self._client = client
        self._user_id = user_id
        self._connection = connection
        self._token: str | None = None
        self._renewed = False
        self._lock = asyncio.Lock()

    @property
    def scheme(self) -> str:
        """The authorization scheme this provider documents for its own API."""
        return _AUTH_SCHEMES.get(self._connection.provider, _DEFAULT_AUTH_SCHEME)

    async def token(self) -> str:
        async with self._lock:
            if self._token is None:
                self._token, self._renewed = await self._owner._current_token(
                    self._client, self._user_id, self._connection
                )
            return self._token

    async def renew(self, stale: str) -> bool:
        """Renew once after ``stale`` was refused. False when nothing more can be done."""
        async with self._lock:
            if self._token != stale:
                return True  # another request of this read already renewed it
            if self._renewed:
                return False
            self._renewed = True
            self._token = await self._owner._renew_token(
                self._client, self._user_id, self._connection
            )
            return True


# --- the client ----------------------------------------------------------------------------


class ProviderDataClient:
    """Bounded reads from a user's Google and Microsoft accounts.

    ``transport`` is for tests (``httpx.MockTransport``); production uses the
    default. ``clock`` decides whether a stored token has expired.
    """

    def __init__(
        self,
        store: ConnectionStore,
        providers: ProviderRegistry,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = TIMEOUT_SECONDS,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        fetch_budget_seconds: float = FETCH_BUDGET_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        for name, value, upper in (
            ("timeout_seconds", timeout_seconds, _MAX_TIMEOUT_SECONDS),
            ("fetch_budget_seconds", fetch_budget_seconds, _MAX_BUDGET_SECONDS),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number of seconds")
            if not 0 < float(value) <= upper:
                raise ValueError(f"{name} must be within (0, {upper}]")
        if isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, int):
            raise ValueError("max_response_bytes must be a whole number of bytes")
        if not _MIN_RESPONSE_BYTES <= max_response_bytes <= _MAX_RESPONSE_BYTES_LIMIT:
            raise ValueError(
                f"max_response_bytes must be within [{_MIN_RESPONSE_BYTES}, "
                f"{_MAX_RESPONSE_BYTES_LIMIT}]"
            )
        self._store = store
        self._providers = providers
        self._transport = transport
        self._timeout = float(timeout_seconds)
        self._budget = float(fetch_budget_seconds)
        self._max_bytes = int(max_response_bytes)
        self._clock = clock

    def __repr__(self) -> str:
        return (
            f"ProviderDataClient(timeout_seconds={self._timeout}, "
            f"fetch_budget_seconds={self._budget}, max_response_bytes={self._max_bytes})"
        )

    @property
    def timeout_seconds(self) -> float:
        return self._timeout

    @property
    def fetch_budget_seconds(self) -> float:
        return self._budget

    @property
    def max_response_bytes(self) -> int:
        return self._max_bytes

    def _now(self) -> int:
        return int(self._clock())

    def _http(self) -> httpx.AsyncClient:
        timeout = httpx.Timeout(self._timeout, connect=min(CONNECT_TIMEOUT_SECONDS, self._timeout))
        return httpx.AsyncClient(
            transport=self._transport,
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            headers=dict([("Accept", "application/json"), ("User-Agent", _USER_AGENT)]),
        )

    # --- tokens ----------------------------------------------------------------------

    async def _current_token(
        self, client: httpx.AsyncClient, user_id: str, connection: ProviderConnection
    ) -> tuple[str, bool]:
        """The stored access token, renewed first when it is at or past its expiry."""
        credentials = self._store.credentials(user_id, connection.connection_id)
        if credentials is None:
            raise ProviderDataError("No usable credentials are stored", reason="reconnect_required")
        expires_at = credentials.token_expires_at
        if expires_at is None or expires_at > self._now() + TOKEN_EXPIRY_SKEW_SECONDS:
            return credentials.access_token, False
        return await self._refresh(client, user_id, connection, credentials), True

    async def _renew_token(
        self, client: httpx.AsyncClient, user_id: str, connection: ProviderConnection
    ) -> str:
        credentials = self._store.credentials(user_id, connection.connection_id)
        if credentials is None:
            raise ProviderDataError("No usable credentials are stored", reason="reconnect_required")
        return await self._refresh(client, user_id, connection, credentials)

    async def _refresh(
        self,
        client: httpx.AsyncClient,
        user_id: str,
        connection: ProviderConnection,
        credentials: ConnectionCredentials,
    ) -> str:
        """One ``refresh_token`` grant at the provider's token endpoint; store what it yields."""
        if credentials.refresh_token is None:
            raise ProviderDataError(
                "The access token expired and cannot be renewed", reason="reconnect_required"
            )
        config: ProviderConfig | None = self._providers.get(connection.provider)
        if config is None:
            raise ProviderDataError("The provider is no longer configured", reason="not_configured")
        form = dict(
            grant_type="refresh_token",
            refresh_token=credentials.refresh_token,
            client_id=config.client_id,
            client_secret=config.client_secret,
        )
        request = client.build_request("POST", config.token_endpoint, data=form)
        status, body = await self._send(client, request)
        data = _json_object(body)
        if data is None:
            reason = "malformed_response" if 200 <= status < 300 else "provider_refused"
            raise ProviderDataError("The token endpoint did not answer JSON", reason=reason)
        error = _short_code(data.get("error"))
        if error is not None or not 200 <= status < 300:
            if error in _DEAD_GRANT_CODES or (error is None and status == 401):
                raise ProviderDataError(
                    "The refresh grant was refused",
                    reason="reconnect_required",
                    provider_code=error,
                )
            raise ProviderDataError(
                "The token endpoint refused the renewal",
                reason="provider_refused",
                provider_code=error,
            )
        access = _token_like(data.get("access_token"))
        if access is None:
            raise ProviderDataError("No usable access token", reason="malformed_response")
        refresh: str | None = None
        if data.get("refresh_token") is not None:
            refresh = _token_like(data.get("refresh_token"))
            if refresh is None:
                raise ProviderDataError("Unusable refresh token", reason="malformed_response")
        stored = self._store.refresh_credentials(
            user_id,
            connection.connection_id,
            access_token=access,
            refresh_token=refresh,
            expires_in=_expires_in(data.get("expires_in")),
            now=self._now(),
        )
        if stored is None:
            raise ProviderDataError("The connection is no longer live", reason="reconnect_required")
        logger.info("Renewed the access token for a %s connection", connection.provider)
        return access

    # --- transport -----------------------------------------------------------------------

    async def _send(self, client: httpx.AsyncClient, request: httpx.Request) -> tuple[int, bytes]:
        """Send one request; return the status and a body read within the byte budget."""
        if not _is_https(str(request.url)):
            raise ProviderDataError("Refusing a non-https request", reason="transport")
        try:
            response = await client.send(request, stream=True)
        except (httpx.HTTPError, OSError) as exc:
            raise ProviderDataError(
                f"Transport failure: {type(exc).__name__}", reason="transport"
            ) from exc
        try:
            declared = response.headers.get("content-length")
            if declared is not None and (not declared.isdigit() or int(declared) > self._max_bytes):
                raise ProviderDataError("Response too large", reason="malformed_response")
            chunks: list[bytes] = []
            total = 0
            try:
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > self._max_bytes:
                        raise ProviderDataError("Response too large", reason="malformed_response")
                    chunks.append(chunk)
            except (httpx.HTTPError, OSError) as exc:
                raise ProviderDataError(
                    f"Transport failure: {type(exc).__name__}", reason="transport"
                ) from exc
        finally:
            await response.aclose()
        return response.status_code, b"".join(chunks)

    async def _authorized_get(
        self,
        client: httpx.AsyncClient,
        url: str,
        params: dict[str, Any],
        headers: dict[str, str] | None,
        token: str,
        scheme: str,
    ) -> tuple[int, bytes]:
        sent = dict(headers or ())
        sent["Authorization"] = f"{scheme} {token}"
        request = client.build_request("GET", url, params=params, headers=sent)
        return await self._send(client, request)

    async def _get_json(
        self,
        client: httpx.AsyncClient,
        bearer: _Bearer,
        url: str,
        params: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """One authenticated GET, retried once with a renewed token after a ``401``."""
        token = await bearer.token()
        scheme = bearer.scheme
        status, body = await self._authorized_get(client, url, params, headers, token, scheme)
        if status == 401 and await bearer.renew(token):
            token = await bearer.token()
            status, body = await self._authorized_get(client, url, params, headers, token, scheme)
        if status == 401:
            raise ProviderDataError(
                "The provider refused the access token",
                reason="reconnect_required",
                status=status,
            )
        data = _json_object(body)
        if status == 403:
            raise self._forbidden(data)
        if 300 <= status < 400:
            raise ProviderDataError(
                "The provider redirected", reason="malformed_response", status=status
            )
        if not 200 <= status < 300:
            raise ProviderDataError(
                f"The provider answered {status}",
                reason="provider_refused",
                provider_code=_short_code(_error_code_of(data)) or f"http_{status}",
                status=status,
            )
        if data is None:
            raise ProviderDataError("The provider did not answer JSON", reason="malformed_response")
        return data

    @staticmethod
    def _forbidden(data: dict[str, Any] | None) -> ProviderDataError:
        code = _error_code_of(data)
        short = _short_code(code)
        if code in _GOOGLE_NON_SCOPE_403:
            return ProviderDataError(
                "The provider refused the read",
                reason="provider_refused",
                provider_code=short,
                status=403,
            )
        return ProviderDataError(
            "The read scope was not granted",
            reason="insufficient_scope",
            provider_code=short,
            status=403,
        )

    # --- reads ---------------------------------------------------------------------------

    async def calendar_events(
        self,
        user_id: str,
        connection: ProviderConnection,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[CalendarEvent]:
        """Events from one connection within ``[start, end)``, in the provider's order."""
        _check_window(start, end)
        count = _check_limit(limit)
        _require_scope(connection, _CALENDAR_SCOPES)
        async with self._http() as client:
            return await self._calendar_with(client, user_id, connection, start, end, count)

    async def inbox_messages(
        self, user_id: str, connection: ProviderConnection, *, limit: int
    ) -> list[InboxMessage]:
        """The newest inbox messages from one connection, newest first."""
        count = _check_limit(limit)
        _require_scope(connection, _MAIL_SCOPES)
        async with self._http() as client:
            return await self._inbox_with(client, user_id, connection, count)

    async def calendar_feed(
        self,
        user_id: str,
        connections: Sequence[ProviderConnection],
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[AccountFetch]:
        """Every connection's events, read concurrently; one failure hides no other account."""
        _check_window(start, end)
        count = _check_limit(limit)

        async def one(client: httpx.AsyncClient, connection: ProviderConnection) -> AccountFetch:
            try:
                _require_scope(connection, _CALENDAR_SCOPES)
                events = await self._calendar_with(client, user_id, connection, start, end, count)
            except ProviderDataError as exc:
                return AccountFetch(connection, error=self._failed("calendar", connection, exc))
            except Exception as exc:  # noqa: BLE001 - never echoed
                return AccountFetch(connection, error=self._unexpected("calendar", connection, exc))
            return AccountFetch(connection, events=tuple(events))

        async with self._http() as client:
            return list(await asyncio.gather(*(one(client, item) for item in connections)))

    async def inbox_feed(
        self, user_id: str, connections: Sequence[ProviderConnection], *, limit: int
    ) -> list[AccountFetch]:
        """Every connection's newest messages, read concurrently; failures are per account."""
        count = _check_limit(limit)

        async def one(client: httpx.AsyncClient, connection: ProviderConnection) -> AccountFetch:
            try:
                _require_scope(connection, _MAIL_SCOPES)
                messages = await self._inbox_with(client, user_id, connection, count)
            except ProviderDataError as exc:
                return AccountFetch(connection, error=self._failed("inbox", connection, exc))
            except Exception as exc:  # noqa: BLE001 - never echoed
                return AccountFetch(connection, error=self._unexpected("inbox", connection, exc))
            return AccountFetch(connection, messages=tuple(messages))

        async with self._http() as client:
            return list(await asyncio.gather(*(one(client, item) for item in connections)))

    @staticmethod
    def _failed(
        what: str, connection: ProviderConnection, exc: ProviderDataError
    ) -> ProviderDataError:
        if exc.provider_code is None:
            logger.warning("Reading %s from %s failed: %s", what, connection.provider, exc.reason)
        else:
            logger.warning(
                "Reading %s from %s failed: %s (provider_code=%s)",
                what,
                connection.provider,
                exc.reason,
                exc.provider_code,
            )
        return exc

    @staticmethod
    def _unexpected(what: str, connection: ProviderConnection, exc: Exception) -> ProviderDataError:
        logger.error("Reading %s from %s raised %s", what, connection.provider, type(exc).__name__)
        return ProviderDataError("Unexpected failure while reading", reason="transport")

    async def _bounded(self, reader: Any) -> Any:
        try:
            return await asyncio.wait_for(reader, timeout=self._budget)
        except asyncio.TimeoutError as exc:
            raise ProviderDataError(
                "The read exceeded its time budget", reason="transport"
            ) from exc

    def _zoho_origin(self, label: str) -> str:
        """The Zoho API origin for one service, derived from the configured accounts host.

        Zoho pairs ``accounts.zoho.<dc>`` with ``mail.zoho.<dc>`` and
        ``calendar.zoho.<dc>``. Anything else is not guessed at.
        """
        config: ProviderConfig | None = self._providers.get("zoho")
        if config is None:
            raise ProviderDataError("The provider is no longer configured", reason="not_configured")
        host = urlsplit(config.token_endpoint).netloc
        if not host.startswith(ZOHO_ACCOUNTS_LABEL) or "@" in host:
            raise ProviderDataError(
                "The Zoho accounts host is not one this adapter can map", reason="not_configured"
            )
        return "https://" + label + "." + host[len(ZOHO_ACCOUNTS_LABEL) :]

    async def _calendar_with(
        self,
        client: httpx.AsyncClient,
        user_id: str,
        connection: ProviderConnection,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[CalendarEvent]:
        bearer = _Bearer(self, client, user_id, connection)
        if connection.provider == "google":
            return await self._bounded(self._google_calendar(client, bearer, start, end, limit))
        if connection.provider == "zoho":
            origin = self._zoho_origin("calendar")
            return await self._bounded(
                self._zoho_calendar(client, bearer, origin, start, end, limit)
            )
        return await self._bounded(self._graph_calendar(client, bearer, start, end, limit))

    async def _inbox_with(
        self,
        client: httpx.AsyncClient,
        user_id: str,
        connection: ProviderConnection,
        limit: int,
    ) -> list[InboxMessage]:
        bearer = _Bearer(self, client, user_id, connection)
        if connection.provider == "google":
            return await self._bounded(self._gmail(client, bearer, limit))
        if connection.provider == "zoho":
            origin = self._zoho_origin("mail")
            return await self._bounded(self._zoho_inbox(client, bearer, origin, limit))
        return await self._bounded(self._graph_inbox(client, bearer, limit))

    # --- providers ---------------------------------------------------------------------

    async def _google_calendar(
        self,
        client: httpx.AsyncClient,
        bearer: _Bearer,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[CalendarEvent]:
        params = dict(
            timeMin=_iso_utc(start),
            timeMax=_iso_utc(end),
            singleEvents="true",
            orderBy="startTime",
            maxResults=str(limit),
            showDeleted="false",
            fields="items(id,summary,start,end,location,status,htmlLink)",
        )
        data = await self._get_json(client, bearer, _GOOGLE_CALENDAR_EVENTS, params)
        items = data.get("items", [])
        if not isinstance(items, list):
            raise ProviderDataError("Unexpected events shape", reason="malformed_response")
        return [event for event in map(_google_event, items[:limit]) if event is not None]

    async def _gmail(
        self, client: httpx.AsyncClient, bearer: _Bearer, limit: int
    ) -> list[InboxMessage]:
        count = min(limit, MAX_GMAIL_MESSAGES)
        listing = await self._get_json(
            client, bearer, _GMAIL_MESSAGES, dict(maxResults=str(count), labelIds="INBOX")
        )
        refs = listing.get("messages", [])
        if not isinstance(refs, list):
            raise ProviderDataError("Unexpected message list shape", reason="malformed_response")
        ids: list[str] = []
        for ref in refs[:count]:
            message_id = _id_like(ref.get("id")) if isinstance(ref, dict) else None
            if message_id is not None:
                ids.append(message_id)
        gate = asyncio.Semaphore(GMAIL_METADATA_CONCURRENCY)

        async def metadata(message_id: str) -> dict[str, Any] | None:
            async with gate:
                try:
                    return await self._get_json(
                        client,
                        bearer,
                        f"{_GMAIL_MESSAGES}/{message_id}",
                        dict(format="metadata", metadataHeaders=["Subject", "From"]),
                    )
                except ProviderDataError as exc:
                    if exc.status == 404:
                        return None  # gone since it was listed
                    raise

        details = await asyncio.gather(*(metadata(message_id) for message_id in ids))
        parsed = (_gmail_message(detail) for detail in details if detail is not None)
        messages = [message for message in parsed if message is not None]
        messages.sort(key=lambda message: message.received_at, reverse=True)
        return messages

    async def _graph_calendar(
        self,
        client: httpx.AsyncClient,
        bearer: _Bearer,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[CalendarEvent]:
        params = dict(
            [
                ("startDateTime", _iso_utc(start)),
                ("endDateTime", _iso_utc(end)),
                ("$top", str(limit)),
                ("$select", "id,subject,start,end,isAllDay,isCancelled,location,webLink,showAs"),
            ]
        )
        prefer = dict(Prefer='outlook.timezone="UTC"')
        data = await self._get_json(client, bearer, _GRAPH_CALENDAR_VIEW, params, prefer)
        value = data.get("value", [])
        if not isinstance(value, list):
            raise ProviderDataError("Unexpected events shape", reason="malformed_response")
        events = [event for event in map(_graph_event, value[:limit]) if event is not None]
        events.sort(key=lambda event: event.start)
        return events

    async def _graph_inbox(
        self, client: httpx.AsyncClient, bearer: _Bearer, limit: int
    ) -> list[InboxMessage]:
        params = dict(
            [
                ("$top", str(limit)),
                ("$orderby", "receivedDateTime desc"),
                ("$select", "id,subject,from,receivedDateTime,isRead,bodyPreview,webLink"),
            ]
        )
        data = await self._get_json(client, bearer, _GRAPH_INBOX_MESSAGES, params)
        value = data.get("value", [])
        if not isinstance(value, list):
            raise ProviderDataError("Unexpected messages shape", reason="malformed_response")
        parsed = map(_graph_message, value[:limit])
        messages = [message for message in parsed if message is not None]
        messages.sort(key=lambda message: message.received_at, reverse=True)
        return messages

    # --- Zoho ---------------------------------------------------------------------------

    @staticmethod
    def _zoho_path_ids(entries: object, key: str, limit: int) -> list[str]:
        """Identifiers safe to place in a URL path, bounded in number."""
        found: list[str] = []
        if not isinstance(entries, list):
            return found
        for entry in entries[: limit * 8]:
            value = _id_like(entry.get(key)) if isinstance(entry, dict) else None
            if value is not None and _PATH_TOKEN.fullmatch(value) is not None:
                found.append(value)
            if len(found) == limit:
                break
        return found

    async def _zoho_calendar(
        self,
        client: httpx.AsyncClient,
        bearer: _Bearer,
        origin: str,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[CalendarEvent]:
        """The user's own Zoho calendars, then the events each holds in the window."""
        listing = await self._get_json(
            client, bearer, origin + ZOHO_CALENDAR_PATH, dict(category="own")
        )
        calendars = listing.get("calendars")
        if not isinstance(calendars, list):
            raise ProviderDataError("Unexpected calendars shape", reason="malformed_response")
        uids = self._zoho_path_ids(calendars, "uid", MAX_ZOHO_CALENDARS)
        # Zoho documents a maximum range; ask for no more than it accepts.
        window_end = min(end, start + timedelta(days=ZOHO_MAX_RANGE_DAYS))
        window = json.dumps(
            dict(start=_zoho_stamp(start), end=_zoho_stamp(window_end)), separators=(",", ":")
        )
        gate = asyncio.Semaphore(ZOHO_CALENDAR_CONCURRENCY)

        async def events_of(uid: str) -> list[CalendarEvent]:
            async with gate:
                data = await self._get_json(
                    client,
                    bearer,
                    f"{origin}{ZOHO_CALENDAR_PATH}/{uid}/events",
                    dict(range=window),
                )
            found = data.get("events")
            if not isinstance(found, list):
                raise ProviderDataError("Unexpected events shape", reason="malformed_response")
            return [event for event in map(_zoho_event, found[:limit]) if event is not None]

        collected = await asyncio.gather(*(events_of(uid) for uid in uids))
        events = [event for calendar in collected for event in calendar]
        events.sort(key=lambda event: event.start)
        return events[:limit]

    async def _zoho_inbox(
        self, client: httpx.AsyncClient, bearer: _Bearer, origin: str, limit: int
    ) -> list[InboxMessage]:
        """The Zoho mail account this grant covers, then its newest mail."""
        listing = await self._get_json(client, bearer, origin + ZOHO_MAIL_PATH, {})
        accounts = self._zoho_path_ids(listing.get("data"), "accountId", MAX_ZOHO_ACCOUNTS)
        if not accounts:
            raise ProviderDataError(
                "No usable Zoho mail account was named", reason="malformed_response"
            )
        data = await self._get_json(
            client,
            bearer,
            f"{origin}{ZOHO_MAIL_PATH}/{accounts[0]}/messages/search",
            dict(searchKey="newMails", start="1", limit=str(limit)),
        )
        rows = data.get("data")
        if not isinstance(rows, list):
            raise ProviderDataError("Unexpected messages shape", reason="malformed_response")
        parsed = map(_zoho_message, rows[:limit])
        messages = [message for message in parsed if message is not None]
        messages.sort(key=lambda message: message.received_at, reverse=True)
        return messages
