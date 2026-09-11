"""Voice tools over the connected-account knowledge of the signed-in user.

Five user-scoped tools -- recent email, email search, one email read as a
safe summary, the upcoming schedule, and an event-existence check -- that
answer from :class:caal.knowledge.KnowledgeService. Each returns the voice
tool contract, a dict with status, message and data, where message is a
short sentence or two meant to be spoken as-is: no ids, no links, no bodies,
times in the deployment time zone in plain words, and an honest note when an
account could not be refreshed or nothing is connected.

The user is bound by the runtime (see :mod:caal.user_scope), never by the
model. A session that is not signed in, or a backend without the knowledge
runtime, gets an explicit unavailable answer rather than a guess.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from caal.knowledge import (
    MAX_ANSWER_ITEMS,
    MAX_QUERY_LENGTH,
    AccountAnswer,
    KnowledgeService,
    query_terms,
)
from caal.knowledge_store import IndexedEvent, IndexedMessage

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_FIND_DAYS",
    "DEFAULT_NEXT_DAYS",
    "DEFAULT_NEXT_LIMIT",
    "DEFAULT_RECENT_LIMIT",
    "DEFAULT_SCHEDULE_DAYS",
    "DEFAULT_SCHEDULE_LIMIT",
    "MAX_DAYS",
    "MAX_NEXT_LIMIT",
    "configure",
    "find_event",
    "next_events",
    "read_email_summary",
    "recent_email",
    "reset",
    "search_email",
    "session_unavailable_result",
    "upcoming_schedule",
]

DEFAULT_RECENT_LIMIT = 5
DEFAULT_SCHEDULE_LIMIT = 8
DEFAULT_SCHEDULE_DAYS = 7
DEFAULT_FIND_DAYS = 14
# "What is my next event" is one event unless the user asked for more; the
# horizon is wide because the answer is the nearest one, not a list.
DEFAULT_NEXT_LIMIT = 1
MAX_NEXT_LIMIT = 5
DEFAULT_NEXT_DAYS = 31
MAX_DAYS = 31
# How many items a spoken summary lists before saying "and N more".
SPOKEN_MESSAGES = 3
SPOKEN_EVENTS = 5
_MAX_ECHO_LENGTH = 80

SESSION_UNAVAILABLE = (
    "Connected email and calendar accounts are only available for a signed-in user profile, "
    "and this session is not signed in, so I have not looked anything up."
)
BACKEND_UNAVAILABLE = (
    "Connected email and calendar knowledge is not available on this JARVIS backend right now."
)
NO_ACCOUNTS = (
    "No email or calendar accounts are connected to your profile yet. You can link Google, "
    "Microsoft or Zoho under Settings, Integrations."
)

_PROVIDER_NAMES = dict(google="Google", microsoft="Microsoft", zoho="Zoho")
_PROBLEMS = dict(
    unavailable="could not be refreshed just now",
    reconnect_required="needs to be reconnected under Settings",
    insufficient_scope=(
        "was linked without permission to read this and needs to be reconnected under Settings"
    ),
    not_configured="cannot be refreshed because its provider is no longer configured on the server",
    unsupported="cannot be read yet",
)
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


# --- binding to the runtime -----------------------------------------------------------------

_lock = threading.Lock()
_provider: Callable[[], KnowledgeService | None] | None = None


def configure(provider: Callable[[], KnowledgeService | None]) -> None:
    """Bind the tools to a knowledge service factory (tests; embedding runtimes)."""
    global _provider
    with _lock:
        _provider = provider


def reset() -> None:
    global _provider
    with _lock:
        _provider = None


def _default_service() -> KnowledgeService | None:
    """The process-wide knowledge service, when multi-user identity is configured."""
    from caal import connections_api, user_api  # lazy: keep the registry import light

    identity = user_api.get_runtime()
    if identity is None or getattr(identity, "store", None) is None:
        return None
    runtime = connections_api.get_connections_runtime(identity)
    return None if runtime is None else runtime.knowledge


def _service() -> KnowledgeService | None:
    with _lock:
        provider = _provider
    try:
        return (provider or _default_service)()
    except Exception as exc:  # noqa: BLE001 - a tool answers, it never raises into the LLM
        logger.error("The knowledge service could not be built: %s", type(exc).__name__)
        return None


# --- the contract ---------------------------------------------------------------------------


def session_unavailable_result() -> dict[str, Any]:
    """The refusal the runtime gives an unidentified session under multi-user.

    Spoken in these tools own words rather than the memory tools words: the
    user asked about mail or calendar, and the refusal must say so.
    """
    return dict(status="unauthorized", message=SESSION_UNAVAILABLE, data=dict())


def _result(status: str, message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    return dict(status=status, message=message, data=data or {})


def _clamp_int(value: object, *, default: int, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        else:
            return default
    return max(low, min(int(value), high))


def _text(value: object, limit: int = MAX_QUERY_LENGTH) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value[: limit * 4].split())[:limit]


def _echo(value: object) -> str:
    """A bounded, single-line echo of what the user asked for, safe to speak."""
    return _text(value, _MAX_ECHO_LENGTH)


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


# --- spoken time ----------------------------------------------------------------------------


def _local(ts: int, zone: ZoneInfo) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(zone)


def _clock(moment: datetime) -> str:
    hour = moment.hour % 12 or 12
    return f"{hour}:{moment.minute:02d} {'AM' if moment.hour < 12 else 'PM'}"


def _day_phrase(day: datetime, today: datetime) -> str:
    """today, tomorrow, yesterday, a near weekday, or the weekday and date."""
    ahead = (day.date() - today.date()).days
    if ahead == 0:
        return "today"
    if ahead == 1:
        return "tomorrow"
    if ahead == -1:
        return "yesterday"
    name = day.strftime("%A")
    if 2 <= ahead <= 5:
        return name
    if -6 <= ahead <= -2:
        return "last " + name
    dated = f"{name}, {day.strftime('%B')} {day.day}"
    return dated if day.year == today.year else f"{dated}, {day.year}"


def _past_phrase(ts: int, now: int, zone: ZoneInfo) -> str:
    delta = max(0, now - ts)
    if delta < 60:
        return "just now"
    if delta < 3600:
        return _plural(delta // 60, "minute") + " ago"
    moment, today = _local(ts, zone), _local(now, zone)
    if delta < 12 * 3600 and moment.date() == today.date():
        return _plural(delta // 3600, "hour") + " ago"
    return f"{_day_phrase(moment, today)} at {_clock(moment)}"


def _event_phrase(row: IndexedEvent, now: int, zone: ZoneInfo) -> str:
    start, today = _local(row.start_ts, zone), _local(now, zone)
    if row.item.all_day:
        last = _local(row.end_ts - 1, zone)
        if last.date() > start.date():
            return f"{_day_phrase(start, today)} through {_day_phrase(last, today)}, all day"
        return f"{_day_phrase(start, today)}, all day"
    return f"{_day_phrase(start, today)} at {_clock(start)}"


def _stale_note(accounts: Sequence[AccountAnswer], now: int, zone: ZoneInfo) -> str:
    notes: list[str] = []
    for account in accounts:
        if not account.stale:
            continue
        who = f"your {_PROVIDER_NAMES.get(account.provider, account.provider)} account"
        # The name the user gave the account is what they call it out loud.
        named = account.user_label or account.account_label
        if named:
            who += f" {named}"
        problem = _PROBLEMS.get(account.status, _PROBLEMS["unavailable"])
        if account.indexed_at:
            since = _past_phrase(account.indexed_at, now, zone)
            if since.endswith("ago") or since == "just now":
                since = _clock(_local(account.indexed_at, zone))
            tail = f", so this is as of {since}"
        else:
            tail = ", and nothing from it is indexed"
        notes.append(f"Note: {who} {problem}{tail}.")
    return " ".join(notes)


def _account_views(accounts: Sequence[AccountAnswer]) -> list[dict[str, Any]]:
    return [
        dict(
            account=account.account_label,
            user_label=account.user_label,
            provider=account.provider,
            status=account.status,
            reason=account.reason,
            stale=account.stale,
            count=account.count,
        )
        for account in accounts
    ]


def _labels(accounts: Sequence[AccountAnswer]) -> dict[str, str | None]:
    return dict((account.connection_id, account.account_label) for account in accounts)


# --- email ------------------------------------------------------------------------------------


def _message_view(
    row: IndexedMessage, labels: dict[str, str | None], now: int, zone: ZoneInfo
) -> dict:
    item = row.item
    received = _received_ts(item.received_at)
    return dict(
        id=item.id,
        account=labels.get(row.connection_id),
        provider=row.provider,
        subject=item.subject,
        sender=item.sender,
        preview=item.preview,
        received_at=item.received_at,
        when=_past_phrase(received, now, zone) if received is not None else None,
        unread=item.unread,
        link=item.link,
    )


def _received_ts(value: str) -> int | None:
    from caal.knowledge_store import parse_instant

    return parse_instant(value)


def _message_phrase(row: IndexedMessage, now: int, zone: ZoneInfo) -> str:
    item = row.item
    who = item.sender or "someone"
    about = item.subject or "no subject"
    received = _received_ts(item.received_at)
    when = _past_phrase(received, now, zone) if received is not None else "at an unknown time"
    return f"{who} about {about}, {when}"


def _spoken_messages(rows: Sequence[IndexedMessage], now: int, zone: ZoneInfo) -> str:
    spoken = "; ".join(_message_phrase(row, now, zone) for row in rows[:SPOKEN_MESSAGES])
    more = len(rows) - SPOKEN_MESSAGES
    return spoken + (f"; and {_plural(more, 'more', 'more')}." if more > 0 else ".")


async def _gate(user_id: object) -> tuple[KnowledgeService | None, dict[str, Any] | None]:
    """The service to answer with, or the explicit answer for why there is none."""
    if user_id is None:
        return None, _result("unavailable", SESSION_UNAVAILABLE)
    service = _service()
    if service is None:
        return None, _result("unavailable", BACKEND_UNAVAILABLE)
    return service, None


def _no_match(account: object) -> dict[str, Any]:
    return _result("not_found", f"I could not find a connected account matching {_echo(account)}.")


def _ambiguous(account: object) -> dict[str, Any]:
    """Several accounts answer to that name, so none of them is read.

    Only the words the user themselves said come back; never a connection id
    and never the addresses of the accounts that matched.
    """
    return _result(
        "ambiguous",
        f"More than one of your connected accounts is called {_echo(account)}. "
        "Please tell me which one you mean, by its provider or by its email address.",
    )


async def recent_email(
    limit: object = DEFAULT_RECENT_LIMIT,
    unread_only: object = False,
    account: object = None,
    *,
    user_id: str | None = None,
) -> dict[str, Any]:
    """The newest mail across the connected accounts, as a short spoken summary."""
    service, refusal = await _gate(user_id)
    if service is None:
        return refusal or _result("unavailable", BACKEND_UNAVAILABLE)
    count = _clamp_int(limit, default=DEFAULT_RECENT_LIMIT, low=1, high=MAX_ANSWER_ITEMS)
    only_unread = unread_only is True
    hint = _text(account) or None
    answer = await service.recent_messages(
        user_id, limit=count, unread_only=only_unread, account=hint
    )
    if not answer.connected:
        return _result("no_accounts", NO_ACCOUNTS)
    if answer.ambiguous:
        return _ambiguous(hint)
    if not answer.accounts:
        return _no_match(hint)
    now, zone = service.now(), service.zone
    rows = answer.messages
    accounts = _plural(len(answer.accounts), "account")
    if not rows:
        kind = "unread" if only_unread else "recent"
        message = f"You have no {kind} email across your {accounts}."
    elif only_unread:
        message = f"You have {_plural(len(rows), 'unread email')} across {accounts}. Newest: "
        message += _spoken_messages(rows, now, zone)
    else:
        message = (
            f"You have {answer.unread_count} unread of {_plural(len(rows), 'recent email')} "
            f"across {accounts}. Newest: " + _spoken_messages(rows, now, zone)
        )
    note = _stale_note(answer.accounts, now, zone)
    labels = _labels(answer.accounts)
    return _result(
        "ok",
        (message + " " + note).strip(),
        dict(
            messages=[_message_view(row, labels, now, zone) for row in rows],
            unread_count=answer.unread_count,
            accounts=_account_views(answer.accounts),
            stale=answer.stale,
        ),
    )


# Words that describe *which* mail rather than what is in it. A request that
# names only these named no search target at all: "show unread email for my
# University account" is the recent-mail question, and a model that reaches for
# the search tool anyway is answering the same question the long way round.
_NOT_A_SEARCH_TARGET = frozenset(
    """
    email emails mail mails inbox inboxes mailbox message messages correspondence
    unread read new newest latest recent last first all any anything everything
    today tonight yesterday
    """.split()
)


def _search_target(words: str) -> bool:
    """Whether these words name something to search for, rather than a state."""
    terms = query_terms(words)
    return bool(terms) and not all(term in _NOT_A_SEARCH_TARGET for term in terms)


async def search_email(
    query: object = "",
    limit: object = DEFAULT_RECENT_LIMIT,
    account: object = None,
    *,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Recent mail matching a sender, subject or preview words.

    A "search" with no search target in it -- ``unread``, ``new email``, the
    name of an account -- is the recent-mail question asked with the wrong
    tool, and is answered as that rather than refused: the user asked for their
    unread mail, and the scope they named is carried across unchanged.
    """
    service, refusal = await _gate(user_id)
    if service is None:
        return refusal or _result("unavailable", BACKEND_UNAVAILABLE)
    words = _text(query)
    if not _search_target(words):
        if query_terms(words):
            return await recent_email(
                limit=limit,
                unread_only=any(term in ("unread",) for term in query_terms(words)),
                account=account,
                user_id=user_id,
            )
        return _result(
            "not_found", "Tell me a name, a subject or a few words to search your recent email for."
        )
    count = _clamp_int(limit, default=DEFAULT_RECENT_LIMIT, low=1, high=MAX_ANSWER_ITEMS)
    hint = _text(account) or None
    answer = await service.recent_messages(user_id, limit=count, query=words, account=hint)
    if not answer.connected:
        return _result("no_accounts", NO_ACCOUNTS)
    if answer.ambiguous:
        return _ambiguous(hint)
    if not answer.accounts:
        return _no_match(hint)
    now, zone = service.now(), service.zone
    note = _stale_note(answer.accounts, now, zone)
    labels = _labels(answer.accounts)
    data = dict(
        query=_echo(words),
        messages=[_message_view(row, labels, now, zone) for row in answer.messages],
        accounts=_account_views(answer.accounts),
        stale=answer.stale,
    )
    if not answer.messages:
        message = (
            f"I could not find any recent email matching {_echo(words)} in your connected accounts."
        )
        return _result("not_found", (message + " " + note).strip(), data)
    message = f"I found {_plural(len(answer.messages), 'email')} matching {_echo(words)}. "
    message += _spoken_messages(answer.messages, now, zone)
    return _result("ok", (message + " " + note).strip(), data)


async def read_email_summary(
    message_id: object = None,
    query: object = None,
    account: object = None,
    *,
    user_id: str | None = None,
) -> dict[str, Any]:
    """One email as a safe spoken summary: sender, subject, when, state, preview."""
    service, refusal = await _gate(user_id)
    if service is None:
        return refusal or _result("unavailable", BACKEND_UNAVAILABLE)
    hint = _text(account) or None
    chosen_id = _text(message_id, 512) or None
    words = _text(query) if chosen_id is None else ""
    answer = await service.recent_messages(
        user_id,
        limit=1,
        account=hint,
        message_id=chosen_id,
        query=words if query_terms(words) else None,
    )
    if not answer.connected:
        return _result("no_accounts", NO_ACCOUNTS)
    if answer.ambiguous:
        return _ambiguous(hint)
    if not answer.accounts:
        return _no_match(hint)
    now, zone = service.now(), service.zone
    note = _stale_note(answer.accounts, now, zone)
    if not answer.messages:
        what = f" matching {_echo(words)}" if words else ""
        message = f"I could not find that email{what} in your connected accounts."
        return _result("not_found", (message + " " + note).strip())
    row = answer.messages[0]
    item = row.item
    received = _received_ts(item.received_at)
    when = _past_phrase(received, now, zone) if received is not None else "at an unknown time"
    preview = (item.preview or "").strip()
    if preview:
        preview_line = "Preview: " + preview + ("" if preview[-1] in ".!?" else ".")
    else:
        preview_line = "There is no preview available for it."
    message = (
        f"Email from {item.sender or 'an unknown sender'}, subject {item.subject or 'no subject'}, "
        f"received {when}, {'unread' if item.unread else 'read'}. {preview_line}"
    )
    return _result(
        "ok",
        (message + " " + note).strip(),
        dict(
            message=_message_view(row, _labels(answer.accounts), now, zone),
            accounts=_account_views(answer.accounts),
            stale=answer.stale,
        ),
    )


# --- calendar ---------------------------------------------------------------------------------


def _at(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _iso(ts: int) -> str:
    return _at(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def _event_view(row: IndexedEvent, labels: dict[str, str | None], now: int, zone: ZoneInfo) -> dict:
    item = row.item
    return dict(
        id=item.id,
        account=labels.get(row.connection_id),
        provider=row.provider,
        title=item.title,
        start=item.start,
        end=item.end,
        all_day=item.all_day,
        location=item.location,
        when=_event_phrase(row, now, zone),
        status=item.status,
        link=item.link,
    )


def _event_line(row: IndexedEvent, now: int, zone: ZoneInfo) -> str:
    title = row.item.title or "an untitled event"
    place = f" at {row.item.location}" if row.item.location else ""
    return f"{_cap(_event_phrase(row, now, zone))}, {title}{place}"


def _spoken_events(rows: Sequence[IndexedEvent], total: int, now: int, zone: ZoneInfo) -> str:
    spoken = " ".join(_event_line(row, now, zone) + "." for row in rows[:SPOKEN_EVENTS])
    more = total - min(len(rows), SPOKEN_EVENTS)
    return spoken + (f" And {_plural(more, 'more', 'more')}." if more > 0 else "")


def _day_offset(value: object, today: datetime) -> int | None:
    """How many days from today a spoken day is; None when it is not a day at all."""
    from caal.knowledge import normalize_text

    text = normalize_text(value)
    if not text:
        return None
    if text in ("today", "tonight", "this morning", "this afternoon", "this evening"):
        return 0
    if text == "tomorrow":
        return 1
    if text in ("day after tomorrow", "the day after tomorrow"):
        return 2
    if text == "yesterday":
        return -1
    name = text.removeprefix("next ").removeprefix("this ").removeprefix("on ")
    if name in _WEEKDAYS:
        return (_WEEKDAYS.index(name) - today.weekday()) % 7
    raw = str(value).strip()[:10]
    try:
        wanted = datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return None
    return (wanted - today.date()).days


def _window(
    service: KnowledgeService, now: int, days: object, day: object, default_days: int
) -> tuple[int, int, str, int] | str:
    """[start, end) epoch seconds and the spoken scope, or an error message."""
    today = _local(now, service.zone)
    offset = _day_offset(day, today) if day is not None else None
    if offset is not None:
        if not -1 <= offset <= MAX_DAYS:
            return f"I can only look at your calendars from yesterday to {MAX_DAYS} days ahead."
        start = service.start_of_day(now, days_ahead=offset)
        end = service.start_of_day(now, days_ahead=offset + 1)
        phrase = _day_phrase(_local(start, service.zone), today)
        scope = phrase if phrase in ("today", "tomorrow", "yesterday") else "on " + phrase
        return start, end, scope, 1
    count = _clamp_int(days, default=default_days, low=1, high=MAX_DAYS)
    scope = "in the next 24 hours" if count == 1 else f"in the next {count} days"
    return now, now + count * 86400, scope, count


def _window_view(start: int, end: int, scope: str, days: int) -> dict[str, Any]:
    return dict(start=_iso(start), end=_iso(end), days=days, label=scope)


async def upcoming_schedule(
    days: object = None,
    day: object = None,
    limit: object = DEFAULT_SCHEDULE_LIMIT,
    account: object = None,
    only_future: object = False,
    *,
    user_id: str | None = None,
) -> dict[str, Any]:
    """The upcoming schedule across the connected calendars, as a short spoken summary.

    ``only_future`` drops whatever is already over: "what is left today" reads
    the rest of the day rather than the whole of it. Events are always given
    soonest first, and an event under way still counts as remaining.
    """
    service, refusal = await _gate(user_id)
    if service is None:
        return refusal or _result("unavailable", BACKEND_UNAVAILABLE)
    count = _clamp_int(limit, default=DEFAULT_SCHEDULE_LIMIT, low=1, high=MAX_ANSWER_ITEMS)
    now, zone = service.now(), service.zone
    window = _window(service, now, days, day, DEFAULT_SCHEDULE_DAYS)
    if isinstance(window, str):
        return _result("not_found", window)
    start, end, scope, span = window
    if only_future is True and start < now:
        # The window keeps its name; only what has already finished is dropped.
        start = now
        if not scope.startswith("in the next"):
            scope = "remaining " + scope
        if start >= end:
            return _result(
                "ok",
                f"You have nothing on your connected calendars {scope}.",
                dict(events=[], count=0, window=_window_view(now, end, scope, span)),
            )
    hint = _text(account) or None
    answer = await service.upcoming_events(
        user_id, start=_at(start), end=_at(end), limit=count, account=hint
    )
    if not answer.connected:
        return _result("no_accounts", NO_ACCOUNTS)
    if answer.ambiguous:
        return _ambiguous(hint)
    if not answer.accounts:
        return _no_match(hint)
    rows = answer.events
    total = sum(item.count for item in answer.accounts)
    if not rows:
        message = f"You have nothing on your connected calendars {scope}."
    else:
        message = f"You have {_plural(total, 'event')} {scope}. " + _spoken_events(
            rows, total, now, zone
        )
    note = _stale_note(answer.accounts, now, zone)
    labels = _labels(answer.accounts)
    return _result(
        "ok",
        (message + " " + note).strip(),
        dict(
            events=[_event_view(row, labels, now, zone) for row in rows],
            count=total,
            window=_window_view(start, end, scope, span),
            accounts=_account_views(answer.accounts),
            stale=answer.stale,
        ),
    )


async def next_events(
    limit: object = DEFAULT_NEXT_LIMIT,
    days: object = DEFAULT_NEXT_DAYS,
    account: object = None,
    *,
    user_id: str | None = None,
) -> dict[str, Any]:
    """The event that starts next on the connected calendars, or the next few.

    Nearest means what starts next: nothing in the past, and nothing already
    under way, is the answer to "what is my next event". The events come back
    soonest first, and one of them unless the caller asked for more.
    """
    service, refusal = await _gate(user_id)
    if service is None:
        return refusal or _result("unavailable", BACKEND_UNAVAILABLE)
    count = _clamp_int(limit, default=DEFAULT_NEXT_LIMIT, low=1, high=MAX_NEXT_LIMIT)
    span = _clamp_int(days, default=DEFAULT_NEXT_DAYS, low=1, high=MAX_DAYS)
    now, zone = service.now(), service.zone
    end = now + span * 86400
    hint = _text(account) or None
    answer = await service.upcoming_events(
        user_id, start=_at(now), end=_at(end), limit=MAX_ANSWER_ITEMS, account=hint
    )
    if not answer.connected:
        return _result("no_accounts", NO_ACCOUNTS)
    if answer.ambiguous:
        return _ambiguous(hint)
    if not answer.accounts:
        return _no_match(hint)
    rows = [row for row in answer.events if row.start_ts >= now][:count]
    note = _stale_note(answer.accounts, now, zone)
    labels = _labels(answer.accounts)
    horizon = "the next 24 hours" if span == 1 else f"the next {span} days"
    if not rows:
        message = f"You have nothing coming up on your connected calendars in {horizon}."
    elif len(rows) == 1:
        row = rows[0]
        place = f" at {row.item.location}" if row.item.location else ""
        title = row.item.title or "an untitled event"
        message = f"Your next event is {title}, {_event_phrase(row, now, zone)}{place}."
    else:
        counted = _plural(len(rows), "event")
        message = "Your next " + counted + ": " + _spoken_events(rows, len(rows), now, zone)
    return _result(
        "ok",
        (message + " " + note).strip(),
        dict(
            events=[_event_view(row, labels, now, zone) for row in rows],
            count=len(rows),
            window=_window_view(now, end, "in " + horizon, span),
            accounts=_account_views(answer.accounts),
            stale=answer.stale,
        ),
    )


async def find_event(
    query: object = "",
    days: object = DEFAULT_FIND_DAYS,
    day: object = None,
    account: object = None,
    *,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Whether an event matching a few words exists on the connected calendars, and when."""
    service, refusal = await _gate(user_id)
    if service is None:
        return refusal or _result("unavailable", BACKEND_UNAVAILABLE)
    words = _text(query)
    if not query_terms(words):
        return _result(
            "not_found", "Tell me a few words from the title of the event you want me to look for."
        )
    now, zone = service.now(), service.zone
    window = _window(service, now, days, day, DEFAULT_FIND_DAYS)
    if isinstance(window, str):
        return _result("not_found", window)
    start, end, scope, span = window
    hint = _text(account) or None
    answer = await service.upcoming_events(
        user_id, start=_at(start), end=_at(end), limit=MAX_ANSWER_ITEMS, query=words, account=hint
    )
    if not answer.connected:
        return _result("no_accounts", NO_ACCOUNTS)
    if answer.ambiguous:
        return _ambiguous(hint)
    if not answer.accounts:
        return _no_match(hint)
    rows = answer.events
    note = _stale_note(answer.accounts, now, zone)
    labels = _labels(answer.accounts)
    data = dict(
        exists=bool(rows),
        query=_echo(words),
        events=[_event_view(row, labels, now, zone) for row in rows],
        window=_window_view(start, end, scope, span),
        accounts=_account_views(answer.accounts),
        stale=answer.stale,
    )
    if not rows:
        message = (
            f"No. I do not see any event matching {_echo(words)} {scope} "
            "on your connected calendars."
        )
        return _result("not_found", (message + " " + note).strip(), data)
    if len(rows) == 1:
        row = rows[0]
        place = f" at {row.item.location}" if row.item.location else ""
        title = row.item.title or "An untitled event"
        message = f"Yes. {title} is {_event_phrase(row, now, zone)}{place}."
    else:
        message = f"Yes. I see {_plural(len(rows), 'event')} matching {_echo(words)}. "
        message += _spoken_events(rows, len(rows), now, zone)
    return _result("ok", (message + " " + note).strip(), data)
