"""Persistent local reminders, owned by one user each.

Two kinds of reminder exist here and they are described to the user as what
they are:

* a **timed** reminder, which is delivered once on each of the channels its
  owner chose -- spoken in a voice session of the same user, sent to the
  Telegram chat bound to that one profile, or placed as a call to the number
  already approved on it. Speech is still the default and still the only
  channel a live room can serve; the rest are carried out by the durable
  worker, so they survive the room closing. See :mod:`caal.tools.reminder_delivery`;
* an **undated** reminder, which is a persistent list item. It is never
  announced and the reply never claims that it will be.

This is the local CAAL store. It is not Apple Reminders and nothing here writes
to any external service.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from caal.tools import alarms_tools, reminder_delivery
from caal.tools.errors import SafeToolError, safe_error_result
from caal.tools.scheduled_time import WHEN_HINT, parse_when
from caal.user_scope import require_user_id

logger = logging.getLogger(__name__)

STORE_PATH = Path(os.getenv("CAAL_DATA_DIR", "/app/data")) / "assistant.sqlite3"

LEGACY_SCOPE = ""
MAX_TITLE_LENGTH = 200
MAX_NOTES_LENGTH = 2000
MAX_REMINDERS_LISTED = 200
_BUSY_TIMEOUT_SECONDS = 5.0

_CREATE = """
    CREATE TABLE IF NOT EXISTS {name} (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL DEFAULT '',
        title TEXT NOT NULL,
        due TEXT,
        due_at INTEGER,
        list_name TEXT NOT NULL,
        notes TEXT NOT NULL,
        completed INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    )
"""


def _result(message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"status": "ok", "message": message, "data": data or dict()}


def _connect() -> sqlite3.Connection:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(STORE_PATH, timeout=_BUSY_TIMEOUT_SECONDS, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout = {int(_BUSY_TIMEOUT_SECONDS * 1000)}")
    _ensure_schema(connection)
    return connection


def _columns(connection: sqlite3.Connection) -> set[str]:
    return {row["name"] for row in connection.execute("PRAGMA table_info(reminders)")}


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """Create the owned table, keeping pre-multi-user rows in the legacy scope."""
    columns = _columns(connection)
    if columns and {"user_id", "due_at"} <= columns:
        return
    connection.execute("BEGIN IMMEDIATE")
    try:
        columns = _columns(connection)  # re-check under the write lock
        if not columns:
            connection.execute(_CREATE.format(name="reminders"))
        elif not {"user_id", "due_at"} <= columns:
            connection.execute(_CREATE.format(name="reminders_owned"))
            connection.execute(
                "INSERT INTO reminders_owned (id,user_id,title,due,due_at,list_name,notes,"
                "completed,created_at) SELECT id,'',title,due,NULL,list_name,notes,completed,"
                "created_at FROM reminders"
            )
            connection.execute("DROP TABLE reminders")
            connection.execute("ALTER TABLE reminders_owned RENAME TO reminders")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS reminders_scope ON reminders (user_id, completed, due)"
        )
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise


def _scope(user_id: object) -> str:
    if user_id is None:
        return LEGACY_SCOPE
    return require_user_id(user_id)


def _clean(value: object, name: str, limit: int, required: bool = True) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise SafeToolError(f"The reminder {name} has to be something I can read back.")
    cleaned = " ".join(value.split())[:limit]
    if required and not cleaned:
        raise SafeToolError("I need a few words for the reminder itself.")
    return cleaned


# --- how a reminder is delivered, in words --------------------------------------------------

_OFFERS = {
    reminder_delivery.SPEAK: "say it here",
    reminder_delivery.TELEGRAM: "send it to your Telegram",
    reminder_delivery.CALL: "call you",
}
_PROMISES = {
    reminder_delivery.SPEAK: "say it out loud when you are here with me",
    reminder_delivery.TELEGRAM: "send it to your Telegram",
    reminder_delivery.CALL: "call you",
}
SPOKEN_ONLY_PROMISE = "I will say it once when you are here with me."


def _join(phrases: list[str]) -> str:
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def promise(channels: tuple[str, ...]) -> str:
    """What will actually happen, said plainly and only about channels that are armed."""
    if channels == (reminder_delivery.SPEAK,):
        return SPOKEN_ONLY_PROMISE
    return "When it comes due I will " + _join([_PROMISES[c] for c in channels]) + "."


NOTHING_OFFER = "do nothing about it"
NO_DELIVERY_PROMISE = (
    "I will not tell you about it when it comes due. It stays on your list, "
    "and you can ask me to change that any time."
)


def _question(available: tuple[str, ...]) -> str:
    """The one choice question. There is always a choice, because doing nothing is one.

    Every channel this owner may actually use is named, and so is the option of
    no alert at all. Saying no is a decision a person is allowed to make out
    loud, so it is offered in the same breath as the rest rather than left as
    something they have to think to ask for.
    """
    offers = [_OFFERS[c] for c in available]
    question = (
        "When it comes due, do you want me to " + ", ".join([*offers, "or " + NOTHING_OFFER]) + "?"
    )
    if len(offers) > 1:
        question += " Any combination of those is fine."
    return question


def _offered(user_id: object) -> tuple[str, ...]:
    availability = reminder_delivery.channel_availability(user_id)
    return tuple(c for c in reminder_delivery.CHANNELS if availability.get(c))


def _selected(
    scope: object, requested: tuple[str, ...] | None
) -> tuple[tuple[str, ...], list[str], bool]:
    """The channels to arm, what has to be said about the refused ones, and whether to ask.

    A caller who named channels is never asked again -- including the caller
    who named ``none``, which is a channel selection of nothing at all. A caller
    who did not gets the spoken channel armed provisionally and the question,
    always: however few channels this owner has, no alert is always the other
    answer, so there is always something to choose.

    A saved dashboard default is deliberately not consulted here. It is a
    preference someone set once in a browser; a reminder asked for out loud with
    no channel named is not consent to send a message or place a call, and a
    silently applied telegram or call is exactly the kind of thing that must
    never happen without the person saying so in that conversation. So the
    saved value stays what the dashboard shows and what the dashboard writes,
    and this asks. The one-word answer to the question is handled in
    :func:`caal.tools.reminder_delivery.parse_channels`, where ``default`` means
    the spoken channel alone rather than whatever the browser has stored.
    """
    asking = False
    if requested == ():
        # They said "nothing". That is an answer, so there is nothing to ask.
        return (), [], False
    if requested is None:
        requested = reminder_delivery.DEFAULT_CHANNELS
        # Always: doing nothing is always the other option, so a person who did
        # not say is always being offered a real choice. The spoken channel is
        # armed meanwhile, so a due time cannot pass in silence while they think.
        asking = True
    permitted, refused = reminder_delivery.allowed(scope, requested)
    if not permitted:
        # Truthful fallback: the reminder still exists and still reaches them
        # the way it always did, and the refusal is said out loud.
        permitted = (reminder_delivery.SPEAK,)
    return permitted, refused, asking


def _arm(
    reminder_id: str,
    scope: object,
    user_id: str | None,
    title: str,
    permitted: tuple[str, ...],
    due_at: int,
    now: int | None,
) -> tuple[str, ...]:
    """Schedule the alarm behind the spoken channel, then record every channel.

    The alarm row is written first and its id is carried into the ledger, so
    the spoken channel is settled by the session that actually said the words
    and by nothing else. A refused alarm drops only the spoken channel.
    """
    alarm_id: str | None = None
    if reminder_delivery.SPEAK in permitted:
        alarm_id = str(uuid.uuid4())
        try:
            alarms_tools.schedule(
                title, due_at, "reminder", user_id=user_id, now=now, alarm_id=alarm_id
            )
        except SafeToolError:
            logger.info("Stored a reminder whose spoken channel could not be scheduled")
            permitted = tuple(c for c in permitted if c != reminder_delivery.SPEAK)
            alarm_id = None
    reminder_delivery.arm(reminder_id, scope, permitted, due_at, now=now, alarm_id=alarm_id)
    return permitted


def create_reminder(
    title: str,
    due: str | None = None,
    list_name: str | None = None,
    list: str | None = None,
    notes: str | None = None,
    delivery: list[str] | None = None,
    user_id: str | None = None,
    now: int | None = None,
) -> dict[str, Any]:
    """Create a reminder in the local CAAL store of this user.

    With a ``due`` time it is also delivered, once on each channel its owner
    chose. Without one it is a list item: nothing is armed, nothing is sent,
    and the reply says so rather than promising an alert -- even when channels
    were asked for, because there is no moment for them to happen at.

    ``delivery`` is a bounded enum and the only thing a caller may choose. The
    owner comes from the verified session, and every destination behind a
    channel is resolved server-side.
    """
    try:
        scope = _scope(user_id)
        clean_title = _clean(title, "title", MAX_TITLE_LENGTH)
        clean_notes = _clean(notes, "notes", MAX_NOTES_LENGTH, required=False)
        clean_list = _clean(list or list_name or "Reminders", "list", 80, required=False)
        requested = None if delivery is None else reminder_delivery.parse_channels(delivery)
        due_at: int | None = None
        if isinstance(due, str) and due.strip():
            current = int(datetime.now(timezone.utc).timestamp()) if now is None else int(now)
            due_at = parse_when(due, current)
    except SafeToolError as error:
        logger.info("Refused a reminder request: %s", type(error).__name__)
        return safe_error_result(error, status="invalid_request")

    reminder = dict(
        id=str(uuid.uuid4()),
        user_id=scope,
        title=clean_title,
        due=None if due_at is None else _iso(due_at),
        due_at=due_at,
        list=clean_list or "Reminders",
        notes=clean_notes,
        completed=False,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    with closing(_connect()) as connection:
        connection.execute(
            "INSERT INTO reminders (id,user_id,title,due,due_at,list_name,notes,completed,"
            "created_at) VALUES (:id,:user_id,:title,:due,:due_at,:list,:notes,:completed,"
            ":created_at)",
            reminder,
        )
    data = dict(
        title=reminder["title"],
        due=reminder["due"],
        list=reminder["list"],
        timed=due_at is not None,
        delivery=[],
        delivery_pending=False,
    )
    if due_at is None:
        logger.info("Stored an undated reminder")
        extra = (
            " Nothing will be sent or called about it either."
            if requested
            else ""
        )
        return _result(
            f"Added to your {reminder['list']} list: {reminder['title']}. "
            "There is no time on it, so I will not alert you." + extra + " Tell me a clear "
            "time if you wanted an alert, or ask me for the list any time.",
            data,
        )
    # Only after the reminder itself is stored: delivery is the second half of
    # the same promise, and the reply below is the only thing that claims it.
    permitted, refused, asking = _selected(scope, requested)
    if permitted:
        permitted = _arm(reminder["id"], scope, user_id, reminder["title"], permitted, due_at, now)
    data["delivery"] = [*permitted]
    data["delivery_pending"] = bool(asking)
    # An unanswered question is bound to the reminder it was asked about, so a
    # two-word answer on the next turn can be read as one without guessing.
    # A caller who did choose closes any question still open for this owner.
    if asking and permitted:
        reminder_delivery.await_answer(reminder["id"], scope, due_at, now=now)
    else:
        reminder_delivery.clear_awaiting(scope)
    delay = alarms_tools.describe_delay(
        due_at - (int(now) if now is not None else int(datetime.now(timezone.utc).timestamp()))
    )
    if not permitted and requested == ():
        # They asked for no alert, and that is exactly what they get: a reminder
        # that is on the list, at a time, and silent.
        logger.info("Stored a timed reminder its owner asked not to be told about")
        return _result(f"Reminder set {delay}: {reminder['title']}. " + NO_DELIVERY_PROMISE, data)
    if not permitted:
        logger.info("Stored a reminder with no channel armed")
        return _result(
            f"Added to your {reminder['list']} list: {reminder['title']}. "
            "I could not arm any way of telling you, so I will not be able to remind you.",
            data | dict(timed=False),
        )
    logger.info("Stored a timed reminder on %d channel(s)", len(permitted))
    tail = _question(_offered(scope)) if asking else promise(permitted)
    sentences = [f"Reminder set {delay}: {reminder['title']}."] + refused + [tail]
    return _result(" ".join(sentences), data)


def set_delivery(
    delivery: list[str],
    user_id: str | None = None,
    now: int | None = None,
) -> dict[str, Any]:
    """Choose the channels of the reminder this user most recently set.

    This is the answer to the question :func:`create_reminder` asks. There is
    no reminder id anywhere in it: the target is resolved from the verified
    owner own most recent timed reminder that has not come due, so a caller
    can never reach a reminder that is not theirs by naming one.

    ``["none"]`` is the answer "do nothing about it": it cancels every channel
    of that one reminder and returns an honest no-delivery result. It never
    deletes the reminder, and it can no more reach somebody else reminder than
    any other answer can.
    """
    try:
        scope = _scope(user_id)
        chosen = reminder_delivery.parse_channels(delivery)
    except SafeToolError as error:
        logger.info("Refused a delivery choice: %s", type(error).__name__)
        return safe_error_result(error, status="invalid_request")

    current = int(datetime.now(timezone.utc).timestamp()) if now is None else int(now)
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT id, title, due, due_at FROM reminders WHERE user_id = ? AND completed = 0 "
            "AND due_at IS NOT NULL AND due_at > ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (scope, current),
        ).fetchone()
    if row is None:
        logger.info("No timed reminder of this owner to change the delivery of")
        return safe_error_result(
            SafeToolError(
                "I do not have a reminder of yours still to come, so there is nothing to change."
            ),
            status="invalid_request",
        )

    if chosen == ():
        # "Nothing." Every channel of this one reminder is stood down, the
        # reminder itself is left exactly where it is, and the reply says so
        # rather than quietly leaving something armed.
        armed = reminder_delivery.channels_of(row["id"])
        if reminder_delivery.SPEAK in armed:
            alarms_tools.cancel_pending(
                reminder_delivery.alarm_of(row["id"]) or "", user_id=user_id
            )
        reminder_delivery.cancel(row["id"], armed)
        reminder_delivery.clear_awaiting(scope)
        logger.info("Stood down %d channel(s) at the request of their owner", len(armed))
        return _result(
            NO_DELIVERY_PROMISE,
            dict(title=row["title"], due=row["due"], delivery=[], delivery_pending=False),
        )

    permitted, refused = reminder_delivery.allowed(scope, chosen)
    if not permitted:
        permitted = (reminder_delivery.SPEAK,)
    already = reminder_delivery.channels_of(row["id"])
    dropped = tuple(c for c in already if c not in permitted)
    if reminder_delivery.SPEAK in dropped:
        alarms_tools.cancel_pending(reminder_delivery.alarm_of(row["id"]) or "", user_id=user_id)
    reminder_delivery.cancel(row["id"], dropped)
    added = tuple(c for c in permitted if c not in already)
    if added:
        _arm(row["id"], scope, user_id, row["title"], added, int(row["due_at"]), now)
    live = reminder_delivery.channels_of(row["id"])
    reminder_delivery.clear_awaiting(scope)
    logger.info("Changed a reminder to %d channel(s)", len(live))
    data = dict(title=row["title"], due=row["due"], delivery=list(live), delivery_pending=False)
    return _result(" ".join([*refused, promise(live)]), data)


def _iso(due_at: int) -> str:
    return datetime.fromtimestamp(due_at, tz=timezone.utc).isoformat()


def _rows(scope: str, include_completed: bool) -> list[Any]:
    query = (
        "SELECT id, title, due, due_at, list_name, notes, completed, created_at FROM reminders "
        "WHERE user_id = ?"
    )
    if not include_completed:
        query += " AND completed = 0"
    query += " ORDER BY due IS NULL, due, created_at LIMIT ?"
    with closing(_connect()) as connection:
        return connection.execute(query, (scope, MAX_REMINDERS_LISTED)).fetchall()


def list_reminders(
    include_completed: bool = False,
    user_id: str | None = None,
) -> dict[str, Any]:
    """List the local reminders of this user, soonest due first, undated last."""
    scope = _scope(user_id)
    rows = _rows(scope, include_completed)
    states = reminder_delivery.states_of([row["id"] for row in rows])
    reminders = [
        dict(
            title=row["title"],
            due=row["due"],
            list=row["list_name"],
            notes=row["notes"],
            completed=bool(row["completed"]),
            timed=row["due"] is not None,
            created_at=row["created_at"],
            delivery=[
                channel
                for channel in reminder_delivery.CHANNELS
                if channel in states.get(row["id"], {})
            ],
        )
        for row in rows
    ]
    count = len(reminders)
    return _result(
        f"Found {count} {'reminder' if count == 1 else 'reminders'}.",
        dict(reminders=reminders),
    )


def announcement_for(reminder_id: object, user_id: str | None = None) -> str | None:
    """The words of one reminder of this owner, for a call that has been answered.

    Scoped to the owner the call was placed for, so an answered leg can only
    ever speak that owner own reminder, and only when the row is still there.
    """
    if not isinstance(reminder_id, str) or not reminder_id:
        return None
    scope = _scope(user_id)
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT title FROM reminders WHERE id = ? AND user_id = ?", (reminder_id, scope)
        ).fetchone()
    if row is None:
        logger.info("An answered reminder call found no reminder of its own owner")
        return None
    return reminder_delivery.reminder_message(row["title"])


def dashboard_reminders(user_id: str | None = None, include_completed: bool = False) -> list[dict]:
    """The owner own reminders, as the authenticated dashboard shows them.

    Every field is the owner own words or a bounded state; the opaque row id
    travels so a reader can key a list on it, and nothing else leaves. This is
    a read: it arms nothing, sends nothing, and settles nothing.
    """
    scope = _scope(user_id)
    rows = _rows(scope, include_completed)
    states = reminder_delivery.states_of([row["id"] for row in rows])
    return [
        dict(
            id=row["id"],
            title=row["title"],
            due=row["due"],
            timed=row["due"] is not None,
            list=row["list_name"],
            notes=row["notes"],
            completed=bool(row["completed"]),
            created_at=row["created_at"],
            delivery=[
                dict(channel=channel, state=states.get(row["id"], {})[channel])
                for channel in reminder_delivery.CHANNELS
                if channel in states.get(row["id"], {})
            ],
        )
        for row in rows
    ]


DUE_DESCRIPTION = WHEN_HINT
DELIVERY_DESCRIPTION = (
    "How the user wants to be told when it comes due, as any combination of "
    "speak (out loud in a live session), telegram (a message to their authorised "
    "Telegram) and call (a call to the number approved on their profile); all means "
    "every one of those, and default means only speak. none means they said they want "
    "no alert at all -- nothing, no notification, do not tell me, just put it on the "
    "list -- and it cancels every channel while keeping the reminder; it cannot be "
    "combined with a channel. Leave it out when the user did not say, and CAAL asks "
    "them which ways they want, including doing nothing."
)
