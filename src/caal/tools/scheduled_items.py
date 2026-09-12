"""Changing a scheduled item that already exists, for its owner and nobody else.

CAAL could set an alarm, a timer and a timed reminder. It could not change one.
Asked to turn an alarm into a reminder it searched the schedule, found the
alarm, and then reached for the only scheduled write it had -- so a *second*
alarm appeared while the reply sounded like something had been converted.

This is the write that was missing. One bounded surface, three actions:

* **cancel**  the item stops existing, and every pending delivery row of it
  goes with it, so nothing is left for the durable worker to call or message
  about;
* **update**  a new time, a new title, or both. Everything not named is kept;
* **convert** an alarm or a timer becomes a reminder, or a reminder becomes an
  alarm or a timer. The source is removed and the target written inside one
  transaction, so there is never a moment with both, and never a moment with
  neither.

The safety line is the same one every scheduled tool here holds:

* a caller names an *action*, a natural *reference*, and at most a kind, a new
  time, a new title and a conversion target. There is no row id, no owner id,
  no channel, no destination and no column name anywhere in the surface, so
  none of them is smuggleable;
* the owner comes from the verified session and is applied to every statement.
  A reference resolves only among the pending items of that one owner;
* a delivered item, an overdue one and a completed one are not candidates at
  all. What already happened is not editable here;
* a reference that matches nothing, or matches several things, is a refusal in
  plain words that writes nothing.

Nothing here logs a label, a title, a row id or an owner.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from caal.tools import alarms_tools, reminder_delivery, reminders_tools
from caal.tools.errors import SafeToolError, safe_error_result
from caal.tools.scheduled_time import WHEN_HINT, parse_when
from caal.user_scope import require_user_id, scheduling_unavailable_result

logger = logging.getLogger(__name__)

STORE_PATH = Path(os.getenv("CAAL_DATA_DIR", "/app/data")) / "assistant.sqlite3"

#: The one native tool name this write is exposed under.
CHANGE_TOOL = "scheduled.change"

ACTIONS: tuple[str, ...] = ("cancel", "update", "convert")
#: What a person calls the thing they want changed, and what they want it to
#: become. Deliberately the words they say, never a table or a column.
KINDS: tuple[str, ...] = ("reminder", "alarm", "timer")

ALARM = "alarm"
TIMER = "timer"
REMINDER = "reminder"
#: Which store a candidate lives in. Never a caller argument.
ALARM_SOURCE = "alarm"
REMINDER_SOURCE = "reminder"

MAX_CANDIDATES = 12
_BUSY_TIMEOUT_SECONDS = 5.0
_OK = "ok"


REFERENCE_DESCRIPTION = (
    "How the user referred to the item, in their own words: that one, my last timer, "
    "the laundry reminder, the standup alarm. Pass what they said and nothing else. "
    "Leave it out when they plainly meant the one they just set."
)
WHEN_CHANGE_DESCRIPTION = "The new time, if they gave one. " + WHEN_HINT
TITLE_CHANGE_DESCRIPTION = (
    "The new wording for it, if they gave one, in their own words and with no time in it."
)
KIND_DESCRIPTION = (
    "What the user called the thing they want changed, when they said: reminder, alarm "
    "or timer. Leave it out when they did not say."
)
TARGET_KIND_DESCRIPTION = (
    "For convert only: what they want it to become. reminder, alarm or timer."
)

CHANGE_SCHEMA: dict[str, Any] = dict(
    type="object",
    properties=dict(
        action=dict(
            type="string",
            enum=list(ACTIONS),
            description=(
                "cancel to stop it happening at all, update to change its time or its "
                "wording, convert to make it a different kind of thing."
            ),
        ),
        reference=dict(type="string", description=REFERENCE_DESCRIPTION),
        kind=dict(type="string", enum=list(KINDS), description=KIND_DESCRIPTION),
        when=dict(type="string", description=WHEN_CHANGE_DESCRIPTION),
        title=dict(type="string", description=TITLE_CHANGE_DESCRIPTION),
        target_kind=dict(type="string", enum=list(KINDS), description=TARGET_KIND_DESCRIPTION),
    ),
    required=["action"],
    additionalProperties=False,
)

CHANGE_DESCRIPTION = (
    "Change a local alarm, timer or reminder the signed-in user already has. Use for: "
    "cancel that alarm, delete my last timer, move my reminder to in an hour, rename the "
    "alarm, make that alarm a reminder instead, turn the reminder into a timer. Never "
    "answer one of those by setting a new one. It only ever reaches the items of the "
    "signed-in user that are still to come, and there is no way to name a row, an owner, "
    "a phone number or a chat."
)


# --- the store --------------------------------------------------------------------------------


def _connect() -> sqlite3.Connection:
    """One connection over the whole scheduled slice of the shared store.

    A conversion touches three tables at once, so it needs one transaction over
    all three rather than three modules each opening their own. The schemas
    still belong to the modules that own them, so they are created by those
    modules first and only then written through here.
    """
    for module in (alarms_tools, reminders_tools, reminder_delivery):
        with closing(module._connect()):
            pass
    connection = sqlite3.connect(STORE_PATH, timeout=_BUSY_TIMEOUT_SECONDS, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = " + str(int(_BUSY_TIMEOUT_SECONDS * 1000)))
    return connection


def _scope(user_id: object) -> str:
    if user_id is None:
        return ""
    return require_user_id(user_id)


def _now() -> int:
    import time

    return int(time.time())


def _iso(due_at: int) -> str:
    return datetime.fromtimestamp(int(due_at), tz=timezone.utc).isoformat()


def _created_epoch(value: object) -> int:
    """The moment a reminder row was written, as a number it can be sorted by."""
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(datetime.fromisoformat(str(value)).timestamp())
    except (TypeError, ValueError):
        return 0


# --- what there is to change ---------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One pending scheduled item of one owner.

    ``row_id`` never leaves this module: it is not returned to a caller, shown
    to a model, or written to a log. ``title`` is the words of the owner and
    goes back only to the session that owns them.
    """

    source: str
    row_id: str
    kind: str
    title: str
    due_at: int | None
    created_at: int


def candidates(
    user_id: object, now: int | None = None, kind: str | None = None
) -> list[Candidate]:
    """The pending items of this one owner, newest first.

    Pending means exactly that: not delivered, not completed, and not already
    past. An undated reminder is pending too -- it is a live list item and can
    be renamed or cancelled -- and it simply has no due time.
    """
    scope = _scope(user_id)
    current = _now() if now is None else int(now)
    wanted = None if kind is None else str(kind).strip().lower()
    if wanted is not None and wanted not in KINDS:
        raise SafeToolError("I can change an alarm, a timer or a reminder.")
    found: list[Candidate] = []
    with closing(_connect()) as connection:
        if wanted != REMINDER:
            rows = connection.execute(
                "SELECT id, kind, label, due_at, created_at FROM alarms WHERE user_id = ? "
                "AND kind IN (?,?) AND delivered_at IS NULL AND fired_at IS NULL "
                "AND due_at > ? ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (scope, ALARM, TIMER, current, MAX_CANDIDATES),
            ).fetchall()
            found.extend(
                Candidate(
                    source=ALARM_SOURCE,
                    row_id=row["id"],
                    kind=row["kind"],
                    title=row["label"],
                    due_at=int(row["due_at"]),
                    created_at=int(row["created_at"]),
                )
                for row in rows
                if wanted is None or row["kind"] == wanted
            )
        if wanted in (None, REMINDER):
            rows = connection.execute(
                "SELECT id, title, due_at, created_at FROM reminders WHERE user_id = ? "
                "AND completed = 0 AND (due_at IS NULL OR due_at > ?) "
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (scope, current, MAX_CANDIDATES),
            ).fetchall()
            found.extend(
                Candidate(
                    source=REMINDER_SOURCE,
                    row_id=row["id"],
                    kind=REMINDER,
                    title=row["title"],
                    due_at=None if row["due_at"] is None else int(row["due_at"]),
                    created_at=_created_epoch(row["created_at"]),
                )
                for row in rows
            )
    found.sort(key=lambda item: item.created_at, reverse=True)
    return found[:MAX_CANDIDATES]


def summarize(items: list[Candidate]) -> list[dict[str, Any]]:
    """The bounded description a local reading may be shown, numbered for one call.

    Their own words, what kind of thing it is, and how far off it is. No row
    id, no owner, no channel, no destination, no absolute clock time.
    """
    summary: list[dict[str, Any]] = []
    for number, item in enumerate(items, start=1):
        when = "no time on it"
        if item.due_at is not None:
            when = alarms_tools.describe_delay(item.due_at - _now())
        summary.append(dict(n=number, kind=item.kind, what=item.title, due=when))
    return summary


def summarize_at(items: list[Candidate], now: int) -> list[dict[str, Any]]:
    """:func:`summarize` measured from a given moment rather than from the clock."""
    return [
        dict(
            n=number,
            kind=item.kind,
            what=item.title,
            due=(
                "no time on it"
                if item.due_at is None
                else alarms_tools.describe_delay(item.due_at - int(now))
            ),
        )
        for number, item in enumerate(items, start=1)
    ]


# --- which one they meant -----------------------------------------------------------------


#: Words that mean "the one we were just talking about". Saying any of these is
#: not naming an item, so the newest one is the only reading of them.
_DEICTIC = frozenset(
    "that this it one last latest newest recent recently just now current".split()
)

#: Words that carry no identity: the kinds themselves, and ordinary glue. What
#: is left after these is what the person actually named the item by.
_NOT_IDENTIFYING = frozenset(
    """
    a an the my mine me i our we your
    alarm alarms timer timers reminder reminders alert alerts
    of for about to at on in with and or please
    set setting scheduled pending upcoming next thing item
    """.split()
) | _DEICTIC

NOTHING_PENDING = (
    "I do not have an alarm, a timer or a reminder of yours still to come, so there is "
    "nothing for me to change."
)
NO_MATCH = (
    "I could not find one of yours that matches, so I have not changed anything. "
    "Tell me which one you mean."
)
AMBIGUOUS = (
    "I have more than one that could be, so I have not changed anything. Which one did "
    "you mean?"
)


def _words(text: object) -> list[str]:
    if not isinstance(text, str):
        return []
    return [
        "".join(character for character in part if character.isalnum())
        for part in text.lower().split()
    ]


def _identifying(reference: object) -> list[str]:
    """The words of a reference that actually pick something out."""
    return [word for word in _words(reference) if word and word not in _NOT_IDENTIFYING]


def _points_at_the_newest(reference: object) -> bool:
    """Whether the reference says only "the one we were just doing"."""
    words = [word for word in _words(reference) if word]
    if not words:
        # Nothing said at all: the one they just set is the only reading.
        return True
    return any(word in _DEICTIC for word in words)


#: Public name for the reading above: a caller outside this module asks it
#: whether a whole turn pointed at the item that was set most recently.
def points_at_the_newest(reference: object) -> bool:
    """Whether a reference says only "the one we were just doing"."""
    return _points_at_the_newest(reference)


def resolve(
    user_id: object, now: int, reference: object = None, kind: str | None = None
) -> Candidate:
    """The one pending item of this owner that a natural reference names.

    Newest for "that one". For a named reference, the unique item whose own
    words carry every identifying word of the reference, and failing that the
    unique one that shares any of them. Zero matches and several matches are
    both refusals: guessing here would cancel or move the wrong thing.
    """
    items = candidates(user_id, now, kind=kind)
    if not items:
        raise SafeToolError(NOTHING_PENDING)
    wanted = _identifying(reference)
    if not wanted:
        if _points_at_the_newest(reference) or len(items) == 1:
            return items[0]
        # "the alarm", with two of them. Naming a kind is not naming one.
        raise SafeToolError(AMBIGUOUS)
    contains = [
        item for item in items if all(word in _words(item.title) for word in wanted)
    ]
    if len(contains) == 1:
        return contains[0]
    if len(contains) > 1:
        raise SafeToolError(AMBIGUOUS)
    overlapping = [
        item for item in items if any(word in _words(item.title) for word in wanted)
    ]
    if len(overlapping) == 1:
        return overlapping[0]
    if len(overlapping) > 1:
        raise SafeToolError(AMBIGUOUS)
    raise SafeToolError(NO_MATCH)


# --- the writes ---------------------------------------------------------------------------


def _speak_alarm_ids(connection: sqlite3.Connection, item: Candidate, scope: str) -> list[str]:
    """The alarm rows that announce one reminder, however they came to exist.

    The ledger links them, and a reminder written before the ledger existed is
    matched on exactly what it is: same owner, the reminder kind, the same
    words and the same moment. Nothing wider than that is ever removed.
    """
    ids = [
        row["alarm_id"]
        for row in connection.execute(
            "SELECT alarm_id FROM reminder_deliveries WHERE reminder_id = ? AND alarm_id "
            "IS NOT NULL",
            (item.row_id,),
        ).fetchall()
    ]
    if item.due_at is not None:
        ids.extend(
            row["id"]
            for row in connection.execute(
                "SELECT id FROM alarms WHERE user_id = ? AND kind = ? AND label = ? "
                "AND due_at = ? AND delivered_at IS NULL",
                (scope, REMINDER, item.title, int(item.due_at)),
            ).fetchall()
        )
    return sorted(set(identifier for identifier in ids if identifier))


def _drop_alarms(connection: sqlite3.Connection, ids: list[str], scope: str) -> None:
    for identifier in ids:
        connection.execute(
            "DELETE FROM alarms WHERE id = ? AND user_id = ? AND delivered_at IS NULL",
            (identifier, scope),
        )


def _stand_down(connection: sqlite3.Connection, item: Candidate, scope: str) -> None:
    """Remove every pending delivery of one reminder, and the alarm behind its voice.

    This runs before the target of a conversion is written and before a
    cancelled reminder disappears, so there is never a queued call or message
    pointing at something that is no longer there.
    """
    _drop_alarms(connection, _speak_alarm_ids(connection, item, scope), scope)
    connection.execute(
        "DELETE FROM reminder_deliveries WHERE reminder_id = ? AND user_id = ? AND state = ?",
        (item.row_id, scope, reminder_delivery.PENDING),
    )
    connection.execute(
        "DELETE FROM reminder_delivery_awaiting WHERE user_id = ? AND reminder_id = ?",
        (scope, item.row_id),
    )


def _insert_reminder(
    connection: sqlite3.Connection, row: dict[str, Any]
) -> None:
    """Write one reminder row. Its own function so a failure here rolls the rest back."""
    connection.execute(
        "INSERT INTO reminders (id,user_id,title,due,due_at,list_name,notes,completed,"
        "created_at) VALUES (:id,:user_id,:title,:due,:due_at,:list_name,:notes,0,:created_at)",
        row,
    )


def _insert_alarm(connection: sqlite3.Connection, row: dict[str, Any]) -> None:
    connection.execute(
        "INSERT INTO alarms (id,user_id,label,kind,due_at,created_at) "
        "VALUES (:id,:user_id,:label,:kind,:due_at,:created_at)",
        row,
    )


def _clean_title(value: object) -> str:
    cleaned = " ".join(str(value).split())
    if not cleaned:
        raise SafeToolError("Tell me what to call it and I will change it.")
    return cleaned[: reminders_tools.MAX_TITLE_LENGTH]


# --- the three actions ----------------------------------------------------------------------


def _cancel(connection: sqlite3.Connection, item: Candidate, scope: str) -> None:
    if item.source == ALARM_SOURCE:
        cursor = connection.execute(
            "DELETE FROM alarms WHERE id = ? AND user_id = ? AND delivered_at IS NULL",
            (item.row_id, scope),
        )
        if not int(cursor.rowcount or 0):
            raise SafeToolError(NOTHING_PENDING)
        return
    _stand_down(connection, item, scope)
    cursor = connection.execute(
        "DELETE FROM reminders WHERE id = ? AND user_id = ? AND completed = 0",
        (item.row_id, scope),
    )
    if not int(cursor.rowcount or 0):
        raise SafeToolError(NOTHING_PENDING)


def _update(
    connection: sqlite3.Connection,
    item: Candidate,
    scope: str,
    due_at: int | None,
    title: str | None,
) -> tuple[str, int | None]:
    """Apply the parts that were named, and keep every part that was not."""
    new_title = item.title if title is None else title
    new_due = item.due_at if due_at is None else due_at
    if item.source == ALARM_SOURCE:
        if new_due is None:
            raise SafeToolError("I need a time for an alarm. " + WHEN_HINT)
        connection.execute(
            "UPDATE alarms SET label = ?, due_at = ? WHERE id = ? AND user_id = ? "
            "AND delivered_at IS NULL",
            (new_title, int(new_due), item.row_id, scope),
        )
        return new_title, new_due
    connection.execute(
        "UPDATE reminders SET title = ?, due = ?, due_at = ? WHERE id = ? AND user_id = ? "
        "AND completed = 0",
        (
            new_title,
            None if new_due is None else _iso(int(new_due)),
            None if new_due is None else int(new_due),
            item.row_id,
            scope,
        ),
    )
    # The spoken channel is an alarm row and a ledger row; both describe the
    # same moment, so both move with it or the reminder would be announced at
    # the old time or not at all.
    for identifier in _speak_alarm_ids(connection, item, scope):
        if new_due is None:
            connection.execute(
                "DELETE FROM alarms WHERE id = ? AND user_id = ? AND delivered_at IS NULL",
                (identifier, scope),
            )
            continue
        connection.execute(
            "UPDATE alarms SET label = ?, due_at = ? WHERE id = ? AND user_id = ? "
            "AND delivered_at IS NULL",
            (new_title, int(new_due), identifier, scope),
        )
    if new_due is not None:
        connection.execute(
            "UPDATE reminder_deliveries SET due_at = ? WHERE reminder_id = ? AND user_id = ? "
            "AND state = ?",
            (int(new_due), item.row_id, scope, reminder_delivery.PENDING),
        )
    return new_title, new_due


def _convert(
    connection: sqlite3.Connection,
    item: Candidate,
    scope: str,
    target_kind: str,
    due_at: int | None,
    title: str | None,
    current: int,
) -> tuple[str, int, str | None]:
    """Remove the source and write the target, inside one transaction.

    Returns the words, the moment, and the id of a reminder that was created --
    which is what the delivery question is asked about, once the write has
    actually committed.
    """
    new_title = item.title if title is None else title
    new_due = item.due_at if due_at is None else due_at
    if new_due is None:
        raise SafeToolError(
            "That one has no time on it, so tell me when it should go off and I will "
            "make the change. " + WHEN_HINT
        )
    if int(new_due) <= current:
        raise SafeToolError("That time has already passed, so tell me a time still to come.")
    if target_kind == REMINDER:
        _drop_alarms(connection, [item.row_id], scope)
        reminder_id = str(uuid.uuid4())
        _insert_reminder(
            connection,
            dict(
                id=reminder_id,
                user_id=scope,
                title=new_title,
                due=_iso(int(new_due)),
                due_at=int(new_due),
                list_name="Reminders",
                notes="",
                created_at=datetime.now(timezone.utc).isoformat(),
            ),
        )
        # Only the spoken channel is armed. Converting is not consent to be
        # called or messaged, so no remote channel is ever chosen here.
        alarm_id = str(uuid.uuid4())
        _insert_alarm(
            connection,
            dict(
                id=alarm_id,
                user_id=scope,
                label=new_title,
                kind=REMINDER,
                due_at=int(new_due),
                created_at=current,
            ),
        )
        connection.execute(
            "INSERT OR IGNORE INTO reminder_deliveries (id,reminder_id,user_id,channel,due_at,"
            "state,attempts,next_attempt_at,alarm_id,created_at) VALUES (?,?,?,?,?,?,0,0,?,?)",
            (
                str(uuid.uuid4()),
                reminder_id,
                scope,
                reminder_delivery.SPEAK,
                int(new_due),
                reminder_delivery.PENDING,
                alarm_id,
                current,
            ),
        )
        return new_title, int(new_due), reminder_id
    # Into an alarm or a timer: the delivery ledger of the reminder is stood
    # down first, so a queued call or message can never outlive the reminder
    # it belonged to.
    _stand_down(connection, item, scope)
    connection.execute(
        "DELETE FROM reminders WHERE id = ? AND user_id = ? AND completed = 0",
        (item.row_id, scope),
    )
    _insert_alarm(
        connection,
        dict(
            id=str(uuid.uuid4()),
            user_id=scope,
            label=new_title,
            kind=target_kind,
            due_at=int(new_due),
            created_at=current,
        ),
    )
    return new_title, int(new_due), None


# --- one change, said as what it actually was ------------------------------------------------


INTERNAL_FAILURE = (
    "Something went wrong on my end, so nothing was changed. Ask me again and I will try it."
)
NOTHING_TO_CHANGE = (
    "Tell me what to change about it -- a new time, or new wording -- and I will."
)
NO_TARGET = "Tell me what you want it to be instead: a reminder, an alarm or a timer."
ALREADY_THAT = "That one is already what you are asking me to make it."


def _kind_word(item: Candidate) -> str:
    return item.kind


def _result(message: str, data: dict[str, Any]) -> dict[str, Any]:
    return dict(status=_OK, message=message, data=data)


def apply_change(
    item: Candidate,
    action: str,
    *,
    user_id: object = None,
    now: int | None = None,
    when: object = None,
    title: object = None,
    target_kind: object = None,
) -> dict[str, Any]:
    """Carry out one already-resolved change, atomically, and say what happened.

    The candidate came from :func:`resolve` or :func:`candidates`, so it is
    already this owner own and already pending. Everything here runs in one
    transaction: a failure anywhere leaves the item exactly as it was.
    """
    scope = _scope(user_id)
    current = _now() if now is None else int(now)
    try:
        if action not in ACTIONS:
            raise SafeToolError("I can cancel it, change it, or make it a different kind.")
        new_due: int | None = None
        if isinstance(when, str) and when.strip():
            new_due = parse_when(when, current)
        elif when is not None and not isinstance(when, str):
            raise SafeToolError("I could not read that as a time. " + WHEN_HINT)
        new_title: str | None = None
        if title is not None:
            new_title = _clean_title(title)
        wanted_kind: str | None = None
        if action == "convert":
            if not isinstance(target_kind, str) or target_kind.strip().lower() not in KINDS:
                raise SafeToolError(NO_TARGET)
            wanted_kind = target_kind.strip().lower()
            if wanted_kind == item.kind:
                raise SafeToolError(ALREADY_THAT)
        elif target_kind is not None:
            raise SafeToolError("Did you want me to change that one into something else?")
        if action == "update" and new_due is None and new_title is None:
            raise SafeToolError(NOTHING_TO_CHANGE)
    except SafeToolError as error:
        logger.info("Refused a scheduled change: %s", type(error).__name__)
        return safe_error_result(error)

    reminder_id: str | None = None
    final_title = item.title
    final_due = item.due_at
    try:
        with closing(_connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if action == "cancel":
                    _cancel(connection, item, scope)
                elif action == "update":
                    final_title, final_due = _update(
                        connection, item, scope, new_due, new_title
                    )
                else:
                    final_title, final_due, reminder_id = _convert(
                        connection,
                        item,
                        scope,
                        str(wanted_kind),
                        new_due,
                        new_title,
                        current,
                    )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
    except SafeToolError as error:
        logger.info("Refused a scheduled change: %s", type(error).__name__)
        return safe_error_result(error)
    except Exception as exc:  # noqa: BLE001 - the store is the only thing that can fail here
        logger.error("A scheduled change failed (%s)", type(exc).__name__)
        return dict(status="error", message=INTERNAL_FAILURE, data=dict())

    logger.info("Applied a scheduled change: %s", action)
    return _speak(
        action, item, scope, final_title, final_due, reminder_id, wanted_kind, current
    )


def _speak(
    action: str,
    item: Candidate,
    scope: str,
    title: str,
    due_at: int | None,
    reminder_id: str | None,
    wanted_kind: str | None,
    current: int,
) -> dict[str, Any]:
    """What actually happened, in one or two sentences, claiming nothing more."""
    data = dict(
        action=action,
        kind=item.kind,
        target_kind=item.kind if wanted_kind is None else wanted_kind,
        title=title,
        due=None if due_at is None else _iso(int(due_at)),
        delivery=[],
        delivery_pending=False,
    )
    if action == "cancel":
        data["title"] = item.title
        return _result("That " + item.kind + " is cancelled: " + item.title + ".", data)
    delay = "" if due_at is None else alarms_tools.describe_delay(int(due_at) - current)
    if action == "update":
        if due_at is None:
            return _result(
                "Changed. It is on your list as: " + title + ", with no time on it.", data
            )
        return _result(
            "Changed. That " + item.kind + " is now set " + delay + ": " + title + ".", data
        )
    target = str(wanted_kind)
    if reminder_id is not None:
        data["delivery"] = [reminder_delivery.SPEAK]
        data["delivery_pending"] = True
        reminder_delivery.await_answer(reminder_id, scope, int(due_at or 0), now=current)
        question = reminders_tools._question(reminders_tools._offered(scope))
        return _result(
            "That " + item.kind + " is a reminder now, set " + delay + ": " + title + ". "
            + question,
            data,
        )
    return _result(
        "That reminder is " + ("an " if target == ALARM else "a ") + str(target) + " now, set "
        + delay + ": " + title + ". I will say it out loud here when it comes due.",
        data,
    )


# --- the native tool ---------------------------------------------------------------------------


def change_scheduled_item(
    action: object = None,
    reference: object = None,
    kind: object = None,
    when: object = None,
    title: object = None,
    target_kind: object = None,
    user_id: str | None = None,
    now: int | None = None,
    identity_configured: bool = False,
) -> dict[str, Any]:
    """Cancel, change or convert one scheduled item of the signed-in user.

    Every argument is words the person said. The owner is the verified session
    and is never one of them, and neither is a row, a channel or a
    destination: ``identity_configured`` is not a declared parameter, so a
    model cannot set it and a session without a verified user is refused here
    rather than falling back to the unowned legacy store.
    """
    if user_id is None and identity_configured:
        return scheduling_unavailable_result()
    current = _now() if now is None else int(now)
    try:
        if not isinstance(action, str) or action not in ACTIONS:
            raise SafeToolError("I can cancel it, change it, or make it a different kind.")
        wanted = None
        if kind is not None:
            if not isinstance(kind, str) or kind.strip().lower() not in KINDS:
                raise SafeToolError("I can change an alarm, a timer or a reminder.")
            wanted = kind.strip().lower()
        item = resolve(user_id, current, reference=reference, kind=wanted)
    except SafeToolError as error:
        logger.info("Refused a scheduled change: %s", type(error).__name__)
        return safe_error_result(error)
    except ValueError:
        # An owner id of the wrong shape never reaches the store.
        logger.info("Refused a scheduled change for an unusable session scope")
        return scheduling_unavailable_result()
    return apply_change(
        item,
        action,
        user_id=user_id,
        now=current,
        when=when,
        title=title,
        target_kind=target_kind,
    )


# --- the one authorised repair ------------------------------------------------------------------


def pending_counts(user_id: object, now: int | None = None) -> dict[str, int]:
    """How many pending items of each kind this owner has. Counts only, ever."""
    items = candidates(user_id, now)
    return dict(
        alarms=sum(1 for item in items if item.source == ALARM_SOURCE),
        reminders=sum(1 for item in items if item.source == REMINDER_SOURCE),
    )


def repair_latest_alarm_to_reminder(
    user_id: object, now: int | None = None, expect_pending: int | None = None
) -> dict[str, Any]:
    """Convert the single pending alarm of one owner into a reminder, unseen.

    For the one item that was created by a request CAAL could not carry out:
    the person asked for a reminder and got an alarm, because converting did
    not exist yet. It runs through exactly the same owner-scoped write as the
    conversation path, so it can neither cross an owner boundary nor send
    anything, and it refuses unless the owner has exactly one pending alarm or
    timer -- there is no version of this that picks between two.

    It returns counts and kinds. The label of the item is never read out,
    returned or logged.
    """
    current = _now() if now is None else int(now)
    items = [item for item in candidates(user_id, current) if item.source == ALARM_SOURCE]
    if len(items) != 1 or (expect_pending is not None and len(items) != expect_pending):
        logger.info("Refused a repair: %d pending alarm(s) of this owner", len(items))
        return dict(
            status="invalid_request",
            message="This owner does not have exactly one pending alarm to repair.",
            pending=len(items),
        )
    source_kind = items[0].kind
    outcome = apply_change(
        items[0], "convert", user_id=user_id, now=current, target_kind=REMINDER
    )
    counts = pending_counts(user_id, current)
    if outcome.get("status") != _OK:
        return dict(
            status=outcome.get("status", "error"),
            message="The repair did not go through, and nothing was changed.",
            alarms_pending=counts["alarms"],
            reminders_pending=counts["reminders"],
        )
    # The conversation path asks how they want the new reminder delivered, and
    # records that the question is open. A repair asks nobody anything, so the
    # question is withdrawn: an answer to something never said out loud would
    # land on this reminder the next time they spoke.
    reminder_delivery.clear_awaiting(_scope(user_id))
    logger.info("Repaired one scheduled item into a reminder")
    return dict(
        status=_OK,
        source_kind=source_kind,
        target_kind=REMINDER,
        alarms_pending=counts["alarms"],
        reminders_pending=counts["reminders"],
        delivery=outcome["data"]["delivery"],
    )


__all__ = [
    "ACTIONS",
    "CHANGE_DESCRIPTION",
    "CHANGE_SCHEMA",
    "CHANGE_TOOL",
    "KINDS",
    "Candidate",
    "apply_change",
    "AMBIGUOUS",
    "candidates",
    "change_scheduled_item",
    "pending_counts",
    "points_at_the_newest",
    "repair_latest_alarm_to_reminder",
    "resolve",
    "summarize",
    "summarize_at",
]
