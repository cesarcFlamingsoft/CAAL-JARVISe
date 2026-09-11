"""Which ways a timed reminder reaches its owner, and the ledger that settles them.

A reminder used to have one channel: whatever voice session of its owner
happened to be live when it came due. That is still a channel, and still the
only one a live room can serve. The other two outlive the room, so they are
recorded here as their own rows and carried out by the durable worker:

* ``speak``     announced by an active session of the owner, exactly as before.
  Settled by :mod:`caal.alarm_delivery` when the words were actually said;
* ``telegram``  a message to the one chat an administrator bound to that one
  profile. There is no per-user Telegram inbox in this deployment, so every
  other profile is refused rather than quietly given the operator chat;
* ``call``      an outbound call to the number already approved on that
  profile. The number is resolved server-side at dispatch time; it is never
  stored here, never an argument, and never reaches a model.

Nothing in this module takes a destination from a caller. A caller chooses
from :data:`CHANNELS` and nothing else, and ownership comes from the verified
session scope. Nothing here logs a title, an id, a user or a number.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from caal.tools.errors import SafeToolError

logger = logging.getLogger(__name__)

STORE_PATH = Path(os.getenv("CAAL_DATA_DIR", "/app/data")) / "assistant.sqlite3"

SPEAK = "speak"
TELEGRAM = "telegram"
CALL = "call"
CHANNELS: tuple[str, ...] = (SPEAK, TELEGRAM, CALL)
ALL = "all"
#: What a tool schema may offer. ``all`` is a shorthand for a combination, not
#: a fourth channel: it expands to whatever this owner is actually allowed.
CHANNEL_ARGUMENTS: tuple[str, ...] = CHANNELS + (ALL,)
DEFAULT_CHANNELS: tuple[str, ...] = (SPEAK,)
#: The channels a live session cannot serve, so the durable worker owns them.
WORKER_CHANNELS: tuple[str, ...] = (TELEGRAM, CALL)

PENDING = "pending"
DELIVERED = "delivered"
FAILED = "failed"

#: How many hand-off attempts one channel gets before it is left failed.
MAX_ATTEMPTS = 3
#: How long a worker may hold a claim before another worker may take it over.
CLAIM_LEASE_SECONDS = 300
BACKOFF_SECONDS = (60, 300, 900)
MAX_CLAIM = 20
_BUSY_TIMEOUT_SECONDS = 5.0

_CREATE_DELIVERIES = """
    CREATE TABLE IF NOT EXISTS reminder_deliveries (
        id TEXT PRIMARY KEY,
        reminder_id TEXT NOT NULL,
        user_id TEXT NOT NULL DEFAULT '',
        channel TEXT NOT NULL,
        due_at INTEGER NOT NULL,
        state TEXT NOT NULL DEFAULT 'pending',
        attempts INTEGER NOT NULL DEFAULT 0,
        claimed_by TEXT,
        claimed_at INTEGER,
        next_attempt_at INTEGER NOT NULL DEFAULT 0,
        settled_at INTEGER,
        alarm_id TEXT,
        created_at INTEGER NOT NULL,
        UNIQUE (reminder_id, channel)
    )
"""

_CREATE_META = """
    CREATE TABLE IF NOT EXISTS reminder_delivery_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
"""

_ADOPTED = "adopted_pre_ledger_reminders"

_CREATE_DEFAULTS = """
    CREATE TABLE IF NOT EXISTS reminder_delivery_defaults (
        user_id TEXT PRIMARY KEY,
        channels TEXT NOT NULL,
        updated_at INTEGER NOT NULL
    )
"""


def _connect() -> sqlite3.Connection:
    """Open the shared store and make sure this slice of it exists.

    Additive only: the tables are created beside whatever a pre-delivery
    database already holds, and no existing row or column is touched, so an
    older reminders table keeps every reminder it had.
    """
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(STORE_PATH, timeout=_BUSY_TIMEOUT_SECONDS, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout = {int(_BUSY_TIMEOUT_SECONDS * 1000)}")
    connection.execute(_CREATE_DELIVERIES)
    connection.execute(_CREATE_DEFAULTS)
    connection.execute(_CREATE_META)
    fresh = (
        connection.execute(
            "SELECT 1 FROM reminder_delivery_meta WHERE key = ?", (_ADOPTED,)
        ).fetchone()
        is None
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS reminder_deliveries_due "
        "ON reminder_deliveries (state, channel, due_at)"
    )
    if fresh:
        _adopt_existing_reminders(connection)
    return connection


def _adopt_existing_reminders(connection: sqlite3.Connection) -> None:
    """Record the channel the reminders written before this ledger already use.

    A timed reminder from before delivery choices existed is announced by a
    live session of its owner, and nothing else. That *is* the spoken channel,
    so it is recorded as one -- linked to the alarm row that will actually
    announce it, matched exactly on owner, kind, due time and label, and only
    while that alarm is still waiting. A reminder whose alarm has already been
    delivered or was never scheduled adopts nothing: the alternative is a
    dashboard that promises an announcement nobody is going to make.

    Runs exactly once per database, recorded by its own marker rather than by
    the table existing: a deployment that read the ledger before this existed
    must still adopt what it already had. Counts only in the log.
    """
    connection.execute(
        "INSERT OR REPLACE INTO reminder_delivery_meta (key, value) VALUES (?, ?)",
        (_ADOPTED, "1"),
    )
    try:
        rows = connection.execute(
            "SELECT r.id AS reminder_id, r.user_id AS user_id, r.due_at AS due_at, "
            "MIN(a.id) AS alarm_id FROM reminders r JOIN alarms a "
            "ON a.user_id = r.user_id AND a.kind = 'reminder' AND a.label = r.title "
            "AND a.due_at = r.due_at AND a.delivered_at IS NULL "
            "WHERE r.due_at IS NOT NULL AND r.completed = 0 GROUP BY r.id"
        ).fetchall()
    except sqlite3.Error:
        # No alarms table yet, or an older shape: there is nothing to adopt.
        return
    if not rows:
        return
    moment = int(_now())
    connection.executemany(
        "INSERT OR IGNORE INTO reminder_deliveries (id,reminder_id,user_id,channel,due_at,"
        "state,attempts,next_attempt_at,alarm_id,created_at) VALUES (?,?,?,?,?,?,0,0,?,?)",
        [
            (
                str(uuid.uuid4()),
                row["reminder_id"],
                row["user_id"],
                SPEAK,
                int(row["due_at"]),
                PENDING,
                row["alarm_id"],
                moment,
            )
            for row in rows
        ],
    )
    logger.info("Adopted %d reminder(s) written before delivery channels existed", len(rows))


# --- what a caller may ask for -------------------------------------------------------------


def parse_channels(value: object) -> tuple[str, ...]:
    """Read a requested combination of channels, or refuse it in plain words.

    A bounded enum and nothing else: a phone number, a chat id or an invented
    channel is a refusal, never a destination. The order is always the order
    of :data:`CHANNELS`, so a stored selection compares equal however it was
    asked for.
    """
    if isinstance(value, str):
        requested: list[Any] = [value]
    elif isinstance(value, (list, tuple)):
        requested = list(value)
    else:
        raise SafeToolError(
            "Tell me how you want the reminder: out loud here, on Telegram, or a call."
        )
    if not requested:
        raise SafeToolError(
            "Tell me how you want the reminder: out loud here, on Telegram, or a call."
        )
    chosen: set[str] = set()
    for item in requested:
        if not isinstance(item, str):
            raise SafeToolError("I can say it here, send it on Telegram, or call you.")
        name = item.strip().lower()
        if name == ALL:
            chosen.update(CHANNELS)
        elif name in CHANNELS:
            chosen.add(name)
        else:
            raise SafeToolError("I can say it here, send it on Telegram, or call you.")
    return tuple(channel for channel in CHANNELS if channel in chosen)


# --- what a caller is actually allowed --------------------------------------------------


def _scope(user_id: object) -> str:
    """The storage scope of a caller: the legacy scope, or a validated user id."""
    if user_id is None:
        return ""
    from caal.user_scope import require_user_id

    return require_user_id(user_id)


def _settings_file() -> Path | None:
    """The settings file this deployment actually uses, whoever is asking.

    The agent and the durable worker mount the same ``settings.json`` but
    import ``caal`` from different roots, so the module-relative settings path
    resolves in one process and not in the other. A binding that depended on
    that would mean the dashboard offering Telegram while the worker refused
    it at delivery time -- a promise the deployment cannot keep. So the file is
    looked for where it is: an explicit path, the module's own, and finally
    beside the data directory every process already agrees on.
    """
    candidates: list[Path] = []
    explicit = os.getenv("CAAL_SETTINGS_PATH", "").strip()
    if explicit:
        candidates.append(Path(explicit))
    try:
        from caal import settings as settings_module

        candidates.append(Path(settings_module.SETTINGS_PATH))
    except Exception:
        pass
    candidates.append(Path(os.getenv("CAAL_DATA_DIR", "/app/data")).parent / "settings.json")
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _setting(name: str) -> str:
    """One operator setting, read the same way in every process. Never logged."""
    try:
        from caal import settings as settings_module

        value = settings_module.load_settings().get(name)
    except Exception:
        value = None
    if value:
        return str(value)
    path = _settings_file()
    if path is None:
        return ""
    try:
        with path.open() as handle:
            return str(json.load(handle).get(name) or "")
    except (OSError, ValueError):
        # A settings file that cannot be read is a deployment without the
        # binding, which refuses; it is never a reason to widen anything.
        return ""


def telegram_configured() -> bool:
    """Whether this deployment has a Telegram bot at all. Never reads the values out."""
    token = _setting("telegram_bot_token") or os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = _setting("telegram_chat_id") or os.getenv("TELEGRAM_CHAT_ID", "")
    return bool(token and chat_id)


def telegram_owner() -> str:
    """The one opaque profile the configured Telegram chat belongs to, if any.

    An operator may pin it in the environment, which is the one place every
    process is guaranteed to read alike; otherwise it comes from the settings
    file. Empty means no signed-in user reaches that chat at all.
    """
    pinned = os.getenv("CAAL_TELEGRAM_OWNER_USER_ID", "").strip()
    if pinned:
        return pinned
    return _setting("telegram_owner_user_id")


def resolve_callback_number(user_id: str) -> str | None:
    """The owner own approved callback number, resolved server-side, or nothing.

    The value is used to decide whether the channel exists at all; it is never
    returned to a caller, stored here, or written to a log.
    """
    try:
        from caal import user_api

        store = getattr(user_api.get_runtime(), "store", None)
        if store is None:
            return None
        return store.approved_callback_number(user_id)
    except Exception:
        return None


def channel_availability(user_id: object) -> dict[str, bool]:
    """Which channels this owner may actually use, decided entirely server-side."""
    scope = _scope(user_id)
    # A global Telegram chat is an operator channel, not a per-user inbox: a
    # signed-in user may use it only when an administrator bound that chat to
    # their opaque profile id, and a legacy single-user deployment keeps it.
    telegram = telegram_configured() and (scope == "" or scope == telegram_owner())
    call = bool(scope) and bool(resolve_callback_number(scope))
    return {SPEAK: True, TELEGRAM: telegram, CALL: call}


REFUSALS = {
    TELEGRAM: (
        "I do not have a Telegram chat authorised for your profile, so I cannot message you there."
    ),
    CALL: "There is no approved callback number on your profile, so I cannot call you.",
}


def allowed(user_id: object, channels: tuple[str, ...]) -> tuple[tuple[str, ...], list[str]]:
    """Split a requested combination into what may be used and what must be said."""
    availability = channel_availability(user_id)
    permitted = tuple(channel for channel in channels if availability.get(channel))
    refused = [REFUSALS[channel] for channel in channels if not availability.get(channel)]
    if refused:
        logger.info("Refused %d unauthorised reminder channel(s)", len(refused))
    return permitted, refused


# --- the owner default for future reminders ----------------------------------------------


def default_channels(user_id: object) -> tuple[str, ...]:
    """What a new reminder of this owner uses when they did not say."""
    scope = _scope(user_id)
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT channels FROM reminder_delivery_defaults WHERE user_id = ?", (scope,)
        ).fetchone()
    if row is None:
        return DEFAULT_CHANNELS
    stored = tuple(part for part in str(row["channels"]).split(",") if part in CHANNELS)
    return tuple(channel for channel in CHANNELS if channel in stored) or DEFAULT_CHANNELS


def has_default(user_id: object) -> bool:
    """Whether this owner has ever chosen, so an unspecified reminder can ask once."""
    scope = _scope(user_id)
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT 1 FROM reminder_delivery_defaults WHERE user_id = ?", (scope,)
        ).fetchone()
    return row is not None


def set_default_channels(
    user_id: object, channels: object, now: int | None = None
) -> tuple[str, ...]:
    """Save the owner default. A channel they may not use is refused, not saved."""
    scope = _scope(user_id)
    chosen = parse_channels(channels)
    permitted, refused = allowed(scope, chosen)
    if refused or not permitted:
        raise SafeToolError(refused[0] if refused else "Pick at least one way to be reminded.")
    moment = int(now if now is not None else _now())
    with closing(_connect()) as connection:
        connection.execute(
            "INSERT INTO reminder_delivery_defaults (user_id,channels,updated_at) "
            "VALUES (?,?,?) ON CONFLICT(user_id) DO UPDATE SET channels = excluded.channels, "
            "updated_at = excluded.updated_at",
            (scope, ",".join(permitted), moment),
        )
    logger.info("Saved a reminder delivery default of %d channel(s)", len(permitted))
    return permitted


def _now() -> int:
    import time

    return int(time.time())


# --- the per-channel ledger ----------------------------------------------------------------


@dataclass(frozen=True)
class Delivery:
    """One channel of one reminder, claimed by one worker.

    Carries the title because the worker has to write the message, and the
    owner id because every destination is resolved from it. Neither is ever
    logged, and neither came from a model.
    """

    delivery_id: str
    reminder_id: str
    user_id: str
    channel: str
    title: str
    due_at: int
    attempts: int


def arm(
    reminder_id: str,
    user_id: object,
    channels: tuple[str, ...],
    due_at: int,
    now: int | None = None,
    alarm_id: str | None = None,
) -> tuple[str, ...]:
    """Record the channels one timed reminder will be delivered on.

    Idempotent: arming the same channel twice keeps the first row, so nothing
    is ever delivered twice because a caller retried.
    """
    scope = _scope(user_id)
    moment = int(now if now is not None else _now())
    rows = [
        (
            str(uuid.uuid4()),
            reminder_id,
            scope,
            channel,
            int(due_at),
            PENDING,
            0,
            0,
            alarm_id if channel == SPEAK else None,
            moment,
        )
        for channel in CHANNELS
        if channel in channels
    ]
    if not rows:
        return ()
    with closing(_connect()) as connection:
        connection.executemany(
            "INSERT OR IGNORE INTO reminder_deliveries (id,reminder_id,user_id,channel,due_at,"
            "state,attempts,next_attempt_at,alarm_id,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
    logger.info("Armed a timed reminder on %d channel(s)", len(rows))
    return tuple(row[3] for row in rows)


def channels_of(reminder_id: str) -> tuple[str, ...]:
    """The channels currently selected for one reminder, in the canonical order."""
    with closing(_connect()) as connection:
        rows = connection.execute(
            "SELECT channel FROM reminder_deliveries WHERE reminder_id = ?", (reminder_id,)
        ).fetchall()
    selected = {row["channel"] for row in rows}
    return tuple(channel for channel in CHANNELS if channel in selected)


def states_of(reminder_ids: list[str]) -> dict[str, dict[str, str]]:
    """The state of every channel of every named reminder: channel -> state."""
    if not reminder_ids:
        return {}
    report: dict[str, dict[str, str]] = {reminder_id: {} for reminder_id in reminder_ids}
    placeholders = ",".join("?" for _ in reminder_ids)
    with closing(_connect()) as connection:
        rows = connection.execute(
            "SELECT reminder_id, channel, state FROM reminder_deliveries "
            f"WHERE reminder_id IN ({placeholders})",
            tuple(reminder_ids),
        ).fetchall()
    for row in rows:
        report.setdefault(row["reminder_id"], {})[row["channel"]] = row["state"]
    return report


def cancel(reminder_id: str, channels: tuple[str, ...]) -> int:
    """Drop channels that have not been delivered yet. A delivered one stays recorded."""
    if not channels:
        return 0
    placeholders = ",".join("?" for _ in channels)
    with closing(_connect()) as connection:
        cursor = connection.execute(
            f"DELETE FROM reminder_deliveries WHERE reminder_id = ? AND channel IN "
            f"({placeholders}) AND state = ?",
            (reminder_id, *channels, PENDING),
        )
    return int(cursor.rowcount or 0)


def alarm_of(reminder_id: str) -> str | None:
    """The alarm row the spoken channel of this reminder is waiting on, if any."""
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT alarm_id FROM reminder_deliveries WHERE reminder_id = ? AND channel = ?",
            (reminder_id, SPEAK),
        ).fetchone()
    return None if row is None else row["alarm_id"]


def claim_due(
    claimant: str,
    now: int,
    channels: tuple[str, ...] = WORKER_CHANNELS,
    limit: int = MAX_CLAIM,
) -> list[Delivery]:
    """Atomically claim the due, unsettled deliveries the durable worker owns.

    A claim is exclusive and leased. A second worker sees nothing while the
    lease holds, and a worker that dies mid-flight loses the lease rather than
    the delivery: the row is claimable again once the lease expires, and it is
    only ever removed from the queue by an actual delivery or a spent budget.
    The spoken channel is never claimed here -- only a live session can settle
    that, so only a live session may take it.
    """
    wanted = tuple(channel for channel in channels if channel in WORKER_CHANNELS)
    if not wanted or limit <= 0:
        return []
    current = int(now)
    stale = current - CLAIM_LEASE_SECONDS
    placeholders = ",".join("?" for _ in wanted)
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            rows = connection.execute(
                "SELECT d.id AS id, d.reminder_id AS reminder_id, d.user_id AS user_id, "
                "d.channel AS channel, d.due_at AS due_at, d.attempts AS attempts, "
                "r.title AS title FROM reminder_deliveries d "
                "JOIN reminders r ON r.id = d.reminder_id "
                f"WHERE d.state = ? AND d.channel IN ({placeholders}) AND d.due_at <= ? "
                "AND d.next_attempt_at <= ? AND (d.claimed_at IS NULL OR d.claimed_at <= ?) "
                "AND r.completed = 0 ORDER BY d.due_at, d.channel LIMIT ?",
                (PENDING, *wanted, current, current, stale, int(limit)),
            ).fetchall()
            if rows:
                connection.executemany(
                    "UPDATE reminder_deliveries SET claimed_by = ?, claimed_at = ?, "
                    "attempts = attempts + 1 WHERE id = ?",
                    [(claimant, current, row["id"]) for row in rows],
                )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return [
        Delivery(
            delivery_id=row["id"],
            reminder_id=row["reminder_id"],
            user_id=row["user_id"],
            channel=row["channel"],
            title=row["title"],
            due_at=int(row["due_at"]),
            attempts=int(row["attempts"]) + 1,
        )
        for row in rows
    ]


def mark_delivered(delivery_id: str, claimant: str, now: int) -> bool:
    """Settle one channel that was actually accepted by that channel.

    Only the worker holding the claim may settle it, and only once: this is
    the single place a delivery is ever called done, and it is called after
    the hand-off returned, never before.
    """
    with closing(_connect()) as connection:
        cursor = connection.execute(
            "UPDATE reminder_deliveries SET state = ?, settled_at = ?, claimed_by = NULL, "
            "claimed_at = NULL WHERE id = ? AND claimed_by = ? AND state = ?",
            (DELIVERED, int(now), delivery_id, claimant, PENDING),
        )
    settled = int(cursor.rowcount or 0) > 0
    if settled:
        logger.info("Settled one reminder channel as delivered")
    return settled


def release(
    delivery_id: str,
    claimant: str,
    delay_seconds: int,
    now: int,
    reset_attempts: bool = False,
) -> str:
    """Hand one unsettled channel back, or leave it truthfully failed.

    Nothing was delivered, so nothing is recorded as delivered. A budget that
    is spent stops here in ``failed`` rather than retrying forever or quietly
    disappearing, and the other channels of the same reminder are untouched.
    """
    current = int(now)
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT attempts FROM reminder_deliveries WHERE id = ? AND claimed_by = ? "
            "AND state = ?",
            (delivery_id, claimant, PENDING),
        ).fetchone()
        if row is None:
            return ""
        if not reset_attempts and int(row["attempts"]) >= MAX_ATTEMPTS:
            connection.execute(
                "UPDATE reminder_deliveries SET state = ?, settled_at = ?, claimed_by = NULL, "
                "claimed_at = NULL WHERE id = ?",
                (FAILED, current, delivery_id),
            )
            logger.warning("A reminder channel spent its attempts and is left failed")
            return FAILED
        connection.execute(
            "UPDATE reminder_deliveries SET claimed_by = NULL, claimed_at = NULL, "
            "attempts = ?, next_attempt_at = ? WHERE id = ?",
            (0 if reset_attempts else int(row["attempts"]), current + max(0, int(delay_seconds)),
             delivery_id),
        )
    return PENDING


def fail(delivery_id: str, claimant: str, now: int) -> bool:
    """Stop one claimed channel for good, without ever claiming it was delivered.

    For a refusal or a hand-off whose outcome cannot be known: retrying either
    would be wrong, so the channel is left in ``failed``, which is what a
    reader is then shown. The other channels of the same reminder go on.
    """
    with closing(_connect()) as connection:
        cursor = connection.execute(
            "UPDATE reminder_deliveries SET state = ?, settled_at = ?, claimed_by = NULL, "
            "claimed_at = NULL WHERE id = ? AND claimed_by = ? AND state = ?",
            (FAILED, int(now), delivery_id, claimant, PENDING),
        )
    return int(cursor.rowcount or 0) > 0


def reminder_message(title: str) -> str:
    """How a due reminder reads on any channel: the owner own words, and nothing else."""
    clean = " ".join(str(title).split())
    return f"Reminder: {clean}." if clean else "Here is your reminder."


def backoff_seconds(attempts: int) -> int:
    """How long to wait before the next attempt on a channel that could not be handed off."""
    index = max(0, min(int(attempts) - 1, len(BACKOFF_SECONDS) - 1))
    return BACKOFF_SECONDS[index]


def settle_speak(alarm_ids: list[str], now: int | None = None) -> int:
    """Settle the spoken channel of every reminder whose alarm was actually said.

    Called by :mod:`caal.alarm_delivery` after the words left the session, so
    an alarm that was claimed but never spoken leaves its channel pending.
    """
    if not alarm_ids:
        return 0
    moment = int(now if now is not None else _now())
    placeholders = ",".join("?" for _ in alarm_ids)
    with closing(_connect()) as connection:
        cursor = connection.execute(
            "UPDATE reminder_deliveries SET state = ?, settled_at = ? WHERE channel = ? "
            f"AND state = ? AND alarm_id IN ({placeholders})",
            (DELIVERED, moment, SPEAK, PENDING, *alarm_ids),
        )
    return int(cursor.rowcount or 0)


def queue_counts(now: int | None = None) -> dict[str, int]:
    """Counts only, for an operator probe. Never an id, a title or a number."""
    moment = int(now if now is not None else _now())
    with closing(_connect()) as connection:
        rows = connection.execute(
            "SELECT state, COUNT(*) AS total FROM reminder_deliveries GROUP BY state"
        ).fetchall()
        (due,) = connection.execute(
            "SELECT COUNT(*) FROM reminder_deliveries WHERE state = ? AND due_at <= ?",
            (PENDING, moment),
        ).fetchone()
    counts = {row["state"]: int(row["total"]) for row in rows}
    counts["due"] = int(due)
    return counts


__all__ = [
    "ALL",
    "BACKOFF_SECONDS",
    "CALL",
    "CHANNELS",
    "CHANNEL_ARGUMENTS",
    "CLAIM_LEASE_SECONDS",
    "DEFAULT_CHANNELS",
    "DELIVERED",
    "FAILED",
    "MAX_ATTEMPTS",
    "PENDING",
    "REFUSALS",
    "SPEAK",
    "TELEGRAM",
    "WORKER_CHANNELS",
    "Delivery",
    "alarm_of",
    "allowed",
    "arm",
    "backoff_seconds",
    "cancel",
    "channel_availability",
    "channels_of",
    "claim_due",
    "default_channels",
    "fail",
    "has_default",
    "mark_delivered",
    "parse_channels",
    "queue_counts",
    "release",
    "reminder_message",
    "resolve_callback_number",
    "set_default_channels",
    "settle_speak",
    "states_of",
    "telegram_configured",
    "telegram_owner",
]
