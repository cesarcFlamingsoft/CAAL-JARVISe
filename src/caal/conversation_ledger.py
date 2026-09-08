"""Private durable ledger of one *live* logical conversation.

A conversation that starts on the web or a LAN voice session and continues on
the approved phone is one logical conversation. This ledger is what lets the
phone leg pick up a long conversation without squeezing it through a tiny
dispatch snapshot: the origin session appends completed visible turns as they
happen, the outbound worker hydrates from it once a human answers, and the
ledger is deleted as soon as the last linked session closes (or its idle TTL
lapses).

What it is not
--------------
It is **not** long-term memory. The ledger is transient conversational state
with the same lifetime as the conversation itself. Nothing here is ever copied
into the explicit preference store or any other durable memory; see
:data:`HERMES_LONG_TERM_MEMORY_SEAM`.

Safety envelope
---------------
* SQLite on the existing ``CAAL_DATA_DIR`` volume, server-side only; the
  opaque conversation ID never enters public room metadata, browser
  responses, or logs;
* only completed ``user``/``assistant`` text turns are appended, with
  prompts, tool calls, tool outputs and non-text content excluded;
* credential-like values are redacted before anything touches disk;
* hard per-turn, recent-window, summary and hydration limits keep the ledger
  bounded no matter how long the conversation runs;
* transcript content is never logged, only counts.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .handoff_context import MAX_TURN_CHARS, redact_sensitive_text
from .user_scope import require_user_id

logger = logging.getLogger(__name__)

STORE_PATH = Path(os.getenv("CAAL_DATA_DIR", "/app/data")) / "assistant.sqlite3"

# HERMES_LONG_TERM_MEMORY_SEAM
# ---------------------------
# CAAL does not make importance decisions. Long-term memory is written only
# (a) through the user's explicit "remember/save this" request, handled by the
# separate explicit memory tool, or (b) by the Hermes agent under its own
# importance policy. This ledger never promotes a transcript, a summary, or a
# turn into either of those stores, and it deliberately has no dependency on
# them. If Hermes ever wants to consult a live conversation, that request must
# come from Hermes through an explicit, authenticated seam; CAAL never pushes.
HERMES_LONG_TERM_MEMORY_SEAM = (
    "hermes-only: long-term memory is written by explicit user request or by the "
    "Hermes agent's own importance policy; CAAL never promotes ledger content"
)

# --- limits ------------------------------------------------------------------

# Verbatim window kept per conversation. Older turns fold into the summary.
MAX_RECENT_TURNS = 24
MAX_RECENT_CHARS = 8000
# Rolling compact summary of everything that aged out of the verbatim window.
# Bounded by lines as well as characters so the very oldest folded turns age
# out even when every line is short.
MAX_SUMMARY_CHARS = 2000
MAX_SUMMARY_LINE_CHARS = 160
MAX_SUMMARY_LINES = 4 * MAX_RECENT_TURNS
# What a phone continuation is hydrated with: summary plus the latest turns.
MAX_HYDRATION_TURNS = 12
MAX_HYDRATION_CHARS = 6000
# A linked session that has neither recorded a turn nor sent a liveness
# heartbeat for this long is treated as gone even if it never closed cleanly.
# A conversation is deleted once it has no live link left; an active session
# that keeps heartbeating keeps its ledger for as long as it is open.
CONVERSATION_TTL_SECONDS = 6 * 60 * 60
# How often a live session should refresh its link (well inside the TTL).
SESSION_LIVENESS_INTERVAL_SECONDS = 10 * 60
# How long an outbound worker has to claim a continuation after dispatch.
CONTINUATION_CLAIM_TTL_SECONDS = 300
# How long a claimed return marker stays reserved before another turn may
# retry it, covering a claimant that died between claim and ack.
RETURN_SYNC_CLAIM_TTL_SECONDS = 60
_RETURN_SYNC_TOKEN_BYTES = 16

MAX_CONVERSATION_ID_LENGTH = 64
# Shape only: URL-safe token characters, non-empty, bounded. Whether an id
# names a live conversation is a separate (LookupError) question.
_CONVERSATION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_ID_BYTES = 24
_BUSY_TIMEOUT_SECONDS = 5.0
_ALLOWED_ROLES = frozenset({"user", "assistant"})

_LINK_ACTIVE = "active"
_LINK_PENDING = "pending"
_LINK_CLOSED = "closed"


# --- models ------------------------------------------------------------------


@dataclass(frozen=True)
class LedgerTurn:
    """One visible turn. Never prints its text."""

    role: str
    text: str = field(repr=False)

    def __repr__(self) -> str:
        return f"LedgerTurn(role={self.role!r}, chars={len(self.text)})"


@dataclass(frozen=True)
class ConversationContext:
    """What a continuation is hydrated with: a compact summary plus recent turns.

    Duck-compatible with :class:`caal.handoff_context.ConversationSnapshot`
    for :func:`caal.handoff_context.restore_conversation_context`. Never prints
    its contents.
    """

    summary: str = field(default="", repr=False)
    turns: tuple[LedgerTurn, ...] = field(default=(), repr=False)
    # True when this hydrates the origin session after a phone leg ended,
    # rather than the phone leg itself.
    returned: bool = field(default=False, repr=False)

    @property
    def total_chars(self) -> int:
        return len(self.summary) + sum(len(turn.text) for turn in self.turns)

    def __repr__(self) -> str:
        return (
            f"ConversationContext(summary_chars={len(self.summary)}, "
            f"turns={len(self.turns)}, chars={self.total_chars})"
        )

    __str__ = __repr__

    def continuation_preamble(self) -> str:
        """Private system text framing the ledger for the session being hydrated."""
        if self.returned:
            framing = (
                "The user continued this conversation on their phone and that call has "
                "now ended, so they are back with you in this session. Carry on "
                "naturally as one conversation, including whatever was settled on the "
                "phone."
            )
        else:
            framing = (
                "The user was just talking with you in another session and asked to "
                "continue this conversation on their phone. This phone call is that "
                "continuation, so pick up naturally where things left off."
            )
        lines = [
            "Continuation context (private, for your reference only).",
            framing,
            "A compact summary of the earlier part of the conversation and its most "
            "recent turns follow. Do not read this back to the user and do not repeat "
            "earlier details unless the user asks about them.",
            "",
        ]
        if self.summary:
            lines.append("[Earlier conversation, summarized]")
            lines.append(self.summary)
            lines.append("")
        lines.append("[Most recent turns]")
        for turn in self.turns:
            label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{label}: {turn.text}")
        lines.append("[End of earlier conversation]")
        return "\n".join(lines)


@dataclass(frozen=True)
class ReturnSyncClaim:
    """A reserved return marker: hydration context plus the token that settles it.

    The marker stays pending until :func:`ack_return_sync` is called with this
    token; :func:`release_return_sync` hands it back for the next turn to
    retry. Never prints its token or contents.
    """

    token: str = field(repr=False)
    context: ConversationContext = field(repr=False)

    def __repr__(self) -> str:
        return f"ReturnSyncClaim({self.context!r})"

    __str__ = __repr__


# --- helpers -----------------------------------------------------------------


def _now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _resolve_now(now: int | None) -> int:
    return _now() if now is None else int(now)


def new_conversation_id() -> str:
    """Return a fresh opaque identifier with no embedded meaning."""
    return secrets.token_urlsafe(_ID_BYTES)


def is_valid_conversation_id(value: object) -> bool:
    """Accept only the opaque shape this module generates."""
    return isinstance(value, str) and _CONVERSATION_ID.fullmatch(value) is not None


def _require_id(value: object) -> str:
    if not is_valid_conversation_id(value):
        raise ValueError("Conversation id has an invalid shape")
    return value  # type: ignore[return-value]


def _require_key(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError("Session key must be non-empty text")
    if not value.isprintable():
        raise ValueError("Session key must not contain control characters")
    return value.strip()


def bound_turn_text(text: str) -> str:
    """Normalize, redact, then truncate so a cut can never expose part of a secret."""
    text = redact_sensitive_text(" ".join(text.split()))
    if len(text) > MAX_TURN_CHARS:
        text = text[:MAX_TURN_CHARS]
    return text


def _summary_line(role: str, text: str) -> str:
    label = "User" if role == "user" else "Assistant"
    body = text if len(text) <= MAX_SUMMARY_LINE_CHARS else text[: MAX_SUMMARY_LINE_CHARS - 1] + "…"
    return f"{label}: {body}"


def _fold_summary(summary: str, folded: Iterable[tuple[str, str]]) -> str:
    """Append folded turns to the rolling summary, dropping the oldest lines first."""
    lines = summary.splitlines() if summary else []
    lines.extend(_summary_line(role, text) for role, text in folded)
    if len(lines) > MAX_SUMMARY_LINES:
        del lines[: len(lines) - MAX_SUMMARY_LINES]
    while lines and sum(len(line) + 1 for line in lines) - 1 > MAX_SUMMARY_CHARS:
        lines.pop(0)
    return "\n".join(lines)


def _connect() -> sqlite3.Connection:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(STORE_PATH, timeout=_BUSY_TIMEOUT_SECONDS, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout = {int(_BUSY_TIMEOUT_SECONDS * 1000)}")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_ledger (
            conversation_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL DEFAULT '',
            folded_turns INTEGER NOT NULL DEFAULT 0,
            next_seq INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_turns (
            conversation_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            role TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            PRIMARY KEY (conversation_id, seq)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_links (
            conversation_id TEXT NOT NULL,
            session_key TEXT NOT NULL,
            state TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY (conversation_id, session_key)
        )
        """
    )
    # One pending "the phone leg ended, catch up" marker per origin session.
    # ``session_key`` names the origin session that may consume it. A claimed
    # marker carries the claimant's token until it is acked or released.
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_return_sync (
            conversation_id TEXT NOT NULL,
            session_key TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            claimed_at INTEGER,
            claim_token TEXT,
            PRIMARY KEY (conversation_id, session_key)
        )
        """
    )
    _ensure_column(connection, "conversation_return_sync", "claimed_at", "INTEGER")
    _ensure_column(connection, "conversation_return_sync", "claim_token", "TEXT")
    # Multi-user: the opaque id of the user a conversation belongs to. Only a
    # continuation for the same user may ever link to or claim it.
    _ensure_column(connection, "conversation_ledger", "user_id", "TEXT")
    return connection


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    """Upgrade a table written before ``column`` existed."""
    columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _delete_conversation(connection: sqlite3.Connection, conversation_id: str) -> None:
    connection.execute(
        "DELETE FROM conversation_turns WHERE conversation_id = ?", (conversation_id,)
    )
    connection.execute(
        "DELETE FROM conversation_links WHERE conversation_id = ?", (conversation_id,)
    )
    connection.execute(
        "DELETE FROM conversation_return_sync WHERE conversation_id = ?", (conversation_id,)
    )
    connection.execute(
        "DELETE FROM conversation_ledger WHERE conversation_id = ?", (conversation_id,)
    )


def _has_live_link(connection: sqlite3.Connection, conversation_id: str, moment: int) -> bool:
    """An active link whose liveness is fresh, or a pending one inside its claim window."""
    row = connection.execute(
        """
        SELECT 1 FROM conversation_links
        WHERE conversation_id = ?
          AND ((state = ? AND updated_at >= ?) OR (state = ? AND created_at >= ?))
        LIMIT 1
        """,
        (
            conversation_id,
            _LINK_ACTIVE,
            moment - CONVERSATION_TTL_SECONDS,
            _LINK_PENDING,
            moment - CONTINUATION_CLAIM_TTL_SECONDS,
        ),
    ).fetchone()
    return row is not None


def _touch_link(
    connection: sqlite3.Connection, conversation_id: str, session_key: str, moment: int
) -> bool:
    """Refresh one active link's liveness. Caller holds the transaction."""
    touched = connection.execute(
        "UPDATE conversation_links SET updated_at = ? "
        "WHERE conversation_id = ? AND session_key = ? AND state = ?",
        (moment, conversation_id, session_key, _LINK_ACTIVE),
    )
    return touched.rowcount == 1


def _delete_if_unlinked(connection: sqlite3.Connection, conversation_id: str, moment: int) -> bool:
    if _has_live_link(connection, conversation_id, moment):
        return False
    _delete_conversation(connection, conversation_id)
    return True


def _close_link(
    connection: sqlite3.Connection, conversation_id: str, session_key: str, moment: int
) -> None:
    """Mark one link closed and drop any return marker only it could consume."""
    connection.execute(
        "UPDATE conversation_links SET state = ?, updated_at = ? "
        "WHERE conversation_id = ? AND session_key = ?",
        (_LINK_CLOSED, moment, conversation_id, session_key),
    )
    connection.execute(
        "DELETE FROM conversation_return_sync WHERE conversation_id = ? AND session_key = ?",
        (conversation_id, session_key),
    )


def _origin_link(connection: sqlite3.Connection, conversation_id: str) -> sqlite3.Row | None:
    """The link :func:`open_conversation` created: the earliest one for the conversation."""
    return connection.execute(
        """
        SELECT session_key, state FROM conversation_links
        WHERE conversation_id = ? ORDER BY created_at, rowid LIMIT 1
        """,
        (conversation_id,),
    ).fetchone()


def _read_hydration(
    connection: sqlite3.Connection, conversation_id: str, moment: int, *, returned: bool
) -> ConversationContext | None:
    """Read summary plus the latest turns and touch activity. Caller holds the transaction."""
    ledger = connection.execute(
        "SELECT summary FROM conversation_ledger WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    if ledger is None:
        return None
    rows = connection.execute(
        """
        SELECT role, text FROM conversation_turns
        WHERE conversation_id = ? ORDER BY seq DESC LIMIT ?
        """,
        (conversation_id, MAX_HYDRATION_TURNS),
    ).fetchall()
    connection.execute(
        "UPDATE conversation_ledger SET updated_at = ? WHERE conversation_id = ?",
        (moment, conversation_id),
    )
    return _hydration_context(
        ledger["summary"], [(r["role"], r["text"]) for r in rows][::-1], returned=returned
    )


# --- lifecycle ---------------------------------------------------------------


def open_conversation(
    *, session_key: str, now: int | None = None, user_id: str | None = None
) -> str:
    """Start a logical conversation owned by ``session_key``; return its opaque ID.

    ``user_id`` binds the conversation to a verified user; ``None`` keeps the
    legacy unscoped behaviour.
    """
    key = _require_key(session_key)
    owner = None if user_id is None else require_user_id(user_id)
    moment = _resolve_now(now)
    conversation_id = new_conversation_id()
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                """
                INSERT INTO conversation_ledger
                    (conversation_id, summary, folded_turns, next_seq, created_at, updated_at,
                     user_id)
                VALUES (?, '', 0, 0, ?, ?, ?)
                """,
                (conversation_id, moment, moment, owner),
            )
            connection.execute(
                """
                INSERT INTO conversation_links
                    (conversation_id, session_key, state, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, key, _LINK_ACTIVE, moment, moment),
            )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    logger.debug("Opened conversation ledger")
    return conversation_id


def conversation_exists(conversation_id: object) -> bool:
    if not isinstance(conversation_id, str) or not conversation_id:
        return False
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT 1 FROM conversation_ledger WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()
    return row is not None


def conversation_user_id(conversation_id: object) -> str | None:
    """The opaque user a live conversation belongs to; ``None`` if unscoped or unknown."""
    if not is_valid_conversation_id(conversation_id):
        return None
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT user_id FROM conversation_ledger WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    return row["user_id"] if row is not None else None


def append_turn(
    conversation_id: str,
    role: str,
    text: str,
    *,
    now: int | None = None,
    session_key: str | None = None,
) -> bool:
    """Append one completed visible turn; return whether it was stored.

    Anything other than non-empty ``user``/``assistant`` text is refused, as is
    an unknown conversation. Older turns fold into the rolling summary once the
    verbatim window is full, so the stored size stays bounded. With a
    ``session_key`` the turn also counts as that session's liveness.
    """
    if role not in _ALLOWED_ROLES or not isinstance(text, str):
        return False
    if not isinstance(conversation_id, str) or not conversation_id:
        return False
    bounded = bound_turn_text(text)
    if not bounded:
        return False
    key = _require_key(session_key) if session_key is not None else None
    moment = _resolve_now(now)

    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT summary, folded_turns, next_seq FROM conversation_ledger "
                "WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                connection.execute("ROLLBACK")
                return False
            seq = row["next_seq"]
            connection.execute(
                "INSERT INTO conversation_turns (conversation_id, seq, role, text, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (conversation_id, seq, role, bounded, moment),
            )
            summary, folded = _fold_overflow(connection, conversation_id, row)
            connection.execute(
                """
                UPDATE conversation_ledger
                SET summary = ?, folded_turns = ?, next_seq = ?, updated_at = ?
                WHERE conversation_id = ?
                """,
                (summary, folded, seq + 1, moment, conversation_id),
            )
            if key is not None:
                _touch_link(connection, conversation_id, key, moment)
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return True


def touch_session(conversation_id: object, *, session_key: str, now: int | None = None) -> bool:
    """Liveness heartbeat: refresh ``session_key``'s active link; return whether it was live.

    Only an active (opened or claimed) link can be refreshed; pending, closed
    and unknown links are refused quietly. Reads nothing.
    """
    if not is_valid_conversation_id(conversation_id):
        return False
    key = _require_key(session_key)
    moment = _resolve_now(now)
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            touched = _touch_link(connection, conversation_id, key, moment)
            if touched:
                connection.execute(
                    "UPDATE conversation_ledger SET updated_at = ? WHERE conversation_id = ?",
                    (moment, conversation_id),
                )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return touched


def _fold_overflow(
    connection: sqlite3.Connection, conversation_id: str, row: sqlite3.Row
) -> tuple[str, int]:
    """Move the oldest verbatim turns into the summary until the window fits."""
    turns = connection.execute(
        "SELECT seq, role, text FROM conversation_turns WHERE conversation_id = ? ORDER BY seq",
        (conversation_id,),
    ).fetchall()
    total = sum(len(turn["text"]) for turn in turns)
    folded: list[tuple[str, str]] = []
    index = 0
    while len(turns) - index > MAX_RECENT_TURNS or (
        total > MAX_RECENT_CHARS and len(turns) - index > 1
    ):
        turn = turns[index]
        folded.append((turn["role"], turn["text"]))
        total -= len(turn["text"])
        index += 1
    if not folded:
        return row["summary"], row["folded_turns"]
    connection.execute(
        "DELETE FROM conversation_turns WHERE conversation_id = ? AND seq < ?",
        (conversation_id, turns[index - 1]["seq"] + 1),
    )
    return _fold_summary(row["summary"], folded), row["folded_turns"] + len(folded)


def link_continuation(
    conversation_id: str,
    *,
    session_key: str,
    now: int | None = None,
    user_id: str | None = None,
) -> None:
    """Reserve a continuation slot for ``session_key`` (the outbound attempt).

    Called before dispatch. A pending link keeps the ledger alive after the
    origin session closes, until it is claimed, released, or its claim window
    lapses. Raises ``LookupError`` for an unknown conversation and
    ``PermissionError`` when ``user_id`` is not the conversation's owner (a
    legacy unscoped conversation may only be continued unscoped).
    """
    identifier = _require_id(conversation_id)
    key = _require_key(session_key)
    owner = None if user_id is None else require_user_id(user_id)
    moment = _resolve_now(now)
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            exists = connection.execute(
                "SELECT user_id FROM conversation_ledger WHERE conversation_id = ?", (identifier,)
            ).fetchone()
            if exists is not None and exists["user_id"] != owner:
                # The except clause below rolls the transaction back.
                raise PermissionError("Conversation belongs to a different user")
            if exists is not None:
                connection.execute(
                    """
                    INSERT INTO conversation_links
                        (conversation_id, session_key, state, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(conversation_id, session_key) DO UPDATE SET
                        state = excluded.state,
                        created_at = excluded.created_at,
                        updated_at = excluded.updated_at
                    """,
                    (identifier, key, _LINK_PENDING, moment, moment),
                )
                connection.execute(
                    "UPDATE conversation_ledger SET updated_at = ? WHERE conversation_id = ?",
                    (moment, identifier),
                )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    if exists is None:
        raise LookupError("No live conversation for that id")
    logger.debug("Linked conversation continuation")


def claim_continuation(
    conversation_id: object,
    *,
    session_key: str,
    now: int | None = None,
    user_id: str | None = None,
) -> ConversationContext | None:
    """Claim a pending continuation exactly once and return hydration context.

    Returns ``None`` when the conversation is unknown, the link is missing,
    already claimed, its claim window lapsed, or ``user_id`` is not the
    conversation's owner. This is the only read path that surfaces transcript
    content, and the outbound worker calls it only after AMD has positively
    classified a human.
    """
    if not is_valid_conversation_id(conversation_id):
        return None
    key = _require_key(session_key)
    owner = None if user_id is None else require_user_id(user_id)
    moment = _resolve_now(now)
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            owned = connection.execute(
                "SELECT user_id FROM conversation_ledger WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if owned is None or owned["user_id"] != owner:
                connection.execute("ROLLBACK")
                return None
            claimed = connection.execute(
                """
                UPDATE conversation_links SET state = ?, updated_at = ?
                WHERE conversation_id = ? AND session_key = ? AND state = ? AND created_at >= ?
                """,
                (
                    _LINK_ACTIVE,
                    moment,
                    conversation_id,
                    key,
                    _LINK_PENDING,
                    moment - CONTINUATION_CLAIM_TTL_SECONDS,
                ),
            )
            if claimed.rowcount != 1:
                connection.execute("ROLLBACK")
                return None
            context = _read_hydration(connection, conversation_id, moment, returned=False)
            if context is None:
                connection.execute("ROLLBACK")
                return None
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise

    logger.info(
        "Claimed conversation continuation turns=%d chars=%d",
        len(context.turns),
        context.total_chars,
    )
    return context


def _hydration_context(
    summary: str, turns: list[tuple[str, str]], *, returned: bool = False
) -> ConversationContext:
    """Bound the hydrated context, dropping the summary head and oldest turns first."""
    kept = [LedgerTurn(role=role, text=text) for role, text in turns]
    budget = MAX_HYDRATION_CHARS
    turn_chars = sum(len(turn.text) for turn in kept)
    while kept and turn_chars > budget:
        turn_chars -= len(kept[0].text)
        kept.pop(0)
    summary_lines = summary.splitlines() if summary else []
    while summary_lines and sum(len(line) + 1 for line in summary_lines) - 1 > budget - turn_chars:
        summary_lines.pop(0)
    return ConversationContext(
        summary="\n".join(summary_lines), turns=tuple(kept), returned=returned
    )


def mark_return_sync(
    conversation_id: object, *, phone_session_key: str, now: int | None = None
) -> bool:
    """Flag that the phone leg ended so the origin session catches up once.

    Called when an outbound leg ends, before its link is closed. The marker is
    written if and only if the origin session (the one that opened the
    conversation) still holds an active link and ``phone_session_key`` holds a
    claimed one; otherwise nothing is recorded. Reads no transcript. Returns
    whether a marker is now pending.
    """
    if not is_valid_conversation_id(conversation_id):
        return False
    phone_key = _require_key(phone_session_key)
    moment = _resolve_now(now)
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            phone = connection.execute(
                "SELECT 1 FROM conversation_links "
                "WHERE conversation_id = ? AND session_key = ? AND state = ?",
                (conversation_id, phone_key, _LINK_ACTIVE),
            ).fetchone()
            origin = _origin_link(connection, conversation_id)
            marked = (
                phone is not None
                and origin is not None
                and origin["state"] == _LINK_ACTIVE
                and origin["session_key"] != phone_key
            )
            if marked:
                # A newer return supersedes any claim still in flight, so an
                # ack for the older hydration can never consume these turns.
                connection.execute(
                    """
                    INSERT INTO conversation_return_sync
                        (conversation_id, session_key, created_at, claimed_at, claim_token)
                    VALUES (?, ?, ?, NULL, NULL)
                    ON CONFLICT(conversation_id, session_key) DO UPDATE SET
                        created_at = excluded.created_at,
                        claimed_at = NULL,
                        claim_token = NULL
                    """,
                    (conversation_id, origin["session_key"], moment),
                )
                connection.execute(
                    "UPDATE conversation_ledger SET updated_at = ? WHERE conversation_id = ?",
                    (moment, conversation_id),
                )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    logger.debug("Conversation return sync marked=%s", marked)
    return marked


def claim_return_sync(
    conversation_id: object, *, session_key: str, now: int | None = None
) -> ReturnSyncClaim | None:
    """Reserve the pending return marker for ``session_key`` and read its context.

    The marker is not consumed: it stays reserved under the returned token
    until :func:`ack_return_sync` (applied) or :func:`release_return_sync`
    (retry next turn); a reservation older than
    :data:`RETURN_SYNC_CLAIM_TTL_SECONDS` may be taken over, so a claimant that
    died never strands it. Returns ``None`` when nothing is pending, another
    claim is in flight, or the origin link is no longer active (in which case
    the marker is dropped). The reserve-and-read is one transaction, so
    concurrent turns can never both hydrate.
    """
    if not is_valid_conversation_id(conversation_id):
        return None
    key = _require_key(session_key)
    moment = _resolve_now(now)
    stale_claim = moment - RETURN_SYNC_CLAIM_TTL_SECONDS
    with closing(_connect()) as connection:
        # Every user turn asks; skip the write lock when nothing is claimable.
        pending = connection.execute(
            """
            SELECT 1 FROM conversation_return_sync
            WHERE conversation_id = ? AND session_key = ?
              AND (claimed_at IS NULL OR claimed_at < ?)
            """,
            (conversation_id, key, stale_claim),
        ).fetchone()
        if pending is None:
            return None
        token = secrets.token_urlsafe(_RETURN_SYNC_TOKEN_BYTES)
        connection.execute("BEGIN IMMEDIATE")
        try:
            claimed = connection.execute(
                """
                UPDATE conversation_return_sync SET claimed_at = ?, claim_token = ?
                WHERE conversation_id = ? AND session_key = ?
                  AND (claimed_at IS NULL OR claimed_at < ?)
                """,
                (moment, token, conversation_id, key, stale_claim),
            )
            if claimed.rowcount != 1:
                connection.execute("ROLLBACK")
                return None
            context = None
            if _touch_link(connection, conversation_id, key, moment):
                context = _read_hydration(connection, conversation_id, moment, returned=True)
            if context is None:
                # Only an active origin link may ever consume the marker.
                connection.execute(
                    "DELETE FROM conversation_return_sync "
                    "WHERE conversation_id = ? AND session_key = ?",
                    (conversation_id, key),
                )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    if context is None:
        logger.debug("Conversation return sync dropped: origin session no longer live")
        return None
    logger.info(
        "Claimed conversation return sync turns=%d chars=%d",
        len(context.turns),
        context.total_chars,
    )
    return ReturnSyncClaim(token=token, context=context)


def _settle_return_sync(
    conversation_id: object, *, session_key: str, token: str, consume: bool
) -> bool:
    if not is_valid_conversation_id(conversation_id):
        return False
    key = _require_key(session_key)
    if not isinstance(token, str) or not token:
        return False
    if consume:
        statement = (
            "DELETE FROM conversation_return_sync "
            "WHERE conversation_id = ? AND session_key = ? AND claim_token = ?"
        )
    else:
        statement = (
            "UPDATE conversation_return_sync SET claimed_at = NULL, claim_token = NULL "
            "WHERE conversation_id = ? AND session_key = ? AND claim_token = ?"
        )
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            settled = connection.execute(statement, (conversation_id, key, token))
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    return settled.rowcount == 1


def ack_return_sync(
    conversation_id: object, *, session_key: str, token: str, now: int | None = None
) -> bool:
    """Consume a claimed marker once its context has been applied.

    Only the claim's own token can ack it; a stale, superseded or replayed
    token is a no-op. Returns whether the marker was consumed.
    """
    del now  # accepted for symmetry with the other lifecycle calls
    acked = _settle_return_sync(conversation_id, session_key=session_key, token=token, consume=True)
    logger.debug("Conversation return sync acked=%s", acked)
    return acked


def release_return_sync(
    conversation_id: object, *, session_key: str, token: str, now: int | None = None
) -> bool:
    """Hand a claimed marker back so the next turn retries it.

    Called when applying the context failed part-way. Only the claim's own
    token can release it. Returns whether the marker is pending again.
    """
    del now
    released = _settle_return_sync(
        conversation_id, session_key=session_key, token=token, consume=False
    )
    logger.debug("Conversation return sync released=%s", released)
    return released


def consume_return_sync(
    conversation_id: object, *, session_key: str, now: int | None = None
) -> ConversationContext | None:
    """Claim and immediately ack the pending marker; return its context.

    Convenience for callers that apply the context synchronously. Callers that
    apply it asynchronously should use :func:`claim_return_sync` and ack or
    release the claim themselves so a failure is retried, never lost.
    """
    claim = claim_return_sync(conversation_id, session_key=session_key, now=now)
    if claim is None:
        return None
    ack_return_sync(conversation_id, session_key=session_key, token=claim.token, now=now)
    return claim.context


def release_continuation(
    conversation_id: object, *, session_key: str, now: int | None = None
) -> None:
    """Drop a continuation that will never be claimed (voicemail, IVR, failed dial).

    Reads nothing. If no other session still holds the conversation, it is
    deleted. Unknown conversations are a harmless no-op.
    """
    if not is_valid_conversation_id(conversation_id):
        return
    key = _require_key(session_key)
    moment = _resolve_now(now)
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            _close_link(connection, conversation_id, key, moment)
            _delete_if_unlinked(connection, conversation_id, moment)
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    logger.debug("Released conversation continuation")


def close_session(conversation_id: object, *, session_key: str, now: int | None = None) -> bool:
    """Mark one linked session closed; delete the ledger if none remain live.

    Returns whether the conversation was deleted. A pending continuation that
    is still inside its claim window keeps the ledger alive, so an outbound
    worker can always claim after the origin session has gone.
    """
    if not is_valid_conversation_id(conversation_id):
        return False
    key = _require_key(session_key)
    moment = _resolve_now(now)
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            _close_link(connection, conversation_id, key, moment)
            deleted = _delete_if_unlinked(connection, conversation_id, moment)
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    logger.info("Closed conversation session deleted=%s", deleted)
    return deleted


def purge_expired_conversations(*, now: int | None = None) -> int:
    """Close links whose session went silent, then delete conversations with no live link.

    A session that is still open keeps its link fresh through recorded turns
    and :func:`touch_session` heartbeats, so an active conversation is never
    deleted from under it however long it runs. Returns the number deleted.
    """
    moment = _resolve_now(now)
    removed = 0
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            # Claims never taken, and sessions that neither closed nor heartbeat.
            connection.execute(
                "UPDATE conversation_links SET state = ?, updated_at = ? "
                "WHERE state = ? AND created_at < ?",
                (_LINK_CLOSED, moment, _LINK_PENDING, moment - CONTINUATION_CLAIM_TTL_SECONDS),
            )
            connection.execute(
                "UPDATE conversation_links SET state = ?, updated_at = ? "
                "WHERE state = ? AND updated_at < ?",
                (_LINK_CLOSED, moment, _LINK_ACTIVE, moment - CONVERSATION_TTL_SECONDS),
            )
            orphans = connection.execute(
                """
                SELECT conversation_id FROM conversation_ledger
                WHERE conversation_id NOT IN (
                    SELECT conversation_id FROM conversation_links WHERE state = ?
                )
                """,
                (_LINK_ACTIVE,),
            ).fetchall()
            for row in orphans:
                if _delete_if_unlinked(connection, row["conversation_id"], moment):
                    removed += 1
            # A return marker is only ever consumable by an active origin link.
            connection.execute(
                """
                DELETE FROM conversation_return_sync WHERE NOT EXISTS (
                    SELECT 1 FROM conversation_links AS link
                    WHERE link.conversation_id = conversation_return_sync.conversation_id
                      AND link.session_key = conversation_return_sync.session_key
                      AND link.state = ?
                )
                """,
                (_LINK_ACTIVE,),
            )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    if removed:
        logger.info("Purged expired conversations count=%d", removed)
    return removed


# --- capture -----------------------------------------------------------------


def _message_text(item: Any) -> str | None:
    content = getattr(item, "content", None)
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts = [part for part in content if isinstance(part, str)]
    return "\n".join(parts) if parts else None


class ConversationRecorder:
    """Append completed visible turns from a LiveKit session to the ledger.

    Inert until :meth:`bind` names a conversation. Skips anything that is not
    user/assistant text and any assistant reply listed in
    ``exclude_assistant_texts`` (the handoff's own control replies). Storage
    failures are swallowed with a content-free warning: recording must never
    disturb the live session.
    """

    def __init__(
        self, *, exclude_assistant_texts: Iterable[str] = (), session_key: str | None = None
    ) -> None:
        self._conversation_id: str | None = None
        self._session_key = _require_key(session_key) if session_key is not None else None
        self._skip_assistant = {" ".join(text.split()) for text in exclude_assistant_texts}
        self.recorded = 0

    @property
    def conversation_id(self) -> str | None:
        return self._conversation_id

    def bind(self, conversation_id: str, *, session_key: str | None = None) -> None:
        """Name the conversation (and optionally the session whose liveness turns count for)."""
        self._conversation_id = _require_id(conversation_id)
        if session_key is not None:
            self._session_key = _require_key(session_key)

    def record(self, item: Any) -> bool:
        """Record one history item; return whether it was appended."""
        if self._conversation_id is None:
            return False
        if getattr(item, "type", None) != "message":
            return False
        role = getattr(item, "role", None)
        if role not in _ALLOWED_ROLES:
            return False
        text = _message_text(item)
        if text is None:
            return False
        normalized = " ".join(text.split())
        if not normalized:
            return False
        if role == "assistant" and normalized in self._skip_assistant:
            return False
        try:
            stored = append_turn(
                self._conversation_id, role, normalized, session_key=self._session_key
            )
        except Exception:
            logger.warning("Conversation ledger append failed", exc_info=False)
            return False
        if stored:
            self.recorded += 1
        return stored
