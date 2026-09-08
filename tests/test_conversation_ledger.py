"""Coverage for the private durable conversation ledger behind phone continuity.

The ledger keeps the *logical* conversation alive across a web/voice session
and its outbound phone continuation, and only for as long as that conversation
is live. It is not long-term memory: nothing in it is ever promoted to the
explicit preference store, and Hermes alone decides what deserves remembering.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from types import SimpleNamespace

import pytest
from livekit.agents import llm

from caal import conversation_ledger
from caal.conversation_ledger import (
    CONTINUATION_CLAIM_TTL_SECONDS,
    CONVERSATION_TTL_SECONDS,
    HERMES_LONG_TERM_MEMORY_SEAM,
    MAX_HYDRATION_CHARS,
    MAX_HYDRATION_TURNS,
    MAX_RECENT_CHARS,
    MAX_RECENT_TURNS,
    MAX_SUMMARY_CHARS,
    MAX_TURN_CHARS,
    RETURN_SYNC_CLAIM_TTL_SECONDS,
    ConversationContext,
    ConversationRecorder,
    ack_return_sync,
    append_turn,
    claim_continuation,
    claim_return_sync,
    close_session,
    consume_return_sync,
    conversation_exists,
    is_valid_conversation_id,
    link_continuation,
    mark_return_sync,
    open_conversation,
    purge_expired_conversations,
    release_continuation,
    release_return_sync,
    touch_session,
)
from caal.handoff_context import CONTINUATION_MESSAGE_ID, REDACTED, restore_conversation_context
from caal.tools import memory_tools

ORIGIN = "web-room-1"
PHONE = "attempt-abc"
SECRET = "hunter2-super-secret"


@pytest.fixture(autouse=True)
def store(monkeypatch, tmp_path):
    """Point the ledger and the preference store at one isolated SQLite file."""
    path = tmp_path / "assistant.sqlite3"
    monkeypatch.setattr(conversation_ledger, "STORE_PATH", path)
    monkeypatch.setattr(memory_tools, "STORE_PATH", path)
    return path


def _fill(conversation_id: str, count: int, *, now: int = 1000) -> None:
    for index in range(count):
        role = "user" if index % 2 == 0 else "assistant"
        append_turn(conversation_id, role, f"turn {index}", now=now + index)


# --- identity ----------------------------------------------------------------


def test_open_conversation_returns_an_opaque_unique_id() -> None:
    first = open_conversation(session_key=ORIGIN, now=1000)
    second = open_conversation(session_key="web-room-2", now=1000)

    assert first != second
    assert len(first) >= 24
    assert is_valid_conversation_id(first)
    assert conversation_exists(first)


@pytest.mark.parametrize(
    "value", ["", "   ", None, 42, "has space", "a/b", "x" * 200, "semi;colon", "\x00"]
)
def test_conversation_id_validation_rejects_untrusted_shapes(value: object) -> None:
    assert is_valid_conversation_id(value) is False


# --- append, redaction, bounds ----------------------------------------------


def test_append_keeps_only_user_and_assistant_text_in_order() -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)

    assert append_turn(conversation_id, "user", "How long should the bread proof?", now=1001)
    assert append_turn(conversation_id, "assistant", "About an hour.", now=1002)
    assert append_turn(conversation_id, "system", "operator prompt", now=1003) is False
    assert append_turn(conversation_id, "tool", '{"events": []}', now=1004) is False
    assert append_turn(conversation_id, "user", "   ", now=1005) is False

    link_continuation(conversation_id, session_key=PHONE, now=1006)
    context = claim_continuation(conversation_id, session_key=PHONE, now=1007)

    assert context is not None
    assert [(turn.role, turn.text) for turn in context.turns] == [
        ("user", "How long should the bread proof?"),
        ("assistant", "About an hour."),
    ]
    assert "operator prompt" not in json.dumps(context.continuation_preamble())


def test_append_redacts_credentials_before_they_touch_disk(store) -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)

    append_turn(conversation_id, "user", f"my wifi password is {SECRET} ok", now=1001)
    append_turn(conversation_id, "assistant", "Noted, the garage code is 8675309.", now=1002)

    raw = sqlite3.connect(store)
    stored = " ".join(row[0] for row in raw.execute("SELECT text FROM conversation_turns"))
    raw.close()
    assert SECRET not in stored
    assert "8675309" not in stored
    assert REDACTED in stored


def test_append_bounds_characters_per_turn() -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)

    append_turn(conversation_id, "user", "x" * (MAX_TURN_CHARS * 4), now=1001)
    link_continuation(conversation_id, session_key=PHONE, now=1002)
    context = claim_continuation(conversation_id, session_key=PHONE, now=1003)

    assert context is not None
    assert all(len(turn.text) <= MAX_TURN_CHARS for turn in context.turns)


def test_append_ignores_unknown_conversations() -> None:
    assert append_turn("does-not-exist", "user", "hello", now=1000) is False


def test_long_conversation_folds_old_turns_into_a_bounded_rolling_summary(store) -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)

    _fill(conversation_id, MAX_RECENT_TURNS * 6)

    raw = sqlite3.connect(store)
    recent = raw.execute(
        "SELECT COUNT(*), COALESCE(SUM(LENGTH(text)), 0) FROM conversation_turns"
    ).fetchone()
    summary = raw.execute("SELECT summary FROM conversation_ledger").fetchone()[0]
    raw.close()

    assert recent[0] <= MAX_RECENT_TURNS
    assert recent[1] <= MAX_RECENT_CHARS
    assert summary
    assert len(summary) <= MAX_SUMMARY_CHARS
    # The summary carries the older material the verbatim window dropped.
    assert "turn 0" not in summary  # the very oldest lines age out of the summary too
    assert any(f"turn {index}" in summary for index in range(40, 60))


def test_total_stored_size_stays_bounded_no_matter_how_long_the_conversation(store) -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)

    for index in range(MAX_RECENT_TURNS * 4):
        append_turn(conversation_id, "user", "y" * MAX_TURN_CHARS, now=1000 + index)

    raw = sqlite3.connect(store)
    recent_chars = raw.execute("SELECT COALESCE(SUM(LENGTH(text)), 0) FROM conversation_turns")
    recent_chars = recent_chars.fetchone()[0]
    summary_chars = raw.execute("SELECT LENGTH(summary) FROM conversation_ledger").fetchone()[0]
    raw.close()

    assert recent_chars <= MAX_RECENT_CHARS
    assert summary_chars <= MAX_SUMMARY_CHARS


# --- hydration ---------------------------------------------------------------


def test_claim_hydrates_summary_plus_recent_verbatim_turns_within_bounds() -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    _fill(conversation_id, MAX_RECENT_TURNS * 3)
    link_continuation(conversation_id, session_key=PHONE, now=2000)

    context = claim_continuation(conversation_id, session_key=PHONE, now=2001)

    assert context is not None
    assert isinstance(context, ConversationContext)
    assert context.summary
    assert 0 < len(context.turns) <= MAX_HYDRATION_TURNS
    assert context.turns[-1].text == f"turn {MAX_RECENT_TURNS * 3 - 1}"
    assert context.total_chars <= MAX_HYDRATION_CHARS
    preamble = context.continuation_preamble()
    assert context.summary.splitlines()[0] in preamble
    assert context.turns[-1].text in preamble
    lowered = preamble.lower()
    assert "phone" in lowered
    assert "do not read" in lowered or "never read" in lowered
    assert preamble.index(context.summary.splitlines()[0]) < preamble.index(context.turns[-1].text)


def test_hydration_prefers_the_latest_turns_when_trimming() -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    for index in range(MAX_HYDRATION_TURNS + 4):
        append_turn(conversation_id, "assistant", "z" * MAX_TURN_CHARS, now=1000 + index)
    append_turn(conversation_id, "user", "the newest question", now=2000)
    link_continuation(conversation_id, session_key=PHONE, now=2001)

    context = claim_continuation(conversation_id, session_key=PHONE, now=2002)

    assert context is not None
    assert context.turns[-1].text == "the newest question"
    assert context.total_chars <= MAX_HYDRATION_CHARS


def test_context_never_prints_its_contents() -> None:
    context = ConversationContext(
        summary="User: private summary",
        turns=(conversation_ledger.LedgerTurn(role="user", text="private detail"),),
    )

    for rendered in (repr(context), str(context), f"{context}"):
        assert "private detail" not in rendered
        assert "private summary" not in rendered


@pytest.mark.asyncio
async def test_hydrated_context_injects_once_through_the_existing_restore_seam(caplog) -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    append_turn(conversation_id, "user", "let's plan the garage renovation", now=1001)
    link_continuation(conversation_id, session_key=PHONE, now=1002)
    context = claim_continuation(conversation_id, session_key=PHONE, now=1003)
    assert context is not None

    class _Agent:
        def __init__(self) -> None:
            self.chat_ctx = llm.ChatContext.empty()
            self.updates = 0

        async def update_chat_ctx(self, chat_ctx):
            self.chat_ctx = chat_ctx
            self.updates += 1

    agent = _Agent()
    with caplog.at_level(logging.DEBUG):
        assert await restore_conversation_context(agent, context) is True
        assert await restore_conversation_context(agent, context) is False

    assert agent.updates == 1
    injected = agent.chat_ctx.get_by_id(CONTINUATION_MESSAGE_ID)
    assert injected is not None and injected.role == "system"
    assert "garage renovation" in injected.text_content
    assert all("garage renovation" not in record.getMessage() for record in caplog.records)


# --- lifecycle ---------------------------------------------------------------


def test_closing_the_only_session_deletes_the_conversation(store) -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    append_turn(conversation_id, "user", "private planning detail", now=1001)

    assert close_session(conversation_id, session_key=ORIGIN, now=1002) is True

    assert conversation_exists(conversation_id) is False
    raw = sqlite3.connect(store)
    assert raw.execute("SELECT COUNT(*) FROM conversation_turns").fetchone()[0] == 0
    assert raw.execute("SELECT COUNT(*) FROM conversation_links").fetchone()[0] == 0
    raw.close()


def test_origin_close_keeps_the_ledger_while_a_continuation_is_pending_claim() -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    append_turn(conversation_id, "user", "where were we", now=1001)
    link_continuation(conversation_id, session_key=PHONE, now=1002)

    # The web tab closes before the phone even rings.
    assert close_session(conversation_id, session_key=ORIGIN, now=1003) is False
    assert conversation_exists(conversation_id) is True

    # The outbound worker can still claim it once a human answers.
    context = claim_continuation(conversation_id, session_key=PHONE, now=1010)
    assert context is not None
    assert [turn.text for turn in context.turns] == ["where were we"]

    # The phone session keeps appending, and closing it ends the conversation.
    assert append_turn(conversation_id, "assistant", "We were planning.", now=1011)
    assert close_session(conversation_id, session_key=PHONE, now=1012) is True
    assert conversation_exists(conversation_id) is False


def test_phone_close_keeps_the_ledger_while_the_origin_session_is_still_live() -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    link_continuation(conversation_id, session_key=PHONE, now=1001)
    claim_continuation(conversation_id, session_key=PHONE, now=1002)

    assert close_session(conversation_id, session_key=PHONE, now=1003) is False
    assert conversation_exists(conversation_id) is True
    assert close_session(conversation_id, session_key=ORIGIN, now=1004) is True
    assert conversation_exists(conversation_id) is False


def test_continuation_can_only_be_claimed_once_and_only_while_pending() -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    append_turn(conversation_id, "user", "hello", now=1001)

    assert claim_continuation(conversation_id, session_key=PHONE, now=1002) is None
    link_continuation(conversation_id, session_key=PHONE, now=1003)
    assert claim_continuation(conversation_id, session_key="attempt-other", now=1004) is None
    assert claim_continuation(conversation_id, session_key=PHONE, now=1004) is not None
    assert claim_continuation(conversation_id, session_key=PHONE, now=1005) is None
    assert claim_continuation("unknown", session_key=PHONE, now=1005) is None


def test_unclaimed_continuation_expires_and_no_longer_holds_the_ledger_open() -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    link_continuation(conversation_id, session_key=PHONE, now=1001)
    close_session(conversation_id, session_key=ORIGIN, now=1002)
    assert conversation_exists(conversation_id) is True

    late = 1001 + CONTINUATION_CLAIM_TTL_SECONDS + 1
    assert claim_continuation(conversation_id, session_key=PHONE, now=late) is None
    purge_expired_conversations(now=late)

    assert conversation_exists(conversation_id) is False


def test_release_drops_a_pending_continuation_without_reading_it(store) -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    link_continuation(conversation_id, session_key=PHONE, now=1001)
    close_session(conversation_id, session_key=ORIGIN, now=1002)

    release_continuation(conversation_id, session_key=PHONE, now=1003)

    assert conversation_exists(conversation_id) is False
    # Releasing an unknown continuation is a harmless no-op.
    release_continuation("unknown", session_key=PHONE, now=1003)


def test_stale_conversations_are_ttl_cleaned_even_with_lingering_links() -> None:
    fresh = open_conversation(session_key="fresh-room", now=5000)
    stale = open_conversation(session_key=ORIGIN, now=1000)
    append_turn(stale, "user", "hello", now=1001)

    removed = purge_expired_conversations(now=1001 + CONVERSATION_TTL_SECONDS + 1)

    assert removed == 1
    assert conversation_exists(stale) is False
    assert conversation_exists(fresh) is True


def test_linking_an_unknown_conversation_fails_loudly() -> None:
    with pytest.raises(LookupError):
        link_continuation("unknown", session_key=PHONE, now=1000)


def test_claim_touches_activity_so_a_live_phone_call_is_not_ttl_cleaned() -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    append_turn(conversation_id, "user", "hello", now=1001)
    later = 1001 + CONVERSATION_TTL_SECONDS - 10
    link_continuation(conversation_id, session_key=PHONE, now=later)
    claim_continuation(conversation_id, session_key=PHONE, now=later + 1)

    purge_expired_conversations(now=later + 20)

    assert conversation_exists(conversation_id) is True


# --- liveness keeps an active session's ledger --------------------------------


def test_active_origin_that_keeps_heartbeating_outlives_the_idle_ttl(store) -> None:
    """Regression: a web tab open for more than six hours must not lose its ledger."""
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    append_turn(conversation_id, "user", "let's plan the garage renovation", now=1001)
    beat = 1001
    # No turns for a long time; only liveness heartbeats arrive, well apart.
    for _ in range(4):
        beat += CONVERSATION_TTL_SECONDS // 2
        assert touch_session(conversation_id, session_key=ORIGIN, now=beat) is True
        assert purge_expired_conversations(now=beat + 1) == 0
        assert conversation_exists(conversation_id) is True

    # The origin can still hand off, get its phone leg back and consume it.
    _phone_leg(conversation_id, now=beat + 2)
    assert mark_return_sync(conversation_id, phone_session_key=PHONE, now=beat + 10) is True
    assert close_session(conversation_id, session_key=PHONE, now=beat + 11) is False
    assert consume_return_sync(conversation_id, session_key=ORIGIN, now=beat + 12) is not None
    # And once every link ends, the ledger is deleted immediately as before.
    assert close_session(conversation_id, session_key=ORIGIN, now=beat + 13) is True
    assert conversation_exists(conversation_id) is False
    raw = sqlite3.connect(store)
    assert raw.execute("SELECT COUNT(*) FROM conversation_links").fetchone()[0] == 0
    raw.close()


def test_turns_recorded_for_a_session_count_as_its_liveness() -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    moment = 1000
    for index in range(3):
        moment += CONVERSATION_TTL_SECONDS - 60
        append_turn(conversation_id, "user", f"turn {index}", now=moment, session_key=ORIGIN)
        assert purge_expired_conversations(now=moment + 30) == 0

    assert conversation_exists(conversation_id) is True


def test_recorder_bound_to_a_session_keeps_that_session_live(monkeypatch) -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    recorder = ConversationRecorder(session_key=ORIGIN)
    recorder.bind(conversation_id)
    late = 1000 + CONVERSATION_TTL_SECONDS - 5
    monkeypatch.setattr(conversation_ledger, "_now", lambda: late)

    assert recorder.record(_message("user", "still here")) is True

    assert purge_expired_conversations(now=late + 10) == 0
    assert conversation_exists(conversation_id) is True


def test_silent_origin_link_expires_and_the_ledger_goes_with_it(store) -> None:
    """A session that never closed cleanly and never heartbeats is still cleaned."""
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    touch_session(conversation_id, session_key=ORIGIN, now=2000)

    assert purge_expired_conversations(now=2000 + CONVERSATION_TTL_SECONDS - 1) == 0
    assert conversation_exists(conversation_id) is True
    assert purge_expired_conversations(now=2000 + CONVERSATION_TTL_SECONDS + 1) == 1
    assert conversation_exists(conversation_id) is False


def test_live_phone_leg_keeps_the_ledger_after_a_silent_origin_expired() -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    append_turn(conversation_id, "user", "hello", now=1001)
    late = 1000 + CONVERSATION_TTL_SECONDS + 100
    # The phone leg is dispatched and claimed on the wall clock, long after the
    # silent origin's liveness lapsed; the call itself keeps heartbeating.
    link_continuation(conversation_id, session_key=PHONE, now=late)
    assert claim_continuation(conversation_id, session_key=PHONE, now=late + 1) is not None
    assert purge_expired_conversations(now=late + 2) == 0
    assert conversation_exists(conversation_id) is True
    touch_session(conversation_id, session_key=PHONE, now=late + CONVERSATION_TTL_SECONDS - 10)
    assert purge_expired_conversations(now=late + CONVERSATION_TTL_SECONDS + 10) == 0
    assert conversation_exists(conversation_id) is True

    # The origin lapsed, so the phone ending is the last live link.
    assert close_session(
        conversation_id, session_key=PHONE, now=late + CONVERSATION_TTL_SECONDS + 11
    )
    assert conversation_exists(conversation_id) is False


def test_touch_refuses_closed_links_and_unknown_conversations() -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    link_continuation(conversation_id, session_key=PHONE, now=1001)

    # A pending (unclaimed) continuation is not a live session yet.
    assert touch_session(conversation_id, session_key=PHONE, now=1002) is False
    assert touch_session(conversation_id, session_key="web-room-2", now=1002) is False
    assert touch_session("unknown", session_key=ORIGIN, now=1002) is False
    assert touch_session("has space", session_key=ORIGIN, now=1002) is False
    close_session(conversation_id, session_key=ORIGIN, now=1003)
    assert touch_session(conversation_id, session_key=ORIGIN, now=1004) is False


# --- return sync -------------------------------------------------------------


def _phone_leg(conversation_id: str, *, phone: str = PHONE, now: int = 1010) -> None:
    """Claim a continuation for ``phone`` and speak two turns on it."""
    link_continuation(conversation_id, session_key=phone, now=now)
    assert claim_continuation(conversation_id, session_key=phone, now=now + 1) is not None
    append_turn(conversation_id, "user", "on the phone: pick the oak flooring", now=now + 2)
    append_turn(conversation_id, "assistant", "Oak it is, noted on the phone.", now=now + 3)


def _marker_rows(store) -> int:
    raw = sqlite3.connect(store)
    try:
        return raw.execute("SELECT COUNT(*) FROM conversation_return_sync").fetchone()[0]
    finally:
        raw.close()


def test_phone_leg_end_marks_a_one_time_return_sync_for_a_live_origin() -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    append_turn(conversation_id, "user", "let's plan the garage renovation", now=1001)
    _phone_leg(conversation_id)

    assert mark_return_sync(conversation_id, phone_session_key=PHONE, now=1020) is True
    assert close_session(conversation_id, session_key=PHONE, now=1021) is False

    context = consume_return_sync(conversation_id, session_key=ORIGIN, now=1022)

    assert context is not None
    assert isinstance(context, ConversationContext)
    assert context.returned is True
    texts = [turn.text for turn in context.turns]
    assert texts[-2:] == ["on the phone: pick the oak flooring", "Oak it is, noted on the phone."]
    assert texts[0] == "let's plan the garage renovation"
    assert context.total_chars <= MAX_HYDRATION_CHARS
    # One-time: the very next turn must not hydrate again.
    assert consume_return_sync(conversation_id, session_key=ORIGIN, now=1023) is None
    assert conversation_exists(conversation_id) is True


def test_return_sync_is_not_marked_when_the_origin_already_closed(store) -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    _phone_leg(conversation_id)
    assert close_session(conversation_id, session_key=ORIGIN, now=1015) is False

    assert mark_return_sync(conversation_id, phone_session_key=PHONE, now=1020) is False
    assert _marker_rows(store) == 0
    assert consume_return_sync(conversation_id, session_key=ORIGIN, now=1021) is None
    # Nothing holds the ledger open once the phone leg ends.
    assert close_session(conversation_id, session_key=PHONE, now=1022) is True
    assert conversation_exists(conversation_id) is False


def test_return_sync_is_only_marked_by_a_claimed_phone_link(store) -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)

    # Never linked, still pending, unknown, and malformed all refuse quietly.
    assert mark_return_sync(conversation_id, phone_session_key="attempt-never", now=1001) is False
    link_continuation(conversation_id, session_key=PHONE, now=1002)
    assert mark_return_sync(conversation_id, phone_session_key=PHONE, now=1003) is False
    assert mark_return_sync("unknown", phone_session_key=PHONE, now=1003) is False
    assert mark_return_sync("has space", phone_session_key=PHONE, now=1003) is False
    # The origin cannot mark a return to itself.
    assert mark_return_sync(conversation_id, phone_session_key=ORIGIN, now=1003) is False
    assert _marker_rows(store) == 0


def test_origin_close_after_marking_discards_the_marker_and_the_ledger(store) -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    _phone_leg(conversation_id)
    assert mark_return_sync(conversation_id, phone_session_key=PHONE, now=1020) is True
    close_session(conversation_id, session_key=PHONE, now=1021)
    assert _marker_rows(store) == 1

    # The web tab closes before its next turn ever arrives.
    assert close_session(conversation_id, session_key=ORIGIN, now=1022) is True

    assert _marker_rows(store) == 0
    assert conversation_exists(conversation_id) is False
    assert consume_return_sync(conversation_id, session_key=ORIGIN, now=1023) is None


def test_return_sync_can_only_be_consumed_by_the_origin_session(store) -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    _phone_leg(conversation_id)
    assert mark_return_sync(conversation_id, phone_session_key=PHONE, now=1020) is True

    assert consume_return_sync(conversation_id, session_key=PHONE, now=1021) is None
    assert consume_return_sync(conversation_id, session_key="web-room-2", now=1021) is None
    assert consume_return_sync("unknown", session_key=ORIGIN, now=1021) is None
    assert _marker_rows(store) == 1
    assert consume_return_sync(conversation_id, session_key=ORIGIN, now=1022) is not None
    assert _marker_rows(store) == 0


def test_expired_origin_never_consumes_and_purge_drops_the_marker(store) -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    _phone_leg(conversation_id)
    assert mark_return_sync(conversation_id, phone_session_key=PHONE, now=1020) is True
    close_session(conversation_id, session_key=PHONE, now=1021)

    late = 1021 + CONVERSATION_TTL_SECONDS + 1
    assert purge_expired_conversations(now=late) == 1

    assert conversation_exists(conversation_id) is False
    assert _marker_rows(store) == 0
    assert consume_return_sync(conversation_id, session_key=ORIGIN, now=late + 1) is None


def test_purge_drops_a_marker_whose_origin_link_is_no_longer_active(store) -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    _phone_leg(conversation_id)
    assert mark_return_sync(conversation_id, phone_session_key=PHONE, now=1020) is True
    raw = sqlite3.connect(store)
    raw.execute("UPDATE conversation_links SET state = 'closed' WHERE session_key = ?", (ORIGIN,))
    raw.commit()
    raw.close()

    purge_expired_conversations(now=1021)

    assert _marker_rows(store) == 0
    assert consume_return_sync(conversation_id, session_key=ORIGIN, now=1022) is None


def test_each_phone_leg_marks_its_own_return_sync(store) -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    _phone_leg(conversation_id, phone="attempt-1", now=1010)
    assert mark_return_sync(conversation_id, phone_session_key="attempt-1", now=1020) is True
    close_session(conversation_id, session_key="attempt-1", now=1021)
    assert consume_return_sync(conversation_id, session_key=ORIGIN, now=1022) is not None

    _phone_leg(conversation_id, phone="attempt-2", now=1030)
    assert mark_return_sync(conversation_id, phone_session_key="attempt-2", now=1040) is True
    # Marking twice for the same leg keeps a single pending marker.
    assert mark_return_sync(conversation_id, phone_session_key="attempt-2", now=1041) is True
    assert _marker_rows(store) == 1
    close_session(conversation_id, session_key="attempt-2", now=1042)

    context = consume_return_sync(conversation_id, session_key=ORIGIN, now=1043)
    assert context is not None
    assert consume_return_sync(conversation_id, session_key=ORIGIN, now=1044) is None


def test_returned_context_frames_the_phone_return_privately_and_never_prints() -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    _phone_leg(conversation_id)
    assert mark_return_sync(conversation_id, phone_session_key=PHONE, now=1020) is True
    context = consume_return_sync(conversation_id, session_key=ORIGIN, now=1021)
    assert context is not None

    preamble = context.continuation_preamble()
    lowered = preamble.lower()
    assert "phone" in lowered
    assert "do not read" in lowered or "never read" in lowered
    assert "pick the oak flooring" in preamble
    outbound = ConversationContext(summary=context.summary, turns=context.turns)
    assert outbound.continuation_preamble() != preamble
    assert "returned=True" not in repr(context)
    for rendered in (repr(context), str(context), f"{context}"):
        assert "oak flooring" not in rendered


@pytest.mark.asyncio
async def test_returned_context_injects_once_through_the_restore_seam(caplog) -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    _phone_leg(conversation_id)
    assert mark_return_sync(conversation_id, phone_session_key=PHONE, now=1020) is True
    context = consume_return_sync(conversation_id, session_key=ORIGIN, now=1021)
    assert context is not None

    class _Agent:
        def __init__(self) -> None:
            self.chat_ctx = llm.ChatContext.empty()
            self.updates = 0

        async def update_chat_ctx(self, chat_ctx):
            self.chat_ctx = chat_ctx
            self.updates += 1

    agent = _Agent()
    with caplog.at_level(logging.DEBUG):
        assert await restore_conversation_context(agent, context) is True
        assert await restore_conversation_context(agent, context) is False

    assert agent.updates == 1
    injected = agent.chat_ctx.get_by_id(CONTINUATION_MESSAGE_ID)
    assert injected is not None and injected.role == "system"
    assert "oak flooring" in injected.text_content
    assert all("oak flooring" not in record.getMessage() for record in caplog.records)


def test_return_sync_logs_only_counts_never_transcript_or_ids(caplog) -> None:
    with caplog.at_level(logging.DEBUG):
        conversation_id = open_conversation(session_key=ORIGIN, now=1000)
        _phone_leg(conversation_id)
        mark_return_sync(conversation_id, phone_session_key=PHONE, now=1020)
        close_session(conversation_id, session_key=PHONE, now=1021)
        consume_return_sync(conversation_id, session_key=ORIGIN, now=1022)

    for record in caplog.records:
        message = record.getMessage()
        assert "oak flooring" not in message
        assert conversation_id not in message
        assert ORIGIN not in message
        assert PHONE not in message


# --- return sync: claim, ack, release ----------------------------------------


def _marked(conversation_id: str, *, now: int = 1020) -> None:
    assert mark_return_sync(conversation_id, phone_session_key=PHONE, now=now) is True
    close_session(conversation_id, session_key=PHONE, now=now + 1)


def test_claim_hands_out_context_and_a_token_without_consuming_the_marker(store) -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    _phone_leg(conversation_id)
    _marked(conversation_id)

    claim = claim_return_sync(conversation_id, session_key=ORIGIN, now=1030)

    assert claim is not None
    assert claim.context.returned is True
    assert "pick the oak flooring" in claim.context.continuation_preamble()
    assert isinstance(claim.token, str) and len(claim.token) >= 16
    assert _marker_rows(store) == 1
    for rendered in (repr(claim), str(claim), f"{claim}"):
        assert "oak flooring" not in rendered
        assert claim.token not in rendered


def test_ack_consumes_the_marker_exactly_once(store) -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    _phone_leg(conversation_id)
    _marked(conversation_id)
    claim = claim_return_sync(conversation_id, session_key=ORIGIN, now=1030)
    assert claim is not None

    assert ack_return_sync(conversation_id, session_key=ORIGIN, token=claim.token, now=1031) is True
    assert _marker_rows(store) == 0
    # Replaying the ack, claiming again, or releasing a consumed claim all no-op.
    assert (
        ack_return_sync(conversation_id, session_key=ORIGIN, token=claim.token, now=1032) is False
    )
    assert claim_return_sync(conversation_id, session_key=ORIGIN, now=1033) is None
    assert (
        release_return_sync(conversation_id, session_key=ORIGIN, token=claim.token, now=1034)
        is False
    )
    assert conversation_exists(conversation_id) is True


def test_release_returns_the_marker_so_the_next_turn_retries(store) -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    _phone_leg(conversation_id)
    _marked(conversation_id)
    first = claim_return_sync(conversation_id, session_key=ORIGIN, now=1030)
    assert first is not None

    assert (
        release_return_sync(conversation_id, session_key=ORIGIN, token=first.token, now=1031)
        is True
    )
    assert _marker_rows(store) == 1
    second = claim_return_sync(conversation_id, session_key=ORIGIN, now=1032)

    assert second is not None
    assert second.token != first.token
    assert [t.text for t in second.context.turns] == [t.text for t in first.context.turns]
    # A stale token can neither ack nor release the newer claim.
    assert (
        ack_return_sync(conversation_id, session_key=ORIGIN, token=first.token, now=1033) is False
    )
    assert (
        release_return_sync(conversation_id, session_key=ORIGIN, token=first.token, now=1033)
        is False
    )
    assert _marker_rows(store) == 1
    assert (
        ack_return_sync(conversation_id, session_key=ORIGIN, token=second.token, now=1034) is True
    )
    assert _marker_rows(store) == 0


def test_concurrent_claims_never_both_apply(store) -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    _phone_leg(conversation_id)
    _marked(conversation_id)

    first = claim_return_sync(conversation_id, session_key=ORIGIN, now=1030)
    second = claim_return_sync(conversation_id, session_key=ORIGIN, now=1030)

    assert first is not None
    assert second is None
    assert _marker_rows(store) == 1


def test_an_abandoned_claim_becomes_claimable_again_after_its_window(store) -> None:
    """A process that died between claim and ack must not strand the marker forever."""
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    _phone_leg(conversation_id)
    _marked(conversation_id)
    first = claim_return_sync(conversation_id, session_key=ORIGIN, now=1030)
    assert first is not None

    inside = 1030 + RETURN_SYNC_CLAIM_TTL_SECONDS - 1
    assert claim_return_sync(conversation_id, session_key=ORIGIN, now=inside) is None
    outside = 1030 + RETURN_SYNC_CLAIM_TTL_SECONDS + 1
    second = claim_return_sync(conversation_id, session_key=ORIGIN, now=outside)

    assert second is not None
    assert second.token != first.token
    # The dead claimant's token is now worthless.
    assert (
        ack_return_sync(conversation_id, session_key=ORIGIN, token=first.token, now=outside + 1)
        is False
    )
    assert _marker_rows(store) == 1


def test_claim_is_scoped_to_the_origin_session_and_drops_when_it_closed(store) -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    _phone_leg(conversation_id)
    _marked(conversation_id)

    assert claim_return_sync(conversation_id, session_key=PHONE, now=1030) is None
    assert claim_return_sync(conversation_id, session_key="web-room-2", now=1030) is None
    assert claim_return_sync("unknown", session_key=ORIGIN, now=1030) is None
    assert claim_return_sync("has space", session_key=ORIGIN, now=1030) is None
    assert _marker_rows(store) == 1

    raw = sqlite3.connect(store)
    raw.execute("UPDATE conversation_links SET state = 'closed' WHERE session_key = ?", (ORIGIN,))
    raw.commit()
    raw.close()
    assert claim_return_sync(conversation_id, session_key=ORIGIN, now=1031) is None
    assert _marker_rows(store) == 0


def test_a_newer_phone_return_supersedes_an_unacked_claim(store) -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    _phone_leg(conversation_id, phone="attempt-1", now=1010)
    assert mark_return_sync(conversation_id, phone_session_key="attempt-1", now=1020) is True
    close_session(conversation_id, session_key="attempt-1", now=1021)
    stale = claim_return_sync(conversation_id, session_key=ORIGIN, now=1022)
    assert stale is not None

    # A second leg ends before the first claim was ever acked.
    _phone_leg(conversation_id, phone="attempt-2", now=1030)
    assert mark_return_sync(conversation_id, phone_session_key="attempt-2", now=1040) is True
    close_session(conversation_id, session_key="attempt-2", now=1041)

    # Acking the superseded claim must not consume the newer return.
    assert (
        ack_return_sync(conversation_id, session_key=ORIGIN, token=stale.token, now=1042) is False
    )
    fresh = claim_return_sync(conversation_id, session_key=ORIGIN, now=1043)
    assert fresh is not None
    assert ack_return_sync(conversation_id, session_key=ORIGIN, token=fresh.token, now=1044) is True
    assert _marker_rows(store) == 0


def test_return_sync_schema_upgrades_a_pre_claim_marker_table(store) -> None:
    """A database written before claim tokens existed keeps working."""
    raw = sqlite3.connect(store)
    raw.execute(
        "CREATE TABLE conversation_return_sync ("
        "conversation_id TEXT NOT NULL, session_key TEXT NOT NULL, created_at INTEGER NOT NULL, "
        "PRIMARY KEY (conversation_id, session_key))"
    )
    raw.commit()
    raw.close()

    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    _phone_leg(conversation_id)
    _marked(conversation_id)
    claim = claim_return_sync(conversation_id, session_key=ORIGIN, now=1030)

    assert claim is not None
    assert ack_return_sync(conversation_id, session_key=ORIGIN, token=claim.token, now=1031) is True


def test_claim_ack_release_log_only_counts_never_tokens_or_ids(caplog) -> None:
    with caplog.at_level(logging.DEBUG):
        conversation_id = open_conversation(session_key=ORIGIN, now=1000)
        _phone_leg(conversation_id)
        _marked(conversation_id)
        claim = claim_return_sync(conversation_id, session_key=ORIGIN, now=1030)
        assert claim is not None
        release_return_sync(conversation_id, session_key=ORIGIN, token=claim.token, now=1031)
        again = claim_return_sync(conversation_id, session_key=ORIGIN, now=1032)
        assert again is not None
        ack_return_sync(conversation_id, session_key=ORIGIN, token=again.token, now=1033)

    for record in caplog.records:
        message = record.getMessage()
        assert "oak flooring" not in message
        assert conversation_id not in message
        assert claim.token not in message
        assert again.token not in message
        assert ORIGIN not in message


# --- never long-term memory --------------------------------------------------


def test_ledger_lifecycle_never_promotes_anything_into_explicit_preferences() -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    append_turn(conversation_id, "user", "remember my favourite colour is teal", now=1001)
    append_turn(conversation_id, "assistant", "Noted, teal it is.", now=1002)
    link_continuation(conversation_id, session_key=PHONE, now=1003)
    claim_continuation(conversation_id, session_key=PHONE, now=1004)
    mark_return_sync(conversation_id, phone_session_key=PHONE, now=1005)
    consume_return_sync(conversation_id, session_key=ORIGIN, now=1006)
    close_session(conversation_id, session_key=ORIGIN, now=1007)
    close_session(conversation_id, session_key=PHONE, now=1008)
    purge_expired_conversations(now=1009)

    assert memory_tools.recall()["data"]["memories"] == []


def test_hermes_owns_long_term_memory_and_the_ledger_never_imports_the_memory_tool() -> None:
    import inspect

    assert "hermes" in HERMES_LONG_TERM_MEMORY_SEAM.lower()
    source = inspect.getsource(conversation_ledger)
    assert "memory_tools" not in source.replace("HERMES_LONG_TERM_MEMORY_SEAM", "")
    assert "HERMES_LONG_TERM_MEMORY_SEAM" in source
    assert "importance" in source.lower()


# --- no sensitive logs -------------------------------------------------------


def test_ledger_logs_only_counts_never_transcript_or_ids(caplog) -> None:
    with caplog.at_level(logging.DEBUG):
        conversation_id = open_conversation(session_key=ORIGIN, now=1000)
        append_turn(conversation_id, "user", "the garage renovation budget", now=1001)
        link_continuation(conversation_id, session_key=PHONE, now=1002)
        claim_continuation(conversation_id, session_key=PHONE, now=1003)
        close_session(conversation_id, session_key=ORIGIN, now=1004)
        close_session(conversation_id, session_key=PHONE, now=1005)

    for record in caplog.records:
        message = record.getMessage()
        assert "garage renovation" not in message
        assert conversation_id not in message


# --- recorder ----------------------------------------------------------------


def _message(role: str, text: str) -> llm.ChatMessage:
    return llm.ChatMessage(role=role, content=[text])  # type: ignore[arg-type]


def test_recorder_captures_only_visible_completed_turns_and_skips_control_replies() -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    recorder = ConversationRecorder(exclude_assistant_texts=("Do you want me to call you now?",))
    recorder.bind(conversation_id)

    assert recorder.record(_message("user", "How long should the bread proof?")) is True
    assert recorder.record(_message("assistant", "About an hour.")) is True
    assert recorder.record(_message("assistant", "Do you want me to call you now?")) is False
    assert recorder.record(_message("system", "hidden operator prompt")) is False
    assert recorder.record(_message("developer", "developer-only note")) is False
    assert (
        recorder.record(
            llm.FunctionCall(call_id="c1", name="calendar_list", arguments='{"day": "x"}')
        )
        is False
    )
    assert (
        recorder.record(
            llm.FunctionCallOutput(call_id="c1", name="calendar_list", output="{}", is_error=False)
        )
        is False
    )
    assert recorder.record(SimpleNamespace(type="agent_handoff")) is False
    assert recorder.record(_message("user", "")) is False

    link_continuation(conversation_id, session_key=PHONE, now=1001)
    context = claim_continuation(conversation_id, session_key=PHONE, now=1002)
    assert context is not None
    texts = [turn.text for turn in context.turns]
    assert texts == ["How long should the bread proof?", "About an hour."]
    assert recorder.recorded == 2


def test_recorder_is_inert_until_bound_and_survives_storage_errors(monkeypatch) -> None:
    recorder = ConversationRecorder()

    assert recorder.conversation_id is None
    assert recorder.record(_message("user", "hello before binding")) is False

    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    recorder.bind(conversation_id)

    def _boom(*args, **kwargs):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(conversation_ledger, "append_turn", _boom)
    assert recorder.record(_message("user", "still safe")) is False


def test_recorder_never_logs_transcript_content(caplog) -> None:
    conversation_id = open_conversation(session_key=ORIGIN, now=1000)
    recorder = ConversationRecorder()
    recorder.bind(conversation_id)

    with caplog.at_level(logging.DEBUG):
        recorder.record(_message("user", "the garage code is 8675309"))

    assert all("garage code" not in record.getMessage() for record in caplog.records)
    assert all("8675309" not in record.getMessage() for record in caplog.records)
