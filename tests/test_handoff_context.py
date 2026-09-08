"""Unit coverage for the protected conversation snapshot carried by a phone handoff.

The snapshot is the only conversational state that crosses from the web/voice
session into the outbound call. These tests pin its safety envelope: it holds
nothing but recent user/assistant text, credential-like values are redacted,
its size is bounded, it never prints its own contents, and it is injected into
an outbound session exactly once as a private continuation preamble.
"""

from __future__ import annotations

import json
import logging

import pytest
from livekit.agents import llm

from caal.handoff_context import (
    CONTINUATION_MESSAGE_ID,
    MAX_SNAPSHOT_CHARS,
    MAX_SNAPSHOT_METADATA_BYTES,
    MAX_SNAPSHOT_TURNS,
    MAX_TURN_CHARS,
    REDACTED,
    ConversationSnapshot,
    SnapshotTurn,
    capture_conversation_snapshot,
    inject_continuation_preamble,
    redact_sensitive_text,
    restore_conversation_context,
)
from caal.llm.llm_node import _build_messages_from_context

SECRET = "hunter2-super-secret"


def _history(*turns: tuple[str, str]) -> llm.ChatContext:
    chat_ctx = llm.ChatContext.empty()
    for role, text in turns:
        chat_ctx.add_message(role=role, content=text)  # type: ignore[arg-type]
    return chat_ctx


# --- capture ---------------------------------------------------------------


def test_capture_keeps_only_user_and_assistant_text_in_order() -> None:
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="system", content="You are JARVIS. Hidden operator prompt.")
    chat_ctx.add_message(role="user", content="What's on my calendar tomorrow?")
    chat_ctx.items.append(
        llm.FunctionCall(call_id="call-1", name="calendar_list", arguments='{"day": "tomorrow"}')
    )
    chat_ctx.items.append(
        llm.FunctionCallOutput(
            call_id="call-1", name="calendar_list", output='{"events": ["dentist"]}', is_error=False
        )
    )
    chat_ctx.add_message(role="assistant", content="You have the dentist at nine.")
    chat_ctx.add_message(role="developer", content="developer-only note")

    snapshot = capture_conversation_snapshot(chat_ctx.items)

    assert snapshot is not None
    assert snapshot.turns == (
        SnapshotTurn(role="user", text="What's on my calendar tomorrow?"),
        SnapshotTurn(role="assistant", text="You have the dentist at nine."),
    )
    flat = json.dumps(snapshot.to_metadata())
    assert "Hidden operator prompt" not in flat
    assert "calendar_list" not in flat
    assert "developer-only" not in flat


def test_capture_drops_empty_and_non_text_messages() -> None:
    chat_ctx = _history(("user", "   "), ("assistant", ""), ("user", "still here?"))
    chat_ctx.add_message(
        role="user",
        content=[llm.ImageContent(image="data:image/png;base64,AAAA")],
    )

    snapshot = capture_conversation_snapshot(chat_ctx.items)

    assert snapshot is not None
    assert [turn.text for turn in snapshot.turns] == ["still here?"]


def test_capture_excludes_the_handoff_control_replies() -> None:
    chat_ctx = _history(
        ("user", "How long does the bread need to proof?"),
        ("assistant", "About an hour at room temperature."),
        ("assistant", "I can continue this by calling your approved phone. Call you now?"),
    )

    snapshot = capture_conversation_snapshot(
        chat_ctx.items,
        exclude_assistant_texts=(
            "I can continue this by calling your approved phone. Call you now?",
        ),
    )

    assert snapshot is not None
    assert [turn.text for turn in snapshot.turns] == [
        "How long does the bread need to proof?",
        "About an hour at room temperature.",
    ]


def test_capture_returns_none_when_there_is_nothing_to_carry() -> None:
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="system", content="instructions only")

    assert capture_conversation_snapshot(chat_ctx.items) is None
    assert capture_conversation_snapshot([]) is None


def test_capture_keeps_only_the_most_recent_turns() -> None:
    turns = [("user" if i % 2 == 0 else "assistant", f"turn {i}") for i in range(40)]
    chat_ctx = _history(*turns)

    snapshot = capture_conversation_snapshot(chat_ctx.items)

    assert snapshot is not None
    assert len(snapshot.turns) == MAX_SNAPSHOT_TURNS
    assert snapshot.turns[-1].text == "turn 39"
    assert snapshot.turns[0].text == f"turn {40 - MAX_SNAPSHOT_TURNS}"


def test_capture_bounds_characters_per_turn_and_in_total() -> None:
    long_text = "x" * (MAX_TURN_CHARS * 3)
    chat_ctx = _history(*[("user", long_text)] * MAX_SNAPSHOT_TURNS)

    snapshot = capture_conversation_snapshot(chat_ctx.items)

    assert snapshot is not None
    assert all(len(turn.text) <= MAX_TURN_CHARS for turn in snapshot.turns)
    assert snapshot.total_chars <= MAX_SNAPSHOT_CHARS
    assert len(json.dumps(snapshot.to_metadata()).encode()) <= MAX_SNAPSHOT_METADATA_BYTES


def test_capture_prefers_dropping_old_turns_over_losing_the_latest_one() -> None:
    chat_ctx = _history(
        *[("assistant", "y" * MAX_TURN_CHARS)] * MAX_SNAPSHOT_TURNS,
        ("user", "the newest question"),
    )

    snapshot = capture_conversation_snapshot(chat_ctx.items)

    assert snapshot is not None
    assert snapshot.turns[-1].text == "the newest question"
    assert snapshot.total_chars <= MAX_SNAPSHOT_CHARS


def test_capture_redacts_credentials_before_anything_leaves_the_session() -> None:
    chat_ctx = _history(
        ("user", f"my wifi password is {SECRET} please remember it"),
        ("assistant", "Noted. Your API key sk-live-abcdefghijklmnopqrstuvwxyz0123 is stored."),
    )

    snapshot = capture_conversation_snapshot(chat_ctx.items)

    assert snapshot is not None
    flat = json.dumps(snapshot.to_metadata())
    assert SECRET not in flat
    assert "sk-live-abcdefghijklmnopqrstuvwxyz0123" not in flat
    assert REDACTED in flat


# --- redaction -------------------------------------------------------------


TEST_AWS_ACCESS_KEY = "AK" + "IA" + "TEST" + "123456789012"
TEST_GITHUB_TOKEN = "gh" + "p_" + "testtokenvalue1234567890"
TEST_PRIVATE_KEY = "-" * 5 + "BEGIN PRIVATE KEY" + "-" * 5 + "\nMIIEvQIBADANBg"


@pytest.mark.parametrize(
    "text",
    [
        "my password is Tr0ub4dor&3",
        "password: Tr0ub4dor&3",
        f"the api key = {TEST_AWS_ACCESS_KEY}",
        f"use token {TEST_GITHUB_TOKEN}",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJl",
        "the pin is 4471",
        "card number 4111 1111 1111 1111",
        "call me at +1 (780) 555-8345 instead",
        TEST_PRIVATE_KEY,
        "session id a8f5f167f44f4964e6c998dee827110c",
    ],
)
def test_redaction_masks_credential_like_values(text: str) -> None:
    redacted = redact_sensitive_text(text)

    assert REDACTED in redacted
    for value in (
        "Tr0ub4dor&3",
        TEST_AWS_ACCESS_KEY,
        TEST_GITHUB_TOKEN,
        "eyJhbGciOiJIUzI1NiJ9",
        "4471",
        "4111 1111 1111 1111",
        "555-8345",
        "MIIEvQIBADANBg",
        "a8f5f167f44f4964e6c998dee827110c",
    ):
        assert value not in redacted


@pytest.mark.parametrize(
    "text",
    [
        "What's the weather like tomorrow?",
        "The dentist is at 9 on the 4th of September 2026.",
        "That token expired, can you renew it?",
        "Remind me to buy bread and milk.",
    ],
)
def test_redaction_leaves_ordinary_conversation_alone(text: str) -> None:
    assert redact_sensitive_text(text) == text


# --- serialization ---------------------------------------------------------


def test_snapshot_round_trips_through_dispatch_metadata() -> None:
    snapshot = ConversationSnapshot(
        turns=(
            SnapshotTurn(role="user", text="Where were we?"),
            SnapshotTurn(role="assistant", text="Planning your Friday trip."),
        )
    )

    restored = ConversationSnapshot.from_metadata(json.loads(json.dumps(snapshot.to_metadata())))

    assert restored == snapshot


@pytest.mark.parametrize(
    "raw",
    [
        "not a dict",
        {"v": 2, "turns": []},
        {"v": 1},
        {"v": 1, "turns": "nope"},
        {"v": 1, "turns": [{"role": "system", "text": "x"}]},
        {"v": 1, "turns": [{"role": "tool", "text": "x"}]},
        {"v": 1, "turns": [{"role": "user", "text": 42}]},
        {"v": 1, "turns": [{"role": "user"}]},
        {"v": 1, "turns": [{"role": "user", "text": "x"}] * (MAX_SNAPSHOT_TURNS + 1)},
        {"v": 1, "turns": [{"role": "user", "text": "z" * MAX_SNAPSHOT_METADATA_BYTES}]},
    ],
)
def test_snapshot_rejects_malformed_or_oversized_metadata(raw: object) -> None:
    with pytest.raises(ValueError):
        ConversationSnapshot.from_metadata(raw)


def test_snapshot_parsing_re_applies_bounds_and_redaction() -> None:
    raw = {
        "v": 1,
        "turns": [
            {"role": "user", "text": f"my password is {SECRET}"},
            {"role": "assistant", "text": "a" * (MAX_TURN_CHARS + 50)},
        ],
    }

    snapshot = ConversationSnapshot.from_metadata(raw)

    assert SECRET not in snapshot.turns[0].text
    assert len(snapshot.turns[1].text) <= MAX_TURN_CHARS


def test_snapshot_never_prints_its_contents() -> None:
    snapshot = ConversationSnapshot(turns=(SnapshotTurn(role="user", text="private detail"),))

    assert "private detail" not in repr(snapshot)
    assert "private detail" not in str(snapshot)
    assert "private detail" not in f"{snapshot}"


# --- continuation preamble -------------------------------------------------


def test_preamble_frames_the_snapshot_as_private_continuation_context() -> None:
    snapshot = ConversationSnapshot(
        turns=(
            SnapshotTurn(role="user", text="Where were we?"),
            SnapshotTurn(role="assistant", text="Planning your Friday trip."),
        )
    )

    preamble = snapshot.continuation_preamble()

    assert "Where were we?" in preamble
    assert "Planning your Friday trip." in preamble
    lowered = preamble.lower()
    assert "phone" in lowered
    assert "continu" in lowered
    assert "do not read" in lowered or "never read" in lowered
    assert "unless" in lowered  # only reveal earlier details when asked
    assert lowered.index("where were we?") < lowered.index("planning your friday trip.")


def test_preamble_is_injected_as_a_system_message_exactly_once() -> None:
    snapshot = ConversationSnapshot(turns=(SnapshotTurn(role="user", text="Where were we?"),))
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="system", content="You are JARVIS.", id="lk.agent_task.instructions")

    assert inject_continuation_preamble(chat_ctx, snapshot) is True
    assert inject_continuation_preamble(chat_ctx, snapshot) is False

    system_items = [item for item in chat_ctx.items if item.type == "message"]
    assert len(system_items) == 2
    injected = chat_ctx.get_by_id(CONTINUATION_MESSAGE_ID)
    assert injected is not None
    assert injected.role == "system"
    assert "Where were we?" in injected.text_content
    assert chat_ctx.items[0].id == "lk.agent_task.instructions"


def test_preamble_with_newer_context_replaces_the_existing_one_in_place() -> None:
    first = ConversationSnapshot(turns=(SnapshotTurn(role="user", text="Where were we?"),))
    later = ConversationSnapshot(
        turns=(
            SnapshotTurn(role="user", text="Where were we?"),
            SnapshotTurn(role="assistant", text="Planning your Friday trip."),
        )
    )
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="system", content="You are JARVIS.", id="lk.agent_task.instructions")
    assert inject_continuation_preamble(chat_ctx, first) is True
    chat_ctx.add_message(role="user", content="a later question")
    original_index = chat_ctx.index_by_id(CONTINUATION_MESSAGE_ID)

    assert inject_continuation_preamble(chat_ctx, later) is True

    assert sum(1 for item in chat_ctx.items if item.id == CONTINUATION_MESSAGE_ID) == 1
    assert chat_ctx.index_by_id(CONTINUATION_MESSAGE_ID) == original_index
    injected = chat_ctx.get_by_id(CONTINUATION_MESSAGE_ID)
    assert injected is not None and injected.role == "system"
    assert "Friday trip" in injected.text_content
    # The same context again is a no-op, as before.
    assert inject_continuation_preamble(chat_ctx, later) is False


def test_replacing_the_preamble_never_touches_a_shallow_copy_source() -> None:
    first = ConversationSnapshot(turns=(SnapshotTurn(role="user", text="Where were we?"),))
    later = ConversationSnapshot(turns=(SnapshotTurn(role="user", text="Something newer"),))
    source = llm.ChatContext.empty()
    inject_continuation_preamble(source, first)
    working = source.copy()

    assert inject_continuation_preamble(working, later) is True

    assert "Something newer" in working.get_by_id(CONTINUATION_MESSAGE_ID).text_content
    assert "Something newer" not in source.get_by_id(CONTINUATION_MESSAGE_ID).text_content


class _FakeAgent:
    """Records the chat context an outbound agent is asked to adopt."""

    def __init__(self) -> None:
        self.chat_ctx = llm.ChatContext.empty()
        self.chat_ctx.add_message(role="system", content="You are JARVIS.")
        self.updates: list[llm.ChatContext] = []

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        self.updates.append(chat_ctx)
        self.chat_ctx = chat_ctx


@pytest.mark.asyncio
async def test_restore_updates_the_agent_history_once_and_logs_no_contents(caplog) -> None:
    snapshot = ConversationSnapshot(turns=(SnapshotTurn(role="user", text="secret plan"),))
    agent = _FakeAgent()

    with caplog.at_level(logging.DEBUG):
        assert await restore_conversation_context(agent, snapshot) is True
        assert await restore_conversation_context(agent, snapshot) is False

    assert len(agent.updates) == 1
    assert agent.chat_ctx.get_by_id(CONTINUATION_MESSAGE_ID) is not None
    assert all("secret plan" not in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_restore_with_newer_context_updates_the_agent_history_once_more() -> None:
    first = ConversationSnapshot(turns=(SnapshotTurn(role="user", text="first leg"),))
    later = ConversationSnapshot(
        turns=(
            SnapshotTurn(role="user", text="first leg"),
            SnapshotTurn(role="assistant", text="second leg"),
        )
    )
    agent = _FakeAgent()

    assert await restore_conversation_context(agent, first) is True
    assert await restore_conversation_context(agent, later) is True
    assert await restore_conversation_context(agent, later) is False

    assert len(agent.updates) == 2
    assert sum(1 for item in agent.chat_ctx.items if item.id == CONTINUATION_MESSAGE_ID) == 1
    assert "second leg" in agent.chat_ctx.get_by_id(CONTINUATION_MESSAGE_ID).text_content


def test_llm_messages_keep_the_agent_prompt_and_the_preamble_for_hermes() -> None:
    """A second system message must not replace the JARVIS prompt sent to Hermes."""
    snapshot = ConversationSnapshot(turns=(SnapshotTurn(role="user", text="Where were we?"),))
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="system", content="You are JARVIS.", id="lk.agent_task.instructions")
    inject_continuation_preamble(chat_ctx, snapshot)
    chat_ctx.add_message(role="system", content="Greet the user briefly.")

    messages = _build_messages_from_context(chat_ctx)

    assert [message["role"] for message in messages] == ["system"]
    system = messages[0]["content"]
    assert system.index("You are JARVIS.") < system.index("Where were we?")
    assert system.index("Where were we?") < system.index("Greet the user briefly.")
