"""Regression coverage for caller-authorized JARVIS call termination."""

from __future__ import annotations

import asyncio

import pytest

from caal.call_termination import (
    acknowledge_and_end_call,
    callback_requested,
    end_call_requested,
    end_livekit_room,
)


def test_end_call_requested_accepts_explicit_hang_up_phrases() -> None:
    assert end_call_requested("JARVIS, hang up now")
    assert end_call_requested("Please end the call.")
    assert end_call_requested("goodbye JARVIS")


def test_end_call_requested_rejects_ambiguous_conversation_text() -> None:
    assert not end_call_requested("How do I end a phone call on my iPhone?")
    assert not end_call_requested("I need to call my manager later")
    assert not end_call_requested("")


# --- hang up now, call back when the background task is done -----------------


@pytest.mark.parametrize(
    "text",
    [
        "hang up and call me back when you're done",
        "JARVIS, hang up and call me again when you're done.",
        "Hang up now and call me back once it's finished",
        "end the call and call me back when the task is done",
        "please hang up, then ring me back when you have the results",
        "hang up and give me a call when you're finished",
        "disconnect and call me back as soon as it's ready",
        "okay hang up and call me back with the results",
        "call me back when you're done and hang up",
        "hang up for now and call me when the background task is complete",
    ],
)
def test_callback_requested_accepts_explicit_hang_up_and_call_back_phrases(text: str) -> None:
    assert callback_requested(text) is True
    # It is its own command: the plain hang-up path must not also fire.
    assert end_call_requested(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "hang up",
        "JARVIS, hang up now",
        "end the call",
        "call me back when you're done",  # no hang-up: not a callback authorization
        "hang up and call me back",  # no completion condition
        "should I hang up and have you call me back when you're done?",
        "can you hang up and call me back when you're done?",
        "would you hang up and call me back when it's done",
        "what happens if you hang up and call me back when you're done",
        "if you hang up and call me back when you're done that would be great",
        "hang up and call me back when you're done, and also book the flight",
        "I might tell you to hang up and call me back when you're done",
        "my brother said hang up and call me back when you're done",
        "hang up and call my brother back when you're done",
        "hang up and call me back at +1 780 555 8345 when you're done",
        "",
    ],
)
def test_callback_requested_rejects_incidental_question_and_hypothetical_text(text: str) -> None:
    assert callback_requested(text) is False


def test_callback_requested_is_bounded_and_deterministic() -> None:
    long_text = "hang up and call me back when you're done " * 500
    assert callback_requested(long_text) is False
    assert callback_requested("hang up and call me back when you're done") is True
    assert callback_requested("hang up and call me back when you're done") is True


def test_end_livekit_room_deletes_only_the_current_room() -> None:
    deleted_rooms: list[str] = []

    class FakeRoomService:
        async def delete_room(self, request) -> None:
            deleted_rooms.append(request.room)

    asyncio.run(end_livekit_room(FakeRoomService(), "caal-outbound-safe-id"))

    assert deleted_rooms == ["caal-outbound-safe-id"]


def test_acknowledge_and_end_call_speaks_before_terminating_room() -> None:
    events: list[str] = []

    class FakeSession:
        async def say(self, text: str) -> None:
            events.append(text)

    class FakeRoomService:
        async def delete_room(self, request) -> None:
            events.append(request.room)

    asyncio.run(acknowledge_and_end_call(FakeSession(), FakeRoomService(), "room-1"))

    assert events == ["Ending the call. Goodbye.", "room-1"]
