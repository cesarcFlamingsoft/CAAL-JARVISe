"""Answering the delivery question in plain words, without asking a model.

JARVIS asks "when it comes due, do you want me to say it here, send it to your
Telegram, call you, or do nothing about it?" and the person answers "call me".
That answer is a whole turn on its own, and routing it through the model is
where it goes wrong: the local model reads "call me" as a phone request, as
background work, or as something for the calendar, and the reminder that was
waiting for an answer never gets one.

So the answer is read here instead, deterministically, and only in the one
situation where it can mean nothing else: this signed-in owner has a timed
reminder, still in the future, whose delivery choice was asked about and has
not been answered yet.

The properties these tests pin:

* the awaiting state is a bound fact -- written when a timed reminder is
  created without a delivery choice, cleared the moment one is made, and gone
  once the reminder is past. It is never inferred from a reminder existing;
* recognition is whitelist-only and bounded. Every word of the turn has to be
  one of the accepted answer words; a number, a third party, a hypothetical, a
  question, a calendar request or an extra clause falls through untouched;
* an accepted answer mutates through the same owner-scoped tool the model
  would have called, speaks the handler own message, publishes exactly one
  constant scheduled-change packet, and consumes the turn so no other route
  sees it;
* while callback dispatch is in verification mode, a chosen call is described
  as saved and verified rather than as a call that is going to ring;
* nothing logged, published or spoken carries the utterance, the title, the
  row id, a number or a chat.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from caal.delivery_router import CALL_VERIFICATION_NOTE, DeliveryAnswerHandler
from caal.tools import alarms_tools, reminder_delivery, reminders_tools
from caal.tools.delivery_answer import read_delivery_answer
from caal.user_scope import UserScope

ANA = "usr_" + "a" * 24
BO = "usr_" + "b" * 24
NOW = 1_000_000
SOON = "PT30M"

SPEAK = reminder_delivery.SPEAK
TELEGRAM = reminder_delivery.TELEGRAM
CALL = reminder_delivery.CALL
ALL = reminder_delivery.ALL
DEFAULT = reminder_delivery.DEFAULT
NONE = reminder_delivery.NONE

TITLE = "Take the blue folder to the clinic"


@pytest.fixture
def store(monkeypatch, tmp_path):
    """One SQLite file shared by all three stores, as in the running deployment."""
    path = tmp_path / "assistant.sqlite3"
    monkeypatch.setattr(alarms_tools, "STORE_PATH", path)
    monkeypatch.setattr(reminders_tools, "STORE_PATH", path)
    monkeypatch.setattr(reminder_delivery, "STORE_PATH", path)
    return path


@pytest.fixture
def everything_available(monkeypatch):
    """Both outside channels authorized for Ana and for nobody else."""
    monkeypatch.setattr(reminder_delivery, "telegram_owner", lambda: ANA)
    monkeypatch.setattr(reminder_delivery, "telegram_configured", lambda: True)
    monkeypatch.setattr(
        reminder_delivery,
        "resolve_callback_number",
        lambda user_id: "+15551230000" if user_id == ANA else None,
    )


@pytest.fixture
def not_dry_run(monkeypatch):
    monkeypatch.delenv("CAAL_CALLBACK_DISPATCH_DRY_RUN", raising=False)


@pytest.fixture
def verification_mode(monkeypatch):
    monkeypatch.setenv("CAAL_CALLBACK_DISPATCH_DRY_RUN", "1")


class FakeSession:
    """Records what the agent would speak."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def say(self, text: str, **_: Any) -> None:
        self.spoken.append(text)


class Announcer:
    """Counts the constant scheduled-change packets a route publishes."""

    def __init__(self) -> None:
        self.published = 0

    async def __call__(self) -> None:
        self.published += 1


def _handler(user_id: str | None = ANA, *, announce: Any = None) -> DeliveryAnswerHandler:
    scope = (
        UserScope(user_id=user_id, identity_configured=True)
        if user_id
        else UserScope.anonymous()
    )
    return DeliveryAnswerHandler(scope=scope, announce=announce, now=lambda: NOW)


def _reminder_id(path, user_id: str) -> str:
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT id FROM reminders WHERE user_id = ? ORDER BY created_at DESC, rowid DESC",
            (user_id,),
        ).fetchall()
    finally:
        connection.close()
    assert rows, "expected a reminder for this owner"
    return rows[0][0]


def _ask(user_id: str = ANA, title: str = TITLE, due: str = SOON) -> dict:
    """Create a timed reminder without naming a delivery, which asks the question."""
    created = reminders_tools.create_reminder(title=title, due=due, user_id=user_id, now=NOW)
    assert created["data"]["delivery_pending"] is True
    return created


def run(coroutine):
    return asyncio.run(coroutine)


# --- reading the answer, and refusing to read anything else ----------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("call me", (CALL,)),
        ("Call me.", (CALL,)),
        ("give me a call", (CALL,)),
        ("call", (CALL,)),
        ("please call me", (CALL,)),
        ("yeah, call me", (CALL,)),
        ("call me about that reminder", (CALL,)),
        ("call me about the reminder", (CALL,)),
        ("telegram", (TELEGRAM,)),
        ("message me", (TELEGRAM,)),
        ("text me", (TELEGRAM,)),
        ("send it to my telegram", (TELEGRAM,)),
        ("say it here", (SPEAK,)),
        ("here", (SPEAK,)),
        ("out loud", (SPEAK,)),
        ("tell me here", (SPEAK,)),
        ("all", (ALL,)),
        ("every way", (ALL,)),
        ("all of them", (ALL,)),
        ("default", (DEFAULT,)),
        ("the usual", (DEFAULT,)),
        ("none", (NONE,)),
        ("nothing", (NONE,)),
        ("no notification", (NONE,)),
        ("call and message me", (TELEGRAM, CALL)),
        ("call me and say it here", (SPEAK, CALL)),
    ],
)
def test_the_accepted_answers_are_read_exactly(text, expected) -> None:
    assert read_delivery_answer(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "call me in five minutes",
        "call me in 5 minutes",
        "call me at 4",
        "call my wife",
        "call her",
        "call the clinic",
        "call John",
        "should you call me?",
        "do you want to call me",
        "would you call me",
        "maybe call me",
        "if it is urgent call me",
        "call me when the research is done",
        "hang up and call me back when you are done",
        "put a call on my calendar",
        "schedule a call for tomorrow",
        "add a call to my tasks",
        "remind me about my call",
        "call me about the dentist appointment tomorrow",
        "what are my options",
        "call me later",
        "tell me a joke",
        "call me and also book the room",
        "nothing and call me",
    ],
)
def test_anything_that_is_not_plainly_that_answer_is_left_alone(text) -> None:
    assert read_delivery_answer(text) is None


def test_a_long_utterance_is_never_read_as_an_answer() -> None:
    assert read_delivery_answer("call me " * 40) is None


# --- the awaiting state is bound, not guessed -------------------------------------------------


def test_an_unanswered_timed_reminder_is_recorded_as_awaiting_its_answer(
    store, everything_available
) -> None:
    _ask()

    assert reminder_delivery.awaiting_reminder(ANA, NOW) == _reminder_id(store, ANA)
    assert reminder_delivery.awaiting_reminder(BO, NOW) is None


def test_a_reminder_created_with_a_choice_is_never_awaiting(store, everything_available) -> None:
    created = reminders_tools.create_reminder(
        title=TITLE, due=SOON, delivery=["telegram"], user_id=ANA, now=NOW
    )

    assert created["data"]["delivery_pending"] is False
    assert reminder_delivery.awaiting_reminder(ANA, NOW) is None


def test_an_undated_reminder_is_never_awaiting(store, everything_available) -> None:
    reminders_tools.create_reminder(title=TITLE, user_id=ANA, now=NOW)

    assert reminder_delivery.awaiting_reminder(ANA, NOW) is None


def test_answering_clears_the_awaiting_state(store, everything_available) -> None:
    _ask()

    reminders_tools.set_delivery(delivery=["telegram"], user_id=ANA, now=NOW)

    assert reminder_delivery.awaiting_reminder(ANA, NOW) is None


def test_saying_nothing_at_all_also_clears_the_awaiting_state(store, everything_available) -> None:
    _ask()

    reminders_tools.set_delivery(delivery=["none"], user_id=ANA, now=NOW)

    assert reminder_delivery.awaiting_reminder(ANA, NOW) is None


def test_a_reminder_that_has_come_due_is_no_longer_awaiting(store, everything_available) -> None:
    _ask()

    assert reminder_delivery.awaiting_reminder(ANA, NOW + 3_600) is None


def test_only_the_newest_future_reminder_can_be_the_awaited_one(
    store, everything_available
) -> None:
    """set_delivery targets the newest future reminder, so nothing else may match."""
    _ask()
    first = _reminder_id(store, ANA)
    reminders_tools.create_reminder(
        title="Second thing", due="PT45M", delivery=["speak"], user_id=ANA, now=NOW
    )

    assert _reminder_id(store, ANA) != first
    assert reminder_delivery.awaiting_reminder(ANA, NOW) is None


# --- the route ---------------------------------------------------------------------------------


def test_a_spoken_call_answer_arms_the_call_and_consumes_the_turn(
    store, everything_available, not_dry_run
) -> None:
    _ask()
    session = FakeSession()
    announce = Announcer()

    consumed = run(_handler(announce=announce).handle("call me", session))

    assert consumed is True
    assert reminder_delivery.channels_of(_reminder_id(store, ANA)) == (CALL,)
    assert announce.published == 1
    assert session.spoken and "call you" in session.spoken[0].lower()
    assert reminder_delivery.awaiting_reminder(ANA, NOW) is None


def test_a_typed_additive_answer_arms_both_channels(
    store, everything_available, not_dry_run
) -> None:
    _ask()
    session = FakeSession()
    announce = Announcer()

    consumed = run(_handler(announce=announce).handle("call and message me", session))

    assert consumed is True
    assert reminder_delivery.channels_of(_reminder_id(store, ANA)) == (TELEGRAM, CALL)
    assert announce.published == 1


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("none", ()),
        ("nothing", ()),
        ("default", (SPEAK,)),
        ("the usual", (SPEAK,)),
        ("say it here", (SPEAK,)),
        ("telegram", (TELEGRAM,)),
        ("text me", (TELEGRAM,)),
        ("all", (SPEAK, TELEGRAM, CALL)),
        ("every way", (SPEAK, TELEGRAM, CALL)),
    ],
)
def test_every_accepted_answer_settles_the_reminder_the_way_it_says(
    store, everything_available, not_dry_run, answer, expected
) -> None:
    _ask()
    session = FakeSession()

    assert run(_handler().handle(answer, session)) is True
    assert reminder_delivery.channels_of(_reminder_id(store, ANA)) == expected
    assert reminder_delivery.awaiting_reminder(ANA, NOW) is None
    assert session.spoken


def test_the_same_answer_twice_only_changes_anything_once(
    store, everything_available, not_dry_run
) -> None:
    _ask()
    session = FakeSession()
    announce = Announcer()
    handler = _handler(announce=announce)

    assert run(handler.handle("call me", session)) is True
    assert run(handler.handle("call me", session)) is False
    assert announce.published == 1
    assert reminder_delivery.channels_of(_reminder_id(store, ANA)) == (CALL,)


@pytest.mark.parametrize(
    "text",
    [
        "call me in 5 minutes",
        "call my wife",
        "should you call me?",
        "if it comes to that call me",
        "schedule a call for tomorrow",
        "put a call on my calendar",
        "call me about the dentist appointment tomorrow",
        "call me when the research is done",
    ],
)
def test_a_turn_that_is_not_the_answer_never_touches_the_reminder(
    store, everything_available, not_dry_run, text
) -> None:
    _ask()
    session = FakeSession()
    announce = Announcer()

    assert run(_handler(announce=announce).handle(text, session)) is False
    assert reminder_delivery.channels_of(_reminder_id(store, ANA)) == (SPEAK,)
    assert announce.published == 0
    assert session.spoken == []


def test_with_nothing_awaiting_an_answer_the_turn_is_left_to_the_model(
    store, everything_available, not_dry_run
) -> None:
    reminders_tools.create_reminder(
        title=TITLE, due=SOON, delivery=["speak"], user_id=ANA, now=NOW
    )
    session = FakeSession()

    assert run(_handler().handle("call me", session)) is False
    assert reminder_delivery.channels_of(_reminder_id(store, ANA)) == (SPEAK,)
    assert session.spoken == []


def test_an_expired_reminder_cannot_be_answered(store, everything_available, not_dry_run) -> None:
    _ask()
    session = FakeSession()
    late = DeliveryAnswerHandler(
        scope=UserScope(user_id=ANA, identity_configured=True), now=lambda: NOW + 3_600
    )

    assert run(late.handle("call me", session)) is False
    assert reminder_delivery.channels_of(_reminder_id(store, ANA)) == (SPEAK,)


def test_another_user_answer_cannot_reach_this_owner_reminder(
    store, everything_available, not_dry_run
) -> None:
    _ask()
    session = FakeSession()

    assert run(_handler(BO).handle("call me", session)) is False
    assert reminder_delivery.channels_of(_reminder_id(store, ANA)) == (SPEAK,)
    assert session.spoken == []


def test_an_anonymous_session_never_answers_anybody(
    store, everything_available, not_dry_run
) -> None:
    _ask()
    session = FakeSession()
    anonymous = DeliveryAnswerHandler(scope=UserScope.anonymous(), now=lambda: NOW)

    assert run(anonymous.handle("call me", session)) is False
    assert reminder_delivery.channels_of(_reminder_id(store, ANA)) == (SPEAK,)


# --- verification mode ---------------------------------------------------------------------


def test_a_chosen_call_is_described_truthfully_while_calls_are_held_back(
    store, everything_available, verification_mode
) -> None:
    _ask()
    session = FakeSession()

    assert run(_handler().handle("call me", session)) is True

    spoken = session.spoken[0]
    assert CALL_VERIFICATION_NOTE in spoken
    assert "i will call you" not in spoken.lower()
    # The choice is still saved and still armed: verification mode holds the
    # dial back, it does not change what the person asked for.
    assert reminder_delivery.channels_of(_reminder_id(store, ANA)) == (CALL,)
    assert reminder_delivery.states_of([_reminder_id(store, ANA)]) == dict(
        [(_reminder_id(store, ANA), dict(call=reminder_delivery.PENDING))]
    )


def test_a_held_back_call_still_says_what_the_other_channels_will_do(
    store, everything_available, verification_mode
) -> None:
    _ask()
    session = FakeSession()

    assert run(_handler().handle("call and message me", session)) is True

    spoken = session.spoken[0].lower()
    assert "telegram" in spoken
    assert CALL_VERIFICATION_NOTE.lower() in spoken


def test_verification_mode_says_nothing_extra_about_a_choice_with_no_call(
    store, everything_available, verification_mode
) -> None:
    _ask()
    session = FakeSession()

    assert run(_handler().handle("telegram", session)) is True
    assert CALL_VERIFICATION_NOTE not in session.spoken[0]


def test_an_ordinary_deployment_promises_the_call_plainly(
    store, everything_available, not_dry_run
) -> None:
    _ask()
    session = FakeSession()

    assert run(_handler().handle("call me", session)) is True
    assert CALL_VERIFICATION_NOTE not in session.spoken[0]


# --- privacy ---------------------------------------------------------------------------------


def test_the_route_logs_nothing_that_belongs_to_anybody(
    store, everything_available, not_dry_run, caplog
) -> None:
    _ask()
    session = FakeSession()
    utterance = "call and message me"

    with caplog.at_level(logging.DEBUG):
        assert run(_handler().handle(utterance, session)) is True

    written = " ".join(record.getMessage() for record in caplog.records).lower()
    for secret in (utterance, TITLE.lower(), _reminder_id(store, ANA).lower(), ANA.lower()):
        assert secret not in written


def test_the_published_packet_is_the_shared_constant() -> None:
    from caal import scheduled_events

    assert scheduled_events.EVENT == dict(v=1, kind="scheduled_changed")
    assert "title" not in scheduled_events.PAYLOAD
    assert "usr_" not in scheduled_events.PAYLOAD


# --- the turn handler that owns the order --------------------------------------------------


@pytest.fixture
def voice_agent():
    module_path = Path(__file__).parents[1] / "voice_agent.py"
    spec = importlib.util.spec_from_file_location("voice_agent_for_delivery", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RefusingBackground:
    """Background work that fails the test if an answered turn ever reaches it."""

    def has_deliverable_file_request(self, _text: str) -> bool:
        return False

    async def process_turn(self, *_args, **_kwargs):
        raise AssertionError("a delivery answer must never start background work")


async def _never_end() -> None:
    raise AssertionError("a delivery answer must not end the call")


def test_a_spoken_answer_is_claimed_before_any_other_route(
    store, everything_available, not_dry_run, voice_agent
) -> None:
    _ask()
    session = FakeSession()
    local = voice_agent.LocalTurnHandler(
        phone_handoff=None,
        session=session,
        end_call=_never_end,
        background=RefusingBackground(),
        delivery=_handler(),
    )

    async def turn() -> bool:
        local.on_final_transcript("call me")
        return await local.turn_consumed("call me")

    assert run(turn()) is True
    assert reminder_delivery.channels_of(_reminder_id(store, ANA)) == (CALL,)


def test_typed_chat_gets_the_same_route(
    store, everything_available, not_dry_run, voice_agent
) -> None:
    _ask()
    session = FakeSession()
    local = voice_agent.LocalTurnHandler(
        phone_handoff=None,
        session=session,
        end_call=_never_end,
        background=RefusingBackground(),
        delivery=_handler(),
    )

    assert run(local.turn_consumed("call and message me")) is True
    assert reminder_delivery.channels_of(_reminder_id(store, ANA)) == (TELEGRAM, CALL)


def test_an_ordinary_turn_still_flows_past_the_route(
    store, everything_available, not_dry_run, voice_agent
) -> None:
    _ask()
    session = FakeSession()
    local = voice_agent.LocalTurnHandler(
        phone_handoff=None,
        session=session,
        end_call=_never_end,
        delivery=_handler(),
    )

    assert run(local.turn_consumed("what is on my calendar tomorrow")) is False
    assert session.spoken == []
    assert reminder_delivery.channels_of(_reminder_id(store, ANA)) == (SPEAK,)


def test_the_voice_agent_builds_the_route_only_for_a_signed_in_session(voice_agent) -> None:
    signed_in = UserScope(user_id=ANA, identity_configured=True)

    assert voice_agent.build_delivery_answer_handler(signed_in) is not None
    assert voice_agent.build_delivery_answer_handler(UserScope.anonymous()) is None
    assert voice_agent.build_delivery_answer_handler(UserScope.legacy()) is None
