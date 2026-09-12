"""Reading a request to change a scheduled item, on the local model only.

Asked to turn an alarm into a reminder, JARVIS looked at the schedule and then
called the one scheduled write it had: it set a *second* alarm, and said
something that sounded like a change. The native surface that can actually
cancel, move, rename and convert now exists; this is the layer that recognises
the request when the person says it in their own words rather than in the shape
a tool schema wants.

It is built like the delivery reader beside it and holds the same line:

* the offline gate is the safety boundary, not the model. A turn only becomes
  askable when it is short, names one of alarm, timer or reminder, carries a
  cue that something should *change*, is not a question, a hypothetical, about
  somebody else, or carrying a phone number;
* the payload is the shared system prompt, the bounded turn, and a bounded
  summary of the pending items of this one owner -- their own words and a
  relative time, numbered for this one call. No row id, no owner id, no
  channel, no destination, no conversation history, no cached provider data,
  no tool schemas, and nothing belonging to anybody else;
* the reply is strict top-level JSON in a bounded enum. A new time or a new
  title has to be words the person actually said, an item number has to be one
  the server offered, and anything else decides nothing;
* ambiguity fails closed: it asks, and changes nothing;
* the local model is the only model. Hermes is not reachable from here, and a
  slow, unreachable or unusable reading leaves the turn on its ordinary route.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from caal.schedule_router import ScheduledChangeHandler
from caal.tools import alarms_tools, reminder_delivery, reminders_tools, scheduled_items
from caal.tools.schedule_semantics import (
    MAX_SEMANTIC_INPUT_CHARS,
    SCHEDULE_CHANGE_SYSTEM_PROMPT,
    SemanticScheduleReader,
    askable,
    parse_change_reply,
)
from caal.user_scope import UserScope
from caal.work_router import provider_classifier

ANA = "usr_" + "a" * 24
BO = "usr_" + "b" * 24
NOW = 1_000_000
HOUR = 3600

LAUNDRY = "take the laundry out"
STANDUP = "standup with the team"

CONVERT = json.dumps(dict(action="convert", item=1, target_kind="reminder"))
CANCEL = json.dumps(dict(action="cancel", item=1))


class FakeLocalModel:
    """The local provider, recording exactly what it was asked and offered."""

    def __init__(self, reply: str = CONVERT, *, error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[tuple[Any, Any]] = []

    async def chat(self, messages: Any, tools: Any = None, **_: Any) -> Any:
        self.calls.append((messages, tools))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=self.reply, tool_calls=[])


class RefusingModel(FakeLocalModel):
    """A model whose answer would be wrong if it were ever consulted."""

    def __init__(self) -> None:
        super().__init__(json.dumps(dict(action="cancel", item=1)))


class FakeSession:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def say(self, text: str, **_: Any) -> None:
        self.spoken.append(text)


class Announcer:
    def __init__(self) -> None:
        self.published = 0

    async def __call__(self) -> None:
        self.published += 1


class RefusingBackground:
    async def process_turn(self, *_: Any, **__: Any) -> Any:
        raise AssertionError("a scheduled change must never reach background work")

    def has_deliverable_file_request(self, *_: Any, **__: Any) -> bool:
        return False


@pytest.fixture
def store(monkeypatch, tmp_path):
    path = tmp_path / "assistant.sqlite3"
    monkeypatch.setattr(alarms_tools, "STORE_PATH", path)
    monkeypatch.setattr(reminders_tools, "STORE_PATH", path)
    monkeypatch.setattr(reminder_delivery, "STORE_PATH", path)
    monkeypatch.setattr(scheduled_items, "STORE_PATH", path)
    return path


def run(coroutine):
    return asyncio.run(coroutine)


def set_alarm(label: str, seconds: int, kind: str = "alarm", user_id: str = ANA) -> None:
    assert alarms_tools.schedule(
        label, NOW + seconds, kind, user_id=user_id, now=NOW
    )["status"] == "ok"


def set_reminder(title: str, seconds: int, user_id: str = ANA) -> None:
    assert reminders_tools.create_reminder(
        title=title, due="PT" + str(seconds) + "S", user_id=user_id, now=NOW
    )["status"] == "ok"


def rows(path: Path, sql: str, *parameters) -> list[tuple]:
    connection = sqlite3.connect(path)
    try:
        return connection.execute(sql, parameters).fetchall()
    finally:
        connection.close()


def alarm_rows(path: Path, user_id: str = ANA) -> list[tuple]:
    return rows(
        path,
        "SELECT kind, label, due_at FROM alarms WHERE user_id = ? AND delivered_at IS NULL "
        "ORDER BY rowid",
        user_id,
    )


def reminder_rows(path: Path, user_id: str = ANA) -> list[tuple]:
    return rows(
        path,
        "SELECT title, due_at, completed FROM reminders WHERE user_id = ? ORDER BY rowid",
        user_id,
    )


def _reader(model: FakeLocalModel) -> SemanticScheduleReader:
    return SemanticScheduleReader(classify=provider_classifier(model))


def _handler(
    model: FakeLocalModel,
    user_id: str | None = ANA,
    *,
    announce: Any = None,
    now: int = NOW,
) -> ScheduledChangeHandler:
    scope = (
        UserScope(user_id=user_id, identity_configured=True) if user_id else UserScope.anonymous()
    )
    return ScheduledChangeHandler(
        scope=scope, announce=announce, now=lambda: now, semantic=_reader(model)
    )


# --- the offline gate decides what a model may ever see ---------------------------------------


NATURAL = (
    "change that alarm to a reminder",
    "cancel my last timer",
    "move my reminder to in an hour",
    "rename the alarm to pick up the parcel",
    "actually make that alarm a reminder instead",
    "scrap the laundry reminder",
    "turn my standup alarm into a reminder please",
    "push the laundry reminder back by 2 hours",
    "i do not need that timer any more, drop it",
    "that alarm should have been a reminder",
    "call the alarm something else, the bread",
    "no longer need the standup reminder",
)


@pytest.mark.parametrize("text", NATURAL)
def test_a_natural_change_request_may_be_read_by_the_local_model(text) -> None:
    assert askable(text) is not None


NEVER_ASKED = (
    "",
    "   ",
    None,
    "set an alarm for ten minutes",
    "set a reminder for the standup",
    "remind me to take the bread out",
    "what alarms do i have",
    "which reminders are still on",
    "do i have a timer running?",
    "can you change alarms at all",
    "if i cancel that alarm will it still ring",
    "maybe cancel the timer later",
    "would you cancel her alarm",
    "cancel his reminder",
    "cancel their timer",
    "change the alarm and call 5551234567",
    "cancel the meeting with the bank",
    "tell me a joke",
    "what is on my calendar",
    "delete that email",
    "sure",
    "ok thanks",
)


@pytest.mark.parametrize("text", NEVER_ASKED)
def test_anything_conservative_safeguards_catch_never_reaches_a_model(text) -> None:
    assert askable(text) is None


def test_a_long_turn_is_never_shown_to_a_model() -> None:
    assert askable("cancel that alarm " * 40) is None


def test_the_text_shown_to_a_model_is_bounded_and_plain() -> None:
    shown = askable("Cancel My Last TIMER, please!")
    assert shown == "cancel my last timer please"
    assert len(shown) <= MAX_SEMANTIC_INPUT_CHARS


# --- the payload: the turn, and a bounded summary of what this owner has ----------------------


def _messages(model: FakeLocalModel) -> list[dict]:
    assert len(model.calls) == 1
    messages, tools = model.calls[0]
    assert tools is None
    return messages


def test_the_payload_is_the_prompt_the_turn_and_the_owner_own_candidates(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    model = FakeLocalModel()
    handler = _handler(model)
    session = FakeSession()

    assert run(handler.handle("change that alarm to a reminder", session)) is True

    messages = _messages(model)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == SCHEDULE_CHANGE_SYSTEM_PROMPT
    assert "change that alarm to a reminder" in messages[1]["content"]
    assert LAUNDRY in messages[1]["content"]


def test_the_payload_never_carries_an_id_an_owner_or_a_destination(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    identifiers = rows(store, "SELECT id FROM alarms")
    model = FakeLocalModel()

    run(_handler(model).handle("change that alarm to a reminder", FakeSession()))

    payload = json.dumps(_messages(model))
    assert ANA not in payload
    for (identifier,) in identifiers:
        assert identifier not in payload
    for forbidden in ("user_id", "alarm_id", "reminder_id", "telegram", "phone", "+1555"):
        assert forbidden not in payload.lower()


def test_the_payload_never_carries_the_items_of_another_owner(store) -> None:
    set_alarm(LAUNDRY, HOUR, user_id=BO)
    set_alarm(STANDUP, HOUR, user_id=ANA)
    model = FakeLocalModel(json.dumps(dict(action="cancel", item=1)))

    run(_handler(model).handle("cancel that alarm", FakeSession()))

    payload = json.dumps(_messages(model))
    assert LAUNDRY not in payload
    assert STANDUP in payload
    assert len(alarm_rows(store, BO)) == 1


def test_the_payload_never_carries_history_a_cache_or_a_tool_schema(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    model = FakeLocalModel()

    run(_handler(model).handle("change that alarm to a reminder", FakeSession()))

    messages = _messages(model)
    assert len(messages) == 2
    payload = json.dumps(messages).lower()
    for forbidden in ("assistant", "tool_call", "function", "inbox", "schedule.next", "email"):
        assert forbidden not in payload


def test_the_prompt_never_names_a_tool_a_destination_or_an_escalation() -> None:
    lowered = SCHEDULE_CHANGE_SYSTEM_PROMPT.lower()
    for forbidden in ("scheduled.change", "tool", "hermes", "http", "phone number", "chat id"):
        assert forbidden not in lowered


def test_nothing_of_the_turn_or_the_candidates_is_ever_logged(store, caplog) -> None:
    caplog.set_level("DEBUG")
    set_alarm(LAUNDRY, HOUR)
    model = FakeLocalModel()

    run(_handler(model).handle("change that alarm to a reminder", FakeSession()))

    written = " ".join(record.getMessage() for record in caplog.records).lower()
    assert "laundry" not in written
    assert "change that alarm" not in written
    assert ANA not in written


# --- the reply is read strictly, or not at all -------------------------------------------------


TURN = "move my reminder to in an hour"


def _parsed(reply: str, *, turn: str = TURN, count: int = 2):
    return parse_change_reply(reply, turn=turn, candidate_count=count)


def test_a_well_formed_reply_is_read() -> None:
    change = _parsed(json.dumps(dict(action="update", item=2, when="in an hour")))
    assert change is not None
    assert (change.action, change.index, change.when) == ("update", 2, "in an hour")
    assert change.title is None
    assert change.target_kind is None


def test_a_conversion_reply_carries_its_target() -> None:
    change = _parsed(
        json.dumps(dict(action="convert", item=1, target_kind="reminder")),
        turn="change that alarm to a reminder",
    )
    assert change is not None
    assert (change.action, change.target_kind) == ("convert", "reminder")


def test_no_item_means_the_newest_and_is_allowed() -> None:
    change = _parsed(json.dumps(dict(action="cancel", item=None)), turn="cancel my last timer")
    assert change is not None
    assert change.index is None


UNUSABLE = (
    "",
    "   ",
    "I have cancelled it for you.",
    "```json {}```",
    json.dumps(dict(action="cancel", item=1)) + " and also nothing else",
    json.dumps([dict(action="cancel")]),
    json.dumps(dict(action="cancel", item=1, user_id=ANA)),
    json.dumps(dict(action="cancel", item=1, reminder_id="abc")),
    json.dumps(dict(action="cancel", item=1, delivery=["call"])),
    json.dumps(dict(item=1)),
    json.dumps(dict(action="delete", item=1)),
    json.dumps(dict(action="complete", item=1)),
    json.dumps(dict(action="cancel", item=0)),
    json.dumps(dict(action="cancel", item=3)),
    json.dumps(dict(action="cancel", item=-1)),
    json.dumps(dict(action="cancel", item="1; drop table alarms")),
    json.dumps(dict(action="cancel", item=True)),
    json.dumps(dict(action="convert", item=1)),
    json.dumps(dict(action="convert", item=1, target_kind="calendar")),
    json.dumps(dict(action="update", item=1)),
    json.dumps(dict(action="update", item=1, when="tomorrow at nine")),
    json.dumps(dict(action="update", item=1, title="the dentist appointment")),
)


@pytest.mark.parametrize("reply", UNUSABLE)
def test_a_reply_off_contract_decides_nothing(reply) -> None:
    assert _parsed(reply) is None


def test_a_reply_saying_nothing_was_meant_decides_nothing() -> None:
    assert _parsed(json.dumps(dict(action="none"))) is None


def test_a_new_time_has_to_be_words_the_person_actually_said() -> None:
    assert _parsed(json.dumps(dict(action="update", item=1, when="in an hour"))) is not None
    assert _parsed(json.dumps(dict(action="update", item=1, when="in 5 minutes"))) is None


def test_a_new_title_has_to_be_words_the_person_actually_said() -> None:
    turn = "rename the alarm to pick up the parcel"
    assert _parsed(
        json.dumps(dict(action="update", item=1, title="pick up the parcel")), turn=turn
    ) is not None
    assert _parsed(
        json.dumps(dict(action="update", item=1, title="collect the package")), turn=turn
    ) is None


def test_an_enormous_reply_is_discarded_whatever_is_inside_it() -> None:
    assert _parsed(json.dumps(dict(action="cancel", item=1, when="x" * 5000))) is None


# --- an unreachable or slow model leaves the turn exactly where it was -------------------------


def test_an_unreachable_model_changes_nothing(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    model = FakeLocalModel(error=RuntimeError("nope"))
    session = FakeSession()

    assert run(_handler(model).handle("cancel that alarm", session)) is False
    assert len(alarm_rows(store)) == 1
    assert session.spoken == []


def test_a_slow_model_changes_nothing(store) -> None:
    set_alarm(LAUNDRY, HOUR)

    class SlowModel(FakeLocalModel):
        async def chat(self, messages: Any, tools: Any = None, **_: Any) -> Any:
            await asyncio.sleep(5)
            raise AssertionError("the reading should have timed out")

    reader = SemanticScheduleReader(
        classify=provider_classifier(SlowModel()), timeout_seconds=0.1
    )
    handler = ScheduledChangeHandler(
        scope=UserScope(user_id=ANA, identity_configured=True),
        now=lambda: NOW,
        semantic=reader,
    )
    assert run(handler.handle("cancel that alarm", FakeSession())) is False
    assert len(alarm_rows(store)) == 1


def test_with_nothing_pending_no_model_is_asked_at_all(store) -> None:
    model = RefusingModel()
    assert run(_handler(model).handle("cancel that alarm", FakeSession())) is False
    assert model.calls == []


def test_an_anonymous_session_never_reaches_a_model(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    model = RefusingModel()
    assert run(_handler(model, user_id=None).handle("cancel that alarm", FakeSession())) is False
    assert model.calls == []
    assert len(alarm_rows(store)) == 1


# --- the route, end to end ---------------------------------------------------------------------


def test_change_that_alarm_to_a_reminder_really_converts_it(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    model = FakeLocalModel(CONVERT)
    session = FakeSession()
    announce = Announcer()
    handler = _handler(model, announce=announce)

    assert run(handler.handle("change that alarm to a reminder", session)) is True

    assert reminder_rows(store) == [(LAUNDRY, NOW + HOUR, 0)]
    assert alarm_rows(store) == [("reminder", LAUNDRY, NOW + HOUR)]
    assert announce.published == 1
    assert session.spoken and "reminder" in session.spoken[0].lower()


def test_cancel_my_last_timer_really_cancels_it(store) -> None:
    set_alarm(LAUNDRY, HOUR, kind="timer")
    model = FakeLocalModel(CANCEL)
    session = FakeSession()
    announce = Announcer()

    assert run(_handler(model, announce=announce).handle("cancel my last timer", session)) is True

    assert alarm_rows(store) == []
    assert announce.published == 1


def test_move_my_reminder_to_in_an_hour_really_moves_it(store) -> None:
    set_reminder(STANDUP, 10 * 60)
    reply = json.dumps(dict(action="update", item=1, when="in an hour"))
    model = FakeLocalModel(reply)

    assert run(_handler(model).handle("move my reminder to in an hour", FakeSession())) is True

    assert reminder_rows(store) == [(STANDUP, NOW + HOUR, 0)]


def test_rename_the_alarm_really_renames_it(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    reply = json.dumps(dict(action="update", item=1, title="pick up the parcel"))
    model = FakeLocalModel(reply)
    turn = "rename the alarm to pick up the parcel"

    assert run(_handler(model).handle(turn, FakeSession())) is True

    assert alarm_rows(store) == [("alarm", "pick up the parcel", NOW + HOUR)]


LESS_FORMULAIC = (
    "actually make that alarm a reminder instead",
    "that alarm should have been a reminder",
    "turn my laundry alarm into a reminder please",
)


@pytest.mark.parametrize("text", LESS_FORMULAIC)
def test_a_less_formulaic_phrasing_reaches_the_same_safe_mutation(store, text) -> None:
    set_alarm(LAUNDRY, HOUR)

    assert run(_handler(FakeLocalModel(CONVERT)).handle(text, FakeSession())) is True

    assert reminder_rows(store) == [(LAUNDRY, NOW + HOUR, 0)]
    assert alarm_rows(store) == [("reminder", LAUNDRY, NOW + HOUR)]


# --- and everything that must change nothing ----------------------------------------------------


@pytest.mark.parametrize("text", NEVER_ASKED[3:])
def test_a_turn_the_gate_refuses_is_left_to_ordinary_routing(store, text) -> None:
    set_alarm(LAUNDRY, HOUR)
    model = RefusingModel()
    session = FakeSession()
    announce = Announcer()

    assert run(_handler(model, announce=announce).handle(text, session)) is False

    assert model.calls == []
    assert alarm_rows(store) == [("alarm", LAUNDRY, NOW + HOUR)]
    assert session.spoken == []
    assert announce.published == 0


def test_a_reading_the_model_could_not_make_is_left_to_ordinary_routing(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    model = FakeLocalModel(json.dumps(dict(action="none")))
    announce = Announcer()

    assert run(_handler(model, announce=announce).handle("cancel that alarm", FakeSession())) is (
        False
    )
    assert alarm_rows(store) == [("alarm", LAUNDRY, NOW + HOUR)]
    assert announce.published == 0


def test_a_named_reference_that_is_ambiguous_asks_and_changes_nothing(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    set_alarm(STANDUP, 2 * HOUR)
    model = FakeLocalModel(json.dumps(dict(action="cancel", item=None)))
    session = FakeSession()
    announce = Announcer()

    claimed = run(_handler(model, announce=announce).handle("cancel the alarm", session))

    assert claimed is True
    assert len(alarm_rows(store)) == 2
    assert announce.published == 0
    assert session.spoken and "which" in session.spoken[0].lower()


def test_an_item_number_the_server_never_offered_changes_nothing(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    model = FakeLocalModel(json.dumps(dict(action="cancel", item=4)))
    announce = Announcer()

    assert run(_handler(model, announce=announce).handle("cancel that alarm", FakeSession())) is (
        False
    )
    assert alarm_rows(store) == [("alarm", LAUNDRY, NOW + HOUR)]
    assert announce.published == 0


def test_a_malformed_reading_changes_nothing(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    model = FakeLocalModel("I went ahead and cancelled every alarm you have.")
    announce = Announcer()

    assert run(_handler(model, announce=announce).handle("cancel that alarm", FakeSession())) is (
        False
    )
    assert alarm_rows(store) == [("alarm", LAUNDRY, NOW + HOUR)]
    assert announce.published == 0


def test_an_item_that_already_came_due_is_never_reached_this_way(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    model = RefusingModel()
    late = _handler(model, now=NOW + 2 * HOUR)

    assert run(late.handle("cancel that alarm", FakeSession())) is False
    assert model.calls == []
    assert len(alarm_rows(store)) == 1


def test_one_owner_can_never_reach_the_item_of_another(store) -> None:
    set_alarm(LAUNDRY, HOUR, user_id=BO)
    model = RefusingModel()

    assert run(_handler(model, user_id=ANA).handle("cancel that alarm", FakeSession())) is False
    assert model.calls == []
    assert len(alarm_rows(store, BO)) == 1


def test_a_failed_change_speaks_no_success_and_announces_nothing(store, monkeypatch) -> None:
    set_alarm(LAUNDRY, HOUR)

    def _explode(*_, **__):
        raise sqlite3.OperationalError("no")

    monkeypatch.setattr(scheduled_items, "_insert_reminder", _explode)
    session = FakeSession()
    announce = Announcer()

    claimed = run(
        _handler(FakeLocalModel(CONVERT), announce=announce).handle(
            "change that alarm to a reminder", session
        )
    )

    assert claimed is True
    assert announce.published == 0
    assert alarm_rows(store) == [("alarm", LAUNDRY, NOW + HOUR)]
    assert reminder_rows(store) == []
    assert session.spoken
    assert "reminder is set" not in session.spoken[0].lower()


# --- the direct reply, and the stream it must never start --------------------------------------


def test_the_result_is_spoken_directly_and_no_second_generation_is_asked_for(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    model = FakeLocalModel(CONVERT)
    session = FakeSession()

    assert run(_handler(model).handle("change that alarm to a reminder", session)) is True

    # One call: the bounded classification. The outcome is spoken as it stands,
    # so no tool history is ever handed back to a model to narrate.
    assert len(model.calls) == 1
    assert len(session.spoken) == 1


def test_the_spoken_result_never_claims_a_call_or_a_message(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    session = FakeSession()

    run(_handler(FakeLocalModel(CANCEL)).handle("cancel my last alarm", session))

    lowered = " ".join(session.spoken).lower()
    for claim in ("i called", "calling you", "sent to your telegram", "messaged you"):
        assert claim not in lowered


# --- what the dashboard feed of this owner is then shown ---------------------------------------


def test_the_owner_feed_reflects_a_conversion(store) -> None:
    set_alarm(LAUNDRY, HOUR)

    run(_handler(FakeLocalModel(CONVERT)).handle("change that alarm to a reminder", FakeSession()))

    assert alarms_tools.dashboard_alarms(user_id=ANA, now=NOW) == []
    assert [item["title"] for item in reminders_tools.dashboard_reminders(user_id=ANA)] == [
        LAUNDRY
    ]


def test_the_owner_feed_reflects_a_cancellation(store) -> None:
    set_alarm(LAUNDRY, HOUR)

    run(_handler(FakeLocalModel(CANCEL)).handle("cancel my last alarm", FakeSession()))

    assert alarms_tools.dashboard_alarms(user_id=ANA, now=NOW) == []


def test_the_owner_feed_reflects_an_update(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    reply = json.dumps(dict(action="update", item=1, when="in 2 hours"))

    run(_handler(FakeLocalModel(reply)).handle("move that alarm to in 2 hours", FakeSession()))

    shown = alarms_tools.dashboard_alarms(user_id=ANA, now=NOW)
    assert [item["due"] for item in shown] == [alarms_tools._iso(NOW + 2 * HOUR)]


# --- the authorised one-off repair of the item that is already out there -----------------------


def test_the_latest_pending_alarm_can_be_repaired_into_a_reminder_unseen(store, caplog) -> None:
    caplog.set_level("DEBUG")
    set_alarm(LAUNDRY, HOUR)

    outcome = scheduled_items.repair_latest_alarm_to_reminder(ANA, now=NOW)

    assert outcome["status"] == "ok"
    assert outcome["source_kind"] == "alarm"
    assert outcome["target_kind"] == "reminder"
    # Counts and kinds only: the label of the item never leaves and never lands
    # in a log, which is the whole point of doing the repair through this path.
    assert LAUNDRY not in json.dumps(outcome)
    assert "laundry" not in " ".join(r.getMessage() for r in caplog.records).lower()

    assert [kind for kind, _, _ in alarm_rows(store)] == ["reminder"]
    assert len(reminder_rows(store)) == 1
    assert outcome["alarms_pending"] == 0
    assert outcome["reminders_pending"] == 1


def test_the_repair_never_sends_or_calls_anything(store) -> None:
    set_alarm(LAUNDRY, HOUR)

    scheduled_items.repair_latest_alarm_to_reminder(ANA, now=NOW)

    channels = rows(store, "SELECT DISTINCT channel FROM reminder_deliveries")
    assert channels == [(reminder_delivery.SPEAK,)]
    assert reminder_delivery.claim_due("test-worker", NOW + 10 * HOUR) == []


def test_the_repair_refuses_when_there_is_no_single_pending_alarm(store) -> None:
    assert scheduled_items.repair_latest_alarm_to_reminder(ANA, now=NOW)["status"] != "ok"
    set_alarm(LAUNDRY, HOUR)
    set_alarm(STANDUP, 2 * HOUR)
    outcome = scheduled_items.repair_latest_alarm_to_reminder(ANA, now=NOW, expect_pending=1)
    assert outcome["status"] != "ok"
    assert len(alarm_rows(store)) == 2


def test_the_repair_leaves_no_question_nobody_was_asked(store) -> None:
    set_alarm(LAUNDRY, HOUR)

    scheduled_items.repair_latest_alarm_to_reminder(ANA, now=NOW)

    # The conversation path arms the spoken channel and asks how they want it.
    # A repair has no conversation in it, so there is nobody to have asked --
    # and an open question nobody heard would read a later "call me" as an
    # answer to it.
    assert reminder_delivery.awaiting_reminder(ANA, NOW) is None


def test_the_repair_never_crosses_an_owner_boundary(store) -> None:
    set_alarm(LAUNDRY, HOUR, user_id=BO)

    assert scheduled_items.repair_latest_alarm_to_reminder(ANA, now=NOW)["status"] != "ok"
    assert len(alarm_rows(store, BO)) == 1
    assert reminder_rows(store, BO) == []


# --- the wiring: the local model, and only the local model -------------------------------------


def _voice_agent(name: str):
    module_path = Path(__file__).parents[1] / "voice_agent.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_reader_runs_on_the_local_model_and_never_on_hermes(store) -> None:
    voice_agent = _voice_agent("voice_agent_for_schedule_semantics")
    set_alarm(LAUNDRY, HOUR)
    primary = FakeLocalModel(CONVERT)
    escalation = RefusingModel()
    routed = SimpleNamespace(
        provider_instance=SimpleNamespace(primary=primary, escalation=escalation)
    )

    handler = voice_agent.build_schedule_change_handler(
        UserScope(user_id=ANA, identity_configured=True), provider=routed, now=lambda: NOW
    )

    assert handler is not None
    assert voice_agent.work_router_provider(routed) is primary
    assert run(handler.handle("change that alarm to a reminder", FakeSession())) is True
    assert len(primary.calls) == 1
    assert escalation.calls == []


def test_without_a_provider_there_is_no_semantic_route_at_all() -> None:
    voice_agent = _voice_agent("voice_agent_schedule_no_provider")

    handler = voice_agent.build_schedule_change_handler(
        UserScope(user_id=ANA, identity_configured=True)
    )

    assert handler is None or handler.semantic is None or handler.semantic.enabled is False


def test_an_anonymous_session_gets_no_schedule_route() -> None:
    voice_agent = _voice_agent("voice_agent_schedule_anonymous")

    assert voice_agent.build_schedule_change_handler(UserScope.anonymous()) is None


async def _never_end() -> None:
    raise AssertionError("a scheduled change must not end the call")


def _local_turn(voice_agent, session, handler):
    return voice_agent.LocalTurnHandler(
        phone_handoff=None,
        session=session,
        end_call=_never_end,
        background=RefusingBackground(),
        schedule=handler,
    )


def test_a_spoken_change_request_is_claimed_before_any_other_route(store) -> None:
    voice_agent = _voice_agent("voice_agent_schedule_speech")
    set_alarm(LAUNDRY, HOUR)
    session = FakeSession()
    text = "change that alarm to a reminder"
    local = _local_turn(voice_agent, session, _handler(FakeLocalModel(CONVERT)))

    async def turn() -> bool:
        local.on_final_transcript(text)
        return await local.turn_consumed(text)

    assert run(turn()) is True
    assert reminder_rows(store) == [(LAUNDRY, NOW + HOUR, 0)]


def test_typed_chat_gets_the_same_change_route(store) -> None:
    voice_agent = _voice_agent("voice_agent_schedule_typed")
    set_alarm(LAUNDRY, HOUR, kind="timer")
    session = FakeSession()
    local = _local_turn(voice_agent, session, _handler(FakeLocalModel(CANCEL)))

    assert run(local.turn_consumed("cancel my last timer")) is True
    assert alarm_rows(store) == []
