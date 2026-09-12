"""Changing, converting and cancelling a scheduled item that already exists.

Until this existed CAAL could only ever *add* one. Asked to turn an alarm into
a reminder it searched the schedule, found the alarm, and then called the only
scheduled write it had -- ``alarms.set`` -- so a second alarm appeared and the
spoken reply sounded like a change had been made. Nothing had been changed.

These tests pin the server-side half of the fix: one bounded mutation surface
that resolves the pending items of its own owner and then cancels, updates or
converts exactly one of them, atomically.

The boundaries they hold:

* the caller never names a row: no id, no owner id, no destination, no column.
  It names an action, a natural reference, optionally a kind, a new time, a new
  title and a conversion target, and nothing else;
* resolution is the business of the server. It sees only the items of this one
  owner, only ones still pending, resolves "that one" as the newest, and
  refuses -- in plain words, changing nothing -- when a named reference matches
  nothing or matches several;
* a conversion is atomic: the source is gone, exactly one target exists, and
  the delivery ledger of the source is stood down before the target is written,
  so nothing is ever armed twice and no call or message is left queued;
* a cancellation removes the item and every pending delivery row with it, is
  idempotent, and never claims anything was sent;
* nothing here ever places a call, sends a message, or reads a delivered item.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from caal.tools import alarms_tools, reminder_delivery, reminders_tools, scheduled_items
from caal.tools.errors import SafeToolError
from caal.user_scope import SCHEDULING_UNAVAILABLE_REPLY

ANA = "usr_" + "a" * 24
BO = "usr_" + "b" * 24
NOW = 1_000_000
HOUR = 3600

LAUNDRY = "take the laundry out"
STANDUP = "standup with the team"


@pytest.fixture
def store(monkeypatch, tmp_path):
    path = tmp_path / "assistant.sqlite3"
    monkeypatch.setattr(alarms_tools, "STORE_PATH", path)
    monkeypatch.setattr(reminders_tools, "STORE_PATH", path)
    monkeypatch.setattr(reminder_delivery, "STORE_PATH", path)
    monkeypatch.setattr(scheduled_items, "STORE_PATH", path)
    return path


@pytest.fixture
def everything_available(monkeypatch):
    monkeypatch.setattr(reminder_delivery, "telegram_owner", lambda: ANA)
    monkeypatch.setattr(reminder_delivery, "telegram_configured", lambda: True)
    monkeypatch.setattr(
        reminder_delivery,
        "resolve_callback_number",
        lambda user_id: "+15551230000" if user_id == ANA else None,
    )


# --- little readers over the store, so a test asserts on state and not on words ---------------


def _rows(path: Path, sql: str, *parameters) -> list[tuple]:
    connection = sqlite3.connect(path)
    try:
        return connection.execute(sql, parameters).fetchall()
    finally:
        connection.close()


def alarm_rows(path: Path, user_id: str = ANA) -> list[tuple]:
    return _rows(
        path,
        "SELECT kind, label, due_at FROM alarms WHERE user_id = ? AND delivered_at IS NULL "
        "ORDER BY rowid",
        user_id,
    )


def reminder_rows(path: Path, user_id: str = ANA) -> list[tuple]:
    return _rows(
        path,
        "SELECT title, due_at, completed FROM reminders WHERE user_id = ? ORDER BY rowid",
        user_id,
    )


def delivery_rows(path: Path, user_id: str = ANA) -> list[tuple]:
    return _rows(
        path,
        "SELECT channel, state FROM reminder_deliveries WHERE user_id = ? ORDER BY rowid",
        user_id,
    )


def nothing_left_to_dispatch(now: int = NOW + 10 * HOUR) -> bool:
    """Whether a durable worker would find any call or message to make."""
    return reminder_delivery.claim_due("test-worker", now) == []


def set_alarm(label: str, seconds: int, kind: str = "alarm", user_id: str = ANA) -> None:
    result = alarms_tools.schedule(label, NOW + seconds, kind, user_id=user_id, now=NOW)
    assert result["status"] == "ok"


def set_reminder(
    title: str, seconds: int | None, user_id: str = ANA, delivery: list[str] | None = None
) -> dict:
    result = reminders_tools.create_reminder(
        title=title,
        due=None if seconds is None else "PT" + str(seconds) + "S",
        delivery=delivery,
        user_id=user_id,
        now=NOW,
    )
    assert result["status"] == "ok"
    return result


# --- what a caller may say ---------------------------------------------------------------------


def test_the_mutation_surface_accepts_no_identifier_of_any_kind() -> None:
    schema = scheduled_items.CHANGE_SCHEMA
    assert set(schema["properties"]) == {
        "action",
        "reference",
        "kind",
        "when",
        "title",
        "target_kind",
    }
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["action"]
    assert schema["properties"]["action"]["enum"] == list(scheduled_items.ACTIONS)
    assert schema["properties"]["kind"]["enum"] == list(scheduled_items.KINDS)
    assert schema["properties"]["target_kind"]["enum"] == list(scheduled_items.KINDS)


@pytest.mark.parametrize(
    "argument",
    [
        "id",
        "item_id",
        "alarm_id",
        "reminder_id",
        "user_id",
        "owner",
        "phone",
        "chat_id",
        "delivery",
        "completed",
        "channels",
        "number",
    ],
)
def test_no_identifier_or_destination_is_a_declared_argument(argument) -> None:
    assert argument not in scheduled_items.CHANGE_SCHEMA["properties"]


def test_an_unsigned_in_session_can_change_nothing(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    result = scheduled_items.change_scheduled_item(
        action="cancel", reference="that one", user_id=None, now=NOW, identity_configured=True
    )
    assert result["status"] == "unauthorized"
    assert result["message"] == SCHEDULING_UNAVAILABLE_REPLY
    assert len(alarm_rows(store)) == 1


@pytest.mark.parametrize("action", ["delete", "", None, 7, "drop table", "convert_all"])
def test_an_action_outside_the_enum_changes_nothing(store, action) -> None:
    set_alarm(LAUNDRY, HOUR)
    result = scheduled_items.change_scheduled_item(
        action=action, reference="that one", user_id=ANA, now=NOW
    )
    assert result["status"] != "ok"
    assert len(alarm_rows(store)) == 1


# --- resolution is the business of the server --------------------------------------------------


def test_that_one_is_the_newest_pending_item_of_this_owner(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    set_alarm(STANDUP, 2 * HOUR, kind="timer")
    found = scheduled_items.resolve(ANA, NOW, reference="that one", kind=None)
    assert (found.kind, found.title) == ("timer", STANDUP)


@pytest.mark.parametrize("reference", ["that one", "this one", "the last one", "it", "", None])
def test_every_way_of_saying_the_last_one_resolves_the_same(store, reference) -> None:
    set_alarm(LAUNDRY, HOUR)
    assert scheduled_items.resolve(ANA, NOW, reference=reference, kind=None).title == LAUNDRY


def test_a_named_reference_matches_the_words_of_the_owner(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    set_alarm(STANDUP, 2 * HOUR)
    found = scheduled_items.resolve(ANA, NOW, reference="the laundry alarm", kind=None)
    assert found.title == LAUNDRY


def test_a_kind_narrows_the_candidates(store) -> None:
    set_alarm(LAUNDRY, HOUR, kind="alarm")
    set_reminder(STANDUP, 2 * HOUR)
    alarm = scheduled_items.resolve(ANA, NOW, reference="the last one", kind="alarm")
    reminder = scheduled_items.resolve(ANA, NOW, reference="the last one", kind="reminder")
    assert alarm.kind == "alarm"
    assert reminder.kind == "reminder"


def test_a_reference_matching_several_items_refuses_rather_than_guessing(store) -> None:
    set_alarm("meeting with the bank", HOUR)
    set_alarm("meeting with the school", 2 * HOUR)
    with pytest.raises(SafeToolError):
        scheduled_items.resolve(ANA, NOW, reference="the meeting alarm", kind=None)


def test_a_reference_matching_nothing_refuses(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    with pytest.raises(SafeToolError):
        scheduled_items.resolve(ANA, NOW, reference="the dentist alarm", kind=None)


def test_with_nothing_pending_there_is_nothing_to_resolve(store) -> None:
    with pytest.raises(SafeToolError):
        scheduled_items.resolve(ANA, NOW, reference="that one", kind=None)


def test_an_item_that_has_already_come_due_is_never_a_candidate(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    with pytest.raises(SafeToolError):
        scheduled_items.resolve(ANA, NOW + 2 * HOUR, reference="that one", kind=None)


def test_a_delivered_item_is_never_a_candidate(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    claimed = alarms_tools.claim_due_alarms(now=NOW + 2 * HOUR, user_id=ANA)
    alarms_tools.mark_delivered([row["id"] for row in claimed], now=NOW + 2 * HOUR)
    with pytest.raises(SafeToolError):
        scheduled_items.resolve(ANA, NOW, reference="that one", kind=None)


def test_a_completed_reminder_is_never_a_candidate(store) -> None:
    set_reminder(STANDUP, HOUR)
    connection = sqlite3.connect(store)
    connection.execute("UPDATE reminders SET completed = 1")
    connection.commit()
    connection.close()
    with pytest.raises(SafeToolError):
        scheduled_items.resolve(ANA, NOW, reference="the standup reminder", kind="reminder")


def test_the_spoken_channel_of_a_reminder_is_not_a_second_candidate(store) -> None:
    set_reminder(STANDUP, HOUR)
    found = scheduled_items.candidates(ANA, NOW)
    assert [(item.kind, item.title) for item in found] == [("reminder", STANDUP)]


def test_one_owner_never_sees_or_touches_an_item_of_another(store) -> None:
    set_alarm(LAUNDRY, HOUR, user_id=BO)
    assert scheduled_items.candidates(ANA, NOW) == []
    result = scheduled_items.change_scheduled_item(
        action="cancel", reference="the laundry alarm", user_id=ANA, now=NOW
    )
    assert result["status"] != "ok"
    assert len(alarm_rows(store, BO)) == 1


# --- cancel ------------------------------------------------------------------------------------


def test_cancelling_an_alarm_removes_it_and_says_only_that(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    result = scheduled_items.change_scheduled_item(
        action="cancel", reference="that alarm", user_id=ANA, now=NOW
    )
    assert result["status"] == "ok"
    assert alarm_rows(store) == []
    assert result["data"]["action"] == "cancel"
    assert result["data"]["kind"] == "alarm"
    lowered = result["message"].lower()
    assert "cancel" in lowered
    for claim in ("called", "calling", "telegram", "message", "sent"):
        assert claim not in lowered


def test_cancelling_a_reminder_takes_every_pending_delivery_with_it(
    store, everything_available
) -> None:
    set_reminder(STANDUP, HOUR, delivery=["all"])
    armed = set(channel for channel, _ in delivery_rows(store))
    assert armed == set(reminder_delivery.CHANNELS)
    result = scheduled_items.change_scheduled_item(
        action="cancel", reference="the standup reminder", user_id=ANA, now=NOW
    )
    assert result["status"] == "ok"
    assert reminder_rows(store) == []
    assert delivery_rows(store) == []
    assert alarm_rows(store) == []
    assert nothing_left_to_dispatch()


def test_cancelling_twice_is_safe_and_the_second_time_says_there_is_nothing(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    first = scheduled_items.change_scheduled_item(
        action="cancel", reference="the laundry alarm", user_id=ANA, now=NOW
    )
    second = scheduled_items.change_scheduled_item(
        action="cancel", reference="the laundry alarm", user_id=ANA, now=NOW
    )
    assert first["status"] == "ok"
    assert second["status"] != "ok"
    assert alarm_rows(store) == []


def test_cancelling_an_undated_reminder_works_too(store) -> None:
    set_reminder(STANDUP, None)
    result = scheduled_items.change_scheduled_item(
        action="cancel", reference="the standup reminder", user_id=ANA, now=NOW
    )
    assert result["status"] == "ok"
    assert reminder_rows(store) == []


# --- update ------------------------------------------------------------------------------------


def test_moving_an_alarm_keeps_its_label_and_changes_only_its_time(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    result = scheduled_items.change_scheduled_item(
        action="update", reference="that alarm", when="in 2 hours", user_id=ANA, now=NOW
    )
    assert result["status"] == "ok"
    assert alarm_rows(store) == [("alarm", LAUNDRY, NOW + 2 * HOUR)]


def test_renaming_an_alarm_keeps_its_time_and_changes_only_its_label(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    result = scheduled_items.change_scheduled_item(
        action="update", reference="that alarm", title=STANDUP, user_id=ANA, now=NOW
    )
    assert result["status"] == "ok"
    assert alarm_rows(store) == [("alarm", STANDUP, NOW + HOUR)]


def test_moving_a_reminder_moves_its_spoken_alarm_and_its_delivery_rows(
    store, everything_available
) -> None:
    set_reminder(STANDUP, HOUR, delivery=["speak", "call"])
    result = scheduled_items.change_scheduled_item(
        action="update", reference="that reminder", when="in 2 hours", user_id=ANA, now=NOW
    )
    assert result["status"] == "ok"
    assert reminder_rows(store) == [(STANDUP, NOW + 2 * HOUR, 0)]
    assert alarm_rows(store) == [("reminder", STANDUP, NOW + 2 * HOUR)]
    due = _rows(store, "SELECT DISTINCT due_at FROM reminder_deliveries WHERE user_id = ?", ANA)
    assert due == [(NOW + 2 * HOUR,)]


def test_renaming_a_reminder_renames_the_alarm_behind_its_spoken_channel(store) -> None:
    set_reminder(STANDUP, HOUR)
    result = scheduled_items.change_scheduled_item(
        action="update", reference="that reminder", title=LAUNDRY, user_id=ANA, now=NOW
    )
    assert result["status"] == "ok"
    assert reminder_rows(store) == [(LAUNDRY, NOW + HOUR, 0)]
    assert alarm_rows(store) == [("reminder", LAUNDRY, NOW + HOUR)]


def test_an_update_that_changes_nothing_asks_rather_than_pretending(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    result = scheduled_items.change_scheduled_item(
        action="update", reference="that alarm", user_id=ANA, now=NOW
    )
    assert result["status"] != "ok"
    assert alarm_rows(store) == [("alarm", LAUNDRY, NOW + HOUR)]


@pytest.mark.parametrize("when", ["later", "soon", "in a bit", "sometime", "tomorrowish"])
def test_a_vague_time_is_refused_and_the_item_is_left_alone(store, when) -> None:
    set_alarm(LAUNDRY, HOUR)
    result = scheduled_items.change_scheduled_item(
        action="update", reference="that alarm", when=when, user_id=ANA, now=NOW
    )
    assert result["status"] != "ok"
    assert alarm_rows(store) == [("alarm", LAUNDRY, NOW + HOUR)]


def test_a_new_time_already_past_is_refused(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    result = scheduled_items.change_scheduled_item(
        action="update",
        reference="that alarm",
        when="2000-01-01T00:00:00+00:00",
        user_id=ANA,
        now=NOW,
    )
    assert result["status"] != "ok"
    assert alarm_rows(store) == [("alarm", LAUNDRY, NOW + HOUR)]


# --- convert -----------------------------------------------------------------------------------


def test_an_alarm_becomes_a_reminder_at_the_same_time_with_the_same_words(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    result = scheduled_items.change_scheduled_item(
        action="convert", reference="that alarm", target_kind="reminder", user_id=ANA, now=NOW
    )
    assert result["status"] == "ok"
    assert reminder_rows(store) == [(LAUNDRY, NOW + HOUR, 0)]
    # The source alarm is gone; the only alarm row left is the spoken channel
    # of the new reminder, which is what announces it.
    assert alarm_rows(store) == [("reminder", LAUNDRY, NOW + HOUR)]
    assert [channel for channel, _ in delivery_rows(store)] == [reminder_delivery.SPEAK]


def test_converting_an_alarm_arms_no_remote_channel_at_all(store, everything_available) -> None:
    set_alarm(LAUNDRY, HOUR)
    scheduled_items.change_scheduled_item(
        action="convert", reference="that alarm", target_kind="reminder", user_id=ANA, now=NOW
    )
    channels = set(channel for channel, _ in delivery_rows(store))
    assert reminder_delivery.TELEGRAM not in channels
    assert reminder_delivery.CALL not in channels
    assert nothing_left_to_dispatch()


def test_converting_an_alarm_leaves_the_delivery_question_open_for_this_owner(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    result = scheduled_items.change_scheduled_item(
        action="convert", reference="that alarm", target_kind="reminder", user_id=ANA, now=NOW
    )
    assert result["data"]["delivery_pending"] is True
    assert reminder_delivery.awaiting_reminder(ANA, NOW) is not None


def test_a_timer_becomes_a_reminder_too(store) -> None:
    set_alarm(LAUNDRY, HOUR, kind="timer")
    result = scheduled_items.change_scheduled_item(
        action="convert", reference="that timer", target_kind="reminder", user_id=ANA, now=NOW
    )
    assert result["status"] == "ok"
    assert reminder_rows(store) == [(LAUNDRY, NOW + HOUR, 0)]


def test_a_reminder_becomes_an_alarm_and_its_delivery_ledger_stands_down(
    store, everything_available
) -> None:
    set_reminder(STANDUP, HOUR, delivery=["all"])
    result = scheduled_items.change_scheduled_item(
        action="convert", reference="that reminder", target_kind="alarm", user_id=ANA, now=NOW
    )
    assert result["status"] == "ok"
    assert reminder_rows(store) == []
    assert delivery_rows(store) == []
    assert alarm_rows(store) == [("alarm", STANDUP, NOW + HOUR)]
    assert nothing_left_to_dispatch()


def test_a_reminder_becomes_a_timer_keeping_its_due_time_and_words(store) -> None:
    set_reminder(STANDUP, HOUR)
    result = scheduled_items.change_scheduled_item(
        action="convert", reference="that reminder", target_kind="timer", user_id=ANA, now=NOW
    )
    assert result["status"] == "ok"
    assert alarm_rows(store) == [("timer", STANDUP, NOW + HOUR)]
    assert reminder_rows(store) == []


def test_a_conversion_may_carry_a_new_time_and_a_new_title(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    result = scheduled_items.change_scheduled_item(
        action="convert",
        reference="that alarm",
        target_kind="reminder",
        when="in 2 hours",
        title=STANDUP,
        user_id=ANA,
        now=NOW,
    )
    assert result["status"] == "ok"
    assert reminder_rows(store) == [(STANDUP, NOW + 2 * HOUR, 0)]


def test_converting_an_undated_reminder_to_an_alarm_asks_for_a_time(store) -> None:
    set_reminder(STANDUP, None)
    result = scheduled_items.change_scheduled_item(
        action="convert", reference="that reminder", target_kind="alarm", user_id=ANA, now=NOW
    )
    assert result["status"] != "ok"
    assert reminder_rows(store) == [(STANDUP, None, 0)]
    assert alarm_rows(store) == []


def test_converting_something_to_what_it_already_is_changes_nothing(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    result = scheduled_items.change_scheduled_item(
        action="convert", reference="that alarm", target_kind="alarm", user_id=ANA, now=NOW
    )
    assert result["status"] != "ok"
    assert alarm_rows(store) == [("alarm", LAUNDRY, NOW + HOUR)]


def test_a_conversion_with_no_target_named_refuses(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    result = scheduled_items.change_scheduled_item(
        action="convert", reference="that alarm", user_id=ANA, now=NOW
    )
    assert result["status"] != "ok"
    assert alarm_rows(store) == [("alarm", LAUNDRY, NOW + HOUR)]


def test_a_failed_conversion_leaves_the_source_exactly_as_it_was(store, monkeypatch) -> None:
    set_alarm(LAUNDRY, HOUR)

    def _explode(*_, **__):
        raise sqlite3.OperationalError("no")

    monkeypatch.setattr(scheduled_items, "_insert_reminder", _explode)
    result = scheduled_items.change_scheduled_item(
        action="convert", reference="that alarm", target_kind="reminder", user_id=ANA, now=NOW
    )
    assert result["status"] == "error"
    assert alarm_rows(store) == [("alarm", LAUNDRY, NOW + HOUR)]
    assert reminder_rows(store) == []


# --- what the dashboard is then shown ----------------------------------------------------------


def test_the_dashboard_stops_showing_a_cancelled_alarm(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    scheduled_items.change_scheduled_item(
        action="cancel", reference="that alarm", user_id=ANA, now=NOW
    )
    assert alarms_tools.dashboard_alarms(user_id=ANA, now=NOW) == []


def test_the_dashboard_shows_only_the_target_after_a_conversion(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    scheduled_items.change_scheduled_item(
        action="convert", reference="that alarm", target_kind="reminder", user_id=ANA, now=NOW
    )
    assert alarms_tools.dashboard_alarms(user_id=ANA, now=NOW) == []
    shown = reminders_tools.dashboard_reminders(user_id=ANA)
    assert [item["title"] for item in shown] == [LAUNDRY]


def test_the_dashboard_shows_the_new_time_after_an_update(store) -> None:
    set_alarm(LAUNDRY, HOUR)
    scheduled_items.change_scheduled_item(
        action="update", reference="that alarm", when="in 2 hours", user_id=ANA, now=NOW
    )
    shown = alarms_tools.dashboard_alarms(user_id=ANA, now=NOW)
    assert len(shown) == 1
    assert shown[0]["due"] == alarms_tools._iso(NOW + 2 * HOUR)


# --- the change is announced exactly once, and only when it happened ---------------------------


def test_a_successful_change_counts_as_a_scheduled_mutation() -> None:
    from caal import scheduled_events

    assert scheduled_items.CHANGE_TOOL in scheduled_events.SCHEDULED_TOOLS
    assert scheduled_events.is_scheduled_mutation(
        scheduled_items.CHANGE_TOOL, dict(status="ok")
    )


@pytest.mark.parametrize("status", ["invalid_request", "unauthorized", "error"])
def test_a_refused_change_announces_nothing(status) -> None:
    from caal import scheduled_events

    assert not scheduled_events.is_scheduled_mutation(
        scheduled_items.CHANGE_TOOL, dict(status=status)
    )


def test_the_change_tool_is_spoken_directly_rather_than_through_a_model() -> None:
    from caal.llm.scheduled_reply import spoken_outcome

    spoken = spoken_outcome(scheduled_items.CHANGE_TOOL, dict(status="ok", message="Done."))
    assert spoken == "Done."


# --- it is a registered native tool, bound to the owner of the session -------------------------


def test_the_change_tool_is_registered_and_user_scoped() -> None:
    from caal.tools.registry import create_default_registry

    tool = create_default_registry().get(scheduled_items.CHANGE_TOOL)
    assert tool.user_scoped is True
    assert tool.parameters == scheduled_items.CHANGE_SCHEMA
    assert tool.handler is scheduled_items.change_scheduled_item


def test_the_model_cannot_smuggle_an_owner_into_the_change_tool() -> None:
    from caal.tools.registry import create_default_registry
    from caal.user_scope import UserScope, scoped_tool_arguments

    tool = create_default_registry().get(scheduled_items.CHANGE_TOOL)
    bound = scoped_tool_arguments(
        tool,
        dict(action="cancel", user_id=BO, reminder_id="abc", phone="+1555"),
        UserScope(user_id=ANA, identity_configured=True),
    )
    assert bound == dict(action="cancel", user_id=ANA)


def test_nothing_in_this_module_logs_a_title_or_a_label(store, caplog) -> None:
    caplog.set_level("DEBUG")
    set_alarm(LAUNDRY, HOUR)
    scheduled_items.change_scheduled_item(
        action="convert", reference="the laundry alarm", target_kind="reminder", user_id=ANA,
        now=NOW,
    )
    written = " ".join(record.getMessage() for record in caplog.records)
    assert LAUNDRY not in written
    assert "laundry" not in written.lower()
    assert ANA not in written
