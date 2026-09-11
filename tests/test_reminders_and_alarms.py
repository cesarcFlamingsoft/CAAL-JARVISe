"""Local alarms, timers and reminders: parsing, ownership and delivery."""

from __future__ import annotations

import asyncio
import importlib
import logging
import sqlite3
from types import SimpleNamespace

import pytest

from caal.tools import alarms_tools, reminder_delivery, reminders_tools
from caal.tools.errors import SafeToolError
from caal.tools.registry import create_default_registry
from caal.tools.scheduled_time import parse_when
from caal.user_scope import UserScope, scoped_tool_arguments

ANA = "usr_" + "a" * 24
BO = "usr_" + "b" * 24
NOW = 1_000_000


@pytest.fixture
def store(monkeypatch, tmp_path):
    """One SQLite file shared by both stores, as in the running deployment."""
    path = tmp_path / "assistant.sqlite3"
    monkeypatch.setattr(alarms_tools, "STORE_PATH", path)
    monkeypatch.setattr(reminders_tools, "STORE_PATH", path)
    monkeypatch.setattr(reminder_delivery, "STORE_PATH", path)
    return path


# --- the production failure -------------------------------------------------------------


def test_iso_duration_pt2m_sets_an_alarm(store):
    """The exact call the local model made in production, which used to be refused."""
    created = alarms_tools.set_alarm(
        label="Take the bread out", when="PT2M", kind="alarm", now=NOW, user_id=ANA
    )

    assert created["status"] == "ok"
    assert created["data"]["due_at"] == NOW + 120
    assert "in 2 minutes" in created["message"]
    assert alarms_tools.claim_due_alarms(now=NOW + 119, user_id=ANA) == []
    assert [row["label"] for row in alarms_tools.claim_due_alarms(now=NOW + 120, user_id=ANA)] == [
        "Take the bread out"
    ]


def test_the_model_tool_call_payload_reaches_the_handler(store):
    """A whole tool call as the local model emits it: registry, scoping, handler."""
    registry = create_default_registry()
    tool = registry.get("alarms.set")
    arguments = dict(kind="alarm", label="Take the bread out", when="PT2M", user_id=BO)

    bound = scoped_tool_arguments(tool, arguments, UserScope.for_user(SimpleNamespace(user_id=ANA)))

    assert bound is not None and bound["user_id"] == ANA  # never the invented BO
    result = tool.handler(**bound, now=NOW)
    assert result["status"] == "ok"
    assert alarms_tools.claim_due_alarms(now=NOW + 120, user_id=BO) == []


@pytest.mark.parametrize(
    "when,seconds",
    [
        ("PT2M", 120),
        ("PT30S", 30),
        ("PT1H30M", 5_400),
        ("P1D", 86_400),
        ("pt45s", 45),
        ("P1DT2H", 93_600),
        ("P1W", 604_800),
        ("10m", 600),
        ("2 hours", 7_200),
        ("45 minutes", 2_700),
        ("1 day", 86_400),
    ],
)
def test_accepted_duration_forms(when, seconds):
    assert parse_when(when, NOW) == NOW + seconds


def test_a_timezone_aware_timestamp_is_accepted():
    assert parse_when("2026-09-09T18:30:00-06:00", 1_788_976_800) == 1_789_000_200
    assert parse_when("2026-09-09T18:30:00Z", 1_788_976_800) == 1_788_978_600


@pytest.mark.parametrize(
    "when",
    [
        "PT0S",
        "P0D",
        "-PT2M",
        "PT",
        "P",
        "banana",
        "in a bit",
        "2026-09-09T18:30:00",
        "1999-01-01T00:00:00Z",
        "P9999D",
        "PT99999999999H",
        "9" * 400,
        "",
        None,
        17,
    ],
)
def test_refused_time_forms_are_refused_safely(when):
    with pytest.raises(SafeToolError) as refusal:
        parse_when(when, 1_788_976_800)
    message = str(refusal.value)
    assert message and message[0].isupper() and "Traceback" not in message


def test_a_refused_alarm_answers_with_a_speakable_message(store):
    refused = alarms_tools.set_alarm(
        label="Nap", when="tomorrow-ish", kind="alarm", now=NOW, user_id=ANA
    )

    assert refused["status"] == "invalid_time"
    assert "PT2M" in refused["message"]
    assert refused["data"] == dict()
    assert alarms_tools.pending_count(user_id=ANA, now=NOW) == 0


def test_an_unknown_kind_is_refused_without_storing(store):
    refused = alarms_tools.set_alarm(
        label="Nap", when="PT2M", kind="countdown", now=NOW, user_id=ANA
    )

    assert refused["status"] == "invalid_time"
    assert alarms_tools.pending_count(user_id=ANA, now=NOW) == 0


def test_an_empty_label_is_refused(store):
    refused = alarms_tools.set_alarm(label="   ", when="PT2M", kind="timer", now=NOW, user_id=ANA)

    assert refused["status"] == "invalid_time"
    assert alarms_tools.pending_count(user_id=ANA, now=NOW) == 0


# --- reminders: timed and undated -------------------------------------------------------


def test_an_undated_reminder_is_a_list_item_and_never_claims_an_alert(store):
    created = reminders_tools.create_reminder(
        title="Buy stamps", list_name="Errands", notes="the small ones", user_id=ANA, now=NOW
    )

    assert created["status"] == "ok"
    assert created["data"]["timed"] is False
    assert "will not alert you" in created["message"]
    assert alarms_tools.claim_due_alarms(now=NOW + 10**6, user_id=ANA) == []

    listed = reminders_tools.list_reminders(user_id=ANA)
    assert [item["title"] for item in listed["data"]["reminders"]] == ["Buy stamps"]
    assert listed["data"]["reminders"][0]["timed"] is False


def test_a_timed_reminder_is_scheduled_and_announced_once(store):
    created = reminders_tools.create_reminder(
        title="Call the dentist", due="PT2M", user_id=ANA, now=NOW
    )

    assert created["data"]["timed"] is True
    assert "in 2 minutes" in created["message"]

    due = alarms_tools.claim_due_alarms(now=NOW + 120, user_id=ANA)
    assert [row["kind"] for row in due] == ["reminder"]
    from caal import alarm_delivery

    assert alarm_delivery.alarm_message(due[0]) == "Reminder: Call the dentist."


def test_a_reminder_with_an_unreadable_due_is_refused_rather_than_silently_undated(store):
    refused = reminders_tools.create_reminder(title="Call the dentist", due="soon", user_id=ANA)

    assert refused["status"] == "invalid_request"
    assert reminders_tools.list_reminders(user_id=ANA)["data"]["reminders"] == []


# --- ownership --------------------------------------------------------------------------


def test_alarms_of_one_user_are_invisible_to_another(store):
    alarms_tools.set_alarm(label="Ana laundry", when="PT1M", kind="alarm", now=NOW, user_id=ANA)
    alarms_tools.set_alarm(label="Bo standup", when="PT1M", kind="alarm", now=NOW, user_id=BO)

    ana_due = alarms_tools.claim_due_alarms(now=NOW + 60, user_id=ANA)
    bo_due = alarms_tools.claim_due_alarms(now=NOW + 60, user_id=BO)

    assert [row["label"] for row in ana_due] == ["Ana laundry"]
    assert [row["label"] for row in bo_due] == ["Bo standup"]


def test_reminders_of_one_user_are_invisible_to_another(store):
    reminders_tools.create_reminder(title="Ana errand", user_id=ANA, now=NOW)
    reminders_tools.create_reminder(title="Bo errand", user_id=BO, now=NOW)

    assert [
        item["title"] for item in reminders_tools.list_reminders(user_id=BO)["data"]["reminders"]
    ] == ["Bo errand"]


def test_a_forged_user_id_shape_is_rejected(store):
    with pytest.raises(ValueError):
        alarms_tools.set_alarm(
            label="x", when="PT1M", kind="alarm", now=NOW, user_id="usr_../../etc"
        )


def test_an_anonymous_session_is_refused_before_any_store_opens():
    registry = create_default_registry()
    for name in ("alarms.set", "reminders.create", "reminders.list"):
        tool = registry.get(name)
        assert tool.user_scoped is True
        assert scoped_tool_arguments(tool, dict(label="x"), UserScope.anonymous()) is None


def test_a_legacy_deployment_keeps_its_own_unowned_scope(store):
    created = alarms_tools.set_alarm(label="Tea", when="PT1M", kind="timer", now=NOW)

    assert created["status"] == "ok"
    assert [row["label"] for row in alarms_tools.claim_due_alarms(now=NOW + 60)] == ["Tea"]


# --- migration of the pre-ownership tables ----------------------------------------------


def _write_unowned_tables(path):
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE alarms (id TEXT PRIMARY KEY, label TEXT NOT NULL, kind TEXT NOT NULL, "
        "due_at INTEGER NOT NULL, fired_at INTEGER, created_at INTEGER NOT NULL)"
    )
    connection.execute(
        "INSERT INTO alarms VALUES ('old-1','Old alarm','alarm',?,NULL,?)", (NOW + 30, NOW)
    )
    connection.execute(
        "CREATE TABLE reminders (id TEXT PRIMARY KEY, title TEXT NOT NULL, due TEXT, "
        "list_name TEXT NOT NULL, notes TEXT NOT NULL, completed INTEGER NOT NULL DEFAULT 0, "
        "created_at TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO reminders VALUES ('old-2','Old reminder',NULL,'Reminders','',0,'2026-01-01')"
    )
    connection.commit()
    connection.close()


def test_unowned_rows_survive_migration_and_never_reach_a_signed_in_user(store):
    _write_unowned_tables(store)

    assert alarms_tools.claim_due_alarms(now=NOW + 60, user_id=ANA) == []
    assert reminders_tools.list_reminders(user_id=ANA)["data"]["reminders"] == []
    # Kept, not dropped: the legacy scope still holds them.
    assert [row["label"] for row in alarms_tools.claim_due_alarms(now=NOW + 60)] == ["Old alarm"]
    assert [item["title"] for item in reminders_tools.list_reminders()["data"]["reminders"]] == [
        "Old reminder"
    ]


def test_migration_is_idempotent_and_keeps_new_rows(store):
    _write_unowned_tables(store)
    alarms_tools.set_alarm(label="New", when="PT1M", kind="alarm", now=NOW, user_id=ANA)

    assert alarms_tools.pending_count(user_id=ANA, now=NOW) == 1
    assert [row["label"] for row in alarms_tools.claim_due_alarms(now=NOW + 60, user_id=ANA)] == [
        "New"
    ]


# --- delivery ---------------------------------------------------------------------------


class _Session:
    def __init__(self, fail_at=None):
        self.messages = []
        self._fail_at = fail_at

    async def say(self, message):
        if self._fail_at is not None and len(self.messages) == self._fail_at:
            raise RuntimeError("the room went away")
        self.messages.append(message)


def _scope_for(user_id):
    return UserScope.for_user(SimpleNamespace(user_id=user_id))


def test_a_due_alarm_is_spoken_exactly_once_across_sessions(store):
    from caal import alarm_delivery

    alarms_tools.set_alarm(label="Tea", when="PT1M", kind="timer", now=NOW, user_id=ANA)

    first = _Session()
    assert asyncio.run(alarm_delivery.announce_due_alarms(first, _scope_for(ANA))) == 1
    assert first.messages == ["Your timer is finished: Tea."]

    second = _Session()
    assert asyncio.run(alarm_delivery.announce_due_alarms(second, _scope_for(ANA))) == 0
    assert second.messages == []


def test_a_session_never_announces_the_alarm_of_another_user(store):
    from caal import alarm_delivery

    alarms_tools.set_alarm(label="Ana alarm", when="PT1M", kind="alarm", now=NOW, user_id=ANA)

    bo_session = _Session()
    assert asyncio.run(alarm_delivery.announce_due_alarms(bo_session, _scope_for(BO))) == 0
    assert bo_session.messages == []
    # And it is still there for the user who set it.
    ana_session = _Session()
    assert asyncio.run(alarm_delivery.announce_due_alarms(ana_session, _scope_for(ANA))) == 1


def test_an_anonymous_session_delivers_nothing_and_preserves_the_alarm(store):
    from caal import alarm_delivery

    alarms_tools.set_alarm(label="Ana alarm", when="PT1M", kind="alarm", now=NOW, user_id=ANA)
    session = _Session()

    assert alarm_delivery.may_deliver(UserScope.anonymous()) is False
    assert asyncio.run(alarm_delivery.announce_due_alarms(session, UserScope.anonymous())) == 0
    assert session.messages == []
    assert asyncio.run(alarm_delivery.announce_due_alarms(_Session(), _scope_for(ANA))) == 1


def test_an_alarm_that_cannot_be_spoken_is_preserved_for_the_next_session(store):
    from caal import alarm_delivery

    alarms_tools.set_alarm(label="First", when="PT1M", kind="alarm", now=NOW, user_id=ANA)
    alarms_tools.set_alarm(label="Second", when="PT2M", kind="alarm", now=NOW, user_id=ANA)
    broken = _Session(fail_at=1)

    with pytest.raises(RuntimeError):
        asyncio.run(alarm_delivery.announce_due_alarms(broken, _scope_for(ANA)))
    assert broken.messages == ["Alarm: First."]

    recovered = _Session()
    assert asyncio.run(alarm_delivery.announce_due_alarms(recovered, _scope_for(ANA))) == 1
    assert recovered.messages == ["Alarm: Second."]


def test_announcements_name_the_label_only(store):
    from caal import alarm_delivery

    assert alarm_delivery.alarm_message(dict(kind="timer", label="Tea", id="row-1")) == (
        "Your timer is finished: Tea."
    )
    assert (
        alarm_delivery.alarm_message(dict(kind="alarm", label="Tea", id="row-1")) == "Alarm: Tea."
    )
    for kind in ("alarm", "timer", "reminder"):
        assert "row-1" not in alarm_delivery.alarm_message(dict(kind=kind, label="x", id="row-1"))


# --- what the model is told, and what the log keeps --------------------------------------


def test_the_tool_schemas_teach_the_accepted_time_forms():
    registry = create_default_registry()
    when = registry.get("alarms.set").parameters["properties"]["when"]["description"]
    due = registry.get("reminders.create").parameters["properties"]["due"]["description"]

    for description in (when, due):
        assert "PT2M" in description
        assert "10m" in description
        assert "timezone offset" in description


def test_no_schema_lets_the_model_choose_whose_alarm_it_is():
    registry = create_default_registry()
    for name in ("alarms.set", "reminders.create", "reminders.list"):
        properties = registry.get(name).parameters["properties"]
        assert "user_id" not in properties
        assert registry.get(name).parameters["additionalProperties"] is False


def test_the_reminder_tool_does_not_claim_an_integration_it_does_not_have():
    description = create_default_registry().get("reminders.create").description

    assert "does not write to Apple Reminders" in description


def test_creating_an_alarm_needs_no_confirmation_but_reports_only_stored_state(store):
    registry = create_default_registry()

    assert registry.get("alarms.set").requires_confirmation is False
    result = alarms_tools.set_alarm(label="Tea", when="PT1M", kind="timer", now=NOW, user_id=ANA)
    assert result["status"] == "ok"
    assert alarms_tools.pending_count(user_id=ANA, now=NOW) == 1


def test_nothing_private_reaches_the_log(store, caplog):
    with caplog.at_level(logging.DEBUG):
        alarms_tools.set_alarm(
            label="Divorce lawyer", when="PT2M", kind="alarm", now=NOW, user_id=ANA
        )
        reminders_tools.create_reminder(
            title="Biopsy results", notes="oncology", user_id=ANA, now=NOW
        )
        alarms_tools.set_alarm(label="x", when="nonsense", kind="alarm", now=NOW, user_id=ANA)

    text = caplog.text
    for secret in ("Divorce lawyer", "Biopsy results", "oncology", ANA):
        assert secret not in text


def test_a_result_carries_no_row_id_or_owner(store):
    created = alarms_tools.set_alarm(label="Tea", when="PT1M", kind="timer", now=NOW, user_id=ANA)
    reminder = reminders_tools.create_reminder(title="Buy stamps", user_id=ANA, now=NOW)

    assert "id" not in created["data"] and "user_id" not in created["data"]
    assert "id" not in reminder["data"] and "user_id" not in reminder["data"]


def test_the_scheduling_tools_keep_their_arguments_out_of_the_llm_log():
    from caal.llm.llm_node import _keeps_contents_private

    for name in ("alarms.set", "reminders.create", "reminders.list"):
        assert _keeps_contents_private(name) is True


@pytest.mark.asyncio
async def test_an_anonymous_session_is_refused_by_the_llm_dispatch(monkeypatch, store):
    llm_node_module = importlib.import_module("caal.llm.llm_node")
    from caal.user_scope import SCHEDULING_UNAVAILABLE_REPLY

    monkeypatch.setattr(
        llm_node_module.settings_module, "get_setting", lambda key, default=None: True
    )

    class _Agent:
        _user_scope = UserScope.anonymous()

    refusal = await llm_node_module._execute_single_tool(
        _Agent(), "alarms.set", dict(label="Tea", when="PT2M", kind="timer")
    )

    assert refusal["message"] == SCHEDULING_UNAVAILABLE_REPLY
    assert refusal["status"] == "unauthorized"
    assert alarms_tools.pending_count(now=NOW) == 0


@pytest.mark.asyncio
async def test_a_tool_failure_never_hands_the_model_a_traceback(monkeypatch):
    llm_node_module = importlib.import_module("caal.llm.llm_node")

    async def _boom(agent, tool_name, arguments):
        raise RuntimeError("sqlite3 OperationalError at /app/data/assistant.sqlite3")

    monkeypatch.setattr(llm_node_module, "_execute_single_tool", _boom)

    class _Provider:
        def format_tool_call_message(self, content, tool_calls):
            return dict(role="assistant")

        def format_tool_result(self, content, tool_call_id, tool_name):
            return dict(role="tool", content=content)

    messages, _ = await llm_node_module._execute_tool_calls(
        object(),
        [],
        [SimpleNamespace(name="alarms.set", arguments=dict(label="Tea"), id="1")],
        None,
        _Provider(),
    )

    reported = messages[-1]["content"]
    assert "sqlite3" not in reported and "Traceback" not in reported
    assert "did not go through" in reported
