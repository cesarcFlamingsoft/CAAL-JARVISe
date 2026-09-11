"""Reminder delivery: the channels a reminder is announced on, and exactly once.

A timed reminder used to have exactly one way of reaching its owner: a live
voice session of that same owner. This is the rest of it. The owner picks any
combination of

* ``speak``     the existing active-session announcement, unchanged;
* ``telegram``  a message to the Telegram chat an administrator bound to that
  one profile, and to no other profile ever;
* ``call``      an outbound call to the number already approved on that
  profile, resolved server-side at dispatch time and never supplied by anyone.

The rules the tests below hold to:

* channels are a bounded enum. No id, number or chat ever crosses the tool
  boundary, in either direction;
* every channel is settled on its own. One failing channel neither suppresses
  nor re-sends another, and nothing is recorded as delivered until the channel
  actually accepted it;
* a channel claim is atomic and leased, so a worker that dies mid-flight hands
  the delivery back instead of losing or duplicating it;
* an undated reminder arms nothing and claims nothing;
* an anonymous session gets none of this.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from types import SimpleNamespace

import pytest

from caal.tools import alarms_tools, reminder_delivery, reminders_tools
from caal.tools.errors import SafeToolError
from caal.tools.registry import create_default_registry
from caal.user_scope import UserScope, scoped_tool_arguments

ANA = "usr_" + "a" * 24
BO = "usr_" + "b" * 24
NOW = 1_000_000
SOON = "PT30M"
DUE = NOW + 1_800

SPEAK = reminder_delivery.SPEAK
TELEGRAM = reminder_delivery.TELEGRAM
CALL = reminder_delivery.CALL


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


def _only_reminder(path, user_id) -> str:
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT id FROM reminders WHERE user_id = ?", (user_id or "",)
        ).fetchall()
    finally:
        connection.close()
    assert len(rows) == 1
    return rows[0][0]


def _armed(store, channels, user_id=ANA, title="Call the clinic"):
    reminders_tools.create_reminder(
        title=title, due=SOON, delivery=list(channels), user_id=user_id, now=NOW
    )
    return _only_reminder(store, user_id)


# --- the bounded enum -------------------------------------------------------------------


def test_channels_are_a_bounded_enum():
    assert reminder_delivery.CHANNELS == (SPEAK, TELEGRAM, CALL)
    assert reminder_delivery.parse_channels(["call", "speak"]) == (SPEAK, CALL)
    assert reminder_delivery.parse_channels("telegram") == (TELEGRAM,)
    assert reminder_delivery.parse_channels(["telegram", "telegram"]) == (TELEGRAM,)


def test_all_is_the_whole_combination_the_user_asked_for():
    """Remind me every way you can is a combination, not a fourth channel."""
    assert reminder_delivery.parse_channels("all") == (SPEAK, TELEGRAM, CALL)
    assert reminder_delivery.parse_channels(["all", "speak"]) == (SPEAK, TELEGRAM, CALL)


@pytest.mark.parametrize(
    "value",
    ["sms", ["email"], ["speak", "whatsapp"], 7, [7], ["+15551230000"], [""], dict(speak=True)],
)
def test_anything_outside_the_enum_is_refused_safely(value):
    with pytest.raises(SafeToolError):
        reminder_delivery.parse_channels(value)


def test_no_reminder_schema_lets_the_model_name_a_destination():
    """The model may pick channels. It may never pick a number, a chat or an owner."""
    registry = create_default_registry()
    forbidden = ("user_id", "chat_id", "telegram_chat_id", "phone", "number", "destination", "to")
    for name in ("reminders.create", "reminders.set_delivery"):
        schema = registry.get(name).parameters
        properties = schema["properties"]
        assert not set(properties) & set(forbidden), name
        delivery = properties["delivery"]
        assert delivery["type"] == "array"
        assert set(delivery["items"]["enum"]) == set(reminder_delivery.CHANNEL_ARGUMENTS)


# --- creating with, and without, a choice -----------------------------------------------


def test_an_explicit_combination_is_applied_without_a_question(store, everything_available):
    created = reminders_tools.create_reminder(
        title="Call the clinic", due=SOON, delivery=["call", "telegram"], user_id=ANA, now=NOW
    )

    assert created["status"] == "ok"
    assert created["data"]["delivery"] == [TELEGRAM, CALL]
    assert created["data"]["delivery_pending"] is False
    assert "?" not in created["message"]
    assert reminder_delivery.channels_of(_only_reminder(store, ANA)) == (TELEGRAM, CALL)


def test_an_unspecified_timed_reminder_asks_which_ways_and_says_when_they_happen(
    store, everything_available
):
    created = reminders_tools.create_reminder(
        title="Call the clinic", due=SOON, user_id=ANA, now=NOW
    )

    assert created["data"]["delivery_pending"] is True
    message = created["message"].lower()
    assert message.count("?") == 1
    for cue in ("say it", "telegram", "call you"):
        assert cue in message
    assert "when it comes due" in message
    assert created["data"]["delivery"] == [SPEAK]


def test_the_follow_up_answer_applies_to_the_reminder_just_created(store, everything_available):
    reminders_tools.create_reminder(title="Call the clinic", due=SOON, user_id=ANA, now=NOW)

    answered = reminders_tools.set_delivery(delivery=["all"], user_id=ANA, now=NOW)

    assert answered["status"] == "ok"
    assert answered["data"]["delivery"] == [SPEAK, TELEGRAM, CALL]
    assert reminder_delivery.channels_of(_only_reminder(store, ANA)) == (SPEAK, TELEGRAM, CALL)


def test_dropping_the_spoken_channel_cancels_its_alarm(store, everything_available):
    reminders_tools.create_reminder(title="Call the clinic", due=SOON, user_id=ANA, now=NOW)
    assert alarms_tools.pending_count(user_id=ANA, now=NOW) == 1

    reminders_tools.set_delivery(delivery=["telegram"], user_id=ANA, now=NOW)

    assert alarms_tools.pending_count(user_id=ANA, now=NOW) == 0
    assert reminder_delivery.channels_of(_only_reminder(store, ANA)) == (TELEGRAM,)


def test_set_delivery_with_nothing_to_apply_to_is_refused_not_invented(store):
    answered = reminders_tools.set_delivery(delivery=["telegram"], user_id=ANA, now=NOW)

    assert answered["status"] == "invalid_request"
    assert "reminder" in answered["message"].lower()


def test_set_delivery_never_reaches_the_reminder_of_another_user(store, everything_available):
    reminders_tools.create_reminder(title="Ana only", due=SOON, user_id=ANA, now=NOW)

    answered = reminders_tools.set_delivery(delivery=["telegram"], user_id=BO, now=NOW)

    assert answered["status"] == "invalid_request"
    assert reminder_delivery.channels_of(_only_reminder(store, ANA)) == (SPEAK,)


# --- an undated reminder promises nothing ------------------------------------------------


def test_an_undated_reminder_arms_nothing_even_when_channels_were_asked_for(
    store, everything_available
):
    created = reminders_tools.create_reminder(
        title="Buy milk", delivery=["call", "telegram"], user_id=ANA, now=NOW
    )

    assert created["data"]["timed"] is False
    assert created["data"]["delivery"] == []
    assert created["data"]["delivery_pending"] is False
    message = created["message"].lower()
    assert "no time on it" in message
    assert "?" not in message
    assert reminder_delivery.claim_due("worker-1", now=NOW + 10_000) == []


# --- authorization ------------------------------------------------------------------------


def test_telegram_is_refused_for_a_user_that_chat_was_never_bound_to(store, everything_available):
    created = reminders_tools.create_reminder(
        title="Standup", due=SOON, delivery=["telegram"], user_id=BO, now=NOW
    )

    assert created["status"] == "ok"
    assert TELEGRAM not in created["data"]["delivery"]
    assert "telegram" in created["message"].lower()
    assert reminder_delivery.channels_of(_only_reminder(store, BO)) == (SPEAK,)


def test_a_call_is_refused_for_a_profile_with_no_approved_number(store, everything_available):
    created = reminders_tools.create_reminder(
        title="Standup", due=SOON, delivery=["call"], user_id=BO, now=NOW
    )

    assert CALL not in created["data"]["delivery"]
    assert "number" in created["message"].lower()


def test_an_anonymous_session_can_neither_choose_nor_create(store):
    registry = create_default_registry()
    for name in ("reminders.create", "reminders.set_delivery"):
        tool = registry.get(name)
        assert tool.user_scoped is True
        assert scoped_tool_arguments(tool, dict(delivery=["call"]), UserScope.anonymous()) is None


def test_a_forged_owner_never_survives_the_tool_boundary(store, everything_available):
    registry = create_default_registry()
    tool = registry.get("reminders.create")
    arguments = dict(title="Standup", due=SOON, delivery=["telegram"], user_id=BO)

    bound = scoped_tool_arguments(tool, arguments, UserScope.for_user(SimpleNamespace(user_id=ANA)))

    assert bound is not None and bound["user_id"] == ANA
    result = tool.handler(**bound, now=NOW)
    assert result["status"] == "ok"
    assert reminders_tools.list_reminders(user_id=BO)["data"]["reminders"] == []


# --- the owner-scoped default for future reminders ---------------------------------------


def test_the_default_for_future_reminders_is_owned_and_migration_safe(store, everything_available):
    assert reminder_delivery.default_channels(ANA) == reminder_delivery.DEFAULT_CHANNELS

    reminder_delivery.set_default_channels(ANA, [TELEGRAM, SPEAK], now=NOW)

    assert reminder_delivery.default_channels(ANA) == (SPEAK, TELEGRAM)
    assert reminder_delivery.default_channels(BO) == reminder_delivery.DEFAULT_CHANNELS


def test_a_saved_default_is_applied_instead_of_asking(store, everything_available):
    reminder_delivery.set_default_channels(ANA, [SPEAK, TELEGRAM], now=NOW)

    created = reminders_tools.create_reminder(title="Standup", due=SOON, user_id=ANA, now=NOW)

    assert created["data"]["delivery"] == [SPEAK, TELEGRAM]
    assert created["data"]["delivery_pending"] is False
    assert "?" not in created["message"]


def test_a_default_a_user_cannot_use_is_not_saved(store, everything_available):
    """An owner may only default to a channel that is actually authorized for them."""
    with pytest.raises(SafeToolError):
        reminder_delivery.set_default_channels(BO, [TELEGRAM], now=NOW)


def test_the_delivery_tables_are_created_beside_an_older_reminders_database(store):
    """A database written before delivery existed keeps its rows and gains the columns."""
    connection = sqlite3.connect(store)
    connection.execute(
        "CREATE TABLE reminders (id TEXT PRIMARY KEY, user_id TEXT NOT NULL DEFAULT '', "
        "title TEXT NOT NULL, due TEXT, due_at INTEGER, list_name TEXT NOT NULL, "
        "notes TEXT NOT NULL, completed INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO reminders VALUES ('old','','Water the plants',NULL,NULL,'Reminders','',0,'x')"
    )
    connection.commit()
    connection.close()

    listed = reminders_tools.list_reminders(user_id=None)

    assert [row["title"] for row in listed["data"]["reminders"]] == ["Water the plants"]
    assert listed["data"]["reminders"][0]["delivery"] == []
    assert len(reminders_tools.list_reminders(user_id=None)["data"]["reminders"]) == 1


# --- delivery: exactly once, per channel, independently ----------------------------------


def test_an_outside_channel_is_claimed_by_exactly_one_worker(store, everything_available):
    _armed(store, [TELEGRAM, CALL])

    first = reminder_delivery.claim_due("worker-1", now=DUE)
    second = reminder_delivery.claim_due("worker-2", now=DUE)

    assert sorted(item.channel for item in first) == [CALL, TELEGRAM]
    assert second == []


def test_the_spoken_channel_is_not_claimed_by_the_worker(store, everything_available):
    """Speech belongs to a live session; the worker must never settle it."""
    _armed(store, [SPEAK, TELEGRAM])

    claimed = reminder_delivery.claim_due("worker-1", now=DUE)

    assert [item.channel for item in claimed] == [TELEGRAM]


def test_one_channel_failing_neither_suppresses_nor_re_sends_the_other(store, everything_available):
    _armed(store, [TELEGRAM, CALL])
    claimed = dict((item.channel, item) for item in reminder_delivery.claim_due("worker-1", now=DUE))

    reminder_delivery.mark_delivered(claimed[TELEGRAM].delivery_id, "worker-1", now=DUE)
    reminder_delivery.release(claimed[CALL].delivery_id, "worker-1", delay_seconds=0, now=DUE)

    again = reminder_delivery.claim_due("worker-1", now=DUE + 1)
    assert [item.channel for item in again] == [CALL]
    assert again[0].attempts == 2


def test_a_spent_retry_budget_leaves_a_truthful_failed_state(store, everything_available):
    reminder_id = _armed(store, [CALL])
    at = DUE
    for _ in range(reminder_delivery.MAX_ATTEMPTS):
        claimed = reminder_delivery.claim_due("worker-1", now=at)
        assert len(claimed) == 1
        reminder_delivery.release(claimed[0].delivery_id, "worker-1", delay_seconds=0, now=at)
        at += 1

    assert reminder_delivery.claim_due("worker-1", now=at + 10_000) == []
    states = reminder_delivery.states_of([reminder_id])[reminder_id]
    assert states[CALL] == reminder_delivery.FAILED


def test_a_worker_that_dies_mid_flight_hands_the_delivery_back(store, everything_available):
    """Restart recovery: a lease that expires is re-claimable, a delivery is not."""
    _armed(store, [TELEGRAM, CALL])
    claimed = dict((item.channel, item) for item in reminder_delivery.claim_due("worker-1", now=DUE))
    reminder_delivery.mark_delivered(claimed[TELEGRAM].delivery_id, "worker-1", now=DUE)

    still_leased = reminder_delivery.claim_due("worker-2", now=DUE + 5)
    recovered = reminder_delivery.claim_due(
        "worker-2", now=DUE + reminder_delivery.CLAIM_LEASE_SECONDS + 1
    )

    assert still_leased == []
    assert [item.channel for item in recovered] == [CALL]


def test_a_settled_delivery_is_never_claimed_again(store, everything_available):
    _armed(store, [TELEGRAM])
    claimed = reminder_delivery.claim_due("worker-1", now=DUE)
    reminder_delivery.mark_delivered(claimed[0].delivery_id, "worker-1", now=DUE)

    assert reminder_delivery.claim_due("worker-1", now=DUE + 100_000) == []
    assert reminder_delivery.mark_delivered(claimed[0].delivery_id, "worker-2", now=DUE) is False


def test_nothing_is_claimed_before_it_is_due(store, everything_available):
    _armed(store, [TELEGRAM])

    assert reminder_delivery.claim_due("worker-1", now=DUE - 1) == []


def test_the_spoken_channel_settles_only_when_the_session_actually_spoke(store):
    """The existing active-session path stays the owner of the spoken channel."""
    import asyncio

    from caal import alarm_delivery

    reminder_id = _armed(store, [SPEAK])

    class _Session:
        def __init__(self) -> None:
            self.said: list[str] = []

        async def say(self, text: str) -> None:
            self.said.append(text)

    scope = UserScope.for_user(SimpleNamespace(user_id=ANA))
    session = _Session()

    spoken = asyncio.run(alarm_delivery.announce_due_alarms(session, scope, now=DUE))

    assert spoken == 1
    assert session.said == ["Reminder: Call the clinic."]
    assert reminder_delivery.states_of([reminder_id])[reminder_id][SPEAK] == (
        reminder_delivery.DELIVERED
    )
    assert asyncio.run(alarm_delivery.announce_due_alarms(_Session(), scope, now=DUE)) == 0


def test_an_unspoken_alarm_leaves_the_spoken_channel_pending(store):
    import asyncio

    from caal import alarm_delivery

    reminder_id = _armed(store, [SPEAK])

    class _MuteSession:
        async def say(self, text: str) -> None:
            raise RuntimeError("the room went away")

    with pytest.raises(RuntimeError):
        asyncio.run(alarm_delivery.announce_due_alarms(_MuteSession(), UserScope.for_user(
            SimpleNamespace(user_id=ANA)), now=DUE))

    assert reminder_delivery.states_of([reminder_id])[reminder_id][SPEAK] == (
        reminder_delivery.PENDING
    )
    assert alarms_tools.claim_due_alarms(now=DUE, user_id=ANA) != []


# --- privacy -------------------------------------------------------------------------------


def test_nothing_private_reaches_the_log(store, everything_available, caplog):
    caplog.set_level(logging.DEBUG)
    reminder_id = _armed(store, [TELEGRAM, CALL], title="Divorce lawyer at four")
    claimed = [i for i in reminder_delivery.claim_due("worker-1", now=DUE) if i.channel == TELEGRAM]
    reminder_delivery.mark_delivered(claimed[0].delivery_id, "worker-1", now=DUE)
    reminder_delivery.set_default_channels(ANA, [SPEAK], now=NOW)

    text = "\n".join(record.getMessage() for record in caplog.records)
    for secret in ("Divorce", "lawyer", ANA, BO, reminder_id, claimed[0].delivery_id, "+1555"):
        assert secret not in text


def test_a_tool_result_carries_no_id_and_no_owner(store, everything_available):
    created = reminders_tools.create_reminder(
        title="Standup", due=SOON, delivery=["telegram"], user_id=ANA, now=NOW
    )
    listed = reminders_tools.list_reminders(user_id=ANA)

    for payload in (created, listed):
        rendered = repr(payload)
        assert ANA not in rendered
        assert "user_id" not in rendered
        assert "id" not in payload["data"]
    assert set(listed["data"]["reminders"][0]["delivery"]) == set([TELEGRAM])


def test_the_delivery_choice_tool_call_reaches_the_handler(store, everything_available):
    """A whole tool call as the local model emits it: registry, scoping, handler."""
    reminders_tools.create_reminder(title="Call the clinic", due=SOON, user_id=ANA, now=NOW)
    registry = create_default_registry()
    tool = registry.get("reminders.set_delivery")

    bound = scoped_tool_arguments(
        tool,
        dict(delivery=["call", "telegram"], user_id=BO, chat_id="1234"),
        UserScope.for_user(SimpleNamespace(user_id=ANA)),
    )

    assert bound is not None and bound["user_id"] == ANA and "chat_id" not in bound
    result = tool.handler(**bound, now=NOW)
    assert result["status"] == "ok"
    assert reminder_delivery.channels_of(_only_reminder(store, ANA)) == (TELEGRAM, CALL)


def test_the_reminder_tools_keep_their_arguments_out_of_the_llm_log():
    from caal.llm.llm_node import _keeps_contents_private

    for name in ("reminders.create", "reminders.list", "reminders.set_delivery", "alarms.set"):
        assert _keeps_contents_private(name) is True


# --- what was already there when this ledger arrived --------------------------------------


def _legacy_rows(path, due_at, delivered=False):
    """A timed reminder and its alarm, exactly as the pre-delivery code wrote them."""
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS reminders (id TEXT PRIMARY KEY, user_id TEXT NOT NULL "
        "DEFAULT '', title TEXT NOT NULL, due TEXT, due_at INTEGER, list_name TEXT NOT NULL, "
        "notes TEXT NOT NULL, completed INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS alarms (id TEXT PRIMARY KEY, user_id TEXT NOT NULL "
        "DEFAULT '', label TEXT NOT NULL, kind TEXT NOT NULL, due_at INTEGER NOT NULL, "
        "fired_at INTEGER, delivered_at INTEGER, created_at INTEGER NOT NULL)"
    )
    connection.execute(
        "INSERT INTO reminders VALUES ('old-1',?,'Call the clinic','x',?,'Reminders','',0,'t')",
        (ANA, due_at),
    )
    connection.execute(
        "INSERT INTO alarms VALUES ('alarm-1',?,'Call the clinic','reminder',?,NULL,?,1)",
        (ANA, due_at, NOW if delivered else None),
    )
    connection.commit()
    connection.close()


def test_a_reminder_from_before_this_ledger_keeps_the_channel_it_already_had(store):
    """It was always announced by a live session; the dashboard says exactly that."""
    _legacy_rows(store, DUE)

    assert reminder_delivery.channels_of("old-1") == (SPEAK,)
    assert reminder_delivery.states_of(["old-1"])["old-1"][SPEAK] == reminder_delivery.PENDING
    # And it is settled by the session that says it, like any other spoken channel.
    assert reminder_delivery.settle_speak(["alarm-1"], now=DUE) == 1
    assert reminder_delivery.states_of(["old-1"])["old-1"][SPEAK] == reminder_delivery.DELIVERED


def test_a_reminder_whose_alarm_was_already_delivered_adopts_nothing(store):
    """Never promise an announcement that has already happened or cannot happen."""
    _legacy_rows(store, DUE, delivered=True)

    assert reminder_delivery.channels_of("old-1") == ()
    assert reminder_delivery.claim_due("worker-1", now=DUE + 10_000) == []


def test_adoption_happens_once_and_never_arms_an_outside_channel(store):
    _legacy_rows(store, DUE)
    reminder_delivery.channels_of("old-1")
    reminder_delivery.channels_of("old-1")

    assert reminder_delivery.channels_of("old-1") == (SPEAK,)
    assert reminder_delivery.claim_due("worker-1", now=DUE + 10_000) == []


def test_adoption_runs_once_even_for_a_ledger_that_already_existed(store):
    """A deployment that read the ledger before adoption existed still adopts."""
    reminder_delivery.channels_of("nothing-here")  # creates the tables, adopts nothing
    _legacy_rows(store, DUE)
    import sqlite3 as _sqlite3

    connection = _sqlite3.connect(store)
    connection.execute("DELETE FROM reminder_delivery_meta")
    connection.commit()
    connection.close()

    assert reminder_delivery.channels_of("old-1") == (SPEAK,)


# --- the owner binding has to mean the same thing in every process -------------------------


def _deployment_settings(tmp_path, monkeypatch, owner):
    """A deployment whose settings.json sits beside its data directory.

    This is the real layout: the agent and the durable worker mount the same
    settings.json and the same data directory, but they import caal from
    different roots, so the module-relative settings path resolves in one
    process and not in the other. The binding may not depend on which process
    is asking: the dashboard would offer Telegram and the worker would refuse
    it, and the owner would be told about neither.
    """
    from caal import settings as settings_module

    data = tmp_path / "data"
    data.mkdir()
    (tmp_path / "settings.json").write_text(
        json.dumps(dict(telegram_owner_user_id=owner, telegram_bot_token="t", telegram_chat_id="c"))
    )
    monkeypatch.setenv("CAAL_DATA_DIR", str(data))
    monkeypatch.delenv("CAAL_TELEGRAM_OWNER_USER_ID", raising=False)
    # The path this process would resolve on its own does not exist, exactly as
    # in the container where caal is imported from site-packages.
    monkeypatch.setattr(settings_module, "SETTINGS_PATH", tmp_path / "nowhere" / "settings.json")
    monkeypatch.setattr(settings_module, "_settings_cache", None)
    monkeypatch.delenv("CAAL_SETTINGS_PATH", raising=False)


def test_the_telegram_binding_is_found_however_the_process_imported_caal(tmp_path, monkeypatch):
    _deployment_settings(tmp_path, monkeypatch, ANA)

    assert reminder_delivery.telegram_owner() == ANA
    assert reminder_delivery.telegram_configured() is True


def test_a_profile_that_is_not_the_bound_owner_is_still_refused(store, tmp_path, monkeypatch):
    _deployment_settings(tmp_path, monkeypatch, ANA)
    monkeypatch.setattr(reminder_delivery, "STORE_PATH", store)
    monkeypatch.setattr(reminder_delivery, "resolve_callback_number", lambda user_id: None)

    assert reminder_delivery.channel_availability(ANA)[TELEGRAM] is True
    assert reminder_delivery.channel_availability(BO)[TELEGRAM] is False


def test_an_operator_can_bind_the_chat_from_the_environment(tmp_path, monkeypatch):
    """One place both processes always agree on, for a deployment that wants it."""
    _deployment_settings(tmp_path, monkeypatch, ANA)
    monkeypatch.setenv("CAAL_TELEGRAM_OWNER_USER_ID", BO)

    assert reminder_delivery.telegram_owner() == BO


def test_no_binding_anywhere_means_no_signed_in_user_reaches_that_chat(tmp_path, monkeypatch):
    _deployment_settings(tmp_path, monkeypatch, "")

    assert reminder_delivery.telegram_owner() == ""
    assert reminder_delivery.channel_availability(ANA)[TELEGRAM] is False
    # A legacy single-user deployment keeps the operator chat it always had.
    assert reminder_delivery.channel_availability(None)[TELEGRAM] is True
