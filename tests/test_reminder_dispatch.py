"""The durable worker half of reminder delivery: Telegram and the callback number.

Speech belongs to a live room. These two do not, so they are carried out by
the same supervised process that dispatches background-task callbacks, and
they survive the room that created them closing.

Everything in here re-decides authorization at delivery time, from the stored
owner and nothing else:

* Telegram is only ever sent to the one chat an administrator bound to that
  one profile. A reminder of any other owner is refused outright rather than
  delivered to the shared operator chat;
* a call resolves the owner own approved number at dispatch time. No number is
  stored on a reminder, no caller supplies one, and a profile without an
  approved number is refused, not guessed.

Nothing is recorded as delivered before the channel accepted it, every channel
is settled on its own, and no log line here carries a title, an id or a number.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from caal.tools import alarms_tools, reminder_delivery, reminders_tools

ANA = "usr_" + "a" * 24
BO = "usr_" + "b" * 24
NOW = 1_000_000
DUE = NOW + 1_800
NUMBER = "+15551230000"


@pytest.fixture
def store(monkeypatch, tmp_path):
    path = tmp_path / "assistant.sqlite3"
    monkeypatch.setattr(alarms_tools, "STORE_PATH", path)
    monkeypatch.setattr(reminders_tools, "STORE_PATH", path)
    monkeypatch.setattr(reminder_delivery, "STORE_PATH", path)
    monkeypatch.setattr(reminder_delivery, "telegram_owner", lambda: ANA)
    monkeypatch.setattr(reminder_delivery, "telegram_configured", lambda: True)
    monkeypatch.setattr(
        reminder_delivery,
        "resolve_callback_number",
        lambda user_id: NUMBER if user_id == ANA else None,
    )
    return path


class FakeTelegram:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.error: Exception | None = None

    async def __call__(self, text: str) -> None:
        if self.error is not None:
            raise self.error
        self.sent.append(text)


class FakePlacer:
    def __init__(self) -> None:
        self.reserved: list[object] = []
        self.dispatched: list[tuple[str, object]] = []
        self.reserve_error: Exception | None = None
        self.dispatch_error: Exception | None = None

    async def reserve(self, request) -> str:
        if self.reserve_error is not None:
            raise self.reserve_error
        self.reserved.append(request)
        return "caal-outbound-" + request.attempt_id

    async def dispatch(self, room_name: str, request) -> None:
        if self.dispatch_error is not None:
            raise self.dispatch_error
        self.dispatched.append((room_name, request))


def _dispatcher(telegram=None, placer=None, dry_run=False, claimant="worker-1"):
    from caal.reminder_dispatch import ReminderDispatcher

    return ReminderDispatcher(
        claimant=claimant,
        send_telegram=telegram,
        placer=placer,
        resolve_user_destination=reminder_delivery.resolve_callback_number,
        dry_run=dry_run,
    )


def _reminder(channels, user_id=ANA, title="Call the clinic"):
    reminders_tools.create_reminder(
        title=title, due="PT30M", delivery=list(channels), user_id=user_id, now=NOW
    )


def _state_of(store, channel, user_id=ANA):
    import sqlite3

    connection = sqlite3.connect(store)
    try:
        row = connection.execute(
            "SELECT state FROM reminder_deliveries WHERE user_id = ? AND channel = ?",
            (user_id, channel),
        ).fetchone()
    finally:
        connection.close()
    return None if row is None else row[0]


# --- telegram -------------------------------------------------------------------------------


def test_a_telegram_reminder_is_sent_once_and_settled(store):
    _reminder([reminder_delivery.TELEGRAM])
    telegram = FakeTelegram()
    dispatcher = _dispatcher(telegram=telegram)

    first = asyncio.run(dispatcher.deliver_due(now=DUE))
    second = asyncio.run(dispatcher.deliver_due(now=DUE + 1))

    assert telegram.sent == ["Reminder: Call the clinic."]
    assert first.get(reminder_delivery.DELIVERED) == 1
    assert second == {}
    assert _state_of(store, reminder_delivery.TELEGRAM) == reminder_delivery.DELIVERED


def test_telegram_is_refused_for_a_reminder_of_another_profile(store, monkeypatch):
    """The bound chat belongs to one profile; nobody else reaches it, ever."""
    _reminder([reminder_delivery.TELEGRAM])
    monkeypatch.setattr(reminder_delivery, "telegram_owner", lambda: BO)
    telegram = FakeTelegram()

    outcomes = asyncio.run(_dispatcher(telegram=telegram).deliver_due(now=DUE))

    assert telegram.sent == []
    assert outcomes.get("refused") == 1
    assert _state_of(store, reminder_delivery.TELEGRAM) == reminder_delivery.FAILED


def test_a_telegram_failure_is_retried_within_a_bound_and_then_left_failed(store):
    _reminder([reminder_delivery.TELEGRAM])
    telegram = FakeTelegram()
    telegram.error = RuntimeError("telegram said no")
    dispatcher = _dispatcher(telegram=telegram)

    at = DUE
    for _ in range(reminder_delivery.MAX_ATTEMPTS):
        asyncio.run(dispatcher.deliver_due(now=at))
        at += reminder_delivery.BACKOFF_SECONDS[-1] + 1

    assert _state_of(store, reminder_delivery.TELEGRAM) == reminder_delivery.FAILED
    assert asyncio.run(dispatcher.deliver_due(now=at + 100_000)) == {}


def test_an_unconfigured_telegram_leaves_the_channel_pending_not_delivered(store):
    _reminder([reminder_delivery.TELEGRAM])

    outcomes = asyncio.run(_dispatcher(telegram=None).deliver_due(now=DUE))

    assert outcomes.get("unavailable") == 1
    assert _state_of(store, reminder_delivery.TELEGRAM) == reminder_delivery.PENDING


# --- the call ------------------------------------------------------------------------------


def test_a_call_is_placed_to_the_number_resolved_from_the_owner_profile(store):
    _reminder([reminder_delivery.CALL])
    placer = FakePlacer()

    outcomes = asyncio.run(_dispatcher(placer=placer).deliver_due(now=DUE))

    assert outcomes.get(reminder_delivery.DELIVERED) == 1
    assert len(placer.dispatched) == 1
    request = placer.dispatched[0][1]
    assert request.destination == NUMBER
    assert request.user_id == ANA
    assert request.reminder_id is not None
    assert request.callback_task_id is None
    assert _state_of(store, reminder_delivery.CALL) == reminder_delivery.DELIVERED


def test_a_profile_without_an_approved_number_is_refused_not_dialled(store, monkeypatch):
    _reminder([reminder_delivery.CALL])
    monkeypatch.setattr(reminder_delivery, "resolve_callback_number", lambda user_id: None)
    placer = FakePlacer()

    outcomes = asyncio.run(
        _dispatcher(placer=placer).deliver_due(now=DUE)
    )

    assert placer.reserved == [] and placer.dispatched == []
    assert outcomes.get("refused") == 1
    assert _state_of(store, reminder_delivery.CALL) == reminder_delivery.FAILED


def test_dry_run_builds_the_whole_request_and_places_nothing(store):
    """The verification path: everything up to the hand-off, and no phone rings."""
    _reminder([reminder_delivery.CALL])
    placer = FakePlacer()

    outcomes = asyncio.run(_dispatcher(placer=placer, dry_run=True).deliver_due(now=DUE))

    assert placer.reserved == [] and placer.dispatched == []
    assert outcomes.get("dry-run") == 1
    assert _state_of(store, reminder_delivery.CALL) == reminder_delivery.PENDING


def test_a_hand_off_that_may_have_been_received_is_never_retried(store):
    _reminder([reminder_delivery.CALL])
    placer = FakePlacer()
    placer.dispatch_error = RuntimeError("livekit went away mid hand-off")

    outcomes = asyncio.run(_dispatcher(placer=placer).deliver_due(now=DUE))

    assert outcomes.get("uncertain") == 1
    assert _state_of(store, reminder_delivery.CALL) == reminder_delivery.FAILED


def test_a_reservation_failure_is_retried_because_nothing_was_dispatched(store):
    _reminder([reminder_delivery.CALL])
    placer = FakePlacer()
    placer.reserve_error = RuntimeError("no room")

    outcomes = asyncio.run(_dispatcher(placer=placer).deliver_due(now=DUE))

    assert outcomes.get("retry") == 1
    assert _state_of(store, reminder_delivery.CALL) == reminder_delivery.PENDING


# --- independence ----------------------------------------------------------------------------


def test_one_channel_failing_does_not_suppress_the_other(store):
    _reminder([reminder_delivery.TELEGRAM, reminder_delivery.CALL])
    telegram = FakeTelegram()
    telegram.error = RuntimeError("telegram said no")
    placer = FakePlacer()

    asyncio.run(_dispatcher(telegram=telegram, placer=placer).deliver_due(now=DUE))

    assert len(placer.dispatched) == 1
    assert _state_of(store, reminder_delivery.CALL) == reminder_delivery.DELIVERED
    assert _state_of(store, reminder_delivery.TELEGRAM) == reminder_delivery.PENDING


def test_the_spoken_channel_is_never_touched_by_the_worker(store):
    _reminder([reminder_delivery.SPEAK, reminder_delivery.TELEGRAM])
    telegram = FakeTelegram()

    asyncio.run(_dispatcher(telegram=telegram).deliver_due(now=DUE))

    assert _state_of(store, reminder_delivery.SPEAK) == reminder_delivery.PENDING
    assert alarms_tools.claim_due_alarms(now=DUE, user_id=ANA) != []


def test_nothing_private_reaches_the_log(store, caplog):
    caplog.set_level(logging.DEBUG)
    _reminder([reminder_delivery.TELEGRAM, reminder_delivery.CALL], title="Biopsy results")
    asyncio.run(_dispatcher(telegram=FakeTelegram(), placer=FakePlacer()).deliver_due(now=DUE))

    text = "\n".join(record.getMessage() for record in caplog.records)
    for secret in ("Biopsy", "results", ANA, NUMBER, "+1555"):
        assert secret not in text


# --- wiring ------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_supervisor_delivers_reminders_on_every_tick(monkeypatch, tmp_path):
    from caal import background_tasks
    from caal.durable_work import DurableWorkSupervisor

    monkeypatch.setattr(background_tasks, "STORE_PATH", tmp_path / "tasks.sqlite3")

    class Recorder:
        def __init__(self) -> None:
            self.calls = 0

        async def deliver_due(self, limit: int = 10) -> dict:
            self.calls += 1
            return {}

    reminders = Recorder()
    supervisor = DurableWorkSupervisor(
        worker=lambda task: None, reminders=reminders, execute_work=False
    )

    await supervisor.tick()

    assert reminders.calls == 1
    assert supervisor.status()["reminder_delivery_configured"] is True


def test_the_worker_service_builds_a_reminder_dispatcher(monkeypatch):
    """Wired for real in the durable worker, not only reachable from a test."""
    from caal import work_service

    assert hasattr(work_service, "build_reminder_dispatcher")
    source = Path(work_service.__file__).read_text()
    assert "build_reminder_dispatcher" in source
    assert "reminders=" in source
