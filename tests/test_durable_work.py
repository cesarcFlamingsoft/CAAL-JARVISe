"""Regression coverage for the durable, user-scoped background work runtime.

These tests describe the failure that motivated the runtime: an outbound
phone call where the caller explicitly confirmed a callback, whose LiveKit
room and job process ended two seconds later. The work runner lived in that
job, so the work was interrupted and the armed callback was never dispatched.

Everything here uses fakes. No LiveKit API, no SIP participant, no dialing.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

import pytest

from caal import background_tasks, durable_work
from caal.background_tasks import (
    FAILED,
    QUEUED,
    RUNNING,
    SUCCEEDED,
    enqueue,
    get_task,
)
from caal.durable_work import (
    CallbackDispatcher,
    CallbackRefusedError,
    DispatchOutcome,
    DurableWorkSupervisor,
    build_callback_request,
)

USER_A = "usr_" + "a" * 24
USER_B = "usr_" + "b" * 24
NUMBER_A = "+15551230001"
NUMBER_B = "+15551230002"
OWNER_KEY = "conv-" + "f" * 64
SECRET_TEXT = "book the Lisbon flight for Alice"
TOKEN_HEADER = "X-CAAL-Worker-Token"
FAKE_TASK_ID = "bt_" + "0" * 16


@pytest.fixture
def store(monkeypatch, tmp_path):
    """Point every durable-store read and write at a throwaway database."""
    path = tmp_path / "assistant.sqlite3"
    monkeypatch.setattr(background_tasks, "STORE_PATH", path)
    return path


def _resolver(mapping):
    def _resolve(user_id: str) -> str | None:
        return mapping.get(user_id)

    return _resolve


async def _answer(text: str) -> str:
    return text


class FakePlacer:
    """Stand-in for the LiveKit outbound placer, with per-step failure control."""

    def __init__(self, *, fail_reserve: int = 0, fail_dispatch: int = 0) -> None:
        self.reserved: list[str] = []
        self.dispatched: list[tuple[str, object]] = []
        self._fail_reserve = fail_reserve
        self._fail_dispatch = fail_dispatch

    async def reserve(self, request) -> str:
        if self._fail_reserve > 0:
            self._fail_reserve -= 1
            raise RuntimeError("livekit room creation refused")
        room = "caal-outbound-" + request.attempt_id
        self.reserved.append(room)
        return room

    async def dispatch(self, room_name: str, request) -> None:
        if self._fail_dispatch > 0:
            self._fail_dispatch -= 1
            raise RuntimeError("livekit dispatch refused")
        self.dispatched.append((room_name, request))


def _dispatcher(placer, *, mapping=None, claimant="dw_test", clock=None, **kwargs):
    approved = mapping if mapping is not None else dict([(USER_A, NUMBER_A)])
    return CallbackDispatcher(
        placer=placer,
        resolve_user_destination=_resolver(approved),
        claimant=claimant,
        clock=clock or time.time,
        **kwargs,
    )


def _arm(task_id: str, *, user_id: str | None = USER_A, session_key: str = OWNER_KEY) -> None:
    assert background_tasks.arm_callback(
        task_id, None if user_id else NUMBER_A, session_key=session_key, user_id=user_id
    )


def _settle(task_id: str, status: str = SUCCEEDED, result: str = "done") -> None:
    background_tasks._mark_running(task_id)
    assert background_tasks._finish(task_id, status, result=result)


def _supervisor(dispatcher, *, answer: str = "flights are 180 euros"):
    return DurableWorkSupervisor(
        worker=lambda item: _answer(answer),
        dispatcher=dispatcher,
        worker_id="dw_supervisor",
    )


# ---------------------------------------------------------------------------
# 1. The incident: the room and job end after the callback is armed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_room_exit_after_armed_callback_keeps_work_and_dispatches_later(store) -> None:
    """A job that ends mid-task must hand the work back, not interrupt it."""
    task = enqueue(SECRET_TEXT, session_key=OWNER_KEY, user_id=USER_A)
    _arm(task.task_id)

    # The phone job leased the work, then its room was deleted and it exited.
    leased = background_tasks.lease_next_queued("job-phone-1")
    assert leased is not None and leased.task_id == task.task_id
    assert background_tasks.release_lease(task.task_id, "job-phone-1")

    assert get_task(task.task_id).status == QUEUED
    assert background_tasks.callback_pending(task.task_id)

    placer = FakePlacer()
    supervisor = _supervisor(_dispatcher(placer))
    await supervisor.tick()
    await supervisor.drain_active()
    await supervisor.tick()

    assert get_task(task.task_id).status == SUCCEEDED
    assert len(placer.dispatched) == 1
    _, request = placer.dispatched[0]
    assert request.callback_task_id == task.task_id
    assert request.user_id == USER_A
    assert request.destination == NUMBER_A


# ---------------------------------------------------------------------------
# 2. Ordinary work with no session left
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_with_no_session_settles_and_stays_announceable(store) -> None:
    task = enqueue("summarise the quarterly report", session_key=OWNER_KEY, user_id=USER_A)
    supervisor = _supervisor(_dispatcher(FakePlacer()), answer="summary ready")
    await supervisor.tick()
    await supervisor.drain_active()

    settled = get_task(task.task_id)
    assert settled.status == SUCCEEDED
    assert settled.notified_at is None
    pending = background_tasks.pending_notifications(user_id=USER_A)
    assert [item.task_id for item in pending] == [task.task_id]


# ---------------------------------------------------------------------------
# 3. Restart recovery
# ---------------------------------------------------------------------------


def test_expired_leases_are_requeued_and_poison_work_is_dead_lettered(store) -> None:
    task = enqueue("long research job", session_key=OWNER_KEY, user_id=USER_A)
    for _ in range(background_tasks.MAX_TASK_ATTEMPTS):
        assert background_tasks.lease_next_queued("dw_dead", lease_seconds=1) is not None
        requeued, dead = background_tasks.requeue_expired_leases(now=int(time.time()) + 3600)
        assert requeued + dead == 1

    settled = get_task(task.task_id)
    assert settled.status == FAILED
    assert background_tasks.lease_next_queued("dw_next") is None


def test_orphaned_running_work_is_never_adopted_automatically(store) -> None:
    """Rows left RUNNING by the old session-bound runner carry no lease.

    Restarting must not silently re-run them (and so must not silently place
    the callback armed on them): adoption is an explicit operator action.
    """
    task = enqueue("work from the old runner", session_key=OWNER_KEY, user_id=USER_A)
    background_tasks._mark_running(task.task_id)

    assert background_tasks.requeue_expired_leases(now=int(time.time()) + 3600) == (0, 0)
    assert get_task(task.task_id).status == RUNNING
    assert background_tasks.orphaned_running_count() == 1

    assert background_tasks.adopt_orphaned_running() == 1
    assert get_task(task.task_id).status == QUEUED


# ---------------------------------------------------------------------------
# 4. Exactly-once dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_dispatchers_place_exactly_one_callback(store) -> None:
    task = enqueue(SECRET_TEXT, session_key=OWNER_KEY, user_id=USER_A)
    _arm(task.task_id)
    _settle(task.task_id)

    placer = FakePlacer()
    first = _dispatcher(placer, claimant="dw_one")
    second = _dispatcher(placer, claimant="dw_two")
    await asyncio.gather(first.dispatch_due(), second.dispatch_due())

    assert len(placer.dispatched) == 1
    # And a third pass, after the fact, still places nothing.
    await first.dispatch_due()
    assert len(placer.dispatched) == 1
    assert get_task(task.task_id).notified_at is not None


# ---------------------------------------------------------------------------
# 5 and 6. Failed dispatch: release, retry, and the ambiguous case
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_reservation_releases_the_claim_and_retries_with_backoff(store) -> None:
    task = enqueue(SECRET_TEXT, session_key=OWNER_KEY, user_id=USER_A)
    _arm(task.task_id)
    _settle(task.task_id)

    now = time.time()
    placer = FakePlacer(fail_reserve=1)
    dispatcher = _dispatcher(placer, clock=lambda: now)
    assert await dispatcher.dispatch_due() == dict([(DispatchOutcome.RETRY, 1)])

    # Nothing was dialed, nothing was announced, and the authorization survives.
    assert placer.dispatched == []
    assert get_task(task.task_id).notified_at is None
    assert background_tasks.callback_pending(task.task_id)
    # Backed off: an immediate second pass does not even claim it.
    assert await dispatcher.dispatch_due() == dict()

    later = now + background_tasks.CALLBACK_BACKOFF_MAX_SECONDS + 1
    dispatcher = _dispatcher(placer, clock=lambda: later)
    assert await dispatcher.dispatch_due() == dict([(DispatchOutcome.SENT, 1)])
    assert len(placer.dispatched) == 1


@pytest.mark.asyncio
async def test_ambiguous_dispatch_never_dials_again_and_never_claims_delivery(store) -> None:
    """A dispatch call that failed after being issued may already have a job."""
    task = enqueue(SECRET_TEXT, session_key=OWNER_KEY, user_id=USER_A)
    _arm(task.task_id)
    _settle(task.task_id)

    placer = FakePlacer(fail_dispatch=1)
    dispatcher = _dispatcher(placer)
    assert await dispatcher.dispatch_due() == dict([(DispatchOutcome.UNCERTAIN, 1)])

    assert placer.dispatched == []
    # Never retried: a second job would be a second call to the same person.
    assert await dispatcher.dispatch_due() == dict()
    assert not background_tasks.callback_pending(task.task_id)
    # Delivery was not claimed, so the ordinary fallback still reports it.
    assert get_task(task.task_id).notified_at is None


@pytest.mark.asyncio
async def test_dispatch_gives_up_after_the_attempt_budget(store) -> None:
    task = enqueue(SECRET_TEXT, session_key=OWNER_KEY, user_id=USER_A)
    _arm(task.task_id)
    _settle(task.task_id)

    placer = FakePlacer(fail_reserve=background_tasks.MAX_CALLBACK_ATTEMPTS + 2)
    moment = time.time()

    def _clock() -> float:
        return moment

    for _ in range(background_tasks.MAX_CALLBACK_ATTEMPTS):
        dispatcher = _dispatcher(placer, clock=_clock)
        await dispatcher.dispatch_due()
        moment += background_tasks.CALLBACK_BACKOFF_MAX_SECONDS + 1

    assert placer.dispatched == []
    assert not background_tasks.callback_pending(task.task_id)
    # Given up honestly: the outcome is still unannounced, so it can be
    # delivered by the session or the fallback instead.
    assert get_task(task.task_id).notified_at is None


# ---------------------------------------------------------------------------
# 7. Ownership
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_user_and_anonymous_callbacks_are_refused(store) -> None:
    task = enqueue(SECRET_TEXT, session_key=OWNER_KEY, user_id=USER_A)
    _arm(task.task_id)
    _settle(task.task_id)

    placer = FakePlacer()
    # The resolver knows only another user's number; USER_A has none.
    dispatcher = _dispatcher(placer, mapping=dict([(USER_B, NUMBER_B)]))
    assert await dispatcher.dispatch_due() == dict([(DispatchOutcome.REFUSED, 1)])
    assert placer.reserved == [] and placer.dispatched == []
    assert get_task(task.task_id).notified_at is None

    anonymous = enqueue("anonymous work", session_key="room-anon")
    assert background_tasks.arm_callback(anonymous.task_id, NUMBER_B, session_key="room-anon")
    _settle(anonymous.task_id)
    assert await dispatcher.dispatch_due() == dict([(DispatchOutcome.REFUSED, 1)])
    assert placer.dispatched == []


def test_a_callback_may_never_name_its_own_number(store) -> None:
    with pytest.raises(CallbackRefusedError):
        build_callback_request(
            task_id=FAKE_TASK_ID,
            user_id=USER_A,
            destination="+19998887777",
            resolve_user_destination=_resolver(dict([(USER_A, NUMBER_A)])),
        )
    with pytest.raises(CallbackRefusedError):
        build_callback_request(
            task_id=FAKE_TASK_ID,
            user_id=USER_A,
            destination=None,
            resolve_user_destination=_resolver(dict()),
        )


def test_standalone_outbound_request_construction_is_safe(store) -> None:
    request = build_callback_request(
        task_id=FAKE_TASK_ID,
        user_id=USER_A,
        destination=None,
        resolve_user_destination=_resolver(dict([(USER_A, NUMBER_A)])),
    )
    assert request.destination == NUMBER_A
    assert request.user_id == USER_A
    assert request.callback_task_id == FAKE_TASK_ID
    metadata = request.dispatch_metadata()
    assert metadata["caal_outbound"] is True
    assert metadata["callback_task_id"] == FAKE_TASK_ID
    assert "handoff_context" not in metadata
    # Nothing the user said, and no number, rides along in a printable form.
    assert SECRET_TEXT not in repr(request)
    assert NUMBER_A not in repr(request)


# ---------------------------------------------------------------------------
# 8. One owner, one channel
# ---------------------------------------------------------------------------


def test_an_armed_callback_preempts_session_and_fallback_delivery(store) -> None:
    armed = enqueue(SECRET_TEXT, session_key=OWNER_KEY, user_id=USER_A)
    plain = enqueue("other work", session_key=OWNER_KEY, user_id=USER_A)
    _arm(armed.task_id)
    _settle(armed.task_id)
    _settle(plain.task_id)

    visible = background_tasks.pending_notifications(user_id=USER_A, exclude_callback_armed=True)
    assert [item.task_id for item in visible] == [plain.task_id]

    claimed = background_tasks.claim_next_notification(
        OWNER_KEY, OWNER_KEY, user_id=USER_A, exclude_callback_armed=True
    )
    assert claimed is not None and claimed.task_id == plain.task_id
    assert (
        background_tasks.claim_next_notification(
            OWNER_KEY, OWNER_KEY, user_id=USER_A, exclude_callback_armed=True
        )
        is None
    )
    # The armed task is untouched and still callable back.
    assert get_task(armed.task_id).notified_at is None
    assert background_tasks.callback_pending(armed.task_id)


@pytest.mark.asyncio
async def test_bridge_teardown_releases_leases_instead_of_interrupting(store) -> None:
    """The session-bound path must hand work back to the durable runtime."""
    from caal.background_task_session import BackgroundTaskBridge

    gate = asyncio.Event()

    async def _execute(request: str, context: str) -> str:
        await gate.wait()
        return "done"

    bridge = BackgroundTaskBridge(execute=_execute, session_key=OWNER_KEY, user_id=USER_A)
    await bridge.start()
    enqueue(SECRET_TEXT, session_key=OWNER_KEY, user_id=USER_A)
    task_id = background_tasks.list_tasks()[0].task_id
    bridge._runner.poke()
    await asyncio.sleep(0.05)
    assert get_task(task_id).status == RUNNING

    await bridge.abandon()
    assert get_task(task_id).status == QUEUED
    gate.set()


# ---------------------------------------------------------------------------
# 9. Logging hygiene
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nothing_logs_user_text_numbers_or_opaque_ids(store, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    task = enqueue(SECRET_TEXT, session_key=OWNER_KEY, user_id=USER_A)
    _arm(task.task_id)

    supervisor = _supervisor(_dispatcher(FakePlacer()), answer="flights are 180 euros to Lisbon")
    await supervisor.tick()
    await supervisor.drain_active()
    await supervisor.tick()
    status = supervisor.status()

    blob = "\n".join(record.getMessage() for record in caplog.records) + repr(status)
    for forbidden in (SECRET_TEXT, "Lisbon", NUMBER_A, USER_A, task.task_id, OWNER_KEY):
        assert forbidden not in blob
    assert not re.search(r"\+\d{8,}", blob)
    assert not re.search(r"\bbt_[0-9a-f]{16}\b", blob)
    assert not re.search(r"\busr_[0-9a-zA-Z_-]{20,}\b", blob)


def test_status_reports_counts_only(store) -> None:
    enqueue(SECRET_TEXT, session_key=OWNER_KEY, user_id=USER_A)
    supervisor = _supervisor(_dispatcher(FakePlacer()))
    status = supervisor.status()
    assert status["queued"] == 1
    assert status["active"] == 0
    assert status["orphaned_running"] == 0
    assert all(isinstance(value, (int, str, bool)) for value in status.values())


# ---------------------------------------------------------------------------
# The worker service front door
# ---------------------------------------------------------------------------


def test_worker_service_status_requires_the_internal_token(store, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from caal import work_service

    secret = "s" * 48
    monkeypatch.setenv("CAAL_INTERNAL_AUTH_SECRET", secret)
    supervisor = _supervisor(_dispatcher(FakePlacer()))
    client = TestClient(work_service.build_app(supervisor, wake=asyncio.Event()))

    assert client.get("/healthz").status_code == 200
    assert client.get("/status").status_code == 401
    assert client.post("/internal/wake").status_code == 401
    bad = dict([(TOKEN_HEADER, "wrong")])
    good = dict([(TOKEN_HEADER, secret)])
    assert client.get("/status", headers=bad).status_code == 401

    ok = client.get("/status", headers=good)
    assert ok.status_code == 200
    assert ok.json()["queued"] == 0
    assert client.post("/internal/wake", headers=good).status_code == 200


def test_worker_service_refuses_to_serve_without_an_internal_secret(store, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from caal import work_service

    monkeypatch.delenv("CAAL_INTERNAL_AUTH_SECRET", raising=False)
    supervisor = _supervisor(_dispatcher(FakePlacer()))
    client = TestClient(work_service.build_app(supervisor, wake=asyncio.Event()))
    # Liveness stays available for the container probe; everything else is shut.
    assert client.get("/healthz").status_code == 200
    headers = dict([(TOKEN_HEADER, "anything")])
    assert client.get("/status", headers=headers).status_code == 503


@pytest.mark.asyncio
async def test_dry_run_dispatch_constructs_without_placing_a_call(store) -> None:
    """The operational verification path: build the request, dial nothing."""
    task = enqueue(SECRET_TEXT, session_key=OWNER_KEY, user_id=USER_A)
    _arm(task.task_id)
    _settle(task.task_id)

    placer = FakePlacer()
    dispatcher = _dispatcher(placer, dry_run=True)
    assert await dispatcher.dispatch_due() == dict([(DispatchOutcome.DRY_RUN, 1)])
    assert placer.reserved == [] and placer.dispatched == []
    # The authorization is left exactly as it was found.
    assert background_tasks.callback_pending(task.task_id)
    assert get_task(task.task_id).notified_at is None


def test_module_exports_are_stable() -> None:
    for name in ("CallbackDispatcher", "DurableWorkSupervisor", "build_callback_request"):
        assert hasattr(durable_work, name)


@pytest.mark.asyncio
async def test_dry_run_never_spends_the_real_retry_budget(store) -> None:
    """Verification mode must not exhaust the attempts a real dispatch needs."""
    task = enqueue(SECRET_TEXT, session_key=OWNER_KEY, user_id=USER_A)
    _arm(task.task_id)
    _settle(task.task_id)

    placer = FakePlacer()
    moment = time.time()
    for _ in range(background_tasks.MAX_CALLBACK_ATTEMPTS + 3):
        verifier = _dispatcher(placer, dry_run=True, clock=lambda: moment)
        await verifier.dispatch_due()
        moment += background_tasks.CALLBACK_BACKOFF_MAX_SECONDS + 1

    assert placer.reserved == [] and placer.dispatched == []
    # Flipped to live, the very next pass still places the call.
    live = _dispatcher(placer, clock=lambda: moment)
    assert await live.dispatch_due() == dict([(DispatchOutcome.SENT, 1)])
