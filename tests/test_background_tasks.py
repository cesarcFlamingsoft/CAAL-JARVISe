"""Coverage for the durable background task manager.

The manager is a small vertical slice: a persistent queue in the shared CAAL
SQLite store, an async runner fed by an injected worker coroutine, and an
exact-once notification claim so a finished task is announced to the user
only once. Task text is bounded and redacted before it is persisted, and it
never appears in logs or in ``repr`` output.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
import threading

import pytest

from caal import background_tasks
from caal.background_tasks import (
    CANCELLED,
    DEFAULT_MAX_CONCURRENCY,
    FAILED,
    INTERRUPTED,
    MAX_CONCURRENCY_LIMIT,
    MAX_ERROR_CHARS,
    MAX_QUEUE_DEPTH,
    MAX_REQUEST_CHARS,
    MAX_RESULT_CHARS,
    QUEUED,
    RUNNING,
    SUCCEEDED,
    TERMINAL_STATUSES,
    BackgroundTask,
    BackgroundTaskRunner,
    QueueFullError,
    background_task_requested,
    cancel,
    claim_next_notification,
    claim_notification,
    enqueue,
    get_task,
    list_tasks,
    pending_notifications,
    recover_interrupted,
    redact_secrets,
)

SECRET = "sk-live-ZZZ0SUPERSECRET0ZZZ"
PASSWORD = "hunter2hunter2hunter2"


@pytest.fixture
def store(monkeypatch, tmp_path):
    path = tmp_path / "assistant.sqlite3"
    monkeypatch.setattr(background_tasks, "STORE_PATH", path)
    return path


@pytest.fixture
def clock(monkeypatch):
    state = {"now": 1_700_000_000}

    def tick(seconds: int = 1) -> int:
        state["now"] += seconds
        return state["now"]

    monkeypatch.setattr(background_tasks, "_now", lambda: state["now"])
    tick.now = lambda: state["now"]  # type: ignore[attr-defined]
    return tick


@pytest.fixture
def task_logs(caplog):
    """Capture the module logger directly; other tests may stop it propagating."""
    target = logging.getLogger("caal.background_tasks")
    previous_level = target.level
    target.setLevel(logging.DEBUG)
    target.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        target.removeHandler(caplog.handler)
        target.setLevel(previous_level)


def _raw_dump(path) -> str:
    connection = sqlite3.connect(path)
    try:
        return "\n".join(connection.iterdump())
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Redaction and bounds
# ---------------------------------------------------------------------------


def test_redact_secrets_masks_common_credential_shapes() -> None:
    aws_access_key = "AK" + "IA" + "TEST" + "123456789012"
    github_token = "gh" + "p_" + "testtokenvalue1234567890"
    text = (
        f"use api_key={SECRET} and password: {PASSWORD}; "
        "Authorization: Bearer abcdefghijklmnop.qrstuvwxyz "
        f"aws {aws_access_key} github {github_token}"
    )
    redacted = redact_secrets(text)
    assert SECRET not in redacted
    assert PASSWORD not in redacted
    assert "abcdefghijklmnop.qrstuvwxyz" not in redacted
    assert aws_access_key not in redacted
    assert github_token not in redacted
    assert "[REDACTED]" in redacted
    # Ordinary text is left alone.
    assert redact_secrets("look up tomorrow's weather in Austin") == (
        "look up tomorrow's weather in Austin"
    )
    assert redact_secrets("") == ""


def test_enqueue_bounds_and_redacts_request(store, clock) -> None:
    long_request = "summarise " + ("x" * (MAX_REQUEST_CHARS * 2)) + " " + SECRET
    task = enqueue(long_request)
    assert len(task.request) <= MAX_REQUEST_CHARS
    assert SECRET not in task.request
    assert SECRET not in _raw_dump(store)


def test_enqueue_rejects_empty_request(store, clock) -> None:
    with pytest.raises(ValueError):
        enqueue("   ")


def test_queue_depth_is_bounded(store, clock) -> None:
    for _ in range(MAX_QUEUE_DEPTH):
        enqueue("task")
    with pytest.raises(QueueFullError):
        enqueue("one too many")
    # Finished tasks no longer count against the bound.
    first = list_tasks(statuses=[QUEUED])[0]
    assert cancel(first.task_id)
    enqueue("room again")


# ---------------------------------------------------------------------------
# Opaque IDs, snapshots, statuses
# ---------------------------------------------------------------------------


def test_task_ids_are_opaque_and_unique(store, clock) -> None:
    request = "find cheap flights to Lisbon"
    a = enqueue(request)
    b = enqueue(request)
    assert a.task_id != b.task_id
    assert re.fullmatch(r"bt_[0-9a-f]{16}", a.task_id)
    assert "lisbon" not in a.task_id.lower()
    assert "1" not in a.task_id[:3]  # no sequential prefix


def test_snapshot_repr_omits_text(store, clock) -> None:
    task = enqueue(f"ask the bank with password: {PASSWORD} and remember cheddar")
    for rendered in (repr(task), str(task), f"{task}"):
        assert task.task_id in rendered
        assert QUEUED in rendered
        assert "cheddar" not in rendered
        assert PASSWORD not in rendered
        assert "request_chars=" in rendered
    # The text is still reachable on purpose for whoever renders it to the user.
    assert "cheddar" in task.request
    assert isinstance(task, BackgroundTask)
    assert task.is_terminal is False


def test_statuses_and_terminal_set() -> None:
    assert TERMINAL_STATUSES == frozenset({SUCCEEDED, FAILED, CANCELLED, INTERRUPTED})
    assert QUEUED not in TERMINAL_STATUSES
    assert RUNNING not in TERMINAL_STATUSES
    assert 1 <= DEFAULT_MAX_CONCURRENCY <= MAX_CONCURRENCY_LIMIT


def test_get_and_list_tasks(store, clock) -> None:
    assert get_task("bt_0000000000000000") is None
    older = enqueue("first", session_key="web:1")
    clock()
    newer = enqueue("second", session_key="phone:2")
    assert get_task(older.task_id) == older
    assert [t.task_id for t in list_tasks()] == [older.task_id, newer.task_id]
    assert [t.task_id for t in list_tasks(session_key="phone:2")] == [newer.task_id]
    assert list_tasks(statuses=[SUCCEEDED]) == []
    assert len(list_tasks(limit=1)) == 1


# ---------------------------------------------------------------------------
# Cancel and recovery
# ---------------------------------------------------------------------------


def test_cancel_queued_task(store, clock) -> None:
    task = enqueue("cancel me")
    clock(5)
    assert cancel(task.task_id) is True
    after = get_task(task.task_id)
    assert after.status == CANCELLED
    assert after.finished_at == clock.now()
    # Cancelling again, or cancelling a terminal task, is a no-op.
    assert cancel(task.task_id) is False
    assert cancel("bt_doesnotexist0000") is False


def test_recover_running_to_interrupted(store, clock) -> None:
    running = enqueue("was running when the process died")
    queued = enqueue("still waiting")
    background_tasks._mark_running(running.task_id)
    assert get_task(running.task_id).status == RUNNING
    clock(10)
    assert recover_interrupted() == 1
    assert recover_interrupted() == 0
    recovered = get_task(running.task_id)
    assert recovered.status == INTERRUPTED
    assert recovered.finished_at == clock.now()
    assert recovered.error is not None and SECRET not in recovered.error
    assert get_task(queued.task_id).status == QUEUED
    # Interrupted tasks are announced to the user exactly like other finished tasks.
    assert [t.task_id for t in pending_notifications()] == [running.task_id]


# ---------------------------------------------------------------------------
# Exact-once notification claim
# ---------------------------------------------------------------------------


def test_notification_claim_is_exact_once(store, clock) -> None:
    task = enqueue("finish me", session_key="web:1")
    assert claim_notification(task.task_id, "web:1") is None  # not finished yet
    background_tasks._mark_running(task.task_id)
    background_tasks._finish(task.task_id, SUCCEEDED, result="done")
    assert [t.task_id for t in pending_notifications(session_key="web:1")] == [task.task_id]
    assert pending_notifications(session_key="phone:9") == []

    claimed = claim_notification(task.task_id, "web:1")
    assert claimed is not None and claimed.notified_at == clock.now()
    assert claim_notification(task.task_id, "web:1") is None
    assert claim_notification(task.task_id, "phone:9") is None
    assert pending_notifications() == []


def test_claim_next_notification_walks_oldest_first(store, clock) -> None:
    first = enqueue("a", session_key="web:1")
    clock()
    second = enqueue("b", session_key="web:1")
    for task in (first, second):
        background_tasks._mark_running(task.task_id)
        background_tasks._finish(task.task_id, FAILED, error="boom")
    assert claim_next_notification("web:1", "web:1").task_id == first.task_id
    assert claim_next_notification("web:1", "web:1").task_id == second.task_id
    assert claim_next_notification("web:1", "web:1") is None


def test_concurrent_claims_yield_exactly_one_winner(store, clock) -> None:
    task = enqueue("race me")
    background_tasks._mark_running(task.task_id)
    background_tasks._finish(task.task_id, SUCCEEDED, result="ok")
    barrier = threading.Barrier(8)
    winners: list[str] = []
    lock = threading.Lock()

    def attempt(name: str) -> None:
        barrier.wait()
        if claim_notification(task.task_id, name) is not None:
            with lock:
                winners.append(name)

    threads = [threading.Thread(target=attempt, args=(f"c{i}",)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(winners) == 1


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_executes_queued_tasks_with_injected_worker(store, clock) -> None:
    seen: list[str] = []

    async def worker(task: BackgroundTask) -> str:
        seen.append(task.request)
        return f"result for {task.request}"

    runner = BackgroundTaskRunner(worker)
    await runner.start()
    try:
        task = await runner.submit("look up train times", session_key="web:1")
        finished = await runner.wait(task.task_id, timeout=2)
    finally:
        await runner.stop()
    assert seen == ["look up train times"]
    assert finished.status == SUCCEEDED
    assert finished.result == "result for look up train times"
    assert finished.started_at is not None and finished.finished_at is not None
    assert [t.task_id for t in pending_notifications(session_key="web:1")] == [task.task_id]


@pytest.mark.asyncio
async def test_runner_drains_tasks_persisted_before_start(store, clock) -> None:
    before = enqueue("queued before restart")
    background_tasks._mark_running(enqueue("died mid-flight").task_id)

    async def worker(task: BackgroundTask) -> str:
        return "ok"

    runner = BackgroundTaskRunner(worker)
    await runner.start()
    try:
        assert (await runner.wait(before.task_id, timeout=2)).status == SUCCEEDED
    finally:
        await runner.stop()
    # Starting a runner never destroys work it does not own. The row left
    # running by a pre-lease runner holds no lease, so it is neither resumed
    # nor declared interrupted here: it waits for explicit operator adoption,
    # because resuming it can end in an outbound callback.
    statuses = sorted(t.status for t in list_tasks())
    assert statuses == sorted([SUCCEEDED, RUNNING])
    assert background_tasks.orphaned_running_count() == 1


@pytest.mark.asyncio
async def test_runner_respects_max_concurrency(store, clock) -> None:
    gate = asyncio.Event()
    active = 0
    peak = 0

    async def worker(task: BackgroundTask) -> str:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await gate.wait()
        active -= 1
        return "ok"

    runner = BackgroundTaskRunner(worker, max_concurrency=2)
    await runner.start()
    try:
        tasks = [await runner.submit(f"job {i}") for i in range(5)]
        await asyncio.sleep(0.05)
        assert runner.running_count == 2
        assert peak == 2
        assert sorted(t.status for t in list_tasks()) == sorted(
            [RUNNING, RUNNING, QUEUED, QUEUED, QUEUED]
        )
        gate.set()
        for task in tasks:
            assert (await runner.wait(task.task_id, timeout=2)).status == SUCCEEDED
    finally:
        await runner.stop()
    assert peak == 2
    assert runner.running_count == 0


def test_runner_validates_concurrency() -> None:
    async def worker(task: BackgroundTask) -> str:
        return ""

    with pytest.raises(ValueError):
        BackgroundTaskRunner(worker, max_concurrency=0)
    with pytest.raises(ValueError):
        BackgroundTaskRunner(worker, max_concurrency=MAX_CONCURRENCY_LIMIT + 1)


@pytest.mark.asyncio
async def test_runner_records_failures_bounded_and_redacted(store, clock, task_logs) -> None:
    async def worker(task: BackgroundTask) -> str:
        raise RuntimeError(f"upstream rejected token={SECRET} " + ("y" * MAX_ERROR_CHARS * 2))

    runner = BackgroundTaskRunner(worker)
    await runner.start()
    try:
        task = await runner.submit("call the api")
        finished = await runner.wait(task.task_id, timeout=2)
    finally:
        await runner.stop()
    assert finished.status == FAILED
    assert finished.error.startswith("RuntimeError")
    assert len(finished.error) <= MAX_ERROR_CHARS
    assert SECRET not in finished.error
    assert SECRET not in _raw_dump(store)
    assert "failed" in task_logs.text  # proves the capture is live
    assert task.task_id not in task_logs.text  # opaque ids stay out of logs
    assert SECRET not in task_logs.text
    assert "call the api" not in task_logs.text


@pytest.mark.asyncio
async def test_runner_bounds_and_redacts_results_and_logs(store, clock, task_logs) -> None:
    async def worker(task: BackgroundTask) -> str:
        return f"password: {PASSWORD} " + ("z" * MAX_RESULT_CHARS * 2)

    runner = BackgroundTaskRunner(worker)
    await runner.start()
    try:
        task = await runner.submit(f"log into the portal with password: {PASSWORD}")
        finished = await runner.wait(task.task_id, timeout=2)
    finally:
        await runner.stop()
    assert finished.status == SUCCEEDED
    assert len(finished.result) <= MAX_RESULT_CHARS
    assert PASSWORD not in finished.result
    assert PASSWORD not in _raw_dump(store)
    assert "succeeded" in task_logs.text  # proves the capture is live
    assert task.task_id not in task_logs.text  # opaque ids stay out of logs
    assert PASSWORD not in task_logs.text
    assert "portal" not in task_logs.text


@pytest.mark.asyncio
async def test_runner_cancels_running_task(store, clock) -> None:
    started = asyncio.Event()
    cancelled_inside = asyncio.Event()

    async def worker(task: BackgroundTask) -> str:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled_inside.set()
            raise
        return "never"

    runner = BackgroundTaskRunner(worker)
    await runner.start()
    try:
        task = await runner.submit("slow job")
        await asyncio.wait_for(started.wait(), 2)
        assert await runner.cancel(task.task_id) is True
        finished = await runner.wait(task.task_id, timeout=2)
        assert cancelled_inside.is_set()
        assert finished.status == CANCELLED
        assert runner.running_count == 0
        assert await runner.cancel(task.task_id) is False
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_runner_cancels_queued_task_before_it_runs(store, clock) -> None:
    gate = asyncio.Event()
    ran: list[str] = []

    async def worker(task: BackgroundTask) -> str:
        ran.append(task.request)
        await gate.wait()
        return "ok"

    runner = BackgroundTaskRunner(worker, max_concurrency=1)
    await runner.start()
    try:
        blocker = await runner.submit("blocker")
        waiting = await runner.submit("waiting")
        await asyncio.sleep(0.02)
        assert await runner.cancel(waiting.task_id) is True
        gate.set()
        await runner.wait(blocker.task_id, timeout=2)
        await asyncio.sleep(0.02)
    finally:
        await runner.stop()
    assert ran == ["blocker"]
    assert get_task(waiting.task_id).status == CANCELLED


@pytest.mark.asyncio
async def test_stop_hands_in_flight_tasks_back_to_the_queue(store, clock) -> None:
    """Stopping a runner must not destroy the work it happens to be holding.

    The runner leases work; a process on its way out releases the lease so the
    durable work service resumes the task. Recording it as interrupted -- which
    is what this used to do -- dropped any callback armed on it, which is
    exactly how a caller who had confirmed a callback was never called back.
    """
    started = asyncio.Event()

    async def worker(task: BackgroundTask) -> str:
        started.set()
        await asyncio.sleep(30)
        return "never"

    runner = BackgroundTaskRunner(worker)
    await runner.start()
    task = await runner.submit("long haul")
    await asyncio.wait_for(started.wait(), 2)
    await runner.stop()
    assert get_task(task.task_id).status == QUEUED
    assert get_task(task.task_id).finished_at is None
    assert runner.running_count == 0
    # A stopped runner refuses new work instead of silently dropping it.
    with pytest.raises(RuntimeError):
        await runner.submit("after stop")


@pytest.mark.asyncio
async def test_late_worker_result_does_not_overwrite_cancellation(store, clock) -> None:
    release = asyncio.Event()

    async def worker(task: BackgroundTask) -> str:
        try:
            await release.wait()
        except asyncio.CancelledError:
            pass  # swallow: a badly behaved worker that keeps going
        return "late result"

    runner = BackgroundTaskRunner(worker)
    await runner.start()
    try:
        task = await runner.submit("stubborn")
        await asyncio.sleep(0.02)
        assert await runner.cancel(task.task_id) is True
        finished = await runner.wait(task.task_id, timeout=2)
    finally:
        await runner.stop()
    assert finished.status == CANCELLED
    assert finished.result is None


@pytest.mark.asyncio
async def test_wait_times_out_for_unfinished_task(store, clock) -> None:
    async def worker(task: BackgroundTask) -> str:
        await asyncio.sleep(30)
        return ""

    runner = BackgroundTaskRunner(worker)
    await runner.start()
    try:
        task = await runner.submit("slow")
        with pytest.raises(asyncio.TimeoutError):
            await runner.wait(task.task_id, timeout=0.05)
        with pytest.raises(KeyError):
            await runner.wait("bt_doesnotexist0000", timeout=0.05)
    finally:
        await runner.stop()


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Research flight prices to Tokyo in the background and let me know when it's done",
        "run this as a background task",
        "Can you work on the quarterly summary in the background?",
        "compare those three laptops and get back to me when you're done",
        "Look into it and notify me when it's finished.",
        "Start a background job to dig through my inbox for the invoice",
        "keep working on that while I take this call",
    ],
)
def test_classifier_accepts_explicit_background_requests(text: str) -> None:
    assert background_task_requested(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "what's the weather tomorrow",
        "set a timer for five minutes",
        "play some jazz in the background",
        "change my background color to blue",
        "what is a background task?",
        "tell me when the meeting is",
        "remind me to call mom later",
        "how do I get back to the main menu",
    ],
)
def test_classifier_rejects_ordinary_requests(text: str) -> None:
    assert background_task_requested(text) is False


def test_classifier_is_deterministic_and_bounded() -> None:
    text = "please handle this in the background and ping me when it's done"
    assert all(background_task_requested(text) for _ in range(5))
    assert background_task_requested(("blah " * 10_000) + text) is False
    assert background_task_requested(text + (" blah" * 10_000)) is True
    assert background_task_requested(None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# One-time callback authorization
# ---------------------------------------------------------------------------

DESTINATION = "+17805558345"


def test_task_id_shape_is_exact() -> None:
    assert background_tasks.is_valid_task_id(enqueue.__name__) is False
    assert background_tasks.is_valid_task_id("bt_" + "0" * 16) is True
    assert background_tasks.is_valid_task_id("bt_" + "0" * 15) is False
    assert background_tasks.is_valid_task_id("bt_" + "g" * 16) is False
    assert background_tasks.is_valid_task_id(None) is False


def test_arm_callback_requires_an_open_task_of_the_same_session(store, clock) -> None:
    task = enqueue("look it up", session_key="phone-1")
    assert (
        background_tasks.arm_callback("bt_doesnotexist0000", DESTINATION, session_key="phone-1")
        is False
    )
    assert background_tasks.arm_callback(task.task_id, DESTINATION, session_key="other") is False
    assert background_tasks.arm_callback(task.task_id, "", session_key="phone-1") is False
    assert background_tasks.callback_armed(task.task_id) is False

    assert background_tasks.arm_callback(task.task_id, DESTINATION, session_key="phone-1") is True
    assert background_tasks.callback_armed(task.task_id) is True
    # Idempotent: arming again keeps exactly one authorization.
    assert background_tasks.arm_callback(task.task_id, DESTINATION, session_key="phone-1") is True

    background_tasks._mark_running(task.task_id)
    background_tasks._finish(task.task_id, SUCCEEDED, result="done")
    assert background_tasks.arm_callback(task.task_id, DESTINATION, session_key="phone-1") is False


def test_claim_callback_is_exact_once_and_only_after_settling(store, clock) -> None:
    task = enqueue("look it up", session_key="phone-1")
    assert background_tasks.arm_callback(task.task_id, DESTINATION, session_key="phone-1")
    # Still running: nothing to call about yet.
    assert background_tasks.claim_callback(task.task_id, "worker") is None
    background_tasks._mark_running(task.task_id)
    background_tasks._finish(task.task_id, FAILED, error="boom")

    assert background_tasks.claim_callback(task.task_id, "worker") == DESTINATION
    assert background_tasks.claim_callback(task.task_id, "worker") is None
    assert background_tasks.callback_armed(task.task_id) is False
    # A claimed authorization cannot be re-armed or disarmed.
    assert background_tasks.disarm_callback(task.task_id) is False


def test_cancel_and_interruption_withdraw_the_callback(store, clock) -> None:
    cancelled = enqueue("cancel me", session_key="phone-1")
    interrupted = enqueue("interrupt me", session_key="phone-1")
    for task in (cancelled, interrupted):
        assert background_tasks.arm_callback(task.task_id, DESTINATION, session_key="phone-1")

    assert cancel(cancelled.task_id) is True
    assert background_tasks.callback_armed(cancelled.task_id) is False
    assert background_tasks.claim_callback(cancelled.task_id, "worker") is None

    background_tasks._mark_running(interrupted.task_id)
    assert recover_interrupted() == 1
    assert background_tasks.callback_armed(interrupted.task_id) is False
    assert background_tasks.claim_callback(interrupted.task_id, "worker") is None


def test_disarm_callback_withdraws_only_an_unclaimed_authorization(store, clock) -> None:
    task = enqueue("look it up", session_key="phone-1")
    assert background_tasks.disarm_callback(task.task_id) is False
    assert background_tasks.arm_callback(task.task_id, DESTINATION, session_key="phone-1")
    assert background_tasks.disarm_callback(task.task_id) is True
    assert background_tasks.callback_armed(task.task_id) is False


def test_callback_logs_never_carry_the_destination_or_task_id(store, clock, task_logs) -> None:
    task = enqueue("look it up", session_key="phone-1")
    background_tasks.arm_callback(task.task_id, DESTINATION, session_key="phone-1")
    background_tasks._mark_running(task.task_id)
    background_tasks._finish(task.task_id, SUCCEEDED, result="done")
    background_tasks.claim_callback(task.task_id, "worker")

    assert DESTINATION not in task_logs.text
    assert task.task_id not in task_logs.text
    assert re.search(r"bt_[0-9a-f]{16}", task_logs.text) is None
