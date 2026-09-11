"""Anti-spam policy for callback and outbound-handoff failure notifications.

The incident these tests pin down: after outgoing and callback failures the
owner received a stream of identical Telegram prompts of the generic form "The
call was not answered, so I hung up without leaving a message. Reply with one
of these: another number, retry later, or continue through chat." That text
solicited a phone number over chat -- a destination that is profile-admin
controlled only -- and it arrived once per failed attempt.

The policy proven here:

* no automated output anywhere invites an alternate number;
* a callback retried under the bounded policy says nothing at all, no matter
  how many attempts, worker restarts, or lease reclaims it takes;
* an uncertain hand-off (it may have been received) says nothing;
* a callback truly abandoned after its retry budget, or explicitly cancelled,
  yields exactly one concise, deduplicated, owner-scoped notice that names no
  task, no id and no number;
* verification (dry-run) mode never notifies;
* non-callback outbound handoff failures are silent unless an operator opts in.

Everything here uses fakes. No LiveKit API, no SIP participant, no dialing.
"""

from __future__ import annotations

import re
import time

import pytest

from caal import background_task_session, background_tasks, telegram_notify
from caal.background_task_session import (
    CALLBACK_ABANDONED_NOTICE,
    BackgroundTaskBridge,
)
from caal.background_tasks import SUCCEEDED, enqueue
from caal.durable_work import CallbackDispatcher, DispatchOutcome
from caal.outbound_runtime import (
    HANDOFF_FAILURE_NOTICE,
    handoff_failure_notice_enabled,
    requires_fallback_notification,
)

USER_A = "usr_" + "a" * 24
USER_B = "usr_" + "b" * 24
NUMBER_A = "+15551230001"
OWNER_KEY_A = "room-a"
OWNER_KEY_B = "room-b"
SECRET_TEXT = "book the Lisbon flight for Alice"

# The exact shapes that must never appear in anything automated output sends.
_FORBIDDEN = (
    re.compile(r"another number", re.I),
    re.compile(r"reply with one of these", re.I),
    re.compile(r"retry later", re.I),
    re.compile(r"continue through chat", re.I),
    re.compile(r"not answered", re.I),
    re.compile(r"[+]\d{7,}"),
)


def assert_safe(text: str) -> None:
    for pattern in _FORBIDDEN:
        assert pattern.search(text) is None, "unsafe automated text: " + repr(text)


@pytest.fixture
def store(monkeypatch, tmp_path):
    path = tmp_path / "assistant.sqlite3"
    monkeypatch.setattr(background_tasks, "STORE_PATH", path)
    return path


class FakePlacer:
    """Stand-in for the LiveKit placer, with per-step failure control."""

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


def _dispatcher(placer, *, claimant="dw_test", mapping=None, **kwargs):
    approved = mapping if mapping is not None else dict([(USER_A, NUMBER_A), (USER_B, NUMBER_A)])
    return CallbackDispatcher(
        placer=placer,
        claimant=claimant,
        resolve_user_destination=lambda user_id: approved.get(user_id),
        clock=time.time,
        **kwargs,
    )


def _settled_armed_task(*, user_id: str = USER_A, session_key: str = OWNER_KEY_A) -> str:
    task = enqueue(SECRET_TEXT, session_key=session_key, user_id=user_id)
    assert background_tasks.arm_callback(
        task.task_id, None, session_key=session_key, user_id=user_id
    )
    background_tasks._mark_running(task.task_id)
    assert background_tasks._finish(task.task_id, SUCCEEDED, result="done")
    return task.task_id


class FakeFallback:
    """The owner-scoped out-of-session channel (Telegram in production)."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def __call__(self, text: str) -> None:
        self.sent.append(text)


def _bridge(fallback, *, user_id=USER_A, session_key=OWNER_KEY_A) -> BackgroundTaskBridge:
    async def _never(request: str, context: str) -> str:
        raise AssertionError("no work should run in these tests")

    return BackgroundTaskBridge(
        execute=_never, session_key=session_key, fallback=fallback, user_id=user_id
    )


# ---------------------------------------------------------------------------
# 1. The generic prompt is gone from the codebase entirely
# ---------------------------------------------------------------------------


def test_notifier_cannot_send_the_generic_unanswered_prompt() -> None:
    """The API that produced the spam no longer exists."""
    assert not hasattr(telegram_notify.TelegramCallNotifier, "notify_unanswered")


def test_no_automated_text_invites_an_alternate_number() -> None:
    for text in (
        CALLBACK_ABANDONED_NOTICE,
        HANDOFF_FAILURE_NOTICE,
        background_task_session._FAILED_MESSAGE,
        background_task_session._INTERRUPTED_MESSAGE,
    ):
        assert_safe(text)


def test_the_abandoned_notice_exposes_nothing_private(store) -> None:
    task_id = _settled_armed_task()
    assert task_id not in CALLBACK_ABANDONED_NOTICE
    assert SECRET_TEXT not in CALLBACK_ABANDONED_NOTICE
    assert "bt_" not in CALLBACK_ABANDONED_NOTICE
    # It must not claim the owner received a call.
    assert "you answered" not in CALLBACK_ABANDONED_NOTICE.lower()
    assert "spoke" not in CALLBACK_ABANDONED_NOTICE.lower()


# ---------------------------------------------------------------------------
# 2. Retries, restarts and uncertainty are silent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeated_retries_and_restarts_notify_nobody(store) -> None:
    """Every attempt below the budget leaves the notice queue empty."""
    task_id = _settled_armed_task()
    budget = background_tasks.MAX_CALLBACK_ATTEMPTS
    placer = FakePlacer(fail_reserve=budget - 1)
    for attempt in range(budget - 1):
        # A fresh claimant each pass: a worker restart or a reclaimed lease.
        dispatcher = _dispatcher(placer, claimant="dw_restart_" + str(attempt), max_attempts=budget)
        outcome = await dispatcher._dispatch_one(task_id, now=int(time.time()) + attempt * 10000)
        assert outcome is DispatchOutcome.RETRY
        assert background_tasks.pending_callback_notices(user_id=USER_A) == []


@pytest.mark.asyncio
async def test_an_uncertain_handoff_never_notifies(store) -> None:
    """It may have been received; a notice could contradict a real call."""
    task_id = _settled_armed_task()
    dispatcher = _dispatcher(FakePlacer(fail_dispatch=1))
    outcome = await dispatcher._dispatch_one(task_id, now=int(time.time()))
    assert outcome is DispatchOutcome.UNCERTAIN
    assert background_tasks.pending_callback_notices(user_id=USER_A) == []


@pytest.mark.asyncio
async def test_dry_run_never_notifies(store) -> None:
    task_id = _settled_armed_task()
    dispatcher = _dispatcher(FakePlacer(fail_reserve=99), dry_run=True, max_attempts=1)
    now = int(time.time())
    for pass_number in range(3):
        outcome = await dispatcher._dispatch_one(task_id, now=now + pass_number * 10000)
        assert outcome is DispatchOutcome.DRY_RUN
    assert background_tasks.pending_callback_notices(user_id=USER_A) == []


# ---------------------------------------------------------------------------
# 3. Exactly one terminal notice
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exhausting_the_budget_queues_exactly_one_notice(store) -> None:
    task_id = _settled_armed_task()
    budget = background_tasks.MAX_CALLBACK_ATTEMPTS
    placer = FakePlacer(fail_reserve=budget + 5)
    now = int(time.time())
    outcomes = []
    for attempt in range(budget):
        dispatcher = _dispatcher(placer, claimant="dw_" + str(attempt), max_attempts=budget)
        outcomes.append(await dispatcher._dispatch_one(task_id, now=now + attempt * 10000))
    assert outcomes[-1] is DispatchOutcome.ABANDONED
    assert outcomes[:-1] == [DispatchOutcome.RETRY] * (budget - 1)
    assert background_tasks.pending_callback_notices(user_id=USER_A) == [task_id]


def test_a_notice_is_claimed_exactly_once(store) -> None:
    task_id = _settled_armed_task()
    assert background_tasks.mark_callback_notice(task_id, user_id=USER_A) is True
    # Re-marking is a no-op: restarts and re-entry cannot multiply the notice.
    assert background_tasks.mark_callback_notice(task_id, user_id=USER_A) is False
    assert background_tasks.claim_callback_notice(task_id, "owner-1") is True
    assert background_tasks.claim_callback_notice(task_id, "owner-2") is False
    assert background_tasks.pending_callback_notices(user_id=USER_A) == []


def test_an_explicit_cancellation_queues_one_notice(store) -> None:
    task = enqueue(SECRET_TEXT, session_key=OWNER_KEY_A, user_id=USER_A)
    assert background_tasks.arm_callback(
        task.task_id, None, session_key=OWNER_KEY_A, user_id=USER_A
    )
    assert background_tasks.disarm_callback(task.task_id, notify_owner=True) is True
    assert background_tasks.pending_callback_notices(user_id=USER_A) == [task.task_id]


def test_an_in_session_cancellation_stays_silent(store) -> None:
    """The caller was told when they asked; chat must not say it again."""
    task = enqueue(SECRET_TEXT, session_key=OWNER_KEY_A, user_id=USER_A)
    assert background_tasks.arm_callback(
        task.task_id, None, session_key=OWNER_KEY_A, user_id=USER_A
    )
    assert background_tasks.disarm_callback(task.task_id) is True
    assert background_tasks.pending_callback_notices(user_id=USER_A) == []


# ---------------------------------------------------------------------------
# 4. Delivery: owner-scoped, deduplicated, one message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_bridge_sends_one_notice_and_never_repeats_it(store) -> None:
    task_id = _settled_armed_task()
    assert background_tasks.mark_callback_notice(task_id, user_id=USER_A)
    fallback = FakeFallback()
    bridge = _bridge(fallback)
    assert await bridge.flush_callback_notices() == 1
    assert await bridge.flush_callback_notices() == 0
    assert fallback.sent == [CALLBACK_ABANDONED_NOTICE]
    assert_safe(fallback.sent[0])


@pytest.mark.asyncio
async def test_a_notice_never_crosses_to_another_user(store) -> None:
    task_id = _settled_armed_task(user_id=USER_A)
    assert background_tasks.mark_callback_notice(task_id, user_id=USER_A)
    other = FakeFallback()
    stranger = _bridge(other, user_id=USER_B, session_key=OWNER_KEY_B)
    assert await stranger.flush_callback_notices() == 0
    assert other.sent == []
    mine = FakeFallback()
    assert await _bridge(mine).flush_callback_notices() == 1
    assert mine.sent == [CALLBACK_ABANDONED_NOTICE]


# ---------------------------------------------------------------------------
# 5. Non-callback outbound handoff failures: opt-in, default off
# ---------------------------------------------------------------------------


def test_handoff_failure_notification_defaults_to_disabled(monkeypatch) -> None:
    monkeypatch.delenv("CAAL_OUTBOUND_NOTIFY_FAILED_HANDOFF", raising=False)
    assert handoff_failure_notice_enabled() is False
    for category in ("machine-vm", "machine-ivr", "machine-unavailable", "unanswered", "human"):
        assert requires_fallback_notification(category, notify_enabled=False) is False


def test_handoff_failure_notification_opt_in_keeps_human_silent(monkeypatch) -> None:
    monkeypatch.setenv("CAAL_OUTBOUND_NOTIFY_FAILED_HANDOFF", "1")
    assert handoff_failure_notice_enabled() is True
    assert requires_fallback_notification("human", notify_enabled=True) is False
    # Opted in, the machine verdicts still leave no voicemail on the call
    # itself: only this one bounded chat line is sent.
    for category in ("machine-vm", "machine-ivr", "unanswered"):
        assert requires_fallback_notification(category, notify_enabled=True) is True
    assert_safe(HANDOFF_FAILURE_NOTICE)


# ---------------------------------------------------------------------------
# 6. An unanswered callback leg returns the outcome to the ordinary channel
# ---------------------------------------------------------------------------


def test_an_unanswered_callback_leg_returns_the_outcome_without_messaging(store) -> None:
    """The leg says nothing; the result goes back to the owner-scoped channel."""
    task_id = _settled_armed_task()
    claim = background_tasks.claim_callback_dispatch(task_id, "dw_1")
    assert claim is not None
    assert background_tasks.complete_callback_dispatch(task_id, "dw_1")
    assert background_tasks.get_task(task_id).notified_at is not None
    assert background_tasks.release_callback_announcement(task_id) is True
    assert background_tasks.get_task(task_id).notified_at is None
    assert background_tasks.pending_callback_notices(user_id=USER_A) == []
