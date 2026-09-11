"""Durable, supervised background work: execution and callback dispatch.

Background work used to run inside whichever LiveKit job happened to schedule
it. That made the work only as durable as a phone call: when an outbound room
was deleted and its job exited, the runner went with it, the work was recorded
as interrupted, and any callback the caller had explicitly asked for was
silently dropped. Nothing failed loudly, because nothing was left to fail.

This module owns that lifecycle instead, with no dependency on a room, a job,
or a session being alive:

* work is taken under a renewable **lease** (see :mod:`caal.background_tasks`),
  so a process that dies returns its work to the queue rather than losing it;
* a settled task's **callback authorization** is won atomically together with
  the right to announce the outcome, so exactly one owner is told exactly once,
  on exactly one channel;
* the callback is placed as an ordinary isolated outbound LiveKit job carrying
  only the opaque task id and the verified user id. The number is resolved
  server-side from that user's profile at dispatch time and re-resolved again
  by the outbound worker before any SIP participant is created. No model, no
  caller, and no stored task text can influence it;
* a hand-off that fails before any SIP participant can exist releases the
  authorization for a bounded, backed-off retry, and explicitly un-claims the
  announcement, so nothing is ever recorded as delivered that was not.

Nothing here logs task text, task ids, user ids, or phone numbers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any, Protocol

from livekit.protocol import agent_dispatch
from livekit.protocol import room as room_protocol

from . import background_tasks
from .background_tasks import (
    DEFAULT_LEASE_SECONDS,
    DEFAULT_MAX_CONCURRENCY,
    DISPATCH_SENT,
    DISPATCH_UNCERTAIN,
    FAILED,
    MAX_CALLBACK_ATTEMPTS,
    SUCCEEDED,
    BackgroundTask,
    CallbackDispatch,
)
from .outbound_calls import OutboundCallPolicy, OutboundCallRequest

logger = logging.getLogger(__name__)

DEFAULT_POLL_SECONDS = 5.0
ROOM_PREFIX = "caal-outbound-"

# Look up a user's *current* approved callback number by opaque id. None when
# the user has none, is suspended, or is unknown.
ResolveUserDestination = Callable[[str], "str | None"]
Worker = Callable[[BackgroundTask], Awaitable[object]]


class CallbackRefusedError(PermissionError):
    """The callback may not be placed. Messages never name a number or a user."""


class DispatchOutcome(StrEnum):
    """What one dispatch attempt did. Safe to count and to log."""

    SENT = "sent"
    RETRY = "retry"
    UNCERTAIN = "uncertain"
    REFUSED = "refused"
    ABANDONED = "abandoned"
    DRY_RUN = "dry-run"


def build_callback_request(
    *,
    task_id: str,
    user_id: str | None,
    destination: str | None,
    resolve_user_destination: ResolveUserDestination | None = None,
    allowed_destinations: str = "",
) -> OutboundCallRequest:
    """Build the server-side outbound request for one settled task.

    Standalone and side-effect free: it creates no room, dispatches no job and
    dials nothing, so the operational check can exercise exactly the path a
    real callback takes without anyone's phone ringing.

    The destination is never an input. For a user-bound callback it is the
    number resolved from that user's own profile right now; a stored value that
    disagrees refuses the call rather than widening it. A legacy callback with
    no user falls back to the static operator allowlist.
    """
    if not background_tasks.is_valid_task_id(task_id):
        raise CallbackRefusedError("Callback task id is malformed")
    attempt_id = secrets.token_urlsafe(18)
    if user_id is not None:
        if resolve_user_destination is None:
            raise CallbackRefusedError("No resolver for a user-bound callback")
        try:
            approved = resolve_user_destination(user_id)
        except Exception as exc:
            raise CallbackRefusedError("Could not resolve the owner's callback number") from exc
        if not approved:
            raise CallbackRefusedError("The owner has no approved callback number")
        if destination and destination != approved:
            raise CallbackRefusedError("A stored destination may not override the owner's number")
        policy = OutboundCallPolicy.for_destination(approved)
        target = approved
    else:
        if not destination:
            raise CallbackRefusedError("An unowned callback carries no destination")
        try:
            policy = OutboundCallPolicy.from_csv(allowed_destinations)
        except ValueError as exc:
            raise CallbackRefusedError("No outbound allowlist is configured") from exc
        target = destination
    try:
        return policy.authorize(
            target, attempt_id=attempt_id, callback_task_id=task_id, user_id=user_id
        )
    except (PermissionError, ValueError) as exc:
        raise CallbackRefusedError("The callback destination is not approved") from exc


class CallbackPlacer(Protocol):
    """Two-step hand-off, so a failure can be told apart from an ambiguity."""

    async def reserve(self, request: OutboundCallRequest) -> str:
        """Create the isolated room. A failure here means nothing was dispatched."""

    async def dispatch(self, room_name: str, request: OutboundCallRequest) -> None:
        """Dispatch the outbound job. A failure here may still have been received."""


class LiveKitCallbackPlacer:
    """The production placer: an ordinary isolated outbound job, nothing special.

    The room metadata carries only the attempt id. Everything private -- the
    opaque task id and the verified user id -- travels in the dispatch
    metadata, which only the trusted worker reads, exactly as the interactive
    outbound path already does.
    """

    def __init__(self, *, livekit: Any, agent_name: str) -> None:
        self._livekit = livekit
        self._agent_name = agent_name

    async def reserve(self, request: OutboundCallRequest) -> str:
        room_name = ROOM_PREFIX + request.attempt_id
        metadata = json.dumps(dict(caal_outbound=True, attempt_id=request.attempt_id))
        await self._livekit.room.create_room(
            room_protocol.CreateRoomRequest(name=room_name, metadata=metadata, empty_timeout=120)
        )
        return room_name

    async def dispatch(self, room_name: str, request: OutboundCallRequest) -> None:
        await self._livekit.agent_dispatch.create_dispatch(
            agent_dispatch.CreateAgentDispatchRequest(
                agent_name=self._agent_name,
                room=room_name,
                metadata=json.dumps(request.dispatch_metadata()),
            )
        )


class CallbackDispatcher:
    """Places the callbacks of settled work, exactly once each.

    Safe to run in more than one process: every attempt begins with an atomic
    claim, so a second dispatcher finds nothing to do rather than dialing the
    same person twice.
    """

    def __init__(
        self,
        *,
        placer: CallbackPlacer,
        claimant: str,
        resolve_user_destination: ResolveUserDestination | None = None,
        allowed_destinations: str = "",
        max_attempts: int = MAX_CALLBACK_ATTEMPTS,
        clock: Callable[[], float] = time.time,
        dry_run: bool = False,
    ) -> None:
        if not claimant:
            raise ValueError("a dispatcher needs a claimant id")
        self._placer = placer
        self._claimant = claimant
        self._resolve = resolve_user_destination
        self._allowed = allowed_destinations or ""
        self._max_attempts = max(1, int(max_attempts))
        self._clock = clock
        self._dry_run = bool(dry_run)

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    async def dispatch_due(self, *, limit: int = 10) -> dict[DispatchOutcome, int]:
        """Attempt every callback that is authorized, settled and due. Counts only."""
        now = int(self._clock())
        counts: dict[DispatchOutcome, int] = {}
        for task_id in background_tasks.due_callbacks(now=now, limit=limit):
            try:
                outcome = await self._dispatch_one(task_id, now=now)
            except Exception:
                # A dispatcher that raises would stop every other callback in
                # the queue. No exception text: it can carry a destination.
                logger.warning("callback dispatch raised unexpectedly")
                continue
            if outcome is not None:
                counts[outcome] = counts.get(outcome, 0) + 1
        return counts

    async def _dispatch_one(self, task_id: str, *, now: int) -> DispatchOutcome | None:
        claim = background_tasks.claim_callback_dispatch(task_id, self._claimant, now=now)
        if claim is None:
            return None
        if claim.user_id is not None and claim.user_id != claim.task_user_id:
            # A callback may only ever reach the owner of the work it reports.
            background_tasks.abandon_callback_dispatch(task_id, self._claimant, now=now)
            logger.warning("refused a callback whose owner is not the owner of the work")
            return DispatchOutcome.REFUSED
        try:
            request = build_callback_request(
                task_id=task_id,
                user_id=claim.user_id,
                destination=claim.destination,
                resolve_user_destination=self._resolve,
                allowed_destinations=self._allowed,
            )
        except CallbackRefusedError:
            background_tasks.abandon_callback_dispatch(task_id, self._claimant, now=now)
            logger.warning("callback refused: no approved owner-bound destination")
            return DispatchOutcome.REFUSED
        if self._dry_run:
            # The verification path: everything up to the hand-off, then put
            # the authorization back unspent -- no attempt counted against the
            # retry budget a real dispatch will need, and not due again
            # immediately, so verification mode cannot spin.
            background_tasks.release_callback_dispatch(
                task_id,
                self._claimant,
                delay_seconds=background_tasks.CALLBACK_BACKOFF_MAX_SECONDS,
                now=now,
                reset_attempts=True,
            )
            logger.info("callback dispatch verified end to end without placing a call")
            return DispatchOutcome.DRY_RUN
        try:
            room_name = await self._placer.reserve(request)
        except Exception:
            return self._retry(task_id, claim, now=now)
        try:
            await self._placer.dispatch(room_name, request)
        except Exception:
            # The dispatch was issued and may have been accepted. Retrying it
            # could ring the same person twice, so this authorization is
            # consumed and the outcome is handed back to the ordinary
            # channels; delivery is never claimed on its behalf.
            background_tasks.abandon_callback_dispatch(
                task_id, self._claimant, state=DISPATCH_UNCERTAIN, now=now
            )
            logger.warning("callback hand-off outcome is unknown; not retrying, using fallback")
            return DispatchOutcome.UNCERTAIN
        background_tasks.complete_callback_dispatch(
            task_id, self._claimant, state=DISPATCH_SENT, now=now
        )
        logger.info("background task callback dispatched as an isolated outbound job")
        return DispatchOutcome.SENT

    def _retry(self, task_id: str, claim: CallbackDispatch, *, now: int) -> DispatchOutcome:
        """Nothing was dispatched, so the authorization can safely go back.

        Retrying is silent, always: an attempt that failed before a SIP
        participant could exist is an internal event, and telling the owner
        about each one is how a single undelivered callback turned into a
        stream of chat prompts. Only the end of the budget is an outcome the
        owner is owed, and it is owed exactly once.
        """
        if claim.attempts >= self._max_attempts:
            background_tasks.abandon_callback_dispatch(task_id, self._claimant, now=now)
            background_tasks.mark_callback_notice(task_id, user_id=claim.task_user_id, now=now)
            logger.warning("callback dispatch exhausted its attempts; one terminal notice is owed")
            return DispatchOutcome.ABANDONED
        background_tasks.release_callback_dispatch(
            task_id,
            self._claimant,
            delay_seconds=background_tasks.callback_backoff_seconds(claim.attempts),
            now=now,
        )
        logger.info("callback could not be handed off; released silently for a later attempt")
        return DispatchOutcome.RETRY


class DurableWorkSupervisor:
    """Runs queued work under leases and dispatches the callbacks it settles.

    One tick is the whole job: renew what this process holds, reclaim what a
    dead process left behind, start as much new work as the concurrency cap
    allows, then place any callback that has come due. Ticks are idempotent
    and safe to run in parallel with another supervisor or with a live voice
    session, because every state change is an atomic claim in the store.
    """

    def __init__(
        self,
        *,
        worker: Worker,
        dispatcher: CallbackDispatcher | None = None,
        reminders: Any | None = None,
        worker_id: str | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        clock: Callable[[], float] = time.time,
        execute_work: bool = True,
    ) -> None:
        self._worker = worker
        self._execute_work = bool(execute_work)
        self._dispatcher = dispatcher
        # Due reminders on the channels a live room cannot serve. Same shape as
        # the callback dispatcher: an atomic claim per channel, so running this
        # beside another supervisor or a voice session stays correct.
        self._reminders = reminders
        self._worker_id = worker_id or background_tasks.new_runner_id()
        self._max_concurrency = max(1, int(max_concurrency))
        self._lease_seconds = max(2, int(lease_seconds))
        self._poll_seconds = max(0.5, float(poll_seconds))
        self._clock = clock
        self._active: dict[str, asyncio.Task[None]] = {}
        self._started_at = int(clock())
        self._last_tick_at = 0
        self._settled = 0
        self._failed = 0
        self._dispatched = 0
        self._reminders_delivered = 0

    @property
    def worker_id(self) -> str:
        """Opaque, random per-process id. It names no host and no user."""
        return self._worker_id

    @property
    def active_count(self) -> int:
        return len(self._active)

    async def tick(self) -> None:
        """One supervision pass. Never raises: the loop must outlive its work."""
        try:
            self._renew_leases()
            background_tasks.requeue_expired_leases(now=int(self._clock()))
        except Exception:
            logger.warning("durable work recovery pass failed")
        try:
            self._start_new_work()
        except Exception:
            logger.warning("durable work could not start queued work")
        if self._dispatcher is not None:
            try:
                counts = await self._dispatcher.dispatch_due()
            except Exception:
                logger.warning("durable callback dispatch pass failed")
            else:
                self._dispatched += counts.get(DispatchOutcome.SENT, 0)
        if self._reminders is not None:
            try:
                delivered = await self._reminders.deliver_due()
            except Exception:
                logger.warning("durable reminder delivery pass failed")
            else:
                self._reminders_delivered += int(delivered.get("delivered", 0))
        self._last_tick_at = int(self._clock())

    async def run(self, *, stop: asyncio.Event, wake: asyncio.Event | None = None) -> None:
        """Tick until asked to stop, waking early on an internal signal."""
        logger.info("durable work supervisor started (concurrency %d)", self._max_concurrency)
        while not stop.is_set():
            await self.tick()
            waiters = [asyncio.ensure_future(stop.wait())]
            if wake is not None:
                waiters.append(asyncio.ensure_future(wake.wait()))
            done, pending = await asyncio.wait(
                waiters, timeout=self._poll_seconds, return_when=asyncio.FIRST_COMPLETED
            )
            for waiter in pending:
                waiter.cancel()
            if wake is not None and wake.is_set():
                wake.clear()
        logger.info("durable work supervisor stopping")

    async def drain_active(self) -> None:
        """Wait for in-flight work to finish normally."""
        if self._active:
            await asyncio.gather(*list(self._active.values()), return_exceptions=True)

    async def shutdown(self) -> None:
        """Stop without losing anything: every lease goes back to the queue."""
        for task_id, aio_task in list(self._active.items()):
            background_tasks.release_lease(task_id, self._worker_id)
            aio_task.cancel()
        for aio_task in list(self._active.values()):
            try:
                await aio_task
            except (asyncio.CancelledError, Exception):
                pass
        self._active.clear()

    def status(self) -> dict[str, object]:
        """Counts and timings only. Never an id, a number, or any task text."""
        report: dict[str, object] = dict(background_tasks.queue_counts())
        report["worker_id"] = self._worker_id
        report["active"] = len(self._active)
        report["concurrency"] = self._max_concurrency
        report["settled"] = self._settled
        report["failed"] = self._failed
        report["dispatched"] = self._dispatched
        report["uptime_seconds"] = max(0, int(self._clock()) - self._started_at)
        report["last_tick_age_seconds"] = (
            max(0, int(self._clock()) - self._last_tick_at) if self._last_tick_at else -1
        )
        report["dispatch_dry_run"] = bool(self._dispatcher is not None and self._dispatcher.dry_run)
        report["dispatch_configured"] = self._dispatcher is not None
        report["reminder_delivery_configured"] = self._reminders is not None
        report["reminders_delivered"] = self._reminders_delivered
        report["executing"] = self._execute_work
        return report

    def _renew_leases(self) -> None:
        for task_id in list(self._active):
            background_tasks.renew_lease(
                task_id, self._worker_id, lease_seconds=self._lease_seconds
            )

    def _start_new_work(self) -> None:
        if not self._execute_work:
            return
        loop = asyncio.get_running_loop()
        while len(self._active) < self._max_concurrency:
            task = background_tasks.lease_next_queued(
                self._worker_id, lease_seconds=self._lease_seconds
            )
            if task is None:
                return
            aio_task = loop.create_task(self._run(task))
            self._active[task.task_id] = aio_task
            aio_task.add_done_callback(
                lambda _done, task_id=task.task_id: self._active.pop(task_id, None)
            )

    async def _run(self, task: BackgroundTask) -> None:
        logger.info("durable background task started (%d in flight)", len(self._active))
        try:
            result = await self._worker(task)
        except asyncio.CancelledError:
            # The process is going away; the work is not finished and must not
            # be reported as anything. Hand the lease back and let the next
            # supervisor resume it.
            background_tasks.release_lease(task.task_id, self._worker_id)
            raise
        except Exception as exc:
            background_tasks._finish(
                task.task_id, FAILED, error=type(exc).__name__ + ": " + str(exc)
            )
            self._failed += 1
            logger.warning("durable background task failed (%s)", type(exc).__name__)
        else:
            background_tasks._finish(
                task.task_id, SUCCEEDED, result="" if result is None else result
            )
            self._settled += 1
            logger.info("durable background task succeeded")


__all__ = [
    "DEFAULT_POLL_SECONDS",
    "CallbackDispatcher",
    "CallbackPlacer",
    "CallbackRefusedError",
    "DispatchOutcome",
    "DurableWorkSupervisor",
    "LiveKitCallbackPlacer",
    "ResolveUserDestination",
    "build_callback_request",
]
