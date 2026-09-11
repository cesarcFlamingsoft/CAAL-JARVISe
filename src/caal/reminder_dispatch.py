"""Delivering a due reminder on the channels a live room cannot serve.

Speech is announced by a session of the owner. Telegram and a call are not: the
room that created the reminder is usually long gone by the time it comes due,
so they are carried out here, by the same supervised worker that dispatches
background-task callbacks, and they survive restarts.

Every decision is made again at delivery time, server-side, from the stored
owner alone:

* the configured Telegram chat is an operator channel bound to exactly one
  profile. A reminder of any other owner is refused, never delivered there;
* a call resolves that owner own approved number now. No number is stored on a
  reminder, no caller ever supplies one, and a profile without an approved
  number is refused rather than guessed.

Nothing is recorded as delivered until the channel accepted it. A hand-off that
failed before anything could have been received goes back for a bounded retry;
one that may have been received is never retried and is left failed rather than
claimed. One channel neither suppresses nor re-sends another.

No log line here carries a title, a row id, an owner or a number.
"""

from __future__ import annotations

import logging
import secrets
import time
from collections.abc import Awaitable, Callable
from typing import Protocol

from .outbound_calls import OutboundCallPolicy, OutboundCallRequest
from .tools import reminder_delivery
from .tools.reminder_delivery import (
    CALL,
    DELIVERED,
    FAILED,
    PENDING,
    TELEGRAM,
    Delivery,
    reminder_message,
)

logger = logging.getLogger(__name__)

REFUSED = "refused"
RETRY = "retry"
UNCERTAIN = "uncertain"
DRY_RUN = "dry-run"
UNAVAILABLE = "unavailable"

#: A channel this process cannot serve at all waits instead of burning its
#: budget: the reminder is still true, this worker just is not wired for it.
UNAVAILABLE_DELAY_SECONDS = 900

SendTelegram = Callable[[str], Awaitable[None]]
ResolveUserDestination = Callable[[str], "str | None"]


class CallPlacer(Protocol):
    """Two-step hand-off, so a failure can be told apart from an ambiguity."""

    async def reserve(self, request: OutboundCallRequest) -> str: ...

    async def dispatch(self, room_name: str, request: OutboundCallRequest) -> None: ...


class ReminderDispatcher:
    """Delivers the due reminders of every owner on the channels it can serve.

    Safe in more than one process: every attempt begins with an atomic, leased
    claim, so a second dispatcher finds nothing to do rather than messaging or
    calling the same person twice.
    """

    def __init__(
        self,
        *,
        claimant: str,
        send_telegram: SendTelegram | None = None,
        placer: CallPlacer | None = None,
        resolve_user_destination: ResolveUserDestination | None = None,
        clock: Callable[[], float] = time.time,
        dry_run: bool = False,
    ) -> None:
        if not claimant:
            raise ValueError("a reminder dispatcher needs a claimant id")
        self._claimant = claimant
        self._telegram = send_telegram
        self._placer = placer
        self._resolve = resolve_user_destination
        self._clock = clock
        self._dry_run = bool(dry_run)

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    async def deliver_due(self, limit: int = 10, now: int | None = None) -> dict[str, int]:
        """Attempt every due, unsettled channel once. Counts only, never content."""
        moment = int(self._clock()) if now is None else int(now)
        counts: dict[str, int] = {}
        for delivery in reminder_delivery.claim_due(self._claimant, now=moment, limit=limit):
            try:
                outcome = await self._deliver_one(delivery, now=moment)
            except Exception:
                # A raising channel must not stop the queue, and no exception
                # text is logged: it can carry a destination or a title.
                logger.warning("reminder delivery raised unexpectedly")
                outcome = self._retry(delivery, now=moment)
            if outcome:
                counts[outcome] = counts.get(outcome, 0) + 1
        return counts

    async def _deliver_one(self, delivery: Delivery, *, now: int) -> str:
        if delivery.channel == TELEGRAM:
            return await self._deliver_telegram(delivery, now=now)
        if delivery.channel == CALL:
            return await self._deliver_call(delivery, now=now)
        return self._refuse(delivery, now=now, why="an unknown channel")

    # --- telegram ---------------------------------------------------------------------

    async def _deliver_telegram(self, delivery: Delivery, *, now: int) -> str:
        owner = reminder_delivery.telegram_owner()
        if delivery.user_id and delivery.user_id != owner:
            return self._refuse(delivery, now=now, why="a chat bound to another profile")
        if self._telegram is None or not reminder_delivery.telegram_configured():
            return self._unavailable(delivery, now=now)
        await self._telegram(reminder_message(delivery.title))
        # Only now: the message left this process and the channel accepted it.
        reminder_delivery.mark_delivered(delivery.delivery_id, self._claimant, now=now)
        logger.info("delivered one due reminder on its message channel")
        return DELIVERED

    # --- the call ----------------------------------------------------------------------

    async def _deliver_call(self, delivery: Delivery, *, now: int) -> str:
        if self._resolve is None or not delivery.user_id:
            return self._refuse(delivery, now=now, why="no owner-bound destination")
        try:
            approved = self._resolve(delivery.user_id)
        except Exception:
            approved = None
        if not approved:
            return self._refuse(delivery, now=now, why="no approved number on the profile")
        try:
            request = OutboundCallPolicy.for_destination(approved).authorize(
                approved,
                attempt_id=secrets.token_urlsafe(18),
                reminder_id=delivery.reminder_id,
                user_id=delivery.user_id,
            )
        except (PermissionError, ValueError):
            return self._refuse(delivery, now=now, why="an unapproved destination")
        if self._placer is None:
            return self._unavailable(delivery, now=now)
        if self._dry_run:
            # Everything up to the hand-off, then the claim goes back unspent:
            # verification must not cost a real attempt, and no phone rings.
            self._park(delivery, now=now)
            logger.info("reminder call verified end to end without placing a call")
            return DRY_RUN
        try:
            room_name = await self._placer.reserve(request)
        except Exception:
            return self._retry(delivery, now=now)
        try:
            await self._placer.dispatch(room_name, request)
        except Exception:
            # It may have been accepted. Retrying could ring the same person
            # twice, so the channel stops here, failed rather than claimed.
            reminder_delivery.fail(delivery.delivery_id, self._claimant, now=now)
            logger.warning("reminder call hand-off outcome is unknown; not retrying")
            return UNCERTAIN
        reminder_delivery.mark_delivered(delivery.delivery_id, self._claimant, now=now)
        logger.info("dispatched one due reminder as an isolated outbound job")
        return DELIVERED

    # --- outcomes -----------------------------------------------------------------------

    def _park(self, delivery: Delivery, *, now: int) -> None:
        """Put a claim back unspent, and not due again immediately."""
        reminder_delivery.release(
            delivery.delivery_id,
            self._claimant,
            delay_seconds=UNAVAILABLE_DELAY_SECONDS,
            now=now,
            reset_attempts=True,
        )

    def _unavailable(self, delivery: Delivery, *, now: int) -> str:
        """This process is not wired for the channel. Pending is the truthful state."""
        self._park(delivery, now=now)
        logger.info("a reminder channel is not wired in this process; left pending")
        return UNAVAILABLE

    def _retry(self, delivery: Delivery, *, now: int) -> str:
        """Nothing was delivered, so the claim goes back for a bounded retry."""
        state = reminder_delivery.release(
            delivery.delivery_id,
            self._claimant,
            delay_seconds=reminder_delivery.backoff_seconds(delivery.attempts),
            now=now,
        )
        if state == FAILED:
            return FAILED
        logger.info("a reminder channel could not be handed off; released for a later attempt")
        return RETRY if state == PENDING else ""

    def _refuse(self, delivery: Delivery, *, now: int, why: str) -> str:
        """Permanently not allowed. Left failed, truthfully, and never retried."""
        reminder_delivery.fail(delivery.delivery_id, self._claimant, now=now)
        logger.warning("refused a reminder channel: %s", why)
        return REFUSED


__all__ = [
    "DRY_RUN",
    "REFUSED",
    "RETRY",
    "UNAVAILABLE",
    "UNCERTAIN",
    "CallPlacer",
    "ReminderDispatcher",
    "reminder_message",
]
