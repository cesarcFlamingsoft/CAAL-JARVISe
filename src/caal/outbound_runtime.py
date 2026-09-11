"""Trusted-worker configuration for an already-dispatched outbound room."""

from __future__ import annotations

import inspect
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta

from .background_tasks import is_valid_task_id
from .conversation_ledger import is_valid_conversation_id
from .handoff_context import ConversationSnapshot
from .outbound_calls import OutboundCallPolicy, is_valid_reminder_id
from .user_scope import is_valid_user_id

# Look up a user's *current* approved callback number by opaque id; None when
# the user has none, is suspended, or is unknown.
ResolveUserDestination = Callable[[str], "str | None"]


logger = logging.getLogger(__name__)


async def shutdown_job(ctx: object, reason: str) -> None:
    """End the job, whichever shape this LiveKit version gives ``ctx.shutdown``.

    ``JobContext.shutdown`` returns ``None`` in the runtime version and a
    coroutine in others (and in test doubles). Awaiting the sync one raised
    ``TypeError: object NoneType can not be used in await expression`` in the
    middle of a failed dial, so the job died with an unhandled error instead of
    shutting down cleanly. The call sites all mean the same thing -- end this
    job now -- so they say it once, here.

    A context that cannot be shut down is logged and left: the caller is
    already on its way out, and a second failure must not replace the first.
    The reason is a fixed phrase from the caller, never anything the user said.
    """
    shutdown = getattr(ctx, "shutdown", None)
    if not callable(shutdown):
        logger.warning("The job context has no shutdown; leaving the job to end on its own")
        return
    try:
        result = shutdown(reason)
        if inspect.isawaitable(result):
            await result
    except Exception as exc:  # noqa: BLE001 - the job is ending either way
        logger.warning("Could not shut the job down cleanly (%s)", type(exc).__name__)


@dataclass(frozen=True)
class OutboundRoomConfig:
    attempt_id: str
    destination: str = field(repr=False)
    # Conversation carried over from the session that asked for the call.
    # Only ever surfaced to the LLM after AMD confirms a human answered.
    snapshot: ConversationSnapshot | None = field(default=None, repr=False)
    # Opaque id of the ledger conversation to hydrate from instead. Same rule:
    # the ledger is only ever read after AMD confirms a human answered.
    conversation_id: str | None = field(default=None, repr=False)
    # Opaque id of the settled background task this call was placed to report.
    # Its outcome is spoken only after AMD confirms a human answered.
    callback_task_id: str | None = field(default=None, repr=False)
    # Opaque id of the local reminder this call exists to deliver. Its words
    # are read from the store, for this same owner, only after AMD confirms a
    # human answered.
    reminder_id: str | None = field(default=None, repr=False)
    # Opaque id of the verified user this call is for. When present, the
    # destination was re-resolved from that user's profile, not an allowlist.
    user_id: str | None = field(default=None, repr=False)

    @property
    def carries_continuation(self) -> bool:
        """Whether this call continues an earlier conversation by any means."""
        return self.snapshot is not None or self.conversation_id is not None

    @property
    def is_callback(self) -> bool:
        """Whether this call exists to report a finished background task."""
        return self.callback_task_id is not None

    @property
    def is_reminder(self) -> bool:
        """Whether this call exists to deliver one due reminder of its owner."""
        return self.reminder_id is not None

    @classmethod
    def from_dispatch_metadata(
        cls,
        metadata: str,
        *,
        allowed_destinations: str,
        resolve_user_destination: ResolveUserDestination | None = None,
    ) -> OutboundRoomConfig | None:
        """Parse and re-authorize a dispatched outbound job before anything is dialed.

        A job for a verified user (``user_id`` in the metadata) is authorized
        against that user's *current* approved number, looked up through
        ``resolve_user_destination`` at this moment: a number that was cleared
        or changed since dispatch, a suspended user, or a missing resolver all
        refuse the call (``PermissionError``). Only a legacy job without a user
        falls back to the static allowlist.
        """
        try:
            raw = json.loads(metadata or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("Outbound room metadata is invalid") from exc
        if raw.get("caal_outbound") is not True:
            return None
        attempt_id = raw.get("attempt_id")
        destination = raw.get("destination")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("Outbound room lacks an attempt ID")
        if not isinstance(destination, str):
            raise ValueError("Outbound room lacks a destination")
        user_id = raw.get("user_id")
        if user_id is not None:
            if not is_valid_user_id(user_id):
                raise ValueError("Outbound room carries a malformed user id")
            if resolve_user_destination is None:
                raise PermissionError("Cannot revalidate the user's callback number")
            approved = resolve_user_destination(user_id)
            if approved is None:
                raise PermissionError("The user has no approved callback number")
            policy = OutboundCallPolicy.for_destination(approved)
        else:
            policy = OutboundCallPolicy.from_csv(allowed_destinations)
        request = policy.authorize(destination, attempt_id=attempt_id, user_id=user_id)
        snapshot = None
        raw_context = raw.get("handoff_context")
        if raw_context is not None:
            # Malformed context rejects the whole job: a dispatch that carries a
            # bad snapshot did not come from the trusted coordinator.
            snapshot = ConversationSnapshot.from_metadata(raw_context)
        conversation_id = raw.get("conversation_id")
        if conversation_id is not None and not is_valid_conversation_id(conversation_id):
            # Same reasoning as a bad snapshot: not from the trusted coordinator.
            raise ValueError("Outbound room carries a malformed conversation id")
        callback_task_id = raw.get("callback_task_id")
        if callback_task_id is not None and not is_valid_task_id(callback_task_id):
            raise ValueError("Outbound room carries a malformed callback task id")
        reminder_id = raw.get("reminder_id")
        if reminder_id is not None and not is_valid_reminder_id(reminder_id):
            # Same reasoning as the ids above: not from the trusted dispatcher.
            raise ValueError("Outbound room carries a malformed reminder id")
        return cls(
            attempt_id=request.attempt_id,
            destination=request.destination,
            snapshot=snapshot,
            conversation_id=conversation_id,
            callback_task_id=callback_task_id,
            reminder_id=reminder_id,
            user_id=request.user_id,
        )


# Operators opt in to a chat line about a non-callback outbound hand-off that
# reached no human. Default off: a hand-off the user themselves asked for in
# the moment does not need a message after the fact, and the generic prompt
# that used to be sent here arrived once per failure and asked the user to
# supply another phone number over chat.
HANDOFF_NOTICE_ENV = "CAAL_OUTBOUND_NOTIFY_FAILED_HANDOFF"

# What that opt-in line says. No number, no invitation to give one, no claim
# that a call took place, and nothing about what the call was for.
HANDOFF_FAILURE_NOTICE = (
    "JARVIS: I could not complete an outbound call I was asked to place, and I left no message."
)


def handoff_failure_notice_enabled(value: str | None = None) -> bool:
    """Whether the operator turned the outbound hand-off notice on. Default off."""
    raw = os.getenv(HANDOFF_NOTICE_ENV, "") if value is None else value
    return (raw or "").strip().lower() in ("1", "true", "yes", "on")


def requires_fallback_notification(category: str, *, notify_enabled: bool = False) -> bool:
    """Whether a non-callback outbound hand-off may send its one chat line.

    The call itself stays silent on every non-human verdict exactly as before:
    voicemail, an automated menu and a line that cannot take a message are all
    hung up on without speaking. This decides only whether the operator asked
    to be told about it afterwards, and the answer is no unless they opted in.
    """
    return bool(notify_enabled) and category != "human"


def call_timeouts(ringing_seconds: str, max_duration_seconds: str) -> tuple[timedelta, timedelta]:
    """Parse bounded SIP timeouts for the protobuf API."""
    try:
        ringing = int(ringing_seconds)
        duration = int(max_duration_seconds)
    except ValueError as exc:
        raise ValueError("Outbound call timeouts must be integer seconds") from exc
    if not 5 <= ringing <= 120:
        raise ValueError("Outbound ringing timeout must be between 5 and 120 seconds")
    if not 30 <= duration <= 3600:
        raise ValueError("Outbound max duration must be between 30 and 3600 seconds")
    return timedelta(seconds=ringing), timedelta(seconds=duration)
