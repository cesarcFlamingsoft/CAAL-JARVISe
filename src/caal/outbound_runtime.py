"""Trusted-worker configuration for an already-dispatched outbound room."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta

from .background_tasks import is_valid_task_id
from .conversation_ledger import is_valid_conversation_id
from .handoff_context import ConversationSnapshot
from .outbound_calls import OutboundCallPolicy
from .user_scope import is_valid_user_id

# Look up a user's *current* approved callback number by opaque id; None when
# the user has none, is suspended, or is unknown.
ResolveUserDestination = Callable[[str], "str | None"]


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
        return cls(
            attempt_id=request.attempt_id,
            destination=request.destination,
            snapshot=snapshot,
            conversation_id=conversation_id,
            callback_task_id=callback_task_id,
            user_id=request.user_id,
        )


def requires_fallback_notification(category: str) -> bool:
    """Use a conservative no-message policy for every non-human verdict."""
    return category != "human"


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
