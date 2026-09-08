"""Server-side policy and request model for safe CAAL outbound calls."""

from __future__ import annotations

import hmac
import json
import re
import secrets
from dataclasses import dataclass, field
from enum import StrEnum

from livekit import api
from livekit.protocol import agent_dispatch, room

from . import conversation_ledger
from .background_tasks import is_valid_task_id
from .handoff_context import ConversationSnapshot
from .user_scope import require_user_id

_E164 = re.compile(r"^\+[1-9]\d{7,14}$")


def verify_control_token(expected: str, provided: str | None) -> bool:
    """Constant-time check for the private outbound-dial control plane."""
    if not expected or provided is None:
        return False
    return hmac.compare_digest(expected, provided)


class OutboundCallStatus(StrEnum):
    """A request state that is safe to expose to callers."""

    AUTHORIZED = "authorized"


@dataclass(frozen=True)
class OutboundCallRequest:
    """Validated outbound call request; destination stays server-side."""

    destination: str
    attempt_id: str
    status: OutboundCallStatus = OutboundCallStatus.AUTHORIZED
    # Recent conversation carried into the call. Private: it only ever appears
    # in dispatch metadata, never in room metadata or logs.
    snapshot: ConversationSnapshot | None = field(default=None, repr=False)
    # Opaque id of the live ledger conversation this call continues. Private
    # for the same reason as the snapshot: dispatch metadata only, never logs.
    conversation_id: str | None = field(default=None, repr=False)
    # Opaque id of the settled background task this call reports on. Private
    # like the ids above: dispatch metadata only, never room metadata or logs.
    callback_task_id: str | None = field(default=None, repr=False)
    # Opaque id of the verified user this call is placed for. The outbound
    # worker re-resolves that user's approved number before dialing.
    user_id: str | None = field(default=None, repr=False)

    def dispatch_metadata(self) -> dict[str, object]:
        """Return metadata private to the LiveKit agent dispatch job."""
        metadata: dict[str, object] = {
            "caal_outbound": True,
            "attempt_id": self.attempt_id,
            "destination": self.destination,
        }
        if self.snapshot is not None:
            metadata["handoff_context"] = self.snapshot.to_metadata()
        if self.conversation_id is not None:
            metadata["conversation_id"] = self.conversation_id
        if self.callback_task_id is not None:
            metadata["callback_task_id"] = self.callback_task_id
        if self.user_id is not None:
            metadata["user_id"] = self.user_id
        return metadata


@dataclass(frozen=True)
class OutboundCallPolicy:
    """Immutable E.164 destination allowlist."""

    allowed_destinations: frozenset[str]

    @classmethod
    def from_csv(cls, destinations: str) -> OutboundCallPolicy:
        parsed = frozenset(item.strip() for item in destinations.split(",") if item.strip())
        if not parsed:
            raise ValueError("At least one approved E.164 destination is required")
        invalid = [destination for destination in parsed if not _E164.fullmatch(destination)]
        if invalid:
            raise ValueError("Approved destinations must use E.164 format")
        return cls(allowed_destinations=parsed)

    @classmethod
    def for_destination(cls, destination: str) -> OutboundCallPolicy:
        """A policy that permits exactly one number: a user's approved callback number."""
        if not isinstance(destination, str) or not _E164.fullmatch(destination):
            raise ValueError("Approved destination must use E.164 format")
        return cls(allowed_destinations=frozenset({destination}))

    def authorize(
        self,
        destination: str,
        *,
        attempt_id: str = "pending",
        snapshot: ConversationSnapshot | None = None,
        conversation_id: str | None = None,
        callback_task_id: str | None = None,
        user_id: str | None = None,
    ) -> OutboundCallRequest:
        if not _E164.fullmatch(destination):
            raise ValueError("Destination must use E.164 format")
        if destination not in self.allowed_destinations:
            raise PermissionError("Destination is not approved for outbound calling")
        if callback_task_id is not None and not is_valid_task_id(callback_task_id):
            raise ValueError("Callback task id is malformed")
        owner = None if user_id is None else require_user_id(user_id)
        return OutboundCallRequest(
            destination=destination,
            attempt_id=attempt_id,
            snapshot=snapshot,
            conversation_id=conversation_id,
            callback_task_id=callback_task_id,
            user_id=owner,
        )


class OutboundCallCoordinator:
    """Create an isolated room and dispatch CAAL before any SIP dial occurs."""

    def __init__(
        self, *, policy: OutboundCallPolicy, livekit: api.LiveKitAPI, agent_name: str
    ) -> None:
        self._policy = policy
        self._livekit = livekit
        self._agent_name = agent_name

    async def start(
        self,
        destination: str,
        *,
        context: ConversationSnapshot | None = None,
        conversation_id: str | None = None,
        callback_task_id: str | None = None,
        user_id: str | None = None,
    ) -> OutboundCallRequest:
        """Authorize, reserve the ledger continuation, then create room and dispatch.

        The allowlist check always runs first: neither a ``conversation_id``
        nor a ``callback_task_id`` bypasses it, and nothing is linked for a
        refused destination. The continuation is linked as pending *before*
        dispatch so the ledger survives the origin session closing while the
        phone rings. A malformed id raises ``ValueError`` and an unknown one
        ``LookupError``, both before any room exists. With ``user_id`` the
        continuation must belong to that same user (``PermissionError``).
        """
        attempt_id = secrets.token_urlsafe(18)
        request = self._policy.authorize(
            destination,
            attempt_id=attempt_id,
            snapshot=context,
            conversation_id=conversation_id,
            callback_task_id=callback_task_id,
            user_id=user_id,
        )
        if conversation_id is not None:
            conversation_ledger.link_continuation(
                conversation_id, session_key=attempt_id, user_id=request.user_id
            )
        room_name = f"caal-outbound-{attempt_id}"
        room_metadata = json.dumps({"caal_outbound": True, "attempt_id": request.attempt_id})
        try:
            await self._livekit.room.create_room(
                room.CreateRoomRequest(name=room_name, metadata=room_metadata, empty_timeout=120)
            )
            await self._livekit.agent_dispatch.create_dispatch(
                agent_dispatch.CreateAgentDispatchRequest(
                    agent_name=self._agent_name,
                    room=room_name,
                    metadata=json.dumps(request.dispatch_metadata()),
                )
            )
        except BaseException:
            # No worker will ever claim this attempt; do not hold the ledger open.
            if conversation_id is not None:
                conversation_ledger.release_continuation(conversation_id, session_key=attempt_id)
            raise
        return request
