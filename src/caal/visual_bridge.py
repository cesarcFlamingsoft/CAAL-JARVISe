"""Explicit current-view requests, scoped to one ephemeral browser voice turn."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

_TOPIC = re.compile(r"^[a-f0-9]{32}$")
_LAST_OBSERVATION_SECONDS = 5 * 60


class VisualBridge:
    def __init__(
        self,
        *,
        user: str | None,
        room: str,
        send: Callable[[dict[str, Any], str], Awaitable[None]],
    ) -> None:
        self.user, self.room = user, room
        self._send = send
        self._binding: tuple[str, str] | None = None
        self._seq = 0
        self._closed = False
        self.pending: asyncio.Future[str] | None = None
        self._command: dict[str, Any] | None = None
        # A text-only, short-lived observation lets a spoken follow-up refer to
        # this same preview without retaining the camera frame.
        self._last_description: tuple[str, float] | None = None

    def cancel(self) -> None:
        if self._command is not None and self._binding is not None:
            packet = {**self._command, "action": "vision.cancel"}
            participant = self._binding[0]

            async def send_cancel() -> None:
                try:
                    await self._send(packet, participant)
                except Exception:
                    pass

            asyncio.create_task(send_cancel())
        if self.pending is not None and not self.pending.done():
            self.pending.set_result("")
        self.pending = None
        self._command = None

    def close(self) -> None:
        self._closed = True
        self.cancel()
        self._binding = None
        self._last_description = None

    def disconnect(self, participant: str) -> None:
        if self._binding is not None and self._binding[0] == participant:
            self.close()

    def receive(self, payload: bytes, participant: str) -> None:
        if self._closed or not self.user or len(payload) > 8192:
            return
        try:
            value = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            return
        if (
            not isinstance(value, dict)
            or value.get("user") != self.user
            or value.get("room") != self.room
        ):
            return
        action = value.get("action")
        if action == "vision.ready":
            if set(value) != {"action", "user", "room", "epoch"}:
                return
            epoch = value.get("epoch")
            if not isinstance(epoch, str) or not _TOPIC.fullmatch(epoch):
                return
            if self._binding and self._binding[0] != participant:
                return
            binding = (participant, epoch)
            if binding != self._binding:
                self.cancel()
                self._binding = binding
                self._last_description = None
            return
        if action == "vision.close":
            if self._binding == (participant, value.get("epoch")):
                self.cancel()
                self._binding = None
                self._last_description = None
            return
        command = self._command
        if action != "vision.result" or command is None or self.pending is None:
            return
        if self._binding != (participant, value.get("epoch")):
            return
        if set(value) != {*command, "description"}:
            return
        if any(value.get(k) != v for k, v in command.items() if k != "action"):
            return
        description = value.get("description")
        if (
            not isinstance(description, str)
            or len(description) > 1200
            or any(ord(c) < 32 or ord(c) == 127 for c in description)
        ):
            return
        if command["expires"] < time.time() * 1000:
            self.cancel()
            return
        self._command = None
        self._last_description = (description, time.monotonic())
        if not self.pending.done():
            self.pending.set_result(description)

    def available(self) -> bool:
        """Whether this exact personal browser session has an active camera binding."""
        return not self._closed and self.user is not None and self._binding is not None

    def last_observation(self) -> str | None:
        """Return only the recent text description for this live preview."""
        if not self.available() or self._last_description is None:
            return None
        description, observed_at = self._last_description
        if time.monotonic() - observed_at > _LAST_OBSERVATION_SECONDS:
            self._last_description = None
            return None
        return description

    async def analyze(self) -> str:
        """Capture one bound camera frame and return its local description to the LLM tool."""
        if not self.available():
            raise ValueError("vision_unavailable")
        self.cancel()
        binding = self._binding
        assert binding is not None
        self._seq += 1
        command = dict(
            action="vision.analyze",
            user=self.user,
            room=self.room,
            epoch=binding[1],
            seq=self._seq,
            expires=int(time.time() * 1000) + 30000,
        )
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self.pending, self._command = future, command
        try:
            await self._send(command, binding[0])
            description = await asyncio.wait_for(asyncio.shield(future), 30)
            if not description:
                raise ValueError("vision_unavailable")
            return description
        finally:
            if self.pending is future:
                self.cancel()
