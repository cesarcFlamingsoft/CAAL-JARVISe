"""Explicit current-view requests, scoped to one ephemeral browser voice turn."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

_TOPIC = re.compile(r"^[a-f0-9]{32}$")
_QUESTIONS = re.compile(
    r"(?:friday[, ]+)?(?:what (?:is|s) this (?:that )?i (?:am|m) (?:handling|holding)|"
    r"what (?:am i holding(?: (?:in|on) my hand)?|(?:is|s) in my hand|"
    r"can you see|does this look like|is in front of me)|"
    r"(?:can you |could you )?tell me what i (?:am|m) holding|"
    r"(?:friday[, ]+)?(?:can you |could you )?see what i have in my hand|"
    r"(?:can you |could you |please )?(?:describe|tell me) what you (?:see|can see)|"
    r"(?:please )?(?:analyze|describe) (?:the |my )?(?:current )?(?:camera view|vision preview))"
)
OPEN_VISION = "Open Vision in Personal mode with a live camera preview, then ask again."
UNAVAILABLE = "The local visual answer is unavailable. Please try again."


def explicit_visual_question(text: str) -> bool:
    if not isinstance(text, str) or len(text) > 160:
        return False
    normalized = re.sub(r"[’']", " ", text.lower()).strip(" .?!")
    return _QUESTIONS.fullmatch(" ".join(normalized.split())) is not None


class VisualBridge:
    def __init__(
        self,
        *,
        user: str | None,
        room: str,
        send: Callable[[dict[str, Any], str], Awaitable[None]],
        speak: Callable[[str], Awaitable[None]],
    ) -> None:
        self.user, self.room = user, room
        self._send, self._speak = send, speak
        self._binding: tuple[str, str] | None = None
        self._seq = 0
        self._closed = False
        self.pending: asyncio.Future[str] | None = None
        self._command: dict[str, Any] | None = None
        self._speech_task: asyncio.Future[None] | None = None

    def cancel(self) -> None:
        if self._speech_task is not None:
            self._speech_task.cancel()
            self._speech_task = None
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
            return
        if action == "vision.close":
            if self._binding == (participant, value.get("epoch")):
                self.cancel()
                self._binding = None
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
        if not self.pending.done():
            self.pending.set_result(description)

    async def handle(self, text: str, *, company: bool = False) -> bool:
        self.cancel()
        if not explicit_visual_question(text):
            return False
        if self._closed:
            return True
        if company or not self.user or self._binding is None:
            await self._speak(OPEN_VISION)
            return True
        binding = self._binding
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
            if self.pending is future and self._binding == binding and not self._closed:
                self._speech_task = asyncio.ensure_future(self._speak(description or OPEN_VISION))
                description = ""
                await self._speech_task
                self._speech_task = None
        except asyncio.CancelledError:
            if self.pending is future:
                raise
        except Exception:
            # Fail closed; no exception or packet content is emitted.
            if self.pending is future and not self._closed:
                try:
                    await self._speak(UNAVAILABLE)
                except Exception:
                    pass
        finally:
            if self.pending is future:
                self.cancel()
        return True
