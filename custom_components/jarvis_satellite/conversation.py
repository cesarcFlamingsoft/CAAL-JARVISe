"""HA ChatLog streaming to the enrolled non-person Jarvis conversation scope."""

import asyncio
import time
import uuid
from collections import OrderedDict
from contextlib import aclosing, suppress

from homeassistant.components.conversation import ConversationEntity, ConversationResult
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent

from . import verify_registry


async def async_setup_entry(hass, entry, async_add_entities):
    async_add_entities([JarvisConversation(entry)])


class JarvisConversation(ConversationEntity):
    _attr_name = "Jarvis Bedroom Pilot"
    _attr_supports_streaming = True

    def __init__(self, entry):
        self._attr_name = getattr(entry, "title", None) or self._attr_name
        self._attr_unique_id = entry.entry_id + "_conversation"
        self.client = entry.runtime_data
        self.scopes = OrderedDict()
        self.active = None

    @property
    def supported_languages(self):
        return ["en", "en-CA", "en-GB"]

    async def _async_handle_message(self, user_input, chat_log):
        identity = self.client.identity_data
        if self.hass is not None:
            verify_registry(self.hass, identity)
        if (
            user_input.satellite_id != identity["satellite_id"]
            or user_input.device_id != identity["device_id"]
        ):
            raise HomeAssistantError(
                "This integration is restricted to the enrolled voice satellite"
            )
        # Context.user_id and HA history are deliberately never forwarded as authority/history.
        now = time.monotonic()
        for key, value in list(self.scopes.items()):
            if value[0] < now - 300:
                del self.scopes[key]
        key = chat_log.conversation_id
        if key not in self.scopes:
            self.scopes[key] = (now, str(uuid.uuid4()))
        self.scopes.move_to_end(key)
        self.scopes[key] = (now, self.scopes[key][1])
        while len(self.scopes) > 16:
            self.scopes.popitem(last=False)
        if self.active is not None:
            previous, task = self.active
            task.cancel()
            with suppress(Exception):
                await self.client.cancel(previous)
        rid = str(uuid.uuid4())
        self.active = (rid, asyncio.current_task())
        body = {
            "text": user_input.text,
            "satellite_id": identity["satellite_id"],
            "device_id": identity["device_id"],
            "conversation_id": self.scopes[key][1],
            "request_id": rid,
        }
        answer = ""
        success = False

        async def deltas():
            nonlocal answer
            yield {"role": "assistant"}
            async with aclosing(self.client.turn(body)) as stream:
                async for text in stream:
                    answer += text
                    yield {"content": text}

        try:
            async with asyncio.timeout(95):
                async for _ in chat_log.async_add_delta_content_stream(self.entity_id, deltas()):
                    pass
            response = intent.IntentResponse(language=user_input.language)
            response.async_set_speech(answer)
            success = True
            return ConversationResult(
                response,
                chat_log.conversation_id,
                continue_conversation=answer.rstrip().endswith("?"),
            )
        finally:
            if self.active and self.active[0] == rid:
                self.active = None
            if not success:
                with suppress(Exception):
                    await self.client.cancel(rid)
