"""Native streaming input/output; no provider file cache or HA cache mutation."""

import asyncio
import re
import struct
from contextlib import aclosing

from homeassistant.components.tts import TextToSpeechEntity, TTSAudioRequest, TTSAudioResponse
from homeassistant.exceptions import HomeAssistantError

from .client import MAX_AUDIO, MAX_TEXT

# Streaming WAV uses unknown RIFF/data lengths, as supported by ffmpeg.
WAV_HEADER = struct.pack(
    "<4sI4s4sIHHIIHH4sI",
    b"RIFF",
    0xFFFFFFFF,
    b"WAVE",
    b"fmt ",
    16,
    1,
    1,
    24000,
    48000,
    2,
    16,
    b"data",
    0xFFFFFFFF,
)


async def async_setup_entry(hass, entry, async_add_entities):
    async_add_entities([JarvisTTS(entry)])


class JarvisTTS(TextToSpeechEntity):
    _attr_name = "FRIDAY Bedroom Pilot Voice"
    _attr_default_language = "en"
    _attr_supported_languages = ["en", "en-CA", "en-GB"]

    def __init__(self, entry):
        if getattr(entry, "title", None):
            self._attr_name = entry.title + " Voice"
        self._attr_unique_id = entry.entry_id + "_tts"
        self.client = entry.runtime_data
        self._busy = False

    async def async_speak(
        self, media_player_entity_id, message, cache=False, language=None, options=None
    ):
        # This public provider-local hook can force no disk caching for speak.
        await super().async_speak(media_player_entity_id, message, False, language, options)

    async def async_stream_tts_audio(self, request: TTSAudioRequest) -> TTSAudioResponse:
        if request.language not in self.supported_languages or request.options:
            raise HomeAssistantError("Unsupported pilot voice options")

        async def generate():
            if self._busy:
                raise HomeAssistantError("Pilot voice busy")
            self._busy = True
            try:
                async with asyncio.timeout(90), aclosing(request.message_gen):
                    yield WAV_HEADER
                    pending = ""
                    text_size = audio_size = 0
                    async for chunk in request.message_gen:
                        if not isinstance(chunk, str):
                            raise HomeAssistantError("Invalid text stream")
                        text_size += len(chunk)
                        if text_size > MAX_TEXT:
                            raise HomeAssistantError("Pilot text limit exceeded")
                        pending += chunk
                        while pending:
                            match = re.search(r"[.!?](?:\s|$)", pending)
                            cut = (
                                min(match.end(), 240)
                                if match
                                else (240 if len(pending) >= 240 else 0)
                            )
                            if not cut:
                                break
                            sentence, pending = pending[:cut], pending[cut:]
                            async with aclosing(self.client.audio(sentence)) as audio:
                                async for data in audio:
                                    audio_size += len(data)
                                    if audio_size > MAX_AUDIO:
                                        raise HomeAssistantError("Pilot audio limit exceeded")
                                    yield data
                    if pending.strip():
                        async with aclosing(self.client.audio(pending)) as audio:
                            async for data in audio:
                                audio_size += len(data)
                                if audio_size > MAX_AUDIO:
                                    raise HomeAssistantError("Pilot audio limit exceeded")
                                yield data
            finally:
                self._busy = False

        return TTSAudioResponse("wav", generate())
