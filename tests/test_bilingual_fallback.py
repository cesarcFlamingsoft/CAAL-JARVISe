"""Stage 2: a Spanish turn is never quietly spoken by the English fallback.

`QwenTTS` falls back to Kokoro when the private service fails. Kokoro is
configured with an English voice, so for a Spanish turn that fallback produces a
confident, fluent, wrong answer — the worst available outcome, because nothing
in the system notices. These tests pin that it does not happen.
"""

import asyncio

import httpx
import pytest
from livekit.agents import APIConnectionError

from caal.qwen_tts import QwenTTS

TOKEN = "t" * 40
ENDPOINT = "http://127.0.0.1:18003"


class RecordingFallback:
    """A stand-in for the Kokoro provider that records whether it was used."""

    supports_language = ("en",)

    def __init__(self):
        self.used = []

    def synthesize(self, text, *, conn_options=None):
        self.used.append(text)
        raise AssertionError("the English fallback must not speak a Spanish turn")

    async def aclose(self):
        return None


def failing_client(status=502):
    async def handler(request):
        return httpx.Response(status, text="nope")

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def run(provider, text="Hola."):
    async def go():
        chunks = []
        stream = provider.synthesize(text)
        try:
            async for event in stream:
                chunks.append(bytes(event.frame.data))
        finally:
            await stream.aclose()
        return b"".join(chunks)

    return asyncio.run(go())


@pytest.mark.parametrize("status", [400, 500, 502, 504])
def test_a_failed_spanish_synthesis_does_not_fall_back_to_the_english_voice(status):
    fallback = RecordingFallback()
    provider = QwenTTS(
        endpoint=ENDPOINT, token=TOKEN, client=failing_client(status), fallback=fallback
    )
    provider.language = "es"
    with pytest.raises(APIConnectionError):
        run(provider)
    assert fallback.used == []


def test_an_english_turn_still_uses_the_fallback_exactly_as_before():
    used = []

    class EnglishFallback(RecordingFallback):
        def synthesize(self, text, *, conn_options=None):
            used.append(text)
            raise RuntimeError("reached")

    fallback = EnglishFallback()
    for language in (None, "en"):
        provider = QwenTTS(
            endpoint=ENDPOINT, token=TOKEN, client=failing_client(), fallback=fallback
        )
        provider.language = language
        with pytest.raises(RuntimeError):
            run(provider, "Hello.")
    assert used == ["Hello.", "Hello."]


def test_a_fallback_that_can_speak_the_language_is_still_used():
    class BilingualFallback(RecordingFallback):
        supports_language = ("en", "es")

        def synthesize(self, text, *, conn_options=None):
            self.used.append(text)
            raise RuntimeError("reached")

    fallback = BilingualFallback()
    provider = QwenTTS(endpoint=ENDPOINT, token=TOKEN, client=failing_client(), fallback=fallback)
    provider.language = "es"
    with pytest.raises(RuntimeError):
        run(provider)
    assert fallback.used == ["Hola."]
