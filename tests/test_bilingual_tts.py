"""Per-request language on the way out to speech synthesis.

The Qwen trial protocol has no language field today and the private service
hard-codes `lang_code="English"`. This adds the field on the client side so a
Spanish turn is *asked* for in Spanish, while an English turn's request body
stays exactly what it is now -- the live British profile must not shift.
"""

import httpx
import pytest
from livekit.agents import APIConnectOptions

from caal.language_policy import EN, ES
from caal.qwen_tts import QwenTTS, sentence_adapter
from caal.tts_selection import set_language

ENDPOINT = "http://127.0.0.1:18003"
TOKEN = "t" * 40
PCM_HEADERS = {
    "content-type": "audio/pcm",
    "x-audio-sample-rate": "24000",
    "x-audio-channels": "1",
}


def client(seen):
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"\x00\x01" * 480, headers=PCM_HEADERS)

    return httpx.AsyncClient(transport=httpx.MockTransport(handle), trust_env=False)


async def speak(text, seen, *, language=None):
    tts = QwenTTS(endpoint=ENDPOINT, token=TOKEN, client=client(seen))
    if language is not None:
        tts.language = language
    stream = tts.synthesize(text, conn_options=APIConnectOptions(max_retry=0, timeout=5))
    try:
        async for _ in stream:
            pass
    finally:
        await stream.aclose()
        await tts.aclose()


@pytest.mark.asyncio
async def test_an_english_turn_sends_the_request_body_it_sends_today():
    seen: list[httpx.Request] = []
    await speak("All set, sir.", seen, language=EN)
    import json

    body = json.loads(seen[0].content)
    assert body == {
        "input": "All set, sir.",
        "model": "qwen-trial",
        "voice": "jarvis-designed",
        "response_format": "pcm",
    }


@pytest.mark.asyncio
async def test_a_default_instance_with_no_language_is_also_unchanged():
    seen: list[httpx.Request] = []
    await speak("All set, sir.", seen)
    import json

    assert "language" not in json.loads(seen[0].content)


@pytest.mark.asyncio
async def test_a_spanish_turn_asks_for_spanish():
    seen: list[httpx.Request] = []
    await speak("Listo, señor.", seen, language=ES)
    import json

    body = json.loads(seen[0].content)
    assert body["language"] == "es"
    assert body["voice"] == "jarvis-designed"
    assert body["input"] == "Listo, señor."


def test_set_language_reaches_a_provider_behind_the_sentence_adapter():
    tts = QwenTTS(endpoint=ENDPOINT, token=TOKEN, client=client([]))
    adapted = sentence_adapter(tts)
    set_language(adapted, ES)
    assert tts.language == ES
    set_language(adapted, EN)
    assert tts.language == EN


def test_set_language_is_a_no_op_for_a_provider_that_has_no_language():
    class Plain:
        pass

    plain = Plain()
    set_language(plain, ES)  # must not raise
    assert not hasattr(plain, "language")
