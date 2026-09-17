"""The speech server already returns the language it detected; keep it.

`livekit.plugins.openai.stt` only asks for `verbose_json` when the model is
literally `whisper-1`, so for the local `mlx-community/whisper-medium-mlx` the
detected code is parsed and then dropped, and every transcript arrives tagged
`en`. That is the single reason a Spanish turn cannot be noticed today.
"""

import httpx
import numpy as np
import pytest
from livekit import rtc
from livekit.agents import stt as stt_module

from caal.stt.language_aware import LanguageAwareSTT

MODEL = "mlx-community/whisper-medium-mlx"


def frames(seconds=0.5, sample_rate=16000):
    samples = np.zeros(int(seconds * sample_rate), dtype=np.int16)
    return [
        rtc.AudioFrame(
            data=samples.tobytes(),
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=len(samples),
        )
    ]


def transport(payload, seen):
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handle)


async def recognize(payload, seen, **kwargs):
    client = LanguageAwareSTT(
        base_url="http://speech.invalid", model=MODEL, transport=transport(payload, seen), **kwargs
    )
    try:
        return await client.recognize(frames())
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_a_spanish_transcript_keeps_the_detected_language():
    seen: list[httpx.Request] = []
    event = await recognize(
        {"text": " enciende la lámpara de la oficina.", "language": "es"}, seen
    )
    assert event.type == stt_module.SpeechEventType.FINAL_TRANSCRIPT
    alternative = event.alternatives[0]
    assert alternative.text == " enciende la lámpara de la oficina."
    assert alternative.language == "es"


@pytest.mark.asyncio
async def test_an_english_transcript_still_reports_english():
    seen: list[httpx.Request] = []
    event = await recognize({"text": " Turn on the office lamp.", "language": "en"}, seen)
    assert event.alternatives[0].text == " Turn on the office lamp."
    assert event.alternatives[0].language == "en"


@pytest.mark.asyncio
async def test_no_language_is_pinned_so_the_server_detects_freely():
    seen: list[httpx.Request] = []
    await recognize({"text": "hi", "language": "en"}, seen)
    body = seen[0].content.decode("utf-8", "replace")
    assert 'name="model"' in body
    assert MODEL in body
    assert 'name="language"' not in body
    assert str(seen[0].url) == "http://speech.invalid/v1/audio/transcriptions"


@pytest.mark.asyncio
async def test_a_missing_language_field_leaves_the_transcript_untagged():
    seen: list[httpx.Request] = []
    event = await recognize({"text": "hello"}, seen)
    assert event.alternatives[0].text == "hello"
    assert event.alternatives[0].language == ""


@pytest.mark.asyncio
async def test_an_upstream_failure_surfaces_as_a_livekit_api_error():
    from livekit.agents import APIConnectionError, APIStatusError

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = LanguageAwareSTT(
        base_url="http://speech.invalid", model=MODEL, transport=httpx.MockTransport(handle)
    )
    with pytest.raises((APIStatusError, APIConnectionError)):
        await client.recognize(frames())
    await client.aclose()


@pytest.mark.asyncio
async def test_capabilities_match_the_non_streaming_whisper_endpoint():
    client = LanguageAwareSTT(base_url="http://speech.invalid", model=MODEL)
    assert client.capabilities.streaming is False
    assert client.capabilities.interim_results is False
    await client.aclose()
