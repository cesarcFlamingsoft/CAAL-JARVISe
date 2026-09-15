"""The opt-in TTS consumes real PCM incrementally through the LiveKit SDK."""

import asyncio

import httpx
import pytest

from caal import qwen_tts


@pytest.mark.asyncio
async def test_qwen_first_frame_precedes_response_completion_and_reuses_client():
    release = asyncio.Event()
    requests = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"\x01\x00" * 1920
            await release.wait()
            yield b"\x02\x00" * 1920

    def handler(req):
        requests.append(req)
        return httpx.Response(
            200,
            headers={
                "content-type": "audio/pcm",
                "x-audio-sample-rate": "24000",
                "x-audio-channels": "1",
            },
            stream=Body(),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = qwen_tts.QwenTTS(
            endpoint="http://127.0.0.1:18003", token="a" * 32, client=client
        )
        stream = provider.synthesize("Understood.")
        try:
            first = await asyncio.wait_for(anext(stream), 1)
            assert 0 < first.frame.samples_per_channel <= 480
            assert bytes(first.frame.data) == b"\x01\x00" * first.frame.samples_per_channel
            assert not release.is_set()
            release.set()
            frames = [first] + [x async for x in stream]
            actual = b"".join(bytes(x.frame.data) for x in frames)
            expected = b"\x01\x00" * 1920 + b"\x02\x00" * 1920
            assert actual[: len(expected)] == expected
            assert not any(actual[len(expected) :])  # SDK final marker is silence.
            async with provider.synthesize("Again.") as again:
                _ = [x async for x in again]
            assert len(requests) == 2
            assert requests[0].headers["authorization"] == "Bearer " + "a" * 32
        finally:
            release.set()
            await stream.aclose()
            await provider.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("after_audio", [False, True])
async def test_timeout_falls_back_only_before_audio_and_never_replays(after_audio):
    from types import SimpleNamespace

    from livekit import rtc
    from livekit.agents import APIConnectionError

    calls = []

    class FallbackStream:
        async def __aiter__(self):
            yield SimpleNamespace(frame=rtc.AudioFrame(b"\x03\x00" * 960, 24000, 1, 960))

        async def aclose(self):
            pass

    class Fallback:
        def synthesize(self, text, **kwargs):
            calls.append(text)
            return FallbackStream()

        async def aclose(self):
            pass

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            if after_audio:
                yield b"\x01\x00" * 1920
                await asyncio.sleep(0.02)
            raise httpx.ReadTimeout("private upstream details")

    request_count = []

    def handler(req):
        request_count.append(req)
        return httpx.Response(
            200,
            headers={
                "content-type": "audio/pcm",
                "x-audio-sample-rate": "24000",
                "x-audio-channels": "1",
            },
            stream=Body(),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = qwen_tts.QwenTTS(
            endpoint="http://127.0.0.1:18003", token="a" * 32, client=client, fallback=Fallback()
        )
        stream = provider.synthesize("Safe text.")
        try:
            if after_audio:
                with pytest.raises(APIConnectionError):
                    _ = [event async for event in stream]
                assert calls == []
            else:
                frames = [event async for event in stream]
                actual = b"".join(bytes(x.frame.data) for x in frames)
                assert actual[:1920] == b"\x03\x00" * 960
                assert not any(actual[1920:])
                assert calls == ["Safe text."]
            assert len(request_count) == 1
        finally:
            await stream.aclose()
            await provider.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers,body",
    [
        ({"content-type": "application/json"}, b"bad audio!"),
        (
            {"content-type": "audio/pcm", "x-audio-sample-rate": "22050", "x-audio-channels": "1"},
            b"\x01\x00" * 960,
        ),
        (
            {"content-type": "audio/pcm", "x-audio-sample-rate": "24000", "x-audio-channels": "1"},
            b"x",
        ),
        (
            {"content-type": "audio/pcm", "x-audio-sample-rate": "24000", "x-audio-channels": "1"},
            b"",
        ),
    ],
)
async def test_malformed_response_fails_without_retry(headers, body):
    from livekit.agents import APIConnectionError

    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(200, headers=headers, content=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = qwen_tts.QwenTTS(
            endpoint="http://127.0.0.1:18003", token="a" * 32, client=client
        )
        async with provider.synthesize("Test.") as stream:
            with pytest.raises(APIConnectionError):
                _ = [x async for x in stream]
        assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("after_audio", [False, True])
async def test_cancellation_closes_http_and_never_falls_back(after_audio):
    entered = asyncio.Event()
    closed = asyncio.Event()

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            if after_audio:
                yield b"\x01\x00" * 1920
            entered.set()
            await asyncio.sleep(10)

        async def aclose(self):
            closed.set()

    class Fallback:
        def synthesize(self, *a, **k):
            pytest.fail("cancel must not replay through fallback")

        async def aclose(self):
            pass

    def handler(req):
        return httpx.Response(
            200,
            headers={
                "content-type": "audio/pcm",
                "x-audio-sample-rate": "24000",
                "x-audio-channels": "1",
            },
            stream=Body(),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = qwen_tts.QwenTTS(
            endpoint="http://127.0.0.1:18003", token="a" * 32, client=client, fallback=Fallback()
        )
        stream = provider.synthesize("Cancel test.")
        if after_audio:
            await anext(stream)
        await asyncio.wait_for(entered.wait(), 1)
        await stream.aclose()
        assert closed.is_set()
        await provider.aclose()


@pytest.mark.asyncio
async def test_total_deadline_closes_stalled_request():
    from livekit.agents import APIConnectionError

    closed = asyncio.Event()

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(10)
            yield b"\x01\x00"

        async def aclose(self):
            closed.set()

    def handler(req):
        return httpx.Response(
            200,
            headers={
                "content-type": "audio/pcm",
                "x-audio-sample-rate": "24000",
                "x-audio-channels": "1",
            },
            stream=Body(),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = qwen_tts.QwenTTS(
            endpoint="http://127.0.0.1:18003", token="a" * 32, client=client, total_timeout=0.02
        )
        async with provider.synthesize("Deadline.") as stream:
            with pytest.raises(APIConnectionError):
                await asyncio.wait_for(anext(stream), 0.2)
        assert closed.is_set()


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://8.8.8.8:18003",
        "http://example.com:18003",
        "http://user:pass@localhost:18003",
        "http://localhost:18003/wrong",
        "http://localhost:18003?token=private",
    ],
)
def test_only_fixed_private_trial_origins_are_allowed(endpoint):
    with pytest.raises(ValueError):
        qwen_tts.QwenTTS(endpoint=endpoint, token="a" * 32)


@pytest.mark.asyncio
async def test_short_first_sentence_starts_without_waiting_for_remaining_text():
    calls = []

    def handler(req):
        import json

        calls.append(json.loads(req.content)["input"])
        return httpx.Response(
            200,
            headers={
                "content-type": "audio/pcm",
                "x-audio-sample-rate": "24000",
                "x-audio-channels": "1",
            },
            content=b"\x01\x00" * 5760,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = qwen_tts.QwenTTS(
            endpoint="http://127.0.0.1:18003", token="a" * 32, client=client
        )
        adapter = qwen_tts.sentence_adapter(provider)
        stream = adapter.stream()
        try:
            stream.push_text("Understood. I ")
            first = await asyncio.wait_for(anext(stream), 1)
            assert first.frame.samples_per_channel > 0
            assert calls == ["Understood."]
        finally:
            await stream.aclose()
            await adapter.aclose()


@pytest.mark.asyncio
async def test_metrics_identify_actual_trial_model():
    provider = qwen_tts.QwenTTS(endpoint="http://127.0.0.1:18003", token="a" * 32)
    try:
        assert provider.model == "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit"
        assert provider.provider == "qwen-trial-mlx"
    finally:
        await provider.aclose()
