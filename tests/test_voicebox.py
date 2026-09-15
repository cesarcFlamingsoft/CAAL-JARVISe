"""Genuine Voicebox: approved origins, native WAV contract, no implicit profiles."""

import importlib.util

import pytest


def test_only_explicit_approved_origin_is_accepted(monkeypatch):
    assert importlib.util.find_spec("caal.voicebox") is not None
    from caal.voicebox import checked_origin

    monkeypatch.setenv("CAAL_VOICEBOX_ALLOWED_ENDPOINTS", "http://127.0.0.1:8000")
    assert checked_origin("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
    for bad in [
        "http://127.0.0.1:8001",
        "http://169.254.169.254:8000",
        "http://8.8.8.8:8000",
        "http://192.168.1.10:8000",
        "http://localhost:8000",
        "http://u:p@127.0.0.1:8000",
        "http://127.0.0.1:8000/",
        "http://127.0.0.1:8000?",
        "http://127.0.0.1:8000#",
        "http://127.0.0.1:8000/%2f",
        "http://127.0.0.1:8000\\@evil.com",
        " http://127.0.0.1:8000",
    ]:
        with pytest.raises(ValueError):
            checked_origin(bad)


@pytest.mark.asyncio
async def test_dns_is_revalidated_pinned_and_redirects_are_refused(monkeypatch):
    import httpx

    from caal import voicebox

    assert hasattr(voicebox, "VoiceboxClient")
    monkeypatch.setenv("CAAL_VOICEBOX_ALLOWED_IPS", "127.0.0.1")
    calls = []
    answers = ["127.0.0.1"]

    def resolve(host):
        return answers

    def handle(request):
        calls.append(request)
        assert request.url.host == "127.0.0.1"
        assert request.headers["host"] == "host.docker.internal:8000"
        return httpx.Response(302, headers={"location": "http://169.254.169.254/"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        client = voicebox.VoiceboxClient(
            "http://host.docker.internal:8000", "", client=http, resolver=resolve
        )
        with pytest.raises(ValueError):
            await client.json("/health")
        answers[:] = ["8.8.8.8"]
        with pytest.raises(ValueError):
            await client.json("/health")
        with pytest.raises(ValueError):
            await client.json("/arbitrary")
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_native_voicebox_wav_request_and_pcm_frames(monkeypatch):
    import io
    import json
    import wave

    import httpx

    from caal import voicebox

    assert hasattr(voicebox, "VoiceboxTTS")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x01\x00" * 2400)

    def handler(request):
        assert request.url.path == "/generate/stream"
        payload = json.loads(request.content)
        assert payload["profile_id"] == "chosen-profile"
        assert payload["engine"] == "kokoro"
        assert payload["model_size"] == "1.7B"
        assert payload["personality"] is False
        assert payload["text"] == "Hello."
        return httpx.Response(200, content=buffer.getvalue(), headers={"content-type": "audio/wav"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = voicebox.VoiceboxTTS(
            endpoint="http://127.0.0.1:8000",
            credential="",
            profile_id="chosen-profile",
            engine="kokoro",
            model_size="1.7B",
            client=http,
        )
        async with provider.synthesize("Hello.") as stream:
            frames = [event.frame async for event in stream]
        assert frames and all(f.sample_rate == 24000 for f in frames)
        assert b"".join(bytes(f.data) for f in frames).startswith(b"\x01\x00" * 2400)
        await provider.aclose()


@pytest.mark.asyncio
async def test_voicebox_failure_before_audio_falls_back_once_without_retry():
    from types import SimpleNamespace

    import httpx
    from livekit import rtc

    from caal.voicebox import VoiceboxTTS

    calls = []
    requests = []

    class Fallback:
        def synthesize(self, text, **kwargs):
            calls.append(text)
            return Frames()

        async def aclose(self):
            pass

    class Frames:
        async def __aiter__(self):
            yield SimpleNamespace(frame=rtc.AudioFrame(b"\x03\x00" * 960, 24000, 1, 960))

        async def aclose(self):
            pass

    def handle(req):
        requests.append(req)
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        p = VoiceboxTTS(
            endpoint="http://127.0.0.1:8000",
            credential="",
            profile_id="chosen",
            engine="kokoro",
            model_size="1.7B",
            client=http,
            fallback=Fallback(),
        )
        async with p.synthesize("Hello.") as stream:
            frames = [e.frame async for e in stream]
        assert frames and calls == ["Hello."] and len(requests) == 1
        await p.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("after_audio", [False, True])
async def test_cancellation_closes_native_response_without_fallback(after_audio):
    import asyncio
    import io
    import wave

    import httpx

    from caal.voicebox import VoiceboxTTS

    started, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x01\x00" * 48000)

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            if not after_audio:
                await release.wait()
            yield buffer.getvalue()

        async def aclose(self):
            closed.set()

    class Fallback:
        def synthesize(self, *args, **kwargs):
            raise AssertionError("Cancelled speech must never replay")

        async def aclose(self):
            pass

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, stream=Body(), headers={"content-type": "audio/wav"})
        )
    ) as http:
        p = VoiceboxTTS(
            endpoint="http://127.0.0.1:8000",
            credential="",
            profile_id="chosen",
            engine="kokoro",
            model_size="1.7B",
            client=http,
            fallback=Fallback(),
        )
        stream = p.synthesize("Hello.")
        pending = asyncio.create_task(anext(stream))
        await asyncio.wait_for(started.wait(), 1)
        if after_audio:
            first = await asyncio.wait_for(pending, 1)
            assert first.frame.samples_per_channel > 0
        await stream.aclose()
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        assert closed.is_set()
        await p.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", [16000, 22050, 44100, 48000])
async def test_native_wav_sample_rates_resample_for_kokoro_fallback(rate):
    import io
    import wave

    import httpx

    from caal.voicebox import VoiceboxTTS

    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        w.writeframes(b"\x01\x00" * (rate // 10))
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, content=b.getvalue(), headers={"content-type": "audio/wav"}
            )
        )
    ) as http:
        p = VoiceboxTTS(
            endpoint="http://127.0.0.1:8000",
            credential="",
            profile_id="chosen",
            engine="kokoro",
            model_size="1.7B",
            client=http,
        )
        async with p.synthesize("Hello.") as stream:
            frames = [e.frame async for e in stream]
        assert all(f.sample_rate == 24000 and f.num_channels == 1 for f in frames)
        assert 2300 <= sum(f.samples_per_channel for f in frames) <= 2900
        await p.aclose()
