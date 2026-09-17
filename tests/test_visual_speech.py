import httpx
import pytest

from caal.visual_speech import speak_visual


@pytest.mark.asyncio
async def test_only_audio_enters_sdk_and_text_goes_to_validated_local_tts():
    calls = []

    class Session:
        async def say(self, text, *, audio, add_to_chat_ctx):
            assert text == ""
            assert add_to_chat_ctx is False
            calls.extend([frame async for frame in audio])

    def handle(request):
        assert request.url.host == "localhost"
        assert b"A mug." in request.content
        return httpx.Response(200, content=b"\0\0" * 240, headers={"content-type": "audio/pcm"})

    await speak_visual(
        Session(),
        "A mug.",
        endpoint="http://localhost:8880",
        model="kokoro",
        voice="af_heart",
        transport=httpx.MockTransport(handle),
    )
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_remote_tts_is_refused_before_network():
    with pytest.raises(ValueError):
        await speak_visual(
            None, "private", endpoint="https://example.com:443", model="x", voice="x"
        )


@pytest.mark.asyncio
async def test_selected_local_voice_is_preserved_and_cloud_provider_is_refused():
    from livekit.plugins import openai

    from caal.visual_speech import speech_target

    provider = openai.TTS(
        base_url="http://localhost:8880/v1", api_key="unused", model="kokoro", voice="af_heart"
    )
    try:
        assert speech_target(provider) == dict(
            endpoint="http://localhost:8880", model="kokoro", voice="af_heart"
        )
    finally:
        await provider.aclose()
    with pytest.raises(ValueError):
        speech_target(object())


def test_private_speech_never_gives_sdk_description_text():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "src/caal/visual_speech.py").read_text()
    assert 'session.say("", audio=audio(), add_to_chat_ctx=False)' in source
    for forbidden in (
        "logger",
        "synthesize(",
        "generate_reply",
        "conversation_ledger",
        "chat_ctx.add",
        "print(",
    ):
        assert forbidden not in source


@pytest.mark.asyncio
async def test_selected_voicebox_uses_native_local_api_without_sdk_text_or_fallback():
    import io
    import wave

    from caal.visual_speech import speech_target
    from caal.voicebox import VoiceboxTTS

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\0\0" * 240)
    captured = []

    class Session:
        async def say(self, text, *, audio, add_to_chat_ctx):
            assert text == "" and add_to_chat_ctx is False
            captured.extend([frame async for frame in audio])

    def handle(request):
        import json

        assert str(request.url) == "http://127.0.0.1:8000/generate/stream"
        payload = json.loads(request.content)
        assert payload["profile_id"] == "selected-profile"
        assert payload["text"] == "A mug."
        assert "input" not in payload
        return httpx.Response(200, content=output.getvalue(), headers={"content-type": "audio/wav"})

    provider = VoiceboxTTS(
        endpoint="http://127.0.0.1:8000",
        credential="",
        profile_id="selected-profile",
        engine="qwen",
        model_size="small",
    )
    try:
        await speak_visual(
            Session(), "A mug.", **speech_target(provider), transport=httpx.MockTransport(handle)
        )
    finally:
        await provider.aclose()
    assert len(captured) == 1
