"""Private local speech: only audio, never caption text, enters the voice SDK."""

from __future__ import annotations

import asyncio
import io
import wave
from typing import Any
from urllib.parse import urlsplit

import httpx
from livekit import rtc

from .local_ollama import normalize_endpoint, resolve_local_alias


def speech_target(provider: Any) -> dict[str, Any]:
    """Use the selected local voice, refusing unsupported providers, never substituting."""
    from livekit.plugins import openai

    from .qwen_tts import QwenTTS
    from .voicebox import VoiceboxTTS

    for _ in range(4):
        wrapped = getattr(provider, "_wrapped_tts", None) or getattr(provider, "_tts", None)
        if wrapped is None:
            break
        provider = wrapped
    if isinstance(provider, VoiceboxTTS):
        return dict(
            endpoint=provider.native.endpoint,
            model=provider.engine,
            voice=provider.profile_id,
            voicebox=provider.native,
            model_size=provider.model_size,
        )
    if isinstance(provider, QwenTTS):
        return dict(
            endpoint=provider.endpoint,
            token=provider.token,
            model="qwen-trial",
            voice="jarvis-designed",
        )
    if isinstance(provider, openai.TTS):
        endpoint = str(provider._client.base_url).removesuffix("/").removesuffix("/v1")
        return dict(endpoint=endpoint, model=provider._opts.model, voice=provider._opts.voice)
    raise ValueError("private_speech_unavailable")


async def speak_visual(
    session: Any,
    description: str,
    *,
    endpoint: str,
    model: str,
    voice: str,
    token: str = "",
    voicebox: Any = None,
    model_size: str = "",
    transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    # Only the two fixed internal speech services may use Docker service DNS.
    if endpoint not in ("http://kokoro:8880", "http://speaches:8000"):
        endpoint = normalize_endpoint(endpoint)
        resolve_local_alias(urlsplit(endpoint).hostname or "")
    if not 0 < len(description) <= 1200:
        raise ValueError("private_speech_unavailable")
    pcm = bytearray()
    try:
        target = endpoint + "/v1/audio/speech"
        headers = {"Authorization": "Bearer " + token} if token else {}
        payload: dict[str, Any] = dict(
            input=description, model=model, voice=voice, response_format="pcm"
        )
        if voicebox is not None:
            # Keep the existing native client's endpoint allowlist and pinned DNS.
            target, headers = voicebox.target("/generate/stream")
            payload = dict(
                profile_id=voice,
                text=description[:600],
                engine=model,
                model_size=model_size,
                language="en",
                personality=False,
                effects_chain=[],
            )

        async def request_audio() -> None:
            async with httpx.AsyncClient(
                timeout=30, trust_env=False, transport=transport
            ) as client:
                async with client.stream(
                    "POST",
                    target,
                    follow_redirects=False,
                    headers=headers,
                    json=payload,
                ) as response:
                    formats = (
                        ("audio/wav",)
                        if voicebox is not None
                        else ("audio/pcm", "application/octet-stream")
                    )
                    if (
                        response.status_code != 200
                        or response.headers.get("content-type", "").split(";")[0] not in formats
                    ):
                        raise ValueError("private_speech_unavailable")
                    async for chunk in response.aiter_bytes():
                        pcm.extend(chunk)
                        if len(pcm) > 4 * 1024 * 1024:
                            raise ValueError("private_speech_unavailable")

        try:
            await asyncio.wait_for(request_audio(), 30)
        finally:
            payload.clear()
        if voicebox is not None:
            with wave.open(io.BytesIO(pcm), "rb") as wav:
                rate, count = wav.getframerate(), wav.getnframes()
                if (
                    wav.getnchannels() != 1
                    or wav.getsampwidth() != 2
                    or rate not in (16000, 22050, 24000, 44100, 48000)
                    or not 0 < count <= rate * 60
                ):
                    raise ValueError("private_speech_unavailable")
                pcm[:] = wav.readframes(count)
                if len(pcm) != count * 2:
                    raise ValueError("private_speech_unavailable")
            if rate != 24000:
                resampler = rtc.AudioResampler(input_rate=rate, output_rate=24000, num_channels=1)
                frames = resampler.push(rtc.AudioFrame(bytes(pcm), rate, 1, count))
                frames += resampler.flush()
                pcm[:] = b"".join(bytes(frame.data) for frame in frames)
        if len(pcm) > 24000 * 2 * 60:
            raise ValueError("private_speech_unavailable")
        description = ""
        if not pcm or len(pcm) % 2:
            raise ValueError("private_speech_unavailable")

        async def audio():
            for offset in range(0, len(pcm), 960):
                chunk = bytes(pcm[offset : offset + 960])
                yield rtc.AudioFrame(chunk, 24000, 1, len(chunk) // 2)

        handle = session.say("", audio=audio(), add_to_chat_ctx=False)
        try:
            await handle
        except asyncio.CancelledError:
            handle.interrupt(force=True)
            raise
    except Exception:
        raise ValueError("private_speech_unavailable") from None
    finally:
        pcm.clear()
        description = ""
