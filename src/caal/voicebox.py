"""Narrow client for the local Voicebox native API, never OpenAI compatibility."""

import asyncio
import io
import ipaddress
import os
import re
import wave
from dataclasses import replace
from urllib.parse import urlsplit

import httpx
from livekit import rtc
from livekit.agents import APIConnectionError, tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from .local_ollama import normalize_endpoint, resolve_local_alias


def checked_origin(value):
    normalized = normalize_endpoint(value)
    # Require a canonical origin: reject even empty query/fragment and trailing slash.
    if normalized != value:
        raise ValueError("invalid_voicebox_endpoint")
    allowed = os.environ.get(
        "CAAL_VOICEBOX_ALLOWED_ENDPOINTS",
        "http://127.0.0.1:8000,http://localhost:8000,http://host.docker.internal:8000",
    ).split(",")
    if value not in allowed or "?" in value or "#" in value or "\\" in value:
        raise ValueError("voicebox_endpoint_not_approved")
    return normalized


class VoiceboxClient:
    def __init__(self, endpoint, credential="", *, client=None, resolver=resolve_local_alias):
        self.endpoint = checked_origin(endpoint)
        if (
            not isinstance(credential, str)
            or len(credential) > 4096
            or any(ord(c) < 33 or ord(c) > 126 for c in credential)
        ):
            raise ValueError("invalid_voicebox_credential")
        self.credential = credential
        self.resolver = resolver
        self.owned = client is None
        self.client = client or httpx.AsyncClient(
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(60, connect=2),
            limits=httpx.Limits(max_connections=2),
        )

    def target(self, path):
        if path not in {
            "/health",
            "/models/status",
            "/generate/stream",
            "/openapi.json",
        } and not re.fullmatch(r"/profiles/[A-Za-z0-9_-]{1,100}", path):
            raise ValueError("invalid_voicebox_operation")
        parts = urlsplit(checked_origin(self.endpoint))
        allowed = set(os.environ.get("CAAL_VOICEBOX_ALLOWED_IPS", "127.0.0.1,::1").split(","))
        addresses = self.resolver(parts.hostname)
        if not addresses or any(
            a not in allowed or ipaddress.ip_address(a).is_link_local for a in addresses
        ):
            raise ValueError("voicebox_dns_not_approved")
        # Connect to the validated literal, not a second DNS lookup.
        address = addresses[0]
        host = "[" + address + "]" if ":" in address else address
        headers = {"Host": parts.netloc}
        if self.credential:
            headers["Authorization"] = "Bearer " + self.credential
        return f"http://{host}:{parts.port}{path}", headers

    async def json(self, path):
        url, headers = self.target(path)
        async with self.client.stream(
            "GET", url, headers=headers, follow_redirects=False
        ) as response:
            if response.status_code != 200:
                raise ValueError("voicebox_unavailable")
            content = bytearray()
            async for chunk in response.aiter_bytes():
                content.extend(chunk)
                if len(content) > 1048576:
                    raise ValueError("invalid_voicebox_response")
            import json

            return json.loads(content)

    async def aclose(self):
        if self.owned:
            await self.client.aclose()


class VoiceboxTTS(tts.TTS):
    """Voicebox synthesizes a complete WAV before transfer. No startup-streaming claim."""

    def __init__(
        self,
        *,
        endpoint,
        credential,
        profile_id,
        engine,
        model_size,
        client=None,
        fallback=None,
        total_timeout=60,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False), sample_rate=24000, num_channels=1
        )
        self.native = VoiceboxClient(endpoint, credential, client=client)
        self.profile_id, self.engine, self.model_size = profile_id, engine, model_size
        self.fallback, self.total_timeout = fallback, total_timeout

    @property
    def model(self):
        return self.engine + ":" + self.model_size

    @property
    def provider(self):
        return "voicebox-native"

    def synthesize(self, text, *, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        return VoiceboxStream(
            tts=self, input_text=text, conn_options=replace(conn_options, max_retry=0)
        )

    async def aclose(self):
        await self.native.aclose()
        if self.fallback:
            await self.fallback.aclose()


class VoiceboxStream(tts.ChunkedStream):
    async def _run(self, output_emitter):
        p = self._tts
        output_emitter.initialize(
            request_id="voicebox",
            sample_rate=24000,
            num_channels=1,
            mime_type="audio/pcm",
            frame_size_ms=20,
        )

        async def generate():
            if not 0 < len(self.input_text) <= 600:
                raise ValueError("text_too_long")
            url, headers = p.native.target("/generate/stream")
            payload = dict(
                profile_id=p.profile_id,
                text=self.input_text,
                engine=p.engine,
                model_size=p.model_size,
                language="en",
                personality=False,
                effects_chain=[],
            )
            content = bytearray()
            async with p.native.client.stream(
                "POST", url, headers=headers, json=payload, follow_redirects=False
            ) as response:
                if (
                    response.status_code != 200
                    or response.headers.get("content-type", "").split(";")[0] != "audio/wav"
                ):
                    raise ValueError("invalid_voicebox_audio")
                async for chunk in response.aiter_bytes():
                    content.extend(chunk)
                    if len(content) > 4 * 1024 * 1024:
                        raise ValueError("voicebox_audio_too_large")
            with wave.open(io.BytesIO(content), "rb") as wav:
                rate, count = wav.getframerate(), wav.getnframes()
                if (
                    wav.getnchannels() != 1
                    or wav.getsampwidth() != 2
                    or rate not in (16000, 22050, 24000, 44100, 48000)
                    or not 0 < count <= rate * 90
                ):
                    raise ValueError("unsupported_voicebox_audio")
                pcm = wav.readframes(count)
                if len(pcm) != count * 2:
                    raise ValueError("truncated_voicebox_audio")
            if rate != 24000:
                resampler = rtc.AudioResampler(input_rate=rate, output_rate=24000, num_channels=1)
                frames = resampler.push(rtc.AudioFrame(pcm, rate, 1, count)) + resampler.flush()
                pcm = b"".join(bytes(f.data) for f in frames)
            return pcm

        try:
            pcm = await asyncio.wait_for(generate(), p.total_timeout)
        except Exception:
            if p.fallback is None:
                raise APIConnectionError(
                    "Voicebox synthesis unavailable", retryable=False
                ) from None
            stream = p.fallback.synthesize(self.input_text, conn_options=self._conn_options)
            try:
                async for event in stream:
                    output_emitter.push(bytes(event.frame.data))
            finally:
                await stream.aclose()
        else:
            # Outside the fallback catch: failure/cancellation after speech never replays.
            output_emitter.push(pcm)
        output_emitter.flush()


MODEL_IDS = {
    ("qwen", "1.7B"): "qwen-tts-1.7B",
    ("qwen", "0.6B"): "qwen-tts-0.6B",
    ("qwen_custom_voice", "1.7B"): "qwen-custom-voice-1.7B",
    ("qwen_custom_voice", "0.6B"): "qwen-custom-voice-0.6B",
    ("luxtts", "1.7B"): "luxtts",
    ("chatterbox", "1.7B"): "chatterbox-tts",
    ("chatterbox_turbo", "1.7B"): "chatterbox-turbo",
    ("kokoro", "1.7B"): "kokoro",
    ("tada", "1B"): "tada-1b",
    ("tada", "3B"): "tada-3b-ml",
}


async def verify_profile(client, choice):
    model = MODEL_IDS.get((choice.engine, choice.model_size))
    if not model:
        raise ValueError("unsupported_voicebox_model")
    profile = await client.json("/profiles/" + choice.profile_id)
    if profile.get("id") != choice.profile_id or profile.get("language") != "en":
        raise ValueError("unsupported_voicebox_profile")
    kind = profile.get("voice_type")
    if kind == "preset":
        if profile.get("preset_engine") != choice.engine or not profile.get("preset_voice_id"):
            raise ValueError("incompatible_voicebox_profile")
    elif kind == "cloned":
        if (
            choice.engine not in {"qwen", "luxtts", "chatterbox", "chatterbox_turbo", "tada"}
            or not isinstance(profile.get("sample_count"), int)
            or profile["sample_count"] < 1
        ):
            raise ValueError("incompatible_voicebox_profile")
    else:
        # Designed profile execution is marked future in this Voicebox checkout.
        raise ValueError("unsupported_voicebox_profile")
    models = await client.json("/models/status")
    if not any(
        row.get("model_name") == model and row.get("downloaded") is True for row in models["models"]
    ):
        raise ValueError("voicebox_model_not_cached")
