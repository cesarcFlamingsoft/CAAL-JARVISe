"""Opt-in private Qwen PCM adapter. Kokoro remains the normal provider."""

import asyncio
from contextvars import ContextVar
from dataclasses import replace

import httpx
from livekit.agents import APIConnectionError, APIConnectOptions, tokenize, tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from .speech_request import SpeechProfile, speech_profile


class QwenTTS(tts.TTS):
    @property
    def model(self) -> str:
        return "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit"

    @property
    def provider(self) -> str:
        return "qwen-trial-mlx"

    def __init__(
        self, *, endpoint: str, token: str, client=None, fallback=None, total_timeout=60.0
    ):
        if endpoint not in {
            "http://127.0.0.1:18003",
            "http://localhost:18003",
            "http://host.docker.internal:18003",
        }:
            raise ValueError("Qwen trial requires its fixed private origin")
        if not isinstance(token, str) or len(token) < 32 or not token.isascii():
            raise ValueError("Qwen trial requires a private token")
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False), sample_rate=24000, num_channels=1
        )
        self.endpoint = endpoint
        self.token = token
        # Which language to ask this turn's speech in. ``None`` (and "en") send
        # the request body unchanged, so the approved British English profile
        # keeps exactly the request it has today.
        self._language = ContextVar(f"qwen_language_{id(self)}", default=None)
        self.fallback = fallback
        self.total_timeout = total_timeout
        self.owns_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(15, connect=2),
            trust_env=False,
            limits=httpx.Limits(
                max_connections=2, max_keepalive_connections=2, keepalive_expiry=120
            ),
        )

    @property
    def language(self):
        return self._language.get()

    @language.setter
    def language(self, value):
        self._language.set(value)

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ):
        return QwenStream(
            tts=self, input_text=text, conn_options=replace(conn_options, max_retry=0),
            profile=speech_profile.get() or SpeechProfile(self.language),
            fallback=self.fallback,
        )

    async def aclose(self):
        if self.owns_client:
            await self.client.aclose()
        if self.fallback:
            await self.fallback.aclose()


class QwenStream(tts.ChunkedStream):
    def __init__(self, *, profile, fallback, **kwargs):
        self.profile = profile
        self.fallback = fallback
        self.fallback_allowed = (
            not profile.language or profile.language == "en"
            or profile.language in getattr(fallback, "supports_language", ())
        )
        super().__init__(**kwargs)

    async def _run(self, output_emitter: tts.AudioEmitter):
        provider = self._tts
        output_emitter.initialize(
            request_id="qwen-trial",
            sample_rate=24000,
            num_channels=1,
            mime_type="audio/pcm",
            frame_size_ms=20,
        )
        emitted = False

        body = {
            "input": self.input_text,
            "model": self.profile.model,
            "voice": self.profile.voice,
            "response_format": "pcm",
        }
        # Additive: only a non-English turn carries the field, so an English
        # request is byte-identical to the one the live service answers today
        # and an older service that ignores the field still speaks English.
        language = self.profile.language
        if language and language != "en":
            body["language"] = language

        async def primary():
            nonlocal emitted
            async with provider.client.stream(
                "POST",
                provider.endpoint + "/v1/audio/speech",
                headers={"Authorization": "Bearer " + provider.token},
                json=body,
            ) as response:
                response.raise_for_status()
                if (
                    response.headers.get("content-type", "").split(";")[0] != "audio/pcm"
                    or response.headers.get("x-audio-sample-rate") != "24000"
                    or response.headers.get("x-audio-channels") != "1"
                ):
                    raise ValueError("Invalid audio format")
                pending = b""
                async for chunk in response.aiter_bytes():
                    pending += chunk
                    length = len(pending) // 2 * 2
                    if length:
                        emitted = True
                        output_emitter.push(pending[:length])
                        pending = pending[length:]
                if pending or not emitted:
                    raise ValueError("Incomplete audio")

        try:
            await asyncio.wait_for(primary(), provider.total_timeout)
        except Exception:
            if emitted or self.fallback is None or not self.fallback_allowed:
                raise APIConnectionError("Qwen synthesis failed", retryable=False) from None
            fallback = self.fallback.synthesize(
                self.input_text, conn_options=self._conn_options
            )
            try:
                async for event in fallback:
                    output_emitter.push(bytes(event.frame.data))
            finally:
                await fallback.aclose()
        output_emitter.flush()


class _SentenceAdapter(tts.StreamAdapter):
    async def aclose(self):
        await super().aclose()
        await self._wrapped_tts.aclose()


def sentence_adapter(provider):
    return _SentenceAdapter(
        tts=provider,
        sentence_tokenizer=tokenize.basic.SentenceTokenizer(
            min_sentence_len=1,
            stream_context_len=1,
            retain_format=True,
        ),
    )
