"""Speech recognition that keeps the language the server detected.

The local speech server (``mlx_audio.server``) returns a top-level ``language``
on every transcription, and ignores the ``language`` form field entirely --
both measured, see ``reports/bilingual/evidence/stt-probe-baseline.json``.

``livekit.plugins.openai.stt`` asks for ``response_format=verbose_json`` only
when the model is literally ``whisper-1``; for any other model it parses the
reply as ``openai.types.audio.Transcription`` and reads ``resp.languages``,
which this server does not send. The detected code is therefore discarded and
every transcript is tagged with the plugin's default hint, ``"en"``.

This is the same single HTTP request the plugin already makes -- no second
model, no classification call, no added latency -- with the language kept.
"""

from __future__ import annotations

import httpx
from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

#: Codes this assistant speaks. Anything else is reported as detected but the
#: reply-language policy (see ``caal.language_policy``) will not act on it.
SPOKEN = ("en", "es")


class LanguageAwareSTT(stt.STT):
    """Non-streaming Whisper transcription against an OpenAI-shaped endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        language: str | None = None,
        timeout: float = 30.0,
        connect_timeout: float = 15.0,
        transport: httpx.BaseTransport | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._url = base_url.rstrip("/") + "/v1/audio/transcriptions"
        self._model = model
        # Left unset by default so the server detects freely. Pinning it is
        # accepted but, against this server, has no effect on recognition.
        self._language = language or None
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            trust_env=False,
            transport=transport,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "local-whisper"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str | list[str]] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        pinned = self._language
        if is_given(language) and language:
            pinned = language if isinstance(language, str) else language[0]

        wav = rtc.combine_audio_frames(buffer).to_wav_bytes()
        data = {"model": self._model}
        if pinned:
            data["language"] = pinned
        try:
            response = await self._client.post(
                self._url,
                files={"file": ("file.wav", wav, "audio/wav")},
                data=data,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException:
            raise APITimeoutError() from None
        except httpx.HTTPStatusError as exc:
            raise APIStatusError(
                "speech recognition failed", status_code=exc.response.status_code
            ) from None
        except Exception as exc:
            raise APIConnectionError() from exc

        detected = payload.get("language")
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text=payload.get("text") or "",
                    language=LanguageCode(detected if isinstance(detected, str) else ""),
                )
            ],
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
