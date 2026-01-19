"""Noise-suppressed STT wrapper using DeepFilterNet.

Wraps any STT and applies real-time noise suppression to audio
before forwarding to the inner STT.

This wrapper is stackable - can be combined with WakeWordGatedSTT:
    base_stt = openai.STT(...)
    noise_stt = NoiseSuppressedSTT(base_stt)
    wake_stt = WakeWordGatedSTT(noise_stt, ...)  # Optional
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from livekit import rtc

from livekit.agents.stt import (
    RecognizeStream,
    SpeechEvent,
    STT,
    STTCapabilities,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import aio

logger = logging.getLogger(__name__)


@dataclass
class NoiseSuppressionConfig:
    """Configuration for noise suppression."""

    enabled: bool = True
    # DeepFilterNet attenuation limit in dB (higher = more aggressive)
    atten_lim_db: float = 100.0


class NoiseSuppressedSTT(STT):
    """STT wrapper that applies DeepFilterNet noise suppression.

    Processes audio frames through DeepFilterNet before forwarding
    to the inner STT, improving recognition in noisy environments.

    Typical latency: ~10-20ms per frame on CPU.
    """

    def __init__(
        self,
        inner_stt: STT,
        *,
        config: NoiseSuppressionConfig | None = None,
    ) -> None:
        """Initialize noise-suppressed STT.

        Args:
            inner_stt: The actual STT to forward processed audio to.
            config: Noise suppression configuration.
        """
        super().__init__(capabilities=inner_stt.capabilities)
        self._inner = inner_stt
        self._config = config or NoiseSuppressionConfig()

        # Initialize DeepFilterNet
        self._model = None
        self._df_state = None
        self._lock = threading.Lock()
        self._initialized = False

        if self._config.enabled:
            self._init_deepfilter()

    def _init_deepfilter(self) -> bool:
        """Initialize DeepFilterNet model."""
        if self._initialized:
            return self._model is not None

        try:
            from df import init_df

            logger.info("Loading DeepFilterNet model...")
            self._model, self._df_state, _ = init_df(
                post_filter=True,
                atten_lim_db=self._config.atten_lim_db,
            )
            self._initialized = True
            logger.info(
                f"DeepFilterNet initialized: atten_lim={self._config.atten_lim_db}dB"
            )
            return True

        except ImportError:
            logger.warning(
                "DeepFilterNet not installed - noise suppression disabled. "
                "Run: pip install deepfilternet"
            )
            self._initialized = True
            return False

        except Exception as e:
            logger.error(f"Failed to initialize DeepFilterNet: {e}")
            self._initialized = True
            return False

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def provider(self) -> str:
        return self._inner.provider

    @classmethod
    def create(
        cls,
        inner_stt: STT,
        settings: dict,
    ) -> STT:
        """Create noise-suppressed STT from settings.

        Returns inner_stt unchanged if noise suppression is disabled
        or unavailable.
        """
        enabled = settings.get("noise_suppression_enabled", False)

        if not enabled:
            logger.info("Noise suppression: disabled")
            return inner_stt

        # Check if DeepFilterNet is available
        try:
            import df
            logger.debug("DeepFilterNet package found")
        except ImportError:
            logger.warning(
                "Noise suppression enabled but DeepFilterNet not installed - skipping"
            )
            return inner_stt

        config = NoiseSuppressionConfig(
            enabled=True,
            atten_lim_db=settings.get("noise_suppression_atten_db", 100.0),
        )

        return cls(inner_stt, config=config)

    async def _recognize_impl(
        self,
        buffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        """Pass through to inner STT (batch mode)."""
        # For batch recognition, could process the entire buffer
        # but this adds latency - skip for now
        return await self._inner.recognize(
            buffer, language=language, conn_options=conn_options
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        """Create streaming recognition with noise suppression."""
        inner_stream = self._inner.stream(
            language=language, conn_options=conn_options
        )

        if self._model is None:
            # No noise suppression - return inner stream directly
            return inner_stream

        return NoiseSuppressedStream(
            stt=self,
            inner_stream=inner_stream,
            model=self._model,
            df_state=self._df_state,
            config=self._config,
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        await self._inner.aclose()


class NoiseSuppressedStream(RecognizeStream):
    """Streaming STT with noise suppression preprocessing."""

    # DeepFilterNet uses 48kHz internally
    DF_SAMPLE_RATE = 48000

    def __init__(
        self,
        stt: NoiseSuppressedSTT,
        *,
        inner_stream: RecognizeStream,
        model,
        df_state,
        config: NoiseSuppressionConfig,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._inner_stream = inner_stream
        self._model = model
        self._df_state = df_state
        self._config = config
        self._lock = threading.Lock()

        # Track processing stats
        self._frames_processed = 0
        self._total_latency_ms = 0.0

    async def _run(self) -> None:
        """Main processing loop."""
        import time

        async def _process_audio() -> None:
            """Process and forward audio frames."""
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    self._inner_stream.flush()
                    continue

                frame = data

                # Apply noise suppression
                start_time = time.perf_counter()
                enhanced_frame = self._process_frame(frame)
                latency = (time.perf_counter() - start_time) * 1000

                # Track stats (every 100 frames)
                self._frames_processed += 1
                self._total_latency_ms += latency
                if self._frames_processed % 100 == 0:
                    avg_latency = self._total_latency_ms / self._frames_processed
                    logger.debug(
                        f"Noise suppression: {self._frames_processed} frames, "
                        f"avg latency: {avg_latency:.1f}ms"
                    )

                # Forward to inner stream
                self._inner_stream.push_frame(enhanced_frame)

            self._inner_stream.end_input()

        async def _forward_events() -> None:
            """Forward events from inner stream."""
            async for event in self._inner_stream:
                self._event_ch.send_nowait(event)

        tasks = [
            aio.create_task(_process_audio()),
            aio.create_task(_forward_events()),
        ]

        try:
            await aio.gather(*tasks)
        finally:
            await aio.cancel_and_wait(*tasks)
            await self._inner_stream.aclose()

    def _process_frame(self, frame) -> "rtc.AudioFrame":
        """Apply noise suppression to a single frame."""
        from livekit import rtc

        try:
            # Convert to numpy
            audio_int16 = np.frombuffer(frame.data, dtype=np.int16)

            # Handle multi-channel
            if frame.num_channels > 1:
                # Take first channel for processing
                audio_int16 = audio_int16[:: frame.num_channels]

            # Convert to float32 [-1, 1]
            audio_float = audio_int16.astype(np.float32) / 32768.0

            # Resample to 48kHz if needed
            if frame.sample_rate != self.DF_SAMPLE_RATE:
                audio_float = self._resample(
                    audio_float, frame.sample_rate, self.DF_SAMPLE_RATE
                )

            # Apply DeepFilterNet
            with self._lock:
                enhanced = self._enhance_audio(audio_float)

            # Resample back
            if frame.sample_rate != self.DF_SAMPLE_RATE:
                enhanced = self._resample(
                    enhanced, self.DF_SAMPLE_RATE, frame.sample_rate
                )

            # Ensure length matches
            target_len = frame.samples_per_channel
            if len(enhanced) < target_len:
                enhanced = np.pad(enhanced, (0, target_len - len(enhanced)))
            elif len(enhanced) > target_len:
                enhanced = enhanced[:target_len]

            # Convert back to int16
            enhanced_int16 = (
                (enhanced * 32768.0).clip(-32768, 32767).astype(np.int16)
            )

            # Recreate multi-channel if needed
            if frame.num_channels > 1:
                enhanced_int16 = np.repeat(enhanced_int16, frame.num_channels)

            return rtc.AudioFrame(
                data=enhanced_int16.tobytes(),
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
                samples_per_channel=frame.samples_per_channel,
            )

        except Exception as e:
            logger.warning(f"Noise suppression failed: {e}")
            return frame

    def _enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        """Process audio through DeepFilterNet."""
        from df import enhance

        # DeepFilterNet expects (batch, samples) or (samples,)
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)

        enhanced = enhance(
            self._model,
            self._df_state,
            audio,
            atten_lim_db=self._config.atten_lim_db,
        )

        return enhanced.squeeze()

    def _resample(
        self, audio: np.ndarray, from_rate: int, to_rate: int
    ) -> np.ndarray:
        """Linear resampling."""
        if from_rate == to_rate:
            return audio

        ratio = to_rate / from_rate
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
