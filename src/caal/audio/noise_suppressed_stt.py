"""Noise-suppressed STT wrapper using DeepFilterNet.

Wraps any STT and applies real-time noise suppression to audio
before forwarding to the inner STT.

Supports two modes:
1. Local: DeepFilterNet runs in-process (requires torch + deepfilternet)
2. Remote: Calls external DeepFilterNet server (for Docker + native GPU setup)

This wrapper is stackable - can be combined with WakeWordGatedSTT:
    base_stt = openai.STT(...)
    noise_stt = NoiseSuppressedSTT(base_stt)
    wake_stt = WakeWordGatedSTT(noise_stt, ...)  # Optional
"""

from __future__ import annotations

import asyncio
import logging
import os
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
    # Remote server URL (if set, uses HTTP instead of local)
    remote_url: str | None = None


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

        # Initialize DeepFilterNet (local mode)
        self._model = None
        self._df_state = None
        self._lock = threading.Lock()
        self._initialized = False

        # Remote mode
        self._use_remote = self._config.remote_url is not None
        self._http_client = None

        if self._config.enabled:
            if self._use_remote:
                self._init_remote()
            else:
                self._init_deepfilter()

    def _init_remote(self) -> bool:
        """Initialize remote DeepFilterNet client."""
        try:
            import httpx

            self._http_client = httpx.Client(timeout=5.0)
            # Test connection
            resp = self._http_client.get(f"{self._config.remote_url}/health")
            if resp.status_code == 200:
                data = resp.json()
                logger.info(
                    f"Connected to remote DeepFilterNet server: "
                    f"MPS={data.get('mps_available', False)}"
                )
                self._initialized = True
                return True
            else:
                logger.warning(f"DeepFilterNet server returned {resp.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Could not connect to DeepFilterNet server: {e}")
            self._initialized = True
            return False

    def _init_deepfilter(self) -> bool:
        """Initialize DeepFilterNet model (local mode)."""
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

    @property
    def is_available(self) -> bool:
        """Check if noise suppression is actually available."""
        if self._use_remote:
            return self._http_client is not None
        return self._model is not None

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

        # Check for remote URL first (Docker + native GPU setup)
        remote_url = settings.get("deepfilter_url") or os.getenv("DEEPFILTER_URL")

        if remote_url:
            logger.info(f"Using remote DeepFilterNet server: {remote_url}")
            config = NoiseSuppressionConfig(
                enabled=True,
                atten_lim_db=settings.get("noise_suppression_atten_db", 100.0),
                remote_url=remote_url,
            )
            stt = cls(inner_stt, config=config)
            if stt.is_available:
                return stt
            else:
                logger.warning("Remote DeepFilterNet unavailable, falling back to local")

        # Try local DeepFilterNet
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

        if not self.is_available:
            # No noise suppression - return inner stream directly
            return inner_stream

        if self._use_remote:
            return RemoteNoiseSuppressedStream(
                stt=self,
                inner_stream=inner_stream,
                remote_url=self._config.remote_url,
                config=self._config,
                conn_options=conn_options,
            )
        else:
            return LocalNoiseSuppressedStream(
                stt=self,
                inner_stream=inner_stream,
                model=self._model,
                df_state=self._df_state,
                config=self._config,
                conn_options=conn_options,
            )

    async def aclose(self) -> None:
        if self._http_client:
            self._http_client.close()
        await self._inner.aclose()


class LocalNoiseSuppressedStream(RecognizeStream):
    """Streaming STT with local DeepFilterNet noise suppression."""

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


class RemoteNoiseSuppressedStream(RecognizeStream):
    """Streaming STT with remote DeepFilterNet noise suppression."""

    DF_SAMPLE_RATE = 48000

    def __init__(
        self,
        stt: NoiseSuppressedSTT,
        *,
        inner_stream: RecognizeStream,
        remote_url: str,
        config: NoiseSuppressionConfig,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._inner_stream = inner_stream
        self._remote_url = remote_url
        self._config = config

        # Track processing stats
        self._frames_processed = 0
        self._total_latency_ms = 0.0

    async def _run(self) -> None:
        """Main processing loop."""
        import time

        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            async def _process_audio() -> None:
                """Process and forward audio frames."""
                async for data in self._input_ch:
                    if isinstance(data, self._FlushSentinel):
                        self._inner_stream.flush()
                        continue

                    frame = data

                    # Apply noise suppression
                    start_time = time.perf_counter()
                    enhanced_frame = await self._process_frame(client, frame)
                    latency = (time.perf_counter() - start_time) * 1000

                    # Track stats (every 100 frames)
                    self._frames_processed += 1
                    self._total_latency_ms += latency
                    if self._frames_processed % 100 == 0:
                        avg_latency = self._total_latency_ms / self._frames_processed
                        logger.debug(
                            f"Remote noise suppression: {self._frames_processed} frames, "
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

    async def _process_frame(self, client, frame) -> "rtc.AudioFrame":
        """Apply noise suppression via remote server."""
        from livekit import rtc

        try:
            # Convert to numpy
            audio_int16 = np.frombuffer(frame.data, dtype=np.int16)

            # Handle multi-channel
            if frame.num_channels > 1:
                audio_int16 = audio_int16[:: frame.num_channels]

            # Resample to 48kHz if needed (server expects 48kHz)
            if frame.sample_rate != self.DF_SAMPLE_RATE:
                audio_float = audio_int16.astype(np.float32) / 32768.0
                audio_float = self._resample(
                    audio_float, frame.sample_rate, self.DF_SAMPLE_RATE
                )
                audio_int16 = (audio_float * 32768.0).clip(-32768, 32767).astype(np.int16)

            # Send to remote server
            resp = await client.post(
                f"{self._remote_url}/enhance",
                content=audio_int16.tobytes(),
                params={"atten_lim_db": self._config.atten_lim_db},
                headers={"Content-Type": "application/octet-stream"},
            )

            if resp.status_code != 200:
                logger.warning(f"DeepFilterNet server error: {resp.status_code}")
                return frame

            # Parse response
            enhanced_int16 = np.frombuffer(resp.content, dtype=np.int16)

            # Resample back
            if frame.sample_rate != self.DF_SAMPLE_RATE:
                enhanced_float = enhanced_int16.astype(np.float32) / 32768.0
                enhanced_float = self._resample(
                    enhanced_float, self.DF_SAMPLE_RATE, frame.sample_rate
                )
                enhanced_int16 = (enhanced_float * 32768.0).clip(-32768, 32767).astype(np.int16)

            # Ensure length matches
            target_len = frame.samples_per_channel
            if len(enhanced_int16) < target_len:
                enhanced_int16 = np.pad(enhanced_int16, (0, target_len - len(enhanced_int16)))
            elif len(enhanced_int16) > target_len:
                enhanced_int16 = enhanced_int16[:target_len]

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
            logger.warning(f"Remote noise suppression failed: {e}")
            return frame

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
