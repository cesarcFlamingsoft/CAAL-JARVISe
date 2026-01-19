"""TV Audio Rejection System.

Implements multi-layer audio analysis to distinguish live speech from TV/playback:

1. **Spectral Analysis**: TV audio has codec artifacts, bandwidth limiting
2. **Dynamic Range**: TV audio is compressed (lower crest factor)
3. **Temporal Patterns**: TV is continuous, live speech has natural pauses
4. **Liveness Scoring**: Combines multiple signals to score authenticity

Based on research from commercial voice assistants (Alexa, Siri, Google).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TVRejectionConfig:
    """Configuration for TV rejection system."""

    enabled: bool = True

    # Energy gate (first filter - reject quiet/distant sounds)
    energy_threshold_db: float = -35.0  # RMS threshold

    # Crest factor (peak-to-RMS ratio) - TV audio is compressed
    # Note: Live speech typically 3-6, TV typically 1.5-2.5
    # Using 1.5 to be very permissive - only reject heavily compressed audio
    min_crest_factor: float = 1.5

    # Spectral analysis
    min_spectral_rolloff_hz: float = 3000.0  # TV often bandwidth-limited
    min_spectral_variance: float = 0.15  # Live speech has more variation

    # Temporal analysis (audio continuity)
    max_continuous_speech_sec: float = 10.0  # TV talks continuously
    silence_reset_sec: float = 1.0  # Reset continuity after silence

    # Liveness scoring thresholds
    # Using 0.15 to be very permissive - only reject obviously TV-like audio
    min_liveness_score: float = 0.15  # Minimum score to pass (0-1)

    # Buffered detection (require multiple consecutive passes)
    required_consecutive_passes: int = 4  # Frames that must pass
    analysis_window_sec: float = 0.5  # Window for spectral analysis

    # Debug logging
    debug_logging: bool = False  # Log every frame's features


@dataclass
class AudioFeatures:
    """Extracted audio features for analysis."""

    # Energy features
    rms_db: float = -100.0
    peak_db: float = -100.0
    crest_factor: float = 0.0  # peak / RMS

    # Spectral features
    spectral_rolloff: float = 0.0  # Hz where 85% of energy is below
    spectral_centroid: float = 0.0  # "Center of mass" of spectrum
    spectral_variance: float = 0.0  # Variance of spectral centroid over time
    spectral_flatness: float = 0.0  # How "noisy" vs "tonal" (0-1)

    # Temporal features
    zero_crossing_rate: float = 0.0
    continuous_speech_sec: float = 0.0

    # Computed scores
    liveness_score: float = 0.0


class TVRejectionFilter:
    """Multi-layer filter to reject TV audio and pass live speech.

    Uses spectral analysis, dynamic range detection, and temporal patterns
    to distinguish live human speech from TV/playback audio.
    """

    def __init__(self, config: TVRejectionConfig | None = None, sample_rate: int = 16000):
        self.config = config or TVRejectionConfig()
        self.sample_rate = sample_rate

        # Buffers for temporal analysis
        self._spectral_history: deque[float] = deque(maxlen=50)  # ~1 sec at 20ms frames
        self._speech_start_time: float | None = None
        self._last_speech_time: float = 0.0
        self._consecutive_passes: int = 0

        # Statistics for adaptive thresholds
        self._recent_crest_factors: deque[float] = deque(maxlen=100)
        self._recent_liveness_scores: deque[float] = deque(maxlen=100)

        self._frame_count = 0  # For periodic logging
        logger.info(
            f"TVRejectionFilter initialized: "
            f"energy_threshold={self.config.energy_threshold_db}dB, "
            f"min_crest_factor={self.config.min_crest_factor}, "
            f"min_liveness={self.config.min_liveness_score}, "
            f"consecutive_passes={self.config.required_consecutive_passes}"
        )

    def analyze(self, audio: np.ndarray) -> AudioFeatures:
        """Extract features from audio for TV rejection analysis.

        Args:
            audio: Audio samples (int16 or float32)

        Returns:
            AudioFeatures with all computed metrics
        """
        # Normalize to float32 [-1, 1]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        features = AudioFeatures()

        # === Energy Features ===
        rms = np.sqrt(np.mean(audio ** 2)) + 1e-10
        peak = np.max(np.abs(audio)) + 1e-10

        features.rms_db = 20 * np.log10(rms)
        features.peak_db = 20 * np.log10(peak)
        features.crest_factor = peak / rms

        # === Spectral Features ===
        # Compute FFT
        n_fft = min(len(audio), 2048)
        if len(audio) >= n_fft:
            spectrum = np.abs(np.fft.rfft(audio[:n_fft]))
            freqs = np.fft.rfftfreq(n_fft, 1 / self.sample_rate)

            # Spectral rolloff (frequency below which 85% of energy lies)
            cumsum = np.cumsum(spectrum ** 2)
            rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
            features.spectral_rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

            # Spectral centroid (weighted mean of frequencies)
            features.spectral_centroid = (
                np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)
            )

            # Spectral flatness (geometric mean / arithmetic mean)
            # Higher = more noise-like, Lower = more tonal
            log_spectrum = np.log(spectrum + 1e-10)
            geo_mean = np.exp(np.mean(log_spectrum))
            arith_mean = np.mean(spectrum) + 1e-10
            features.spectral_flatness = geo_mean / arith_mean

            # Track spectral centroid history for variance
            self._spectral_history.append(features.spectral_centroid)
            if len(self._spectral_history) >= 10:
                features.spectral_variance = np.std(list(self._spectral_history)) / 1000.0

        # === Temporal Features ===
        # Zero-crossing rate
        signs = np.sign(audio)
        features.zero_crossing_rate = np.mean(np.abs(np.diff(signs))) / 2

        # Continuous speech tracking
        now = time.time()
        if features.rms_db > self.config.energy_threshold_db:
            if self._speech_start_time is None:
                self._speech_start_time = now
            features.continuous_speech_sec = now - self._speech_start_time
            self._last_speech_time = now
        else:
            # Reset if silence exceeds threshold
            if now - self._last_speech_time > self.config.silence_reset_sec:
                self._speech_start_time = None
                features.continuous_speech_sec = 0.0

        # === Compute Liveness Score ===
        features.liveness_score = self._compute_liveness_score(features)

        # Track statistics
        self._recent_crest_factors.append(features.crest_factor)
        self._recent_liveness_scores.append(features.liveness_score)

        return features

    def _compute_liveness_score(self, features: AudioFeatures) -> float:
        """Compute liveness score (0-1) from features.

        Higher scores indicate more likely to be live speech.
        TV audio typically has lower scores.
        """
        score = 0.0
        weights_sum = 0.0

        # 1. Crest factor (30% weight)
        # Live speech: typically 3-6, TV: typically 2-3
        crest_score = min(1.0, (features.crest_factor - 2.0) / 3.0)
        score += 0.30 * max(0, crest_score)
        weights_sum += 0.30

        # 2. Spectral rolloff (20% weight)
        # Live speech has more high-frequency content
        rolloff_score = min(1.0, features.spectral_rolloff / 5000.0)
        score += 0.20 * rolloff_score
        weights_sum += 0.20

        # 3. Spectral variance (25% weight)
        # Live speech has more variation in spectral content
        variance_score = min(1.0, features.spectral_variance / 0.3)
        score += 0.25 * variance_score
        weights_sum += 0.25

        # 4. Continuous speech penalty (15% weight)
        # TV talks continuously; penalize long unbroken speech
        if features.continuous_speech_sec > self.config.max_continuous_speech_sec:
            continuity_score = 0.0
        else:
            continuity_score = 1.0 - (features.continuous_speech_sec / self.config.max_continuous_speech_sec)
        score += 0.15 * continuity_score
        weights_sum += 0.15

        # 5. Spectral flatness (10% weight)
        # Speech is more tonal (lower flatness) than compressed audio
        flatness_score = 1.0 - min(1.0, features.spectral_flatness * 2)
        score += 0.10 * max(0, flatness_score)
        weights_sum += 0.10

        return score / weights_sum if weights_sum > 0 else 0.0

    def should_pass(self, audio: np.ndarray) -> tuple[bool, AudioFeatures]:
        """Determine if audio should pass the TV rejection filter.

        Args:
            audio: Audio samples (int16 or float32)

        Returns:
            Tuple of (should_pass, features)
        """
        if not self.config.enabled:
            return True, AudioFeatures()

        features = self.analyze(audio)
        self._frame_count += 1

        # Debug logging every 50 frames (~1 sec at 50fps)
        if self.config.debug_logging and self._frame_count % 50 == 0:
            logger.info(
                f"TV filter stats: rms={features.rms_db:.1f}dB, "
                f"crest={features.crest_factor:.2f}, "
                f"liveness={features.liveness_score:.2f}, "
                f"passes={self._consecutive_passes}"
            )

        # === Multi-layer filtering ===

        # Layer 1: Energy gate (reject quiet sounds)
        if features.rms_db < self.config.energy_threshold_db:
            self._consecutive_passes = 0
            return False, features

        # Layer 2: Crest factor (reject compressed audio)
        if features.crest_factor < self.config.min_crest_factor:
            logger.debug(
                f"TV rejection: low crest factor {features.crest_factor:.2f} "
                f"(min: {self.config.min_crest_factor})"
            )
            self._consecutive_passes = 0
            return False, features

        # Layer 3: Liveness score
        if features.liveness_score < self.config.min_liveness_score:
            logger.debug(
                f"TV rejection: low liveness score {features.liveness_score:.2f} "
                f"(min: {self.config.min_liveness_score})"
            )
            self._consecutive_passes = 0
            return False, features

        # Layer 4: Buffered detection (require consecutive passes)
        self._consecutive_passes += 1
        if self._consecutive_passes < self.config.required_consecutive_passes:
            logger.debug(
                f"TV filter: building passes {self._consecutive_passes}/{self.config.required_consecutive_passes}"
            )
            return False, features

        # Audio passed all filters
        if self._consecutive_passes == self.config.required_consecutive_passes:
            logger.info(
                f"TV filter: PASSED (crest={features.crest_factor:.2f}, "
                f"liveness={features.liveness_score:.2f})"
            )

        return True, features

    def reset(self) -> None:
        """Reset temporal state (call when returning to listening mode)."""
        self._spectral_history.clear()
        self._speech_start_time = None
        self._consecutive_passes = 0

    def get_stats(self) -> dict:
        """Get statistics for debugging/tuning."""
        return {
            "avg_crest_factor": (
                np.mean(list(self._recent_crest_factors))
                if self._recent_crest_factors else 0.0
            ),
            "avg_liveness_score": (
                np.mean(list(self._recent_liveness_scores))
                if self._recent_liveness_scores else 0.0
            ),
            "consecutive_passes": self._consecutive_passes,
            "continuous_speech_sec": (
                time.time() - self._speech_start_time
                if self._speech_start_time else 0.0
            ),
        }


# Convenience function
def create_tv_rejection_filter(settings: dict) -> TVRejectionFilter | None:
    """Create TV rejection filter from settings dict.

    Args:
        settings: Settings dict with tv_rejection_* keys

    Returns:
        TVRejectionFilter if enabled, None otherwise
    """
    enabled = settings.get("tv_rejection_enabled", True)

    if not enabled:
        logger.info("TV rejection: disabled")
        return None

    config = TVRejectionConfig(
        enabled=True,
        energy_threshold_db=settings.get("energy_gate_threshold_db", -35.0),
        min_crest_factor=settings.get("tv_rejection_min_crest_factor", 1.5),
        min_liveness_score=settings.get("tv_rejection_min_liveness", 0.15),
        required_consecutive_passes=settings.get("tv_rejection_consecutive_passes", 4),
        debug_logging=settings.get("tv_rejection_debug", False),
    )

    return TVRejectionFilter(config)
