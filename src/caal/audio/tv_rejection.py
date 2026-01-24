"""TV/Media Audio Rejection System.

Implements multi-layer audio analysis to distinguish live speech from TV/radio/video:

1. **Spectral Analysis**: TV audio has codec artifacts, bandwidth limiting
2. **Dynamic Range**: TV audio is compressed (lower crest factor)
3. **Temporal Patterns**: TV is continuous, live speech has natural pauses
4. **Liveness Scoring**: Combines multiple signals to score authenticity
5. **Music Detection**: Radio/video music has distinct harmonic patterns
6. **Stereo Correlation**: Media is often stereo, live speech is directional mono

Based on research from commercial voice assistants (Alexa, Siri, Google).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# Optional scipy import for advanced signal processing
try:
    from scipy import signal as scipy_signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    scipy_signal = None

logger = logging.getLogger(__name__)


@dataclass
class TVRejectionConfig:
    """Configuration for TV/media rejection system."""

    enabled: bool = True

    # Energy gate (first filter - reject quiet/distant sounds)
    energy_threshold_db: float = -35.0  # RMS threshold

    # Crest factor (peak-to-RMS ratio) - TV audio is compressed
    # Note: Live speech typically 3-6, TV typically 1.5-2.5
    # Using 1.8 for stricter filtering of compressed audio
    min_crest_factor: float = 1.8

    # Spectral analysis
    min_spectral_rolloff_hz: float = 3000.0  # TV often bandwidth-limited
    min_spectral_variance: float = 0.15  # Live speech has more variation

    # Temporal analysis (audio continuity)
    max_continuous_speech_sec: float = 10.0  # TV talks continuously
    silence_reset_sec: float = 1.0  # Reset continuity after silence

    # Liveness scoring thresholds
    # Using 0.20 for stricter filtering of TV-like audio
    min_liveness_score: float = 0.20  # Minimum score to pass (0-1)

    # Buffered detection (require multiple consecutive passes)
    required_consecutive_passes: int = 3  # Frames that must pass
    analysis_window_sec: float = 0.5  # Window for spectral analysis

    # Music detection (for radio/video)
    music_detection_enabled: bool = True
    max_harmonic_ratio: float = 0.7  # Music has strong harmonics (reject if above)
    max_rhythm_regularity: float = 0.8  # Music has regular beat patterns

    # Stereo correlation (media is typically stereo, live speech is directional)
    stereo_detection_enabled: bool = True
    min_stereo_correlation: float = 0.3  # Live speech tends to have high correlation

    # Codec artifact detection
    codec_artifact_detection: bool = True
    max_band_energy_ratio: float = 0.9  # Codecs often cut high frequencies

    # Playback voice detection (video with voice content)
    playback_voice_detection: bool = True
    # Room acoustic analysis - playback lacks natural room reflections
    min_room_reverb_ratio: float = 0.05  # Live speech has room reflections
    # Spectral consistency - playback has unnaturally consistent characteristics
    max_spectral_consistency: float = 0.85  # Live speech varies more
    # Dynamic micro-variations - live speech has natural micro-modulations
    min_pitch_variation: float = 0.02  # Live speech has natural pitch changes

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

    # Music detection features (for radio/video)
    harmonic_ratio: float = 0.0  # Ratio of harmonic to total energy
    rhythm_regularity: float = 0.0  # How regular the beat pattern is
    is_music_detected: bool = False

    # Stereo analysis features
    stereo_correlation: float = 1.0  # 1.0 = mono, 0.0 = uncorrelated stereo
    stereo_width: float = 0.0  # How wide the stereo image is

    # Codec artifact detection
    high_freq_cutoff_detected: bool = False  # True if sharp frequency cutoff
    band_energy_ratio: float = 0.0  # Ratio of high to low frequency energy

    # Playback voice detection (video with voice)
    room_reverb_ratio: float = 0.0  # Ratio of reverberant to direct sound
    spectral_consistency: float = 0.0  # How consistent the spectrum is over time
    pitch_variation: float = 0.0  # Natural pitch micro-variations
    is_playback_voice: bool = False  # True if detected as playback voice

    # Computed scores
    liveness_score: float = 0.0
    media_rejection_score: float = 0.0  # Combined media detection score


class TVRejectionFilter:
    """Multi-layer filter to reject TV/radio/video audio and pass live speech.

    Uses spectral analysis, dynamic range detection, temporal patterns,
    music detection, and stereo analysis to distinguish live human speech
    from media playback audio.
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

        # Music detection buffers
        self._onset_history: deque[float] = deque(maxlen=100)  # For rhythm detection
        self._energy_history: deque[float] = deque(maxlen=50)  # For onset detection

        # Stereo analysis buffers (for multi-channel audio)
        self._stereo_history: deque[float] = deque(maxlen=20)

        # Playback voice detection buffers
        self._spectrum_history: deque[np.ndarray] = deque(maxlen=30)  # For spectral consistency
        self._pitch_history: deque[float] = deque(maxlen=50)  # For pitch variation
        self._reverb_history: deque[float] = deque(maxlen=20)  # For room acoustic smoothing

        self._frame_count = 0  # For periodic logging
        logger.info(
            f"TVRejectionFilter initialized: "
            f"energy_threshold={self.config.energy_threshold_db}dB, "
            f"min_crest_factor={self.config.min_crest_factor}, "
            f"min_liveness={self.config.min_liveness_score}, "
            f"consecutive_passes={self.config.required_consecutive_passes}, "
            f"music_detection={self.config.music_detection_enabled}, "
            f"stereo_detection={self.config.stereo_detection_enabled}"
        )

    def analyze(self, audio: np.ndarray, stereo_audio: np.ndarray | None = None) -> AudioFeatures:
        """Extract features from audio for TV/media rejection analysis.

        Args:
            audio: Audio samples (int16 or float32), mono
            stereo_audio: Optional stereo audio for stereo correlation analysis

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

            # === Music Detection (for radio/video) ===
            if self.config.music_detection_enabled:
                features.harmonic_ratio = self._detect_harmonics(spectrum, freqs)
                features.rhythm_regularity = self._detect_rhythm(rms)
                features.is_music_detected = (
                    features.harmonic_ratio > self.config.max_harmonic_ratio or
                    features.rhythm_regularity > self.config.max_rhythm_regularity
                )

            # === Codec Artifact Detection ===
            if self.config.codec_artifact_detection:
                features.band_energy_ratio, features.high_freq_cutoff_detected = (
                    self._detect_codec_artifacts(spectrum, freqs)
                )

            # === Playback Voice Detection (video with voice) ===
            if self.config.playback_voice_detection:
                # Detect room acoustics (live speech has natural room reflections)
                features.room_reverb_ratio = self._detect_room_acoustics(spectrum, freqs)

                # Detect spectral consistency (playback is unnaturally consistent)
                features.spectral_consistency = self._detect_spectral_consistency(spectrum)

                # Detect pitch micro-variations (live speech has natural variations)
                features.pitch_variation = self._detect_pitch_variation(audio)

                # Combine signals to detect playback voice
                features.is_playback_voice = (
                    features.room_reverb_ratio < self.config.min_room_reverb_ratio or
                    features.spectral_consistency > self.config.max_spectral_consistency or
                    features.pitch_variation < self.config.min_pitch_variation
                )

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

        # === Stereo Correlation Analysis ===
        if self.config.stereo_detection_enabled and stereo_audio is not None:
            features.stereo_correlation, features.stereo_width = (
                self._analyze_stereo(stereo_audio)
            )

        # === Compute Liveness Score ===
        features.liveness_score = self._compute_liveness_score(features)

        # === Compute Media Rejection Score ===
        features.media_rejection_score = self._compute_media_score(features)

        # Track statistics
        self._recent_crest_factors.append(features.crest_factor)
        self._recent_liveness_scores.append(features.liveness_score)

        return features

    def _detect_harmonics(self, spectrum: np.ndarray, freqs: np.ndarray) -> float:
        """Detect harmonic content indicating music.

        Music typically has strong harmonic series (fundamental + overtones).
        Speech has more irregular spectral content.

        Returns:
            Harmonic ratio (0-1), higher = more harmonic/musical
        """
        if len(spectrum) < 100:
            return 0.0

        # Find peaks in spectrum
        peak_indices = []
        for i in range(1, len(spectrum) - 1):
            if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
                if spectrum[i] > np.mean(spectrum) * 2:  # Significant peak
                    peak_indices.append(i)

        if len(peak_indices) < 2:
            return 0.0

        # Check if peaks follow harmonic series (f, 2f, 3f, ...)
        peak_freqs = freqs[peak_indices]
        if len(peak_freqs) < 2:
            return 0.0

        # Check ratio between consecutive peaks
        harmonic_count = 0
        fundamental = peak_freqs[0] if peak_freqs[0] > 50 else peak_freqs[1] if len(peak_freqs) > 1 else 100

        for freq in peak_freqs[1:]:
            ratio = freq / fundamental
            # Check if ratio is close to an integer (harmonic)
            if abs(ratio - round(ratio)) < 0.1 and round(ratio) <= 10:
                harmonic_count += 1

        return min(1.0, harmonic_count / max(1, len(peak_freqs) - 1))

    def _detect_rhythm(self, current_rms: float) -> float:
        """Detect rhythmic patterns indicating music.

        Music has regular beat patterns, speech has irregular energy.

        Returns:
            Rhythm regularity (0-1), higher = more regular/musical
        """
        # Track energy for onset detection
        self._energy_history.append(current_rms)

        if len(self._energy_history) < 20:
            return 0.0

        energy = np.array(list(self._energy_history))

        # Detect onsets (energy increases)
        diff = np.diff(energy)
        onsets = np.where(diff > np.std(diff) * 0.5)[0]

        if len(onsets) < 3:
            return 0.0

        # Calculate inter-onset intervals
        intervals = np.diff(onsets)

        if len(intervals) < 2:
            return 0.0

        # Check regularity of intervals (low variance = regular beat)
        mean_interval = np.mean(intervals)
        if mean_interval < 1:
            return 0.0

        regularity = 1.0 - min(1.0, np.std(intervals) / mean_interval)
        return regularity

    def _detect_codec_artifacts(self, spectrum: np.ndarray, freqs: np.ndarray) -> tuple[float, bool]:
        """Detect codec compression artifacts.

        Streaming media often has:
        - Sharp high-frequency cutoff (e.g., 16kHz for MP3)
        - Reduced high-frequency energy

        Returns:
            (band_energy_ratio, cutoff_detected)
        """
        if len(freqs) < 10:
            return 0.0, False

        # Find frequency bins for different bands
        low_band = (freqs >= 100) & (freqs <= 2000)
        high_band = (freqs >= 4000) & (freqs <= 8000)

        low_energy = np.sum(spectrum[low_band] ** 2) if np.any(low_band) else 1e-10
        high_energy = np.sum(spectrum[high_band] ** 2) if np.any(high_band) else 0

        band_ratio = high_energy / (low_energy + 1e-10)

        # Detect sharp cutoff (typical of codecs)
        # Look for sudden drop in spectrum at common codec cutoff frequencies
        cutoff_detected = False
        cutoff_freqs = [4000, 8000, 11025, 16000]  # Common codec cutoffs

        for cutoff_freq in cutoff_freqs:
            idx = np.argmin(np.abs(freqs - cutoff_freq))
            if idx > 5 and idx < len(spectrum) - 5:
                before = np.mean(spectrum[idx-5:idx])
                after = np.mean(spectrum[idx:idx+5])
                if before > 0 and after / before < 0.3:  # Sharp drop
                    cutoff_detected = True
                    break

        return band_ratio, cutoff_detected

    def _detect_room_acoustics(self, spectrum: np.ndarray, freqs: np.ndarray) -> float:
        """Detect room acoustic characteristics.

        Live speech in a room has natural reflections/reverb.
        Playback from speakers lacks these natural room characteristics
        (already has its own recorded room sound baked in).

        Returns:
            Room reverb ratio (0-1), higher = more natural room sound
        """
        if len(spectrum) < 100:
            return 0.0

        # Analyze energy decay in different frequency bands
        # Natural room acoustics have specific decay patterns

        # Low frequencies (room modes) - 50-200 Hz
        low_band = (freqs >= 50) & (freqs <= 200)
        # Mid frequencies (direct speech) - 200-2000 Hz
        mid_band = (freqs >= 200) & (freqs <= 2000)
        # High frequencies (air absorption, decay faster) - 4000-8000 Hz
        high_band = (freqs >= 4000) & (freqs <= 8000)

        low_energy = np.sum(spectrum[low_band] ** 2) if np.any(low_band) else 1e-10
        mid_energy = np.sum(spectrum[mid_band] ** 2) if np.any(mid_band) else 1e-10
        high_energy = np.sum(spectrum[high_band] ** 2) if np.any(high_band) else 1e-10

        # In natural room acoustics, there's a specific ratio between bands
        # Playback tends to have more compressed dynamics and different ratios
        low_mid_ratio = low_energy / mid_energy
        high_mid_ratio = high_energy / mid_energy

        # Natural room sound has moderate low frequencies (room resonance)
        # and natural high-frequency decay
        room_score = 0.0

        # Check for natural room resonance (not too much, not too little)
        if 0.05 < low_mid_ratio < 0.5:
            room_score += 0.4

        # Check for natural high-frequency content (not cut off by codecs)
        if high_mid_ratio > 0.01:
            room_score += 0.3

        # Check for spectral irregularities (natural rooms have modes)
        if len(spectrum) > 50:
            spectral_irregularity = np.std(spectrum[10:50]) / (np.mean(spectrum[10:50]) + 1e-10)
            if spectral_irregularity > 0.5:
                room_score += 0.3

        # Smooth over time
        self._reverb_history.append(room_score)
        return np.mean(list(self._reverb_history))

    def _detect_spectral_consistency(self, spectrum: np.ndarray) -> float:
        """Detect unnatural spectral consistency.

        Playback audio (same speaker recording) has very consistent spectral shape.
        Live speech varies more frame-to-frame due to natural speech dynamics.

        Returns:
            Spectral consistency (0-1), higher = more consistent (more likely playback)
        """
        if len(spectrum) < 50:
            return 0.0

        # Normalize spectrum for shape comparison
        norm_spectrum = spectrum / (np.sum(spectrum) + 1e-10)

        # Store in history
        self._spectrum_history.append(norm_spectrum[:50])  # Use first 50 bins

        if len(self._spectrum_history) < 10:
            return 0.0

        # Calculate correlation between consecutive frames
        correlations = []
        spectra = list(self._spectrum_history)
        for i in range(1, len(spectra)):
            if len(spectra[i]) == len(spectra[i-1]):
                corr = np.corrcoef(spectra[i], spectra[i-1])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        if not correlations:
            return 0.0

        # High average correlation = consistent (playback-like)
        return np.mean(correlations)

    def _detect_pitch_variation(self, audio: np.ndarray) -> float:
        """Detect natural pitch micro-variations.

        Live speech has natural micro-variations in pitch (jitter).
        Playback from recordings may lack these natural variations or have
        different characteristics.

        Returns:
            Pitch variation (0-1), higher = more natural variation
        """
        if len(audio) < 256:
            return 0.0

        # Simple pitch estimation using autocorrelation
        # This is a basic approach - could be enhanced with more sophisticated methods

        # Use autocorrelation to find fundamental frequency
        correlation = np.correlate(audio, audio, mode='full')
        correlation = correlation[len(correlation)//2:]

        # Find first peak after initial decay (fundamental period)
        # Look in range corresponding to 80-400 Hz (typical speech range)
        min_period = int(self.sample_rate / 400)  # 400 Hz
        max_period = int(self.sample_rate / 80)   # 80 Hz

        if max_period >= len(correlation):
            return 0.0

        # Find peak in valid range
        search_range = correlation[min_period:max_period]
        if len(search_range) == 0:
            return 0.0

        peak_idx = np.argmax(search_range) + min_period
        peak_value = correlation[peak_idx]

        # Estimate pitch confidence
        pitch_confidence = peak_value / (correlation[0] + 1e-10)

        # Store pitch estimate
        if pitch_confidence > 0.3:  # Only track confident estimates
            estimated_pitch = self.sample_rate / peak_idx
            self._pitch_history.append(estimated_pitch)

        if len(self._pitch_history) < 20:
            return 0.5  # Not enough data, assume neutral

        # Calculate pitch variation (coefficient of variation)
        pitches = np.array(list(self._pitch_history))
        mean_pitch = np.mean(pitches)
        if mean_pitch < 1:
            return 0.0

        # Natural speech has 2-10% pitch variation
        # Playback may have less (compressed) or more (different speaker)
        cv = np.std(pitches) / mean_pitch

        # Map to 0-1 range (0.02-0.10 is natural range)
        if cv < 0.01:
            return 0.0  # Too consistent (robotic/playback)
        elif cv < 0.02:
            return cv / 0.02 * 0.5
        elif cv <= 0.10:
            return 0.5 + (cv - 0.02) / 0.08 * 0.5  # Natural range
        else:
            return max(0, 1.0 - (cv - 0.10) / 0.10)  # Too variable

    def _analyze_stereo(self, stereo_audio: np.ndarray) -> tuple[float, float]:
        """Analyze stereo characteristics.

        Live speech from a person is typically mono-like (high correlation).
        Media often has wide stereo separation (low correlation).

        Args:
            stereo_audio: Interleaved stereo samples [L, R, L, R, ...]

        Returns:
            (stereo_correlation, stereo_width)
        """
        if stereo_audio.dtype == np.int16:
            stereo_audio = stereo_audio.astype(np.float32) / 32768.0

        # Deinterleave to left/right channels
        if len(stereo_audio) < 4:
            return 1.0, 0.0

        left = stereo_audio[0::2]
        right = stereo_audio[1::2]

        if len(left) != len(right) or len(left) < 10:
            return 1.0, 0.0

        # Calculate correlation coefficient
        left_mean = np.mean(left)
        right_mean = np.mean(right)
        left_centered = left - left_mean
        right_centered = right - right_mean

        numerator = np.sum(left_centered * right_centered)
        denominator = np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2)) + 1e-10

        correlation = numerator / denominator

        # Calculate stereo width (difference between channels)
        mid = (left + right) / 2
        side = (left - right) / 2
        mid_energy = np.sum(mid ** 2) + 1e-10
        side_energy = np.sum(side ** 2)
        stereo_width = side_energy / mid_energy

        # Track correlation history for smoothing
        self._stereo_history.append(correlation)
        smoothed_correlation = np.mean(list(self._stereo_history))

        return smoothed_correlation, stereo_width

    def _compute_liveness_score(self, features: AudioFeatures) -> float:
        """Compute liveness score (0-1) from features.

        Higher scores indicate more likely to be live speech.
        TV/radio/video audio typically has lower scores.
        """
        score = 0.0
        weights_sum = 0.0

        # 1. Crest factor (25% weight)
        # Live speech: typically 3-6, TV: typically 2-3
        crest_score = min(1.0, (features.crest_factor - 2.0) / 3.0)
        score += 0.25 * max(0, crest_score)
        weights_sum += 0.25

        # 2. Spectral rolloff (15% weight)
        # Live speech has more high-frequency content
        rolloff_score = min(1.0, features.spectral_rolloff / 5000.0)
        score += 0.15 * rolloff_score
        weights_sum += 0.15

        # 3. Spectral variance (20% weight)
        # Live speech has more variation in spectral content
        variance_score = min(1.0, features.spectral_variance / 0.3)
        score += 0.20 * variance_score
        weights_sum += 0.20

        # 4. Continuous speech penalty (10% weight)
        # TV talks continuously; penalize long unbroken speech
        if features.continuous_speech_sec > self.config.max_continuous_speech_sec:
            continuity_score = 0.0
        else:
            continuity_score = 1.0 - (features.continuous_speech_sec / self.config.max_continuous_speech_sec)
        score += 0.10 * continuity_score
        weights_sum += 0.10

        # 5. Spectral flatness (10% weight)
        # Speech is more tonal (lower flatness) than compressed audio
        flatness_score = 1.0 - min(1.0, features.spectral_flatness * 2)
        score += 0.10 * max(0, flatness_score)
        weights_sum += 0.10

        # 6. Music detection penalty (10% weight)
        # If music is detected, penalize liveness score
        if self.config.music_detection_enabled and features.is_music_detected:
            score += 0.10 * 0.0  # Music detected = 0 score for this component
        else:
            score += 0.10 * 1.0
        weights_sum += 0.10

        # 7. Stereo correlation bonus (8% weight)
        # High correlation = more mono-like = more likely live speech
        if self.config.stereo_detection_enabled:
            stereo_score = max(0, features.stereo_correlation)
            score += 0.08 * stereo_score
            weights_sum += 0.08

        # 8. Playback voice detection penalty (12% weight)
        # If detected as playback voice, heavily penalize liveness score
        if self.config.playback_voice_detection:
            if features.is_playback_voice:
                score += 0.12 * 0.0  # Playback detected = 0 score
            else:
                # Combine room acoustics, spectral consistency, pitch variation
                playback_score = (
                    features.room_reverb_ratio * 0.4 +  # Room sounds natural
                    (1.0 - features.spectral_consistency) * 0.3 +  # Spectrum varies
                    features.pitch_variation * 0.3  # Pitch varies naturally
                )
                score += 0.12 * playback_score
            weights_sum += 0.12

        return score / weights_sum if weights_sum > 0 else 0.0

    def _compute_media_score(self, features: AudioFeatures) -> float:
        """Compute media rejection score (0-1) from features.

        Higher scores indicate more likely to be media (TV/radio/video).
        Used as additional rejection criteria.
        """
        score = 0.0
        count = 0

        # 1. Music detection
        if self.config.music_detection_enabled:
            if features.is_music_detected:
                score += 1.0
            else:
                # Partial score based on harmonic/rhythm ratio
                score += (features.harmonic_ratio * 0.5 + features.rhythm_regularity * 0.5)
            count += 1

        # 2. Low stereo correlation (media is typically wide stereo)
        if self.config.stereo_detection_enabled:
            if features.stereo_correlation < self.config.min_stereo_correlation:
                score += 1.0
            else:
                score += max(0, 1.0 - features.stereo_correlation)
            count += 1

        # 3. Codec artifacts
        if self.config.codec_artifact_detection:
            if features.high_freq_cutoff_detected:
                score += 0.8
            if features.band_energy_ratio > self.config.max_band_energy_ratio:
                score += 0.5
            count += 1

        # 4. Continuous speech (TV talks without pauses)
        if features.continuous_speech_sec > self.config.max_continuous_speech_sec * 0.7:
            score += 0.6
            count += 1

        # 5. Playback voice detection (video with voice)
        if self.config.playback_voice_detection:
            playback_score = 0.0

            # Lack of natural room acoustics
            if features.room_reverb_ratio < self.config.min_room_reverb_ratio:
                playback_score += 0.4

            # Unnaturally consistent spectrum
            if features.spectral_consistency > self.config.max_spectral_consistency:
                playback_score += 0.4

            # Lack of natural pitch variation
            if features.pitch_variation < self.config.min_pitch_variation:
                playback_score += 0.4

            # Direct playback voice flag
            if features.is_playback_voice:
                playback_score += 0.5

            score += min(1.0, playback_score)
            count += 1

        return score / max(1, count)

    def should_pass(self, audio: np.ndarray, stereo_audio: np.ndarray | None = None) -> tuple[bool, AudioFeatures]:
        """Determine if audio should pass the TV/media rejection filter.

        Args:
            audio: Audio samples (int16 or float32), mono
            stereo_audio: Optional stereo audio for stereo correlation analysis

        Returns:
            Tuple of (should_pass, features)
        """
        if not self.config.enabled:
            return True, AudioFeatures()

        features = self.analyze(audio, stereo_audio)
        self._frame_count += 1

        # Debug logging every 50 frames (~1 sec at 50fps)
        if self.config.debug_logging and self._frame_count % 50 == 0:
            logger.info(
                f"Media filter stats: rms={features.rms_db:.1f}dB, "
                f"crest={features.crest_factor:.2f}, "
                f"liveness={features.liveness_score:.2f}, "
                f"media_score={features.media_rejection_score:.2f}, "
                f"music={features.is_music_detected}, "
                f"playback_voice={features.is_playback_voice}, "
                f"room_reverb={features.room_reverb_ratio:.2f}, "
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
                f"Media rejection: low crest factor {features.crest_factor:.2f} "
                f"(min: {self.config.min_crest_factor})"
            )
            self._consecutive_passes = 0
            return False, features

        # Layer 3: Music detection (reject radio/video music)
        if self.config.music_detection_enabled and features.is_music_detected:
            logger.debug(
                f"Media rejection: music detected (harmonic={features.harmonic_ratio:.2f}, "
                f"rhythm={features.rhythm_regularity:.2f})"
            )
            self._consecutive_passes = 0
            return False, features

        # Layer 4: Stereo correlation (reject wide stereo media)
        if self.config.stereo_detection_enabled:
            if features.stereo_correlation < self.config.min_stereo_correlation:
                logger.debug(
                    f"Media rejection: low stereo correlation {features.stereo_correlation:.2f} "
                    f"(min: {self.config.min_stereo_correlation})"
                )
                self._consecutive_passes = 0
                return False, features

        # Layer 5: Playback voice detection (video with voice content)
        if self.config.playback_voice_detection and features.is_playback_voice:
            logger.debug(
                f"Media rejection: playback voice detected "
                f"(room_reverb={features.room_reverb_ratio:.2f}, "
                f"spectral_consistency={features.spectral_consistency:.2f}, "
                f"pitch_variation={features.pitch_variation:.2f})"
            )
            self._consecutive_passes = 0
            return False, features

        # Layer 6: Liveness score
        if features.liveness_score < self.config.min_liveness_score:
            logger.debug(
                f"Media rejection: low liveness score {features.liveness_score:.2f} "
                f"(min: {self.config.min_liveness_score})"
            )
            self._consecutive_passes = 0
            return False, features

        # Layer 7: High media rejection score
        if features.media_rejection_score > 0.7:
            logger.debug(
                f"Media rejection: high media score {features.media_rejection_score:.2f}"
            )
            self._consecutive_passes = 0
            return False, features

        # Layer 8: Buffered detection (require consecutive passes)
        self._consecutive_passes += 1
        if self._consecutive_passes < self.config.required_consecutive_passes:
            logger.debug(
                f"Media filter: building passes {self._consecutive_passes}/{self.config.required_consecutive_passes}"
            )
            return False, features

        # Audio passed all filters
        if self._consecutive_passes == self.config.required_consecutive_passes:
            logger.info(
                f"Media filter: PASSED (crest={features.crest_factor:.2f}, "
                f"liveness={features.liveness_score:.2f}, "
                f"media_score={features.media_rejection_score:.2f})"
            )

        return True, features

    def reset(self) -> None:
        """Reset temporal state (call when returning to listening mode)."""
        self._spectral_history.clear()
        self._onset_history.clear()
        self._energy_history.clear()
        self._stereo_history.clear()
        self._spectrum_history.clear()
        self._pitch_history.clear()
        self._reverb_history.clear()
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
            "avg_stereo_correlation": (
                np.mean(list(self._stereo_history))
                if self._stereo_history else 1.0
            ),
            "avg_room_reverb": (
                np.mean(list(self._reverb_history))
                if self._reverb_history else 0.0
            ),
            "avg_pitch_variation": (
                np.std(list(self._pitch_history)) / (np.mean(list(self._pitch_history)) + 1e-10)
                if len(self._pitch_history) > 5 else 0.0
            ),
            "consecutive_passes": self._consecutive_passes,
            "continuous_speech_sec": (
                time.time() - self._speech_start_time
                if self._speech_start_time else 0.0
            ),
            "music_detection_enabled": self.config.music_detection_enabled,
            "stereo_detection_enabled": self.config.stereo_detection_enabled,
            "playback_voice_detection_enabled": self.config.playback_voice_detection,
        }


# Convenience function
def create_tv_rejection_filter(settings: dict) -> TVRejectionFilter | None:
    """Create TV/media rejection filter from settings dict.

    Args:
        settings: Settings dict with tv_rejection_* and media_noise_* keys

    Returns:
        TVRejectionFilter if enabled, None otherwise
    """
    enabled = settings.get("tv_rejection_enabled", True)

    if not enabled:
        logger.info("Media noise rejection: disabled")
        return None

    config = TVRejectionConfig(
        enabled=True,
        energy_threshold_db=settings.get("energy_gate_threshold_db", -35.0),
        min_crest_factor=settings.get("tv_rejection_min_crest_factor", 1.8),
        min_liveness_score=settings.get("tv_rejection_min_liveness", 0.20),
        required_consecutive_passes=settings.get("tv_rejection_consecutive_passes", 3),
        debug_logging=settings.get("tv_rejection_debug", False),
        # Music detection (for radio/video)
        music_detection_enabled=settings.get("media_noise_music_detection", True),
        max_harmonic_ratio=settings.get("media_noise_max_harmonic_ratio", 0.7),
        max_rhythm_regularity=settings.get("media_noise_max_rhythm_regularity", 0.8),
        # Stereo detection
        stereo_detection_enabled=settings.get("media_noise_stereo_detection", True),
        min_stereo_correlation=settings.get("media_noise_min_stereo_correlation", 0.3),
        # Codec artifact detection
        codec_artifact_detection=settings.get("media_noise_codec_detection", True),
        # Playback voice detection (video with voice)
        playback_voice_detection=settings.get("media_noise_playback_voice_detection", True),
        min_room_reverb_ratio=settings.get("media_noise_min_room_reverb", 0.05),
        max_spectral_consistency=settings.get("media_noise_max_spectral_consistency", 0.85),
        min_pitch_variation=settings.get("media_noise_min_pitch_variation", 0.02),
    )

    logger.info(
        f"Media noise rejection: enabled (music={config.music_detection_enabled}, "
        f"stereo={config.stereo_detection_enabled}, "
        f"codec={config.codec_artifact_detection}, "
        f"playback_voice={config.playback_voice_detection})"
    )

    return TVRejectionFilter(config)
