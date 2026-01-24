"""Speaker Recognition and Voice Biometrics.

Provides speaker verification similar to Alexa/Google voice match:
- Enroll authorized speakers by recording voice samples
- Verify if incoming audio matches an enrolled speaker
- Optional: Only respond to recognized speakers

Uses speaker embeddings for voice biometric comparison.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# Try to import resemblyzer for speaker embeddings
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    HAS_RESEMBLYZER = True
except ImportError:
    HAS_RESEMBLYZER = False
    VoiceEncoder = None
    preprocess_wav = None

if TYPE_CHECKING:
    from livekit import rtc

logger = logging.getLogger(__name__)


@dataclass
class SpeakerProfile:
    """Stored profile for an enrolled speaker."""

    name: str
    embedding: np.ndarray  # Voice embedding vector
    enrollment_samples: int = 1  # Number of samples used for enrollment
    created_at: float = 0.0  # Unix timestamp


@dataclass
class SpeakerRecognitionConfig:
    """Configuration for speaker recognition."""

    enabled: bool = True

    # Verification threshold (cosine similarity)
    # Higher = more strict, fewer false positives
    # Lower = more lenient, fewer false negatives
    verification_threshold: float = 0.75  # 0.7-0.85 typical range

    # Minimum audio duration for verification (seconds)
    min_audio_duration: float = 1.0

    # Only activate for recognized speakers
    require_recognized_speaker: bool = False

    # Path to store speaker profiles
    profiles_path: str = "speaker_profiles.json"

    # Sample rate for audio processing
    sample_rate: int = 16000


class SpeakerRecognition:
    """Speaker recognition and voice biometrics system.

    Provides voice verification similar to Alexa Voice ID or Google Voice Match:
    1. Enroll speakers by recording their voice
    2. Verify incoming audio against enrolled speakers
    3. Optionally require recognized speaker for activation

    Usage:
        recognition = SpeakerRecognition(config)

        # Enroll a speaker
        recognition.enroll_speaker("John", audio_samples)

        # Verify audio
        result = recognition.verify_speaker(audio)
        if result.is_recognized:
            print(f"Recognized as {result.speaker_name}")
    """

    def __init__(self, config: SpeakerRecognitionConfig | None = None):
        self.config = config or SpeakerRecognitionConfig()
        self._encoder: VoiceEncoder | None = None
        self._profiles: dict[str, SpeakerProfile] = {}
        self._audio_buffer: list[np.ndarray] = []

        if not HAS_RESEMBLYZER:
            logger.warning(
                "Speaker recognition requires 'resemblyzer' package. "
                "Install with: pip install resemblyzer"
            )
            self.config.enabled = False
        else:
            self._load_profiles()
            logger.info(
                f"SpeakerRecognition initialized: "
                f"threshold={self.config.verification_threshold}, "
                f"profiles={len(self._profiles)}, "
                f"require_recognized={self.config.require_recognized_speaker}"
            )

    def _ensure_encoder(self) -> VoiceEncoder | None:
        """Lazy-load the voice encoder model."""
        if not HAS_RESEMBLYZER:
            return None

        if self._encoder is None:
            logger.info("Loading speaker recognition model (resemblyzer)...")
            self._encoder = VoiceEncoder()
            logger.info("Speaker recognition model loaded")

        return self._encoder

    def _load_profiles(self) -> None:
        """Load speaker profiles from disk."""
        profiles_path = Path(self.config.profiles_path)
        if not profiles_path.exists():
            return

        try:
            with open(profiles_path) as f:
                data = json.load(f)

            for name, profile_data in data.get("profiles", {}).items():
                self._profiles[name] = SpeakerProfile(
                    name=name,
                    embedding=np.array(profile_data["embedding"]),
                    enrollment_samples=profile_data.get("enrollment_samples", 1),
                    created_at=profile_data.get("created_at", 0.0),
                )

            logger.info(f"Loaded {len(self._profiles)} speaker profile(s)")
        except Exception as e:
            logger.warning(f"Failed to load speaker profiles: {e}")

    def _save_profiles(self) -> None:
        """Save speaker profiles to disk."""
        try:
            data = {
                "profiles": {
                    name: {
                        "embedding": profile.embedding.tolist(),
                        "enrollment_samples": profile.enrollment_samples,
                        "created_at": profile.created_at,
                    }
                    for name, profile in self._profiles.items()
                }
            }

            with open(self.config.profiles_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self._profiles)} speaker profile(s)")
        except Exception as e:
            logger.warning(f"Failed to save speaker profiles: {e}")

    def _audio_to_embedding(self, audio: np.ndarray) -> np.ndarray | None:
        """Convert audio to speaker embedding vector.

        Args:
            audio: Audio samples (float32, 16kHz)

        Returns:
            256-dimensional embedding vector or None if failed
        """
        encoder = self._ensure_encoder()
        if encoder is None:
            return None

        try:
            # Preprocess audio for resemblyzer
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0

            # Resemblyzer expects specific preprocessing
            wav = preprocess_wav(audio, self.config.sample_rate)

            if len(wav) < self.config.sample_rate * self.config.min_audio_duration:
                logger.debug("Audio too short for speaker embedding")
                return None

            # Generate embedding
            embedding = encoder.embed_utterance(wav)
            return embedding

        except Exception as e:
            logger.warning(f"Failed to generate speaker embedding: {e}")
            return None

    def enroll_speaker(
        self,
        name: str,
        audio_samples: list[np.ndarray],
        update_existing: bool = True,
    ) -> bool:
        """Enroll a new speaker with voice samples.

        Args:
            name: Speaker name/identifier
            audio_samples: List of audio samples (each should be 2-10 seconds)
            update_existing: If True, update existing profile; if False, fail

        Returns:
            True if enrollment successful, False otherwise
        """
        if not self.config.enabled:
            logger.warning("Speaker recognition is disabled")
            return False

        if name in self._profiles and not update_existing:
            logger.warning(f"Speaker '{name}' already enrolled")
            return False

        # Generate embeddings for each sample
        embeddings = []
        for audio in audio_samples:
            embedding = self._audio_to_embedding(audio)
            if embedding is not None:
                embeddings.append(embedding)

        if not embeddings:
            logger.warning(f"No valid audio samples for speaker '{name}'")
            return False

        # Average embeddings for robust profile
        avg_embedding = np.mean(embeddings, axis=0)

        # Normalize embedding
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-10)

        import time
        self._profiles[name] = SpeakerProfile(
            name=name,
            embedding=avg_embedding,
            enrollment_samples=len(embeddings),
            created_at=time.time(),
        )

        self._save_profiles()
        logger.info(
            f"Enrolled speaker '{name}' with {len(embeddings)} sample(s)"
        )
        return True

    def remove_speaker(self, name: str) -> bool:
        """Remove an enrolled speaker.

        Args:
            name: Speaker name to remove

        Returns:
            True if removed, False if not found
        """
        if name not in self._profiles:
            logger.warning(f"Speaker '{name}' not found")
            return False

        del self._profiles[name]
        self._save_profiles()
        logger.info(f"Removed speaker '{name}'")
        return True

    def list_speakers(self) -> list[str]:
        """List all enrolled speakers."""
        return list(self._profiles.keys())

    def verify_speaker(
        self,
        audio: np.ndarray,
        target_speaker: str | None = None,
    ) -> VerificationResult:
        """Verify if audio matches an enrolled speaker.

        Args:
            audio: Audio samples to verify
            target_speaker: If provided, only check against this speaker

        Returns:
            VerificationResult with match information
        """
        if not self.config.enabled or not self._profiles:
            return VerificationResult(
                is_recognized=False,
                speaker_name=None,
                confidence=0.0,
                all_scores={},
            )

        # Generate embedding for input audio
        embedding = self._audio_to_embedding(audio)
        if embedding is None:
            return VerificationResult(
                is_recognized=False,
                speaker_name=None,
                confidence=0.0,
                all_scores={},
            )

        # Normalize embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

        # Compare against enrolled speakers
        scores = {}
        profiles_to_check = (
            {target_speaker: self._profiles[target_speaker]}
            if target_speaker and target_speaker in self._profiles
            else self._profiles
        )

        for name, profile in profiles_to_check.items():
            # Cosine similarity
            similarity = np.dot(embedding, profile.embedding)
            scores[name] = float(similarity)

        # Find best match
        if scores:
            best_speaker = max(scores, key=scores.get)
            best_score = scores[best_speaker]

            is_recognized = best_score >= self.config.verification_threshold

            if is_recognized:
                logger.info(
                    f"Speaker recognized: {best_speaker} "
                    f"(confidence={best_score:.2f})"
                )
            else:
                logger.debug(
                    f"Speaker not recognized: best match {best_speaker} "
                    f"(confidence={best_score:.2f}, threshold={self.config.verification_threshold})"
                )

            return VerificationResult(
                is_recognized=is_recognized,
                speaker_name=best_speaker if is_recognized else None,
                confidence=best_score,
                all_scores=scores,
            )

        return VerificationResult(
            is_recognized=False,
            speaker_name=None,
            confidence=0.0,
            all_scores={},
        )

    def accumulate_audio(self, frame: "rtc.AudioFrame") -> np.ndarray | None:
        """Accumulate audio frames for verification.

        Call this with each audio frame. Returns accumulated audio
        when enough has been collected for verification.

        Args:
            frame: LiveKit audio frame

        Returns:
            Accumulated audio if enough collected, None otherwise
        """
        # Convert frame to numpy
        audio = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0

        # Handle multi-channel by averaging
        if frame.num_channels > 1:
            audio = audio.reshape(-1, frame.num_channels).mean(axis=1)

        self._audio_buffer.append(audio)

        # Check if we have enough audio
        total_samples = sum(len(chunk) for chunk in self._audio_buffer)
        min_samples = int(self.config.sample_rate * self.config.min_audio_duration)

        if total_samples >= min_samples:
            accumulated = np.concatenate(self._audio_buffer)
            self._audio_buffer.clear()
            return accumulated

        return None

    def reset_buffer(self) -> None:
        """Reset the audio accumulation buffer."""
        self._audio_buffer.clear()

    def should_activate(self, audio: np.ndarray) -> tuple[bool, VerificationResult]:
        """Check if audio passes speaker verification for activation.

        Use this when require_recognized_speaker is True.

        Args:
            audio: Audio to verify

        Returns:
            (should_activate, verification_result)
        """
        result = self.verify_speaker(audio)

        if not self.config.require_recognized_speaker:
            # Don't require recognized speaker, but still provide info
            return True, result

        return result.is_recognized, result


@dataclass
class VerificationResult:
    """Result of speaker verification."""

    is_recognized: bool  # True if matched an enrolled speaker
    speaker_name: str | None  # Name of matched speaker (if recognized)
    confidence: float  # Similarity score (0-1)
    all_scores: dict[str, float] = field(default_factory=dict)  # Scores for all speakers


def create_speaker_recognition(settings: dict) -> SpeakerRecognition | None:
    """Create speaker recognition from settings dict.

    Args:
        settings: Settings dict with speaker_recognition_* keys

    Returns:
        SpeakerRecognition if enabled, None otherwise
    """
    enabled = settings.get("speaker_recognition_enabled", False)

    if not enabled:
        logger.info("Speaker recognition: disabled")
        return None

    if not HAS_RESEMBLYZER:
        logger.warning(
            "Speaker recognition requires 'resemblyzer'. "
            "Install with: pip install resemblyzer"
        )
        return None

    config = SpeakerRecognitionConfig(
        enabled=True,
        verification_threshold=settings.get("speaker_recognition_threshold", 0.75),
        min_audio_duration=settings.get("speaker_recognition_min_duration", 1.0),
        require_recognized_speaker=settings.get("speaker_recognition_required", False),
        profiles_path=settings.get("speaker_recognition_profiles_path", "speaker_profiles.json"),
    )

    return SpeakerRecognition(config)
