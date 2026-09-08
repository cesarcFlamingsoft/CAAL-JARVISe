"""Audio processing utilities for CAAL voice agent."""

from .energy_gate import AudioEnergyGate, EnergyGateConfig
from .noise_suppressed_stt import NoiseSuppressedSTT, NoiseSuppressionConfig
from .speaker_recognition import (
    SpeakerProfile,
    SpeakerRecognition,
    SpeakerRecognitionConfig,
    VerificationResult,
    create_speaker_recognition,
)
from .tv_rejection import (
    AudioFeatures,
    TVRejectionConfig,
    TVRejectionFilter,
    create_tv_rejection_filter,
)

__all__ = [
    "NoiseSuppressedSTT",
    "NoiseSuppressionConfig",
    "AudioEnergyGate",
    "EnergyGateConfig",
    "TVRejectionFilter",
    "TVRejectionConfig",
    "AudioFeatures",
    "create_tv_rejection_filter",
    "SpeakerRecognition",
    "SpeakerRecognitionConfig",
    "SpeakerProfile",
    "VerificationResult",
    "create_speaker_recognition",
]
