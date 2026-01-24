"""Audio processing utilities for CAAL voice agent."""

from .noise_suppressed_stt import NoiseSuppressedSTT, NoiseSuppressionConfig
from .energy_gate import AudioEnergyGate, EnergyGateConfig
from .tv_rejection import (
    TVRejectionFilter,
    TVRejectionConfig,
    AudioFeatures,
    create_tv_rejection_filter,
)
from .speaker_recognition import (
    SpeakerRecognition,
    SpeakerRecognitionConfig,
    SpeakerProfile,
    VerificationResult,
    create_speaker_recognition,
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
