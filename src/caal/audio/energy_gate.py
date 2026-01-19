"""Audio energy gate for filtering distant/quiet sounds.

This module provides a simple but effective filter that distinguishes
between nearby speech (loud) and distant sounds like TV (quiet).

The energy gate calculates RMS (root mean square) energy of audio frames
and only passes through frames above a configurable threshold.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from livekit import rtc

logger = logging.getLogger(__name__)


@dataclass
class EnergyGateConfig:
    """Configuration for audio energy gate."""

    enabled: bool = True
    # RMS threshold in dB (relative to full scale)
    # -40 dB is quiet room, -30 dB is normal speech, -20 dB is loud speech
    # Default -35 dB filters out most TV/background but passes nearby speech
    threshold_db: float = -35.0
    # Minimum consecutive frames above threshold to "open" the gate
    # Prevents single loud noises from triggering
    min_frames_above: int = 2
    # Frames to keep gate open after speech stops (smoothing)
    hold_frames: int = 5
    # Attack time - how quickly gate opens (frames)
    attack_frames: int = 1
    # Release time - how quickly gate closes (frames)
    release_frames: int = 3


class AudioEnergyGate:
    """Energy-based audio gate for filtering distant sounds.

    Calculates RMS energy of audio frames and gates based on threshold.
    Nearby speech (louder) passes through, TV/distant sounds (quieter) are muted.

    Usage:
        gate = AudioEnergyGate(config)
        for frame in audio_stream:
            if gate.should_pass(frame):
                process(frame)
            else:
                # Frame is below threshold, likely TV/background
                pass
    """

    def __init__(self, config: EnergyGateConfig | None = None):
        """Initialize energy gate.

        Args:
            config: Gate configuration
        """
        self.config = config or EnergyGateConfig()
        self._frames_above = 0
        self._hold_counter = 0
        self._gate_open = False
        self._smoothed_db = -60.0  # Start closed

        # Convert threshold to linear for faster comparison
        self._threshold_linear = 10 ** (self.config.threshold_db / 20)

        logger.info(
            f"AudioEnergyGate initialized: threshold={self.config.threshold_db}dB, "
            f"min_frames={self.config.min_frames_above}, hold={self.config.hold_frames}"
        )

    @classmethod
    def from_settings(cls, settings: dict) -> "AudioEnergyGate | None":
        """Create from settings dict."""
        enabled = settings.get("energy_gate_enabled", True)
        if not enabled:
            return None

        config = EnergyGateConfig(
            enabled=True,
            threshold_db=settings.get("energy_gate_threshold_db", -35.0),
            min_frames_above=settings.get("energy_gate_min_frames", 2),
            hold_frames=settings.get("energy_gate_hold_frames", 5),
        )
        return cls(config)

    def calculate_rms_db(self, frame: "rtc.AudioFrame") -> float:
        """Calculate RMS energy in dB for an audio frame.

        Args:
            frame: LiveKit audio frame

        Returns:
            RMS energy in dB (relative to full scale, typically -60 to 0)
        """
        # Convert to numpy array
        audio = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)

        # Handle multi-channel by averaging
        if frame.num_channels > 1:
            audio = audio.reshape(-1, frame.num_channels).mean(axis=1)

        # Normalize to [-1, 1]
        audio = audio / 32768.0

        # Calculate RMS
        rms = np.sqrt(np.mean(audio ** 2))

        # Convert to dB (with floor to avoid log(0))
        if rms < 1e-10:
            return -100.0
        return 20 * math.log10(rms)

    def should_pass(self, frame: "rtc.AudioFrame") -> bool:
        """Determine if frame should pass through the gate.

        Args:
            frame: LiveKit audio frame

        Returns:
            True if frame should be processed, False if it should be gated
        """
        if not self.config.enabled:
            return True

        current_db = self.calculate_rms_db(frame)

        # Check if above threshold
        above_threshold = current_db >= self.config.threshold_db

        if above_threshold:
            self._frames_above += 1
            self._hold_counter = self.config.hold_frames

            # Open gate after minimum consecutive frames
            if self._frames_above >= self.config.min_frames_above:
                if not self._gate_open:
                    logger.debug(f"Gate OPEN: {current_db:.1f}dB")
                self._gate_open = True
        else:
            self._frames_above = 0

            # Hold gate open for smoothing
            if self._hold_counter > 0:
                self._hold_counter -= 1
            else:
                if self._gate_open:
                    logger.debug(f"Gate CLOSED: {current_db:.1f}dB")
                self._gate_open = False

        return self._gate_open

    def process_frame(self, frame: "rtc.AudioFrame") -> "rtc.AudioFrame | None":
        """Process frame through gate, returning None if gated.

        Args:
            frame: Input audio frame

        Returns:
            Frame if passed, None if gated
        """
        if self.should_pass(frame):
            return frame
        return None

    def get_status(self) -> dict:
        """Get current gate status for debugging."""
        return {
            "gate_open": self._gate_open,
            "frames_above": self._frames_above,
            "hold_counter": self._hold_counter,
            "threshold_db": self.config.threshold_db,
        }

    def reset(self) -> None:
        """Reset gate state."""
        self._frames_above = 0
        self._hold_counter = 0
        self._gate_open = False
        self._smoothed_db = -60.0
