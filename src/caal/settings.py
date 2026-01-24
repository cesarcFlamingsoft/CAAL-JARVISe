"""Runtime settings management for CAAL voice agent.

Settings persist to a JSON file and are loaded at session start.
Some settings are hot-swappable mid-session.

Settings hierarchy:
1. settings.json - Runtime-configurable values
2. .env - Infrastructure values (URLs, tokens) - fallback only

Prompt files:
- prompt/default.md - Ships with CAAL, used by default
- prompt/custom.md - User's custom prompt (configured via settings menu)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Paths - use environment variable for Docker, fallback for local dev
_SCRIPT_DIR = Path(__file__).parent.parent.parent  # src/caal -> project root
SETTINGS_PATH = Path(os.getenv("CAAL_SETTINGS_PATH", _SCRIPT_DIR / "settings.json"))
PROMPT_DIR = Path(os.getenv("CAAL_PROMPT_DIR", _SCRIPT_DIR / "prompt"))

DEFAULT_SETTINGS = {
    # First-launch flag
    "first_launch_completed": False,
    # Agent identity
    "agent_name": "Cal",
    "prompt": "default",  # "default" | "custom"
    "wake_greetings": [
        "Hey, what's up?",
        "Hi there!",
        "Yeah?",
        "What can I do for you?",
        "Hey!",
        "Yo!",
        "What's up?",
    ],
    # Provider settings (UI sets both together, but stored separately for power users)
    "stt_provider": "speaches",  # "speaches" | "groq"
    "llm_provider": "ollama",  # "ollama" | "groq"
    "tts_provider": "kokoro",  # "kokoro" | "piper"
    # TTS settings - voice selection (Kokoro uses voice param, Piper bakes voice into model)
    "tts_voice_kokoro": "am_puck",
    "tts_voice_piper": "speaches-ai/piper-en_US-ryan-high",
    "temperature": 0.7,
    # Ollama settings
    "ollama_host": "http://localhost:11434",
    "ollama_model": "ministral-3:8b",
    "num_ctx": 8192,
    # Groq settings
    "groq_api_key": "",  # API key from console.groq.com
    "groq_model": "llama-3.3-70b-versatile",
    # Home Assistant integration
    "hass_enabled": False,
    "hass_host": "",
    "hass_token": "",  # Long-lived access token
    "hass_agent_id": "conversation.home_assistant",  # Conversation agent entity_id
    # n8n integration
    "n8n_enabled": False,
    "n8n_url": "",
    "n8n_token": "",
    # Shared settings
    "max_turns": 20,
    "tool_cache_size": 3,
    # Wake word detection (server-side OpenWakeWord)
    "wake_word_enabled": True,
    "wake_word_model": "models/hey_jarvis.onnx",
    "wake_word_threshold": 0.5,  # Detection threshold (0.3-0.7, lower = more sensitive)
    "wake_word_timeout": 3.0,  # seconds of silence before returning to listening
    # Turn detection settings (advanced)
    "allow_interruptions": True,  # Whether user can interrupt agent mid-speech
    "min_endpointing_delay": 0.5,  # Seconds to wait before considering turn complete
    # VAD tuning parameters (voice vs noise separation)
    "vad_min_speech_duration": 0.1,  # Min duration to start speech chunk (filter noise bursts)
    "vad_min_silence_duration": 0.4,  # Wait before ending speech (lower = faster response)
    "vad_prefix_padding": 0.3,  # Padding before speech onset
    "vad_activation_threshold": 0.7,  # Higher = less sensitive to noise (0.0-1.0)
    # Adaptive endpointing (natural conversation flow)
    "adaptive_endpointing_enabled": True,  # Enable context-aware endpointing delays
    "endpointing_delay_after_question": 0.25,  # Short delay after agent asks question
    "endpointing_delay_after_statement": 0.5,  # Normal delay after agent statement
    "endpointing_delay_initial_turns": 0.7,  # Longer delay for first few turns (be patient)
    # Noise suppression (DeepFilterNet)
    "noise_suppression_enabled": False,  # Off by default (requires deepfilternet package)
    "noise_suppression_atten_db": 100.0,  # Attenuation limit in dB (higher = more aggressive)
    # Energy gate (filter quiet/distant sounds like TV)
    "energy_gate_enabled": True,  # On by default - filters TV and distant sounds
    "energy_gate_threshold_db": -35.0,  # RMS threshold in dB (-40=quiet room, -30=speech)
    # TV/Media rejection (advanced spectral/temporal analysis)
    "tv_rejection_enabled": True,  # Filters TV, radio, video playback
    "tv_rejection_min_crest_factor": 1.8,  # Min peak/RMS ratio (stricter filtering)
    "tv_rejection_min_liveness": 0.20,  # Min liveness score (stricter filtering)
    "tv_rejection_consecutive_passes": 3,  # Require N consecutive passes
    "tv_rejection_debug": False,  # Enable debug logging for tuning
    # Media noise filter (music/radio/video detection)
    "media_noise_filter_enabled": True,  # Master switch for media detection
    "media_noise_music_detection": True,  # Detect and reject music (radio/video)
    "media_noise_stereo_detection": True,  # Detect wide stereo (media vs live speech)
    "media_noise_codec_detection": True,  # Detect codec compression artifacts
    "media_noise_max_harmonic_ratio": 0.7,  # Max harmonic ratio before rejecting as music
    "media_noise_max_rhythm_regularity": 0.8,  # Max rhythm regularity before rejecting
    "media_noise_min_stereo_correlation": 0.3,  # Min correlation (live speech is mono-like)
    # Playback voice detection (video with voice)
    "media_noise_playback_voice_detection": True,  # Detect video/podcast voice content
    "media_noise_min_room_reverb": 0.05,  # Min room acoustic ratio
    "media_noise_max_spectral_consistency": 0.85,  # Max spectral consistency
    "media_noise_min_pitch_variation": 0.02,  # Min natural pitch variation
    # Speaker recognition (voice biometrics like Alexa/Google)
    "speaker_recognition_enabled": False,  # Requires resemblyzer package
    "speaker_recognition_threshold": 0.75,  # Verification threshold (0.7-0.85)
    "speaker_recognition_min_duration": 1.0,  # Min audio duration for verification
    "speaker_recognition_required": False,  # Only activate for recognized speakers
    "speaker_recognition_profiles_path": "config/speaker_profiles.json",  # Path to store profiles
    # Visualization type for frontend
    "visualization_type": "jarvis",  # "jarvis" or "soundbars"
}

# Keys that should never be returned via API (security)
SENSITIVE_KEYS: set[str] = set()  # All keys returned - shown as dots in password fields

# Cached settings (reloaded on save)
_settings_cache: dict | None = None


def load_settings() -> dict:
    """Load settings from JSON file, merged with defaults.

    Returns:
        Settings dict with all keys from DEFAULT_SETTINGS,
        overridden by values from settings.json if present.
    """
    global _settings_cache

    if _settings_cache is not None:
        return _settings_cache

    settings = DEFAULT_SETTINGS.copy()

    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH) as f:
                user_settings = json.load(f)
            # Only apply keys that exist in defaults (ignore unknown keys)
            for key in DEFAULT_SETTINGS:
                if key in user_settings:
                    settings[key] = user_settings[key]

            # Migration: existing users who upgraded from before first_launch_completed existed.
            # They would have a settings file with values that differ from defaults.
            # Skip migration if first_launch_completed is already set (either true or false).
            if "first_launch_completed" not in user_settings:
                # Check if user has customized beyond default values (not just having the keys)
                # A true existing user would have changed ollama_host or n8n_url
                env_configured = (
                    os.getenv("OLLAMA_HOST") and user_settings.get("ollama_host") or
                    os.getenv("N8N_MCP_URL") and user_settings.get("n8n_url")
                )
                if env_configured:
                    settings["first_launch_completed"] = True
                    # Migrate .env values to settings
                    settings = _migrate_env_to_settings(settings)
                    # Save migrated settings
                    save_settings(settings)
                    logger.info("Migrated existing user settings")

            logger.debug(f"Loaded settings from {SETTINGS_PATH}")
        except Exception as e:
            logger.warning(f"Failed to load settings from {SETTINGS_PATH}: {e}")
    else:
        logger.debug(f"No settings file at {SETTINGS_PATH}, using defaults")

    _settings_cache = settings
    return settings


def _migrate_env_to_settings(settings: dict) -> dict:
    """Migrate .env values to settings for existing users.

    Args:
        settings: Current settings dict

    Returns:
        Settings with .env values migrated
    """
    # Ollama settings
    if ollama_host := os.getenv("OLLAMA_HOST"):
        settings["ollama_host"] = ollama_host
    if ollama_model := os.getenv("OLLAMA_MODEL"):
        settings["ollama_model"] = ollama_model

    # n8n settings
    if n8n_url := os.getenv("N8N_MCP_URL"):
        settings["n8n_enabled"] = True
        settings["n8n_url"] = n8n_url
    if n8n_token := os.getenv("N8N_MCP_TOKEN"):
        settings["n8n_token"] = n8n_token

    return settings


def load_settings_safe() -> dict:
    """Load settings without sensitive keys (for API responses).

    Returns:
        Settings dict with sensitive keys removed.
    """
    settings = load_settings().copy()
    for key in SENSITIVE_KEYS:
        settings.pop(key, None)
    return settings


def load_user_settings() -> dict:
    """Load only user-specified settings from JSON file (no defaults).

    Returns:
        Dict with only keys explicitly set by the user in settings.json.
        Empty dict if file doesn't exist or is empty.
    """
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load user settings from {SETTINGS_PATH}: {e}")
    return {}


def save_settings(settings: dict) -> None:
    """Save settings to JSON file.

    Args:
        settings: Settings dict to merge with existing settings.
                  Only keys in DEFAULT_SETTINGS are saved.
                  Existing settings are preserved unless explicitly overwritten.
    """
    global _settings_cache

    # Load existing settings first
    existing = load_user_settings()

    # Merge: existing settings + new settings (new overwrites existing)
    merged = {**existing, **settings}

    # Filter to only known keys
    filtered = {k: v for k, v in merged.items() if k in DEFAULT_SETTINGS}

    try:
        # Ensure parent directory exists
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(SETTINGS_PATH, "w") as f:
            json.dump(filtered, f, indent=2)

        # Invalidate cache
        _settings_cache = None

        logger.info(f"Saved settings to {SETTINGS_PATH}")
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")
        raise


def get_setting(key: str, default: Any = None) -> Any:
    """Get a single setting value.

    Args:
        key: Setting key name
        default: Fallback if key not found (defaults to value from DEFAULT_SETTINGS)

    Returns:
        Setting value
    """
    settings = load_settings()
    if default is None:
        default = DEFAULT_SETTINGS.get(key)
    return settings.get(key, default)


def reload_settings() -> dict:
    """Force reload settings from disk.

    Returns:
        Fresh settings dict
    """
    global _settings_cache
    _settings_cache = None
    return load_settings()


# =============================================================================
# Prompt File Management
# =============================================================================


def get_prompt_path(prompt_name: str) -> Path:
    """Get path to a prompt file.

    Args:
        prompt_name: "default" or "custom"

    Returns:
        Path to the prompt .md file
    """
    return PROMPT_DIR / f"{prompt_name}.md"


def load_prompt_content(prompt_name: str | None = None) -> str:
    """Load raw prompt content from file.

    Args:
        prompt_name: "default" or "custom". If None, uses settings["prompt"].

    Returns:
        Prompt file content, or default content if file doesn't exist.
    """
    if prompt_name is None:
        prompt_name = get_setting("prompt", "default")

    prompt_path = get_prompt_path(prompt_name)

    # If custom doesn't exist, fall back to default
    if prompt_name == "custom" and not prompt_path.exists():
        prompt_path = get_prompt_path("default")

    try:
        return prompt_path.read_text()
    except Exception as e:
        logger.error(f"Failed to load prompt from {prompt_path}: {e}")
        return ""


def save_custom_prompt(content: str) -> None:
    """Save content to prompt/custom.md.

    Args:
        content: Prompt content to save
    """
    prompt_path = get_prompt_path("custom")

    try:
        PROMPT_DIR.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(content)
        logger.info(f"Saved custom prompt to {prompt_path}")
    except Exception as e:
        logger.error(f"Failed to save custom prompt: {e}")
        raise


def load_prompt_with_context(
    timezone_id: str = "America/Los_Angeles",
    timezone_display: str = "Pacific Time",
) -> str:
    """Load prompt and populate with date/time context.

    This is the main function used by voice_agent.py to get the
    fully-populated system prompt.

    Args:
        timezone_id: IANA timezone ID for current time
        timezone_display: Human-readable timezone name

    Returns:
        Prompt with {{CURRENT_DATE_CONTEXT}} and {{TIMEZONE}} replaced
    """
    from caal.utils.formatting import (
        format_date_speech_friendly,
        format_time_speech_friendly,
    )

    template = load_prompt_content()

    now = datetime.now(ZoneInfo(timezone_id))
    date_context = (
        f"Today is {format_date_speech_friendly(now)}. "
        f"The current time is {format_time_speech_friendly(now)} {timezone_display}."
    )

    prompt = template.replace("{{CURRENT_DATE_CONTEXT}}", date_context)
    prompt = prompt.replace("{{TIMEZONE}}", timezone_display)

    return prompt


def custom_prompt_exists() -> bool:
    """Check if a custom prompt file exists."""
    return get_prompt_path("custom").exists()
