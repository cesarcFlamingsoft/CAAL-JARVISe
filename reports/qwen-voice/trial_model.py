"""Pinned, per-request Qwen backend for the approved female FRIDAY voice.

English uses the approved VoiceDesign direction. Spanish uses Qwen Base voice
cloning from the corresponding approved synthetic English FRIDAY reference.
Language selects immutable configuration for that request; no shared speaker or
language state can leak between sessions.
"""

from pathlib import Path

import numpy as np

MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit"
REVISION = "5c390979e4b93af5f2932f90742ca99c7dd04687"
MODEL_PATH = (
    "/Users/cesar/.cache/huggingface/hub/models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit/snapshots/"
    + REVISION
)
SPANISH_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit"
SPANISH_REVISION = "37e955a1deb861c088ae5f3a67043185f3d1a60c"
SPANISH_MODEL_PATH = (
    "/Users/cesar/.cache/huggingface/models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-4bit/snapshots/"
    + SPANISH_REVISION
)

STYLE = (
    "A poised adult British woman in her late thirties with a clear, warm, naturally feminine "
    "mid-low voice and a subtle standard southern English accent. She is a calm, highly capable "
    "personal assistant speaking to her principal: precise, reassuring and quietly confident. "
    "Natural conversational pace, controlled pitch movement and brief thoughtful pauses. Elegant "
    "and human, never robotic, girlish, breathy, theatrical, overly cheerful, or like a "
    "commercial announcer."
)
#: Metadata only. Spanish synthesis uses the immutable clone reference below.
SPANISH_STYLE = "Approved female FRIDAY synthetic-reference clone."
SPANISH_REFERENCE_AUDIO = (
    Path(__file__).with_name("friday-female-audition") / "friday-female-en.wav"
)
SPANISH_REFERENCE_TEXT = (
    "Good evening, sir. I have checked the weather and your calendar. "
    "The evening is clear, and you have no meetings remaining today."
)

LANG_CODES = {"en": "English", "es": "Spanish"}
STYLES = {"en": STYLE, "es": SPANISH_STYLE}
SUPPORTED = tuple(LANG_CODES)
DEFAULT_LANGUAGE = "en"
SEED = 42
MAX_TOKENS = 700


class UnsupportedLanguage(ValueError):  # noqa: N818 - public API compatibility
    """Asked for a language this approved voice has no configuration for."""


def design_for(language):
    """The language code and immutable configuration label for a request."""
    code = DEFAULT_LANGUAGE if language is None else language
    if code not in LANG_CODES:
        raise UnsupportedLanguage(f"No approved voice design for language {code!r}")
    return LANG_CODES[code], STYLES[code]


class QwenModel:
    def __init__(self, load=None, seed=None, interval=0.24):
        if load is None:
            from mlx_audio.tts.utils import load_model

            load = load_model
        if seed is None:
            import mlx.core as mx

            mx.set_memory_limit(6 * 1024**3)
            mx.set_cache_limit(256 * 1024**2)
            seed = mx.random.seed
        self.load = load
        self.seed = seed
        self.interval = interval
        self.models = {}

    def _model(self, path):
        if path not in self.models:
            self.models[path] = self.load(path)
        return self.models[path]

    def generate(self, text, language=DEFAULT_LANGUAGE):
        lang_code, _ = design_for(language)
        self.seed(SEED)
        if language in (None, "en"):
            model = self._model(MODEL_PATH)
            results = model.generate(
                text,
                instruct=STYLE,
                lang_code=lang_code,
                stream=True,
                streaming_interval=self.interval,
                max_tokens=MAX_TOKENS,
                verbose=False,
            )
        else:
            if not SPANISH_REFERENCE_AUDIO.is_file():
                raise FileNotFoundError("Approved Spanish clone reference is missing")
            model = self._model(SPANISH_MODEL_PATH)
            results = model.generate(
                text,
                lang_code=lang_code,
                ref_audio=str(SPANISH_REFERENCE_AUDIO),
                ref_text=SPANISH_REFERENCE_TEXT,
                stream=True,
                streaming_interval=self.interval,
                max_tokens=MAX_TOKENS,
                verbose=False,
            )
        try:
            for result in results:
                audio = np.asarray(result.audio)
                if result.sample_rate != 24000 or audio.ndim != 1 or not np.isfinite(audio).all():
                    raise ValueError("Invalid model audio")
                yield (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        finally:
            close = getattr(results, "close", None)
            if close:
                close()
            model.speech_tokenizer.decoder.reset_streaming_state()
