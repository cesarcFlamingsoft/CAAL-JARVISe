"""Pinned MLX VoiceDesign backend; uses no reference recording."""

import numpy as np

MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit"
REVISION = "5c390979e4b93af5f2932f90742ca99c7dd04687"
MODEL_PATH = (
    "/Users/cesar/.cache/huggingface/hub/models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit/snapshots/"
    + REVISION
)
STYLE = 'A British man in his late thirties with a natural standard southern English accent and a smooth, clear mid-range male voice. He is speaking normally to a colleague beside him: polite, friendly and matter-of-fact. An easy everyday speaking pace, modest pitch variation and short natural pauses. Clear and engaged, without dramatic emphasis.'


class QwenModel:
    def __init__(self, load=None, seed=None, interval=0.24):
        if load is None:
            from mlx_audio.tts.utils import load_model

            load = load_model
        if seed is None:
            import mlx.core as mx

            mx.set_memory_limit(4 * 1024**3)
            mx.set_cache_limit(256 * 1024**2)
            seed = mx.random.seed
        self.load = load
        self.seed = seed
        self.interval = interval
        self.model = None

    def generate(self, text):
        if self.model is None:
            self.model = self.load(MODEL_PATH)
        self.seed(42)
        results = self.model.generate(
            text,
            instruct=STYLE,
            lang_code="English",
            stream=True,
            streaming_interval=self.interval,
            max_tokens=700,
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
            self.model.speech_tokenizer.decoder.reset_streaming_state()
