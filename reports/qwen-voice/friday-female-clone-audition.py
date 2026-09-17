"""Isolated Qwen Base voice-clone audition from the approved synthetic FRIDAY reference."""
from __future__ import annotations

import hashlib
import json
import time
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model

MODEL = Path(
    "/Users/cesar/.cache/huggingface/"
    "models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-4bit/"
    "snapshots/37e955a1deb861c088ae5f3a67043185f3d1a60c"
)
REFERENCE_AUDIO = Path(__file__).with_name("friday-female-audition") / "friday-female-en.wav"
REFERENCE_TEXT = (
    "Good evening, sir. I have checked the weather and your calendar. "
    "The evening is clear, and you have no meetings remaining today."
)
TEXTS = {
    "weather": "Buenas noches, señor. Ya revisé el clima y tu calendario. La noche está despejada y no tienes más reuniones hoy.",
    "calendar": "He revisado tu agenda para mañana. Tu primera reunión es a las nueve y todavía tienes tiempo para prepararte con calma.",
}
OUT = Path(__file__).with_name("friday-female-clone-audition")


def render(model, name: str, text: str) -> dict[str, object]:
    mx.random.seed(42)
    started = time.monotonic()
    results = list(
        model.generate(
            text,
            lang_code="Spanish",
            ref_audio=str(REFERENCE_AUDIO),
            ref_text=REFERENCE_TEXT,
            stream=False,
            max_tokens=700,
            verbose=False,
        )
    )
    if len(results) != 1:
        raise RuntimeError(f"Expected one result for {name}, got {len(results)}")
    result = results[0]
    pcm = (np.clip(np.asarray(result.audio), -1, 1) * 32767).astype("<i2").tobytes()
    if not pcm:
        raise RuntimeError(f"Empty audio for {name}")
    path = OUT / f"friday-female-clone-es-{name}.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(result.sample_rate)
        audio.writeframes(pcm)
    return {
        "file": path.name,
        "text": text,
        "seconds": round(len(pcm) / 2 / result.sample_rate, 3),
        "render_seconds": round(time.monotonic() - started, 3),
        "sample_rate": result.sample_rate,
        "sha256": hashlib.sha256(pcm).hexdigest(),
    }


def main() -> None:
    if not REFERENCE_AUDIO.is_file():
        raise FileNotFoundError(REFERENCE_AUDIO)
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_model(MODEL)
    items = [render(model, name, text) for name, text in TEXTS.items()]
    manifest = {
        "purpose": "isolated Qwen Base cross-language clone audition; live service unchanged",
        "model": str(MODEL),
        "reference": {"audio": str(REFERENCE_AUDIO), "text": REFERENCE_TEXT, "synthetic": True},
        "seed": 42,
        "items": items,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
