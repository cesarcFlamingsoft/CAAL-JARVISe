"""Isolated female-FRIDAY Qwen audition; it never changes the live service."""
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
    "/Users/cesar/.cache/huggingface/hub/"
    "models--mlx-community--Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit/"
    "snapshots/5c390979e4b93af5f2932f90742ca99c7dd04687"
)
STYLES = {
    "en": (
        "A poised adult British woman in her late thirties with a clear, warm, naturally "
        "feminine mid-low voice and a subtle standard southern English accent. She is a calm, "
        "highly capable personal assistant speaking to her principal: precise, reassuring and "
        "quietly confident. Natural conversational pace, controlled pitch movement and brief "
        "thoughtful pauses. Elegant and human, never robotic, girlish, breathy, theatrical, "
        "overly cheerful, or like a commercial announcer."
    ),
    "es": (
        "Una mujer adulta británica de unos cuarenta años, con una voz femenina clara, cálida y "
        "de registro medio-grave. Habla español neutro con pronunciación natural, como una asistente "
        "personal muy capaz: precisa, serena, cercana y discretamente segura. Ritmo conversacional, "
        "entonación controlada y pausas breves. Elegante y humana; nunca robótica, infantil, chillona, "
        "jadeante, teatral ni con tono de locutora comercial."
    ),
}
TEXTS = {
    "en": "Good evening, sir. I have checked the weather and your calendar. The evening is clear, and you have no meetings remaining today.",
    "es": "Buenas noches, señor. Ya revisé el clima y tu calendario. La noche está despejada y no tienes más reuniones hoy.",
}
OUT = Path(__file__).with_name("friday-female-audition")


def render(model, language: str) -> dict[str, object]:
    mx.random.seed(42)
    started = time.monotonic()
    parts = list(
        model.generate(
            TEXTS[language],
            instruct=STYLES[language],
            lang_code={"en": "English", "es": "Spanish"}[language],
            stream=False,
            max_tokens=700,
            verbose=False,
        )
    )
    if len(parts) != 1:
        raise RuntimeError(f"Expected one result for {language}, got {len(parts)}")
    result = parts[0]
    pcm = (np.clip(np.asarray(result.audio), -1, 1) * 32767).astype("<i2").tobytes()
    if not pcm:
        raise RuntimeError(f"Empty audio for {language}")
    path = OUT / f"friday-female-{language}.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(result.sample_rate)
        audio.writeframes(pcm)
    return {
        "file": path.name,
        "text": TEXTS[language],
        "style": STYLES[language],
        "seconds": round(len(pcm) / 2 / result.sample_rate, 3),
        "render_seconds": round(time.monotonic() - started, 3),
        "sample_rate": result.sample_rate,
        "sha256": hashlib.sha256(pcm).hexdigest(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_model(MODEL)
    items = [render(model, language) for language in ("en", "es")]
    manifest = {
        "purpose": "isolated female FRIDAY direction audition; live service unchanged",
        "inspiration": "MCU F.R.I.D.A.Y. qualities only, not a voice clone",
        "model": str(MODEL),
        "seed": 42,
        "items": items,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
