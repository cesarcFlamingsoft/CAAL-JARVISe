"""Render only the faster Spanish variant of the approved female FRIDAY direction."""
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
STYLE = (
    "Una mujer adulta británica de unos cuarenta años, con una voz femenina clara, cálida y "
    "de registro medio-grave. Habla español neutro con pronunciación natural, como una asistente "
    "personal muy capaz: precisa, serena, cercana y discretamente segura. Mantiene un ritmo "
    "conversacional ágil y fluido, ligeramente más rápido de lo normal, con pausas cortas y "
    "funcionales. Elegante y humana; nunca robótica, infantil, chillona, jadeante, teatral, lenta, "
    "ni con palabras arrastradas o tono de locutora comercial."
)
TEXT = "Buenas noches, señor. Ya revisé el clima y tu calendario. La noche está despejada y no tienes más reuniones hoy."
OUT = Path(__file__).with_name("friday-female-audition") / "spanish-faster-v2"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_model(MODEL)
    mx.random.seed(42)
    started = time.monotonic()
    results = list(model.generate(TEXT, instruct=STYLE, lang_code="Spanish", stream=False, max_tokens=700, verbose=False))
    if len(results) != 1:
        raise RuntimeError(f"Expected one result, got {len(results)}")
    result = results[0]
    pcm = (np.clip(np.asarray(result.audio), -1, 1) * 32767).astype("<i2").tobytes()
    if not pcm:
        raise RuntimeError("Empty audio")
    output = OUT / "friday-female-es-faster.wav"
    with wave.open(str(output), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(result.sample_rate)
        audio.writeframes(pcm)
    manifest = {
        "purpose": "isolated faster Spanish female FRIDAY audition; live service unchanged",
        "style": STYLE,
        "text": TEXT,
        "sample_rate": result.sample_rate,
        "seconds": round(len(pcm) / 2 / result.sample_rate, 3),
        "render_seconds": round(time.monotonic() - started, 3),
        "sha256": hashlib.sha256(pcm).hexdigest(),
        "file": output.name,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
