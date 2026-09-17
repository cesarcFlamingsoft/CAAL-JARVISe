"""Isolated Qwen VoiceDesign audition; never imports or changes the live service."""
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
    "Un único hombre adulto hispanohablante, de unos cuarenta años, con una voz "
    "claramente masculina, grave, estable y cálida. Mantiene exactamente la misma "
    "identidad vocal en toda la respuesta. Habla español neutro con pronunciación nativa, "
    "ritmo natural y pausas breves, como un asistente profesional que conversa con una "
    "persona cercana. Sereno, atento y expresivo con moderación; nunca agudo, femenino, "
    "cantado, teatral, lento ni con tono de locutor."
)
TEXTS = {
    "weather": "Buenos días. El clima está despejado y seco en Edmonton. La humedad es moderada y el viento se mantiene suave.",
    "calendar": "He revisado tu agenda. Tienes la tarde libre y no hay reuniones pendientes. ¿En qué más puedo ayudarte?",
}
OUT = Path(__file__).with_name("spanish-identity-lock-audition")


def render(model, name: str, text: str) -> dict[str, object]:
    mx.random.seed(42)
    started = time.monotonic()
    parts = list(
        model.generate(
            text,
            instruct=STYLE,
            lang_code="Spanish",
            stream=False,
            max_tokens=700,
            verbose=False,
        )
    )
    if len(parts) != 1:
        raise RuntimeError(f"Expected one result for {name}, got {len(parts)}")
    result = parts[0]
    pcm = (np.clip(np.asarray(result.audio), -1, 1) * 32767).astype("<i2").tobytes()
    if not pcm:
        raise RuntimeError(f"Empty audio for {name}")
    path = OUT / f"es-lock-{name}.wav"
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
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_model(MODEL)
    items = [render(model, name, text) for name, text in TEXTS.items()]
    manifest = {
        "purpose": "isolated stricter Spanish male identity audition; live service unchanged",
        "model": str(MODEL),
        "seed": 42,
        "style": STYLE,
        "items": items,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
