"""Isolated preset-speaker audition; never touches the live VoiceDesign service."""
from __future__ import annotations

import hashlib
import json
import time
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model

ROOT = Path(__file__).resolve().parent / "customvoice-audition"
MODEL = Path(
    "/Users/cesar/.cache/huggingface/hub/"
    "models--mlx-community--Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit/"
    "snapshots/08c72cad5e2fd0f41730c8bd1f28149585e46361"
)
SAMPLES = {
    "en": "Hello, sir. I have checked the weather and your calendar. How may I help?",
    "es": "Hola, señor. Ya revisé el clima y tu calendario. ¿Cómo puedo ayudarte?",
}
SPEAKERS = ("Aiden", "Ryan")


def write_wav(path: Path, audio: object, sample_rate: int) -> dict[str, object]:
    pcm = (np.clip(np.asarray(audio), -1, 1) * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return {"wav": path.name, "pcm_bytes": len(pcm), "sha256": hashlib.sha256(pcm).hexdigest()}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    model = load_model(MODEL)
    speakers = list(model.get_supported_speakers())
    languages = list(model.get_supported_languages())
    available = {speaker.lower() for speaker in speakers}
    if not {"aiden", "ryan"}.issubset(available):
        raise RuntimeError(f"Expected preset speakers unavailable: {speakers}")
    if not {"english", "spanish"}.issubset(set(x.lower() for x in languages)):
        raise RuntimeError(f"Expected languages unavailable: {languages}")
    output: dict[str, object] = {
        "model_path": str(MODEL),
        "speakers": speakers,
        "languages": languages,
        "instruct": None,
        "seed": 42,
        "samples": [],
    }
    for speaker in SPEAKERS:
        for language, text in SAMPLES.items():
            mx.random.seed(42)
            started = time.monotonic()
            results = model.generate_custom_voice(
                text=text,
                speaker=speaker,
                language={"en": "English", "es": "Spanish"}[language],
                instruct=None,
                temperature=0.0,
                max_tokens=700,
                stream=False,
                verbose=False,
            )
            rendered = list(results)
            if len(rendered) != 1:
                raise RuntimeError(f"Expected one rendering, got {len(rendered)}")
            result = rendered[0]
            row = {
                "speaker": speaker,
                "language": language,
                "seconds": round(time.monotonic() - started, 3),
                "sample_rate": result.sample_rate,
                "text": text,
                **write_wav(ROOT / f"{speaker.lower()}-{language}.wav", result.audio, result.sample_rate),
            }
            output["samples"].append(row)
            print(json.dumps({k: row[k] for k in ("speaker", "language", "seconds", "wav", "pcm_bytes")}))
    (ROOT / "manifest.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
