"""Render one CustomVoice audition in a fresh process/model instance."""
from __future__ import annotations

import argparse
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
    "models--mlx-community--Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit/"
    "snapshots/08c72cad5e2fd0f41730c8bd1f28149585e46361"
)
TEXTS = {
    "en": "Hello, sir. I have checked the weather and your calendar. How may I help?",
    "es": "Hola, señor. Ya revisé el clima y tu calendario. ¿Cómo puedo ayudarte?",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", required=True, choices=("aiden", "ryan", "uncle_fu"))
    parser.add_argument("--language", required=True, choices=("en", "es"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model = load_model(MODEL)
    mx.random.seed(42)
    results = list(
        model.generate_custom_voice(
            text=TEXTS[args.language],
            speaker=args.speaker,
            language={"en": "English", "es": "Spanish"}[args.language],
            instruct=None,
            # Qwen's CustomVoice decoder expects sampled generation.  A fixed MLX
            # seed makes the normal 0.9 sampler reproducible; forcing greedy
            # decoding can collapse non-English/preset outputs into noise.
            temperature=0.9,
            max_tokens=700,
            stream=False,
            verbose=False,
        )
    )
    if len(results) != 1:
        raise RuntimeError(f"Expected one result, got {len(results)}")
    result = results[0]
    pcm = (np.clip(np.asarray(result.audio), -1, 1) * 32767).astype("<i2").tobytes()
    if not pcm:
        raise RuntimeError("Empty audio")
    with wave.open(str(args.output), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(result.sample_rate)
        handle.writeframes(pcm)
    print(json.dumps({
        "speaker": args.speaker,
        "language": args.language,
        "seconds": round(time.monotonic() - started, 3),
        "sample_rate": result.sample_rate,
        "pcm_bytes": len(pcm),
        "sha256": hashlib.sha256(pcm).hexdigest(),
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
