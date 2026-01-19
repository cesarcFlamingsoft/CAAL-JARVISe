#!/usr/bin/env python3
"""Noise Suppression Server - Native macOS service for audio noise reduction.

Supports two backends:
1. DeepFilterNet (if available) - Neural network based, GPU accelerated
2. noisereduce (fallback) - Spectral gating, CPU based but fast

Runs natively on macOS. The Docker agent sends audio to this service for processing.

Usage:
    python deepfilter_server.py [--port 8002] [--host 0.0.0.0]
"""

import argparse
import logging
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
noise_reducer = None
backend_name = None
SAMPLE_RATE = 48000


class DeepFilterBackend:
    """DeepFilterNet backend for GPU-accelerated noise suppression."""

    def __init__(self):
        from df import init_df
        import torch

        self.mps_available = torch.backends.mps.is_available()
        logger.info(f"Initializing DeepFilterNet (MPS={self.mps_available})...")

        self.model, self.state, _ = init_df()
        logger.info("DeepFilterNet initialized")

    def enhance(self, audio: np.ndarray, atten_lim_db: float = 100.0) -> np.ndarray:
        from df import enhance
        return enhance(self.model, self.state, audio, atten_lim_db=atten_lim_db)

    @property
    def name(self) -> str:
        return "deepfilter"


class NoiseReduceBackend:
    """noisereduce backend for spectral gating noise suppression."""

    def __init__(self):
        import noisereduce as nr
        self.nr = nr
        logger.info("Initialized noisereduce (spectral gating)")

    def enhance(self, audio: np.ndarray, atten_lim_db: float = 100.0) -> np.ndarray:
        # noisereduce expects float audio in [-1, 1] range
        # prop_decrease controls how much noise is reduced (0-1)
        prop_decrease = min(1.0, atten_lim_db / 100.0)
        return self.nr.reduce_noise(
            y=audio,
            sr=SAMPLE_RATE,
            prop_decrease=prop_decrease,
            stationary=False,  # Non-stationary for speech
        )

    @property
    def name(self) -> str:
        return "noisereduce"


def init_backend():
    """Initialize the best available noise suppression backend."""
    global noise_reducer, backend_name

    # Try DeepFilterNet first
    try:
        noise_reducer = DeepFilterBackend()
        backend_name = "deepfilter"
        return True
    except Exception as e:
        logger.warning(f"DeepFilterNet not available: {e}")

    # Fall back to noisereduce
    try:
        noise_reducer = NoiseReduceBackend()
        backend_name = "noisereduce"
        return True
    except Exception as e:
        logger.error(f"noisereduce not available: {e}")

    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize noise suppression on startup."""
    if not init_backend():
        logger.error("No noise suppression backend available!")
    else:
        logger.info(f"Using {backend_name} backend for noise suppression")
    yield
    logger.info("Shutting down noise suppression server")


app = FastAPI(
    title="Noise Suppression Server",
    description="GPU-accelerated noise suppression for CAAL",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    import torch

    return {
        "status": "ok",
        "model_loaded": noise_reducer is not None,
        "backend": backend_name,
        "mps_available": torch.backends.mps.is_available() if backend_name == "deepfilter" else False,
        "sample_rate": SAMPLE_RATE,
    }


@app.post("/enhance")
async def enhance_audio(request: Request):
    """Enhance audio by removing noise.

    Expects raw PCM audio (int16, mono, 48kHz) in request body.
    Returns enhanced PCM audio in same format.

    Query params:
        atten_lim_db: Attenuation limit in dB (default: 100)
    """
    if noise_reducer is None:
        raise HTTPException(status_code=503, detail="Noise suppression not initialized")

    try:
        # Get attenuation limit from query params
        atten_lim_db = float(request.query_params.get("atten_lim_db", 100))

        # Read raw PCM data
        pcm_data = await request.body()

        if len(pcm_data) == 0:
            return Response(content=b"", media_type="application/octet-stream")

        # Convert to numpy array (int16)
        audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)

        # Convert to float32 [-1, 1]
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Process through noise reducer
        enhanced = noise_reducer.enhance(audio_float, atten_lim_db=atten_lim_db)

        # Ensure output is the right shape
        if isinstance(enhanced, np.ndarray):
            enhanced = enhanced.squeeze()

        # Convert back to int16
        enhanced_int16 = (enhanced * 32768.0).clip(-32768, 32767).astype(np.int16)

        return Response(
            content=enhanced_int16.tobytes(),
            media_type="application/octet-stream",
        )

    except Exception as e:
        logger.error(f"Error enhancing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Noise Suppression Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to listen on")
    args = parser.parse_args()

    logger.info(f"Starting noise suppression server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
