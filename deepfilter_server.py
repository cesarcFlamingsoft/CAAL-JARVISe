#!/usr/bin/env python3
"""DeepFilterNet Server - Native macOS service for GPU-accelerated noise suppression.

Runs natively on macOS to leverage MPS (Metal Performance Shaders) for
GPU-accelerated noise suppression. The Docker agent sends audio to this
service for processing.

Usage:
    python deepfilter_server.py [--port 8002] [--host 0.0.0.0]
"""

import argparse
import io
import logging
import struct
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global DeepFilterNet state
df_model = None
df_state = None
DF_SAMPLE_RATE = 48000


def init_deepfilter():
    """Initialize DeepFilterNet model."""
    global df_model, df_state

    try:
        from df import init_df

        # Check MPS availability
        if torch.backends.mps.is_available():
            logger.info("MPS (Metal) GPU available - using GPU acceleration")
        else:
            logger.info("MPS not available - using CPU")

        logger.info("Loading DeepFilterNet model...")
        start = time.time()
        df_model, df_state, _ = init_df()
        elapsed = time.time() - start
        logger.info(f"DeepFilterNet loaded in {elapsed:.2f}s")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize DeepFilterNet: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DeepFilterNet on startup."""
    if not init_deepfilter():
        logger.error("DeepFilterNet initialization failed!")
    yield
    logger.info("Shutting down DeepFilterNet server")


app = FastAPI(
    title="DeepFilterNet Server",
    description="GPU-accelerated noise suppression for CAAL",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": df_model is not None,
        "mps_available": torch.backends.mps.is_available(),
        "sample_rate": DF_SAMPLE_RATE,
    }


@app.post("/enhance")
async def enhance_audio(request: Request):
    """Enhance audio by removing noise.

    Expects raw PCM audio (int16, mono, 48kHz) in request body.
    Returns enhanced PCM audio in same format.

    Query params:
        atten_lim_db: Attenuation limit in dB (default: 100)
    """
    global df_model, df_state

    if df_model is None:
        raise HTTPException(status_code=503, detail="DeepFilterNet not initialized")

    try:
        from df import enhance

        # Get attenuation limit from query params
        atten_lim_db = float(request.query_params.get("atten_lim_db", 100))

        # Read raw PCM data
        pcm_data = await request.body()

        if len(pcm_data) == 0:
            return Response(content=b"", media_type="application/octet-stream")

        # Convert to numpy array (int16)
        audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)

        # Convert to float32 [-1, 1] for DeepFilterNet
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # DeepFilterNet expects (samples,) or (batch, samples)
        # Process through DeepFilterNet
        enhanced = enhance(
            df_model,
            df_state,
            audio_float,
            atten_lim_db=atten_lim_db,
        )

        # Convert back to int16
        enhanced_int16 = (enhanced * 32768.0).clip(-32768, 32767).astype(np.int16)

        return Response(
            content=enhanced_int16.tobytes(),
            media_type="application/octet-stream",
        )

    except Exception as e:
        logger.error(f"Error enhancing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enhance_batch")
async def enhance_audio_batch(request: Request):
    """Enhance multiple audio chunks in a single request.

    Expects JSON with:
        - chunks: list of base64-encoded PCM audio chunks
        - atten_lim_db: optional attenuation limit

    Returns JSON with:
        - chunks: list of base64-encoded enhanced audio chunks
    """
    global df_model, df_state

    if df_model is None:
        raise HTTPException(status_code=503, detail="DeepFilterNet not initialized")

    try:
        import base64
        from df import enhance

        data = await request.json()
        chunks = data.get("chunks", [])
        atten_lim_db = float(data.get("atten_lim_db", 100))

        enhanced_chunks = []
        for chunk_b64 in chunks:
            # Decode base64
            pcm_data = base64.b64decode(chunk_b64)

            if len(pcm_data) == 0:
                enhanced_chunks.append("")
                continue

            # Convert to numpy
            audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0

            # Enhance
            enhanced = enhance(
                df_model,
                df_state,
                audio_float,
                atten_lim_db=atten_lim_db,
            )

            # Convert back
            enhanced_int16 = (enhanced * 32768.0).clip(-32768, 32767).astype(np.int16)

            # Encode to base64
            enhanced_chunks.append(base64.b64encode(enhanced_int16.tobytes()).decode())

        return JSONResponse({"chunks": enhanced_chunks})

    except Exception as e:
        logger.error(f"Error in batch enhance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="DeepFilterNet Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to listen on")
    args = parser.parse_args()

    logger.info(f"Starting DeepFilterNet server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
