"""Private, opt-in Qwen audition endpoint. No profile files or arbitrary models accepted.

Bilingual as of the en/es activation, with three deliberate properties:

* ``language`` is per request and defaults to ``"en"``, so a client that sends
  the old body -- the one this service answered before -- gets byte-for-byte the
  English behaviour it had.
* An unsupported language is a **400, not English**. Silently speaking English
  at someone who asked for Spanish is the failure mode this slice exists to
  avoid.
* ``/voice`` still reports the English ``style``/``style_sha256`` that the
  startup recovery manifest pins, and now also reports every approved design.
"""

import asyncio
import hashlib
import secrets
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from trial_model import (
    DEFAULT_LANGUAGE,
    LANG_CODES,
    MODEL,
    REVISION,
    SEED,
    STYLE,
    STYLES,
    SUPPORTED,
    UnsupportedLanguage,
)
from trial_stream import BusyError


class Speech(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: str = Field(min_length=1, max_length=600)
    model: Literal["qwen-trial"] = "qwen-trial"
    voice: Literal["jarvis-designed"] = "jarvis-designed"
    response_format: Literal["pcm"] = "pcm"
    #: Per request. The service holds no language state between requests.
    language: Literal["en", "es"] = DEFAULT_LANGUAGE

    @field_validator("input")
    @classmethod
    def nonblank(cls, text):
        if not text.strip():
            raise ValueError("Empty text")
        return text


async def prefetch(stream, request):
    async def disconnected():
        while True:
            if (await request.receive())["type"] == "http.disconnect":
                return

    first = asyncio.create_task(anext(stream))
    gone = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait([first, gone], return_when=asyncio.FIRST_COMPLETED)
        if gone in done:
            raise asyncio.CancelledError()
        return first.result()
    finally:
        gone.cancel()
        first.cancel()
        await asyncio.gather(gone, first, return_exceptions=True)


def create_app(engine, token, *, prewarm=False):
    if not isinstance(token, str) or len(token) < 32 or not token.isascii():
        raise ValueError("A private token of at least 32 ASCII characters is required")

    @asynccontextmanager
    async def lifespan(app):
        try:
            if prewarm:
                # Prewarm each immutable language path. Spanish is a separate
                # Base clone model and must be warm before the first live turn.
                warmup = {
                    "en": "Understood. I am ready to help.",
                    "es": "Entendido. Estoy lista para ayudar.",
                }
                for language in SUPPORTED:
                    async for _ in engine.stream(warmup[language], language):
                        pass
            yield
        finally:
            await engine.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def authenticate(authorization: str = Header(default="")):
        if not authorization.isascii() or not secrets.compare_digest(
            authorization, "Bearer " + token
        ):
            raise HTTPException(401, "Unauthorized")

    @app.get("/health", dependencies=[Depends(authenticate)])
    async def health():
        return {"status": "ok", "busy": engine.busy, "languages": list(SUPPORTED)}

    @app.get("/voice", dependencies=[Depends(authenticate)])
    async def voice():
        """Read back the design loaded by this service process, without credentials."""
        return {
            "model": MODEL,
            "revision": REVISION,
            "voice": "jarvis-designed",
            "seed": SEED,
            # The English design, under the key the startup recovery manifest
            # pins. Unchanged, and deliberately still top level.
            "style": STYLE,
            "style_sha256": hashlib.sha256(STYLE.encode()).hexdigest(),
            "default_language": DEFAULT_LANGUAGE,
            "designs": {
                code: {
                    "lang_code": LANG_CODES[code],
                    "style": STYLES[code],
                    "style_sha256": hashlib.sha256(STYLES[code].encode()).hexdigest(),
                }
                for code in SUPPORTED
            },
        }

    @app.post("/v1/audio/speech", dependencies=[Depends(authenticate)])
    async def speech(body: Speech, request: Request):
        stream = engine.stream(body.input, body.language)
        try:
            first = await prefetch(stream, request)
        except BusyError:
            raise HTTPException(409, "Synthesis busy") from None
        except UnsupportedLanguage:
            # Never degrade to English here; the caller must know.
            raise HTTPException(400, "Unsupported language") from None
        except TimeoutError:
            raise HTTPException(504, "Synthesis timed out") from None
        except Exception:
            raise HTTPException(502, "Synthesis failed") from None

        async def output():
            try:
                yield first
                async for chunk in stream:
                    yield chunk
            finally:
                await stream.aclose()

        return StreamingResponse(
            output(),
            media_type="audio/pcm",
            headers={
                "X-Audio-Sample-Rate": "24000",
                "X-Audio-Channels": "1",
                "X-Audio-Language": body.language,
                "Cache-Control": "no-store",
            },
        )

    return app
