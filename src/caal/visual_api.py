"""One-shot, user-bound visual analysis through the configured local Ollama only."""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import os
import threading
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from . import settings as settings_module
from . import user_api
from .internal_auth import RateLimiter
from .local_ollama import is_model_name, normalize_endpoint, resolve_local_alias
from .user_api import CurrentUser, IdentityRuntime, require_user

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_IMAGE_BYTES",
    "MAX_IMAGE_EDGE",
    "MAX_IMAGE_PIXELS",
    "VisualRuntime",
    "decode_jpeg",
    "compact_jpeg_for_model",
    "get_visual_runtime",
    "router",
]

MAX_IMAGE_EDGE = 640
MAX_IMAGE_PIXELS = MAX_IMAGE_EDGE * MAX_IMAGE_EDGE
MAX_IMAGE_BYTES = 400 * 1024
# Gemma receives a separately re-encoded model input: keeping it at 384px
# prevents a high-detail browser frame from producing empty visual completions.
MODEL_IMAGE_EDGE = 384
MODEL_JPEG_QUALITY = 70
MAX_DECODED_BYTES = MAX_IMAGE_PIXELS * 3
MAX_BASE64_CHARS = ((MAX_IMAGE_BYTES + 2) // 3) * 4
MAX_REQUEST_BYTES = MAX_BASE64_CHARS + 1024
MAX_PROMPT_CHARS = 240
VISUAL_PROMPT = "Briefly describe what is visible in this camera view."
MAX_OUTPUT_CHARS = 1200
# Ollama's /api/show includes model metadata and can exceed the bounded chat
# response envelope; only its capability list is used here.
MAX_SHOW_BYTES = 256 * 1024
MAX_UPSTREAM_BYTES = 64 * 1024
ANALYSIS_LIMIT_PER_MINUTE = 6
OLLAMA_TIMEOUT = httpx.Timeout(connect=2.0, read=25.0, write=5.0, pool=2.0)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class VisualRequest(_Strict):
    image: str = Field(min_length=4, max_length=MAX_BASE64_CHARS)
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    company_private: bool

    @field_validator("prompt")
    @classmethod
    def safe_prompt(cls, value: str) -> str:
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("control character")
        if value != VISUAL_PROMPT:
            raise ValueError("unsupported prompt")
        return value


class VisualResponse(_Strict):
    description: str


def decode_jpeg(value: str) -> bytes:
    """Decode and fully load a bounded JPEG, refusing deceptive headers."""
    if len(value) > MAX_BASE64_CHARS or len(value) % 4:
        raise ValueError("invalid_image")
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid_image") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("invalid_image")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "JPEG":
                raise ValueError("invalid_image")
            width, height = image.size
            if (
                width < 1
                or height < 1
                or width > MAX_IMAGE_EDGE
                or height > MAX_IMAGE_EDGE
                or width * height > MAX_IMAGE_PIXELS
            ):
                raise ValueError("invalid_image")
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            decoded = image.convert("RGB").tobytes()
            if len(decoded) > MAX_DECODED_BYTES:
                raise ValueError("invalid_image")
    except (Image.DecompressionBombError, OSError, UnidentifiedImageError) as exc:
        raise ValueError("invalid_image") from exc
    return raw


def compact_jpeg_for_model(value: str) -> str:
    """Re-encode the validated camera image to a small, model-stable JPEG."""
    raw = decode_jpeg(value)
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            compact = image.convert("RGB")
            compact.thumbnail((MODEL_IMAGE_EDGE, MODEL_IMAGE_EDGE), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            compact.save(
                output,
                format="JPEG",
                quality=MODEL_JPEG_QUALITY,
                optimize=True,
                progressive=False,
            )
        encoded = output.getvalue()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("invalid_image") from exc
    if not encoded or len(encoded) > MAX_IMAGE_BYTES:
        raise ValueError("invalid_image")
    return base64.b64encode(encoded).decode("ascii")


class VisionUnavailableError(Exception):
    pass


class VisionNoDescriptionError(VisionUnavailableError):
    """The local model answered but did not produce a caption."""


class VisualRuntime:
    """The only network path: two bounded calls to one validated local Ollama."""

    def __init__(
        self,
        identity: IdentityRuntime,
        *,
        load: Callable[[], dict[str, Any]] = settings_module.load_settings,
        transport: httpx.AsyncBaseTransport | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        self.identity = identity
        self._load = load
        self._transport = transport
        self.limiter = limiter or RateLimiter(limit=ANALYSIS_LIMIT_PER_MINUTE, window_seconds=60)

    def allow(self, user_id: str) -> bool:
        return self.limiter.allow(user_id, now=self.identity.clock())

    async def analyze(self, *, image: str, prompt: str) -> str:
        settings = self._load() or {}
        # A live endpoint is authoritative; invalid configuration fails closed.
        try:
            endpoint = normalize_endpoint(os.getenv("OLLAMA_HOST") or settings.get("ollama_host"))
        except ValueError as exc:
            raise VisionUnavailableError from exc
        # Vision must follow the model selected by the live runtime. A saved
        # dashboard preference may lag an image/manifest rollout, so it is only
        # the fallback when no active OLLAMA_MODEL is supplied.
        model = os.getenv("OLLAMA_MODEL") or settings.get("ollama_model")
        if not is_model_name(model):
            raise VisionUnavailableError
        host = urlsplit(endpoint).hostname or ""
        try:
            resolve_local_alias(host)
        except ValueError as exc:
            raise VisionUnavailableError from exc

        client = httpx.AsyncClient(
            timeout=OLLAMA_TIMEOUT, trust_env=False, transport=self._transport
        )
        stage = "model_capabilities"
        try:
            shown = await client.post(
                f"{endpoint}/api/show",
                json={"model": model},
                follow_redirects=False,
                headers={"Accept": "application/json"},
            )
            show = self._json(shown, maximum=MAX_SHOW_BYTES)
            capabilities = show.get("capabilities") if isinstance(show, dict) else None
            if not isinstance(capabilities, list) or "vision" not in capabilities:
                raise VisionUnavailableError
            for attempt in range(2):
                stage = "camera_analysis" if attempt == 0 else "camera_analysis_retry"
                try:
                    answered = await client.post(
                        f"{endpoint}/api/chat",
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": prompt, "images": [image]}],
                            "stream": False,
                            # A visual caption is a bounded observation, not a reasoning
                            # turn: reserve the output budget for the description itself.
                            "think": False,
                            "options": {"num_predict": 300, "temperature": 0.2},
                        },
                        follow_redirects=False,
                        headers={"Accept": "application/json"},
                    )
                    payload = self._json(answered)
                    message = payload.get("message") if isinstance(payload, dict) else None
                    content = message.get("content") if isinstance(message, dict) else None
                    concise = (
                        " ".join(content.split())[:MAX_OUTPUT_CHARS].strip()
                        if isinstance(content, str)
                        else ""
                    )
                    if concise:
                        return concise
                except (httpx.HTTPError, ValueError, VisionUnavailableError) as exc:
                    if attempt:
                        raise
                    logger.warning(
                        "Visual analysis transient failure kind=%s; retrying once",
                        type(exc).__name__,
                    )
                    continue
                if attempt == 0:
                    logger.warning("Visual analysis returned no description; retrying once")
            raise VisionNoDescriptionError
        except VisionNoDescriptionError:
            raise
        except (httpx.HTTPError, ValueError, VisionUnavailableError) as exc:
            logger.warning(
                "Visual analysis unavailable at stage=%s kind=%s", stage, type(exc).__name__
            )
            raise VisionUnavailableError from exc
        finally:
            await client.aclose()

    @staticmethod
    def _json(response: httpx.Response, *, maximum: int = MAX_UPSTREAM_BYTES) -> object:
        if response.status_code != 200 or len(response.content) > maximum:
            raise VisionUnavailableError
        try:
            return response.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise VisionUnavailableError from exc


_runtime_lock = threading.Lock()
_cached: tuple[IdentityRuntime, VisualRuntime] | None = None


def get_visual_runtime(
    identity: IdentityRuntime | None = Depends(user_api.get_runtime),
) -> VisualRuntime | None:
    global _cached
    if identity is None:
        return None
    with _runtime_lock:
        if _cached is None or _cached[0] is not identity:
            _cached = (identity, VisualRuntime(identity))
        return _cached[1]


def require_visual_runtime(
    runtime: VisualRuntime | None = Depends(get_visual_runtime),
) -> VisualRuntime:
    if runtime is None:
        logger.warning("Visual runtime unavailable because identity runtime is absent")
        raise HTTPException(status_code=503, detail="vision_unavailable")
    return runtime


async def _bounded_request(request: Request) -> VisualRequest:
    length = request.headers.get("content-length")
    if length:
        try:
            if int(length) > MAX_REQUEST_BYTES:
                raise HTTPException(status_code=422, detail="invalid")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid") from exc
    received = bytearray()
    async for chunk in request.stream():
        received.extend(chunk)
        if len(received) > MAX_REQUEST_BYTES:
            raise HTTPException(status_code=422, detail="invalid")
    try:
        raw = json.loads(received)
        return VisualRequest.model_validate(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        # Never let Pydantic echo the image or prompt in a validation response.
        raise HTTPException(status_code=422, detail="invalid") from exc


router = APIRouter(tags=["visual"])


@router.post("/users/me/visual/analyze", response_model=VisualResponse)
async def analyze_camera_view(
    request: Request,
    user: CurrentUser = Depends(require_user),
    runtime: VisualRuntime = Depends(require_visual_runtime),
) -> VisualResponse:
    if user.session_binding is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="owner_binding_required")
    body = await _bounded_request(request)
    if body.company_private:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="company_mode_blocked")
    if not runtime.allow(user.profile.user_id):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="rate_limited")
    try:
        model_image = compact_jpeg_for_model(body.image)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="invalid_image") from exc
    try:
        description = await runtime.analyze(image=model_image, prompt=body.prompt)
    except VisionNoDescriptionError as exc:
        raise HTTPException(status_code=502, detail="vision_no_description") from exc
    except VisionUnavailableError as exc:
        raise HTTPException(status_code=503, detail="vision_unavailable") from exc
    return VisualResponse(description=description)
