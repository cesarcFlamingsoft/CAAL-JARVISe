"""Webhook server for external triggers (announcements, tool reload, wake word, settings).

This module provides HTTP endpoints that allow external systems (like n8n)
and the frontend to trigger actions on the running voice agent.

Endpoints:
    POST /announce           - Make the agent speak a message
    POST /reload-tools       - Refresh MCP tool cache and optionally announce
    POST /wake               - Handle wake word detection (greet user)
    GET  /health             - Health check
    GET  /settings           - Get current settings
    POST /settings           - Update settings
    GET  /prompt             - Get current prompt content
    POST /prompt             - Save custom prompt
    GET  /voices             - List available TTS voices
    GET  /models             - List available LLM models
    GET  /wake-word/status   - Get wake word detection status
    POST /wake-word/enable   - Enable server-side wake word detection
    POST /wake-word/disable  - Disable server-side wake word detection
    GET  /wake-word/models   - List available wake word models

Usage:
    # Start in a background thread from voice_agent.py:
    import threading
    import uvicorn
    from caal.webhooks import app

    def run_webhook_server():
        uvicorn.run(app, host="0.0.0.0", port=8889, log_level="info")

    webhook_thread = threading.Thread(target=run_webhook_server, daemon=True)
    webhook_thread.start()
"""

from __future__ import annotations

import logging
import os
import random

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import settings as settings_module

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CAAL Webhook API",
    description="External triggers for CAAL voice agent",
    version="1.0.0",
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend can be on different port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnnounceRequest(BaseModel):
    """Request body for /announce endpoint."""

    message: str
    room_name: str = "voice_assistant_room"


class ReloadToolsRequest(BaseModel):
    """Request body for /reload-tools endpoint."""

    tool_name: str | None = None  # Optional: announce specific tool name
    message: str | None = None  # Optional: custom announcement message (overrides tool_name)
    room_name: str = "voice_assistant_room"


class WakeRequest(BaseModel):
    """Request body for /wake endpoint."""

    room_name: str = "voice_assistant_room"


class WakeResponse(BaseModel):
    """Response body for /wake endpoint."""

    status: str
    room_name: str


class AnnounceResponse(BaseModel):
    """Response body for /announce endpoint."""

    status: str
    room_name: str


class ReloadToolsResponse(BaseModel):
    """Response body for /reload-tools endpoint."""

    status: str
    tool_count: int
    room_name: str


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str
    active_sessions: list[str]


@app.post("/announce", response_model=AnnounceResponse)
async def announce(req: AnnounceRequest) -> AnnounceResponse:
    """Make the agent speak a message.

    This endpoint injects an announcement into an active voice session.
    The agent will speak the provided message using TTS.

    Args:
        req: AnnounceRequest with message and optional room_name

    Returns:
        AnnounceResponse with status

    Raises:
        HTTPException: 404 if no active session in the specified room
    """
    from . import session_registry

    result = session_registry.get(req.room_name)
    if not result:
        logger.warning(f"Announce failed: no session in room {req.room_name}")
        raise HTTPException(
            status_code=404,
            detail=f"No active session in room: {req.room_name}",
        )

    session, _agent = result
    logger.info(f"Announcing to room {req.room_name}: {req.message[:50]}...")

    # Say the message directly (bypasses LLM for instant response)
    await session.say(req.message)

    return AnnounceResponse(status="announced", room_name=req.room_name)


@app.post("/reload-tools", response_model=ReloadToolsResponse)
async def reload_tools(req: ReloadToolsRequest) -> ReloadToolsResponse:
    """Refresh MCP tool cache and optionally announce new tool availability.

    This endpoint clears the n8n workflow cache and re-discovers available
    workflows. Optionally announces the change:
    - If `message` is provided, speaks that exact message
    - If only `tool_name` is provided, speaks "A new tool called '{tool_name}' is now available."
    - If neither is provided, reloads silently

    Args:
        req: ReloadToolsRequest with optional message, tool_name, and room_name

    Returns:
        ReloadToolsResponse with status and tool count

    Raises:
        HTTPException: 404 if no active session in the specified room
    """
    from . import session_registry
    from .integrations import n8n

    result = session_registry.get(req.room_name)
    if not result:
        logger.warning(f"Reload failed: no session in room {req.room_name}")
        raise HTTPException(
            status_code=404,
            detail=f"No active session in room: {req.room_name}",
        )

    session, agent = result
    logger.info(f"Reloading tools for room {req.room_name}")

    # Clear all caches
    agent._ollama_tools_cache = None
    n8n.clear_caches()

    # Re-discover n8n workflows if MCP is configured
    tool_count = 0
    if agent._n8n_mcp and agent._n8n_base_url:
        try:
            tools, name_map = await n8n.discover_n8n_workflows(
                agent._n8n_mcp, agent._n8n_base_url
            )
            agent._n8n_workflow_tools = tools
            agent._n8n_workflow_name_map = name_map
            tool_count = len(tools)
            logger.info(f"Discovered {tool_count} n8n workflows")
        except Exception as e:
            logger.error(f"Failed to re-discover n8n workflows: {e}")

    # Announce: custom message takes priority, then tool_name format
    if req.message:
        await session.say(req.message)
    elif req.tool_name:
        await session.say(f"A new tool called '{req.tool_name}' is now available.")

    return ReloadToolsResponse(
        status="reloaded",
        tool_count=tool_count,
        room_name=req.room_name,
    )


@app.post("/wake", response_model=WakeResponse)
async def wake(req: WakeRequest) -> WakeResponse:
    """Handle wake word detection - greet the user.

    This endpoint is primarily for:
    - Client-side wake word detection (Picovoice - deprecated)
    - Manual testing via curl

    Server-side wake word detection (OpenWakeWord) handles greetings
    directly in voice_agent.py for lower latency.

    Args:
        req: WakeRequest with room_name

    Returns:
        WakeResponse with status

    Raises:
        HTTPException: 404 if no active session in the specified room
    """
    from . import session_registry

    result = session_registry.get(req.room_name)
    if not result:
        logger.warning(f"Wake failed: no session in room {req.room_name}")
        raise HTTPException(
            status_code=404,
            detail=f"No active session in room: {req.room_name}",
        )

    session, _agent = result
    logger.info(f"Wake word detected in room {req.room_name}")

    # Get greeting
    greetings = settings_module.get_setting("wake_greetings")
    greeting = random.choice(greetings)

    # Call TTS directly and push to audio output, bypassing agent turn-taking
    tts = session.tts
    audio_output = session.output.audio
    audio_stream = tts.synthesize(greeting)

    # Push audio frames directly to the audio output
    async for event in audio_stream:
        if hasattr(event, 'frame') and event.frame:
            await audio_output.capture_frame(event.frame)

    # Flush to complete the segment
    audio_output.flush()

    return WakeResponse(status="greeted", room_name=req.room_name)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint.

    Returns:
        HealthResponse with status and list of active session room names
    """
    from . import session_registry

    return HealthResponse(
        status="ok",
        active_sessions=session_registry.list_rooms(),
    )


# =============================================================================
# Settings Endpoints
# =============================================================================


class SettingsResponse(BaseModel):
    """Response body for /settings endpoint."""

    settings: dict
    prompt_content: str
    custom_prompt_exists: bool


class SettingsUpdateRequest(BaseModel):
    """Request body for POST /settings endpoint."""

    settings: dict


class PromptResponse(BaseModel):
    """Response body for /prompt endpoint."""

    prompt: str  # "default" or "custom"
    content: str
    is_custom: bool


class PromptUpdateRequest(BaseModel):
    """Request body for POST /prompt endpoint."""

    content: str


class VoicesResponse(BaseModel):
    """Response body for /voices endpoint."""

    voices: list[str]


class ModelsResponse(BaseModel):
    """Response body for /models endpoint."""

    models: list[str]


@app.get("/settings", response_model=SettingsResponse)
async def get_settings() -> SettingsResponse:
    """Get current settings and prompt content.

    Returns:
        SettingsResponse with current settings, prompt content, and custom prompt status.
        Sensitive keys (tokens) are excluded for security.
    """
    settings = settings_module.load_settings_safe()  # Excludes sensitive keys
    prompt_content = settings_module.load_prompt_content()
    custom_exists = settings_module.custom_prompt_exists()

    return SettingsResponse(
        settings=settings,
        prompt_content=prompt_content,
        custom_prompt_exists=custom_exists,
    )


@app.post("/settings", response_model=SettingsResponse)
async def update_settings(req: SettingsUpdateRequest) -> SettingsResponse:
    """Update settings.

    Args:
        req: SettingsUpdateRequest with settings dict to merge

    Returns:
        SettingsResponse with updated settings
    """
    # Load current settings
    current = settings_module.load_settings()

    # Merge with new settings (only known keys)
    for key, value in req.settings.items():
        if key in settings_module.DEFAULT_SETTINGS:
            current[key] = value

    # Save merged settings
    settings_module.save_settings(current)

    # Reload and return
    settings = settings_module.reload_settings()
    prompt_content = settings_module.load_prompt_content()
    custom_exists = settings_module.custom_prompt_exists()

    logger.info(f"Settings updated: {list(req.settings.keys())}")

    return SettingsResponse(
        settings=settings,
        prompt_content=prompt_content,
        custom_prompt_exists=custom_exists,
    )


@app.get("/prompt", response_model=PromptResponse)
async def get_prompt() -> PromptResponse:
    """Get current prompt content.

    Returns:
        PromptResponse with prompt name and content
    """
    prompt_name = settings_module.get_setting("prompt", "default")
    content = settings_module.load_prompt_content(prompt_name)
    is_custom = prompt_name == "custom" and settings_module.custom_prompt_exists()

    return PromptResponse(
        prompt=prompt_name,
        content=content,
        is_custom=is_custom,
    )


@app.post("/prompt", response_model=PromptResponse)
async def save_prompt(req: PromptUpdateRequest) -> PromptResponse:
    """Save custom prompt content.

    Args:
        req: PromptUpdateRequest with content to save

    Returns:
        PromptResponse with saved prompt info
    """
    # Save to custom.md
    settings_module.save_custom_prompt(req.content)

    # Update settings to use custom prompt
    current = settings_module.load_settings()
    current["prompt"] = "custom"
    settings_module.save_settings(current)

    logger.info("Custom prompt saved and activated")

    return PromptResponse(
        prompt="custom",
        content=req.content,
        is_custom=True,
    )


@app.get("/voices", response_model=VoicesResponse)
async def get_voices() -> VoicesResponse:
    """Get available TTS voices from Kokoro.

    Returns:
        VoicesResponse with list of voice IDs
    """
    kokoro_url = os.getenv("KOKORO_URL", "http://kokoro:8880")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{kokoro_url}/v1/audio/voices",
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            # Kokoro returns {"voices": [{"id": "...", ...}, ...]}
            voices = [v.get("id") or v.get("voice_id") for v in data.get("voices", [])]
            voices = [v for v in voices if v]  # Filter None values

            return VoicesResponse(voices=voices)
    except Exception as e:
        logger.warning(f"Failed to fetch voices from Kokoro: {e}")
        # Return default voices as fallback
        return VoicesResponse(
            voices=["af_heart", "af_bella", "af_sarah", "am_adam", "am_puck"]
        )


@app.get("/models", response_model=ModelsResponse)
async def get_models() -> ModelsResponse:
    """Get available LLM models from Ollama.

    Returns:
        ModelsResponse with list of model names
    """
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{ollama_host}/api/tags",
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            # Ollama returns {"models": [{"name": "...", ...}, ...]}
            models = [m.get("name") for m in data.get("models", [])]
            models = [m for m in models if m]  # Filter None values

            return ModelsResponse(models=models)
    except Exception as e:
        logger.warning(f"Failed to fetch models from Ollama: {e}")
        # Return empty list on failure
        return ModelsResponse(models=[])


# =============================================================================
# Wake Word Control Endpoints
# =============================================================================


class WakeWordStatusResponse(BaseModel):
    """Response body for /wake-word/status endpoint."""

    enabled: bool
    model: str
    threshold: float
    timeout: float


class WakeWordUpdateRequest(BaseModel):
    """Request body for wake word enable/disable."""

    enabled: bool


@app.get("/wake-word/status", response_model=WakeWordStatusResponse)
async def get_wake_word_status() -> WakeWordStatusResponse:
    """Get current wake word detection status.

    Returns:
        WakeWordStatusResponse with enabled state and configuration
    """
    settings = settings_module.load_settings()

    return WakeWordStatusResponse(
        enabled=settings.get("wake_word_enabled", False),
        model=settings.get("wake_word_model", "models/hey_jarvis.onnx"),
        threshold=settings.get("wake_word_threshold", 0.5),
        timeout=settings.get("wake_word_timeout", 3.0),
    )


@app.post("/wake-word/enable", response_model=WakeWordStatusResponse)
async def enable_wake_word() -> WakeWordStatusResponse:
    """Enable wake word detection.

    Note: This updates the setting but requires agent restart to take effect.

    Returns:
        WakeWordStatusResponse with updated configuration
    """
    current = settings_module.load_settings()
    current["wake_word_enabled"] = True
    settings_module.save_settings(current)

    settings = settings_module.reload_settings()
    logger.info("Wake word detection enabled (requires agent restart)")

    return WakeWordStatusResponse(
        enabled=True,
        model=settings.get("wake_word_model", "models/hey_jarvis.onnx"),
        threshold=settings.get("wake_word_threshold", 0.5),
        timeout=settings.get("wake_word_timeout", 3.0),
    )


@app.post("/wake-word/disable", response_model=WakeWordStatusResponse)
async def disable_wake_word() -> WakeWordStatusResponse:
    """Disable wake word detection.

    Note: This updates the setting but requires agent restart to take effect.

    Returns:
        WakeWordStatusResponse with updated configuration
    """
    current = settings_module.load_settings()
    current["wake_word_enabled"] = False
    settings_module.save_settings(current)

    settings = settings_module.reload_settings()
    logger.info("Wake word detection disabled (requires agent restart)")

    return WakeWordStatusResponse(
        enabled=False,
        model=settings.get("wake_word_model", "models/hey_jarvis.onnx"),
        threshold=settings.get("wake_word_threshold", 0.5),
        timeout=settings.get("wake_word_timeout", 3.0),
    )


class WakeWordModelsResponse(BaseModel):
    """Response containing available wake word models."""

    models: list[str]


@app.get("/wake-word/models", response_model=WakeWordModelsResponse)
async def get_wake_word_models() -> WakeWordModelsResponse:
    """List available wake word models.

    Scans the models/ directory for .onnx files, excluding infrastructure
    models (embedding_model, melspectrogram).

    Returns:
        WakeWordModelsResponse with list of model paths
    """
    from pathlib import Path

    models_dir = Path("models")
    models = []

    if models_dir.exists():
        for f in models_dir.glob("*.onnx"):
            # Skip infrastructure models used by OpenWakeWord
            if f.name not in ("embedding_model.onnx", "melspectrogram.onnx"):
                models.append(f"models/{f.name}")

    return WakeWordModelsResponse(models=sorted(models))


# =============================================================================
# Setup Wizard Endpoints
# =============================================================================


class SetupStatusResponse(BaseModel):
    """Response body for /setup/status endpoint."""

    completed: bool


class SetupCompleteRequest(BaseModel):
    """Request body for /setup/complete endpoint."""

    llm_provider: str  # "ollama" | "groq"
    # Ollama settings
    ollama_host: str | None = None
    ollama_model: str | None = None
    # Groq settings
    groq_api_key: str | None = None
    groq_model: str | None = None
    # TTS provider
    tts_provider: str = "kokoro"  # "kokoro" | "piper"
    # Integrations (optional)
    hass_enabled: bool = False
    hass_host: str | None = None
    hass_token: str | None = None
    n8n_enabled: bool = False
    n8n_url: str | None = None
    n8n_token: str | None = None


class SetupCompleteResponse(BaseModel):
    """Response body for /setup/complete endpoint."""

    success: bool
    message: str


class TestConnectionResponse(BaseModel):
    """Response body for connection test endpoints."""

    success: bool
    error: str | None = None
    models: list[str] | None = None  # For Ollama
    device_count: int | None = None  # For Home Assistant
    workflow_count: int | None = None  # For n8n


class TestOllamaRequest(BaseModel):
    """Request body for /setup/test-ollama endpoint."""

    host: str


class TestGroqRequest(BaseModel):
    """Request body for /setup/test-groq endpoint."""

    api_key: str


class TestHassRequest(BaseModel):
    """Request body for /setup/test-hass endpoint."""

    host: str
    token: str


class TestN8nRequest(BaseModel):
    """Request body for /setup/test-n8n endpoint."""

    url: str
    token: str | None = None


@app.get("/setup/status", response_model=SetupStatusResponse)
async def get_setup_status() -> SetupStatusResponse:
    """Check if first-launch setup has been completed.

    Returns:
        SetupStatusResponse with completed flag
    """
    settings = settings_module.load_settings()
    return SetupStatusResponse(completed=settings.get("first_launch_completed", False))


@app.post("/setup/complete", response_model=SetupCompleteResponse)
async def complete_setup(req: SetupCompleteRequest) -> SetupCompleteResponse:
    """Complete the first-launch setup wizard.

    Saves all settings and marks setup as completed.
    UI sets both stt_provider and llm_provider together based on llm_provider choice.

    Args:
        req: SetupCompleteRequest with provider and integration settings

    Returns:
        SetupCompleteResponse with success status
    """
    try:
        current = settings_module.load_settings()

        # Provider settings - UI couples STT and LLM together
        current["llm_provider"] = req.llm_provider
        if req.llm_provider == "groq":
            current["stt_provider"] = "groq"
            if req.groq_model:
                current["groq_model"] = req.groq_model
        else:
            current["stt_provider"] = "speaches"
            if req.ollama_host:
                current["ollama_host"] = req.ollama_host
            if req.ollama_model:
                current["ollama_model"] = req.ollama_model

        # Home Assistant integration
        current["hass_enabled"] = req.hass_enabled
        if req.hass_enabled:
            if req.hass_host:
                current["hass_host"] = req.hass_host
            if req.hass_token:
                current["hass_token"] = req.hass_token

        # n8n integration
        current["n8n_enabled"] = req.n8n_enabled
        if req.n8n_enabled:
            if req.n8n_url:
                current["n8n_url"] = req.n8n_url
            if req.n8n_token:
                current["n8n_token"] = req.n8n_token

        # TTS provider (placeholder for now - piper not yet implemented)
        current["tts_provider"] = req.tts_provider

        # Mark setup as complete
        current["first_launch_completed"] = True

        # Save settings
        settings_module.save_settings(current)

        # Store Groq API key in environment (not in settings.json for security)
        if req.groq_api_key:
            os.environ["GROQ_API_KEY"] = req.groq_api_key
            # TODO: Persist to .env file or secure storage

        logger.info("First-launch setup completed")
        return SetupCompleteResponse(success=True, message="Setup completed successfully")

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return SetupCompleteResponse(success=False, message=str(e))


@app.post("/setup/test-ollama", response_model=TestConnectionResponse)
async def test_ollama(req: TestOllamaRequest) -> TestConnectionResponse:
    """Test Ollama connection and list available models.

    Args:
        req: TestOllamaRequest with host URL

    Returns:
        TestConnectionResponse with success status and model list
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{req.host}/api/tags",
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            models = [m.get("name") for m in data.get("models", [])]
            models = [m for m in models if m]

            return TestConnectionResponse(success=True, models=models)
    except httpx.ConnectError:
        return TestConnectionResponse(
            success=False, error=f"Cannot connect to Ollama at {req.host}"
        )
    except Exception as e:
        return TestConnectionResponse(success=False, error=str(e))


@app.post("/setup/test-groq", response_model=TestConnectionResponse)
async def test_groq(req: TestGroqRequest) -> TestConnectionResponse:
    """Test Groq API key validity.

    Args:
        req: TestGroqRequest with API key

    Returns:
        TestConnectionResponse with success status
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {req.api_key}"},
                timeout=10.0,
            )
            if response.status_code == 401:
                return TestConnectionResponse(success=False, error="Invalid API key")
            response.raise_for_status()

            data = response.json()
            models = [m.get("id") for m in data.get("data", [])]
            models = [m for m in models if m]

            return TestConnectionResponse(success=True, models=models)
    except Exception as e:
        return TestConnectionResponse(success=False, error=str(e))


@app.post("/setup/test-hass", response_model=TestConnectionResponse)
async def test_hass(req: TestHassRequest) -> TestConnectionResponse:
    """Test Home Assistant connection.

    Args:
        req: TestHassRequest with host and token

    Returns:
        TestConnectionResponse with success status and device count
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{req.host}/api/states",
                headers={"Authorization": f"Bearer {req.token}"},
                timeout=10.0,
            )
            if response.status_code == 401:
                return TestConnectionResponse(
                    success=False, error="Invalid access token"
                )
            response.raise_for_status()

            states = response.json()
            device_count = len(states)

            return TestConnectionResponse(success=True, device_count=device_count)
    except httpx.ConnectError:
        return TestConnectionResponse(
            success=False, error=f"Cannot connect to Home Assistant at {req.host}"
        )
    except Exception as e:
        return TestConnectionResponse(success=False, error=str(e))


@app.post("/setup/test-n8n", response_model=TestConnectionResponse)
async def test_n8n(req: TestN8nRequest) -> TestConnectionResponse:
    """Test n8n MCP connection.

    Args:
        req: TestN8nRequest with URL and optional token

    Returns:
        TestConnectionResponse with success status and workflow count
    """
    try:
        headers = {}
        if req.token:
            headers["Authorization"] = f"Bearer {req.token}"

        async with httpx.AsyncClient() as client:
            # Try to connect to n8n MCP endpoint
            response = await client.get(
                req.url,
                headers=headers,
                timeout=10.0,
            )
            # n8n MCP might return various status codes, just check connectivity
            if response.status_code == 401:
                return TestConnectionResponse(
                    success=False, error="Invalid authentication token"
                )

            # Try to get workflow count if possible
            # n8n MCP discovery happens via SSE, so we just verify connectivity
            return TestConnectionResponse(success=True, workflow_count=0)
    except httpx.ConnectError:
        return TestConnectionResponse(
            success=False, error=f"Cannot connect to n8n at {req.url}"
        )
    except Exception as e:
        return TestConnectionResponse(success=False, error=str(e))


# =============================================================================
# Prewarm Endpoint
# =============================================================================


class PrewarmResponse(BaseModel):
    """Response body for /prewarm endpoint."""

    status: str
    message: str


@app.post("/prewarm", response_model=PrewarmResponse)
async def prewarm() -> PrewarmResponse:
    """Trigger model preloading for Ollama.

    This endpoint triggers the preload_models() function to warm up
    the STT and LLM models. It returns immediately - preloading happens
    in background.

    Returns:
        PrewarmResponse with status
    """
    settings = settings_module.load_settings()

    # Only prewarm if using Ollama
    if settings.get("llm_provider", "ollama") != "ollama":
        return PrewarmResponse(
            status="skipped",
            message="Prewarm only applies to Ollama provider",
        )

    # Import and call preload in background
    import asyncio

    async def do_prewarm():
        try:
            # Import here to avoid circular imports
            from voice_agent import preload_models

            ollama_host = settings.get("ollama_host", "http://localhost:11434")
            ollama_model = settings.get("ollama_model", "ministral-3:8b")
            await preload_models(ollama_host, ollama_model)
            logger.info("Model prewarm completed")
        except Exception as e:
            logger.error(f"Prewarm failed: {e}")

    # Fire and forget
    asyncio.create_task(do_prewarm())

    return PrewarmResponse(
        status="started",
        message="Model preloading started in background",
    )
