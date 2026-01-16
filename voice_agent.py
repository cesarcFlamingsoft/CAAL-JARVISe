#!/usr/bin/env python3
"""
CAAL Voice Framework - Voice Agent
==================================

A voice assistant with MCP integrations for n8n workflows.

Usage:
    python voice_agent.py dev

Configuration:
    - .env: Environment variables (MCP URL, model settings)
    - prompt/default.md: Agent system prompt

Environment Variables:
    SPEACHES_URL        - Speaches STT service URL (default: "http://speaches:8000")
    KOKORO_URL          - Kokoro TTS service URL (default: "http://kokoro:8880")
    WHISPER_MODEL       - Whisper model for STT (default: "Systran/faster-whisper-small")
    TTS_VOICE           - Kokoro voice name (default: "af_heart")
    OLLAMA_MODEL        - Ollama model name (default: "ministral-3:8b")
    OLLAMA_THINK        - Enable thinking mode (default: "false")
    TIMEZONE            - Timezone for date/time (default: "Pacific Time")
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

import requests

# Add src directory to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv

# Load environment variables from .env
_script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_script_dir, ".env"))

from livekit import agents
from livekit.agents import AgentSession, Agent, mcp, function_tool
from livekit.plugins import silero, openai, groq as groq_plugin

from caal import CAALLLM
from caal.integrations import (
    load_mcp_config,
    initialize_mcp_servers,
    WebSearchTools,
    discover_n8n_workflows,
)
from caal.llm import llm_node, ToolDataCache
from caal.stt import WakeWordGatedSTT

# Configure logging - LiveKit adds LogQueueHandler to root in worker processes,
# so we use non-propagating loggers with our own handler to avoid duplicates
_log_handler = logging.StreamHandler()
_log_handler.setFormatter(logging.Formatter("%(message)s"))

# voice-agent logger (this file)
logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.addHandler(_log_handler)

# caal package logger (src/caal/*)
_caal_logger = logging.getLogger("caal")
_caal_logger.setLevel(logging.INFO)
_caal_logger.propagate = False
_caal_logger.addHandler(_log_handler)

# Suppress verbose logs from dependencies
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("groq._base_client").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("livekit").setLevel(logging.WARNING)
logging.getLogger("livekit_api").setLevel(logging.WARNING)
logging.getLogger("livekit.agents.tts").setLevel(logging.ERROR)  # Suppress "no request_id" warnings
logging.getLogger("livekit.agents.voice").setLevel(logging.WARNING)
logging.getLogger("livekit.plugins.openai.tts").setLevel(logging.WARNING)

# =============================================================================
# Configuration
# =============================================================================

# Infrastructure config (from .env only - URLs, tokens, etc.)
SPEACHES_URL = os.getenv("SPEACHES_URL", "http://speaches:8000")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "Systran/faster-whisper-small")
KOKORO_URL = os.getenv("KOKORO_URL", "http://kokoro:8880")
TTS_MODEL = os.getenv("TTS_MODEL", "kokoro")  # "kokoro" for Kokoro-FastAPI, "prince-canuma/Kokoro-82M" for mlx-audio
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "false").lower() == "true"
TIMEZONE_ID = os.getenv("TIMEZONE", "America/Los_Angeles")
TIMEZONE_DISPLAY = os.getenv("TIMEZONE_DISPLAY", "Pacific Time")

# Import settings module for runtime-configurable values
from caal import settings as settings_module


def get_runtime_settings() -> dict:
    """Get runtime-configurable settings.

    These can be changed via the settings UI without rebuilding.
    Falls back to .env values for backwards compatibility.

    Priority: settings.json (explicit) > .env > DEFAULT_SETTINGS
    """
    settings = settings_module.load_settings()
    user_settings = settings_module.load_user_settings()  # Only explicitly set values

    return {
        # TTS settings
        "tts_provider": user_settings.get("tts_provider") or os.getenv("TTS_PROVIDER", "kokoro"),
        "tts_voice_kokoro": settings.get("tts_voice_kokoro") or os.getenv("TTS_VOICE", "am_puck"),
        "tts_voice_piper": settings.get("tts_voice_piper") or "speaches-ai/piper-en_US-ryan-high",
        # STT Provider settings
        "stt_provider": user_settings.get("stt_provider") or os.getenv("STT_PROVIDER", "speaches"),
        # LLM Provider settings - .env overrides default, user setting overrides .env
        "llm_provider": user_settings.get("llm_provider") or os.getenv("LLM_PROVIDER", "ollama"),
        "temperature": settings.get("temperature", float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))),
        # Ollama settings
        "ollama_host": user_settings.get("ollama_host") or os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "ollama_model": user_settings.get("ollama_model") or os.getenv("OLLAMA_MODEL", "ministral-3:8b"),
        "num_ctx": settings.get("num_ctx", int(os.getenv("OLLAMA_NUM_CTX", "8192"))),
        "think": OLLAMA_THINK,  # Only applies to Ollama
        # Groq settings
        "groq_api_key": settings.get("groq_api_key") or os.getenv("GROQ_API_KEY", ""),
        "groq_model": user_settings.get("groq_model") or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        # Shared settings
        "max_turns": settings.get("max_turns", int(os.getenv("OLLAMA_MAX_TURNS", "20"))),
        "tool_cache_size": settings.get("tool_cache_size", int(os.getenv("TOOL_CACHE_SIZE", "3"))),
        # Turn detection settings
        "allow_interruptions": settings.get("allow_interruptions", True),
        "min_endpointing_delay": settings.get("min_endpointing_delay", 0.5),
    }


def load_prompt() -> str:
    """Load and populate prompt template with date context."""
    return settings_module.load_prompt_with_context(
        timezone_id=TIMEZONE_ID,
        timezone_display=TIMEZONE_DISPLAY,
    )


# =============================================================================
# Agent Definition
# =============================================================================

# Type alias for tool status callback
ToolStatusCallback = callable  # async (bool, list[str], list[dict]) -> None


# =========================================================================
# Home Assistant Wrapper Tools
# =========================================================================
# These provide a simplified interface to HASS MCP, matching the n8n
# hass_control workflow signature for prompt compatibility.
# Tools are only added when Home Assistant is connected.


def create_hass_tools(hass_server: mcp.MCPServerHTTP) -> tuple[list[dict], dict]:
    """Create Home Assistant tools bound to the given MCP server.

    Returns:
        tuple: (tool_definitions, tool_callables)
        - tool_definitions: List of tool definitions in OpenAI format for LLM
        - tool_callables: Dict mapping tool name to callable function
    """
    async def hass_control(action: str, target: str, value: int = None) -> str:
        """Control Home Assistant devices.
        Parameters: action (required: turn_on, turn_off, volume_up, volume_down, set_volume, mute, unmute, pause, play, next, previous), target (required: device name), value (optional: for set_volume 0-100).
        """
        if not hass_server or not hasattr(hass_server, "_client"):
            return "Home Assistant is not connected"

        # Map action to HASS MCP tool and build arguments
        action_map = {
            "turn_on": ("HassTurnOn", {"name": target}),
            "turn_off": ("HassTurnOff", {"name": target}),
            "pause": ("HassMediaPause", {"name": target}),
            "play": ("HassMediaUnpause", {"name": target}),
            "next": ("HassMediaNext", {"name": target}),
            "previous": ("HassMediaPrevious", {"name": target}),
            "volume_up": ("HassSetVolumeRelative", {"name": target, "volume_step": "up"}),
            "volume_down": ("HassSetVolumeRelative", {"name": target, "volume_step": "down"}),
            "set_volume": ("HassSetVolume", {"name": target, "volume_level": value or 50}),
            "mute": ("HassMediaPlayerMute", {"name": target}),
            "unmute": ("HassMediaPlayerUnmute", {"name": target}),
        }

        if action not in action_map:
            return f"Unknown action: {action}. Valid actions: {', '.join(action_map.keys())}"

        tool_name, args = action_map[action]

        try:
            result = await hass_server._client.call_tool(tool_name, args)

            # Check for errors
            if result.isError:
                error_texts = [c.text for c in result.content if hasattr(c, "text") and c.text]
                return f"Error: {' '.join(error_texts)}"

            # Extract success message
            texts = [c.text for c in result.content if hasattr(c, "text") and c.text]
            return " ".join(texts) if texts else f"Done: {action} {target}"

        except Exception as e:
            logger.error(f"hass_control error: {e}")
            return f"Failed to {action} {target}: {e}"

    async def hass_get_state(target: str = None) -> str:
        """Get the current state of Home Assistant devices.
        Parameters: target (optional: device name to filter, or omit for all devices).
        """
        if not hass_server or not hasattr(hass_server, "_client"):
            return "Home Assistant is not connected"

        try:
            result = await hass_server._client.call_tool("GetLiveContext", {})

            # Check for errors
            if result.isError:
                error_texts = [c.text for c in result.content if hasattr(c, "text") and c.text]
                return f"Error: {' '.join(error_texts)}"

            # Extract content
            texts = [c.text for c in result.content if hasattr(c, "text") and c.text]
            full_context = " ".join(texts) if texts else "No devices found"

            # If target specified, filter to just that device
            if target:
                target_lower = target.lower()
                # Simple filter: look for lines containing the target name
                lines = full_context.split("\n")
                filtered = []
                capturing = False
                for line in lines:
                    if "names:" in line.lower() and target_lower in line.lower():
                        capturing = True
                    elif "names:" in line.lower() and capturing:
                        capturing = False
                    if capturing:
                        filtered.append(line)
                if filtered:
                    return "\n".join(filtered)
                return f"Device '{target}' not found"

            return full_context

        except Exception as e:
            logger.error(f"hass_get_state error: {e}")
            return f"Failed to get state: {e}"

    # Tool definitions in OpenAI format for LLM
    tool_definitions = [
        {
            "type": "function",
            "function": {
                "name": "hass_control",
                "description": "Control Home Assistant devices. Parameters: action (required: turn_on, turn_off, volume_up, volume_down, set_volume, mute, unmute, pause, play, next, previous), target (required: device name), value (optional: for set_volume 0-100).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "target": {"type": "string"},
                        "value": {"type": "integer"},
                    },
                    "required": ["action", "target"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "hass_get_state",
                "description": "Get the current state of Home Assistant devices. Parameters: target (optional: device name to filter, or omit for all devices).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                    },
                    "required": [],
                },
            },
        },
    ]

    # Callable functions for tool execution
    tool_callables = {
        "hass_control": hass_control,
        "hass_get_state": hass_get_state,
    }

    return tool_definitions, tool_callables


class VoiceAssistant(WebSearchTools, Agent):
    """Voice assistant with MCP tools and web search."""

    def __init__(
        self,
        caal_llm: CAALLLM,
        mcp_servers: dict[str, mcp.MCPServerHTTP] | None = None,
        n8n_workflow_tools: list[dict] | None = None,
        n8n_workflow_name_map: dict[str, str] | None = None,
        n8n_base_url: str | None = None,
        on_tool_status: ToolStatusCallback | None = None,
        tool_cache_size: int = 3,
        max_turns: int = 20,
        hass_tool_definitions: list[dict] | None = None,
        hass_tool_callables: dict | None = None,
    ) -> None:
        super().__init__(
            instructions=load_prompt(),
            llm=caal_llm,  # Satisfies LLM interface requirement
        )

        # Store provider for llm_node access
        self._provider = caal_llm.provider_instance

        # All MCP servers (for multi-MCP support)
        # Named _caal_mcp_servers to avoid conflict with LiveKit's internal _mcp_servers handling
        self._caal_mcp_servers = mcp_servers or {}

        # n8n-specific for workflow execution (n8n uses webhook-based execution)
        self._n8n_workflow_tools = n8n_workflow_tools or []
        self._n8n_workflow_name_map = n8n_workflow_name_map or {}
        self._n8n_base_url = n8n_base_url

        # Home Assistant tools (only if HASS is connected)
        self._hass_tool_definitions = hass_tool_definitions or []
        self._hass_tool_callables = hass_tool_callables or {}

        # Callback for publishing tool status to frontend
        self._on_tool_status = on_tool_status

        # Context management: tool data cache and sliding window
        self._tool_data_cache = ToolDataCache(max_entries=tool_cache_size)
        self._max_turns = max_turns

    async def llm_node(self, chat_ctx, tools, model_settings):
        """Custom LLM node using provider-agnostic interface."""
        async for chunk in llm_node(
            self,
            chat_ctx,
            provider=self._provider,
            tool_data_cache=self._tool_data_cache,
            max_turns=self._max_turns,
        ):
            yield chunk


# =============================================================================
# Agent Entrypoint
# =============================================================================

async def entrypoint(ctx: agents.JobContext) -> None:
    """Main entrypoint for the voice agent."""
    # Note: Webhook server is started in background thread at agent startup (main block)
    # This ensures /setup/status is available before users connect

    logger.debug(f"Joining room: {ctx.room.name}")
    await ctx.connect()

    # Load MCP servers from config
    mcp_servers = {}
    mcp_errors = []
    try:
        mcp_configs = load_mcp_config()
        mcp_servers, mcp_errors = await initialize_mcp_servers(mcp_configs)
    except Exception as e:
        logger.error(f"Failed to load MCP config: {e}")
        mcp_configs = []  # Ensure mcp_configs is defined for later use

    # Send MCP connection errors to frontend
    if mcp_errors:
        error_messages = []
        for err in mcp_errors:
            # Friendly names for known servers
            if err.name == "n8n":
                error_messages.append("n8n enabled but could not connect - check URL and token in Settings")
            elif err.name == "home_assistant":
                error_messages.append("Home Assistant enabled but could not connect - check URL and token in Settings")
            else:
                error_messages.append(f"MCP server '{err.name}' failed to connect: {err.error}")

        # Send error to frontend via data channel
        import json as json_module
        payload = json_module.dumps({
            "type": "mcp_error",
            "errors": error_messages,
        })
        try:
            await ctx.room.local_participant.publish_data(
                payload.encode("utf-8"),
                reliable=True,
                topic="mcp_error",
            )
        except Exception as e:
            logger.error(f"Failed to send MCP error to frontend: {e}")

    # Discover n8n workflows (n8n uses webhook-based execution, not MCP tools)
    n8n_workflow_tools = []
    n8n_workflow_name_map = {}
    n8n_base_url = None
    n8n_mcp = mcp_servers.get("n8n")
    if n8n_mcp:
        try:
            # Extract base URL from n8n MCP server config
            n8n_config = next((c for c in mcp_configs if c.name == "n8n"), None)
            if n8n_config:
                # URL format: http://HOST:PORT/mcp-server/http
                # Base URL: http://HOST:PORT
                url_parts = n8n_config.url.rsplit("/", 2)
                n8n_base_url = url_parts[0] if len(url_parts) >= 2 else n8n_config.url

            n8n_workflow_tools, n8n_workflow_name_map = await discover_n8n_workflows(
                n8n_mcp, n8n_base_url
            )
        except Exception as e:
            logger.error(f"Failed to discover n8n workflows: {e}")

    # Get runtime settings (from settings.json with .env fallback)
    runtime = get_runtime_settings()

    # Set GROQ_API_KEY env var for plugins that read from environment
    if runtime.get("groq_api_key"):
        os.environ["GROQ_API_KEY"] = runtime["groq_api_key"]

    # Create CAALLLM instance (provider-agnostic wrapper)
    caal_llm = CAALLLM.from_settings(runtime)

    # Log configuration
    logger.info("=" * 60)
    logger.info("STARTING VOICE AGENT")
    logger.info("=" * 60)
    if runtime["stt_provider"] == "groq":
        logger.info("  STT: Groq (whisper-large-v3-turbo)")
    else:
        logger.info(f"  STT: {SPEACHES_URL} ({WHISPER_MODEL})")
    if runtime["tts_provider"] == "piper":
        logger.info(f"  TTS: Piper ({runtime['tts_voice_piper']})")
    else:
        logger.info(f"  TTS: Kokoro ({runtime['tts_voice_kokoro']})")
    if runtime["llm_provider"] == "ollama":
        logger.info(
            f"  LLM: Ollama ({runtime['ollama_model']}, think={runtime['think']}, num_ctx={runtime['num_ctx']})"
        )
    else:
        logger.info(
            f"  LLM: Groq ({runtime['groq_model']})"
        )
    logger.info(f"  MCP: {list(mcp_servers.keys()) or 'None'}")
    logger.info(
        f"  Turn detection: interruptions={runtime['allow_interruptions']}, "
        f"endpointing_delay={runtime['min_endpointing_delay']}s"
    )
    logger.info("=" * 60)

    # Build STT - Speaches (local) or Groq (cloud)
    if runtime["stt_provider"] == "groq":
        base_stt = groq_plugin.STT(
            model="whisper-large-v3-turbo",
            language="en",
        )
    else:
        base_stt = openai.STT(
            base_url=f"{SPEACHES_URL}/v1",
            api_key="not-needed",  # Speaches doesn't require auth
            model=WHISPER_MODEL,
        )

    # Load wake word settings
    all_settings = settings_module.load_settings()
    wake_word_enabled = all_settings.get("wake_word_enabled", False)

    # Session reference for wake word callback (set after session creation)
    _session_ref: AgentSession | None = None

    if wake_word_enabled:
        import json
        import random

        wake_word_model = all_settings.get("wake_word_model", "models/hey_jarvis.onnx")
        wake_word_threshold = all_settings.get("wake_word_threshold", 0.5)
        wake_word_timeout = all_settings.get("wake_word_timeout", 3.0)
        wake_greetings = all_settings.get("wake_greetings", ["Hey, what's up?"])

        async def on_wake_detected():
            """Play wake greeting directly via TTS, bypassing agent turn-taking."""
            nonlocal _session_ref
            if _session_ref is None:
                logger.warning("Wake detected but session not ready yet")
                return

            try:
                # Pick a random greeting
                greeting = random.choice(wake_greetings)
                logger.info(f"Wake word detected, playing greeting: {greeting}")

                # Get TTS and audio output from session
                tts = _session_ref.tts
                audio_output = _session_ref.output.audio

                # Synthesize and push audio frames directly (bypasses turn-taking)
                audio_stream = tts.synthesize(greeting)
                async for event in audio_stream:
                    if hasattr(event, "frame") and event.frame:
                        await audio_output.capture_frame(event.frame)

                # Flush to complete the audio segment
                audio_output.flush()

            except Exception as e:
                logger.warning(f"Failed to play wake greeting: {e}")

        async def on_state_changed(state):
            """Publish wake word state to connected clients."""
            payload = json.dumps({
                "type": "wakeword_state",
                "state": state.value,
            })
            try:
                await ctx.room.local_participant.publish_data(
                    payload.encode("utf-8"),
                    reliable=True,
                    topic="wakeword_state",
                )
                logger.debug(f"Published wake word state: {state.value}")
            except Exception as e:
                logger.warning(f"Failed to publish wake word state: {e}")

        stt_instance = WakeWordGatedSTT(
            inner_stt=base_stt,
            model_path=wake_word_model,
            threshold=wake_word_threshold,
            silence_timeout=wake_word_timeout,
            on_wake_detected=on_wake_detected,
            on_state_changed=on_state_changed,
        )
        logger.info(f"  Wake word: ENABLED (model={wake_word_model}, threshold={wake_word_threshold})")
    else:
        stt_instance = base_stt
        logger.info("  Wake word: disabled")

    # Create TTS instance based on provider
    if runtime["tts_provider"] == "piper":
        # Piper runs through Speaches container - voice is baked into model ID
        tts_instance = openai.TTS(
            base_url=f"{SPEACHES_URL}/v1",
            api_key="not-needed",
            model=runtime["tts_voice_piper"],  # e.g., "speaches-ai/piper-en_US-ljspeech-medium"
            voice="default",  # Ignored by Piper but required by API
        )
    else:
        # Kokoro uses separate model and voice params
        tts_instance = openai.TTS(
            base_url=f"{KOKORO_URL}/v1",
            api_key="not-needed",
            model=TTS_MODEL,
            voice=runtime["tts_voice_kokoro"],
        )

    # Create session with STT and TTS (both OpenAI-compatible)
    logger.info(f"  STT instance type: {type(stt_instance).__name__}")
    logger.info(f"  STT capabilities: streaming={stt_instance.capabilities.streaming}")
    session = AgentSession(
        stt=stt_instance,
        llm=caal_llm,
        tts=tts_instance,
        vad=silero.VAD.load(),
        allow_interruptions=runtime["allow_interruptions"],
        min_endpointing_delay=runtime["min_endpointing_delay"],
    )
    logger.info(f"  Session STT: {type(session.stt).__name__}")

    # Set session reference for wake word callback
    _session_ref = session

    # ==========================================================================
    # Round-trip latency tracking
    # ==========================================================================

    _transcription_time: float | None = None

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev) -> None:
        nonlocal _transcription_time
        _transcription_time = time.perf_counter()
        logger.debug(f"User said: {ev.transcript[:80]}...")

    @session.on("agent_state_changed")
    def on_agent_state_changed(ev) -> None:
        nonlocal _transcription_time
        if ev.new_state == "speaking" and _transcription_time is not None:
            latency_ms = (time.perf_counter() - _transcription_time) * 1000
            logger.info(f"ROUND-TRIP LATENCY: {latency_ms:.0f}ms (LLM + TTS)")
            _transcription_time = None

        # Notify wake word STT of agent state for silence timer management
        if isinstance(stt_instance, WakeWordGatedSTT):
            stt_instance.set_agent_busy(ev.new_state in ("thinking", "speaking"))

    async def _publish_tool_status(
        tool_used: bool,
        tool_names: list[str],
        tool_params: list[dict],
    ) -> None:
        """Publish tool usage status to frontend via data packet."""
        import json
        payload = json.dumps({
            "tool_used": tool_used,
            "tool_names": tool_names,
            "tool_params": tool_params,
        })

        try:
            await ctx.room.local_participant.publish_data(
                payload.encode("utf-8"),
                reliable=True,
                topic="tool_status",
            )
            logger.debug(f"Published tool status: used={tool_used}, names={tool_names}")
        except Exception as e:
            logger.warning(f"Failed to publish tool status: {e}")

    # ==========================================================================

    # Create HASS tools only if Home Assistant is connected
    hass_tool_definitions = []
    hass_tool_callables = {}
    hass_server = mcp_servers.get("home_assistant")
    if hass_server:
        hass_tool_definitions, hass_tool_callables = create_hass_tools(hass_server)
        logger.info("Home Assistant tools enabled: hass_control, hass_get_state")

    # Create agent with CAALLLM and all MCP servers
    assistant = VoiceAssistant(
        caal_llm=caal_llm,
        mcp_servers=mcp_servers,
        n8n_workflow_tools=n8n_workflow_tools,
        n8n_workflow_name_map=n8n_workflow_name_map,
        n8n_base_url=n8n_base_url,
        on_tool_status=_publish_tool_status,
        tool_cache_size=runtime["tool_cache_size"],
        max_turns=runtime["max_turns"],
        hass_tool_definitions=hass_tool_definitions,
        hass_tool_callables=hass_tool_callables,
    )

    # Create event to wait for session close (BEFORE session.start to avoid race condition)
    close_event = asyncio.Event()

    @session.on("close")
    def on_session_close(ev) -> None:
        logger.info(f"Session closed: {ev.reason}")
        close_event.set()

    # ==========================================================================
    # Webhook Command Handler (via LiveKit data channel)
    # ==========================================================================

    async def _handle_webhook_command(data: rtc.DataPacket) -> None:
        """Handle commands from webhook server via LiveKit data channel."""
        if data.topic != "webhook_command":
            return

        try:
            import json

            cmd = json.loads(data.data.decode("utf-8"))
            action = cmd.get("action")
            logger.info(f"Received webhook command: {action}")

            if action == "announce":
                message = cmd.get("message", "")
                if message:
                    await session.say(message)

            elif action == "wake":
                # Get greeting from settings
                greetings = get_setting("wake_greetings")
                greeting = random.choice(greetings)
                await session.say(greeting)

            elif action == "reload_tools":
                # Clear agent's internal caches
                assistant._ollama_tools_cache = None

                # Re-discover n8n workflows if MCP is available
                n8n_mcp = assistant._caal_mcp_servers.get("n8n")
                if n8n_mcp and assistant._n8n_base_url:
                    try:
                        tools, name_map = await discover_n8n_workflows(
                            n8n_mcp, assistant._n8n_base_url
                        )
                        assistant._n8n_workflow_tools = tools
                        assistant._n8n_workflow_name_map = name_map
                        logger.info(f"Reloaded {len(tools)} n8n workflows")
                    except Exception as e:
                        logger.error(f"Failed to re-discover n8n workflows: {e}")

                # Announce if requested
                if msg := cmd.get("message"):
                    await session.say(msg)
                elif tool_name := cmd.get("tool_name"):
                    await session.say(f"A new tool called '{tool_name}' is now available.")

        except Exception as e:
            logger.error(f"Failed to process webhook command: {e}")

    @ctx.room.on("data_received")
    def on_data_received(data: rtc.DataPacket) -> None:
        """Sync wrapper for async webhook command handler."""
        asyncio.create_task(_handle_webhook_command(data))

    # Start session AFTER handlers are registered
    await session.start(
        room=ctx.room,
        agent=assistant,
    )

    # Send initial greeting with timeout to prevent hanging on unresponsive LLM
    try:
        await asyncio.wait_for(
            session.generate_reply(
                instructions="Greet the user briefly and let them know you're ready to help."
            ),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        logger.error("Initial greeting timed out (30s) - LLM may be unresponsive")
        # Continue anyway - user can still speak

    logger.info("Agent ready - listening for speech...")

    # Wait until session closes (room disconnects, etc.)
    await close_event.wait()


# =============================================================================
# Model Preloading
# =============================================================================


def preload_models():
    """Preload STT and LLM models on startup.

    Ensures models are ready before first user connection, avoiding
    delays on first request (especially important on HDDs).

    Skips preloading entirely if wizard not complete (no provider selected yet).
    Skips individual preloads when using cloud providers (Groq).
    Note: Kokoro (remsky/kokoro-fastapi) preloads its own models at startup.
    """
    settings = settings_module.load_settings()

    # Skip all preloading if wizard not complete
    if not settings.get("first_launch_completed", False):
        logger.info("Skipping model preload (wizard not complete)")
        return

    stt_provider = settings.get("stt_provider", "speaches")
    llm_provider = settings.get("llm_provider", "ollama")

    logger.info("Preloading models...")

    # Download Whisper STT model (skip if using Groq cloud STT)
    if stt_provider == "groq":
        logger.info("  Skipping STT preload (using Groq)")
    else:
        speaches_url = os.getenv("SPEACHES_URL", "http://speaches:8000")
        whisper_model = os.getenv("WHISPER_MODEL", "Systran/faster-whisper-medium")
        try:
            logger.info(f"  Loading STT: {whisper_model}")
            response = requests.post(
                f"{speaches_url}/v1/models/{whisper_model}",
                timeout=300
            )
            if response.status_code == 404:
                response = requests.post(
                    f"{speaches_url}/v1/models?model_name={whisper_model}",
                    timeout=300
                )
            if response.status_code == 200:
                logger.info("  ✓ STT ready")
            else:
                logger.warning(f"  STT model download returned {response.status_code}")
        except Exception as e:
            logger.warning(f"  Failed to preload STT model: {e}")

    # Warm up Ollama LLM (skip if using Groq cloud LLM)
    if llm_provider == "groq":
        logger.info("  Skipping LLM preload (using Groq)")
    else:
        ollama_host = settings.get("ollama_host") or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        ollama_model = settings.get("ollama_model") or os.getenv("OLLAMA_MODEL", "ministral-3:8b")
        ollama_num_ctx = settings.get("num_ctx", int(os.getenv("OLLAMA_NUM_CTX", "8192")))
        try:
            logger.info(f"  Loading LLM: {ollama_model} (num_ctx={ollama_num_ctx})")
            response = requests.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": "hi",
                    "stream": False,
                    "keep_alive": -1,
                    "options": {"num_ctx": ollama_num_ctx}
                },
                timeout=180
            )
            if response.status_code == 200:
                logger.info("  ✓ LLM ready")
            else:
                logger.warning(f"  LLM warmup returned {response.status_code}")
        except Exception as e:
            logger.warning(f"  Failed to preload LLM: {e}")


# =============================================================================
# Webhook Server (runs in background thread)
# =============================================================================

WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8889"))


def run_webhook_server_sync():
    """Run webhook server in a separate thread (blocking).

    This runs the webhook server in the same event loop as the LiveKit agent,
    avoiding cross-thread async issues that cause 200x slower MCP calls.
    
    If the port is already in use (another agent process started it), silently skip.
    This starts the webhook server immediately on agent startup,
    so /setup/status and other endpoints are available before
    any user connects.
    """
    import socket
    import uvicorn
    from caal.webhooks import app

    # Check if port is already in use before attempting to start server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(('0.0.0.0', WEBHOOK_PORT))
        sock.close()
    except OSError:
        # Port already in use - another agent started the webhook server
        logger.debug(f"Webhook server already running on port {WEBHOOK_PORT} (started by another agent)")
        return
    
    # Port is available - start the server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=WEBHOOK_PORT,
        log_level="warning",
        log_config=None,  # Don't configure logging (prevents duplicate handlers in forked workers)
    )
    server = uvicorn.Server(config)
    logger.info(f"Starting webhook server on port {WEBHOOK_PORT}")
    server.run()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import threading

    # Start webhook server in background thread (available immediately)
    webhook_thread = threading.Thread(target=run_webhook_server_sync, daemon=True)
    webhook_thread.start()

    # Preload models before starting worker
    preload_models()

    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            # Suppress memory warnings (models use ~1GB, this is expected)
            job_memory_warn_mb=0,
        )
    )
