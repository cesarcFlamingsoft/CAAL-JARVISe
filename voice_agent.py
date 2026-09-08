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

# Loading .env before optional integration imports is intentional.
# ruff: noqa: E402

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import requests

# Add src directory to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv

# Load environment variables from .env
_script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_script_dir, ".env"))

from livekit import agents, api, rtc
from livekit.agents import Agent, AgentSession, StopResponse, mcp
from livekit.plugins import groq as groq_plugin
from livekit.plugins import openai, silero

from caal import CAALLLM, conversation_ledger, user_api
from caal.alarm_delivery import announce_due_alarms
from caal.audio import (
    AudioEnergyGate,
    NoiseSuppressedSTT,
    create_speaker_recognition,
    create_tv_rejection_filter,
)
from caal.background_task_session import (
    BACKGROUND_CONTROL_REPLIES,
    BackgroundTaskBridge,
    DialCallback,
    DialUserCallback,
    LLMBackgroundWorker,
    callback_outcome_message,
    callback_unanswered_notification,
    capture_task_context,
)
from caal.background_tasks import MAX_CONCURRENCY_LIMIT, recover_interrupted
from caal.call_termination import (
    acknowledge_and_end_call,
    callback_requested,
    end_call_requested,
    end_livekit_room,
)
from caal.conversation import AdaptiveEndpointer
from caal.conversation_ledger import SESSION_LIVENESS_INTERVAL_SECONDS, ConversationRecorder
from caal.document_work import DocumentWorker
from caal.end_call_intent import END_CALL_CONTROL_REPLIES, EndCallAction, EndCallIntentMachine
from caal.handoff_context import inject_continuation_preamble, restore_conversation_context
from caal.handoff_intent import (
    HANDOFF_CONTROL_REPLIES,
    PhoneHandoffController,
    sole_handoff_destination,
)
from caal.integrations import (
    WebSearchTools,
    discover_n8n_workflows,
    initialize_mcp_servers,
    load_mcp_config,
)
from caal.internal_auth import AUDIENCE_AGENT, PrincipalError, verify_principal
from caal.llm import ToolDataCache, llm_node
from caal.outbound_calls import OutboundCallCoordinator, OutboundCallPolicy
from caal.outbound_runtime import (
    OutboundRoomConfig,
    call_timeouts,
    requires_fallback_notification,
)
from caal.security_config import load_multi_user_config, log_startup_status
from caal.settings import get_setting
from caal.stt import WakeWordGatedSTT
from caal.telegram_notify import TelegramCallNotifier
from caal.telephony_auth import CallAccessGate, GateState
from caal.user_scope import UserScope
from caal.work_router import (
    DEFAULT_ROUTER_TIMEOUT_SECONDS,
    SemanticWorkRouter,
    provider_classifier,
)

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
TTS_MODEL = os.getenv(
    "TTS_MODEL", "kokoro"
)  # "kokoro" for Kokoro-FastAPI, "prince-canuma/Kokoro-82M" for mlx-audio
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "false").lower() == "true"
TIMEZONE_ID = os.getenv("TIMEZONE", "America/Los_Angeles")
TIMEZONE_DISPLAY = os.getenv("TIMEZONE_DISPLAY", "Pacific Time")

# Import settings module for runtime-configurable values
from caal import settings as settings_module


async def authenticate_sip_call(room: rtc.Room, tts_instance: object) -> bool:
    """Require a keypad PIN before a SIP participant reaches the main agent.

    Web/LAN participants are unaffected. The temporary session has no STT, LLM,
    or tools, so untrusted callers cannot invoke Hermes while authenticating.
    """
    if os.getenv("CAAL_CALL_PIN_ENABLED", "false").lower() != "true":
        return True

    sip_kind = rtc.ParticipantKind.PARTICIPANT_KIND_SIP
    if not any(participant.kind == sip_kind for participant in room.remote_participants.values()):
        return True

    pin_hash = os.getenv("CAAL_CALL_PIN_HASH", "")
    if not pin_hash:
        logger.error("SIP call rejected: CAAL_CALL_PIN_HASH is not configured")
        return False

    try:
        max_attempts = int(os.getenv("CAAL_CALL_PIN_MAX_ATTEMPTS", "3"))
        gate = CallAccessGate(pin_hash, max_attempts=max_attempts)
    except (TypeError, ValueError):
        logger.error("SIP call rejected: invalid PIN-gate configuration")
        return False

    loop = asyncio.get_running_loop()
    outcome: asyncio.Future[bool] = loop.create_future()
    auth_session = AgentSession(tts=tts_instance, allow_interruptions=False)
    auth_agent = Agent(instructions="Telephone keypad authentication is in progress.")

    async def speak(text: str) -> None:
        try:
            auth_session.say(text, allow_interruptions=False, add_to_chat_ctx=False)
        except Exception:
            logger.warning("Could not play SIP access-gate prompt")

    # Digits are queued so the room callback returns immediately; the worker
    # verifies them in order with scrypt running off the audio event loop.
    digits: asyncio.Queue[str] = asyncio.Queue()

    def on_sip_dtmf(event: rtc.SipDTMF) -> None:
        digits.put_nowait(event.digit)

    async def consume_digits() -> None:
        while True:
            digit = await digits.get()
            state = await gate.accept_digit_async(digit)
            if state is GateState.RETRY:
                logger.info("SIP access PIN rejected; retry remaining=%s", gate.attempts_remaining)
                await speak("That code was not accepted. Please try again.")
            elif state is GateState.GRANTED:
                logger.info("SIP access PIN accepted")
                if not outcome.done():
                    outcome.set_result(True)
                return
            elif state is GateState.LOCKED:
                logger.warning("SIP access PIN attempts exhausted")
                if not outcome.done():
                    outcome.set_result(False)
                return

    room.on("sip_dtmf_received", on_sip_dtmf)
    digit_worker = asyncio.create_task(consume_digits())
    try:
        await auth_session.start(room=room, agent=auth_agent)
        await speak("This line is protected. Enter your four digit access code, then press pound.")
        return await asyncio.wait_for(outcome, timeout=60)
    except asyncio.TimeoutError:
        logger.info("SIP access PIN entry timed out")
        return False
    except Exception:
        logger.exception("SIP access gate failed closed")
        return False
    finally:
        room.off("sip_dtmf_received", on_sip_dtmf)
        digit_worker.cancel()
        try:
            await digit_worker
        except (asyncio.CancelledError, Exception):
            pass
        await auth_session.aclose()


def is_outbound_job(metadata: str) -> bool:
    """Report whether this job was dispatched to place an outbound call."""
    try:
        return json.loads(metadata or "{}").get("caal_outbound") is True
    except (json.JSONDecodeError, AttributeError):
        return False


def open_session_conversation(
    job_metadata: str, *, session_key: str, user_id: str | None = None
) -> str | None:
    """Open a private ledger conversation for an inbound session; return its id.

    Outbound jobs never open one: they join the conversation they were
    dispatched to continue (or run without any). A ledger failure disables
    continuity for the session rather than breaking it. ``user_id`` binds the
    conversation to the verified user so only their phone leg can continue it.
    """
    if is_outbound_job(job_metadata):
        return None
    try:
        conversation_ledger.purge_expired_conversations()
    except Exception:
        logger.warning("Could not purge expired conversation ledgers", exc_info=False)
    try:
        return conversation_ledger.open_conversation(session_key=session_key, user_id=user_id)
    except Exception:
        logger.warning("Conversation ledger unavailable; continuing without it", exc_info=False)
        return None


# =============================================================================
# Session identity
# =============================================================================
#
# A session acts for exactly one of: the legacy single user (multi-user not
# configured), nobody (anonymous: identity is configured but nothing verified),
# or a verified user named by an opaque id. Nothing here trusts a room name, a
# participant's own attributes, or anything spoken.


def load_identity_runtime() -> Any:
    """The multi-user identity runtime for this process.

    ``None`` means multi-user is not configured at all (legacy single-user
    mode). A configuration that was attempted but does not validate yields a
    locked runtime that verifies nothing, so every session runs anonymous
    rather than quietly falling back to shared single-user state.
    """
    status = load_multi_user_config()
    if status.enabled:
        return user_api.get_runtime()
    if status.attempted:
        logger.error(
            "SECURITY CONFIGURATION ERROR: %s Voice sessions run without any user identity "
            "(no memory, no phone handoff, no callbacks) until this is fixed.",
            status.describe(),
        )
        return user_api.LockedIdentityRuntime()
    return None


def _identity_store(identity: Any) -> Any:
    return getattr(identity, "store", None) if identity is not None else None


def describe_scope(scope: UserScope) -> str:
    if not scope.identity_configured:
        return "legacy single-user (multi-user identity not configured)"
    if scope.user_id is None:
        return "anonymous (no verified user; memory, handoff and callbacks disabled)"
    return f"verified user (role={scope.role})"


def resolve_inbound_scope(job_metadata: str, *, room_name: str, identity: Any) -> UserScope:
    """Identify a web/mobile session from the principal in its signed room configuration.

    The BFF mints a short-lived ``caal-agent`` principal for the verified user
    and places it in the LiveKit room configuration it signs, so it reaches the
    worker as job metadata that no participant can read or forge. It must be
    bound to this very room. Anything that fails to verify, or names a user
    who is not active, leaves the session anonymous.
    """
    if identity is None:
        return UserScope.legacy()
    store = _identity_store(identity)
    try:
        raw = json.loads(job_metadata or "{}")
    except (json.JSONDecodeError, TypeError):
        raw = None
    token = raw.get("caal_principal") if isinstance(raw, dict) else None
    if not isinstance(token, str) or not token:
        return UserScope.anonymous()
    if store is None:
        logger.warning("Session principal ignored: identity runtime is locked")
        return UserScope.anonymous()
    try:
        principal = verify_principal(
            token,
            secret=identity.config.internal_auth_secret,
            audience=AUDIENCE_AGENT,
            now=identity.now(),
            room=room_name,
        )
    except PrincipalError:
        logger.warning("Session principal failed verification; running anonymous")
        return UserScope.anonymous()
    profile = store.get_user(principal.subject)
    if profile is None or not profile.is_active:
        logger.warning("Session principal names no active user; running anonymous")
        return UserScope.anonymous()
    return UserScope.for_user(profile)


def has_sip_participant(room: Any) -> bool:
    sip_kind = rtc.ParticipantKind.PARTICIPANT_KIND_SIP
    participants = getattr(room, "remote_participants", {}) or {}
    return any(getattr(p, "kind", None) == sip_kind for p in participants.values())


def resolve_sip_caller_scope(room: Any, *, identity: Any) -> UserScope:
    """Identify a telephone caller by matching caller-id to an approved number.

    Only attributes of a *SIP* participant are consulted (a web participant
    claiming a phone number is ignored), and only after the DTMF PIN gate has
    already passed. No match means an anonymous session.
    """
    if identity is None:
        return UserScope.legacy()
    store = _identity_store(identity)
    if store is None:
        return UserScope.anonymous()
    sip_kind = rtc.ParticipantKind.PARTICIPANT_KIND_SIP
    participants = getattr(room, "remote_participants", {}) or {}
    for participant in participants.values():
        if getattr(participant, "kind", None) != sip_kind:
            continue
        attributes = getattr(participant, "attributes", None) or {}
        number = attributes.get("sip.phoneNumber")
        if not isinstance(number, str) or not number:
            continue
        try:
            profile = store.find_user_by_callback_number(number)
        except Exception:
            logger.warning("Caller-id lookup failed; running anonymous", exc_info=False)
            continue
        if profile is not None and profile.is_active:
            return UserScope.for_user(profile)
    return UserScope.anonymous()


def parse_outbound_config(job_metadata: str, *, identity: Any) -> OutboundRoomConfig | None:
    """Parse a dispatched outbound job, re-authorizing the destination first.

    A user-bound job is authorized against the user's *current* approved
    number read from the store now; a legacy job against the env allowlist.
    Raises ``PermissionError``/``ValueError`` exactly like the runtime parser.
    """
    store = _identity_store(identity)
    resolver = store.approved_callback_number if store is not None else None
    return OutboundRoomConfig.from_dispatch_metadata(
        job_metadata,
        allowed_destinations=os.getenv("CAAL_OUTBOUND_ALLOWED_DESTINATIONS", ""),
        resolve_user_destination=resolver,
    )


def outbound_scope(config: OutboundRoomConfig | None, *, identity: Any) -> UserScope:
    """The user an outbound leg acts for: the dispatched id, if still an active user."""
    if identity is None:
        return UserScope.legacy()
    store = _identity_store(identity)
    if config is None or config.user_id is None or store is None:
        return UserScope.anonymous()
    profile = store.get_user(config.user_id)
    if profile is None or not profile.is_active:
        logger.warning("Outbound leg names no active user; running anonymous")
        return UserScope.anonymous()
    return UserScope.for_user(profile)


def attach_conversation_capture(session: Any, recorder: ConversationRecorder) -> None:
    """Feed every history item LiveKit commits to the ledger recorder.

    The recorder decides what is worth keeping (visible user/assistant text
    only) and is inert until bound to a conversation, so this is safe to attach
    to every session. A malformed event never raises into the session.
    """

    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev: Any) -> None:
        try:
            recorder.record(getattr(ev, "item", None))
        except Exception:
            logger.warning("Conversation capture failed for one item", exc_info=False)


def close_session_conversation(recorder: ConversationRecorder, *, session_key: str) -> None:
    """Release this session's hold on its ledger conversation, if it has one."""
    conversation_id = recorder.conversation_id
    if conversation_id is None:
        return
    try:
        conversation_ledger.close_session(conversation_id, session_key=session_key)
    except Exception:
        logger.warning("Could not close conversation ledger session", exc_info=False)


def end_phone_leg_conversation(recorder: ConversationRecorder, *, session_key: str) -> None:
    """Close the phone leg's hold on the ledger, first flagging the origin to catch up.

    The marker is written only while the origin session's link is still
    active, so a web tab that already closed never gets one. Marking must
    never stop the leg from closing.
    """
    conversation_id = recorder.conversation_id
    if conversation_id is None:
        return
    try:
        conversation_ledger.mark_return_sync(conversation_id, phone_session_key=session_key)
    except Exception:
        logger.warning("Could not mark conversation return sync", exc_info=False)
    close_session_conversation(recorder, session_key=session_key)


def touch_session_conversation(recorder: ConversationRecorder, *, session_key: str) -> None:
    """Liveness heartbeat for this session's ledger link, if it has one."""
    conversation_id = recorder.conversation_id
    if conversation_id is None:
        return
    try:
        conversation_ledger.touch_session(conversation_id, session_key=session_key)
    except Exception:
        logger.warning("Could not refresh conversation ledger liveness", exc_info=False)


class ReturnSyncHydrator:
    """Bring a finished phone leg back into the origin session, before Hermes.

    Asked on every final user turn of the origin session. It claims the
    pending return marker atomically and, only when one was pending, injects
    the ledger context through the existing restore seam: durably into the
    agent's history and into the turn that is about to reach the LLM. The
    marker is acked only after both succeed; otherwise it is released so the
    next turn retries. Concurrent turns cannot both claim, and a second phone
    leg refreshes the single continuation preamble rather than duplicating
    it. Nothing is spoken, no content is logged, and a ledger failure leaves
    the turn untouched.
    """

    def __init__(self, *, recorder: ConversationRecorder, session_key: str) -> None:
        self._recorder = recorder
        self._session_key = session_key

    async def hydrate(self, agent: Any, turn_ctx: Any = None) -> bool:
        """Report whether this turn hydrated the origin session."""
        conversation_id = self._recorder.conversation_id
        if conversation_id is None:
            return False
        try:
            claim = conversation_ledger.claim_return_sync(
                conversation_id, session_key=self._session_key
            )
        except Exception:
            logger.warning("Could not claim conversation return sync", exc_info=False)
            return False
        if claim is None:
            return False
        try:
            restored = await restore_conversation_context(agent, claim.context)
            if turn_ctx is not None:
                inject_continuation_preamble(turn_ctx, claim.context)
        except Exception:
            logger.warning("Could not restore conversation return context", exc_info=False)
            self._settle(conversation_id, claim.token, applied=False)
            return False
        self._settle(conversation_id, claim.token, applied=True)
        return restored

    def _settle(self, conversation_id: str, token: str, *, applied: bool) -> None:
        try:
            if applied:
                conversation_ledger.ack_return_sync(
                    conversation_id, session_key=self._session_key, token=token
                )
            else:
                conversation_ledger.release_return_sync(
                    conversation_id, session_key=self._session_key, token=token
                )
        except Exception:
            # An unsettled claim is retried once its window lapses; nothing is lost.
            logger.warning("Could not settle conversation return sync", exc_info=False)


def build_phone_handoff_controller(
    ctx: agents.JobContext,
    *,
    conversation_id: str | None = None,
    user_scope: UserScope | None = None,
    identity: Any = None,
) -> PhoneHandoffController | None:
    """Let an inbound caller move the conversation to their one approved phone.

    Returns None when a handoff must not be offered at all: inside an outbound
    job (which would let a call re-dial itself), for an anonymous session under
    multi-user, or (legacy) when the allowlist does not name exactly one
    destination. For a verified user the destination is their own approved
    number, read from their profile both when they confirm and again when the
    call is dispatched; a caller-supplied number is never used. With a
    ``conversation_id`` the confirmed call carries only that opaque id and the
    phone leg hydrates from the private ledger.
    """
    if is_outbound_job(ctx.job.metadata):
        return None

    scope = user_scope or UserScope.legacy()
    if scope.identity_configured:
        store = _identity_store(identity)
        if scope.user_id is None or store is None:
            logger.info("Phone handoff disabled: session is not bound to a verified user")
            return None
        user_id = scope.user_id
        agent_name = os.getenv("CAAL_AGENT_NAME", "caal")

        def _resolve_destination() -> str | None:
            return store.approved_callback_number(user_id)

        async def _start_user_call(destination: str, **kwargs: Any) -> Any:
            # Independent second read: the policy is built from the profile,
            # then the confirmed destination must match it exactly.
            approved = store.approved_callback_number(user_id)
            if approved is None:
                raise PermissionError("User has no approved callback number")
            coordinator = OutboundCallCoordinator(
                policy=OutboundCallPolicy.for_destination(approved),
                livekit=ctx.api,
                agent_name=agent_name,
            )
            return await coordinator.start(destination, **kwargs)

        return PhoneHandoffController(
            start_call=_start_user_call,
            allowed_destinations="",
            conversation_id=conversation_id,
            user_id=user_id,
            destination_resolver=_resolve_destination,
        )

    allowed = os.getenv("CAAL_OUTBOUND_ALLOWED_DESTINATIONS", "")
    if sole_handoff_destination(allowed) is None:
        logger.info("Phone handoff disabled: allowlist does not name exactly one destination")
        return None

    try:
        policy = OutboundCallPolicy.from_csv(allowed)
    except ValueError:
        logger.warning("Phone handoff disabled: allowlist is not valid E.164 configuration")
        return None

    coordinator = OutboundCallCoordinator(
        policy=policy,
        livekit=ctx.api,
        agent_name=os.getenv("CAAL_AGENT_NAME", "caal"),
    )
    return PhoneHandoffController(
        start_call=coordinator.start,
        allowed_destinations=allowed,
        conversation_id=conversation_id,
    )


class LocalTurnHandler:
    """Route each user turn to the commands CAAL answers itself, exactly once.

    A spoken turn is claimed from ``user_input_transcribed``, which carries the
    final transcript. Typed LiveKit chat never fires that event, so its text
    only ever shows up as the new message on ``on_user_turn_completed``; without
    a second entry point it reaches the LLM, which has no outbound-call tool.
    Handoff replies are gated on the final STT result. LiveKit can commit a VAD
    turn before a slow remote STT emits its final transcript; the handler waits
    on that event for a short bounded interval instead of forwarding an empty or
    stale speech turn to Hermes. Typed LiveKit chat never starts that gate.
    """

    # Remote STT finalization can lag VAD by several seconds on PSTN media.
    # Control turns must wait long enough to speak their confirmation instead of
    # letting an empty VAD turn race ahead to Hermes.
    _FINAL_STT_GRACE_SECONDS = 5.0

    def __init__(
        self,
        *,
        phone_handoff: PhoneHandoffController | None,
        session: AgentSession,
        end_call: Callable[[], Awaitable[None]],
        background: BackgroundTaskBridge | None = None,
        arm_callback_and_end_call: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._phone_handoff = phone_handoff
        self._background = background
        self._session = session
        self._end_call = end_call
        # "Hang up and call me back when you're done" is only a local command
        # when something can actually arm that callback; otherwise Hermes gets it.
        self._arm_callback_and_end_call = arm_callback_and_end_call
        # Natural "we can wrap this up" / "call me when you finish" readings,
        # layered under the literal commands above. Uncertain readings only ask.
        self._end_call_intent = EndCallIntentMachine()
        self._speech_task: asyncio.Task[bool] | None = None
        self._speech_turn_claimed = False
        self._local_command_claimed = False
        self._awaiting_final_stt = False
        self._final_stt_ready = asyncio.Event()
        self._end_call_task: asyncio.Task[None] | None = None

    def on_speech_transcript_started(self) -> None:
        """Mark that the active turn is speech whose final STT is still due."""
        self._awaiting_final_stt = True
        self._final_stt_ready.clear()

    def on_final_transcript(self, transcript: str) -> None:
        """Claim a spoken turn and start handling it from the STT event."""
        self._awaiting_final_stt = False
        self._speech_turn_claimed = True
        self._final_stt_ready.set()
        if self._start_end_call_if_requested(transcript):
            self._local_command_claimed = True
            return
        self._speech_task = asyncio.create_task(self._handle_local_commands(transcript))

    async def turn_consumed(self, text: str) -> bool:
        """Report whether CAAL answered the turn that just ended by itself."""
        if self._awaiting_final_stt:
            try:
                await asyncio.wait_for(
                    self._final_stt_ready.wait(), timeout=self._FINAL_STT_GRACE_SECONDS
                )
            except asyncio.TimeoutError:
                # No transcript means no local action. Hermes receives the turn
                # normally rather than a fabricated handoff decision.
                self._awaiting_final_stt = False
            else:
                # ``on_final_transcript`` owns the completed speech turn now.
                return await self.turn_consumed(text)

        claimed, self._speech_turn_claimed = self._speech_turn_claimed, False
        local_command, self._local_command_claimed = self._local_command_claimed, False
        task, self._speech_task = self._speech_task, None

        if local_command:
            return True
        if task is not None:
            try:
                return await task
            except Exception:
                logger.exception("Phone handoff handling failed")
                return False
        if claimed:
            # A spoken turn already took its decision above; re-reading the same
            # words as typed text would answer or hang up twice.
            return False
        return await self._handle_typed_text(text)

    async def _handle_typed_text(self, text: str) -> bool:
        """Give typed chat the same local commands a spoken turn gets."""
        if not text.strip():
            return False
        if self._start_end_call_if_requested(text):
            return True
        return await self._handle_local_commands(text)

    async def _handle_local_commands(self, text: str) -> bool:
        """Phone handoff decides first; only an unclaimed turn can start background work.

        A pending handoff confirmation must be able to consume its answer even
        when that answer also mentions the background, so the order is fixed.
        Background work is read next: a request that is both work and "call me
        when you're done" is scheduled and asked about once, never told that
        nothing is running. Scheduling withdraws any exit question still
        pending, so a later "yes" can only ever answer the question actually
        asked. The inferred exit comes last, and a callback question must be
        able to consume its own "yes".
        """
        if self._phone_handoff is not None:
            try:
                if await self._phone_handoff.handle_final_transcript(text, self._session):
                    return True
            except Exception:
                logger.exception("Phone handoff handling failed")
                return False
        if self._background is not None:
            try:
                # A file request has a bounded, real delivery path. On an
                # outbound phone leg Cesar asked for it to end and callback
                # automatically after that delivery settles; ordinary work
                # keeps the existing explicit callback question.
                auto_callback = (
                    self._arm_callback_and_end_call
                    if self._arm_callback_and_end_call is not None
                    and self._background.has_deliverable_file_request(text)
                    else None
                )
                outcome = await self._background.process_turn(
                    text,
                    self._session,
                    offer_callback=self._arm_callback_and_end_call is not None and auto_callback is None,
                    auto_callback=auto_callback,
                )
            except Exception:
                logger.exception("Background task handling failed")
                outcome = None
            if outcome is not None and outcome.consumed:
                # A fresh command replaces a pending question rather than answering it.
                self._end_call_intent.reset()
                if outcome.callback_offered:
                    self._end_call_intent.offer_callback()
                return True
        try:
            if await self._handle_inferred_end_call(text):
                return True
        except Exception:
            logger.exception("End-call inference handling failed")
        return False

    def _start_end_call_if_requested(self, text: str) -> bool:
        """Hanging up outranks a handoff, so an explicit request ends the call.

        Returns whether the text was an explicit termination request, which
        stops it from starting or answering a handoff confirmation.

        The callback form ("hang up and call me back when you're done") is
        checked first because it is a distinct authorization, not a plain
        hang-up. It is only claimed here when a callback can be armed; with
        nothing injected the turn is released to Hermes. Both forms share one
        termination slot, so a second request while the room is already
        closing never terminates or arms twice.
        """
        if callback_requested(text):
            if self._arm_callback_and_end_call is None:
                return False
            self._start_termination_once(
                self._arm_callback_and_end_call,
                "Caller requested a callback when the background task finishes",
            )
            return True
        if not end_call_requested(text):
            return False
        self._start_termination_once(
            self._end_call, "Caller requested termination of the current call"
        )
        return True

    def _callback_available(self) -> bool:
        """A callback can only be offered when something can arm it and work is open."""
        if self._arm_callback_and_end_call is None or self._background is None:
            return False
        return self._background.can_arm_callback

    async def _handle_inferred_end_call(self, text: str) -> bool:
        """Act on the natural-language reading of an exit; return whether it consumed the turn.

        A DIRECT exit ends the call through the same single termination slot
        the literal command uses. A confirmed callback arms it the same way.
        Every other outcome is either a question, a refusal to hang up, or a
        turn handed back untouched. The literal commands belong to
        ``_start_end_call_if_requested`` alone: a literal callback command that
        nothing can arm is released to Hermes, never re-read as a question.
        """
        if callback_requested(text) or end_call_requested(text):
            return False
        decision = self._end_call_intent.observe(
            text, callback_available=self._callback_available()
        )
        if decision.action is EndCallAction.END_CALL:
            self._start_termination_once(self._end_call, "Caller signalled the end of the call")
        elif decision.action is EndCallAction.ARM_CALLBACK:
            if self._arm_callback_and_end_call is None:
                return False
            self._start_termination_once(
                self._arm_callback_and_end_call,
                "Caller confirmed a callback when the background task finishes",
            )
        if decision.reply is not None:
            try:
                # A confirmation is an authorization boundary, not filler. Do
                # not let the just-finished turn or a concurrent Hermes reply
                # interrupt it before the caller can hear it.
                try:
                    await self._session.say(decision.reply, allow_interruptions=False)
                except TypeError:
                    # Lightweight test/session doubles may not expose the
                    # LiveKit keyword; production AgentSession does.
                    await self._session.say(decision.reply)
            except Exception:
                logger.warning("End-call inference could not speak its reply", exc_info=True)
        return decision.consumed

    def _start_termination_once(
        self, terminate: Callable[[], Awaitable[None]], reason: str
    ) -> None:
        if self._end_call_task is None or self._end_call_task.done():
            logger.info(reason)
            self._end_call_task = asyncio.create_task(terminate())


DEFAULT_GREETING_INSTRUCTIONS = "Greet the user briefly and let them know you're ready to help."
HANDOFF_GREETING_INSTRUCTIONS = (
    "The user asked to continue an ongoing conversation on their phone and has "
    "just answered this call. Greet them briefly, acknowledge that you are "
    "picking up where you left off, and invite them to continue. Do not repeat "
    "or summarize earlier details unless they ask."
)


CALLBACK_GREETING_INSTRUCTIONS = (
    "You are calling the user back because they asked you to hang up and call them "
    "once the task you were working on in the background finished, and it has. "
    "Greet them briefly, say you are calling back about that task, and stop; the "
    "outcome itself is delivered right after your greeting, so do not guess at it "
    "or repeat earlier details."
)


def greeting_instructions(config: OutboundRoomConfig | None) -> str:
    """Pick the opening line: a callback or continuation greeting only when warranted."""
    if config is not None and config.is_callback:
        return CALLBACK_GREETING_INSTRUCTIONS
    if config is not None and config.carries_continuation:
        return HANDOFF_GREETING_INSTRUCTIONS
    return DEFAULT_GREETING_INSTRUCTIONS


async def announce_callback_outcome(
    session: AgentSession, config: OutboundRoomConfig | None
) -> bool:
    """Speak the settled task's outcome on an answered callback, once, after the greeting.

    Only ever runs on the human-answered path; every other AMD verdict ends
    the job before the greeting. A failure to speak is logged without content.
    """
    if config is None or config.callback_task_id is None:
        return False
    try:
        await session.say(callback_outcome_message(config.callback_task_id))
    except Exception:
        logger.warning("Could not speak background task callback outcome", exc_info=False)
        return False
    return True


async def run_outbound_call(
    ctx: agents.JobContext,
    session: AgentSession,
    config: OutboundRoomConfig,
    *,
    agent: Agent | None = None,
    recorder: ConversationRecorder | None = None,
    background: BackgroundTaskBridge | None = None,
) -> bool:
    """Dial only after AMD starts; never speak to a non-human answer.

    Handoff context, when present, reaches ``agent`` only after AMD positively
    classifies a human. A ledger continuation is claimed (its only read path)
    at that same moment, ``recorder`` is bound so the phone leg keeps
    appending to the same logical conversation, and ``background`` adopts
    that conversation's queued work so the caller can ask about it, cancel
    it, or arm a callback for it. Every other outcome leaves the session
    context-free, adopts nothing, and releases the pending continuation
    without reading it.
    """
    trunk_id = os.getenv("LIVEKIT_OUTBOUND_TRUNK_ID", "")
    if not trunk_id:
        logger.error("Outbound call cancelled: outbound trunk is not configured")
        await ctx.shutdown("outbound trunk unavailable")
        return False

    identity = f"caal-outbound-{config.attempt_id}"
    category = "unanswered"
    try:
        ringing_timeout, max_call_duration = call_timeouts(
            os.getenv("CAAL_OUTBOUND_RING_TIMEOUT_SECONDS", "30"),
            os.getenv("CAAL_OUTBOUND_MAX_DURATION_SECONDS", "900"),
        )
        async with agents.AMD(
            session,
            participant_identity=identity,
            interrupt_on_machine=True,
            ivr_detection=False,
            wait_until_finished=False,
        ) as detector:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    sip_trunk_id=trunk_id,
                    sip_call_to=config.destination,
                    room_name=ctx.room.name,
                    participant_identity=identity,
                    participant_name="JARVIS outbound call",
                    wait_until_answered=True,
                    ringing_timeout=ringing_timeout,
                    max_call_duration=max_call_duration,
                )
            )
            result = await detector.execute()
            category = result.category.value
    except Exception:
        logger.warning("Outbound call ended before a human answer", exc_info=True)
    else:
        if category == "human":
            logger.info("Outbound call answered by a human attempt=%s", config.attempt_id)
            await _hydrate_answered_call(
                config, agent=agent, recorder=recorder, background=background
            )
            return True

    if config.conversation_id is not None:
        # Nobody will pick this continuation up; drop it without reading it.
        try:
            conversation_ledger.release_continuation(
                config.conversation_id, session_key=config.attempt_id
            )
        except Exception:
            logger.warning("Could not release conversation continuation", exc_info=False)

    if requires_fallback_notification(category):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                notifier = TelegramCallNotifier(
                    token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
                    chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
                    client=client,
                )
                if config.callback_task_id is not None:
                    # A callback nobody answered still owes the user its outcome.
                    text = callback_unanswered_notification(config.callback_task_id)
                    if text:
                        await notifier.notify_text(text)
                else:
                    await notifier.notify_unanswered(category)
        except Exception:
            logger.warning("Could not deliver Telegram outbound-call fallback")
    logger.info("Outbound call ended silently category=%s attempt=%s", category, config.attempt_id)
    await ctx.shutdown("outbound call not answered by a human")
    return False


async def _hydrate_answered_call(
    config: OutboundRoomConfig,
    *,
    agent: Agent | None,
    recorder: ConversationRecorder | None,
    background: BackgroundTaskBridge | None = None,
) -> None:
    """Give a human-answered call its earlier conversation, exactly once.

    Only ever called after a positive human AMD verdict. Any failure means the
    call proceeds without history, never that it fails. Logs counts only.
    Ownership of the conversation's background work is adopted together with
    the continuation claim: a leg that could not claim it continues nothing
    and keeps its room-owned work.
    """
    context: object | None = config.snapshot
    if config.conversation_id is not None:
        try:
            context = conversation_ledger.claim_continuation(
                config.conversation_id, session_key=config.attempt_id, user_id=config.user_id
            )
        except Exception:
            logger.warning("Could not claim conversation continuation", exc_info=False)
            context = None
        if context is None:
            # Expired, unknown, or already claimed: answer as a fresh call.
            logger.info("Outbound call has no claimable continuation attempt=%s", config.attempt_id)
        else:
            if recorder is not None:
                try:
                    recorder.bind(config.conversation_id, session_key=config.attempt_id)
                except ValueError:
                    logger.warning("Could not bind conversation recorder", exc_info=False)
            if background is not None:
                try:
                    background.bind_conversation(config.conversation_id)
                except ValueError:
                    logger.warning("Could not adopt conversation background work", exc_info=False)
                else:
                    logger.info("Outbound call adopted the conversation's background work")
    if context is not None and agent is not None:
        try:
            await restore_conversation_context(agent, context)
        except Exception:
            logger.warning("Could not restore handoff context", exc_info=True)


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
        "llm_provider": user_settings.get("llm_provider") or os.getenv("LLM_PROVIDER", "hermes"),
        "temperature": settings.get("temperature", float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))),
        # Ollama settings
        "ollama_host": user_settings.get("ollama_host")
        or os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "ollama_model": user_settings.get("ollama_model")
        or os.getenv("OLLAMA_MODEL", "ministral-3:8b"),
        "num_ctx": settings.get("num_ctx", int(os.getenv("OLLAMA_NUM_CTX", "8192"))),
        "think": OLLAMA_THINK,  # Only applies to Ollama
        # Groq settings
        "groq_api_key": settings.get("groq_api_key") or os.getenv("GROQ_API_KEY", ""),
        "groq_model": user_settings.get("groq_model")
        or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        # Hermes Agent API settings
        "hermes_api_url": user_settings.get("hermes_api_url")
        or settings.get("hermes_api_url", "http://host.docker.internal:8642/v1"),
        "hermes_api_key": settings.get("hermes_api_key") or os.getenv("HERMES_API_KEY", ""),
        "hermes_model": user_settings.get("hermes_model")
        or settings.get("hermes_model", "hermes-agent"),
        # Shared settings
        "max_turns": settings.get("max_turns", int(os.getenv("OLLAMA_MAX_TURNS", "20"))),
        "tool_cache_size": settings.get("tool_cache_size", int(os.getenv("TOOL_CACHE_SIZE", "3"))),
        # Turn detection settings
        "allow_interruptions": settings.get("allow_interruptions", True),
        "min_endpointing_delay": settings.get("min_endpointing_delay", 0.5),
        # VAD tuning parameters (noise rejection + responsiveness)
        "vad_min_speech_duration": settings.get("vad_min_speech_duration", 0.1),
        "vad_min_silence_duration": settings.get("vad_min_silence_duration", 0.4),
        "vad_prefix_padding": settings.get("vad_prefix_padding", 0.3),
        "vad_activation_threshold": settings.get("vad_activation_threshold", 0.7),
        # Adaptive endpointing settings
        "adaptive_endpointing_enabled": settings.get("adaptive_endpointing_enabled", True),
        "endpointing_delay_after_question": settings.get("endpointing_delay_after_question", 0.25),
        "endpointing_delay_after_statement": settings.get("endpointing_delay_after_statement", 0.5),
        "endpointing_delay_initial_turns": settings.get("endpointing_delay_initial_turns", 0.7),
        # Noise suppression settings (DeepFilterNet)
        "noise_suppression_enabled": settings.get(
            "noise_suppression_enabled", False
        ),  # Off by default (requires extra dep)
        "noise_suppression_atten_db": settings.get("noise_suppression_atten_db", 100.0),
        # Energy gate settings (filter quiet/distant sounds like TV)
        "energy_gate_enabled": settings.get("energy_gate_enabled", True),  # On by default
        "energy_gate_threshold_db": settings.get("energy_gate_threshold_db", -35.0),
        # TV rejection settings (advanced spectral/temporal analysis) - disabled by default
        "tv_rejection_enabled": settings.get("tv_rejection_enabled", False),  # Off until tuned
        "tv_rejection_min_crest_factor": settings.get("tv_rejection_min_crest_factor", 1.5),
        "tv_rejection_min_liveness": settings.get("tv_rejection_min_liveness", 0.15),
        "tv_rejection_consecutive_passes": settings.get("tv_rejection_consecutive_passes", 4),
        # Background tasks
        "background_tasks_enabled": bool(settings.get("background_tasks_enabled", True)),
        "background_task_max_concurrency": max(
            1, min(int(settings.get("background_task_max_concurrency", 2)), MAX_CONCURRENCY_LIMIT)
        ),
        "background_task_timeout_seconds": max(
            1.0, float(settings.get("background_task_timeout_seconds", 600))
        ),
        # Telegram fallback channel (settings first, then the env the call fallback uses)
        "telegram_bot_token": settings.get("telegram_bot_token")
        or os.getenv("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": settings.get("telegram_chat_id") or os.getenv("TELEGRAM_CHAT_ID", ""),
        "telegram_owner_user_id": settings.get("telegram_owner_user_id", ""),
    }


BACKGROUND_NOTIFY_POLL_SECONDS = 3.0


def build_background_callback_dialer(ctx: agents.JobContext) -> DialCallback | None:
    """Let a settled background task call the user back, from whichever session settles it.

    Arming a callback is restricted to an outbound phone leg (see
    :func:`build_callback_arming`), but the task it arms usually runs on the
    web session that scheduled it, and that session's runner is the one that
    sees it settle. Every session therefore carries a dialer. Returns ``None``
    when the allowlist is not valid; the dial itself re-runs the same
    allowlist policy the control endpoint uses, so a stored destination can
    never widen it.
    """
    try:
        policy = OutboundCallPolicy.from_csv(os.getenv("CAAL_OUTBOUND_ALLOWED_DESTINATIONS", ""))
    except ValueError:
        logger.warning("Background callback disabled: allowlist is not valid E.164 configuration")
        return None
    coordinator = OutboundCallCoordinator(
        policy=policy,
        livekit=ctx.api,
        agent_name=os.getenv("CAAL_AGENT_NAME", "caal"),
    )

    async def _dial(destination: str, task_id: str) -> None:
        await coordinator.start(destination, callback_task_id=task_id)

    return _dial


def build_user_callback_dialer(ctx: agents.JobContext, *, identity: Any) -> DialUserCallback | None:
    """Dial a settled task's callback to the owning user's *current* approved number.

    The number is read from the user's profile at dial time (never stored with
    the task), the policy is built from that same read, and the dispatched job
    carries the opaque user id so the worker re-resolves it once more before
    the SIP call is placed. A user without a number cannot be called.
    """
    store = _identity_store(identity)
    if store is None:
        return None
    agent_name = os.getenv("CAAL_AGENT_NAME", "caal")

    async def _dial(user_id: str, task_id: str) -> None:
        approved = store.approved_callback_number(user_id)
        if approved is None:
            raise PermissionError("User has no approved callback number")
        coordinator = OutboundCallCoordinator(
            policy=OutboundCallPolicy.for_destination(approved),
            livekit=ctx.api,
            agent_name=agent_name,
        )
        await coordinator.start(approved, callback_task_id=task_id, user_id=user_id)

    return _dial


def build_callback_arming(
    ctx: agents.JobContext,
    *,
    session: AgentSession,
    bridge: BackgroundTaskBridge | None,
    outbound_config: Callable[[], OutboundRoomConfig | None],
) -> Callable[[], Awaitable[None]] | None:
    """Build "hang up and call me back" for an outbound phone leg; ``None`` elsewhere.

    Only a phone leg the user is already on may arm a callback, and only to
    the number that leg was dialed to: the destination is read from the
    verified dispatch configuration, never from the caller's words. The
    configuration is supplied lazily because it is bound after AMD confirms a
    human, which is still before the first user turn can arrive. Arming
    succeeds only for open work the bridge owns; otherwise the bridge says so
    and the line stays open. On success only this room is closed.
    """
    if bridge is None or not is_outbound_job(ctx.job.metadata):
        return None

    async def _arm_callback_and_end_call() -> None:
        try:
            config = outbound_config()
            if config is None:
                logger.warning("Callback requested before the outbound call was configured")
                return
            if config.user_id is not None:
                # User-bound: no number is stored; it is resolved at dial time.
                armed = await bridge.arm_callback(None, session, user_id=config.user_id)
            else:
                armed = await bridge.arm_callback(config.destination, session)
            if armed:
                await end_livekit_room(ctx.api.room, ctx.room.name)
        except Exception:
            logger.exception("Caller-requested callback arming failed")

    return _arm_callback_and_end_call


def build_background_task_bridge(
    runtime: dict,
    *,
    provider: Any,
    session_key: str,
    conversation_id: str | None = None,
    dial_callback: DialCallback | None = None,
    user_scope: UserScope | None = None,
    dial_user_callback: DialUserCallback | None = None,
) -> BackgroundTaskBridge | None:
    """Wire the background queue to the configured LLM and the Telegram fallback.

    Returns ``None`` when background tasks are disabled. Without Telegram
    credentials the bridge still runs; outcomes the session misses simply stay
    unannounced until the same room reconnects. ``session_key`` is the room;
    ``conversation_id`` is the ledger conversation the session belongs to, if
    any, so work is owned by the conversation rather than the room. Work of a
    verified user is filed under their opaque id, and such sessions get no
    shared Telegram fallback: that channel belongs to the operator, not to the
    user, so a user's outcome waits for their own next session or callback.

    Turns are read by a router backed by the same provider, so a request for a
    deliverable is recognised however it is phrased. Its call is bounded by
    ``work_router_timeout_seconds`` and falls back to the offline net, so a
    slow or unreachable model costs a semantic reading and nothing else.
    """
    if not runtime.get("background_tasks_enabled", True):
        return None
    scope = user_scope or UserScope.legacy()
    token = runtime.get("telegram_bot_token") or ""
    chat_id = runtime.get("telegram_chat_id") or ""
    fallback = None
    document_delivery = None
    telegram_owner_user_id = runtime.get("telegram_owner_user_id") or ""
    # A global Telegram chat is an operator channel, not a per-user inbox. A
    # signed-in user may receive a document there only when an administrator
    # explicitly bound that chat to their opaque profile id. Empty preserves
    # legacy-only delivery; every other verified user fails explicitly instead
    # of leaking a document to the shared operator channel.
    telegram_delivery_allowed = scope.user_id is None or scope.user_id == telegram_owner_user_id
    if token and chat_id and telegram_delivery_allowed:

        async def _send_telegram(text: str) -> None:
            async with httpx.AsyncClient(timeout=15.0) as client:
                notifier = TelegramCallNotifier(token=token, chat_id=chat_id, client=client)
                await notifier.notify_text(text)

        async def _send_telegram_document(*, filename: str, content: bytes, caption: str) -> None:
            async with httpx.AsyncClient(timeout=60.0) as client:
                notifier = TelegramCallNotifier(token=token, chat_id=chat_id, client=client)
                await notifier.send_document(filename=filename, content=content, caption=caption)

        fallback = _send_telegram
        document_delivery = _send_telegram_document
    compose = LLMBackgroundWorker(
        provider, timeout_seconds=runtime.get("background_task_timeout_seconds", 600)
    )
    artifact_dir = Path(os.getenv("CAAL_DATA_DIR", "/app/data")) / "documents"
    worker = DocumentWorker(
        compose=compose,
        deliver=document_delivery,
        artifact_dir=artifact_dir,
    )
    router_enabled = bool(runtime.get("work_router_enabled", True))
    # ``provider`` is normally CAALLLM for LiveKit, whose ``chat`` method has
    # the LiveKit streaming signature. The router needs the configured CAAL
    # provider's small completion API instead.
    router_provider = getattr(provider, "provider_instance", provider)
    work_router = SemanticWorkRouter(
        classify=provider_classifier(router_provider) if router_enabled else None,
        timeout_seconds=runtime.get(
            "work_router_timeout_seconds", DEFAULT_ROUTER_TIMEOUT_SECONDS
        ),
        enabled=router_enabled,
    )
    return BackgroundTaskBridge(
        execute=worker,
        session_key=session_key,
        conversation_id=conversation_id,
        fallback=fallback,
        max_concurrency=runtime.get("background_task_max_concurrency", 2),
        dial_callback=dial_callback,
        user_id=scope.user_id,
        dial_user_callback=dial_user_callback,
        work_router=work_router,
    )


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
# Home Assistant Assist API Integration
# =========================================================================
# Calls Home Assistant's Conversation API directly to interact with an
# AI assistant (like JARVIS 2.0) configured in Home Assistant.


def create_hass_tools(
    hass_host: str, hass_token: str, hass_agent_id: str
) -> tuple[list[dict], dict]:
    """Create Home Assistant Assist API tool.

    Args:
        hass_host: Home Assistant URL (e.g., http://10.0.0.50:8123)
        hass_token: Long-lived access token
        hass_agent_id: Conversation agent entity_id (e.g., conversation.ollama_conversation_2)

    Returns:
        tuple: (tool_definitions, tool_callables)
        - tool_definitions: List of tool definitions in OpenAI format for LLM
        - tool_callables: Dict mapping tool name to callable function
    """
    import httpx

    # Track conversation ID for context continuity
    conversation_state = {"conversation_id": None}

    async def hass_assist(text: str) -> str:
        """Send a request to Home Assistant's AI assistant and get a response.
        Use this for ANY smart home control: lights, switches, media, climate, etc.
        Parameters: text (required: what you want to do or ask, in natural language).
        """
        if not hass_host or not hass_token:
            return "Home Assistant is not configured"

        url = f"{hass_host.rstrip('/')}/api/conversation/process"
        headers = {
            "Authorization": f"Bearer {hass_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "agent_id": hass_agent_id,
        }
        # Include conversation_id for context continuity if we have one
        if conversation_state["conversation_id"]:
            payload["conversation_id"] = conversation_state["conversation_id"]

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

            logger.info("Home Assistant API response received")

            # Store conversation_id for follow-up requests
            if "conversation_id" in data:
                conversation_state["conversation_id"] = data["conversation_id"]

            # Extract speech response
            speech = data.get("response", {}).get("speech", {}).get("plain", {}).get("speech", "")
            if speech:
                logger.info(f"hass_assist returning speech: {speech}")
                return speech

            # Fallback to response_type if no speech
            response_type = data.get("response", {}).get("response_type", "unknown")
            fallback = f"Action completed ({response_type})"
            logger.info(f"hass_assist returning fallback: {fallback}")
            return fallback

        except httpx.HTTPStatusError as e:
            logger.error(f"hass_assist HTTP error: {e}")
            return f"Home Assistant error: {e.response.status_code}"
        except Exception as e:
            logger.error(f"hass_assist error: {e}")
            return f"Failed to communicate with Home Assistant: {e}"

    # Tool definitions in OpenAI format for LLM
    tool_definitions = [
        {
            "type": "function",
            "function": {
                "name": "hass_assist",
                "description": (
                    "Send a command or question to Home Assistant's AI assistant for smart home "
                    "control. Use this for ANY smart home request: turning lights on/off, "
                    "controlling media players, checking device states, adjusting climate, etc. "
                    "Pass natural "
                    "language - the assistant understands context."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": (
                                "Natural language command or question (e.g., 'turn on the office "
                                "lamp', 'what's the temperature?', 'play music in the living room')"
                            ),
                        },
                    },
                    "required": ["text"],
                },
            },
        },
    ]

    # Callable functions for tool execution
    tool_callables = {
        "hass_assist": hass_assist,
    }

    return tool_definitions, tool_callables


# =========================================================================
# Friday Assistant (Clawdbot) Integration
# =========================================================================
# Calls the Friday assistant (Clawdbot) for advanced AI assistance.


def create_friday_tools(
    friday_host: str, friday_token: str, friday_agent_id: str
) -> tuple[list[dict], dict]:
    """Create Friday assistant (Clawdbot) tool.

    Args:
        friday_host: Clawdbot host URL (e.g., http://10.0.0.55:18789)
        friday_token: Clawdbot API token
        friday_agent_id: Agent ID for x-clawdbot-agent-id header

    Returns:
        tuple: (tool_definitions, tool_callables)
        - tool_definitions: List of tool definitions in OpenAI format for LLM
        - tool_callables: Dict mapping tool name to callable function
    """
    import httpx

    # Track conversation messages for context continuity
    conversation_state = {"messages": []}

    async def friday(message: str) -> str:
        """Send a message to the Friday assistant and get a response.
        Use this when the user asks to 'call Friday', 'ask Friday',
        'talk to Friday assistant', or similar.
        Parameters: message (required: what you want to ask or say to Friday).
        """
        if not friday_host or not friday_token:
            return "Friday assistant is not configured"

        url = f"{friday_host.rstrip('/')}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {friday_token}",
            "Content-Type": "application/json",
            "x-clawdbot-agent-id": friday_agent_id,
        }

        # Add the new user message to conversation history
        conversation_state["messages"].append({"role": "user", "content": message})

        payload = {
            "model": "clawdbot",
            "messages": conversation_state["messages"],
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

            logger.info("Friday API response received")

            # Extract assistant response
            choices = data.get("choices", [])
            if choices and len(choices) > 0:
                assistant_message = choices[0].get("message", {})
                content = assistant_message.get("content", "")
                if content:
                    # Add assistant response to conversation history
                    conversation_state["messages"].append({"role": "assistant", "content": content})
                    logger.info(f"friday returning: {content[:100]}...")
                    return content

            return "Friday did not provide a response"

        except httpx.HTTPStatusError as e:
            logger.error(f"friday HTTP error: {e}")
            return f"Friday assistant error: {e.response.status_code}"
        except Exception as e:
            logger.error(f"friday error: {e}")
            return f"Failed to communicate with Friday assistant: {e}"

    # Tool definitions in OpenAI format for LLM
    tool_definitions = [
        {
            "type": "function",
            "function": {
                "name": "friday",
                "description": (
                    "Call the Friday assistant for help with complex questions, research, coding, "
                    "analysis, or any advanced AI assistance. Use this when the user explicitly "
                    "asks to 'call Friday', 'ask Friday', 'talk to Friday assistant', "
                    "'get Friday', or "
                    "similar requests for Friday's help."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": (
                                "The message, question, or request to send to the Friday assistant"
                            ),
                        },
                    },
                    "required": ["message"],
                },
            },
        },
    ]

    # Callable functions for tool execution
    tool_callables = {
        "friday": friday,
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
        friday_tool_definitions: list[dict] | None = None,
        friday_tool_callables: dict | None = None,
        turn_consumed: Callable[[str], Awaitable[bool]] | None = None,
        sync_return_context: Callable[[Any, Any], Awaitable[bool]] | None = None,
        user_scope: UserScope | None = None,
    ) -> None:
        super().__init__(
            instructions=load_prompt(),
            llm=caal_llm,  # Satisfies LLM interface requirement
        )

        # Store provider for llm_node access
        self._provider = caal_llm.provider_instance

        # Who this session acts for. User-scoped native tools (memory) are bound
        # to it by the LLM node; the model can never choose another scope.
        self._user_scope = user_scope or UserScope.legacy()

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

        # Friday assistant tools (Clawdbot)
        self._friday_tool_definitions = friday_tool_definitions or []
        self._friday_tool_callables = friday_tool_callables or {}

        # Callback for publishing tool status to frontend
        self._on_tool_status = on_tool_status

        # Context management: tool data cache and sliding window
        self._tool_data_cache = ToolDataCache(max_entries=tool_cache_size)
        self._max_turns = max_turns

        # Asks whether CAAL already handled the turn locally (e.g. phone handoff)
        self._turn_consumed = turn_consumed
        # Catches this session up after a phone leg ended, before the LLM sees the turn
        self._sync_return_context = sync_return_context

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        """Drop a turn CAAL answered itself, so the LLM never sees the command.

        A turn that does reach the LLM first picks up any phone leg that ended
        since the last one, so ``turn_ctx`` already carries that context.
        """
        text = getattr(new_message, "text_content", "")
        if not isinstance(text, str):
            text = ""
        if self._turn_consumed is not None and await self._turn_consumed(text):
            raise StopResponse
        if self._sync_return_context is not None:
            await self._sync_return_context(self, turn_ctx)

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
# Session Integrations (MCP / n8n)
# =============================================================================

INITIAL_GREETING = "JARVIS online. How may I help you?"


class SessionIntegrations:
    """Per-session tool integrations discovered before the agent starts."""

    def __init__(
        self,
        *,
        mcp_servers: dict,
        n8n_workflow_tools: list,
        n8n_workflow_name_map: dict,
        n8n_base_url: str | None = None,
    ) -> None:
        self.mcp_servers = mcp_servers
        self.n8n_workflow_tools = n8n_workflow_tools
        self.n8n_workflow_name_map = n8n_workflow_name_map
        self.n8n_base_url = n8n_base_url

    @classmethod
    def empty(cls) -> SessionIntegrations:
        return cls(mcp_servers={}, n8n_workflow_tools=[], n8n_workflow_name_map={})


def provider_manages_own_tools(llm_provider: str) -> bool:
    """Whether the configured LLM runs its own tool loop.

    Hermes executes tools inside its own process and is never given CAAL tool
    schemas, so a session must not pay for MCP connections or n8n discovery.
    """
    return (llm_provider or "").strip().lower() == "hermes"


async def initialize_session_integrations(
    ctx: agents.JobContext, runtime: dict
) -> SessionIntegrations:
    """Connect MCP servers and discover n8n workflows for this session."""
    if provider_manages_own_tools(runtime.get("llm_provider", "")):
        logger.info("  MCP/n8n: skipped (Hermes runs its own tool runtime)")
        return SessionIntegrations.empty()

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
                error_messages.append(
                    "n8n enabled but could not connect - check URL and token in Settings"
                )
            elif err.name == "home_assistant":
                error_messages.append(
                    "Home Assistant enabled but could not connect - check URL and token in Settings"
                )
            else:
                error_messages.append(f"MCP server '{err.name}' failed to connect: {err.error}")

        # Send error to frontend via data channel
        import json as json_module

        payload = json_module.dumps(
            {
                "type": "mcp_error",
                "errors": error_messages,
            }
        )
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

    return SessionIntegrations(
        mcp_servers=mcp_servers,
        n8n_workflow_tools=n8n_workflow_tools,
        n8n_workflow_name_map=n8n_workflow_name_map,
        n8n_base_url=n8n_base_url,
    )


def prewarm(proc: agents.JobProcess) -> None:
    """Load the Silero VAD once per worker process.

    Loading it on the first job adds startup cost to that caller's first reply;
    session-specific tuning is applied later by :func:`load_tuned_vad`.
    """
    try:
        proc.userdata["vad"] = silero.VAD.load()
        logger.info("Prewarmed Silero VAD")
    except Exception as e:
        logger.warning(f"Failed to prewarm Silero VAD: {e}")


def load_tuned_vad(runtime: dict, prewarmed=None):
    """Return a VAD tuned for this session, reusing the prewarmed model."""
    options = {
        # Higher than default to filter noise/TV
        "min_speech_duration": runtime.get("vad_min_speech_duration", 0.1),
        # Lower than default 0.55 for faster response
        "min_silence_duration": runtime.get("vad_min_silence_duration", 0.4),
        # Capture speech onset
        "prefix_padding_duration": runtime.get("vad_prefix_padding", 0.3),
        # Higher than default 0.5 to reject TV/background
        "activation_threshold": runtime.get("vad_activation_threshold", 0.7),
    }
    if prewarmed is not None:
        prewarmed.update_options(**options)
        return prewarmed
    return silero.VAD.load(**options)


async def deliver_initial_greeting(session) -> None:
    """Speak a fixed greeting instead of waiting on an LLM generation."""
    try:
        await asyncio.wait_for(session.say(INITIAL_GREETING), timeout=30.0)
    except asyncio.TimeoutError:
        logger.error("Initial greeting timed out (30s) - TTS may be unresponsive")
    except Exception:
        logger.warning("Could not deliver the initial greeting", exc_info=True)


# =============================================================================
# Agent Entrypoint
# =============================================================================


async def entrypoint(ctx: agents.JobContext) -> None:
    """Main entrypoint for the voice agent."""
    # Note: Webhook server is started in background thread at agent startup (main block)
    # This ensures /setup/status is available before users connect

    logger.debug(f"Joining room: {ctx.room.name}")
    await ctx.connect()

    # Get runtime settings (from settings.json with .env fallback)
    runtime = get_runtime_settings()

    integrations = await initialize_session_integrations(ctx, runtime)
    mcp_servers = integrations.mcp_servers
    n8n_workflow_tools = integrations.n8n_workflow_tools
    n8n_workflow_name_map = integrations.n8n_workflow_name_map
    n8n_base_url = integrations.n8n_base_url

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
            f"  LLM: Ollama ({runtime['ollama_model']}, think={runtime['think']}, "
            f"num_ctx={runtime['num_ctx']})"
        )
    elif runtime["llm_provider"] == "hermes":
        logger.info(
            f"  LLM: Hermes Agent ({runtime['hermes_api_url']}, model={runtime['hermes_model']})"
        )
    else:
        logger.info(f"  LLM: Groq ({runtime['groq_model']})")
    logger.info(f"  MCP: {list(mcp_servers.keys()) or 'None'}")
    logger.info(
        f"  Turn detection: interruptions={runtime['allow_interruptions']}, "
        f"endpointing_delay={runtime['min_endpointing_delay']}s"
    )
    logger.info(
        f"  VAD tuning: threshold={runtime['vad_activation_threshold']}, "
        f"min_silence={runtime['vad_min_silence_duration']}s, "
        f"min_speech={runtime['vad_min_speech_duration']}s"
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

    # Apply noise suppression wrapper (if enabled and DeepFilterNet available)
    base_stt = NoiseSuppressedSTT.create(base_stt, runtime)
    if runtime.get("noise_suppression_enabled", False):
        if isinstance(base_stt, NoiseSuppressedSTT):
            logger.info(
                f"  Noise suppression: ENABLED (DeepFilterNet, "
                f"atten={runtime.get('noise_suppression_atten_db', 100)}dB)"
            )
        else:
            logger.info("  Noise suppression: enabled but DeepFilterNet not available")
    else:
        logger.info("  Noise suppression: disabled")

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
            payload = json.dumps(
                {
                    "type": "wakeword_state",
                    "state": state.value,
                }
            )
            try:
                await ctx.room.local_participant.publish_data(
                    payload.encode("utf-8"),
                    reliable=True,
                    topic="wakeword_state",
                )
                logger.debug(f"Published wake word state: {state.value}")
            except Exception as e:
                logger.warning(f"Failed to publish wake word state: {e}")

        # Create energy gate for filtering TV/distant sounds
        energy_gate = None
        if runtime.get("energy_gate_enabled", True):
            energy_gate = AudioEnergyGate.from_settings(runtime)
            logger.info(
                "  Energy gate: ENABLED "
                f"(threshold={runtime.get('energy_gate_threshold_db', -35)}dB)"
            )
        else:
            logger.info("  Energy gate: disabled")

        # Create TV rejection filter for advanced spectral/temporal analysis
        tv_rejection_filter = create_tv_rejection_filter(runtime)
        if tv_rejection_filter:
            logger.info(
                "  TV rejection: ENABLED "
                f"(crest>={runtime.get('tv_rejection_min_crest_factor', 1.5)}, "
                f"liveness>={runtime.get('tv_rejection_min_liveness', 0.15)}, "
                f"passes>={runtime.get('tv_rejection_consecutive_passes', 4)})"
            )
        else:
            logger.info("  TV rejection: disabled (enable in settings to filter TV audio)")

        # Create speaker recognition for voice biometrics
        speaker_recognition = create_speaker_recognition(all_settings)
        if speaker_recognition:
            enrolled_count = len(speaker_recognition.list_speakers())
            logger.info(
                f"  Speaker recognition: ENABLED "
                f"(threshold={speaker_recognition.config.verification_threshold}, "
                f"enrolled={enrolled_count})"
            )
        else:
            logger.info("  Speaker recognition: disabled (enable in settings)")

        stt_instance = WakeWordGatedSTT(
            inner_stt=base_stt,
            model_path=wake_word_model,
            threshold=wake_word_threshold,
            silence_timeout=wake_word_timeout,
            on_wake_detected=on_wake_detected,
            on_state_changed=on_state_changed,
            energy_gate=energy_gate,
            tv_rejection_filter=tv_rejection_filter,
            speaker_recognition=speaker_recognition,
        )
        logger.info(
            f"  Wake word: ENABLED (model={wake_word_model}, threshold={wake_word_threshold})"
        )
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

    # SIP callers must clear the isolated DTMF gate before the Hermes-backed
    # session is created. Web/LAN participants proceed without this extra step.
    if not await authenticate_sip_call(ctx.room, tts_instance):
        logger.info("Telephone session ended before access was granted")
        return

    # Who this session acts for (see "Session identity" above). An outbound leg
    # is re-authorized against the dispatched user's current number before any
    # dial; a telephone caller is matched by caller-id only after the PIN gate;
    # a web session is identified by the principal in its signed room config.
    identity = load_identity_runtime()
    outbound_config: OutboundRoomConfig | None = None
    if is_outbound_job(ctx.job.metadata):
        try:
            outbound_config = parse_outbound_config(ctx.job.metadata, identity=identity)
        except (PermissionError, ValueError):
            logger.error("Rejected invalid outbound room configuration", exc_info=True)
            await ctx.shutdown("invalid outbound room configuration")
            return
        user_scope = outbound_scope(outbound_config, identity=identity)
    elif has_sip_participant(ctx.room):
        user_scope = resolve_sip_caller_scope(ctx.room, identity=identity)
    else:
        user_scope = resolve_inbound_scope(
            ctx.job.metadata, room_name=ctx.room.name, identity=identity
        )
    logger.info("  Session identity: %s", describe_scope(user_scope))

    # Create session with STT and TTS (both OpenAI-compatible)
    logger.info(f"  STT instance type: {type(stt_instance).__name__}")
    logger.info(f"  STT capabilities: streaming={stt_instance.capabilities.streaming}")
    # Load VAD with tuned parameters for better noise rejection and responsiveness
    vad_instance = silero.VAD.load(
        min_speech_duration=runtime.get(
            "vad_min_speech_duration", 0.1
        ),  # Higher than default to filter noise/TV
        min_silence_duration=runtime.get(
            "vad_min_silence_duration", 0.4
        ),  # Lower than default 0.55 for faster response
        prefix_padding_duration=runtime.get("vad_prefix_padding", 0.3),  # Capture speech onset
        activation_threshold=runtime.get(
            "vad_activation_threshold", 0.7
        ),  # Higher than default 0.5 to reject TV/background
    )

    # Keep the turn open long enough for remote STT's final result. The prior
    # 0.5s deprecated endpointing setting allowed LiveKit to commit first,
    # sending a spoken handoff to Hermes before CAAL could intercept it.
    endpointing_delay = max(1.0, float(runtime["min_endpointing_delay"]))
    session = AgentSession(
        stt=stt_instance,
        llm=caal_llm,
        tts=tts_instance,
        vad=vad_instance,
        turn_handling={
            "endpointing": {"mode": "fixed", "min_delay": endpointing_delay},
            "interruption": {"enabled": runtime["allow_interruptions"]},
        },
    )
    logger.info(f"  Session STT: {type(session.stt).__name__}")

    # Set session reference for wake word callback
    _session_ref = session

    # ==========================================================================
    # Round-trip latency tracking
    # ==========================================================================

    _transcription_time: float | None = None

    # Private ledger of this logical conversation, so a phone handoff can pick
    # up a long conversation without squeezing it into dispatch metadata. Every
    # visible turn LiveKit commits is recorded; the handoff's own control
    # replies are not. Outbound jobs bind later, once a human answers.
    ledger_session_key = ctx.room.name
    session_conversation_id = open_session_conversation(
        ctx.job.metadata, session_key=ledger_session_key, user_id=user_scope.user_id
    )
    conversation_recorder = ConversationRecorder(
        exclude_assistant_texts=HANDOFF_CONTROL_REPLIES
        + BACKGROUND_CONTROL_REPLIES
        + END_CALL_CONTROL_REPLIES,
        session_key=ledger_session_key,
    )
    if session_conversation_id is not None:
        conversation_recorder.bind(session_conversation_id)
    attach_conversation_capture(session, conversation_recorder)
    # Only an origin session can be caught up after its phone leg ends.
    return_sync = (
        ReturnSyncHydrator(recorder=conversation_recorder, session_key=ledger_session_key)
        if session_conversation_id is not None
        else None
    )

    # CAAL answers local safety commands itself, before they reach Hermes.
    phone_handoff = build_phone_handoff_controller(
        ctx,
        conversation_id=session_conversation_id,
        user_scope=user_scope,
        identity=identity,
    )

    async def _end_call_at_caller_request() -> None:
        """Acknowledge and close only this LiveKit room after an explicit request."""
        try:
            await acknowledge_and_end_call(session, ctx.api.room, ctx.room.name)
        except Exception:
            logger.exception("Caller-requested call termination failed")

    # Long-running requests run on the durable queue and report back here.
    # Background work belongs to the logical conversation, not the transient
    # LiveKit room, so a task started on web stays eligible for a callback
    # request after the user hands that same conversation to the phone. An
    # outbound leg adopts the conversation in run_outbound_call once a human
    # answers; until then it owns only its room. Whichever session's runner
    # settles the task places the callback, so every session carries a dialer.
    background_bridge = build_background_task_bridge(
        runtime,
        provider=caal_llm.provider_instance,
        session_key=ctx.room.name,
        conversation_id=session_conversation_id,
        dial_callback=build_background_callback_dialer(ctx),
        user_scope=user_scope,
        dial_user_callback=build_user_callback_dialer(ctx, identity=identity),
    )
    # "Hang up and call me back" exists only on a phone leg; the outbound
    # configuration it reads is bound below, before the first user turn.
    arm_callback_and_end_call = build_callback_arming(
        ctx,
        session=session,
        bridge=background_bridge,
        outbound_config=lambda: outbound_config,
    )

    local_turn_handler = LocalTurnHandler(
        phone_handoff=phone_handoff,
        session=session,
        end_call=_end_call_at_caller_request,
        background=background_bridge,
        arm_callback_and_end_call=arm_callback_and_end_call,
    )

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev) -> None:
        nonlocal _transcription_time
        _transcription_time = time.perf_counter()
        logger.debug(f"User said: {ev.transcript[:80]}...")
        if ev.is_final:
            local_turn_handler.on_final_transcript(ev.transcript)
        else:
            local_turn_handler.on_speech_transcript_started()

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

    # ==========================================================================
    # Adaptive Endpointing (context-aware turn detection)
    # ==========================================================================

    adaptive_endpointer = AdaptiveEndpointer.from_settings(runtime)
    _last_agent_text: str = ""

    if runtime.get("adaptive_endpointing_enabled", True):
        logger.info(
            f"  Adaptive endpointing: ENABLED "
            f"(question={runtime['endpointing_delay_after_question']}s, "
            f"statement={runtime['endpointing_delay_after_statement']}s, "
            f"initial={runtime['endpointing_delay_initial_turns']}s)"
        )

        @session.on("agent_speech_committed")
        def on_agent_speech_committed(ev) -> None:
            """Track agent utterances for adaptive endpointing."""
            nonlocal _last_agent_text
            if hasattr(ev, "content") and ev.content:
                _last_agent_text += ev.content

        @session.on("agent_state_changed")
        def on_agent_state_adaptive(ev) -> None:
            """Update endpointing delay when agent finishes speaking."""
            nonlocal _last_agent_text
            if ev.new_state == "listening" and _last_agent_text:
                # Agent finished speaking - calculate new delay
                new_delay = adaptive_endpointer.on_agent_utterance(_last_agent_text)
                _last_agent_text = ""

                # Update VAD's min_silence_duration dynamically
                # This affects how quickly we detect end-of-speech
                try:
                    vad_instance.update_options(min_silence_duration=new_delay)
                    logger.debug(f"Updated VAD min_silence_duration to {new_delay:.2f}s")
                except Exception as e:
                    logger.debug(f"Could not update VAD options: {e}")

        @session.on("user_started_speaking")
        def on_user_started_adaptive(ev) -> None:
            """Track when user starts speaking."""
            adaptive_endpointer.on_user_started_speaking()

        @session.on("user_stopped_speaking")
        def on_user_stopped_adaptive(ev) -> None:
            """Track when user finishes speaking."""
            adaptive_endpointer.on_user_finished_speaking()

        @session.on("agent_speech_interrupted")
        def on_agent_interrupted(ev) -> None:
            """Track interruptions for learning user patterns."""
            adaptive_endpointer.on_interruption()
            logger.debug("User interrupted agent - tracking for adaptive delay")
    else:
        logger.info("  Adaptive endpointing: disabled")

    # ==========================================================================

    async def _publish_tool_status(
        tool_used: bool,
        tool_names: list[str],
        tool_params: list[dict],
    ) -> None:
        """Publish tool usage status to frontend via data packet."""
        import json

        payload = json.dumps(
            {
                "tool_used": tool_used,
                "tool_names": tool_names,
                "tool_params": tool_params,
            }
        )

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

    # Create HASS tools if Home Assistant is enabled (uses Assist API directly)
    hass_tool_definitions = []
    hass_tool_callables = {}
    if all_settings.get("hass_enabled", False):
        hass_host = all_settings.get("hass_host", "")
        hass_token = all_settings.get("hass_token", "")
        hass_agent_id = all_settings.get("hass_agent_id", "conversation.home_assistant")
        if hass_host and hass_token:
            hass_tool_definitions, hass_tool_callables = create_hass_tools(
                hass_host=hass_host,
                hass_token=hass_token,
                hass_agent_id=hass_agent_id,
            )
            logger.info(f"Home Assistant Assist enabled: agent={hass_agent_id}")

    # Create Friday tools if Friday assistant is enabled
    friday_tool_definitions = []
    friday_tool_callables = {}
    if all_settings.get("friday_enabled", False):
        friday_host = all_settings.get("friday_host", "")
        friday_token = all_settings.get("friday_token", "")
        friday_agent_id = all_settings.get("friday_agent_id", "main")
        if friday_host and friday_token:
            friday_tool_definitions, friday_tool_callables = create_friday_tools(
                friday_host=friday_host,
                friday_token=friday_token,
                friday_agent_id=friday_agent_id,
            )
            logger.info(f"Friday assistant enabled: host={friday_host}, agent={friday_agent_id}")

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
        friday_tool_definitions=friday_tool_definitions,
        friday_tool_callables=friday_tool_callables,
        turn_consumed=local_turn_handler.turn_consumed,
        sync_return_context=return_sync.hydrate if return_sync is not None else None,
        user_scope=user_scope,
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
                # Clear agent's internal caches (both old and new cache names for compatibility)
                assistant._ollama_tools_cache = None
                assistant._llm_tools_cache = None

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

    async def _alarm_delivery_loop() -> None:
        """Deliver persisted timers once a user has an active voice session."""
        while not close_event.is_set():
            try:
                delivered = await announce_due_alarms(session)
                if delivered:
                    logger.info(f"Delivered {delivered} due alarm(s)")
            except Exception as e:
                logger.warning(f"Failed to deliver due alarms: {e}")
            await asyncio.sleep(5)

    async def _ledger_liveness_loop() -> None:
        """Keep this session's ledger link fresh for as long as the session is open."""
        while not close_event.is_set():
            await asyncio.sleep(SESSION_LIVENESS_INTERVAL_SECONDS)
            touch_session_conversation(conversation_recorder, session_key=ledger_session_key)

    async def _background_task_loop() -> None:
        """Announce finished background work here, and hand stale outcomes to Telegram."""
        while not close_event.is_set():
            try:
                delivered = await background_bridge.deliver_pending(session)
                if delivered:
                    logger.info(f"Announced {delivered} background task outcome(s)")
                sent = await background_bridge.flush_stale_to_fallback()
                if sent:
                    logger.info(f"Sent {sent} stale background task outcome(s) to Telegram")
            except Exception as e:
                logger.warning(f"Background task delivery failed: {e}")
            await asyncio.sleep(BACKGROUND_NOTIFY_POLL_SECONDS)

    async def _cancel(task: asyncio.Task | None) -> None:
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    alarm_delivery_task: asyncio.Task | None = None
    liveness_task: asyncio.Task | None = None
    background_task: asyncio.Task | None = None
    # The ledger lives only as long as the last linked session, so its release
    # must run however this session ends: a failed start, a rejected outbound
    # configuration, an unanswered dial, a greeting error, or a normal close.
    # An idle TTL still covers the case where even this never runs.
    try:
        # Start session AFTER handlers are registered
        await session.start(
            room=ctx.room,
            agent=assistant,
        )
        liveness_task = asyncio.create_task(_ledger_liveness_loop())

        # Outbound rooms are dispatched before the SIP participant is created so
        # AMD can start listening before the callee answers. They never enter the
        # normal greeting path unless AMD positively identifies a human. The
        # configuration was parsed and re-authorized above, before any session
        # state existed.
        if outbound_config is not None:
            ledger_session_key = outbound_config.attempt_id
            # A human answer binds the recorder and hands the conversation's
            # background work (and any callback on it) to this phone leg.
            if not await run_outbound_call(
                ctx,
                session,
                outbound_config,
                agent=assistant,
                recorder=conversation_recorder,
                background=background_bridge,
            ):
                return

        alarm_delivery_task = asyncio.create_task(_alarm_delivery_loop())
        if background_bridge is not None:
            await background_bridge.start(context_source=lambda: capture_task_context(session))
            background_task = asyncio.create_task(_background_task_loop())

        # Send initial greeting with timeout to prevent hanging on unresponsive LLM
        try:
            await asyncio.wait_for(
                session.generate_reply(instructions=greeting_instructions(outbound_config)),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.error("Initial greeting timed out (30s) - LLM may be unresponsive")
            # Continue anyway - user can still speak

        # A callback exists to deliver one settled outcome; say it right away.
        await announce_callback_outcome(session, outbound_config)

        logger.info("Agent ready - listening for speech...")

        # Wait until session closes (room disconnects, etc.)
        await close_event.wait()
    finally:
        # A phone leg on its way out flags the still-open origin session to
        # catch up on its next turn; the origin session merely closes.
        if outbound_config is not None:
            end_phone_leg_conversation(conversation_recorder, session_key=ledger_session_key)
        else:
            close_session_conversation(conversation_recorder, session_key=ledger_session_key)
        await _cancel(liveness_task)
        await _cancel(alarm_delivery_task)
        await _cancel(background_task)
        if background_bridge is not None:
            # The work carries on in this process without blocking teardown;
            # outcomes this session can no longer hear go to Telegram, or wait
            # for the room to reconnect. Only a process restart interrupts them.
            try:
                await background_bridge.close()
            except Exception:
                logger.warning("Background task session close failed", exc_info=True)


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
            response = requests.post(f"{speaches_url}/v1/models/{whisper_model}", timeout=300)
            if response.status_code == 404:
                response = requests.post(
                    f"{speaches_url}/v1/models?model_name={whisper_model}", timeout=300
                )
            if response.status_code == 200:
                logger.info("  ✓ STT ready")
            else:
                logger.warning(f"  STT model download returned {response.status_code}")
        except Exception as e:
            logger.warning(f"  Failed to preload STT model: {e}")

    # Warm up Ollama only when it is the active LLM. Hermes manages its own
    # provider/runtime lifecycle and must not trigger a local-model preload.
    if llm_provider != "ollama":
        logger.info(f"  Skipping Ollama preload (using {llm_provider})")
    else:
        ollama_host = settings.get("ollama_host") or os.getenv(
            "OLLAMA_HOST", "http://localhost:11434"
        )
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
                    "options": {"num_ctx": ollama_num_ctx},
                },
                timeout=180,
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
        sock.bind(("0.0.0.0", WEBHOOK_PORT))
        sock.close()
    except OSError:
        # Port already in use - another agent started the webhook server
        logger.debug(
            f"Webhook server already running on port {WEBHOOK_PORT} (started by another agent)"
        )
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

    # Multi-user identity: validate loudly at startup so a half-configured
    # deployment is noticed before the first session. Until every setting
    # validates, the identity endpoints answer 503 and voice sessions carry no
    # user; nothing pretends to be multi-user.
    log_startup_status(load_multi_user_config(), logger=logger)

    # Preload models before starting worker
    preload_models()

    # Background work left running by a previous worker life can never finish;
    # report it as interrupted once, here, rather than per job process.
    try:
        recover_interrupted()
    except Exception:
        logger.warning("Could not recover interrupted background tasks", exc_info=True)

    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            # Suppress memory warnings (models use ~1GB, this is expected)
            job_memory_warn_mb=0,
            agent_name=os.getenv("CAAL_AGENT_NAME", "caal"),
        )
    )
