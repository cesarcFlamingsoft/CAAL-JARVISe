"""The durable background work service: a small, supervised, internal worker.

This process exists so that queued work and armed callbacks survive the thing
that scheduled them. It joins no LiveKit room, answers no public traffic, and
holds no credentials of its own beyond the ones the agent already has: it
reads the same SQLite queue, runs the same worker pipeline, and hands finished
callbacks to LiveKit as ordinary isolated outbound jobs.

Its HTTP surface is deliberately tiny and internal:

* ``GET  /healthz``       liveness for the container probe; no authentication,
  no state, nothing but the fact that the process is up.
* ``GET  /status``        counts and timings only, behind the shared internal
  secret. Never an id, a number, or any task text.
* ``POST /internal/wake`` an authenticated nudge to tick now instead of at the
  next poll. It is an optimisation, never a requirement: the supervisor polls
  on its own and is correct with this endpoint never called at all.

There is no unauthenticated path to execution here, and no port is published
to the host: the service is reachable only on the internal compose network.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import signal
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException

from . import background_tasks
from . import settings as settings_module
from .background_task_session import LLMBackgroundWorker
from .background_tasks import BackgroundTask
from .coding_delegation import build_coding_delegate
from .document_work import DocumentWorker
from .durable_work import (
    CallbackDispatcher,
    DurableWorkSupervisor,
    LiveKitCallbackPlacer,
)
from .internal_auth import MIN_SECRET_LENGTH
from .reminder_dispatch import ReminderDispatcher
from .llm.providers import create_provider_from_settings
from .local_ollama import configured_endpoint
from .model_routing import Destination, classify_request

logger = logging.getLogger(__name__)

TOKEN_HEADER = "X-CAAL-Worker-Token"
DEFAULT_PORT = 8890


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def worker_settings() -> dict[str, Any]:
    """The subset of the runtime configuration this service needs.

    Same precedence the agent uses (explicit user settings, then environment,
    then defaults), so the worker runs the work on exactly the model the agent
    would have run it on.
    """
    settings = settings_module.load_settings()
    user_settings = settings_module.load_user_settings()
    runtime: dict[str, Any] = {}
    runtime["llm_provider"] = user_settings.get("llm_provider") or os.getenv(
        "LLM_PROVIDER", "routed"
    )
    runtime["temperature"] = settings.get("temperature", 0.7)
    runtime["ollama_host"] = configured_endpoint(user_settings)
    runtime["ollama_model"] = user_settings.get("ollama_model") or os.getenv(
        "OLLAMA_MODEL", "ministral-3:8b"
    )
    runtime["num_ctx"] = settings.get("num_ctx", 8192)
    runtime["groq_api_key"] = settings.get("groq_api_key") or os.getenv("GROQ_API_KEY", "")
    runtime["groq_model"] = user_settings.get("groq_model") or os.getenv(
        "GROQ_MODEL", "llama-3.3-70b-versatile"
    )
    runtime["hermes_api_url"] = user_settings.get("hermes_api_url") or settings.get(
        "hermes_api_url", "http://host.docker.internal:8642/v1"
    )
    runtime["hermes_api_key"] = settings.get("hermes_api_key") or os.getenv("HERMES_API_KEY", "")
    runtime["hermes_model"] = user_settings.get("hermes_model") or settings.get(
        "hermes_model", "hermes-agent"
    )
    runtime["coding_delegation_enabled"] = settings.get("coding_delegation_enabled", True)
    runtime["coding_delegation_timeout_seconds"] = settings.get(
        "coding_delegation_timeout_seconds", 900
    )
    runtime["background_tasks_enabled"] = bool(settings.get("background_tasks_enabled", True))
    runtime["background_task_max_concurrency"] = max(
        1,
        min(
            int(settings.get("background_task_max_concurrency", 2)),
            background_tasks.MAX_CONCURRENCY_LIMIT,
        ),
    )
    runtime["background_task_timeout_seconds"] = max(
        1.0, float(settings.get("background_task_timeout_seconds", 600))
    )
    return runtime


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# The worker pipeline
# ---------------------------------------------------------------------------


class DurableWorker:
    """Runs one queued task with the pipeline a voice session would have used.

    The conversation snapshot a session passes to its own worker lived only in
    that session's memory and was deliberately never persisted, so durable work
    is carried out from the stored request alone. Which worker runs it is
    decided here, offline and deterministically, exactly as the session decides
    it: a coding request goes to the agent runtime, everything else to the
    configured model.
    """

    def __init__(self, *, compose: Any, coding: Any | None = None) -> None:
        self._compose = compose
        self._coding = coding

    async def __call__(self, task: BackgroundTask) -> str:
        if self._coding is not None and self._is_coding(task.request):
            # A coding job gets the request, never the conversation.
            return await self._coding(task.request, "")
        return await self._compose(task.request, "")

    @staticmethod
    def _is_coding(request: str) -> bool:
        try:
            return classify_request(request).destination is Destination.CODING
        except Exception:
            # No exception text: it could carry the request into the log.
            logger.warning("could not classify durable work for coding delegation")
            return False


def build_worker(runtime: dict[str, Any]) -> DurableWorker:
    """The same composition the agent builds, minus anything session-shaped."""
    provider = create_provider_from_settings(runtime)
    escalation = getattr(provider, "escalation", None) or provider
    compose = LLMBackgroundWorker(
        escalation, timeout_seconds=runtime.get("background_task_timeout_seconds", 600)
    )
    artifact_dir = Path(os.getenv("CAAL_DATA_DIR", "/app/data")) / "documents"
    # No Telegram delivery from the worker: that channel belongs to a session's
    # user scope, and this process has no session and therefore no scope. A
    # document request still produces its artifact and its spoken summary.
    document_worker = DocumentWorker(compose=compose, deliver=None, artifact_dir=artifact_dir)
    return DurableWorker(
        compose=document_worker, coding=build_coding_delegate(runtime, provider=provider)
    )


def build_destination_resolver() -> Any | None:
    """Resolve a user's approved callback number, server-side, at dispatch time."""
    from . import user_api
    from .security_config import load_multi_user_config

    status = load_multi_user_config()
    if not status.enabled:
        logger.warning(
            "Multi-user identity is not configured: user-bound callbacks cannot be dispatched"
        )
        return None
    store = getattr(user_api.get_runtime(), "store", None)
    if store is None:
        return None

    def _resolve(user_id: str) -> str | None:
        return store.approved_callback_number(user_id)

    return _resolve


def build_dispatcher(*, claimant: str) -> CallbackDispatcher | None:
    """Wire the callback dispatcher to LiveKit, or refuse to pretend it is wired."""
    from livekit import api

    url = os.getenv("LIVEKIT_URL", "")
    key = os.getenv("LIVEKIT_API_KEY", "")
    secret = os.getenv("LIVEKIT_API_SECRET", "")
    if not (url and key and secret):
        logger.warning("LiveKit is not configured: callbacks will stay pending, not fail silently")
        return None
    placer = LiveKitCallbackPlacer(
        livekit=api.LiveKitAPI(url=url, api_key=key, api_secret=secret),
        agent_name=os.getenv("CAAL_AGENT_NAME", "caal"),
    )
    dry_run = _truthy(os.getenv("CAAL_CALLBACK_DISPATCH_DRY_RUN"))
    if dry_run:
        logger.warning("Callback dispatch is in verification mode: requests are built, not placed")
    return CallbackDispatcher(
        placer=placer,
        claimant=claimant,
        resolve_user_destination=build_destination_resolver(),
        allowed_destinations=os.getenv("CAAL_OUTBOUND_ALLOWED_DESTINATIONS", ""),
        dry_run=dry_run,
    )


def build_reminder_dispatcher(*, claimant: str) -> ReminderDispatcher | None:
    """Wire due-reminder delivery to Telegram and to LiveKit, or leave it pending.

    Both channels are wired from the operator configuration only. The Telegram
    chat is the one an administrator bound to a single profile, and the
    dispatcher refuses any reminder whose owner is not that profile; the call
    resolves the owner own approved number at dispatch time, through the same
    resolver the callback dispatcher uses. A channel this process cannot serve
    leaves its reminder pending rather than failed, and never claimed.
    """
    from livekit import api

    settings = settings_module.load_settings()
    token = settings.get("telegram_bot_token") or os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = settings.get("telegram_chat_id") or os.getenv("TELEGRAM_CHAT_ID", "")
    send_telegram = None
    if token and chat_id:

        async def send_telegram(text: str) -> None:  # noqa: F811
            import httpx

            from .telegram_notify import TelegramCallNotifier

            async with httpx.AsyncClient(timeout=15.0) as client:
                await TelegramCallNotifier(
                    token=token, chat_id=chat_id, client=client
                ).notify_text(text)

    placer = None
    url = os.getenv("LIVEKIT_URL", "")
    key = os.getenv("LIVEKIT_API_KEY", "")
    secret = os.getenv("LIVEKIT_API_SECRET", "")
    if url and key and secret:
        placer = LiveKitCallbackPlacer(
            livekit=api.LiveKitAPI(url=url, api_key=key, api_secret=secret),
            agent_name=os.getenv("CAAL_AGENT_NAME", "caal"),
        )
    if send_telegram is None and placer is None:
        logger.warning("No reminder delivery channel is configured; due reminders stay pending")
        return None
    # The same flag that holds back callbacks holds back reminder calls: a
    # deployment in verification mode must not ring anybody at all.
    dry_run = _truthy(os.getenv("CAAL_CALLBACK_DISPATCH_DRY_RUN"))
    if dry_run:
        logger.warning("Reminder calls are in verification mode: requests are built, not placed")
    return ReminderDispatcher(
        claimant=claimant,
        send_telegram=send_telegram,
        placer=placer,
        resolve_user_destination=build_destination_resolver(),
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# The internal HTTP surface
# ---------------------------------------------------------------------------


def _authorize(token: str | None) -> None:
    """Shared-secret check for the internal control plane. Fails closed."""
    secret = os.getenv("CAAL_INTERNAL_AUTH_SECRET", "")
    if not secret or len(secret) < MIN_SECRET_LENGTH:
        raise HTTPException(status_code=503, detail="Worker control plane is not configured")
    if not isinstance(token, str) or not hmac.compare_digest(token, secret):
        raise HTTPException(status_code=401, detail="Unauthorized")


def build_app(supervisor: DurableWorkSupervisor, *, wake: asyncio.Event) -> FastAPI:
    """The internal probe and wake-up surface for one supervisor."""
    app = FastAPI(title="CAAL durable work", version="1.0.0")

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        """Liveness only: the process is up and its loop object exists."""
        return dict(status="ok", service="caal-durable-work")

    @app.get("/status")
    async def status(x_caal_worker_token: str | None = Header(default=None)) -> dict[str, object]:
        _authorize(x_caal_worker_token)
        return supervisor.status()

    @app.post("/internal/wake")
    async def internal_wake(
        x_caal_worker_token: str | None = Header(default=None),
    ) -> dict[str, object]:
        """Tick sooner than the poll would. Correctness never depends on this."""
        _authorize(x_caal_worker_token)
        wake.set()
        return dict(status="ok")

    return app


# ---------------------------------------------------------------------------
# Process
# ---------------------------------------------------------------------------


async def amain() -> int:
    """Run the supervisor and its internal probe surface until told to stop."""
    logging.basicConfig(
        level=os.getenv("CAAL_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    import uvicorn

    runtime = worker_settings()
    worker_id = background_tasks.new_runner_id()
    execute_work = bool(runtime.get("background_tasks_enabled", True))
    if not execute_work:
        logger.warning("Background work is disabled in settings: this worker only dispatches")
    supervisor = DurableWorkSupervisor(
        worker=build_worker(runtime),
        dispatcher=build_dispatcher(claimant=worker_id),
        reminders=build_reminder_dispatcher(claimant=worker_id),
        worker_id=worker_id,
        max_concurrency=runtime["background_task_max_concurrency"],
        poll_seconds=float(os.getenv("CAAL_DURABLE_WORK_POLL_SECONDS", "5")),
        execute_work=execute_work,
    )
    stop = asyncio.Event()
    wake = asyncio.Event()

    loop = asyncio.get_running_loop()
    for signal_name in ("SIGTERM", "SIGINT"):
        signal_number = getattr(signal, signal_name, None)
        if signal_number is not None:
            try:
                loop.add_signal_handler(signal_number, stop.set)
            except (NotImplementedError, RuntimeError):
                pass

    config = uvicorn.Config(
        build_app(supervisor, wake=wake),
        host=os.getenv("CAAL_WORKER_HOST", "0.0.0.0"),
        port=int(os.getenv("CAAL_WORKER_PORT", str(DEFAULT_PORT))),
        log_level="warning",
        log_config=None,
    )
    server = uvicorn.Server(config)
    probe = asyncio.create_task(server.serve())

    # Counts only: enough for an operator to see the state of the queue on
    # boot, with nothing in it that identifies a user or their work.
    logger.info("Durable work service starting: %s", background_tasks.queue_counts())
    orphaned = background_tasks.orphaned_running_count()
    if orphaned:
        logger.warning(
            "%d background task(s) were left running by a pre-lease runner. They are NOT "
            "resumed automatically, because resuming one can end in an outbound call; "
            "adopt them deliberately with the durable-work tooling.",
            orphaned,
        )
    try:
        await supervisor.run(stop=stop, wake=wake)
    finally:
        await supervisor.shutdown()
        server.should_exit = True
        try:
            await asyncio.wait_for(probe, timeout=10)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            probe.cancel()
        logger.info("Durable work service stopped: %s", background_tasks.queue_counts())
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TOKEN_HEADER",
    "DurableWorker",
    "build_app",
    "build_dispatcher",
    "build_reminder_dispatcher",
    "build_worker",
    "main",
    "worker_settings",
]
