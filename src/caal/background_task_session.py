"""Bridge between one voice session and the durable background task queue.

The bridge answers the local background-task commands itself (schedule,
status, cancel) with fixed replies so Hermes never sees them, runs each task
through an injected worker with a bounded private snapshot of the recent
conversation, and announces the outcome exactly once: spoken into the
session that asked while it is still open, otherwise through the Telegram
fallback. The work outlives the session; only a process restart interrupts
it. Nothing spoken, sent, or logged ever includes the raw request text or
the task id.

From an active outbound phone call the user may also say "hang up and call
me back when you're done". That arms a persisted one-time callback on the
work this conversation scheduled, hangs up, and, once the task settles, places
one policy-checked outbound call whose greeting delivers the outcome.
Cancelling the task withdraws the callback.

Background work is owned by the logical conversation, not the transport: the
web session that schedules a task and the phone leg that later continues the
same ledger conversation share one owner key, so either may inspect, cancel,
or arm a callback for it. A session with no ledger conversation is owned by
its room name. The owner key stored in the queue is a one-way derivation of
the ledger id, never the id itself, so a queue row can neither reveal nor
be used to claim a conversation.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

from caal.background_tasks import (
    DEFAULT_MAX_CONCURRENCY,
    FAILED,
    INTERRUPTED,
    QUEUED,
    RUNNING,
    SUCCEEDED,
    BackgroundTask,
    BackgroundTaskRunner,
    QueueFullError,
    arm_callback,
    cancel,
    claim_callback_target,
    claim_next_notification,
    claim_notification,
    enqueue,
    get_task,
    list_tasks,
    pending_notifications,
    redact_secrets,
)
from caal.conversation_ledger import is_valid_conversation_id
from caal.end_call_intent import END_CALL_CONTROL_REPLIES
from caal.handoff_context import capture_conversation_snapshot
from caal.handoff_intent import HANDOFF_CONTROL_REPLIES
from caal.telegram_notify import MAX_TEXT_CHARS
from caal.user_scope import require_user_id
from caal.work_router import Route, SemanticWorkRouter

logger = logging.getLogger(__name__)

BACKGROUND_ACK_REPLY = (
    "Understood. I'll work on that in the background and let you know when it's done."
)
BACKGROUND_BUSY_REPLY = (
    "I can't take on more background work right now. Let's finish what's already running first."
)
BACKGROUND_STATUS_IDLE_REPLY = "I'm not working on anything in the background right now."
BACKGROUND_STATUS_WORKING_REPLY = (
    "I'm still working on that in the background. I'll let you know as soon as it's done."
)
BACKGROUND_CANCELLED_REPLY = "Understood. I've stopped that background task."
BACKGROUND_NOTHING_TO_CANCEL_REPLY = "There's no background task running right now."
CALLBACK_ARMED_REPLY = (
    "Understood. I'll hang up now and call you back as soon as it's done. Goodbye."
)
CALLBACK_NOTHING_RUNNING_REPLY = (
    "There's no background task running right now, so I'll stay on the line."
)
# Work JARVIS recognised as long-running on its own (a PDF, a report, some
# research) and scheduled without being told "in the background".
LONG_WORK_ACK_REPLY = (
    "Understood. I'll get started on that now and let you know as soon as it's done."
)
# The same, on a phone call: the question is answered by the next turn only.
LONG_WORK_OFFER_CALLBACK_REPLY = (
    "Understood. I'll get started on that now. "
    "This will take a while. Would you like me to call you when it's done?"
)
BACKGROUND_CONTROL_REPLIES = (
    BACKGROUND_ACK_REPLY,
    BACKGROUND_BUSY_REPLY,
    BACKGROUND_STATUS_IDLE_REPLY,
    BACKGROUND_STATUS_WORKING_REPLY,
    BACKGROUND_CANCELLED_REPLY,
    BACKGROUND_NOTHING_TO_CANCEL_REPLY,
    CALLBACK_ARMED_REPLY,
    CALLBACK_NOTHING_RUNNING_REPLY,
    LONG_WORK_ACK_REPLY,
    LONG_WORK_OFFER_CALLBACK_REPLY,
)


@dataclass(frozen=True)
class BackgroundTurnOutcome:
    """What the bridge did with one user turn.

    ``consumed`` means the turn was answered locally and must not reach the
    LLM. ``scheduled`` means new work was queued. ``callback_offered`` means
    the reply ended with the callback question, so the caller's next turn is
    its answer; it is only ever true when work was actually scheduled.
    """

    consumed: bool
    scheduled: bool = False
    callback_offered: bool = False


_SUCCESS_PREFIX = "Here's what I found from the task you asked me to work on in the background. "
_EMPTY_RESULT_MESSAGE = (
    "I finished the task you asked me to work on in the background, "
    "but there wasn't anything to report."
)
_FAILED_MESSAGE = (
    "I couldn't finish the task you asked me to work on in the background. "
    "Let me know if you'd like me to try again."
)
_INTERRUPTED_MESSAGE = (
    "The task I was working on in the background was interrupted before it finished. "
    "Let me know if you'd like me to try again."
)
_FALLBACK_PREFIX = "JARVIS background task: "
_CALLBACK_UNANSWERED_PREFIX = (
    "I tried to call you back about the background task, but the call wasn't answered. "
)
_CALLBACK_EMPTY_OUTCOME = (
    "I'm calling back about the task you asked me to work on in the background, "
    "but I no longer have its outcome to report."
)

MAX_SPOKEN_RESULT_CHARS = 600
MAX_FALLBACK_CHARS = MAX_TEXT_CHARS
MAX_TASK_CONTEXT_CHARS = 4_500
DEFAULT_TIMEOUT_SECONDS = 600.0
# Finished work nobody announced for this long belongs to a session that is
# gone; anything younger may still be picked up by its own session's loop.
STALE_NOTIFICATION_SECONDS = 60

_TRUNCATION_MARK = "…"

BACKGROUND_SYSTEM_PROMPT = (
    "You are JARVIS carrying out a task the user asked you to handle in the background "
    "while the conversation moved on. Complete the task fully using whatever tools you "
    "have, then reply with a concise, spoken-style summary of the outcome in plain "
    "sentences: no headings, lists, links, or markdown. Lead with the answer. If you "
    "could not complete the task, say briefly what you could and could not do."
)

Execute = Callable[[str, str], Awaitable[str]]
Fallback = Callable[[str], Awaitable[None]]
# Place the callback: (destination, task_id). Runs the allowlist check itself.
DialCallback = Callable[[str, str], Awaitable[None]]
# Place a user-bound callback: (user_id, task_id). Resolves the user's own
# approved number from their profile at dial time and re-checks it.
DialUserCallback = Callable[[str, str], Awaitable[None]]

MAX_ROOM_KEY_CHARS = 128
_OWNER_KEY_PREFIX = "conv-"
# Domain separation: the same ledger id hashed for any other purpose yields a
# different digest, so an owner key can never be replayed as anything else.
_OWNER_KEY_DOMAIN = b"caal.background_tasks.owner:"


def owner_key_for(conversation_id: str | None, *, room_key: str) -> str:
    """Return the queue owner key for a session.

    A session inside a ledger conversation is keyed by a one-way digest of
    the conversation id: stable across every leg of that conversation and
    distinct for every other one, but useless for reading or claiming the
    ledger. A session without one is keyed by its room name. Both inputs are
    validated so a malformed id or an empty room can never produce a key that
    silently matches nothing or everything.
    """
    if not isinstance(room_key, str) or not room_key.strip() or len(room_key) > MAX_ROOM_KEY_CHARS:
        raise ValueError("Background task owner needs a non-empty room key")
    if conversation_id is None:
        return room_key.strip()
    if not is_valid_conversation_id(conversation_id):
        raise ValueError("Background task owner needs a well-formed conversation id")
    digest = hashlib.sha256(_OWNER_KEY_DOMAIN + conversation_id.encode("ascii")).hexdigest()
    return f"{_OWNER_KEY_PREFIX}{digest}"


def _bound(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - len(_TRUNCATION_MARK)] + _TRUNCATION_MARK


def _result_text(task: BackgroundTask, limit: int) -> str:
    return _bound(redact_secrets(" ".join((task.result or "").split())), limit)


def spoken_notification(task: BackgroundTask) -> str:
    """Concise, bounded announcement for the session. Empty means say nothing.

    Failures and interruptions are reported without their internals: an error
    string can carry upstream hostnames or tokens that were never meant to be
    read aloud.
    """
    if task.status == SUCCEEDED:
        result = _result_text(task, MAX_SPOKEN_RESULT_CHARS)
        return f"{_SUCCESS_PREFIX}{result}" if result else _EMPTY_RESULT_MESSAGE
    if task.status == FAILED:
        return _FAILED_MESSAGE
    if task.status == INTERRUPTED:
        return _INTERRUPTED_MESSAGE
    # Cancellation was acknowledged when the user asked for it.
    return ""


def fallback_notification(task: BackgroundTask) -> str:
    """Bounded text for the out-of-session channel. Empty means send nothing."""
    if task.status == SUCCEEDED:
        budget = MAX_FALLBACK_CHARS - len(_FALLBACK_PREFIX) - len(_SUCCESS_PREFIX)
        result = _result_text(task, budget)
        body = f"{_SUCCESS_PREFIX}{result}" if result else _EMPTY_RESULT_MESSAGE
    else:
        body = spoken_notification(task)
    if not body:
        return ""
    return _bound(f"{_FALLBACK_PREFIX}{body}", MAX_FALLBACK_CHARS)


def callback_outcome_message(task_id: str) -> str:
    """What the callback call says once a human has answered.

    The outcome is read from the store by opaque id; an unknown or cancelled
    task yields a short apology rather than silence, so an answered call is
    never a dead line.
    """
    task = get_task(task_id)
    message = spoken_notification(task) if task is not None else ""
    return message or _CALLBACK_EMPTY_OUTCOME


def callback_unanswered_notification(task_id: str) -> str:
    """Bounded fallback text when the callback reached no human. Empty means send nothing."""
    task = get_task(task_id)
    if task is None:
        return ""
    body = fallback_notification(task)
    if not body:
        return ""
    return _bound(f"{_CALLBACK_UNANSWERED_PREFIX}{body}", MAX_FALLBACK_CHARS)


def capture_task_context(session: Any) -> str:
    """Bounded, redacted recent conversation framed as private context.

    Reads the LiveKit chat history only. System prompts, tool traffic, and the
    background and handoff control replies are dropped. Any failure yields no
    context rather than no task.
    """
    history = getattr(session, "history", None)
    items = getattr(history, "items", None)
    if items is None:
        return ""
    try:
        snapshot = capture_conversation_snapshot(
            items,
            exclude_assistant_texts=BACKGROUND_CONTROL_REPLIES
            + HANDOFF_CONTROL_REPLIES
            + END_CALL_CONTROL_REPLIES,
        )
    except Exception:
        logger.warning("Could not capture background task context", exc_info=True)
        return ""
    if snapshot is None:
        return ""
    framing = [
        "Conversation context (private, for your reference only).",
        "The user asked for this task during the conversation below. Use it only to "
        "understand what they mean; do not quote it back or repeat earlier details.",
        "",
        "[Recent conversation]",
    ]
    turns = [
        f"{'User' if turn.role == 'user' else 'Assistant'}: {turn.text}" for turn in snapshot.turns
    ]
    closing = "[End of recent conversation]"

    def render(kept: list[str]) -> str:
        return redact_secrets("\n".join([*framing, *kept, closing]))

    text = render(turns)
    while turns and len(text) > MAX_TASK_CONTEXT_CHARS:
        turns.pop(0)
        text = render(turns)
    return text if turns else ""


class LLMBackgroundWorker:
    """Run one background request through the configured LLM provider.

    Hermes owns its own tool loop, so the request is sent as a plain
    completion: a private system frame plus the user's request. Any provider
    exposing ``chat(messages)`` works.
    """

    def __init__(self, provider: Any, *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS):
        self._provider = provider
        self._timeout = float(timeout_seconds)

    async def __call__(self, request: str, context: str) -> str:
        return await self._run(request, context)

    async def run_task(self, task: BackgroundTask, context: str) -> str:
        """Run with an opaque marker that lets Hermes return real artifacts safely."""
        return await self._run(task.request, context, task_id=task.task_id)

    async def _run(self, request: str, context: str, *, task_id: str | None = None) -> str:
        system = BACKGROUND_SYSTEM_PROMPT
        if context:
            system = f"{system}\n\n{context}"
        if task_id:
            system = f"{system}\n\n[CAAL_ARTIFACT_TASK:{task_id}]"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": request},
        ]
        response = await asyncio.wait_for(self._provider.chat(messages), self._timeout)
        content = getattr(response, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("background task produced an empty answer")
        return content.strip()


class BackgroundTaskBridge:
    """Per-session front end over the shared queue, runner, and notifications."""

    def __init__(
        self,
        *,
        execute: Execute,
        session_key: str,
        conversation_id: str | None = None,
        fallback: Fallback | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        dial_callback: DialCallback | None = None,
        user_id: str | None = None,
        dial_user_callback: DialUserCallback | None = None,
        work_router: SemanticWorkRouter | None = None,
    ) -> None:
        """``session_key`` is the room; ``conversation_id`` the ledger conversation, if any.

        The queue never sees either raw ledger id: work is filed under the
        derived owner key (see :func:`owner_key_for`). ``user_id`` is the
        verified user this session acts for; every queue read and write is
        additionally bound to that scope (``None`` = legacy, unscoped work).

        ``work_router`` reads each turn. The default is the offline router,
        which is exactly the deterministic behaviour that predates it; a caller
        that hands over one backed by the configured LLM gets the semantic
        reading on top, and the same deterministic controls underneath.
        """
        if not session_key:
            raise ValueError("background task bridge needs a session key")
        self._execute = execute
        self._room_key = session_key
        self._owner_key = owner_key_for(conversation_id, room_key=session_key)
        self._user_id = None if user_id is None else require_user_id(user_id)
        self._fallback = fallback
        self._dial_callback = dial_callback
        self._dial_user_callback = dial_user_callback
        self._router = work_router or SemanticWorkRouter()
        self._contexts: dict[str, str] = {}
        self._context_source: Callable[[], str] | None = None
        self._runner = BackgroundTaskRunner(
            self._run_task, max_concurrency=max_concurrency, on_settled=self._on_settled
        )
        self._closed = False

    @property
    def has_fallback(self) -> bool:
        return self._fallback is not None

    @property
    def owner_key(self) -> str:
        """The opaque key this session's work is filed under in the queue."""
        return self._owner_key

    @property
    def session_key(self) -> str:
        """Alias of :attr:`owner_key`, kept for callers that predate conversation ownership."""
        return self._owner_key

    @property
    def user_id(self) -> str | None:
        """The opaque user this session's work is scoped to, if any."""
        return self._user_id

    def bind_conversation(self, conversation_id: str | None) -> None:
        """Adopt the background work of a ledger conversation this leg now continues.

        An outbound phone leg is dispatched with only the opaque conversation
        id and joins that conversation once a human answers; from then on the
        task the web session scheduled is its own to report, cancel, or arm a
        callback for. Existing tasks are never moved and the raw id is never
        stored; only the derived owner key changes. ``None`` returns the leg
        to its room-owned work.
        """
        self._owner_key = owner_key_for(conversation_id, room_key=self._room_key)

    @property
    def can_arm_callback(self) -> bool:
        """Whether this session has open work a callback could be armed for."""
        if self._closed:
            return False
        try:
            return bool(self._own_open_tasks())
        except Exception:
            logger.warning("Could not check for open background tasks", exc_info=True)
            return False

    def has_deliverable_file_request(self, text: object) -> bool:
        """Whether the configured worker can make and deliver this file request."""
        checker = getattr(self._execute, "is_deliverable_request", None)
        if not callable(checker):
            return False
        try:
            return bool(checker(text))
        except Exception:
            logger.warning("Could not classify deliverable background work", exc_info=True)
            return False

    async def start(
        self, *, context_source: Callable[[], str] | None = None, recover: bool = False
    ) -> None:
        self._context_source = context_source
        self._closed = False
        await self._runner.start(recover=recover)

    # -- local turn handling -------------------------------------------------

    async def handle_turn(self, text: str, session: Any) -> bool:
        """Answer a background command locally; return whether the turn was consumed."""
        return (await self.process_turn(text, session)).consumed

    async def process_turn(
        self,
        text: str,
        session: Any,
        *,
        offer_callback: bool = False,
        auto_callback: Callable[[], Awaitable[None]] | None = None,
    ) -> BackgroundTurnOutcome:
        """Answer a background command locally and report what was done.

        The router reads the turn: the explicit commands (cancel, status, "in
        the background") are decided offline and keep their fixed replies, and
        a turn read as work is scheduled. Its acknowledgement ends with the
        callback question only when ``offer_callback`` is set, which the caller
        does only on a phone leg that can actually arm one, and only once the
        work is genuinely queued. The offer is reported back so the caller can
        hold the question for the next turn.

        A routing failure can only ever leave the turn as conversation, so an
        unreachable router costs a semantic reading, never a hung turn.
        """
        if not isinstance(text, str) or not text.strip() or self._closed:
            return BackgroundTurnOutcome(consumed=False)
        try:
            decision = await self._router.route(text)
        except Exception:
            logger.warning("Work router failed; treating the turn as conversation", exc_info=True)
            return BackgroundTurnOutcome(consumed=False)
        if decision.route is Route.CANCEL:
            await self._cancel_own(session)
            return BackgroundTurnOutcome(consumed=True)
        if decision.route is Route.STATUS:
            await self._report_status(session)
            return BackgroundTurnOutcome(consumed=True)
        if decision.route is Route.BACKGROUND:
            scheduled = await self._schedule(text, session, ack=BACKGROUND_ACK_REPLY)
            if scheduled and auto_callback is not None:
                await self._run_auto_callback(auto_callback)
            return BackgroundTurnOutcome(consumed=True, scheduled=scheduled)
        if decision.route is Route.WORK:
            ack = LONG_WORK_OFFER_CALLBACK_REPLY if offer_callback else LONG_WORK_ACK_REPLY
            scheduled = await self._schedule(text, session, ack=ack)
            if scheduled and auto_callback is not None:
                await self._run_auto_callback(auto_callback)
            # A work turn is always answered out loud: the acknowledgement when
            # it was queued, the busy reply when it could not be. The callback
            # is only offered for work that actually exists.
            return BackgroundTurnOutcome(
                consumed=True,
                scheduled=scheduled,
                callback_offered=scheduled and offer_callback and auto_callback is None,
            )
        return BackgroundTurnOutcome(consumed=False)

    @staticmethod
    async def _run_auto_callback(callback: Callable[[], Awaitable[None]]) -> None:
        """Arm and leave a phone call only after the durable task has been queued."""
        try:
            await callback()
        except Exception:
            # Scheduling remains successful and its ordinary in-session result
            # path is still available; never turn a callback transport failure
            # into a lost task.
            logger.exception("Automatic background-task callback arming failed")

    async def _schedule(self, text: str, session: Any, *, ack: str) -> bool:
        """Queue the request and speak ``ack``; report whether it was queued."""
        context = ""
        if self._context_source is not None:
            try:
                context = self._context_source() or ""
            except Exception:
                logger.warning("Background task context source failed", exc_info=True)
        try:
            task = enqueue(text, session_key=self._owner_key, user_id=self._user_id)
        except QueueFullError:
            logger.info("Background task declined: queue full")
            await self._say(session, BACKGROUND_BUSY_REPLY)
            return False
        except Exception:
            logger.exception("Background task could not be scheduled")
            await self._say(session, BACKGROUND_BUSY_REPLY)
            return False
        self._contexts[task.task_id] = context
        self._runner.poke()
        logger.info("Background task scheduled (%d context chars)", len(context))
        await self._say(session, ack)
        return True

    async def _report_status(self, session: Any) -> None:
        # A finished result is the best possible status report.
        if await self.deliver_pending(session):
            return
        if self._own_open_tasks():
            await self._say(session, BACKGROUND_STATUS_WORKING_REPLY)
        else:
            await self._say(session, BACKGROUND_STATUS_IDLE_REPLY)

    async def _cancel_own(self, session: Any) -> None:
        cancelled = 0
        for task in self._own_open_tasks():
            if await self._runner.cancel(task.task_id) or cancel(task.task_id):
                cancelled += 1
            # The acknowledgement below settles it; it must never be announced later.
            claim_notification(task.task_id, self._owner_key)
        if cancelled:
            logger.info("Cancelled %d background task(s)", cancelled)
            await self._say(session, BACKGROUND_CANCELLED_REPLY)
        else:
            await self._say(session, BACKGROUND_NOTHING_TO_CANCEL_REPLY)

    def _own_open_tasks(self) -> list[BackgroundTask]:
        return list_tasks(
            session_key=self._owner_key, statuses=[QUEUED, RUNNING], user_id=self._user_id
        )

    # -- callback ------------------------------------------------------------

    async def arm_callback(
        self, destination: str | None, session: Any, *, user_id: str | None = None
    ) -> bool:
        """Authorize one callback per open task of this session; report whether any was armed.

        Only the work this session scheduled can be armed, and only while it is
        still open. With ``user_id`` the callback is bound to that user (who
        must be this session's own user) and the number is resolved from their
        profile at dial time; ``destination`` is the legacy allowlisted number.
        With nothing running the call stays up and says so, so the caller is
        never hung up on for nothing.
        """
        if self._closed:
            return False
        if user_id is not None:
            if user_id != self._user_id:
                logger.warning("Refused to arm a callback for a different user")
                await self._say(session, CALLBACK_NOTHING_RUNNING_REPLY)
                return False
            destination = None
        elif not destination:
            return False
        armed = 0
        for task in self._own_open_tasks():
            if arm_callback(
                task.task_id, destination, session_key=self._owner_key, user_id=self._user_id
            ):
                armed += 1
        if not armed:
            await self._say(session, CALLBACK_NOTHING_RUNNING_REPLY)
            return False
        logger.info("Armed callback for %d background task(s)", armed)
        await self._say(session, CALLBACK_ARMED_REPLY)
        return True

    async def _place_callback(self, task_id: str) -> bool:
        """Dial the armed callback for a settled task, exactly once.

        The authorization is consumed first, then the outcome's notification is
        claimed so nobody else announces what the call is about to deliver. A
        dial that cannot happen hands the outcome to the fallback instead.
        """
        target = claim_callback_target(task_id, self._owner_key)
        if target is None:
            return False
        task = claim_notification(task_id, self._owner_key)
        if task is None:
            # Already announced elsewhere; nothing left to call about.
            return True
        if target.user_id is not None:
            dialer = self._dial_user_callback
            argument: str | None = target.user_id
        else:
            dialer = self._dial_callback
            argument = target.destination
        if dialer is None or argument is None:
            logger.warning("No dialer for a background task callback; using fallback")
            await self._send_fallback(task)
            return True
        try:
            await dialer(argument, task_id)
        except Exception:
            logger.warning("Background task callback dial failed", exc_info=False)
            await self._send_fallback(task)
            return True
        logger.info("Background task callback dispatched")
        return True

    # -- worker --------------------------------------------------------------

    async def _run_task(self, task: BackgroundTask) -> str:
        context = self._contexts.pop(task.task_id, "")
        task_runner = getattr(self._execute, "run_task", None)
        if callable(task_runner):
            runner = cast(Callable[[BackgroundTask, str], Awaitable[str]], task_runner)
            return await runner(task, context)
        return await self._execute(task.request, context)

    async def _on_settled(self, task_id: str) -> None:
        """A task of ours finished after the session ended: hand it to the fallback.

        An armed callback outranks every other channel: it is consumed and
        dialed here whether or not the session is still open. Otherwise, while
        the session is open its delivery loop speaks outcomes, so nothing is
        claimed here. Without a fallback the outcome stays unannounced for the
        room to hear when it reconnects.
        """
        if await self._place_callback(task_id):
            return
        if not self._closed or self._fallback is None:
            return
        task = claim_notification(task_id, self._owner_key)
        if task is None or task.session_key != self._owner_key or task.user_id != self._user_id:
            return
        await self._send_fallback(task)

    # -- notifications -------------------------------------------------------

    async def deliver_pending(self, session: Any) -> int:
        """Speak every unannounced outcome of this session's tasks, once each."""
        delivered = 0
        while True:
            task = claim_next_notification(
                self._owner_key, self._owner_key, user_id=self._user_id
            )
            if task is None:
                return delivered
            message = spoken_notification(task)
            if not message:
                continue
            try:
                await session.say(message)
            except Exception:
                logger.warning("Could not speak background task outcome", exc_info=True)
                await self._send_fallback(task)
                continue
            delivered += 1

    async def flush_stale_to_fallback(
        self, *, min_age_seconds: float = STALE_NOTIFICATION_SECONDS
    ) -> int:
        """Send other sessions' long-unannounced outcomes through the fallback.

        Only outcomes in this session's own user scope are ever flushed: the
        fallback channel is shared, so another user's result must never reach
        it. Legacy (unscoped) sessions flush only legacy work.
        """
        if self._fallback is None:
            return 0
        cutoff = time.time() - min_age_seconds
        sent = 0
        for candidate in pending_notifications(user_id=self._user_id):
            if candidate.session_key == self._owner_key:
                continue
            if (candidate.finished_at or 0) > cutoff:
                continue
            task = claim_notification(candidate.task_id, self._owner_key)
            if task is not None and await self._send_fallback(task):
                sent += 1
        return sent

    async def _send_fallback(self, task: BackgroundTask) -> bool:
        if self._fallback is None:
            logger.warning("No fallback channel for a background task outcome (%s)", task.status)
            return False
        if task.user_id != self._user_id:
            # The fallback channel is shared across everyone this deployment
            # talks to. A task owned by another user (or by a verified user when
            # this session is anonymous or legacy) must never be announced on it,
            # even when this bridge settled it off the shared queue and its own
            # user-bound callback could not be dialed. Every legitimate caller
            # already passes a task in this bridge's own scope; this is the last
            # guard against the callback-failure path spilling a private outcome.
            logger.warning("Refused to route a cross-user task outcome to the shared fallback")
            return False
        message = fallback_notification(task)
        if not message:
            return False
        try:
            await self._fallback(message)
        except Exception:
            logger.warning(
                "Fallback delivery failed for a background task outcome (%s)", task.status
            )
            return False
        return True

    # -- shutdown ------------------------------------------------------------

    async def close(self) -> int:
        """The session is over; its background work is not.

        Returns immediately: queued and running tasks keep going in this
        process, and each outcome is handed to the fallback channel exactly
        once as it lands. Outcomes that already landed go now. Without a
        fallback they stay unannounced so a later session of the same room
        can still speak them. Only a process restart interrupts the work.
        """
        self._closed = True
        if self._fallback is None:
            return 0
        sent = 0
        while True:
            task = claim_next_notification(
                self._owner_key, self._owner_key, user_id=self._user_id
            )
            if task is None:
                return sent
            if await self._send_fallback(task):
                sent += 1

    async def abandon(self) -> None:
        """Process teardown only: interrupt whatever is still running.

        Interrupted work is recorded as such, so it can still be reported by
        the fallback flush of a later session.
        """
        await self._runner.stop()
        self._contexts.clear()

    @staticmethod
    async def _say(session: Any, text: str) -> None:
        """Speak a control reply on a best-effort basis; the decision already stands."""
        try:
            await session.say(text)
        except Exception:
            logger.warning("Background task bridge could not speak its reply", exc_info=True)


__all__ = [
    "BACKGROUND_ACK_REPLY",
    "BACKGROUND_BUSY_REPLY",
    "BACKGROUND_CANCELLED_REPLY",
    "BACKGROUND_CONTROL_REPLIES",
    "BACKGROUND_NOTHING_TO_CANCEL_REPLY",
    "BACKGROUND_STATUS_IDLE_REPLY",
    "BACKGROUND_STATUS_WORKING_REPLY",
    "BACKGROUND_SYSTEM_PROMPT",
    "CALLBACK_ARMED_REPLY",
    "CALLBACK_NOTHING_RUNNING_REPLY",
    "DEFAULT_TIMEOUT_SECONDS",
    "LONG_WORK_ACK_REPLY",
    "LONG_WORK_OFFER_CALLBACK_REPLY",
    "MAX_FALLBACK_CHARS",
    "MAX_ROOM_KEY_CHARS",
    "MAX_SPOKEN_RESULT_CHARS",
    "MAX_TASK_CONTEXT_CHARS",
    "STALE_NOTIFICATION_SECONDS",
    "BackgroundTaskBridge",
    "BackgroundTurnOutcome",
    "DialCallback",
    "DialUserCallback",
    "LLMBackgroundWorker",
    "callback_outcome_message",
    "callback_unanswered_notification",
    "capture_task_context",
    "fallback_notification",
    "owner_key_for",
    "spoken_notification",
]
