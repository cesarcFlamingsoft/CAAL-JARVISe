"""The one turn that answers the delivery question, answered without a model.

CAAL asks "when it comes due, do you want me to say it here, send it to your
Telegram, call you, or do nothing about it?". Cesar answers "call me". That
reply is a whole turn, and in practice the model behind the turn reads it as
something else -- a phone request, background work, a calendar entry -- so the
reminder that asked the question is left exactly as it was.

This route takes that one turn. It is deliberately the narrowest thing that
can work:

* it exists only for a signed-in owner;
* it acts only while that owner has a reminder whose delivery question was
  actually asked and is still open (:func:`reminder_delivery.awaiting_reminder`
  -- never merely because a reminder exists);
* it acts only on a reply that is plainly that answer and nothing else
  (:func:`caal.tools.delivery_answer.read_delivery_answer`);
* it changes nothing itself. The mutation is the same owner-scoped
  ``reminders.set_delivery`` the model would have called, with the owner bound
  from the verified session rather than from the words.

Everything it says is the handler own message, with one exception: while
outbound calls are held back for verification, a chosen call is described as
saved and verified instead of as a call that is going to ring, because the
call will not ring and saying that it will would be a lie.

Nothing here logs, caches or forwards the utterance, the title, the row id, a
number or a chat, and the packet it publishes is the shared constant.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Awaitable, Callable

from caal.llm.scheduled_reply import spoken_outcome
from caal.tools import reminder_delivery, reminders_tools
from caal.tools.delivery_answer import read_delivery_answer
from caal.tools.delivery_semantics import SemanticDeliveryReader
from caal.user_scope import UserScope

logger = logging.getLogger(__name__)

__all__ = [
    "CALL_VERIFICATION_NOTE",
    "DELIVERY_TOOL",
    "DeliveryAnswerHandler",
    "calls_are_held_back",
]

#: The tool the answer runs through. Named here so the direct reply and the
#: dashboard packet describe the same write the model path describes.
DELIVERY_TOOL = "reminders.set_delivery"

#: What is said instead of promising a call, while this deployment is building
#: call requests without placing them. It claims exactly what happened -- the
#: choice is stored and the request is exercised -- and nothing more.
CALL_VERIFICATION_NOTE = (
    "Your call is saved on that reminder and the request is verified, but outbound calls "
    "are in verification mode here, so it will not actually ring until that is switched off."
)

_CALL_PROMISE = "call you"
_SENTENCE = re.compile(r"[^.!?]+[.!?]?")
_OK = "ok"


def calls_are_held_back() -> bool:
    """Whether this deployment builds reminder calls without placing them.

    The same operator flag the durable worker reads, read the same way, so the
    spoken promise and the dialer can never disagree about what will happen.
    """
    return (os.getenv("CAAL_CALLBACK_DISPATCH_DRY_RUN") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _truthful(message: str, live: tuple[str, ...]) -> str:
    """The spoken result, with a held-back call described as what it is.

    Only the sentence that promises the call is replaced. Whatever else the
    handler said -- a refusal, what the other channels will do -- is kept, and
    the other channels are re-stated from the ledger rather than dropped.
    """
    if reminder_delivery.CALL not in live or not calls_are_held_back():
        return message
    kept = [
        sentence.strip()
        for sentence in _SENTENCE.findall(message)
        if sentence.strip() and _CALL_PROMISE not in sentence.lower()
    ]
    others = tuple(channel for channel in live if channel != reminder_delivery.CALL)
    if others:
        kept.append(reminders_tools.promise(others))
    kept.append(CALL_VERIFICATION_NOTE)
    return " ".join(kept)


class DeliveryAnswerHandler:
    """Answer the open delivery question of this session own owner, once."""

    def __init__(
        self,
        *,
        scope: UserScope,
        announce: Callable[[], Awaitable[None]] | None = None,
        now: Callable[[], int] | None = None,
        semantic: SemanticDeliveryReader | None = None,
    ) -> None:
        self._scope = scope
        # The constant scheduled-change packet, published only after the write.
        self._announce = announce
        self._now = now
        # The second, narrower layer: the local model reads the answers people
        # give in their own words, behind the offline whitelist and behind the
        # open question. Without one this route is the whitelist alone.
        self._semantic = semantic

    @property
    def semantic(self) -> SemanticDeliveryReader | None:
        """The local reading layer behind the whitelist, if one is configured."""
        return self._semantic

    @property
    def enabled(self) -> bool:
        """Only a verified owner has an open question that belongs to them."""
        return self._scope.user_id is not None

    def _moment(self) -> int:
        if self._now is not None:
            return int(self._now())
        import time

        return int(time.time())

    async def handle(self, text: object, session: Any) -> bool:
        """Take the turn if it is plainly that answer; report whether it was taken.

        ``False`` leaves the turn exactly where it was going, which is the
        answer in every uncertain case: an unrecognised reply, no open
        question, a reminder already past, somebody else question.
        """
        if not self.enabled:
            return False
        user_id = self._scope.user_id
        chosen = read_delivery_answer(text)
        if chosen is None:
            # Not one of the exact forms. It may still plainly be the answer,
            # said in this person own words, and the local model is asked to
            # read it -- but only after the offline gate has bounded the turn
            # and only while there is a real question of this owner own open.
            if self._semantic is None or not self._semantic.may_read(text):
                return False
            now = self._moment()
            if reminder_delivery.awaiting_reminder(user_id, now) is None:
                return False
            chosen = await self._semantic.read(text)
            if chosen is None:
                return False
        else:
            now = self._moment()
            if reminder_delivery.awaiting_reminder(user_id, now) is None:
                # A delivery word with no open question behind it is ordinary
                # conversation. It must never move a reminder.
                logger.info(
                    "A delivery-shaped reply arrived with no question open; leaving it alone"
                )
                return False

        result = reminders_tools.set_delivery(
            delivery=list(chosen), user_id=user_id, now=now
        )
        if not isinstance(result, dict):
            logger.error("The delivery tool answered off contract")
            return False
        status = result.get("status")
        data = result.get("data") if isinstance(result.get("data"), dict) else dict()
        live = tuple(
            channel
            for channel in reminder_delivery.CHANNELS
            if channel in (data.get("delivery") or ())
        )
        message = spoken_outcome(DELIVERY_TOOL, result) or ""
        if status == _OK:
            message = _truthful(message, live)
            logger.info("Answered an open delivery question on %d channel(s)", len(live))
            await self._publish()
        else:
            logger.info("An answered delivery question was refused by its own tool")
        if message:
            await self._speak(session, message)
        return True

    async def _publish(self) -> None:
        """One constant packet, after the write and never before."""
        if self._announce is None:
            return
        try:
            await self._announce()
        except Exception as exc:  # noqa: BLE001 - the write already happened
            logger.info(
                "Could not tell this room its scheduled items changed (%s)", type(exc).__name__
            )

    @staticmethod
    async def _speak(session: Any, message: str) -> None:
        try:
            await session.say(message)
        except Exception as exc:  # noqa: BLE001 - the turn is claimed either way
            logger.warning("Could not speak the delivery result (%s)", type(exc).__name__)
