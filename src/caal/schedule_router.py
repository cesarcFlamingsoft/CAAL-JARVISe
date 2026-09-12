"""The turn that changes a scheduled item, taken without a model writing it.

The mutation itself is :mod:`caal.tools.scheduled_items`, and the ordinary
route to it is the native schema the local model already holds. This is the
narrower second route underneath that, for the turn the model reads as
something else entirely -- which is exactly what happened to "change that alarm
to a reminder": the schedule was searched, an alarm was set, and the reply
sounded like a conversion that never took place.

It is deliberately the narrowest thing that can work:

* it exists only for a signed-in owner;
* it acts only when the offline gate in :mod:`caal.tools.schedule_semantics`
  has bounded the turn, and only while that owner actually has a pending item;
* the reading is one bounded, classification-only question to the *local*
  model. Escalation is not reachable from here: the caller passes the local
  provider in;
* it changes nothing itself. The write is the same owner-scoped
  ``scheduled.change`` the model would have called, with the owner bound from
  the verified session rather than from the words, and the item re-checked
  against the same server-composed list the reading was given;
* anything ambiguous asks, out loud, and writes nothing.

Everything it says is the message of the handler. Nothing here logs, caches or
forwards the utterance, a title, a row id, a number or a chat, and the packet
it publishes is the shared constant.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from caal.llm.scheduled_reply import spoken_outcome
from caal.tools import scheduled_items
from caal.tools.schedule_semantics import SemanticScheduleReader
from caal.user_scope import UserScope

logger = logging.getLogger(__name__)

__all__ = ["CHANGE_TOOL", "ScheduledChangeHandler"]

CHANGE_TOOL = scheduled_items.CHANGE_TOOL

_OK = "ok"


class ScheduledChangeHandler:
    """Change one scheduled item of the owner of this session, or leave the turn alone."""

    def __init__(
        self,
        *,
        scope: UserScope,
        announce: Callable[[], Awaitable[None]] | None = None,
        now: Callable[[], int] | None = None,
        semantic: SemanticScheduleReader | None = None,
    ) -> None:
        self._scope = scope
        # The constant scheduled-change packet, published only after the write.
        self._announce = announce
        self._now = now
        self._semantic = semantic

    @property
    def semantic(self) -> SemanticScheduleReader | None:
        """The local reading layer, if one is configured."""
        return self._semantic

    @property
    def enabled(self) -> bool:
        """Only a verified owner has items of their own to change."""
        return self._scope.user_id is not None and self._semantic is not None

    def _moment(self) -> int:
        if self._now is not None:
            return int(self._now())
        import time

        return int(time.time())

    async def handle(self, text: object, session: Any) -> bool:
        """Take the turn if it is plainly a change; report whether it was taken.

        ``False`` leaves the turn exactly where it was going, which is the
        answer in every uncertain case: an unrecognised request, nothing
        pending, an item that already came due, a reading that could not be
        made.
        """
        if not self.enabled:
            return False
        semantic = self._semantic
        assert semantic is not None  # narrowed by `enabled`
        if not semantic.may_read(text):
            return False
        user_id = self._scope.user_id
        now = self._moment()
        try:
            items = scheduled_items.candidates(user_id, now)
        except Exception as exc:  # noqa: BLE001 - a store that cannot be read changes nothing
            logger.info("Could not read the schedule of this owner (%s)", type(exc).__name__)
            return False
        if not items:
            # Nothing of theirs to change, so no model is asked anything at all.
            return False
        change = await semantic.read(text, scheduled_items.summarize_at(items, now))
        if change is None:
            return False

        if change.index is None:
            if len(items) > 1 and not scheduled_items.points_at_the_newest(text):
                # More than one it could be and nothing saying which. Asking is
                # the only safe answer; cancelling the wrong one is not.
                logger.info("Asked which scheduled item was meant rather than guessing")
                await self._speak(session, scheduled_items.AMBIGUOUS)
                return True
            item = items[0]
        else:
            item = items[change.index - 1]

        result = scheduled_items.apply_change(
            item,
            change.action,
            user_id=user_id,
            now=now,
            when=change.when,
            title=change.title,
            target_kind=change.target_kind,
        )
        if not isinstance(result, dict):
            logger.error("The scheduled-change tool answered off contract")
            return False
        message = spoken_outcome(CHANGE_TOOL, result) or ""
        if result.get("status") == _OK:
            logger.info("Changed one scheduled item at the request of its owner")
            await self._publish()
        else:
            logger.info("A scheduled change was refused by its own tool")
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
            logger.warning("Could not speak the scheduled change (%s)", type(exc).__name__)
