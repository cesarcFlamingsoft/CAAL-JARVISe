"""Local recognition of a "continue this on my phone" handoff request.

CAAL handles this itself instead of forwarding it to Hermes, which has no
outbound-call tool and would otherwise answer that it cannot do it. People
rarely phrase it as a command, so the request is inferred from cues in the
final transcript ("I need to take this on the road") rather than matched
against a magic phrase. The inference is deterministic and offline: it reads
the text only, never a model or the network.

Inference decides how to *ask*, never whether to dial. A DIRECT reading asks
for confirmation, a CLARIFY reading asks a question that cannot start a call,
and a call still only happens after an explicit confirmation given in the turn
immediately after that question. The destination is never taken from the
caller: it comes from the server-side allowlist.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, cast

from .handoff_context import ConversationSnapshot, capture_conversation_snapshot

logger = logging.getLogger(__name__)

_PREFIX = r"(?:jarvis[,. ]+)?(?:please |can you |could you |i want you to |i'?d like you to )?"
_PHONE = r"(?:my )(?:phone|cell|cell ?phone|mobile(?: phone)?)"
_SUBJECT_WORDS = r"(?:conversation|chat|call|discussion)"
_SUBJECT = rf"(?:(?:this|our|the)(?: {_SUBJECT_WORDS})?|{_SUBJECT_WORDS})"
_SUFFIX = r"(?: please)?[.!]*"

_HANDOFF_PATTERNS = (
    re.compile(
        rf"{_PREFIX}(?:continue|carry on with|move|switch|transfer|shift) {_SUBJECT} "
        rf"(?:on|onto|over to|to) {_PHONE}{_SUFFIX}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"{_PREFIX}continue (?:on|onto|over to|to) {_PHONE}{_SUFFIX}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"{_PREFIX}(?:call|ring|phone) me (?:on|at) {_PHONE}{_SUFFIX}",
        re.IGNORECASE,
    ),
)

# --- Natural-language inference -------------------------------------------
#
# Four groups of cues, read off the normalized text. A DIRECT reading needs the
# caller to be asking for something (INTENT), to be talking about carrying on
# this conversation (CONTINUATION) and to name either the phone (CHANNEL) or a
# reason to leave the keyboard (MOBILITY). A single cue only earns a question.
# Any BLOCKER wins outright, so explanations, hypotheticals, calls aimed at
# somebody else and broken-phone talk stay ordinary conversation.

_CONTRACTIONS = (
    (r"\bi'm\b", "i am"),
    (r"\bwe're\b", "we are"),
    (r"\byou're\b", "you are"),
    (r"\bi'd\b", "i would"),
    (r"\bi'll\b", "i will"),
    (r"\bi've\b", "i have"),
    (r"\bwe'll\b", "we will"),
    (r"\blet's\b", "let us"),
    (r"\bcan't\b", "can not"),
    (r"\bwon't\b", "will not"),
    (r"\bdon't\b", "do not"),
    (r"\bdoesn't\b", "does not"),
    (r"\bisn't\b", "is not"),
    (r"\bgotta\b", "got to"),
    (r"\bgonna\b", "going to"),
    (r"\bwanna\b", "want to"),
)

_INTENT_CUES = (
    r"\bi (?:need|want|have|had|might|may|should|must|would|will|can|could)\b",
    r"\bi am \w+ing\b",
    r"\bi am about to\b",
    r"\b(?:can|could|would|will|should|shall) (?:you|we)\b",
    r"\bwe (?:can|could|should|need)\b",
    r"\blet us\b",
    # A bare imperative is a request too: "call me...", "move this...".
    r"^(?:jarvis[ ,]+)?(?:please )?"
    r"(?:call|continue|carry|move|switch|transfer|shift|take|pick|keep|resume|finish)\b",
)

_CONTINUATION_CUES = (
    r"\bcontinu(?:e|ing)\b",
    r"\bresume\b",
    r"\bkeep (?:talking|chatting|going|this going|it going)\b",
    r"\bpick (?:this|it|things) (?:up|back up)\b",
    r"\bcarry (?:this|it|on)\b",
    r"\bfinish (?:this|it|our|the)\b",
    r"\btake (?:this|it)\b",
    r"\b(?:move|switch|transfer|shift|bring) (?:this|it|our|the)\b",
    r"\b(?:move|switch|transfer|shift|bring) (?:conversation|chat|call|discussion)\b",
    r"\b(?:this|our|the) (?:conversation|chat|discussion)\b",
)

_CHANNEL_CUES = (
    r"\b(?:on|onto|to|over to) (?:my|the) (?:phone|cell|cell phone|mobile|mobile phone)\b",
    r"\b(?:by|over|via|on) (?:the )?phone\b",
    r"\bphone call\b",
    r"\bvoice call\b",
    r"\bcall me\b",
    r"\bgive me a (?:call|ring)\b",
    r"\b(?:can|could|would|will) you call\b",
)

# Weaker than a channel cue: the phone is mentioned, but not as the place this
# conversation would move to. Enough to ask, never enough to act.
_PHONE_MENTION_CUES = (
    r"\bmy (?:phone|cell|cell phone|mobile|mobile phone)\b",
    r"\bcall\b",
)

_MOBILITY_CUES = (
    r"\bon the road\b",
    r"\bon the go\b",
    r"\bin the car\b",
    r"\bdriv(?:e|ing)\b",
    r"\bcommut(?:e|ing)\b",
    r"\bleav(?:e|ing)\b",
    r"\b(?:head|heading|step|stepping|walk|walking) out\b",
    r"\bout the door\b",
    r"\bon my way\b",
    r"\bget going\b",
    r"\bhave to (?:go|run)\b",
    r"\bgot to (?:go|run)\b",
)

_BLOCKER_CUES = (
    # Explanations and capability questions: the caller wants to know, not to go.
    r"\bhow (?:do|does|did|can|could|would|should) (?:i|we|you|it|that|this|the)\b",
    r"\bwhat (?:happens|do you mean|does that mean|is the|is a|is your)\b",
    r"\bwhy (?:do|does|would|can|is)\b",
    r"\b(?:explain|describe|tell me|show me) (?:how|what|why|about)\b",
    r"\bare you able to\b",
    r"\bis it possible\b",
    r"\bdo you (?:know how|support|have)\b",
    # Hypothetical or instructional mentions of the request itself.
    r"\bif (?:i|you|we) (?:say|said|ask|asked|tell|told|want|wanted|were|could|call|called)\b",
    r"\bwhat would (?:you|happen)\b",
    r"\b(?:suppose|hypothetically|for example|for instance|imagine|pretend)\b",
    r"\bwhen i say\b",
    r"\bthe (?:phrase|command|word|words)\b",
    # Calls aimed at somebody other than the caller, or at a supplied number.
    r"\bcall (?:my|your|his|her|their|our|the|a|an) (?!phone\b|cell\b|mobile\b)\w+",
    r"\bcall (?:him|her|them|us|someone|somebody|anyone|everyone|people|back)\b",
    r"\btake (?:this|the|a) call\b",
    r"\bcall (?:my|the) (?:phone|cell|mobile) (?:company|carrier|provider|service)\b",
    r"\b(?:call|dial|ring|phone) \d",
    r"\b(?:call|dial|ring) (?:me )?(?:on|at) \d",
    # A call that already exists on the calendar is not a handoff.
    r"\bi have (?:a|an|another) (?:phone )?(?:call|meeting)\b",
    r"\b(?:call|meeting) (?:at|with) \b",
    # Device troubleshooting.
    r"\b(?:broken|cracked|dead|died|battery|charger|charging|reboot|restart|reset)\b",
    r"\b(?:phone|cell|mobile) (?:store|shop|repair|repairs|service|provider|carrier)\b",
    r"\b(?:repair|fix) (?:my|the) (?:phone|cell|mobile)\b",
    r"\b(?:is|are|was|were) not working\b",
    r"\bwill not (?:turn|charge|connect|work|start)\b",
    r"\bno (?:signal|service|reception|battery)\b",
    # Refusals: never read a request out of the caller declining one.
    r"\b(?:do|would|can) not (?:call|phone|ring)\b",
    r"\bno need to (?:call|phone)\b",
    r"\brather not\b",
    r"\bwithout (?:calling|a call)\b",
    # Plainly past or scheduled elsewhere in time.
    r"\b(?:yesterday|earlier|last night|last week|this morning|the other day)\b",
    r"\bi (?:was|were) (?!thinking|wondering|hoping|about)\w+ing\b",
    r"\bused to\b",
)

_INTENT_PATTERNS = tuple(re.compile(p) for p in _INTENT_CUES)
_CONTINUATION_PATTERNS = tuple(re.compile(p) for p in _CONTINUATION_CUES)
_CHANNEL_PATTERNS = tuple(re.compile(p) for p in _CHANNEL_CUES)
_PHONE_MENTION_PATTERNS = tuple(re.compile(p) for p in _PHONE_MENTION_CUES)
_MOBILITY_PATTERNS = tuple(re.compile(p) for p in _MOBILITY_CUES)
_BLOCKER_PATTERNS = tuple(re.compile(p) for p in _BLOCKER_CUES)

# A quoted fragment is somebody discussing the words, not saying them.
_QUOTED = re.compile(r"[\"“”]")
_LONG_NUMBER = re.compile(r"\d{3}")

# Confirmations and denials are exact phrases on purpose: anything fuzzier risks
# reading a call authorization into ordinary conversation.
_CONFIRMATIONS = frozenset(
    {
        "yes",
        "yes please",
        "yes do it",
        "yes go ahead",
        "yes call me",
        "yes call me now",
        "yeah",
        "yep",
        "yup",
        "sure",
        "affirmative",
        "confirm",
        "confirmed",
        "do it",
        "go ahead",
        "please do",
    }
)

_DENIALS = frozenset(
    {
        "no",
        "no thanks",
        "no thank you",
        "nope",
        "negative",
        "cancel",
        "cancel that",
        "never mind",
        "nevermind",
        "stop",
        "don't",
        "dont",
        "do not",
        "not now",
        "forget it",
    }
)

ASK_CLARIFICATION_REPLY = (
    "Just so I understand: do you want to keep talking by phone instead of here?"
)
ASK_CONFIRMATION_REPLY = (
    "I can continue this conversation by calling your approved phone. "
    "Do you want me to call you now?"
)
CANCELLED_REPLY = "Understood. I won't call your phone. We can keep going here."
STARTING_REPLY = "Calling your phone now. I'll pick up this conversation there."
NO_DESTINATION_REPLY = (
    "I can't do that. I don't have exactly one approved phone number on file, "
    "so I won't place a call. We can keep going here."
)
NO_CALLBACK_NUMBER_REPLY = (
    "I can't call you because there's no approved callback number on your profile. "
    "An administrator needs to set one in your profile first, so we'll keep going here."
)
FAILED_REPLY = "I wasn't able to start that call, so we'll keep going here."


def _normalize(transcript: str) -> str:
    """Collapse whitespace so spoken punctuation and spacing never matter."""
    return " ".join(transcript.split())


def _strip_answer(transcript: str) -> str:
    """Reduce a short answer to bare words for exact confirmation matching."""
    normalized = _normalize(transcript).lower()
    normalized = re.sub(r"^(?:jarvis|hey jarvis)[,. ]+", "", normalized)
    return " ".join(re.sub(r"[.,!?]", " ", normalized).split())


class HandoffIntent(StrEnum):
    """How confidently a transcript reads as "let's keep talking by phone"."""

    NONE = "none"
    CLARIFY = "clarify"
    DIRECT = "direct"


def _for_inference(transcript: str) -> str:
    """Lower-case, de-contract and de-punctuate so only the wording matters."""
    text = _normalize(transcript).lower().replace("’", "'").replace("‘", "'")
    for pattern, replacement in _CONTRACTIONS:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return " ".join(text.split())


def _any(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _is_blocked(raw: str, text: str) -> bool:
    """Reject readings that mention a call without asking for one."""
    if _QUOTED.search(raw):
        return True
    if _any(_BLOCKER_PATTERNS, text):
        return True
    # A number in the text means a destination is being proposed by the caller,
    # which this feature never honours; refuse to read intent out of it at all.
    return bool(_LONG_NUMBER.search(text)) and (
        _any(_CHANNEL_PATTERNS, text) or _any(_PHONE_MENTION_PATTERNS, text)
    )


def classify_handoff_intent(transcript: str) -> HandoffIntent:
    """Read one final transcript as a phone-handoff intent, conservatively.

    DIRECT means the caller asked to carry this conversation on somewhere the
    keyboard cannot follow, CLARIFY means a single cue points that way and only
    a question can settle it, and NONE means ordinary conversation. Neither
    positive answer authorizes a call by itself.
    """
    normalized = _normalize(transcript)
    if not normalized:
        return HandoffIntent.NONE

    # The original explicit commands stay exactly as literal as they were.
    if any(pattern.fullmatch(normalized) for pattern in _HANDOFF_PATTERNS):
        return HandoffIntent.DIRECT

    text = _for_inference(transcript)
    if _is_blocked(transcript, text) or not _any(_INTENT_PATTERNS, text):
        return HandoffIntent.NONE

    channel = _any(_CHANNEL_PATTERNS, text)
    mobility = _any(_MOBILITY_PATTERNS, text)
    if _any(_CONTINUATION_PATTERNS, text) and (channel or mobility):
        return HandoffIntent.DIRECT
    if channel or mobility or _any(_PHONE_MENTION_PATTERNS, text):
        return HandoffIntent.CLARIFY
    return HandoffIntent.NONE


def handoff_requested(transcript: str) -> bool:
    """Return true only for an unambiguous request to move to the phone."""
    return classify_handoff_intent(transcript) is HandoffIntent.DIRECT


def confirmation_given(transcript: str) -> bool:
    """Return true only for an explicit affirmative answer."""
    return _strip_answer(transcript) in _CONFIRMATIONS


def denial_given(transcript: str) -> bool:
    """Return true only for an explicit refusal."""
    return _strip_answer(transcript) in _DENIALS


def sole_handoff_destination(allowed_destinations: str) -> str | None:
    """Return the only approved destination, or None when the choice is not unique.

    A handoff must never guess between numbers, so anything other than exactly
    one distinct approved destination declines instead of dialing.
    """
    unique = {item.strip() for item in allowed_destinations.split(",") if item.strip()}
    if len(unique) != 1:
        return None
    return next(iter(unique))


class HandoffAction(StrEnum):
    """What the agent should do about the transcript it just heard."""

    NONE = "none"
    ASK_CLARIFICATION = "ask_clarification"
    ASK_CONFIRMATION = "ask_confirmation"
    START_HANDOFF = "start_handoff"
    CANCEL = "cancel"


class _Stage(StrEnum):
    """Where the caller is in the ask-then-confirm sequence."""

    IDLE = "idle"
    CLARIFYING = "clarifying"
    AWAITING_CONFIRMATION = "awaiting_confirmation"


@dataclass(frozen=True)
class HandoffDecision:
    """A handoff verdict for one final transcript.

    ``consumed`` means CAAL answered the turn itself, so it must not be sent on
    to the LLM.
    """

    action: HandoffAction
    reply: str | None = None
    consumed: bool = False


class HandoffIntentMachine:
    """Idle, clarifying an implied request, or awaiting the one confirmation.

    An inferred request never shortens the path to a call. A CLARIFY reading
    only asks a question, and answering it "yes" merely earns the ordinary
    confirmation prompt, so a call still needs an explicit confirmation given
    in the turn right after that prompt.
    """

    def __init__(self) -> None:
        self._stage = _Stage.IDLE

    @property
    def awaiting_confirmation(self) -> bool:
        return self._stage is _Stage.AWAITING_CONFIRMATION

    def reset(self) -> None:
        """Drop any pending question, e.g. when the session restarts."""
        self._stage = _Stage.IDLE

    def observe(self, transcript: str) -> HandoffDecision:
        """Advance the machine with one final user transcript."""
        intent = classify_handoff_intent(transcript)

        if intent is HandoffIntent.DIRECT:
            return self._ask_confirmation()

        if self._stage is _Stage.IDLE:
            if intent is HandoffIntent.CLARIFY:
                self._stage = _Stage.CLARIFYING
                return HandoffDecision(
                    HandoffAction.ASK_CLARIFICATION, ASK_CLARIFICATION_REPLY, consumed=True
                )
            return HandoffDecision(HandoffAction.NONE)

        if not _normalize(transcript):
            # The call authorization must be a fresh immediate turn. A final
            # blank turn means no confirmation arrived, so fail closed.
            if self._stage is _Stage.AWAITING_CONFIRMATION:
                self._stage = _Stage.IDLE
                return HandoffDecision(HandoffAction.CANCEL)
            # A clarification cannot dial; silence may simply be an STT gap.
            return HandoffDecision(HandoffAction.NONE)

        if self._stage is _Stage.CLARIFYING:
            return self._answer_clarification(transcript, intent)
        return self._answer_confirmation(transcript, intent)

    def _ask_confirmation(self) -> HandoffDecision:
        self._stage = _Stage.AWAITING_CONFIRMATION
        return HandoffDecision(
            HandoffAction.ASK_CONFIRMATION, ASK_CONFIRMATION_REPLY, consumed=True
        )

    def _answer_clarification(self, transcript: str, intent: HandoffIntent) -> HandoffDecision:
        """A "yes" here confirms the reading, not the call."""
        if confirmation_given(transcript):
            return self._ask_confirmation()

        if denial_given(transcript):
            self._stage = _Stage.IDLE
            return HandoffDecision(HandoffAction.CANCEL, CANCELLED_REPLY, consumed=True)

        if intent is HandoffIntent.CLARIFY:
            # Still only a hint. Asking again is better than sending a phone
            # request to an LLM that has no way to place a call.
            return HandoffDecision(
                HandoffAction.ASK_CLARIFICATION, ASK_CLARIFICATION_REPLY, consumed=True
            )

        # The caller moved on, so the reading was wrong: hand the turn back.
        self._stage = _Stage.IDLE
        return HandoffDecision(HandoffAction.NONE)

    def _answer_confirmation(self, transcript: str, intent: HandoffIntent) -> HandoffDecision:
        if confirmation_given(transcript):
            self._stage = _Stage.IDLE
            return HandoffDecision(HandoffAction.START_HANDOFF, STARTING_REPLY, consumed=True)

        if denial_given(transcript):
            self._stage = _Stage.IDLE
            return HandoffDecision(HandoffAction.CANCEL, CANCELLED_REPLY, consumed=True)

        if intent is HandoffIntent.CLARIFY:
            # Not an answer, but still about the phone: ask once more instead of
            # letting it fall through to an assistant that cannot dial.
            return self._ask_confirmation()

        # Anything else is not a confirmation, so no call happens, but the caller
        # clearly meant to say something else: let that reach the LLM normally.
        self._stage = _Stage.IDLE
        return HandoffDecision(HandoffAction.CANCEL)


class _Speaker(Protocol):
    async def say(self, text: str) -> Any: ...


StartCall = Callable[..., Awaitable[Any]]

# The handoff's own replies never travel to the phone: they are mechanics, not
# conversation, and would read as noise in the continuation context. The
# conversation ledger recorder skips them for the same reason.
HANDOFF_CONTROL_REPLIES = (
    ASK_CLARIFICATION_REPLY,
    ASK_CONFIRMATION_REPLY,
    CANCELLED_REPLY,
    STARTING_REPLY,
    NO_DESTINATION_REPLY,
    NO_CALLBACK_NUMBER_REPLY,
    FAILED_REPLY,
)
_CONTROL_REPLIES = HANDOFF_CONTROL_REPLIES

# Resolve the session user's own approved callback number at confirmation
# time; None when they have none. Never reads the transcript.
DestinationResolver = Callable[[], "str | None"]


class PhoneHandoffController:
    """Drive the intent machine, then dial only through the allowlisted seam.

    ``start_call`` is the trusted server-side entry point (in production,
    ``OutboundCallCoordinator.start``), which re-checks the destination against
    the allowlist before any room, dispatch, or SIP dial happens.

    When the session has a ledger-backed ``conversation_id``, the confirmed
    call is started as ``start_call(destination, context=None,
    conversation_id=...)`` and the phone leg hydrates from the private ledger
    once a human answers. Without one, it falls back to
    ``start_call(destination, context=snapshot)`` where ``snapshot`` is the
    bounded, redacted recent history captured at the moment of confirmation,
    or ``None`` when there is nothing safe to carry.
    """

    def __init__(
        self,
        *,
        start_call: StartCall,
        allowed_destinations: str,
        conversation_id: str | None = None,
        user_id: str | None = None,
        destination_resolver: DestinationResolver | None = None,
    ) -> None:
        """With ``destination_resolver`` the call goes to the session user's own
        approved number, resolved fresh when they confirm; the static allowlist
        is then ignored entirely. ``user_id`` travels with the call so the
        worker can re-resolve the same profile before dialing.
        """
        self._start_call = start_call
        self._allowed_destinations = allowed_destinations
        self._conversation_id = conversation_id
        self._user_id = user_id
        self._destination_resolver = destination_resolver
        self._machine = HandoffIntentMachine()
        # Transcripts consumed by the handoff itself (the request, the "yes").
        self._control_transcripts: list[str] = []

    @property
    def awaiting_confirmation(self) -> bool:
        return self._machine.awaiting_confirmation

    @property
    def conversation_id(self) -> str | None:
        """Opaque ledger id of the session's live conversation, if any."""
        return self._conversation_id

    def reset(self) -> None:
        self._machine.reset()
        self._control_transcripts.clear()

    async def handle_final_transcript(self, transcript: str, session: _Speaker) -> bool:
        """Handle one final transcript; return true when CAAL consumed the turn."""
        decision = self._machine.observe(transcript)

        if decision.consumed:
            self._control_transcripts.append(transcript)
        elif not self._machine.awaiting_confirmation:
            self._control_transcripts.clear()

        if decision.action is HandoffAction.START_HANDOFF:
            await self._start_confirmed_handoff(session)
            return True

        if decision.reply is not None:
            await self._say(session, decision.reply)
        return decision.consumed

    @staticmethod
    async def _say(session: _Speaker, text: str) -> None:
        """Speak a control reply on a best-effort basis.

        The decision has already been taken by the time a reply is spoken, so a
        TTS or session failure must not raise into the turn pipeline or undo a
        dial that already happened.
        """
        try:
            # The explicit confirmation is a safety boundary. Ensure its audio
            # is not immediately pre-empted by the just-completed user turn.
            try:
                await cast(Any, session).say(text, allow_interruptions=False)
            except TypeError:
                # Test doubles and legacy speaker shims may omit this keyword.
                await session.say(text)
        except Exception:
            logger.warning("Phone handoff could not speak its reply", exc_info=True)

    def _capture_context(self, session: object) -> ConversationSnapshot | None:
        """Snapshot recent visible history; any failure means no context, not no call."""
        history = getattr(session, "history", None)
        if history is None:
            return None
        try:
            return capture_conversation_snapshot(
                history.items,
                exclude_assistant_texts=_CONTROL_REPLIES,
                exclude_user_texts=tuple(self._control_transcripts),
            )
        except Exception:
            logger.warning("Phone handoff could not capture conversation context", exc_info=True)
            return None

    def _resolve_destination(self) -> str | None:
        """The one number this session may dial, or None. Never from the transcript."""
        if self._destination_resolver is None:
            return sole_handoff_destination(self._allowed_destinations)
        try:
            return self._destination_resolver()
        except Exception:
            logger.warning("Phone handoff could not resolve the user's callback number")
            return None

    async def _start_confirmed_handoff(self, session: _Speaker) -> None:
        """Dial the single approved destination, or decline without calling."""
        destination = self._resolve_destination()
        if destination is None:
            if self._destination_resolver is not None:
                logger.info("Phone handoff declined: user has no approved callback number")
                await self._say(session, NO_CALLBACK_NUMBER_REPLY)
            else:
                logger.warning("Phone handoff declined: no single approved destination")
                await self._say(session, NO_DESTINATION_REPLY)
            return

        # With a live ledger the phone leg hydrates from it server-side, so no
        # snapshot is captured or carried; only the opaque id travels.
        if self._conversation_id is not None:
            context = None
            call_kwargs: dict[str, Any] = {
                "context": None,
                "conversation_id": self._conversation_id,
            }
            logger.info("Phone handoff carrying ledger continuation")
        else:
            context = self._capture_context(session)
            call_kwargs = {"context": context}
            logger.info(
                "Phone handoff carrying context turns=%d",
                len(context.turns) if context is not None else 0,
            )
        if self._user_id is not None:
            call_kwargs["user_id"] = self._user_id
        self._control_transcripts.clear()

        # Dispatch first, announce second, so a failed dial never leaves the
        # caller waiting for a call that was already contradicted.
        try:
            await self._start_call(destination, **call_kwargs)
        except Exception:
            logger.exception("Phone handoff could not start an outbound call")
            await self._say(session, FAILED_REPLY)
            return
        await self._say(session, STARTING_REPLY)
