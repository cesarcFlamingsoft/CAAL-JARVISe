"""Conservative inference of "we can wrap this up" and "call me when you finish".

The literal commands in ``caal.call_termination`` ("hang up", "end the call",
"hang up and call me back when you're done") stay the only phrases that act
without a question. This module reads the natural-language forms underneath
them, deterministically and offline, and decides how much they earn:

* a strong, first-person, imperative exit ends the call outright;
* a hedged or partial exit only earns "Would you like me to end the call?";
* an exit paired with a completion-conditioned "call me" earns the callback
  question, and a callback without a clear exit or condition earns the same
  question, so a callback is never armed without an explicit "yes";
* questions, hypotheticals, reported speech, third-party callees, dictated
  numbers, refusals and incidental mentions stay ordinary conversation.

A pending question is only ever settled by an explicit answer given in the
turn right after it, inside a short window; anything else drops it.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from .handoff_intent import confirmation_given, denial_given

_MAX_CHARS = 200

_CONTRACTIONS = (
    (r"\bi'm\b", "i am"),
    (r"\bwe're\b", "we are"),
    (r"\byou're\b", "you are"),
    (r"\bit's\b", "it is"),
    (r"\bthat's\b", "that is"),
    (r"\bthat'll\b", "that will"),
    (r"\bthat'd\b", "that would"),
    (r"\bwhat's\b", "what is"),
    (r"\bi'd\b", "i would"),
    (r"\bi'll\b", "i will"),
    (r"\bi've\b", "i have"),
    (r"\bwe'll\b", "we will"),
    (r"\bwe've\b", "we have"),
    (r"\byou've\b", "you have"),
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

# --- blockers: any hit means the turn is ordinary conversation -------------------

_BLOCKER_CUES = (
    # Questions by form, when the transcript lost its question mark.
    r"^(?:jarvis[ ,]+)?(?:okay |ok |so |and )?"
    r"(?:can|could|would|will|should|shall|do|does|did|is|are|was|were) "
    r"(?:you|we|i|that|it|this)\b",
    # Explanations and capability questions.
    r"\bhow (?:do|does|did|can|could|would|should) (?:i|we|you|it|that|this|the)\b",
    r"\bwhat (?:happens|happened|does|do|did|is|are|would|will|about)\b",
    r"\bwhy (?:do|does|did|would|can|is|are)\b",
    r"\b(?:explain|describe) (?:what|how|why)\b",
    r"\b(?:tell|show) me (?:about|what|how|why)\b",
    r"\bare you able to\b",
    r"\bis it possible\b",
    # Hypothetical or instructional mentions of the request itself.
    r"\bif (?:i|you|we|it|that|this)\b",
    r"\bwhen i say\b",
    r"\b(?:suppose|hypothetically|for example|for instance|imagine|pretend)\b",
    r"\bthe (?:phrase|command|word|words)\b",
    r"\bwhat .* means?\b",
    # Reported speech and third parties.
    r"\b(?:said|says|saying|told|tells|telling|asked|asking)\b",
    r"\b(?:he|she|they) (?:needs|need|has|have|had|wants|want|is|are)\b",
    # A call aimed at somebody else, or at a dictated number.
    r"\bcall (?:my|your|his|her|their|our|the|a|an) (?!phone\b|cell\b|mobile\b)\w+",
    r"\bcall (?:him|her|them|us|someone|somebody|anyone|everyone|people)\b",
    r"\b(?:call|dial|ring|phone) (?:me )?(?:back )?(?:on|at) \d",
    r"\d{3}",
    # A call that already exists, or trouble with the line.
    r"\bi have (?:a|an|another) (?:phone )?(?:call|meeting)\b",
    r"\b(?:call|meeting) (?:at|with) \b",
    r"\bcall (?:keeps|is|was) (?:dropping|breaking|cutting)\b",
    # Refusals and negations: never read a request out of the caller declining one.
    r"\bnot (?:done|finished|ready|through|yet|now)\b",
    r"\bdo not (?:hang|end|disconnect|call|ring|phone|let|go|wrap|leave)\b",
    r"\bno need to (?:call|phone|ring)\b",
    r"\bwithout (?:calling|a call)\b",
    # More conversation in the same breath.
    r"\bbut\b",
    r"\bbefore (?:we|you|i)\b",
    r"\bone more\b",
    r"\banother (?:question|thing)\b",
    r"\bfirst\b",
    # Going somewhere, not leaving the call.
    r"\bgo (?:to|into|back to|over to|down to|out to) "
    r"(?:the|a|an|my|our|his|her|their|work|school|bed)\b",
    # Plainly past.
    r"\bi was\b",
    r"\b(?:yesterday|earlier|last night|last week|this morning|the other day)\b",
    r"\bused to\b",
)

# --- exit cues -------------------------------------------------------------------

_LEAVE = r"(?:go|run|leave|get going|head out|head off|get off (?:the |this )?(?:phone|call|line))"
_NOT_A_PLACE = r"(?! (?:to|into|through|over|for|with)\b)"

# Strong: first-person, present, imperative, complete. Enough to end the call.
_STRONG_EXIT_CUES = (
    r"\bi am (?:good|fine|all set|done|finished) for (?:now|today|tonight)\b",
    r"\b(?:you can|you may|go ahead and) let me go\b",
    r"\bi (?:will|am going to|should) let you go\b",
    r"\b(?:let us|we can|we could|we should|go ahead and|time to) wrap (?:this |it |things )?up\b",
    r"\bi (?:need|have|got|have got|must|am going) to " + _LEAVE + _NOT_A_PLACE,
    r"\bi must " + _LEAVE + _NOT_A_PLACE,
    r"\bthat is (?:all|everything|it) for (?:now|today|tonight)\b",
    r"\bthat is (?:all|everything) i (?:needed|need|wanted)\b",
    r"\bthat will be all\b",
    r"\b(?:bye|goodbye|bye bye)\b",
    r"\btalk to you (?:later|soon|tomorrow)\b",
    r"\b(?:talk|catch you|see you|speak) (?:later|soon)\b",
    r"\bwe are (?:done|finished|all done|all set) (?:here|for now|for today)\b",
    r"\bwe are all (?:done|set)\b",
    r"\bgo ahead and (?:hang up|end (?:the|this) call|disconnect)\b",
)

# Weak: exit-ish, but hedged, partial, or a fragment. Only earns a question.
_WEAK_EXIT_CUES = (
    r"\bi am (?:good|fine|all set|done|finished)\b(?! (?:at|with|on|in|to)\b)",
    r"\bwe are (?:done|finished)\b",
    r"\bi think (?:that is|we are|i am) (?:done|it|all|finished)\b",
    r"\bi should (?:probably |really )?" + _LEAVE + _NOT_A_PLACE,
    r"\blet me go\b",
    r"\bwrap(?:ping)? (?:this |it |things )?up\b",
    r"\bcall it a (?:day|night)\b",
    r"\bthat is (?:it|all|everything)\b",
    r"\bhang up\b",
    r"\bend (?:the|this) call\b",
    r"\bdisconnect\b",
    r"\bget off (?:the|this) (?:phone|call|line)\b",
)

# Hedges downgrade a strong exit to a question.
_HEDGE_CUES = (
    r"\bi think\b",
    r"\bi guess\b",
    r"\bi suppose\b",
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bprobably\b",
    r"\bmight\b",
    r"\bkind of\b",
    r"\bsort of\b",
)

# --- callback cues ---------------------------------------------------------------

_CALLBACK_VERB_CUES = (
    r"\b(?:call|ring|phone) me(?: back| again)?\b",
    r"\bgive me a (?:call|ring|buzz)(?: back)?\b",
)

_CALLBACK_CONDITION_CUES = (
    r"\b(?:when|once|as soon as|after|whenever) (?:you|it|that|this|the|everything)\b"
    r".*\b(?:done|finished|finish|complete|completed|ready|results?|through|wrapped up)\b",
    r"\bwith (?:the |your )?results\b",
)

_STRONG_EXIT_PATTERNS = tuple(re.compile(p) for p in _STRONG_EXIT_CUES)
_WEAK_EXIT_PATTERNS = tuple(re.compile(p) for p in _WEAK_EXIT_CUES)
_HEDGE_PATTERNS = tuple(re.compile(p) for p in _HEDGE_CUES)
_BLOCKER_PATTERNS = tuple(re.compile(p) for p in _BLOCKER_CUES)
_CALLBACK_VERB_PATTERNS = tuple(re.compile(p) for p in _CALLBACK_VERB_CUES)
_CALLBACK_CONDITION_PATTERNS = tuple(re.compile(p) for p in _CALLBACK_CONDITION_CUES)

# A quoted fragment is somebody discussing the words, not saying them.
_QUOTED = re.compile(r"[\"“”]")

ASK_END_CALL_REPLY = "It sounds like you're wrapping up. Would you like me to end the call?"
ASK_CALLBACK_REPLY = (
    "It sounds like you'd like me to hang up and call you back once that task is done. "
    "Should I do that?"
)
NO_TASK_ASK_END_CALL_REPLY = (
    "There's no background task running right now, so I can't call you back. "
    "Would you like me to end the call anyway?"
)
NO_TASK_REPLY = (
    "There's no background task running right now, so there's nothing to call you "
    "back about. I'll stay on the line."
)
STAY_ON_LINE_REPLY = "Understood. I'll stay on the line."
# The answer to a declined callback offer (see ``offer_callback``): the work
# JARVIS just scheduled carries on, and the line stays open.
CALLBACK_OFFER_DECLINED_REPLY = (
    "No problem. I'll keep working on it in the background and let you know when it's done."
)

# These replies are mechanics, not conversation: the ledger and every context
# snapshot skip them, exactly as they skip the handoff and background replies.
END_CALL_CONTROL_REPLIES = (
    ASK_END_CALL_REPLY,
    ASK_CALLBACK_REPLY,
    NO_TASK_ASK_END_CALL_REPLY,
    NO_TASK_REPLY,
    STAY_ON_LINE_REPLY,
    CALLBACK_OFFER_DECLINED_REPLY,
)


class EndCallIntent(StrEnum):
    """How confidently a transcript reads as "I'm leaving this call"."""

    NONE = "none"
    CLARIFY = "clarify"
    DIRECT = "direct"


class CallbackIntent(StrEnum):
    """How confidently a transcript reads as "call me back when you finish"."""

    NONE = "none"
    CLARIFY = "clarify"
    DIRECT = "direct"


@dataclass(frozen=True)
class EndCallReading:
    """Both readings of one transcript; equal readings compare equal."""

    end_call: EndCallIntent = EndCallIntent.NONE
    callback: CallbackIntent = CallbackIntent.NONE


_NOTHING = EndCallReading()


def _normalize(transcript: str) -> str:
    return " ".join(transcript.split())


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
    if "?" in raw or _QUOTED.search(raw):
        return True
    return _any(_BLOCKER_PATTERNS, text)


def classify_end_call_intent(transcript: str) -> EndCallReading:
    """Read one final transcript, conservatively, as an exit and/or a callback.

    Bounded and deterministic: the text is whitespace-normalized, capped in
    length, and read against fixed cue lists only. A blocker of any kind wins
    outright. A callback is DIRECT only when it is completion-conditioned and
    rides on a DIRECT exit; every other "call me" is a CLARIFY.
    """
    if not isinstance(transcript, str):
        return _NOTHING
    normalized = _normalize(transcript)
    if not normalized or len(normalized) > _MAX_CHARS:
        return _NOTHING

    text = _for_inference(transcript)
    if _is_blocked(transcript, text):
        return _NOTHING

    if _any(_STRONG_EXIT_PATTERNS, text):
        end_call = EndCallIntent.CLARIFY if _any(_HEDGE_PATTERNS, text) else EndCallIntent.DIRECT
    elif _any(_WEAK_EXIT_PATTERNS, text):
        end_call = EndCallIntent.CLARIFY
    else:
        end_call = EndCallIntent.NONE

    callback = CallbackIntent.NONE
    if _any(_CALLBACK_VERB_PATTERNS, text):
        conditioned = _any(_CALLBACK_CONDITION_PATTERNS, text)
        if conditioned and end_call is EndCallIntent.DIRECT:
            callback = CallbackIntent.DIRECT
        else:
            callback = CallbackIntent.CLARIFY

    return EndCallReading(end_call=end_call, callback=callback)


# --- the state machine -----------------------------------------------------------


class EndCallAction(StrEnum):
    """What the agent should do about the transcript it just heard."""

    NONE = "none"
    ASK_END_CALL = "ask_end_call"
    END_CALL = "end_call"
    ASK_CALLBACK = "ask_callback"
    ARM_CALLBACK = "arm_callback"
    DECLINE_CALLBACK = "decline_callback"
    CANCEL = "cancel"


class _Pending(StrEnum):
    """Which question is waiting for its answer."""

    END_CALL = "end_call"
    CALLBACK = "callback"
    # JARVIS asked, unprompted, whether to call back about work it just
    # scheduled. Same answer rules; a decline keeps the work going.
    OFFERED_CALLBACK = "offered_callback"


@dataclass(frozen=True)
class EndCallDecision:
    """A verdict for one final transcript.

    ``consumed`` means CAAL answered the turn itself, so it must not be sent on
    to the LLM. A CANCEL that is not consumed hands the turn back untouched.
    """

    action: EndCallAction
    reply: str | None = None
    consumed: bool = False


class EndCallIntentMachine:
    """Idle, or holding exactly one question for the very next turn.

    A DIRECT exit ends the call without a question. Every callback, and every
    uncertain exit, only asks; a fresh explicit "yes" in the following turn
    (inside ``CONFIRMATION_TIMEOUT_SECONDS``) is the only thing that acts. A
    denial keeps the line, and anything else drops the question.
    """

    CONFIRMATION_TIMEOUT_SECONDS = 30.0

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._pending: _Pending | None = None
        self._asked_at = 0.0

    @property
    def awaiting_confirmation(self) -> bool:
        self._expire()
        return self._pending is not None

    def reset(self) -> None:
        """Drop any pending question, e.g. when the session closes."""
        self._pending = None

    def offer_callback(self) -> None:
        """Hold the callback question JARVIS itself just asked about scheduled work.

        The caller has already heard the question; this only makes the very
        next turn its answer, under the same rules as every other question:
        a fresh explicit "yes" arms, "no" declines, anything else drops it.
        """
        self._pending = _Pending.OFFERED_CALLBACK
        self._asked_at = self._clock()

    def _expire(self) -> None:
        if self._pending is not None and (
            self._clock() - self._asked_at > self.CONFIRMATION_TIMEOUT_SECONDS
        ):
            self._pending = None

    def _ask(self, pending: _Pending, action: EndCallAction, reply: str) -> EndCallDecision:
        self._pending = pending
        self._asked_at = self._clock()
        return EndCallDecision(action, reply, consumed=True)

    def observe(self, transcript: str, *, callback_available: bool) -> EndCallDecision:
        """Advance the machine with one final user transcript.

        ``callback_available`` says whether something could actually be called
        back about; without it a callback request is explained, never asked.
        """
        self._expire()
        reading = classify_end_call_intent(transcript)

        # A fresh intent always replaces a pending question rather than answering it.
        if reading.callback is not CallbackIntent.NONE:
            if callback_available:
                return self._ask(_Pending.CALLBACK, EndCallAction.ASK_CALLBACK, ASK_CALLBACK_REPLY)
            if reading.end_call is not EndCallIntent.NONE:
                return self._ask(
                    _Pending.END_CALL, EndCallAction.ASK_END_CALL, NO_TASK_ASK_END_CALL_REPLY
                )
            self._pending = None
            return EndCallDecision(EndCallAction.DECLINE_CALLBACK, NO_TASK_REPLY, consumed=True)
        if reading.end_call is EndCallIntent.DIRECT:
            self._pending = None
            return EndCallDecision(EndCallAction.END_CALL, consumed=True)
        if reading.end_call is EndCallIntent.CLARIFY:
            return self._ask(_Pending.END_CALL, EndCallAction.ASK_END_CALL, ASK_END_CALL_REPLY)

        pending, self._pending = self._pending, None
        if pending is None:
            return EndCallDecision(EndCallAction.NONE)

        # The answer must be a fresh, explicit, whole turn; fail closed otherwise.
        if not _normalize(transcript):
            return EndCallDecision(EndCallAction.CANCEL)
        if confirmation_given(transcript):
            if pending is _Pending.END_CALL:
                return EndCallDecision(EndCallAction.END_CALL, consumed=True)
            return EndCallDecision(EndCallAction.ARM_CALLBACK, consumed=True)
        if denial_given(transcript):
            if pending is _Pending.OFFERED_CALLBACK:
                return EndCallDecision(
                    EndCallAction.CANCEL, CALLBACK_OFFER_DECLINED_REPLY, consumed=True
                )
            return EndCallDecision(EndCallAction.CANCEL, STAY_ON_LINE_REPLY, consumed=True)
        return EndCallDecision(EndCallAction.CANCEL)
