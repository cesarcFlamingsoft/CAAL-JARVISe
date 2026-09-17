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
from .language_policy import strip_accents
from .reply_localization import speech_reply

logger = logging.getLogger(__name__)

_PREFIX = (
    r"(?:(?:friday|jarvis)[,. ]+)?"
    r"(?:please |can you |could you |i want you to |i'?d like you to )?"
)
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
    r"^(?:(?:friday|jarvis)[ ,]+)?(?:please )?"
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

# --- Guards for a semantically read request --------------------------------
#
# A Spanish request is read by the local model, not matched here: a list of
# Spanish phrasings would be exactly the magic-phrase list this module already
# refused to be in English. What lives here is the opposite -- the deterministic
# *subtraction* applied to whatever the model said. These lexicons can only ever
# turn a positive reading into NONE, so a model that says "direct" about a
# hypothetical, somebody else's call, a dictated number, or a turn that never
# mentions a phone at all changes nothing. Read on accent-folded text, because
# the speech server returns "llámame" and "llamame" interchangeably.

_ES_CHANNEL_CUES = (
    r"\b(?:telefono|celular|movil|movil|whatsapp)\b",
    r"\bllama(?:me|rme|das?|ndo)?\b",
    r"\bllame(?:s|me)?\b",
    r"\bmarca(?:me|rme)\b",
    r"\bde ?volver la llamada\b",
)

_ES_MOBILITY_CUES = (
    r"\bme voy\b",
    r"\bya (?:salgo|me voy|voy saliendo)\b",
    r"\bsal(?:go|iendo|ir)\b",
    r"\ben (?:el|mi) (?:coche|carro|auto)\b",
    r"\b(?:manejando|conduciendo|maneja|conduzco)\b",
    r"\ben (?:la )?carretera\b",
    r"\bde camino\b",
    r"\ben camino\b",
    r"\bcamino a\b",
    r"\btengo que (?:irme|salir|correr)\b",
)

_ES_BLOCKER_CUES = (
    # Hypotheticals, explanations and capability questions.
    r"\bsi (?:te |le |me )?(?:digo|dijera|dijese|pido|pidiera|pregunto|dices)\b",
    r"\b(?:por ejemplo|supongamos|supon|hipoteticamente|imagina|imaginate|finge)\b",
    r"\b(?:como|que) (?:funciona|haces|harias|pasa|pasaria)\b",
    r"\bque (?:quiere decir|significa)\b",
    r"\b(?:puedes|podrias|sabes) (?:explicar|decirme como)\b",
    r"\bes posible\b",
    r"\bla (?:frase|palabra|orden|instruccion)\b",
    r"\bcuando (?:digo|te digo)\b",
    # A call aimed at somebody who is not the speaker.
    r"\bllama(?:r|me|nos)?\s+a\s+(?!mi (?:telefono|celular|movil)\b)",
    r"\bllama(?:le|lo|la|les|los|las)\b",
    r"\bque (?:me )?llame\s+a\s+",
    # A destination dictated by the speaker, or a call already on the calendar.
    r"\bnumero\b",
    r"\btengo (?:una|otra) (?:llamada|reunion|junta)\b",
    r"\b(?:llamada|reunion|junta) (?:a las|con)\b",
    # Device trouble.
    r"\b(?:roto|rota|descompuesto|bateria|cargador|se apago|no funciona|sin senal"
    r"|sin servicio|reiniciar)\b",
    # Refusals.
    r"\bno (?:me )?(?:llames|llame|hace falta|es necesario)\b",
    r"\bmejor no\b",
    r"\bsin llamar\b",
    # Plainly in the past.
    r"\b(?:ayer|anoche|antier|la semana pasada|esta manana)\b",
)

_INTENT_PATTERNS = tuple(re.compile(p) for p in _INTENT_CUES)
_CONTINUATION_PATTERNS = tuple(re.compile(p) for p in _CONTINUATION_CUES)
_CHANNEL_PATTERNS = tuple(re.compile(p) for p in _CHANNEL_CUES)
_PHONE_MENTION_PATTERNS = tuple(re.compile(p) for p in _PHONE_MENTION_CUES)
_MOBILITY_PATTERNS = tuple(re.compile(p) for p in _MOBILITY_CUES)
_BLOCKER_PATTERNS = tuple(re.compile(p) for p in _BLOCKER_CUES)
_ES_CHANNEL_PATTERNS = tuple(re.compile(p) for p in _ES_CHANNEL_CUES)
_ES_MOBILITY_PATTERNS = tuple(re.compile(p) for p in _ES_MOBILITY_CUES)
_ES_BLOCKER_PATTERNS = tuple(re.compile(p) for p in _ES_BLOCKER_CUES)

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
        # Spanish, one-for-one with the English above and just as exact. These
        # are only ever consulted while a question is already pending, so
        # adding "sí" does not widen the window in which anything authorizes an
        # action; an ordinary sentence that merely contains "sí" still fails
        # the whole-turn match. Accented and unaccented spellings both appear
        # because the speech server returns either.
        "sí",
        "si",
        "sí por favor",
        "si por favor",
        "sí hazlo",
        "si hazlo",
        "sí adelante",
        "si adelante",
        "sí llámame",
        "si llamame",
        "claro",
        "claro que sí",
        "claro que si",
        "por supuesto",
        "afirmativo",
        "confirmo",
        "confirmado",
        "hazlo",
        "adelante",
        "correcto",
        "de acuerdo",
        "está bien",
        "esta bien",
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
        # Spanish denials, matched as exactly as the English ones above.
        "no gracias",
        "no, gracias",
        "para nada",
        "negativo",
        "cancela",
        "cancelar",
        "cancélalo",
        "cancelalo",
        "olvídalo",
        "olvidalo",
        "déjalo",
        "dejalo",
        "ahora no",
        "todavía no",
        "todavia no",
        "detente",
        "detente por favor",
        "para",
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
    normalized = re.sub(r"^(?:(?:hey )?(?:friday|jarvis))[,. ]+", "", normalized)
    return " ".join(re.sub(r"[.,!?]", " ", normalized).split())


class HandoffIntent(StrEnum):
    """How confidently a transcript reads as "let's keep talking by phone"."""

    NONE = "none"
    CLARIFY = "clarify"
    DIRECT = "direct"


def _for_inference(transcript: str) -> str:
    """Lower-case, de-accent, de-contract and de-punctuate so only wording matters.

    Accents are folded rather than stripped as punctuation was: ``llámame``
    became ``ll mame`` when every non-ASCII character was replaced by a space,
    which no cue could match. English text carries no accents of its own, so
    folding leaves every English reading exactly as it was.
    """
    text = strip_accents(_normalize(transcript)).lower().replace("’", "'").replace("‘", "'")
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


def handoff_surface_present(transcript: str) -> bool:
    """Whether the turn names a phone, a call, or a reason to leave the keyboard.

    The cheap precondition for consulting a model at all, in either language,
    and the last guard applied to what it says: a turn with no handoff surface
    is not a handoff request however confidently it was read. It is a
    *necessary* condition, never a sufficient one -- "my phone is broken" has a
    surface and is blocked a line later.
    """
    if not isinstance(transcript, str) or not _normalize(transcript):
        return False
    text = _for_inference(transcript)
    return (
        _any(_CHANNEL_PATTERNS, text)
        or _any(_PHONE_MENTION_PATTERNS, text)
        or _any(_MOBILITY_PATTERNS, text)
        or _any(_ES_CHANNEL_PATTERNS, text)
        or _any(_ES_MOBILITY_PATTERNS, text)
    )


#: The only three things a model may say about a turn. Anything else -- another
#: word, another case, prose, a number, nothing at all -- is no reading.
_SEMANTIC_LABELS = {
    "direct": HandoffIntent.DIRECT,
    "clarify": HandoffIntent.CLARIFY,
    "none": HandoffIntent.NONE,
}


#: Any digit at all. Stricter than the deterministic path's three-digit rule on
#: purpose: on the semantic path the words were read by a model, so a turn that
#: dictates *any* part of a destination is refused before a model sees it.
_ANY_DIGIT = re.compile(r"\d")


def handoff_reading_permitted(transcript: str) -> bool:
    """Whether this turn may be read as a handoff request at all.

    The single offline precondition, applied twice on purpose: once *before* a
    turn is sent to a model, so a dictated number or a quoted phrase never
    leaves the machine as a handoff question, and once again on what comes back,
    so the guard does not depend on the caller having asked first.

    All three clauses subtract. A turn passes only when it names a phone or a
    reason to leave the keyboard, is not one of the blocked readings
    (hypothetical, explanation, quoted, somebody else's call, declined, past,
    broken device), and carries no digits.
    """
    if not handoff_surface_present(transcript):
        return False
    text = _for_inference(transcript)
    if _is_blocked(transcript, text) or _any(_ES_BLOCKER_PATTERNS, text):
        return False
    return not _ANY_DIGIT.search(text)


def semantic_handoff_intent(transcript: str, label: object) -> HandoffIntent:
    """Narrow a model's reading of one turn with the offline guards.

    This is the only place a model-supplied reading becomes a
    :class:`HandoffIntent`, and it can only ever *subtract*: an unknown label, a
    blocked turn, a dictated number, or a turn with no handoff surface is NONE.
    A surviving DIRECT still only earns the confirmation question -- nothing
    here, and nothing a model can say, authorizes a dial.
    """
    if not isinstance(label, str) or not isinstance(transcript, str):
        return HandoffIntent.NONE
    intent = _SEMANTIC_LABELS.get(label)
    if intent is None or intent is HandoffIntent.NONE:
        return HandoffIntent.NONE
    if not handoff_reading_permitted(transcript):
        return HandoffIntent.NONE
    return intent


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

    @property
    def idle(self) -> bool:
        """Whether no question of this machine's is open."""
        return self._stage is _Stage.IDLE

    def reset(self) -> None:
        """Drop any pending question, e.g. when the session restarts."""
        self._stage = _Stage.IDLE

    def observe(
        self, transcript: str, *, semantic: HandoffIntent | None = None
    ) -> HandoffDecision:
        """Advance the machine with one final user transcript.

        ``semantic`` is a reading already narrowed by
        :func:`semantic_handoff_intent`. It is consulted only when the offline
        classification found nothing and only while idle: once a question is
        open, the answer is the fixed confirmation vocabulary and nothing else,
        so no reading can shorten the path from a request to a call.
        """
        intent = classify_handoff_intent(transcript)
        if intent is HandoffIntent.NONE and semantic is not None and self.idle:
            intent = semantic

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


class SemanticReader(Protocol):
    """Reads one turn as a handoff intent. Implemented by ``caal.handoff_semantics``."""

    async def read(self, transcript: str) -> Any: ...


#: Receives ``(LanguageReading, transcript)`` for the turn being handled, before
#: any fixed reply of this controller is spoken. Never raises into the turn.
LanguageBinder = Callable[[Any, str], None]


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
        semantic: "SemanticReader | None" = None,
        bind_language: "LanguageBinder | None" = None,
    ) -> None:
        """With ``destination_resolver`` the call goes to the session user's own
        approved number, resolved fresh when they confirm; the static allowlist
        is then ignored entirely. ``user_id`` travels with the call so the
        worker can re-resolve the same profile before dialing.

        ``semantic`` reads a turn the offline classifier found nothing in --
        which is every Spanish request, since the offline net is English. It is
        consulted only while idle, only for a turn that names a phone at all,
        and what it says is narrowed by :func:`semantic_handoff_intent` before
        the machine sees it. ``bind_language`` receives the language that same
        reply read, before this controller says anything, so the first Spanish
        turn is asked about in Spanish rather than the turn after it.
        """
        self._start_call = start_call
        self._allowed_destinations = allowed_destinations
        self._conversation_id = conversation_id
        self._user_id = user_id
        self._destination_resolver = destination_resolver
        self._semantic = semantic
        self._bind_language = bind_language
        self._machine = HandoffIntentMachine()
        # Transcripts consumed by the handoff itself (the request, the "yes").
        self._control_transcripts: list[str] = []

    @property
    def awaiting_confirmation(self) -> bool:
        return self._machine.awaiting_confirmation

    def set_language_binder(self, binder: "LanguageBinder | None") -> None:
        """Bind where this controller publishes the language it just read.

        The controller is built before the turn handler that owns the session's
        language, so the binder arrives afterwards. Without one the reading
        still decides the intent; only the language of the question falls back
        to whatever the session already had.
        """
        self._bind_language = binder

    @property
    def conversation_id(self) -> str | None:
        """Opaque ledger id of the session's live conversation, if any."""
        return self._conversation_id

    def reset(self) -> None:
        self._machine.reset()
        self._control_transcripts.clear()

    async def handle_final_transcript(self, transcript: str, session: _Speaker) -> bool:
        """Handle one final transcript; return true when CAAL consumed the turn."""
        decision = self._machine.observe(transcript, semantic=await self._read(transcript))

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

    async def _read(self, transcript: str) -> HandoffIntent | None:
        """Ask the reader about a turn the offline net found nothing in.

        Skipped entirely -- no model call at all -- when a question is already
        open, when the offline net already recognised the request, or when the
        turn names no phone: those are the turns a reading could not change.
        Any failure is no reading, so an unreachable model costs exactly the
        Spanish reading and nothing else.
        """
        if self._semantic is None or not self._machine.idle:
            return None
        if classify_handoff_intent(transcript) is not HandoffIntent.NONE:
            return None
        try:
            reading = await self._semantic.read(transcript)
        except Exception:
            logger.warning("Phone handoff could not read the turn semantically")
            return None
        if reading.language is not None and self._bind_language is not None:
            try:
                self._bind_language(reading.language, transcript)
            except Exception:
                logger.debug("Binding the handoff turn language failed", exc_info=True)
        return reading.intent

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
            # The language of a fixed reply is this turn's, decided from the
            # session before anything was said; an unknown string is spoken as
            # written. See caal.reply_localization.
            text = speech_reply(text)
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
