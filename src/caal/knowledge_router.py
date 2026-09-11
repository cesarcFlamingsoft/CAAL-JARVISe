"""Route questions about the connected email and calendar accounts to the knowledge index.

The production LLM (Hermes) runs its own tool loop in its own process and is
never given CAAL native tool schemas, so the user-scoped inbox.* and
schedule.* knowledge tools could not be reached from a voice turn at all: the
model, with no tool to call, said it could not access the calendar or email.
A prompt cannot fix a tool that is not in the request.

A question may also name which connected account it is about -- by provider
("my Google calendar"), by the label the user gave the account when they
linked it ("the Vertex calendar", "unread mail for my work account"), or by
its address -- and may ask about mail and calendar in one breath ("my events
today and unread email for the Vertex account"). The naming is carried as a
bounded argument only; which connection it means is decided inside the
knowledge service, among the live connections of the verified user, and an
account nobody recognises is answered as exactly that rather than widened
back out to every account.

This module is the code-level route instead. It sits with the other things
CAAL answers itself before a turn reaches the LLM (phone handoff, background
work, end of call): a deterministic reading of the turn picks the knowledge
tool and its arguments, the tool runs under the session verified scope, and
the tool own spoken answer is said as-is. The reading is bounded, offline,
and conservative: it claims questions about the user own mail and calendar,
and leaves actions (send, schedule, cancel), other channels (SMS, chat), the
work router control phrases, and everything else to their existing paths.

Nothing here logs the utterance, the arguments, or the answer: a routing log
line names the tool and its status, and nothing else.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from caal.knowledge import MAX_QUERY_LENGTH, query_terms
from caal.llm.context_barrier import record_private_answer
from caal.model_routing import Destination, classify_request
from caal.tools.knowledge_tools import (
    BACKEND_UNAVAILABLE,
    MAX_DAYS,
    session_unavailable_result,
)
from caal.tools.registry import ToolRegistry, create_default_registry
from caal.user_scope import UserScope, scoped_tool_arguments
from caal.work_router import Route, deterministic_route

logger = logging.getLogger(__name__)

__all__ = [
    "COULD_NOT_INTERPRET",
    "KNOWLEDGE_TIMEOUT_SECONDS",
    "MAX_INPUT_CHARS",
    "MAX_PLANS",
    "MAX_READABLE_WORDS",
    "TIMEOUT_REPLY",
    "KnowledgePlan",
    "KnowledgeTurnHandler",
    "LocalToolPath",
    "ToolStatusCallback",
    "is_interpretable",
    "plan_knowledge_turn",
    "plan_knowledge_turns",
]

# The reading only needs the shape of the question; a turn buried past this is
# not a question about mail or calendar.
MAX_INPUT_CHARS = 600
# The knowledge service refreshes within its own budget and then falls back to
# what is indexed; this bound only guards against a wedged backend.
KNOWLEDGE_TIMEOUT_SECONDS = 20.0
TIMEOUT_REPLY = (
    "Your connected accounts took too long to answer just now. Please ask me again in a moment."
)
# One turn may name both domains ("my events today and my unread mail"), and
# that is all: two safe, read-only, user-scoped reads, in the order asked.
MAX_PLANS = 2
# The longest turn this bounded reading will try to represent at all.
MAX_READABLE_WORDS = 25

COULD_NOT_INTERPRET = (
    "My local model is not reachable just now, and I could not work out exactly what you "
    "are asking about your connected accounts, so I have not looked anything up. Please ask "
    "me again more simply, such as your next event, or your unread email."
)

ToolStatusCallback = Callable[[bool, list[str], list[dict[str, Any]]], Awaitable[None]]


@dataclass(frozen=True)
class KnowledgePlan:
    """Which knowledge tool answers a turn, and with what. Never carries a scope."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    kind: str = "email"


# --- normalisation ----------------------------------------------------------------------------

_CONTRACTIONS = (
    (re.compile(r"\b(what|when|who|where|how|there|it|that|here)'s\b"), r"\1 is"),
    (re.compile(r"\bi'm\b"), "i am"),
    (re.compile(r"\bi've\b"), "i have"),
    (re.compile(r"\b(have|has|had|do|does|did|is|are|was|were)n't\b"), r"\1 not"),
    (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\b(\w+)'s\b"), r"\1"),
)
_NON_WORD = re.compile(r"[^a-z0-9' ]+")
_LEAD_IN = re.compile(
    r"^(?:(?:hey|hi|ok|okay|so|um|uh|well|and|also)\s+)*"
    r"(?:jarvis\s*)?"
    r"(?:please\s+)?"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
    r"(?:i\s+(?:want|need|would like|d like)\s+(?:you\s+)?to\s+)?"
    r"(?:please\s+)?(?:just\s+)?(?:quickly\s+)?"
)
_TRAILER = re.compile(r"\s+(?:please|jarvis|thanks|thank you|for me|right now|now)$")


def _normalized(text: object) -> str:
    """Lowercased, bounded, contraction-free words; empty for anything unusable."""
    if not isinstance(text, str):
        return ""
    lowered = text[:MAX_INPUT_CHARS].replace("’", "'").lower()
    lowered = lowered.replace("e-mail", "email").replace("e mail", "email")
    for pattern, replacement in _CONTRACTIONS:
        lowered = pattern.sub(replacement, lowered)
    words = " ".join(_NON_WORD.sub(" ", lowered).replace("'", "").split())
    words = _LEAD_IN.sub("", words, count=1)
    while True:
        trimmed = _TRAILER.sub("", words)
        if trimmed == words:
            return words
        words = trimmed


# --- what the reading refuses -----------------------------------------------------------------

# Another channel entirely: never answered from the mail index.
_OTHER_CHANNELS = re.compile(
    r"\b(?:text message|text messages|texts?|sms|whatsapp|telegram|slack|teams message|"
    r"imessage|voicemails?|voice mail|discord|signal message)\b"
)
# An imperative that changes something is an action for the LLM and its own
# tools, never a question for the index.
_ACTION_VERBS = frozenset(
    """
    accept add archive arrange block book call cancel clear clean compose create decline
    delete draft email empty flag forward invite mail make mark message move open organise
    organize plan put remind remove reply reschedule respond schedule send set setup snooze
    star text unsubscribe write
    """.split()
)
# Questions about how mail or calendars work, not about what is in them.
_ABOUT_THE_FEATURE = re.compile(
    r"^(?:how (?:do|can|could|would|should) (?:i|you|we|one)\b|how to\b|what (?:is|are) (?:a|an)\b|"
    r"(?:did|do|does|can|could|have|has|are|will|would) you\b)"
)
# Words that make a turn a question or a request to be told, rather than a remark.
_QUESTION_CUES = re.compile(
    r"\b(?:what|which|when|who|whom|how|is|are|am i|do i|did i|have i|has|does|any|anything|"
    r"anyone|anybody|check|show|tell|read|list|give|let me know|look|see|find|search|"
    r"summarize|summarise|summary|catch me up|update me|brief|go through|latest|newest|"
    r"recent|unread|next)\b"
)

# Where one turn is two questions: "my calendar today and any unread mail".
_CONJUNCTION = re.compile(r"\s+(?:and\s+(?:also|then)|and|plus|as well as)\s+")
_NUMBER = r"(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty)"
_WEEKDAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"

# --- the two domains --------------------------------------------------------------------------

_EMAIL_NOUNS = re.compile(
    r"\b(?:emails?|mails?|inbox|messages?|unread|correspondence|newsletters?|gmail)\b"
)
# Mail asked about by its sender rather than by name: "anything from Bo", "did Bo
# write me", "has Bo sent me anything", "did I hear back from Bo".
_EMAIL_PHRASES = re.compile(
    r"\b(?:(?:anything|something|nothing|news|word|updates?) from|hear(?:d)? (?:back )?from|"
    r"(?:sent|emailed|mailed|messaged|contacted) me|(?:wrote|written|write) (?:to )?me|"
    r"(?:got|get|gotten) back to me|reach(?:ed)? out|(?:latest|newest|anything new) from|"
    r"did i (?:get|receive) anything)\b"
)
_CALENDAR_NOUNS = re.compile(
    r"\b(?:calendars?|schedule|agenda|meetings?|appointments?|events?|standups?|"
    r"invites?|booked|busy|free|availability|available)\b"
)
_CALENDAR_PHRASES = re.compile(
    r"\b(?:what (?:do|have) i (?:have|got)|what am i doing|"
    r"(?:how|what) (?:does|is|do|will|would) (?:my |the |next |this )?"
    r"(?:day|morning|afternoon|evening|week|weekend|month|today|tomorrow) (?:look|looking|shaping)|"
    r"what is next|what is up next|next up|what is coming up|coming up next|"
    r"what do i have next|anything (?:on|scheduled|planned|booked)|"
    r"(?:do|have) i (?:have|got) (?:anything|something|any)|"
    r"anything (?:today|tomorrow|tonight|this (?:week|weekend|morning|afternoon|evening)|"
    r"next week|on (?:" + _WEEKDAYS + r"))|"
    r"plans (?:for|today|tomorrow|tonight|this|next|on))\b"
)

_ACCOUNTS = (
    (re.compile(r"\b(?:gmail|google)\b"), "google"),
    (re.compile(r"\b(?:outlook|microsoft|office 365|o365|hotmail|exchange)\b"), "microsoft"),
    (re.compile(r"\bzoho\b"), "zoho"),
)

# --- naming one connected account -------------------------------------------------------------

# A question may name one account instead of a provider: "my work account",
# "the Vertex calendar", "unread mail for ana@vertex.example". The reading is
# bounded and it never decides whose account that is -- it carries the words
# the user said as an argument, and :class:`caal.knowledge.KnowledgeService`
# resolves them against the live connections of the verified user, or matches
# nothing at all.

# An address is read from the raw turn: normalisation drops the @ and the dots.
_EMAIL_ADDRESS = re.compile(r"[a-z0-9][a-z0-9._%+\-]*@[a-z0-9][a-z0-9.\-]*\.[a-z]{2,}")
# A label is a name, not a sentence.
_LABEL_WORDS = 3
# Words that are never part of a label: the two domains, the time words, and
# the scaffolding of a spoken request.
_NOT_A_LABEL = (
    r"accounts?|calendars?|schedules?|agendas?|events?|meetings?|appointments?|invites?|"
    r"emails?|mails?|mailbox|inbox|inboxes|messages?|correspondence|newsletters?|"
    r"unread|read|new|newest|latest|recent|last|first|next|upcoming|other|coming|"
    r"today|tonight|tomorrow|yesterday|days?|weeks?|weekend|months?|morning|afternoon|"
    r"evening|now|" + _WEEKDAYS + r"|"
    r"for|from|of|in|on|at|to|with|about|and|or|the|a|an|my|our|me|i|you|your|any|some|"
    r"give|show|tell|check|list|look|see|find|search|what|which|when|who|how|is|are|do|"
    r"does|did|have|has|had|please|got|there|that|this|it|anything|something"
)
_LABEL_WORD = r"(?:(?!(?:" + _NOT_A_LABEL + r")\b)[a-z0-9]+)"
# Real names hold connector words: "u of a", "University of Alberta", "city of
# edmonton", "Smith and Sons". A connector is read only *between* two words of
# the name, so a label still never begins or ends with one, and the scaffolding
# of a request ("the calendar for ...", "email from ...") stays outside it.
_LABEL_JOINER = r"(?:of|the|and)\b"
# What a connector may join to: another word of the name, or the single letter
# of a spoken initialism -- the "a" of "u of a", the "i" of "u of i" -- which is
# refused anywhere else because on its own it is an article.
_LABEL_JOINED_WORD = r"(?:" + _LABEL_WORD + r"|[a-z0-9]\b)"
# One more word of the name, reached either directly or across connectors. The
# connectors are free: the bound counts the words of the name itself.
_LABEL_NEXT_WORD = (
    r"(?:\s+" + _LABEL_JOINER + r"){1,2}\s+" + _LABEL_JOINED_WORD + r"|\s+" + _LABEL_WORD
)
_LABEL = (
    r"(?P<label>"
    + _LABEL_WORD
    + r"(?:"
    + _LABEL_NEXT_WORD
    + r"){0,"
    + str(_LABEL_WORDS - 1)
    + r"})"
)
_MAIL_NOUNS = r"emails?|mails?|mailbox|inbox|messages?"
_DIARY_NOUNS = r"calendars?|schedule|agenda|events?"
# What an account holds, as opposed to what it holds them for: "my Vertex
# calendar" names an account, "my dentist appointment" names an event.
_HELD_IN_AN_ACCOUNT = r"calendars?|schedule|agenda|inbox|inboxes|mailbox|emails?|mails?"
# Each hint names the account in `label`, and keeps in `keep` whatever of the
# question the naming borrowed, so the rest of the reading still sees it.
_ACCOUNT_HINTS = (
    # "my work account", "the Flaming Soft account", "of Vertex account"
    re.compile(r"\b" + _LABEL + r"\s+accounts?\b"),
    # "the Vertex calendar", "my work inbox"
    re.compile(r"\b(?:my|the|our)\s+" + _LABEL + r"\s+(?=(?:" + _HELD_IN_AN_ACCOUNT + r")\b)"),
    # "the calendar for Vertex", "events of Vertex"
    re.compile(
        r"\b(?P<keep>" + _DIARY_NOUNS + r")\s+(?:for|from|of)\s+(?:my|the|our)?\s*" + _LABEL + r"\b"
    ),
    # "the inbox for Vertex". Never "from": after a mail noun that names a sender.
    re.compile(
        r"\b(?P<keep>" + _MAIL_NOUNS + r")\s+(?:for|of)\s+(?:my|the|our)?\s*" + _LABEL + r"\b"
    ),
)

_NUMBER_WORDS = dict(
    one=1,
    two=2,
    three=3,
    four=4,
    five=5,
    six=6,
    seven=7,
    eight=8,
    nine=9,
    ten=10,
    fifteen=15,
    twenty=20,
)


# --- time words -------------------------------------------------------------------------------

_TODAY = re.compile(
    r"\b(?:today|tonight|this (?:morning|afternoon|evening)|later today|"
    r"rest of (?:the|my) day|my (?:day|morning|afternoon|evening))\b"
)
_TOMORROW = re.compile(r"\btomorrow\b")
_DAY_AFTER = re.compile(r"\bday after tomorrow\b")
_WEEKDAY = re.compile(r"\b(?:on |this |next )?(" + _WEEKDAYS + r")\b")
_THIS_WEEK = re.compile(
    r"\b(?:this week|the week|my week|week ahead|rest of the week|coming week|coming days|"
    r"this weekend|the weekend|my weekend)\b"
)
_NEXT_WEEK = re.compile(r"\bnext week\b")
_MONTH = re.compile(r"\b(?:this month|the month|next month|my month)\b")
_NEXT_DAYS = re.compile(r"\b(?:next|coming|following) " + _NUMBER + r" (day|week)s?\b")
# Nearest, soonest, next: a question about the very next thing on the calendar
# rather than about a stretch of time. "next week" is a window and is read as
# one above, before any of this runs.
_NEAREST_PHRASE = re.compile(
    r"\bwhat is (?:coming )?up next\b|\bwhat is next\b|\bnext up\b|\bwhat is coming up\b|"
    r"\bcoming up next\b|\bwhat do i have next\b"
)
_NEAREST_WORD = re.compile(r"\b(?:next|nearest|soonest|closest|first)\b")
_NEAREST_COUNT = re.compile(
    r"\b(?:next|nearest|soonest|closest|first|upcoming)\s+"
    r"(?:(?P<vague>few|couple(?: of)?)|(?P<count>" + _NUMBER + r"))\b"
)
# A plural noun is the only thing that turns "next" into more than one answer.
_SEVERAL_EVENTS = re.compile(
    r"\b(?:meetings|appointments|events|calls|things|items|sessions|standups|invites)\b"
)
NEAREST_DEFAULT = 1
NEAREST_SEVERAL = 3
MAX_NEAREST = 5


def _number(word: str) -> int | None:
    if word.isdigit():
        return int(word)
    return _NUMBER_WORDS.get(word)


def _window(text: str) -> dict[str, Any]:
    """The day or days a calendar question is about, as the schedule tools take them."""
    if _DAY_AFTER.search(text):
        return dict(day="day after tomorrow")
    if _TOMORROW.search(text):
        return dict(day="tomorrow")
    if _TODAY.search(text):
        return dict(day="today")
    weekday = _WEEKDAY.search(text)
    if weekday:
        return dict(day=weekday.group(1))
    span = _NEXT_DAYS.search(text)
    if span:
        count = _number(span.group(1)) or 0
        if span.group(2) == "week":
            count *= 7
        if count >= 1:
            return dict(days=min(count, MAX_DAYS))
    if _NEXT_WEEK.search(text):
        return dict(days=14)
    if _THIS_WEEK.search(text):
        return dict(days=7)
    if _MONTH.search(text):
        return dict(days=MAX_DAYS)
    return dict()


# --- search targets ---------------------------------------------------------------------------

_TARGET_TAIL = re.compile(
    r"\s+(?:today|yesterday|tonight|tomorrow|recently|lately|yet|already|at all|or not|"
    r"this (?:week|morning|afternoon|evening|month|weekend)|next week|last (?:week|night)|"
    r"(?:on |this |next )?(?:" + _WEEKDAYS + r")|"
    r"in (?:my|the) (?:inbox|email|emails|mail|gmail|outlook|zoho|calendar|calendars)|"
    r"on (?:my|the) (?:calendar|calendars|schedule)|"
    r"emails?|mails?|messages?|meeting|appointment|event|call)$"
)
_TARGET_HEAD = re.compile(
    r"^(?:my|the|a|an|our|any|some|that|this|next|first|last|other|new|newest|latest|"
    r"most recent|recent|nearest|soonest|closest|upcoming|coming)\s+"
)
_PRONOUN_TARGETS = frozenset(
    "anyone anybody someone somebody everyone people others them him her me us it".split()
)
# Ordinals and the like name a position, not an event or a sender.
_POSITION_WORDS = frozenset(
    "next first last upcoming new other earliest latest coming newest recent one "
    "nearest soonest closest".split()
)
_FROM = re.compile(r"\bfrom\s+(.+)$")
_ABOUT = re.compile(
    r"\b(?:about|regarding|concerning|mentioning|referencing|on the subject of|"
    r"with the subject|titled|called|named)\s+(.+)$"
)
_SENDER_DID = re.compile(
    r"^(?:did|has|have|had)\s+(.+?)\s+(?:email|emailed|mail|mailed|write|written|wrote|"
    r"send|sent|message|messaged|reply|replied|get back|gotten back|got back|contact|"
    r"contacted|reach out|reached out)\b"
)
_READ = re.compile(
    r"^(?:read|open)\b|\bread (?:me |it |that |out |aloud )|\bwhat does it say\b|"
    r"\bwhat (?:was|is) (?:that|the|this|it) (?:email|message|mail|one) about\b"
)
# "what did Bo say", "what does Bo want": one sender, read as a summary.
_SENDER_SAID = re.compile(r"^what (?:did|does)\s+(.+?)\s+(?:say|write|send|want|ask|need)\b")
_NEWEST = re.compile(r"\b(?:latest|newest|last|most recent|first|top)\b")
_UNREAD = re.compile(r"\bunread\b|\b(?:have not|not|did not) read\b")
_LIMIT = (
    re.compile(r"\b(?:last|latest|newest|top|first|most recent|recent)\s+" + _NUMBER + r"\b"),
    re.compile(
        _NUMBER
        + r"\s+(?:most recent |latest |last |newest |recent |new )?(?:emails?|messages?|mails?)\b"
    ),
)
_WITH = re.compile(r"\bwith\s+(.+)$")
_TITLED_EVENT = re.compile(
    r"\b(?:a|an|any|the|my|our)\s+((?:(?!\b(?:next|first|last|other|new|upcoming|any|"
    r"scheduled|planned)\b)[a-z0-9]+\s+){1,3}?)"
    r"(?:appointment|meeting|call|event|session|interview|lunch|dinner|visit|checkup|"
    r"check up|review|standup|class|lesson)s?\b"
)
_WHEN_IS = re.compile(r"\b(?:when is|what time is|what time does)\s+(?:my|the|our)?\s*(.+)$")
_EVENT_TAIL = re.compile(r"\s+(?:start|starts|begin|begins|scheduled|happening|happen|is)$")


def _target(raw: str | None) -> str | None:
    """A search target trimmed of articles, time words and domain nouns; None if nothing."""
    if not raw:
        return None
    words = raw.strip()
    while True:
        trimmed = _TARGET_TAIL.sub("", _EVENT_TAIL.sub("", words)).strip()
        trimmed = _TARGET_HEAD.sub("", trimmed)
        if trimmed == words:
            break
        words = trimmed
    terms = query_terms(words)
    if not terms or all(term in _PRONOUN_TARGETS or term in _POSITION_WORDS for term in terms):
        return None
    return words[:MAX_QUERY_LENGTH]


def _mangled(address: str) -> str:
    """An address as normalisation leaves it: ana@x.example becomes ana x example."""
    return " ".join(_NON_WORD.sub(" ", address).split())


def _address(raw: object, text: str) -> str | None:
    """An account named in full, when the words of the question came from it."""
    if not isinstance(raw, str):
        return None
    for found in _EMAIL_ADDRESS.finditer(raw[:MAX_INPUT_CHARS].replace("\u2019", "'").lower()):
        if _mangled(found.group(0)) in text:
            return found.group(0)
    return None


def _account(text: str, raw: object = None) -> tuple[dict[str, Any], str]:
    """The one account a question names, and the question with that naming removed.

    A provider word is read exactly as it always was. Otherwise the bounded
    label the user said -- what they called the account when they linked it,
    or its address -- is carried as an argument. Nothing here reads a
    connection, so nothing here can name an account of another user.
    """
    address = _address(raw, text)
    if address is not None:
        return dict(account=address), " ".join(text.replace(_mangled(address), " ", 1).split())
    for pattern, provider in _ACCOUNTS:
        if pattern.search(text):
            return dict(account=provider), text
    for pattern in _ACCOUNT_HINTS:
        found = pattern.search(text)
        if found is None:
            continue
        label = " ".join(found.group("label").split())
        if not label:
            continue
        keep = found.groupdict().get("keep") or ""
        rest = text[: found.start()] + " " + keep + " " + text[found.end() :]
        return dict(account=label[:MAX_QUERY_LENGTH]), " ".join(rest.split())
    return dict(), text


def _email_plan(text: str, account: dict[str, Any]) -> KnowledgePlan:
    arguments: dict[str, Any] = dict()
    source = _FROM.search(text)
    query = _target(source.group(1)) if source else None
    if query is None:
        sender = _SENDER_DID.match(text)
        if sender and not re.match(r"^(?:i|we|you|me)\b", sender.group(1)):
            query = _target(sender.group(1))
    if query is None:
        topic = _ABOUT.search(text)
        query = _target(topic.group(1)) if topic else None
    said = _SENDER_SAID.match(text)
    if said and not re.match(r"^(?:i|we|you)\b", said.group(1)):
        spoken_by = _target(said.group(1))
        if spoken_by is not None:
            return KnowledgePlan("inbox.read_summary", dict(query=spoken_by, **account), "email")
    if _READ.search(text):
        if query is not None and not _NEWEST.search(text):
            arguments["query"] = query
        return KnowledgePlan("inbox.read_summary", dict(arguments, **account), "email")
    if query is not None:
        return KnowledgePlan("inbox.search", dict(query=query, **account), "email")
    if _UNREAD.search(text):
        arguments["unread_only"] = True
    for pattern in _LIMIT:
        found = pattern.search(text)
        if found:
            count = _number(found.group(1))
            if count is not None and 1 <= count <= 25:
                arguments["limit"] = count
            break
    return KnowledgePlan("inbox.recent", dict(arguments, **account), "email")


def _nearest_limit(text: str) -> int | None:
    """How many of the soonest events a question asks for; None if it asks for none.

    A turn that named a window never reaches here: "next week" is a stretch of
    time. What is left is the nearest-thing question, which is one event unless
    the words themselves ask for several.
    """
    if not (_NEAREST_PHRASE.search(text) or _NEAREST_WORD.search(text)):
        return None
    found = _NEAREST_COUNT.search(text)
    if found is not None:
        if found.group("vague"):
            return NEAREST_SEVERAL
        count = _number(found.group("count"))
        if count is not None:
            return max(1, min(count, MAX_NEAREST))
    return NEAREST_SEVERAL if _SEVERAL_EVENTS.search(text) else NEAREST_DEFAULT


def _calendar_plan(text: str, account: dict[str, Any]) -> KnowledgePlan:
    window = _window(text)
    query: str | None = None
    for pattern in (_WITH, _ABOUT, _TITLED_EVENT):
        found = pattern.search(text)
        if found:
            query = _target(found.group(1))
            if query is not None:
                break
    if query is None:
        when = _WHEN_IS.search(text)
        if when:
            query = _target(when.group(1))
    if query is not None:
        return KnowledgePlan(
            "schedule.find_event", dict(query=query, **window, **account), "calendar"
        )
    if not window:
        nearest = _nearest_limit(text)
        if nearest is not None:
            return KnowledgePlan("schedule.next", dict(limit=nearest, **account), "calendar")
    return KnowledgePlan("schedule.upcoming", dict(window, **account), "calendar")


def _domains(text: str) -> tuple[re.Match[str] | None, re.Match[str] | None]:
    """Where this text asks about mail, and where it asks about a calendar."""
    email = _EMAIL_NOUNS.search(text) or _EMAIL_PHRASES.search(text)
    calendar = _CALENDAR_NOUNS.search(text)
    if calendar is None and email is None:
        # "do I have anything today" names no calendar, but asks about one.
        calendar = _CALENDAR_PHRASES.search(text)
    return email, calendar


def _kind(email: re.Match[str] | None, calendar: re.Match[str] | None) -> str:
    """Which domain a text is about; the one asked about first when it is both."""
    if email is not None and (calendar is None or email.start() <= calendar.start()):
        return "email"
    return "calendar"


def _plan_for(kind: str, text: str, account: dict[str, Any]) -> KnowledgePlan:
    return _email_plan(text, account) if kind == "email" else _calendar_plan(text, account)


def _segments(words: str) -> list[str]:
    """The clauses a turn was joined from, each read on its own afterwards."""
    parts = [_LEAD_IN.sub("", part, count=1).strip() for part in _CONJUNCTION.split(words)]
    return [part for part in parts if part]


def _carries_an_action(words: str) -> bool:
    """Whether any clause of the turn asks for something to be done.

    Half of a request is never answered as if it were the whole of it: "my
    calendar today and send Bob the address" is the LLM business, tools and
    all, not a calendar question with the rest of it dropped.
    """
    return any(segment.split()[0] in _ACTION_VERBS for segment in _segments(words))


def _combined(words: str, raw: object) -> list[KnowledgePlan]:
    """One plan per domain for a turn that asks about both; see plan_knowledge_turns.

    Returns no plans when the turn does not split into a mail clause and a
    calendar clause: it is then the single question it always was.
    """
    segments = _segments(words)
    if len(segments) < 2:
        return []
    claimed: dict[str, str] = {}
    for segment in segments:
        email, calendar = _domains(segment)
        if email is None and calendar is None:
            continue
        claimed.setdefault(_kind(email, calendar), segment)
    if len(claimed) < MAX_PLANS:
        return []
    read = dict((kind, _account(segment, raw)) for kind, segment in claimed.items())
    named = set(account["account"] for account, _ in read.values() if account)
    # One account named once qualifies the whole turn: "events today and unread
    # mail for the Vertex account" asks both about Vertex. Two named accounts
    # stay where they were said.
    shared = dict(account=named.pop()) if len(named) == 1 else dict()
    plans = [_plan_for(kind, rest, account or shared) for kind, (account, rest) in read.items()]
    return plans[:MAX_PLANS]


def plan_knowledge_turns(text: object) -> list[KnowledgePlan]:
    """Read one turn offline; no plans means it is not a question for the index.

    The control and work readings of :mod:`caal.work_router` keep their
    authority: anything they claim is theirs. An imperative that changes
    something (send, schedule, cancel) belongs to the LLM and its own tools.
    What is left must ask about mail or calendar in the first person, or name
    one of the user own accounts, or it is not claimed. A turn that asks about
    both domains is answered by one read of each, in the order it asked.
    """
    words = _normalized(text)
    if not words:
        return []
    if deterministic_route(text).route is not Route.CONVERSATION:
        return []
    if _OTHER_CHANNELS.search(words) or _ABOUT_THE_FEATURE.match(words):
        return []
    if _carries_an_action(words):
        return []
    email, calendar = _domains(words)
    if email is None and calendar is None:
        return []
    account, rest = _account(words, text)
    # Naming an account is itself the request: "my Google calendar" asks for it.
    asked = (
        _QUESTION_CUES.search(words)
        or _SENDER_DID.match(words)
        or _SENDER_SAID.match(words)
        or bool(account)
    )
    if not asked:
        return []
    if email is not None and calendar is not None:
        both = _combined(words, text)
        if both:
            return both
    return [_plan_for(_kind(email, calendar), rest, account)]


def plan_knowledge_turn(text: object) -> KnowledgePlan | None:
    """The first plan a turn asks for; None means it is not a question for the index."""
    plans = plan_knowledge_turns(text)
    return plans[0] if plans else None


# --- what the reading cannot represent ---------------------------------------------------------

# Semantics that need a model to read: comparisons, overlaps, arithmetic over a
# window, preference. The deterministic reading would answer one of these with a
# broad "here is your week", which is not what was asked; during a local model
# outage it says so instead.
_TOO_COMPLEX = re.compile(
    r"\b(?:compare|compared|comparison|versus|vs|overlap(?:s|ping)?|conflict(?:s|ing)?|"
    r"clash(?:es|ing)?|double booked|free time|busiest|"
    r"how (?:long|many hours|much time)|instead of|apart from|besides|except|"
    r"prioriti[sz]e|most important|which one should|should i)\b"
)


def is_interpretable(text: object) -> bool:
    """Whether this bounded reading can represent the turn faithfully.

    Only consulted when no local model is available to read it: a turn this
    says no to is answered honestly rather than approximated.
    """
    words = _normalized(text)
    if not words or len(words.split()) > MAX_READABLE_WORDS:
        return False
    return _TOO_COMPLEX.search(words) is None


# --- is there a local model, and will it get this turn -----------------------------------------

# What :meth:`LocalToolPath.status` answers.
LOCAL = "local"
ESCALATED = "escalated"
UNAVAILABLE = "unavailable"
ABSENT = "absent"

# A readiness answer is reused for a while: this runs on the way to a spoken
# reply, so it must cost nothing on most turns.
PROBE_TTL_SECONDS = 30.0
PROBE_FAILURE_TTL_SECONDS = 10.0
PROBE_TIMEOUT_SECONDS = 3.0


class LocalToolPath:
    """Whether a local, tool-capable model will get this turn and can answer it.

    JARVIS main model is the local one, and it is the thing that should read a
    request about the connected accounts: which tool, which window, which
    account, how many results. So when it is going to see the turn *and* it is
    reachable, this route stands aside. It only answers itself when the turn is
    going somewhere that never receives CAAL tool schemas (Hermes), or when the
    local model is not answering at all.

    Nothing here is given the turn to store or log; it is read once, offline.
    """

    def __init__(
        self,
        provider: Any | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        ttl_seconds: float = PROBE_TTL_SECONDS,
        failure_ttl_seconds: float = PROBE_FAILURE_TTL_SECONDS,
        timeout_seconds: float = PROBE_TIMEOUT_SECONDS,
    ) -> None:
        self._provider = provider
        self._clock = clock
        self._ttl = max(0.0, float(ttl_seconds))
        self._failure_ttl = max(0.0, float(failure_ttl_seconds))
        self._timeout = max(0.01, float(timeout_seconds))
        self._answer: bool | None = None
        self._until = 0.0

    @property
    def offers_tools(self) -> bool:
        """Whether this provider is given the CAAL native tool catalog at all."""
        provider = self._provider
        return provider is not None and not getattr(provider, "manages_own_tools", True)

    async def status(self, text: object) -> str:
        """Where this turn is going: local, escalated, unavailable, or absent."""
        if not self.offers_tools:
            return ABSENT
        if self._escalates(text):
            return ESCALATED
        return LOCAL if await self.reachable() else UNAVAILABLE

    def _escalates(self, text: object) -> bool:
        """Whether the routing sends this turn to the agent harness instead."""
        if not getattr(self._provider, "escalation_available", False):
            return False
        return classify_request(text).destination is not Destination.LOCAL

    async def reachable(self) -> bool:
        """Whether the local model answered a readiness probe recently enough."""
        if self._answer is not None and self._clock() < self._until:
            return self._answer
        answer = True
        probe = getattr(self._provider, "reachable", None)
        if callable(probe):
            try:
                result = probe()
                if inspect.isawaitable(result):
                    result = await asyncio.wait_for(result, self._timeout)
                answer = bool(result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - unreachable is an answer
                logger.warning("The local model readiness probe failed (%s)", type(exc).__name__)
                answer = False
        self._answer = answer
        self._until = self._clock() + (self._ttl if answer else self._failure_ttl)
        return answer


# --- executing the plan ----------------------------------------------------------------------


def _unavailable(message: str) -> dict[str, Any]:
    return dict(status="unavailable", message=message, data=dict())


async def _call(handler: Callable[..., Any], arguments: dict[str, Any]) -> Any:
    result = handler(**arguments)
    if inspect.isawaitable(result):
        result = await result
    return result


class KnowledgeTurnHandler:
    """Answer a connected-account question locally, under the session own scope.

    The route exists only when multi-user identity is configured: a legacy
    single-user deployment keeps the settings-configured IMAP/ICS tools of its
    LLM. Under multi-user, a verified user is bound by the runtime and an
    anonymous session is refused in the tools own words, so the LLM never gets
    to guess about either. Every failure becomes a spoken, truthful reply.
    """

    def __init__(
        self,
        *,
        scope: UserScope,
        registry: ToolRegistry | None = None,
        tool_data_cache: Any | None = None,
        on_tool_status: ToolStatusCallback | None = None,
        timeout_seconds: float = KNOWLEDGE_TIMEOUT_SECONDS,
        local_tools: LocalToolPath | None = None,
    ) -> None:
        self._scope = scope
        self._registry = registry
        # Whether the local, tool-capable model is going to see this turn. When
        # it is, this route stands aside and lets it read the request itself.
        self._local = local_tools
        # Shared with the LLM node, which is the one authority on what may
        # outlive a turn: it refuses connected-account data, so this offer is
        # made and declined rather than decided again here.
        self._cache = tool_data_cache
        self._on_tool_status = on_tool_status
        self._timeout = max(0.01, float(timeout_seconds))

    @property
    def enabled(self) -> bool:
        return self._scope.identity_configured

    def bind_tool_status(self, callback: ToolStatusCallback | None) -> None:
        """Attach the frontend tool indicator once the session can publish to it."""
        self._on_tool_status = callback

    async def handle(self, text: object, session: Any) -> bool:
        """Answer the turn if it is a question for the index; report whether it was.

        A turn that asks about both mail and calendar runs one read of each,
        in the order it asked, and is spoken as one answer. A refusal is said
        once: there is nothing to add by repeating it for the second read.
        """
        if not self.enabled:
            return False
        plans = plan_knowledge_turns(text)
        if not plans:
            return False
        if self._local is not None:
            where = await self._local.status(text)
            if where == LOCAL:
                # The local model has these tools and is answering; a
                # deterministic reading must not speak over it.
                logger.info("Leaving a connected-account question to the local model")
                return False
            if where == UNAVAILABLE and not is_interpretable(text):
                # No model to read it, and a broad all-account read would not
                # be the answer to what was actually asked.
                logger.info("No local model, and the request is not one this route can read")
                await self._speak(session, COULD_NOT_INTERPRET, plans[0].tool)
                return True
        spoken: list[str] = []
        for plan in plans:
            result = await self.answer(plan)
            message = result.get("message")
            if isinstance(message, str) and message.strip() not in ("", *spoken):
                spoken.append(message.strip())
            if result.get("status") == "unauthorized":
                break
        await self._speak(
            session, " ".join(spoken) if spoken else BACKEND_UNAVAILABLE, plans[0].tool
        )
        return True

    @staticmethod
    async def _speak(session: Any, message: str, tool: str) -> None:
        """Say one answer; a session that cannot speak is a warning, not a failure.

        What is said from the connected accounts is marked private first: it
        returns on later turns as ordinary assistant transcript, and a turn that
        escalates to Hermes must not carry it. Only a salted hash is kept.
        """
        record_private_answer(message)
        try:
            await session.say(message)
        except Exception as exc:  # noqa: BLE001 - the turn is claimed either way
            logger.warning("Could not speak the %s answer (%s)", tool, type(exc).__name__)

    async def answer(self, plan: KnowledgePlan) -> dict[str, Any]:
        """Run the planned tool under the session scope; always a spoken-contract dict."""
        registry = self._registry
        if registry is None:
            registry = self._registry = create_default_registry()
        try:
            tool = registry.get(plan.tool)
        except KeyError:
            logger.error("Knowledge route names an unregistered tool %s", plan.tool)
            return _unavailable(BACKEND_UNAVAILABLE)
        # The utterance never chooses whose data is read: the plan carries no
        # scope and the session verified scope is bound here or the call is refused.
        bound = scoped_tool_arguments(tool, plan.arguments, self._scope)
        if bound is None:
            logger.info("Refused connected-account tool %s for an unidentified session", plan.tool)
            return session_unavailable_result()
        logger.info("Answering a connected-account question locally with %s", plan.tool)
        await self._publish(plan)
        try:
            result = await asyncio.wait_for(_call(tool.handler, bound), self._timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Connected-account tool %s timed out after %.1fs", plan.tool, self._timeout
            )
            return _unavailable(TIMEOUT_REPLY)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a tool failure is spoken, never raised
            logger.error("Connected-account tool %s failed (%s)", plan.tool, type(exc).__name__)
            return _unavailable(BACKEND_UNAVAILABLE)
        message = result.get("message") if isinstance(result, dict) else None
        if not isinstance(message, str) or not message.strip():
            logger.error("Connected-account tool %s answered off contract", plan.tool)
            return _unavailable(BACKEND_UNAVAILABLE)
        answer: dict[str, Any] = dict(result)
        logger.info("Connected-account tool %s returned status=%s", plan.tool, answer.get("status"))
        data = answer.get("data")
        if self._cache is not None and data:
            try:
                # Refused for these tools: connected-account data must not be
                # injected into the context of a later turn, which may be a
                # turn that goes to Hermes.
                self._cache.add(plan.tool, data)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not cache the %s answer (%s)", plan.tool, type(exc).__name__)
        return answer

    async def _publish(self, plan: KnowledgePlan) -> None:
        if self._on_tool_status is None:
            return
        try:
            await self._on_tool_status(True, [plan.tool], [dict(plan.arguments)])
        except Exception as exc:  # noqa: BLE001 - the indicator is cosmetic
            logger.debug("Tool status publish failed (%s)", type(exc).__name__)
