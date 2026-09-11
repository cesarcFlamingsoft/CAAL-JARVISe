"""Decide which model answers one user turn: the local one, Hermes, or Claude Code.

JARVIS runs on a local Ollama model. Two kinds of turn are not its work:

``coding``
    Anything about code. It is delegated to the Hermes agent runtime
    (:mod:`caal.coding_delegation`), which carries it out with its own Claude
    Code capability at medium effort on Claude Code's own default model. CAAL
    itself never touches a repository: its container has neither.

``harness``
    A request that produces an artifact, needs research, or takes several
    steps before there is an answer. That is what the Hermes agent runtime is
    for, and it is where the existing long-work net already points.

Everything else is ordinary conversation and stays on the local model.

The reading is an authorization, not a guess, so it is made here rather than
by a model: offline, deterministic, bounded, and in a fixed order -- coding
first, harness second, local otherwise. The background-task control commands
(cancel, status) keep their own authority and are never claimed.

Nothing in this module logs, stores, or returns the words of the turn. A
decision names a destination and where it came from, and nothing else.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

from .work_router import Route, deterministic_route

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_ROUTING_INPUT_CHARS",
    "Destination",
    "RoutingDecision",
    "RoutingSource",
    "classify_request",
]

# The reading only needs the shape of the request. A turn buried past this is
# not the request; bounding it also bounds the work this does on every turn.
MAX_ROUTING_INPUT_CHARS = 600


class Destination(str, Enum):
    """Which runtime answers the turn."""

    LOCAL = "local"
    HARNESS = "harness"
    CODING = "coding"


class RoutingSource(str, Enum):
    """Why. Safe to log: it names no content."""

    # A background-task control command; the queue owns it.
    CONTROL = "control"
    CODING = "coding"
    # The existing deterministic long-work net claimed it.
    LONG_WORK = "long_work"
    # An explicit multi-step / research phrasing.
    HARNESS = "harness"
    LOCAL = "local"


@dataclass(frozen=True)
class RoutingDecision:
    """One routing decision. Deliberately holds no request text."""

    destination: Destination
    source: RoutingSource

    @property
    def is_coding(self) -> bool:
        return self.destination is Destination.CODING

    @property
    def escalates(self) -> bool:
        """Whether the turn needs something other than the local model."""
        return self.destination is not Destination.LOCAL


_NON_WORD = re.compile(r"[^a-z0-9+#' ]+")


def _normalized(text: object) -> str:
    """Lowercased, bounded, punctuation-free words; empty for anything unusable."""
    if not isinstance(text, str):
        return ""
    lowered = text[:MAX_ROUTING_INPUT_CHARS].replace("\u2019", "'").lower()
    return " ".join(_NON_WORD.sub(" ", lowered).split())


# --- coding -------------------------------------------------------------------

# Words that are about code whatever else the sentence says.
_STRONG_CODE = re.compile(
    r"\b(?:claude code|codebase|code base|source code|repo|repos|repository|repositories|"
    r"pull request|merge conflict|stack ?trace|traceback|syntax error|type error|"
    r"compiler|compile|segfault|null pointer|unit tests?|test suite|regex|"
    r"regular expression|refactor(?:s|ed|ing)?|linter|lint|"
    r"debug(?:s|ged|ging)?|implement(?:s|ed|ing)?|"
    r"git (?:branch|commit|diff|repo|history|log))\b"
)
# Nouns that are about code only next to something that acts on them.
_CODE_NOUN = re.compile(
    r"\b(?:code|function|functions|method|methods|class|classes|script|scripts|module|modules|"
    r"package|packages|library|libraries|api|apis|endpoint|endpoints|bug|bugs|tests?|"
    r"variable|variables|loop|loops|query|queries|schema|component|components|program|"
    r"algorithm|parser|handler|provider|migration|dependency|dependencies|build)\b"
)
_LANGUAGE = re.compile(
    r"\b(?:python|javascript|typescript|java|kotlin|swift|rust|golang|ruby|php|sql|html|css|"
    r"react|nextjs|node|django|flask|fastapi|pytest|numpy|pandas|"
    r"(?:bash|shell|sh) script)\b"
)
_CODE_VERB = re.compile(
    r"\b(?:write|writes|wrote|code|coded|coding|fix|fixes|fixed|patch|patches|patched|"
    r"rewrite|rewrites|rewrote|optimi[sz]e|optimi[sz]ed|port|ported|migrate|migrated|"
    r"create|creates|created|add|adds|added|update|updates|updated|change|changes|changed|"
    r"modify|modifies|modified|remove|removes|removed|delete|deletes|deleted|rename|renamed|"
    r"review|reviews|reviewed|clean up|cleans up|cleaned up|test|tests|tested|"
    r"build|builds|built|generate|generates|generated|deploy|deploys)\b"
)
# Talking *about* coding is not a coding request.
_CODING_META = (
    re.compile(
        r"^(?:can|could|do|are|will|would) you\b[^.?!]*"
        r"\b(?:code|coding|program|programming|write code)\s*$"
    ),
    re.compile(
        r"^do you know (?:any )?(?:python|javascript|typescript|java|rust|golang|sql|"
        r"how to (?:code|program))\b"
    ),
    re.compile(r"^what (?:is|are) (?:a|an|the)\b"),
)


def _is_coding(words: str) -> bool:
    """Whether the turn asks for something to be done with code."""
    if any(pattern.match(words) for pattern in _CODING_META):
        return False
    if _STRONG_CODE.search(words):
        return True
    if not _CODE_VERB.search(words):
        return False
    return bool(_CODE_NOUN.search(words) or _LANGUAGE.search(words))


# --- harness ------------------------------------------------------------------

# Phrasings that plainly need more than one step, one source, or one answer.
# The deterministic long-work net already claims most of these; this only adds
# the agentic shapes it deliberately leaves alone.
_HARNESS_CUES = re.compile(
    r"\b(?:research|look into|looking into|dig into|digging into|investigate|deep dive|"
    r"find out everything|browse the (?:web|internet)|search the (?:web|internet)|"
    r"go through (?:all|every|the last)|cross[- ]reference|"
    r"step by step|think (?:hard|carefully|it through)|several steps|"
    r"use (?:your|all your) tools|figure out where|work out what it would)\b"
)


def classify_request(text: object) -> RoutingDecision:
    """Read one turn offline and say which runtime should answer it.

    Never raises and never blocks: anything unusable is ordinary local
    conversation, which is what the assistant did before this module existed.
    """
    words = _normalized(text)
    if not words:
        return RoutingDecision(Destination.LOCAL, RoutingSource.LOCAL)
    control = deterministic_route(text)
    if control.route in (Route.CANCEL, Route.STATUS):
        # Cancelling and reporting on the queue are the queue own commands.
        return RoutingDecision(Destination.LOCAL, RoutingSource.CONTROL)
    if _is_coding(words):
        logger.debug("request routed to coding delegation")
        return RoutingDecision(Destination.CODING, RoutingSource.CODING)
    if control.route in (Route.WORK, Route.BACKGROUND):
        return RoutingDecision(Destination.HARNESS, RoutingSource.LONG_WORK)
    if _HARNESS_CUES.search(words):
        return RoutingDecision(Destination.HARNESS, RoutingSource.HARNESS)
    return RoutingDecision(Destination.LOCAL, RoutingSource.LOCAL)
