"""Whether this conversation may leave the machine at all.

The company library holds the owner's policies, HR documents and contracts.
Redacting a *tool result* on the way to the cloud agent runtime does not
protect them, because the question is the leak: "what does our contract with
FIXTURE Corp say about severance for FIXTURE Person" is already the sensitive
part, and the router reads that question and picks a destination **before** any
company tool has run. Detecting the sensitive ones by their wording would be a
guess, and a guess that fails closed only most of the time is not a
confidentiality control.

So this module does not classify questions into "safe" and "unsafe". It
decides, per session and before anything is routed, whether the session is
**local-only**: no escalation to the Hermes agent runtime, no coding
delegation, no durable/harness admission, no external tool surface, and no
fallback to any of them when the local model fails. A local-only session that
cannot answer says so; it does not quietly find another runtime that can.

And it is **query only**. The initial company-private mode offers the two
read-only company tools and nothing else: no memory or reminder or alarm
write, no scheduling, no deterministic local route that schedules, sends or
hands the call off. See :data:`PRIVATE_SESSION_TOOLS` and
:data:`PRIVATE_SESSION_BLOCKED_ROUTES` for why each of those was not safe to
keep -- briefly, a reminder is a deferred send and memory outlives the
session, so both cross a boundary that is defined per session. **No ordinary
session loses anything**: every one of these capabilities is untouched outside
a company session, and the UI states that a company session can only answer
questions about the library.

``CAAL_COMPANY_PRIVATE_MODE`` chooses the policy:

``explicit`` (default)
    an ordinary session of the owner is an ordinary JARVIS session -- Home
    Assistant, the connected mail and calendar accounts, network lookups, the
    agent harness, everything -- and is offered **no company tool at all**. A
    company-private session is entered deliberately from the UI; the request
    rides in the same server-signed, room-bound principal that names the user,
    so a page can ask for one and nothing in a browser can forge one, and this
    module still checks the user against the provisioned library owner.
    Leaving is ending the session: the next one does not carry the flag.

``owner_session``
    every session of the provisioned owner is local-only from its first word.
    Maximal confidentiality, and it costs the owner every networked capability
    in every session, which is why it is no longer the default. An operator who
    wants it must choose it, and the Company Library page says what it does.

``off``
    no local-only mode -- and therefore **no company tools are offered at
    all**, because there would be nothing keeping their contents out of the
    escalation path.

Engagement is sticky and one-way within a session: it never lifts, and nothing
a turn says can grant it. The way back to ordinary cloud-assisted JARVIS is a
fresh session.

The wording of a turn is read for exactly one purpose, and only in the safe
direction. In an *ordinary* owner session, an offline phrase list recognises
company business and pins that one turn local -- it is not routed out, no
company tool becomes available, and the owner is told to open a company
session. The phrase list is deliberately incomplete and says so; it is a
courtesy, not a boundary. The boundary is that company tools exist only inside
a company session, which is why the UI states that company questions must be
asked there.

Nothing here logs the words of a turn.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "ALLOWED_TOOL_CATEGORIES",
    "COMPANY_MODE_NOTICE",
    "LOCAL_ONLY_NO_ANSWER",
    "MODE_EXPLICIT",
    "MODE_OFF",
    "MODE_OWNER_SESSION",
    "PRIVATE_SESSION_BLOCKED_ROUTES",
    "PRIVATE_SESSION_TOOLS",
    "CompanyModeGate",
    "SessionPrivacy",
    "begin_turn",
    "bind_session",
    "company_intent",
    "company_mode_notice",
    "company_tools_offered",
    "current",
    "is_local_only",
    "local_route_allowed",
    "note_company_tool_use",
    "private_mode",
    "private_session_directive",
    "PRIVATE_SESSION_DIRECTIVE",
    "reset",
    "session_is_private",
    "tool_allowed_in_private_session",
]

MODE_EXPLICIT = "explicit"
MODE_OWNER_SESSION = "owner_session"
MODE_OFF = "off"
_MODES = (MODE_EXPLICIT, MODE_OWNER_SESSION, MODE_OFF)

#: The attribute a session's verified entry request is bound to on the agent.
SESSION_REQUEST_ATTR = "_company_session_requested"

# A company-private session is **query only**, and the surface is a list of
# tool *names*, not of categories.
#
# The first attempt at this allowed the ``company``, ``memory``, ``reminders``
# and ``alarms`` categories, on the reasoning that each of them keeps its bytes
# on this machine. An independent probe took that apart twice:
#
# * a reminder is a **deferred send**. ``reminders.create`` accepts a
#   ``delivery`` list, and ``telegram`` or ``call`` on that list means an
#   outbound API call made later, by the worker process, from text the model
#   chose. The probe put a canary in a reminder title inside an engaged private
#   session and watched it arrive at a captured Telegram sink through the real
#   handler and the real dispatcher. "There is no tool here that sends" was
#   false; the deferral was what hid it;
# * ``memory`` and ``reminders`` are **persistence**, and the confidentiality
#   boundary is per session. A clause quoted out of an HR document and written
#   to memory in a private session is read back in an ordinary session, where
#   the escalation path is wide open. Anything that outlives the session
#   crosses the boundary on its own schedule.
#
# So the initial mode allows the two read-only company tools and nothing else.
# It is deliberately the smallest thing that answers the question the mode
# exists for, and it is named rather than derived: a company tool added later
# that *writes* is refused here until someone decides otherwise.
PRIVATE_SESSION_TOOLS = frozenset({"company.search", "company.read"})

#: The one category those tools live in. Kept as a second, coarser check so a
#: name collision cannot smuggle a non-company tool onto the surface.
ALLOWED_TOOL_CATEGORIES = frozenset({"company"})

# The local-command routes a private session is not offered.
#
# These run *beside* the model, in the agent's own turn handling, so narrowing
# the native tool surface does not touch them: a deterministic "remind me at
# four" route creates the same reminder the tool would have. Each name here
# schedules something, sends something, or hands the call to another runtime.
#
# It is an explicit deny list rather than an allow list because the handler has
# routes that are purely local and purely conversational, and a private session
# should keep those. Adding a route that schedules or sends means adding its
# name here; the test suite checks the set by value so that is a visible change.
PRIVATE_SESSION_BLOCKED_ROUTES = frozenset({
    # Answers "call me" / "message me" for a pending reminder: arms a channel.
    "delivery_answer",
    # Creates, changes and cancels reminders and alarms from plain speech.
    "scheduled_change",
    # Hands the live call to the Hermes runtime.
    "phone_handoff",
    # Reads the connected mail and calendar accounts, and can refresh them
    # from the provider over the network.
    "connected_knowledge",
    # Queues work for the separate worker process, which composes on the
    # escalation provider.
    "durable_work",
    # Arms an outbound call back to the user after the call ends.
    "end_call_callback",
})

LOCAL_ONLY_NO_ANSWER = (
    "I couldn't answer that with the local model, and this is a company-private session, "
    "so I won't send it anywhere else. Start a new session if you want me to use my other "
    "runtimes for this."
)

# The one instruction a company-private session adds to the agent's prompt.
#
# Without it the model was given the two company tools and no idea why: asked a
# plain question about the owner's own uploaded documents it answered with a
# general refusal about privacy and personal information, having never called a
# tool. That refusal is not a control -- the control is the session boundary and
# the read-only surface, both of which hold -- it is just a wrong answer, and it
# left the owner unable to read their own library.
#
# It adds to the prompt and never replaces it, it grants nothing (the tools are
# offered by :func:`company_tools_offered`, not by this text), and it asks for
# no behaviour outside the session it is injected into.
PRIVATE_SESSION_DIRECTIVE = (
    "This is a company-private session. The signed-in owner's own uploaded documents -- "
    "policies, HR files, contracts, directories, notes -- are open to you through the "
    "company.search tool, and nothing in this session leaves this machine. When they ask "
    "anything a document could answer, including what a policy or contract says and "
    "anything written about a named person, call company.search first, using their own "
    "words as the query and putting the name in the person argument when they named "
    "someone. Do not refuse on privacy or confidentiality grounds, do not ask for "
    "permission, and do not say you have no access: this is the owner asking about their "
    "own library. Answer only from the passages the tool returns, naming the document and "
    "the place inside it, and repeat any caveat the result carries about an unidentified "
    "person. Never say the library has no record of someone, or nobody by that name, "
    "unless a company.search you actually ran came back saying so; if a tool call fails or "
    "a tool name does not exist, say the lookup failed and try company.search instead."
)

COMPANY_MODE_NOTICE = (
    "That sounds like company business, so I've kept it on this machine and haven't "
    "answered it. Company documents are only readable in a company session: open the "
    "Company Library page and start one, and I'll look it up there. I can't recognise "
    "every way of asking, so please start a company session before you say anything "
    "confidential."
)

# A deliberately small, deliberately literal phrase list. It is not a
# classifier, it grants nothing, and its only effect is to keep a turn on this
# machine and print the notice above.
_COMPANY_INTENT = re.compile(
    r"\b(?:our|the|company|corporate|internal|staff|employee|employment|hr)\s+"
    r"(?:policy|policies|handbook|contract|contracts|agreement|agreements|"
    r"document|documents|paperwork|file|files|record|records|terms)\b"
    r"|\bcompany library\b"
    r"|\b(?:what|does|do)\s+(?:our|the company)\b"
    r"|\b(?:severance|non[- ]?compete|nda|offer letter|payroll|salary band|"
    r"disciplinary|grievance|onboarding policy|leave policy|expense policy)\b",
    re.IGNORECASE,
)


@dataclass
class SessionPrivacy:
    """One session's answer to "may this leave the machine?". Sticky once true."""

    engaged: bool = False
    reason: str = "not_engaged"

    def engage(self, reason: str) -> None:
        if not self.engaged:
            self.engaged = True
            self.reason = reason
            # The reason is a policy name. It names no user and no words.
            logger.info("A session became company-private (%s)", reason)


_lock = threading.Lock()
_current: ContextVar[SessionPrivacy | None] = ContextVar("company_privacy", default=None)
# Set for the duration of one turn when an ordinary owner session said
# something the phrase list recognised. Never grants access to anything.
_turn_pin: ContextVar[bool] = ContextVar("company_privacy_turn_pin", default=False)
_mode_override: str | None = None


def private_mode() -> str:
    """The configured policy. An unknown value is the default, not a weaker one."""
    with _lock:
        override = _mode_override
    raw = (override or os.environ.get("CAAL_COMPANY_PRIVATE_MODE") or "").strip().lower()
    if raw in _MODES:
        return raw
    if raw:
        logger.warning("CAAL_COMPANY_PRIVATE_MODE is not a known mode; using the default")
    return MODE_EXPLICIT


def set_mode_for_tests(mode: str | None) -> None:
    """Override the configured policy in this process. Tests and the harness only."""
    global _mode_override
    if mode is not None and mode not in _MODES:
        raise ValueError("unknown company private mode")
    with _lock:
        _mode_override = mode


def reset() -> None:
    """Forget this context's session state and any mode override."""
    global _mode_override
    _current.set(None)
    _turn_pin.set(False)
    with _lock:
        _mode_override = None


def current() -> SessionPrivacy | None:
    return _current.get()


def session_is_private() -> bool:
    """Whether *the session* is company-private. Only this unlocks company tools."""
    state = _current.get()
    return bool(state is not None and state.engaged)


def is_local_only() -> bool:
    """Whether this turn must stay on this machine.

    True for every turn of a company-private session, and for a single pinned
    turn of an ordinary session. Read by every routing and escalation decision.
    """
    return session_is_private() or bool(_turn_pin.get())


def company_intent(text: object) -> bool:
    """A deterministic, incomplete reading of "this is about the company".

    Local, offline, and never sent anywhere: recognising the intent must not
    itself be the thing that leaks the question.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    return bool(_COMPANY_INTENT.search(text[:600]))


def _configured_owner() -> str | None:
    """The provisioned company owner, or ``None`` when there is no library."""
    try:
        from caal.company import runtime

        config = runtime.get_config()
    except Exception:  # noqa: BLE001 - an unconfigured deployment is simply not private
        return None
    return getattr(config, "owner_user_id", None) if config is not None else None


def _agent_user_id(agent: Any) -> str | None:
    scope = getattr(agent, "_user_scope", None)
    user_id = getattr(scope, "user_id", None)
    return user_id if isinstance(user_id, str) and user_id else None


def _is_owner(agent: Any) -> bool:
    owner = _configured_owner()
    return owner is not None and _agent_user_id(agent) == owner


def bind_session(agent: Any, *, requested: bool) -> None:
    """Record this session's **verified** company-mode request on the agent.

    Called once, at session start, from the value the worker read out of the
    signed room-bound principal. Nothing a model or a turn produces reaches
    here, and clearing it later does not lift a mode that already engaged.
    """
    try:
        setattr(agent, SESSION_REQUEST_ATTR, bool(requested))
    except Exception:  # noqa: BLE001 - a read-only stand-in simply stays ordinary
        logger.warning("Could not bind the company session request; session stays ordinary")


def company_tools_offered(user_id: object) -> bool:
    """Whether this session may see the company tools at all.

    Structural, not advisory: the tools exist only inside an engaged
    company-private session owned by the provisioned owner. An ordinary
    session -- including a turn pinned local by the phrase list -- is not
    offered them, so "ask the model nicely" is not a way in.
    """
    if private_mode() == MODE_OFF or not session_is_private():
        return False
    owner = _configured_owner()
    return owner is not None and isinstance(user_id, str) and user_id == owner


def private_session_directive(agent: Any) -> str | None:
    """The system instruction this session adds, or ``None`` for every other one.

    Tied to the same condition that offers the tools, so a session that cannot
    call the library is never told that it can. Read after
    :func:`begin_turn` has bound the turn.
    """
    scope = getattr(agent, "_user_scope", None)
    user_id = getattr(scope, "user_id", None)
    return PRIVATE_SESSION_DIRECTIVE if company_tools_offered(user_id) else None


def begin_turn(agent: Any, text: object = None) -> SessionPrivacy:
    """Bind this turn to its session's privacy state, before anything is routed.

    The state lives on the agent, so it is sticky across the turns of one
    session, and is mirrored into a context variable for the turn so the
    routing functions -- which are given words, not sessions -- can see it.
    """
    state = getattr(agent, "_company_privacy", None)
    if not isinstance(state, SessionPrivacy):
        state = SessionPrivacy()
        try:
            agent._company_privacy = state
        except Exception:  # noqa: BLE001 - a read-only stand-in still gets this turn
            pass
    _current.set(state)
    _turn_pin.set(False)

    mode = private_mode()
    if mode == MODE_OFF or not _is_owner(agent):
        # Not the company owner, or no library to protect. This session has no
        # company access, so it has nothing to give away.
        return state
    if mode == MODE_OWNER_SESSION:
        state.engage(MODE_OWNER_SESSION)
    elif getattr(agent, SESSION_REQUEST_ATTR, False) is True:
        state.engage("explicit_entry")
    elif not state.engaged and company_intent(text):
        # An ordinary session that wandered into company wording. Keep this one
        # turn here and tell them where to ask it properly. This grants nothing.
        _turn_pin.set(True)
    return state


def company_mode_notice(agent: Any, text: object = None) -> str | None:
    """The answer an ordinary owner session gets when it asks company business.

    ``None`` whenever there is nothing to say: a session that is already
    private answers properly, and anyone who is not the owner has no company
    library in play at all.
    """
    if session_is_private() or private_mode() == MODE_OFF:
        return None
    if not _is_owner(agent):
        return None
    if not (_turn_pin.get() or company_intent(text)):
        return None
    return COMPANY_MODE_NOTICE


def note_company_tool_use() -> None:
    """Engage the current session because a company tool actually ran.

    A backstop, not a route in: the company tools are only reachable from an
    already-private session, so in practice this re-engages what is engaged.
    If that ever stops being true, the conversation still stays here.
    """
    state = _current.get()
    if state is not None:
        state.engage("company_tool_used")


def tool_allowed_in_private_session(category: object, name: object) -> bool:
    """Whether a local-only company session may be offered this native tool.

    Both checks have to pass: the tool must be one of the named read-only
    company tools, *and* it must still be in the ``company`` category. The
    name list is the surface; the category is a second gate so a tool
    registered later under one of these names cannot inherit the permission.
    """
    if not isinstance(name, str) or name not in PRIVATE_SESSION_TOOLS:
        return False
    return isinstance(category, str) and category in ALLOWED_TOOL_CATEGORIES


def local_route_allowed(route: object) -> bool:
    """Whether this local-command route may run for the current turn.

    ``True`` for every turn that is not local-only, and for every route nobody
    named: this is not a kill switch for the turn handler, it is a list of the
    routes that schedule, send or hand off. A route that is blocked simply does
    not see the turn, which leaves the turn to the local model.
    """
    if not is_local_only():
        return True
    return not (isinstance(route, str) and route in PRIVATE_SESSION_BLOCKED_ROUTES)


class CompanyModeGate:
    """Answers a company-shaped turn in an ordinary session, and consumes it.

    Wired into the agent's local-command chain ahead of everything that could
    route, so the turn is spoken to and never reaches a model, a provider or a
    destination. It answers only what :func:`company_mode_notice` answers, so
    an already-private session passes straight through it.
    """

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    async def handle(self, text: str, session: Any) -> bool:
        notice = company_mode_notice(self._agent, text)
        if notice is None:
            return False
        if session is not None:
            try:
                await session.say(notice)
            except Exception:  # noqa: BLE001 - never echo the turn into a log
                logger.error("Could not deliver the company-session notice")
        return True
