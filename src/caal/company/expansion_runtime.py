"""Bind cross-language company search to the *directly configured local model*.

:mod:`caal.company.query_expansion` never builds a client, names a model or
reads a setting: it takes an injected translator. This module is the one place
that decides which provider that translator is, and it decides fail-closed.

What it will not do, structurally rather than by convention:

* **never the routed provider.** A ``RoutedProvider`` may escalate, and a
  question rendered by an escalation is a private question sent to a cloud
  model. It is unwrapped to its ``primary`` and the routed object itself is
  never called;
* **never Hermes, never any non-local provider.** The unwrapped provider must
  name itself ``ollama``. Anything else -- Hermes, Groq, a stand-in, a wrapper
  that forgot to unwrap -- binds nothing;
* **never an unapproved endpoint.** The provider's own ``base_url`` is put
  through :func:`caal.local_ollama.normalize_endpoint`, the same narrow
  local-only allowance the settings UI enforces. A public address, an https
  URL, a URL with credentials or a path, or a host that is not a local alias
  or a private/loopback IP literal binds nothing;
* **never silently.** Every refusal is logged with its reason code. The
  endpoint itself is not logged: an address is infrastructure.

Binding nothing is a complete, correct outcome -- ``company.search`` then
behaves exactly as it did before the feature existed.

The binding is **session-scoped**. :func:`session_expansion` is entered around
one session and exited in its teardown, so two sessions in one process never
share, replace or unbind each other's translator, and a finished session leaves
no translator and no cached question behind.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from . import query_expansion

logger = logging.getLogger(__name__)

__all__ = [
    "LOCAL_PROVIDER_NAME",
    "SESSION_EXPANSION_ATTR",
    "bind_session_expansion",
    "local_provider",
    "local_translator",
    "rebind_session_expansion",
    "release",
    "session_expansion",
    "validated_endpoint",
]

#: The only provider a question may be rendered by.
LOCAL_PROVIDER_NAME = "ollama"

#: Where a session's own expander lives: on that session's agent, and nowhere
#: else. Not a module global and not a registry -- an entry can only re-bind
#: the expander of the agent it was handed, so two sessions in one process can
#: never reach each other's.
SESSION_EXPANSION_ATTR = "_company_expansion_binding"


@dataclass(frozen=True)
class _SessionBinding:
    """One session's expander, and the verified identity it was minted for."""

    expander: query_expansion.QueryExpander | None
    user_id: str | None


def _session_user_id(agent: Any) -> str | None:
    """The verified user of this session, read exactly where privacy reads it."""
    scope = getattr(agent, "_user_scope", None)
    user_id = getattr(scope, "user_id", None)
    return user_id if isinstance(user_id, str) and user_id else None


def local_provider(provider: Any) -> Any | None:
    """The direct local provider behind ``provider``, or ``None``.

    Unwraps the two wrappers the runtime hands around -- ``CAALLLM`` (LiveKit
    streaming signature) via ``provider_instance``, and ``RoutedProvider`` via
    ``primary`` -- exactly as :func:`voice_agent.work_router_provider` does,
    and then *checks* the result instead of trusting it. A routed provider
    reached here is not an error to report to a user; it is simply not the
    thing that may translate.
    """
    if provider is None:
        return None
    inner = getattr(provider, "provider_instance", provider)
    primary = getattr(inner, "primary", inner)
    # An unwrap that still leaves a router (or that produced a router) is a
    # wrapper this module does not know. Refuse rather than call it.
    if hasattr(primary, "primary") or hasattr(primary, "provider_instance"):
        logger.warning(
            "cross-language expansion: the provider did not unwrap to a direct local "
            "model (%s); not binding a translator",
            type(primary).__name__,
        )
        return None
    name = getattr(primary, "provider_name", None)
    if name != LOCAL_PROVIDER_NAME:
        logger.warning(
            "cross-language expansion: the configured model is %s, not the local one; "
            "not binding a translator",
            name or type(primary).__name__,
        )
        return None
    return primary


def validated_endpoint(provider: Any, *, settings: dict[str, Any] | None = None) -> str | None:
    """The provider's endpoint if it is an approved local one, else ``None``.

    ``base_url`` of ``None`` means the ollama client's own default, which is
    resolved the way the rest of the app resolves it
    (:func:`caal.local_ollama.configured_endpoint`) and then validated like any
    other. The endpoint is never logged, only its refusal code.
    """
    from caal import local_ollama

    raw = getattr(provider, "base_url", None)
    if not raw:
        raw = local_ollama.configured_endpoint(settings)
    try:
        return local_ollama.normalize_endpoint(raw)
    except local_ollama.EndpointError as exc:
        logger.warning(
            "cross-language expansion: the local model endpoint is not an approved "
            "local address (%s); not binding a translator",
            exc.code,
        )
        return None


def local_translator(
    provider: Any,
    *,
    settings: dict[str, Any] | None = None,
) -> query_expansion.Translate | None:
    """A translator on the validated local model, or ``None`` to bind nothing."""
    direct = local_provider(provider)
    if direct is None:
        return None
    if validated_endpoint(direct, settings=settings) is None:
        return None
    return query_expansion.provider_translator(direct)


def bind_session_expansion(
    provider: Any,
    *,
    owner: Any = None,
    authorize: query_expansion.Authorize | None = None,
    settings: dict[str, Any] | None = None,
    timeout_seconds: float = query_expansion.DEFAULT_TIMEOUT_SECONDS,
    label: str = "session",
) -> query_expansion.QueryExpander | None:
    """Bind for this session's context; the caller releases with ``release()``.

    The form a long-lived runtime uses, where the session body is not one
    ``with`` block. The binding still lives in the session's own context, so it
    is invisible to every other session; the caller calls :func:`release` in
    the same ``finally`` that tears the session down.

    ``owner`` is that session's agent. The expander is kept on it, together
    with the verified user id the session was minted for, so the turn and tool
    entries can re-bind it into whatever task they actually run in --
    :func:`rebind_session_expansion`. Binding at start alone is not enough: the
    SDK's room ingress tasks already exist by then and carry a context copied
    before this call, and a ContextVar written here never reaches them.

    Never raises: a session must start whatever this decides.
    """
    try:
        translate = local_translator(provider, settings=settings)
    except Exception as exc:  # noqa: BLE001 - a session starts either way
        logger.warning(
            "cross-language expansion could not be bound (%s); "
            "company search runs on the original query only",
            type(exc).__name__,
        )
        translate = None
    expander = query_expansion.bind_session(
        translate, timeout_seconds=timeout_seconds, authorize=authorize, label=label
    )
    if owner is not None:
        _remember(owner, expander)
    if expander is not None:
        logger.info(
            "Cross-language company search: bound to the local model (timeout %.1fs)",
            expander.timeout_seconds,
        )
    return expander


def _remember(owner: Any, expander: query_expansion.QueryExpander | None) -> None:
    """Keep this session's expander on its own agent, with its own identity."""
    try:
        setattr(owner, SESSION_EXPANSION_ATTR, _SessionBinding(expander, _session_user_id(owner)))
    except Exception:  # noqa: BLE001 - a read-only stand-in simply never expands
        logger.warning(
            "Could not attach the cross-language binding to the session; "
            "company search runs on the original query only"
        )


def rebind_session_expansion(agent: Any) -> query_expansion.QueryExpander | None:
    """Re-bind *this* session's expander into the calling task's context.

    Called at the turn and tool entries, the same shape and for the same reason
    as :func:`caal.company_privacy.begin_turn`: those entries run in tasks the
    SDK created -- some of them before the session bound anything -- and a
    ContextVar written at session start is only inherited by tasks created
    after it, in that same context. Typed RoomIO ingress is neither.

    The expander is read from the agent that was handed in, and its recorded
    identity must still be the session's verified identity, so a binding cannot
    be carried onto another session or survive the identity it was minted for.
    Anything unverified, unbound or closed **clears** the context rather than
    leaving whatever it inherited in place: the fail-closed outcome is the
    search that existed before this feature.
    """
    binding = getattr(agent, SESSION_EXPANSION_ATTR, None) if agent is not None else None
    if not isinstance(binding, _SessionBinding):
        return query_expansion.adopt(None)
    if binding.user_id != _session_user_id(agent):
        logger.warning(
            "cross-language expansion: the session identity no longer matches the binding; "
            "not expanding this turn"
        )
        return query_expansion.adopt(None)
    return query_expansion.adopt(binding.expander)


def release(owner: Any = None) -> None:
    """End this session's binding: close the expander and drop its questions.

    Resetting the ContextVar alone left a task that had inherited the handle
    holding a live expander after the session ended. Closing the object is what
    every holder of it sees, so nothing a finished session bound can translate
    again -- and the sessions still running are untouched, because the object
    closed is the one this session owned.
    """
    if owner is not None:
        binding = getattr(owner, SESSION_EXPANSION_ATTR, None)
        if isinstance(binding, _SessionBinding) and binding.expander is not None:
            binding.expander.close()
        try:
            setattr(owner, SESSION_EXPANSION_ATTR, None)
        except Exception:  # noqa: BLE001 - teardown never fails on a stand-in
            logger.debug("Could not clear the cross-language binding from the session")
    else:
        current = query_expansion.get_expander()
        if current is not None:
            current.close()
    query_expansion.reset()


@contextmanager
def session_expansion(
    provider: Any,
    *,
    authorize: query_expansion.Authorize | None = None,
    settings: dict[str, Any] | None = None,
    timeout_seconds: float = query_expansion.DEFAULT_TIMEOUT_SECONDS,
    label: str = "session",
) -> Iterator[query_expansion.QueryExpander | None]:
    """Bind cross-language expansion for one session, and release it afterwards.

    ``authorize`` is asked with the subject of each tool call, immediately
    before a question would be sent, so a session that is not the library owner
    never reaches the translator at all. Runtimes pass the same condition that
    offers the company tools.

    Never raises: a session must start whatever this decides.
    """
    try:
        translate = local_translator(provider, settings=settings)
    except Exception as exc:  # noqa: BLE001 - a session starts either way
        logger.warning(
            "cross-language expansion could not be bound (%s); "
            "company search runs on the original query only",
            type(exc).__name__,
        )
        translate = None
    with query_expansion.session_binding(
        translate,
        timeout_seconds=timeout_seconds,
        authorize=authorize,
        label=label,
    ) as expander:
        if expander is not None:
            logger.info(
                "Cross-language company search: bound to the local model (timeout %.1fs)",
                expander.timeout_seconds,
            )
        yield expander
