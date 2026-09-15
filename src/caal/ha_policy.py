"""Server-bound administrator authority for the existing unscoped harness.

HA grants never authorize delegation. Opaque capabilities are issued only by
trusted identity/worker dispatchers, never reconstructed from model payloads.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from weakref import WeakKeyDictionary


class _Authority:
    pass


_authorities = WeakKeyDictionary()
_dispatch_scope = ContextVar("delegation_dispatch_scope", default=None)


def verified_user_scope(profile, identity):
    """Called after the trusted dispatcher verified this session's identity."""
    from .user_scope import UserScope

    scope = UserScope.for_user(profile)
    current = identity.store.get_user(scope.user_id)
    if profile.role == "admin" and current and current.is_active and current.role == "admin":
        authority = _Authority()
        _authorities[authority] = (identity, scope.user_id)
        scope = replace(scope, delegation_authority=authority)
    return scope


def _administrator(scope):
    from .user_scope import UserScope

    if not isinstance(scope, UserScope):
        return False
    authority = scope.delegation_authority
    if not isinstance(authority, _Authority):
        return False
    binding = _authorities.get(authority)
    if binding is None or binding[1] != scope.user_id:
        return False
    identity, user_id = binding
    try:
        current = identity.store.get_user(user_id)
        return bool(current and current.is_active and current.role == "admin")
    except Exception:
        return False


def restricted_runtime():
    from .user_api import get_runtime

    # Invalid identity configuration must not open a delegation back door.
    try:
        from .security_config import load_multi_user_config

        return get_runtime() is not None or load_multi_user_config().attempted
    except Exception:
        return True


def require_delegated_scope(scope=None):
    effective = _dispatch_scope.get() or scope
    if not _administrator(effective):
        raise PermissionError("delegation_administrator_required")
    return effective


def tool_allowed(agent, name):
    if _administrator(getattr(agent, "_user_scope", None)):
        return True
    if name == "hass_assist":
        return True
    registry = getattr(agent, "_native_tool_registry", None)
    if registry is None:
        from .tools import create_default_registry

        registry = create_default_registry()
    if registry is not None and name in registry.names():
        return True
    # Class tools are an explicit local allowlist; external discovery never enters it.
    from .llm.agent_tools import resolve_agent_method_tool

    return resolve_agent_method_tool(agent, name) is not None


def bind_provider_scope(provider, scope):
    """Bind a session-owned provider graph; never called with model arguments."""
    from .llm.providers.hermes_provider import HermesProvider

    if isinstance(provider, HermesProvider):
        provider._delegation_scope = scope
    from .llm.providers.routed_provider import RoutedProvider

    if isinstance(provider, RoutedProvider):
        bind_provider_scope(provider.primary, scope)
        if provider.escalation is not None:
            bind_provider_scope(provider.escalation, scope)


@contextmanager
def task_dispatch_scope(task):
    """Authorize a persisted server-admitted task, independently of its caller."""
    from . import background_tasks, user_api

    scope = None
    with background_tasks._connect() as connection:
        row = background_tasks._fetch(connection, task.task_id)
        admission = connection.execute(
            "SELECT user_id FROM delegation_admissions WHERE task_id = ?", (task.task_id,)
        ).fetchone()
    if (
        row is None
        or admission is None
        or row.user_id != admission["user_id"]
        or row.user_id != task.user_id
        or row.request != task.request
        or row.session_key != task.session_key
        or row.status not in (background_tasks.QUEUED, background_tasks.RUNNING)
    ):
        raise PermissionError("delegation_task_not_authorized")
    identity = user_api.get_runtime()
    profile = identity.store.get_user(row.user_id) if identity is not None else None
    if profile is None or not profile.is_active or profile.role != "admin":
        raise PermissionError("delegation_administrator_required")
    scope = verified_user_scope(profile, identity)
    require_delegated_scope(scope)
    token = _dispatch_scope.set(scope)
    try:
        yield
    finally:
        # Child asyncio tasks inherit ContextVars. Ending this dispatch must
        # also invalidate their copy, without revoking the session capability.
        _authorities.pop(scope.delegation_authority, None)
        _dispatch_scope.reset(token)
