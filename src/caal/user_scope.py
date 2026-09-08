"""The user scope a session acts under, and how tools are bound to it.

Every session runs as exactly one of:

* **legacy**    multi-user identity is not configured; the deployment is the
  single-user JARVIS it always was and unscoped stores keep working;
* **anonymous** identity is configured but this session carries no verified
  user (a LAN client, a caller whose number matched nobody); user-scoped
  features are refused rather than guessed;
* **a user**    a verified, active user named only by an opaque id.

Tool arguments produced by the LLM are untrusted: a user-scoped tool never
takes its scope from them. :func:`scoped_tool_arguments` strips any ``user_id``
the model may have invented and binds the session's own scope instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MEMORY_UNAVAILABLE_REPLY",
    "USER_ID_PATTERN",
    "UserScope",
    "is_valid_user_id",
    "memory_unavailable_result",
    "require_user_id",
    "scoped_tool_arguments",
]

USER_ID_PATTERN = re.compile(r"^usr_[0-9a-f]{24}$")
SCOPE_ARGUMENT = "user_id"

MEMORY_UNAVAILABLE_REPLY = (
    "I can only save or recall memories for a signed-in user profile, and this session "
    "isn't signed in, so I haven't stored or looked anything up."
)


def is_valid_user_id(value: object) -> bool:
    """Whether ``value`` has the exact opaque shape the user store issues."""
    return isinstance(value, str) and USER_ID_PATTERN.fullmatch(value) is not None


def require_user_id(value: object) -> str:
    if not is_valid_user_id(value):
        raise ValueError("User id has an invalid shape")
    return value  # type: ignore[return-value]


@dataclass(frozen=True)
class UserScope:
    """Who a session acts for. The display name never prints."""

    user_id: str | None
    identity_configured: bool
    display_name: str | None = field(default=None, repr=False)
    role: str | None = None

    @classmethod
    def legacy(cls) -> UserScope:
        return cls(user_id=None, identity_configured=False)

    @classmethod
    def anonymous(cls) -> UserScope:
        return cls(user_id=None, identity_configured=True)

    @classmethod
    def for_user(cls, profile: Any) -> UserScope:
        return cls(
            user_id=require_user_id(getattr(profile, "user_id", None)),
            identity_configured=True,
            display_name=getattr(profile, "display_name", None),
            role=getattr(profile, "role", None),
        )

    @property
    def is_identified(self) -> bool:
        return self.user_id is not None

    @property
    def memory_available(self) -> bool:
        """Legacy sessions keep the single-user store; anonymous ones get nothing."""
        return self.user_id is not None or not self.identity_configured


def memory_unavailable_result() -> dict[str, Any]:
    return {"status": "unauthorized", "message": MEMORY_UNAVAILABLE_REPLY, "data": {}}


def scoped_tool_arguments(
    tool: Any, arguments: dict[str, Any], scope: UserScope | None
) -> dict[str, Any] | None:
    """Bind a tool call to the session scope; ``None`` means the call must be refused.

    Unscoped tools pass through unchanged (as a copy). For a user-scoped tool
    the model-supplied ``user_id`` is always discarded, the session's own id is
    attached, and an anonymous session under multi-user gets ``None``.
    """
    bound = {name: value for name, value in dict(arguments or {}).items()}
    if not getattr(tool, "user_scoped", False):
        return bound
    bound.pop(SCOPE_ARGUMENT, None)
    effective = scope if scope is not None else UserScope.legacy()
    if not effective.memory_available:
        return None
    bound[SCOPE_ARGUMENT] = effective.user_id
    return bound
