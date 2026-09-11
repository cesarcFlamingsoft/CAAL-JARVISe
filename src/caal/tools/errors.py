"""Tool failures that are safe to say out loud.

A tool that rejects an argument owes the user a sentence they can act on, and
owes the model nothing else: no traceback, no module path, no provider detail
and none of the stored text of the user. Anything raised as a
:class:`SafeToolError` carries exactly such a sentence; anything else escaping a
handler is reported to the model as a generic, contentless failure.
"""

from __future__ import annotations

from typing import Any

__all__ = ["SafeToolError", "safe_error_result"]


class SafeToolError(ValueError):
    """A validation failure whose message may be spoken to the user verbatim."""


def safe_error_result(error: SafeToolError, status: str = "invalid_request") -> dict[str, Any]:
    """The refusal a tool returns instead of raising into the turn."""
    return {"status": status, "message": str(error), "data": dict()}
