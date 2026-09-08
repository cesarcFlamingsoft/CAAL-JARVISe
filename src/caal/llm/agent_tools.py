"""Explicit allowlist for agent-method tools.

Tool names arrive from LLM output. Resolving them with ``getattr`` on the agent
would expose every public attribute of the agent (``aclose``, ``llm_node``,
``say``, ...) as a callable tool, so dispatch is restricted to the small set of
``@function_tool`` methods the agent actually publishes.
"""

from __future__ import annotations

from collections.abc import Callable

__all__ = ["AGENT_METHOD_TOOLS", "resolve_agent_method_tool"]

# Names of @function_tool decorated methods defined on the agent classes.
AGENT_METHOD_TOOLS: frozenset[str] = frozenset({"web_search"})


def resolve_agent_method_tool(agent: object, tool_name: str) -> Callable[..., object] | None:
    """Return the agent's handler for ``tool_name``, or None if not allowlisted."""
    if tool_name not in AGENT_METHOD_TOOLS:
        return None
    handler = getattr(agent, tool_name, None)
    return handler if callable(handler) else None
