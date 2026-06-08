"""Native assistant tool registry.

The registry is intentionally lightweight: it describes tools, JSON schemas, and
confirmation policy independently from any specific LLM/provider adapter. Runtime
adapters can expose these definitions to Ollama, LiveKit, MCP, or future UIs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from caal.tools import calendar_tools, email_tools

ToolHandler = Callable[..., Any]


def _not_configured_handler(*_: Any, **__: Any) -> dict[str, str]:
    """Placeholder handler for tools whose concrete provider is not configured yet."""
    return {
        "status": "not_configured",
        "message": "This native assistant tool is registered but no provider is configured yet.",
    }


@dataclass(frozen=True)
class ToolDefinition:
    """Description and execution metadata for a native assistant tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    category: str
    requires_confirmation: bool = False
    handler: ToolHandler = field(default=_not_configured_handler, compare=False)


class ToolRegistry:
    """In-process catalog of native assistant tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def list(self) -> list[ToolDefinition]:
        return [self._tools[name] for name in self.names()]

    def by_category(self, category: str) -> list[ToolDefinition]:
        return [tool for tool in self.list() if tool.category == category]


def _object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def create_default_registry() -> ToolRegistry:
    """Create the default native assistant tool catalog."""
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="email.send",
            description="Send an email through a configured SMTP account.",
            category="email",
            requires_confirmation=True,
            parameters=_object_schema(
                {
                    "account": {"type": "string", "description": "Configured email account name."},
                    "to": {"type": "array", "items": {"type": "string"}},
                    "cc": {"type": "array", "items": {"type": "string"}},
                    "bcc": {"type": "array", "items": {"type": "string"}},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                    "confirmed": {
                        "type": "boolean",
                        "description": "True only after explicit user confirmation.",
                    },
                },
                ["account", "to", "subject", "body"],
            ),
            handler=email_tools.send_email,
        )
    )
    registry.register(
        ToolDefinition(
            name="email.search",
            description="Search mail in a configured IMAP mailbox.",
            category="email",
            parameters=_object_schema(
                {
                    "account": {"type": "string"},
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                ["account", "query"],
            ),
            handler=email_tools.search_email,
        )
    )
    registry.register(
        ToolDefinition(
            name="email.read",
            description="Read a message by provider/account message id.",
            category="email",
            parameters=_object_schema(
                {"account": {"type": "string"}, "message_id": {"type": "string"}},
                ["account", "message_id"],
            ),
            handler=email_tools.read_email,
        )
    )

    registry.register(
        ToolDefinition(
            name="calendar.list_events",
            description=(
                "List events from configured Google, Outlook, Zoho, Apple/iCloud, "
                "CalDAV, or ICS calendars."
            ),
            category="calendar",
            parameters=_object_schema(
                {
                    "source": {"type": "string", "description": "Calendar source name or 'all'."},
                    "start": {"type": "string", "description": "ISO-8601 start datetime."},
                    "end": {"type": "string", "description": "ISO-8601 end datetime."},
                },
                ["source", "start", "end"],
            ),
            handler=calendar_tools.list_calendar_events,
        )
    )
    registry.register(
        ToolDefinition(
            name="calendar.create_event",
            description="Create a calendar event on a writable configured calendar source.",
            category="calendar",
            requires_confirmation=True,
            parameters=_object_schema(
                {
                    "source": {"type": "string"},
                    "title": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "attendees": {"type": "array", "items": {"type": "string"}},
                    "location": {"type": "string"},
                    "notes": {"type": "string"},
                    "confirmed": {
                        "type": "boolean",
                        "description": "True only after explicit user confirmation.",
                    },
                },
                ["source", "title", "start", "end"],
            ),
            handler=calendar_tools.create_calendar_event,
        )
    )

    registry.register(
        ToolDefinition(
            name="reminders.create",
            description="Create a reminder using Apple Reminders or the local CAAL reminder store.",
            category="reminders",
            parameters=_object_schema(
                {
                    "title": {"type": "string"},
                    "due": {"type": "string", "description": "Optional ISO-8601 due datetime."},
                    "list": {"type": "string"},
                    "notes": {"type": "string"},
                },
                ["title"],
            ),
        )
    )
    registry.register(
        ToolDefinition(
            name="alarms.set",
            description="Set a local alarm or timer that JARVIS can announce when due.",
            category="alarms",
            parameters=_object_schema(
                {
                    "label": {"type": "string"},
                    "when": {
                        "type": "string",
                        "description": "ISO-8601 datetime or duration expression.",
                    },
                    "kind": {"type": "string", "enum": ["alarm", "timer"]},
                },
                ["label", "when", "kind"],
            ),
        )
    )

    return registry
