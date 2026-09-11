"""Native assistant tool registry.

The registry is intentionally lightweight: it describes tools, JSON schemas, and
confirmation policy independently from any specific LLM/provider adapter. Runtime
adapters can expose these definitions to Ollama, LiveKit, MCP, or future UIs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from caal.tools import (
    alarms_tools,
    calendar_tools,
    email_tools,
    knowledge_tools,
    memory_tools,
    reminder_delivery,
    reminders_tools,
)

ToolHandler = Callable[..., Any]

# Shared parameter descriptions of the connected-account knowledge tools.
_ACCOUNT_HINT = (
    "Optional account filter, in the user own words: the name or alias they gave a linked "
    "account (work, university, Vertex), the email address of the account, or a provider "
    "name: google, microsoft or zoho. Pass the words the user said and nothing else. Omit "
    "it to read every connected account. If the name matches nothing the answer says so; "
    "if it matches several accounts the answer asks which one is meant."
)
_DAY_HINT = (
    "Optional single day instead of a range: today, tonight, tomorrow, the day after "
    "tomorrow, a weekday name, or a date as YYYY-MM-DD. Use this for a question about one "
    "named day; use days for a span such as this week."
)


_WHEN_DESCRIPTION = (
    "When it is due. Prefer an ISO-8601 duration from now, such as PT2M for two minutes, "
    "PT30S, PT1H30M or P1D; a plain duration such as 10m or 2 hours also works. For a time "
    "of day give a full timestamp that includes the timezone offset, such as "
    "2026-09-09T18:30:00-06:00. A timestamp without an offset is refused, and so is a time "
    "already past."
)


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
    # A user-scoped tool receives the session's verified ``user_id`` from the
    # runtime (see caal.user_scope), never from the model's arguments.
    user_scoped: bool = False


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
            name="calendar.find_free_time",
            description="Find unoccupied calendar slots that fit a requested duration.",
            category="calendar",
            parameters=_object_schema(
                {
                    "source": {"type": "string", "description": "Calendar source name or 'all'."},
                    "start": {"type": "string", "description": "ISO-8601 start datetime."},
                    "end": {"type": "string", "description": "ISO-8601 end datetime."},
                    "duration_minutes": {"type": "integer", "minimum": 1},
                },
                ["source", "start", "end", "duration_minutes"],
            ),
            handler=calendar_tools.find_free_time,
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
            name="calendar.update_event",
            description="Update a CalDAV calendar event after explicit confirmation.",
            category="calendar",
            requires_confirmation=True,
            parameters=_object_schema(
                {
                    "source": {"type": "string"},
                    "event_id": {"type": "string"},
                    "title": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "attendees": {"type": "array", "items": {"type": "string"}},
                    "location": {"type": "string"},
                    "notes": {"type": "string"},
                    "confirmed": {"type": "boolean"},
                },
                ["source", "event_id", "title", "start", "end"],
            ),
            handler=calendar_tools.update_calendar_event,
        )
    )
    registry.register(
        ToolDefinition(
            name="calendar.delete_event",
            description="Delete a CalDAV calendar event after explicit confirmation.",
            category="calendar",
            requires_confirmation=True,
            parameters=_object_schema(
                {
                    "source": {"type": "string"},
                    "event_id": {"type": "string"},
                    "confirmed": {"type": "boolean"},
                },
                ["source", "event_id"],
            ),
            handler=calendar_tools.delete_calendar_event,
        )
    )

    registry.register(
        ToolDefinition(
            name="memory.remember",
            description="Save an explicit preference or fact for future assistant sessions.",
            category="memory",
            parameters=_object_schema(
                {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                ["key", "value"],
            ),
            handler=memory_tools.remember,
            user_scoped=True,
        )
    )
    registry.register(
        ToolDefinition(
            name="memory.recall",
            description="Recall one saved preference, or list all saved preferences.",
            category="memory",
            parameters=_object_schema({"key": {"type": "string"}}),
            handler=memory_tools.recall,
            user_scoped=True,
        )
    )

    # Knowledge tools: answers about the connected (OAuth-linked) email and
    # calendar accounts of the signed-in user, from a bounded per-user index that
    # is refreshed from the providers when it is stale. Distinct from the
    # settings-configured IMAP/ICS tools above. Never a full body.
    registry.register(
        ToolDefinition(
            name="inbox.recent",
            description=(
                "Summarize the newest emails across the connected email accounts of the "
                "signed-in user (Google, Microsoft or Zoho linked under Settings). Use for: "
                "any new email, what is in my inbox, unread mail, recent messages. Returns "
                "short safe summaries (sender, subject, preview, when), never full bodies."
            ),
            category="knowledge",
            parameters=_object_schema(
                dict(
                    limit=dict(
                        type="integer",
                        minimum=1,
                        maximum=25,
                        description="How many recent emails to return; default 5.",
                    ),
                    unread_only=dict(
                        type="boolean", description="True to count and list unread mail only."
                    ),
                    account=dict(type="string", description=_ACCOUNT_HINT),
                )
            ),
            handler=knowledge_tools.recent_email,
            user_scoped=True,
        )
    )
    registry.register(
        ToolDefinition(
            name="inbox.search",
            description=(
                "Search recent email in the connected accounts of the signed-in user by "
                "sender name, subject words or preview words. Use for: did I get an email "
                "from X, anything about Y, find the message about Z. Only when the user "
                "named something to search for: for unread, new or recent mail, or for a "
                "request that only names an account (show unread email for my University "
                "account), use inbox.recent with unread_only and account instead."
            ),
            category="knowledge",
            parameters=_object_schema(
                dict(
                    query=dict(
                        type="string",
                        description="A sender name, subject words, or a few key words.",
                    ),
                    limit=dict(type="integer", minimum=1, maximum=25),
                    account=dict(type="string", description=_ACCOUNT_HINT),
                ),
                ["query"],
            ),
            handler=knowledge_tools.search_email,
            user_scoped=True,
        )
    )
    registry.register(
        ToolDefinition(
            name="inbox.read_summary",
            description=(
                "Read one email aloud as a safe summary: sender, subject, when it arrived, "
                "read state and the short preview the provider supplies. Never the full body. "
                "Pick the email by id from an earlier result, by a few words, or leave both "
                "empty for the newest email in the connected accounts."
            ),
            category="knowledge",
            parameters=_object_schema(
                dict(
                    message_id=dict(
                        type="string", description="The id of an email from an earlier result."
                    ),
                    query=dict(
                        type="string", description="Words from the sender, subject or preview."
                    ),
                    account=dict(type="string", description=_ACCOUNT_HINT),
                )
            ),
            handler=knowledge_tools.read_email_summary,
            user_scoped=True,
        )
    )
    registry.register(
        ToolDefinition(
            name="schedule.upcoming",
            description=(
                "Summarize a window of the connected calendars of the signed-in user "
                "(Google, Microsoft or Zoho linked under Settings). Use for a period of "
                "time: what is on my calendar today or tomorrow, what do I have this week, "
                "anything on Friday. For the next, nearest or soonest event use "
                "schedule.next instead. Events come back sorted by start time, earliest "
                "first; ask for a small limit and speak only what was asked for."
            ),
            category="knowledge",
            parameters=_object_schema(
                dict(
                    days=dict(
                        type="integer",
                        minimum=1,
                        maximum=31,
                        description="How many days ahead from now; default 7.",
                    ),
                    day=dict(type="string", description=_DAY_HINT),
                    limit=dict(
                        type="integer",
                        minimum=1,
                        maximum=25,
                        description="How many events to return, soonest first; default 8.",
                    ),
                    account=dict(type="string", description=_ACCOUNT_HINT),
                    only_future=dict(
                        type="boolean",
                        description=(
                            "True to drop what is already over, for what is left of a day: "
                            "the rest of today, what do I still have this afternoon."
                        ),
                    ),
                )
            ),
            handler=knowledge_tools.upcoming_schedule,
            user_scoped=True,
        )
    )
    registry.register(
        ToolDefinition(
            name="schedule.next",
            description=(
                "The event that starts next on the connected calendars of the signed-in "
                "user, sorted soonest first. Use for: what is my next event, what is coming "
                "up next, my next or nearest or soonest meeting, what is my next "
                "appointment, the next thing on my calendar. It returns exactly one event "
                "unless a larger limit is asked for, never anything in the past, and never "
                "an event already under way. For a whole day or week (today, tomorrow, this "
                "week) use schedule.upcoming instead."
            ),
            category="knowledge",
            parameters=_object_schema(
                dict(
                    limit=dict(
                        type="integer",
                        minimum=1,
                        maximum=5,
                        description=(
                            "How many of the soonest events to return; default 1. Use more "
                            "only when the user asked for several, as in my next three "
                            "meetings or the next few things."
                        ),
                    ),
                    days=dict(
                        type="integer",
                        minimum=1,
                        maximum=31,
                        description="How far ahead to look for one; default 31 days.",
                    ),
                    account=dict(type="string", description=_ACCOUNT_HINT),
                )
            ),
            handler=knowledge_tools.next_events,
            user_scoped=True,
        )
    )
    registry.register(
        ToolDefinition(
            name="schedule.find_event",
            description=(
                "Check whether an event exists on the connected calendars of the signed-in "
                "user and say when it is. Use for: do I have a dentist appointment, when is "
                "my meeting with X, is there anything called Y this week."
            ),
            category="knowledge",
            parameters=_object_schema(
                dict(
                    query=dict(
                        type="string", description="Words from the event title or location."
                    ),
                    days=dict(
                        type="integer",
                        minimum=1,
                        maximum=31,
                        description="How many days ahead to look; default 14.",
                    ),
                    day=dict(type="string", description=_DAY_HINT),
                    account=dict(type="string", description=_ACCOUNT_HINT),
                ),
                ["query"],
            ),
            handler=knowledge_tools.find_event,
            user_scoped=True,
        )
    )

    _DELIVERY_SCHEMA = {
        "type": "array",
        "items": {"type": "string", "enum": list(reminder_delivery.CHANNEL_ARGUMENTS)},
        "description": reminders_tools.DELIVERY_DESCRIPTION,
    }
    registry.register(
        ToolDefinition(
            name="reminders.create",
            description=(
                "Create a reminder in the local CAAL reminder store of the signed-in user. "
                "Use for: remind me to X, add X to my list, do not let me forget X. Give due "
                "when the user said a time, and CAAL announces the reminder once when it comes "
                "due; leave due out for an open list item, which is saved but never announced. "
                "This is the local store only; it does not write to Apple Reminders or any "
                "other outside service."
            ),
            category="reminders",
            parameters=_object_schema(
                {
                    "title": {
                        "type": "string",
                        "description": "What to be reminded about, in the words of the user.",
                    },
                    "due": {"type": "string", "description": _WHEN_DESCRIPTION},
                    "list": {"type": "string", "description": "Optional list name."},
                    "notes": {"type": "string", "description": "Optional extra detail."},
                    "delivery": _DELIVERY_SCHEMA,
                },
                ["title"],
            ),
            handler=reminders_tools.create_reminder,
            user_scoped=True,
        )
    )
    registry.register(
        ToolDefinition(
            name="reminders.set_delivery",
            description=(
                "Set how the signed-in user wants to be told about the timed reminder they "
                "just set, after CAAL asked them which ways they want. Use for their answer: "
                "all of them, call and message me, just tell me here, Telegram only. It "
                "applies to their own most recent reminder still to come; there is no way to "
                "name a different one, a phone number or a chat."
            ),
            category="reminders",
            parameters=_object_schema({"delivery": _DELIVERY_SCHEMA}, ["delivery"]),
            handler=reminders_tools.set_delivery,
            user_scoped=True,
        )
    )
    registry.register(
        ToolDefinition(
            name="reminders.list",
            description=(
                "List the saved reminders of the signed-in user, soonest due first and "
                "undated ones last. Use for: what are my reminders, what is on my list."
            ),
            category="reminders",
            parameters=_object_schema(
                {
                    "include_completed": {
                        "type": "boolean",
                        "description": "True to include reminders already marked done.",
                    }
                }
            ),
            handler=reminders_tools.list_reminders,
            user_scoped=True,
        )
    )
    registry.register(
        ToolDefinition(
            name="alarms.set",
            description=(
                "Set a local alarm or timer for the signed-in user that CAAL announces out "
                "loud, once, when it comes due. Use for: wake me at X, set a timer for X "
                "minutes, alarm in X. Nothing is sent anywhere: the alarm is stored locally "
                "and announced in a live session of the same user."
            ),
            category="alarms",
            parameters=_object_schema(
                {
                    "label": {
                        "type": "string",
                        "description": "A short name for the alarm, in the words of the user.",
                    },
                    "when": {"type": "string", "description": _WHEN_DESCRIPTION},
                    "kind": {
                        "type": "string",
                        "enum": ["alarm", "timer"],
                        "description": (
                            "timer for a countdown the user named a length for, alarm for a "
                            "time of day."
                        ),
                    },
                },
                ["label", "when", "kind"],
            ),
            handler=alarms_tools.set_alarm,
            user_scoped=True,
        )
    )

    return registry
