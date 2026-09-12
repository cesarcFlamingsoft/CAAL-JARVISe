"""Schema validation for a native tool call, before any handler runs.

The model writes the arguments of a native tool, and a model can leave one out
or give it the wrong shape. Until this boundary existed the runtime called
tool.handler(**bound) straight away, so an omitted required argument became a
Python TypeError in the middle of a voice turn and reached the model as a
contentless failure: nothing was created, and the model was never told that one
detail was missing.

The check reads the same JSON schema the registry publishes to the model, and it
answers with a sentence that invites the model to ask the user for what is
missing. Only schema-declared *names* ever appear in that sentence or in the
log: an argument value is the words of the user in another shape.
"""

from __future__ import annotations

from typing import Any

__all__ = ["invalid_arguments_result", "validate_tool_arguments"]

# The session scope is attached by the runtime, never declared in the schema.
_RUNTIME_ARGUMENTS = frozenset({"user_id"})

# A bounded refusal: enough names for the model to act on, never a long list.
_MAX_NAMES = 4
_MAX_MESSAGE = 300

_JSON_TYPES: dict[str, Any] = dict(
    string=str,
    integer=int,
    number=(int, float),
    boolean=bool,
    array=(list, tuple),
    object=dict,
)


def _names(names: list[str]) -> str:
    shown = names[:_MAX_NAMES]
    if len(shown) == 1:
        return shown[0]
    return ", ".join(shown[:-1]) + " and " + shown[-1]


def invalid_arguments_result(message: str) -> dict[str, Any]:
    """The tool result an incomplete or mistyped call gets instead of an exception."""
    return dict(status="invalid_request", message=message[:_MAX_MESSAGE], data=dict())


def _is_missing(value: Any) -> bool:
    """An argument the model left out, or filled with nothing."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False


def _matches(spec: Any, value: Any) -> bool:
    """Whether one value fits the shape its schema fragment declares."""
    if not isinstance(spec, dict):
        return True
    declared = spec.get("type")
    if isinstance(declared, str):
        expected = _JSON_TYPES.get(declared)
        if expected is None:
            return True
        # A bool is an int in Python and nowhere else; keep the schema honest.
        if declared in ("integer", "number") and isinstance(value, bool):
            return False
        if not isinstance(value, expected):
            return False
    choices = spec.get("enum")
    if isinstance(choices, list) and choices and value not in choices:
        return False
    items = spec.get("items")
    if isinstance(value, (list, tuple)) and isinstance(items, dict):
        return all(_matches(items, entry) for entry in value)
    return True


def validate_tool_arguments(tool: Any, arguments: dict[str, Any]) -> dict[str, Any] | None:
    """Check a bound native tool call; a dict result means the call must not run.

    Returns None when the call may proceed to the handler. Undeclared arguments
    are not judged here: the scope binding already drops them.
    """
    schema = getattr(tool, "parameters", None)
    if not isinstance(schema, dict):
        return None
    properties = schema.get("properties")
    properties = properties if isinstance(properties, dict) else dict()
    required = schema.get("required")
    required = required if isinstance(required, list) else []

    missing = [
        name
        for name in required
        if isinstance(name, str) and _is_missing(arguments.get(name, None))
    ]
    if missing:
        return invalid_arguments_result(
            "I need one more detail before I can do that: "
            + _names(sorted(missing))
            + ". Ask the user for it in their own words, then call the tool again."
        )

    mistyped = sorted(
        name
        for name, value in arguments.items()
        if name in properties
        and name not in _RUNTIME_ARGUMENTS
        and not _matches(properties[name], value)
    )
    if mistyped:
        return invalid_arguments_result(
            "That did not come through in a form I can use for "
            + _names(mistyped)
            + ". Check what the user asked for, then call the tool again with it."
        )
    return None
