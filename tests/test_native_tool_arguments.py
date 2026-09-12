"""The native-tool argument boundary: an incomplete model call is answered, not raised.

The production failure this covers: a voice session emitted reminders.create
without a title. The runtime called tool.handler(**bound) directly, Python
raised a TypeError for the missing positional argument, and the turn reported
the generic failure, so the model never learned to ask for the missing detail
and no reminder was ever created.

Nothing here asserts on what a user said: the assertions are about status, about
whether a handler ran, and about what must never appear anywhere.
"""

from __future__ import annotations

import importlib
import json
import logging
from types import SimpleNamespace

import pytest

from caal.tools import alarms_tools, reminder_delivery, reminders_tools
from caal.user_scope import UserScope

llm_node = importlib.import_module("caal.llm.llm_node")

ANA = "usr_" + "a" * 24


@pytest.fixture
def store(monkeypatch, tmp_path):
    path = tmp_path / "assistant.sqlite3"
    monkeypatch.setattr(alarms_tools, "STORE_PATH", path)
    monkeypatch.setattr(reminders_tools, "STORE_PATH", path)
    monkeypatch.setattr(reminder_delivery, "STORE_PATH", path)
    return path


@pytest.fixture
def agent(monkeypatch):
    """An agent as the voice runtime holds it: native tools on, a verified user."""
    monkeypatch.setattr(llm_node.settings_module, "get_setting", lambda key, default=None: True)

    class _Agent:
        _user_scope = UserScope.for_user(SimpleNamespace(user_id=ANA))

    return _Agent()


class _Provider:
    """The minimum of the provider contract that _execute_tool_calls uses."""

    def format_tool_call_message(self, content, tool_calls):
        return dict(role="assistant", content=content)

    def format_tool_result(self, content, tool_call_id, tool_name):
        return dict(role="tool", content=content, tool_call_id=tool_call_id)


def _handler_trap(monkeypatch, module, name):
    """Replace a handler so that reaching it fails the test, and record the call."""
    calls: list[dict] = []

    def _trap(**kwargs):
        calls.append(kwargs)
        raise AssertionError("an incomplete call must never reach a handler")

    monkeypatch.setattr(module, name, _trap)
    return calls


# --- the production failure ----------------------------------------------------------


@pytest.mark.asyncio
async def test_reminders_create_without_a_title_is_answered_not_raised(monkeypatch, agent, store):
    calls = _handler_trap(monkeypatch, reminders_tools, "create_reminder")

    result = await llm_node._execute_single_tool(
        agent, "reminders.create", dict(due="PT10M", list="Home")
    )

    assert calls == []
    assert result["status"] == "invalid_request"
    assert isinstance(result["message"], str) and result["message"].strip()
    assert "title" in result["message"]


@pytest.mark.asyncio
async def test_the_incomplete_call_reaches_the_model_as_a_bounded_invitation(
    monkeypatch, agent, store, caplog
):
    """The whole turn path: no traceback, no generic failure, no echoed arguments."""
    _handler_trap(monkeypatch, reminders_tools, "create_reminder")
    call = SimpleNamespace(
        name="reminders.create",
        arguments=dict(due="PT10M", notes="call the oncologist back"),
        id="call-1",
    )

    with caplog.at_level(logging.DEBUG):
        messages, _ = await llm_node._execute_tool_calls(agent, [], [call], None, _Provider())

    reported = messages[-1]["content"]
    payload = json.loads(reported)
    assert payload["status"] == "invalid_request"
    assert payload["data"] == dict()
    assert len(payload["message"]) <= 300
    assert "did not go through" not in reported
    for leak in ("Traceback", "TypeError", "positional argument", "create_reminder"):
        assert leak not in reported
    for secret in ("oncologist", "PT10M"):
        assert secret not in reported and secret not in caplog.text


@pytest.mark.asyncio
async def test_a_complete_reminder_call_still_reaches_the_handler(agent, store):
    result = await llm_node._execute_single_tool(
        agent, "reminders.create", dict(title="Buy stamps", due="PT10M")
    )

    assert result["status"] == "ok"
    listed = await llm_node._execute_single_tool(agent, "reminders.list", dict())
    assert listed["status"] == "ok"


# --- the rest of the native catalog --------------------------------------------------


@pytest.mark.asyncio
async def test_another_native_tool_with_a_missing_required_argument(monkeypatch, agent, store):
    calls = _handler_trap(monkeypatch, alarms_tools, "set_alarm")

    result = await llm_node._execute_single_tool(agent, "alarms.set", dict(label="Tea"))

    assert calls == []
    assert result["status"] == "invalid_request"
    assert "when" in result["message"] and "kind" in result["message"]


@pytest.mark.asyncio
async def test_a_complete_alarm_call_still_reaches_the_handler(agent, store):
    result = await llm_node._execute_single_tool(
        agent, "alarms.set", dict(label="Tea", when="PT1M", kind="timer")
    )

    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_a_mistyped_scalar_argument_is_refused_before_the_handler(monkeypatch, agent, store):
    calls = _handler_trap(monkeypatch, reminders_tools, "list_reminders")

    result = await llm_node._execute_single_tool(
        agent, "reminders.list", dict(include_completed="yes please")
    )

    assert calls == []
    assert result["status"] == "invalid_request"
    assert "include_completed" in result["message"]
    assert "yes please" not in result["message"]


@pytest.mark.asyncio
async def test_a_mistyped_container_argument_is_refused_before_the_handler(
    monkeypatch, agent, store
):
    calls = _handler_trap(monkeypatch, reminders_tools, "set_delivery")

    result = await llm_node._execute_single_tool(
        agent, "reminders.set_delivery", dict(delivery="call and telegram")
    )

    assert calls == []
    assert result["status"] == "invalid_request"
    assert "call and telegram" not in result["message"]


@pytest.mark.asyncio
async def test_an_undeclared_argument_is_still_dropped_rather_than_refused(agent, store):
    """Pre-existing behaviour: an invented argument is dropped and the call proceeds."""
    result = await llm_node._execute_single_tool(
        agent, "reminders.create", dict(title="Buy stamps", connection_id="cx_1")
    )

    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_an_anonymous_session_is_still_refused_before_any_validation(monkeypatch, store):
    """Scope refusal outranks argument validation: no scope, no answer about arguments."""
    monkeypatch.setattr(llm_node.settings_module, "get_setting", lambda key, default=None: True)
    from caal.user_scope import SCHEDULING_UNAVAILABLE_REPLY

    class _Anonymous:
        _user_scope = UserScope.anonymous()

    result = await llm_node._execute_single_tool(_Anonymous(), "reminders.create", dict())

    assert result["status"] == "unauthorized"
    assert result["message"] == SCHEDULING_UNAVAILABLE_REPLY


# --- the ordinary natural request, as the model actually emits it ---------------------


@pytest.fixture
def everything_available(monkeypatch):
    """Both remote channels authorised for Ana, so a saved default could bite."""
    monkeypatch.setattr(reminder_delivery, "telegram_owner", lambda: ANA)
    monkeypatch.setattr(reminder_delivery, "telegram_configured", lambda: True)
    monkeypatch.setattr(
        reminder_delivery, "resolve_callback_number", lambda user_id: "+15551230000"
    )


@pytest.mark.asyncio
async def test_a_natural_relative_reminder_maps_to_title_and_due(
    agent, store, everything_available, caplog
):
    """remind me in 1 minute: a title and a phrase, no PT2M and no field names."""
    with caplog.at_level(logging.DEBUG):
        result = await llm_node._execute_single_tool(
            agent, "reminders.create", dict(title="take the bread out", due="in 1 minute")
        )

    assert result["status"] == "ok"
    assert result["data"]["timed"] is True
    assert result["data"]["due"] is not None
    assert result["data"]["delivery"] == [reminder_delivery.SPEAK]
    assert result["data"]["delivery_pending"] is True
    for cue in ("say it", "telegram", "call you"):
        assert cue in result["message"].lower()
    for secret in ("bread", "in 1 minute"):
        assert secret not in caplog.text


@pytest.mark.asyncio
async def test_a_saved_dashboard_default_does_not_arm_remote_channels_through_the_runtime(
    agent, store, everything_available
):
    reminder_delivery.set_default_channels(ANA, ["all"])

    result = await llm_node._execute_single_tool(
        agent, "reminders.create", dict(title="Standup", due="in an hour")
    )

    assert result["data"]["delivery"] == [reminder_delivery.SPEAK]
    assert result["data"]["delivery_pending"] is True
    assert reminder_delivery.claim_due("worker-1", now=2**31) == []


@pytest.mark.asyncio
async def test_the_word_default_travels_through_the_schema_as_a_selection(
    agent, store, everything_available
):
    """The enum the registry publishes has to carry it, or the call never runs."""
    result = await llm_node._execute_single_tool(
        agent, "reminders.create", dict(title="Standup", due="in an hour", delivery=["default"])
    )

    assert result["status"] == "ok"
    assert result["data"]["delivery"] == [reminder_delivery.SPEAK]
    assert result["data"]["delivery_pending"] is False


@pytest.mark.asyncio
async def test_an_explicit_remote_combination_is_still_additive_through_the_runtime(
    agent, store, everything_available
):
    created = await llm_node._execute_single_tool(
        agent, "reminders.create", dict(title="Standup", due="in an hour")
    )
    assert created["data"]["delivery_pending"] is True

    answered = await llm_node._execute_single_tool(
        agent, "reminders.set_delivery", dict(delivery=["telegram", "call"])
    )

    assert answered["status"] == "ok"
    assert answered["data"]["delivery"] == [reminder_delivery.TELEGRAM, reminder_delivery.CALL]


@pytest.mark.asyncio
async def test_an_unauthorised_remote_channel_is_still_refused_through_the_runtime(
    agent, store, monkeypatch
):
    monkeypatch.setattr(reminder_delivery, "telegram_configured", lambda: False)
    monkeypatch.setattr(reminder_delivery, "resolve_callback_number", lambda user_id: None)

    result = await llm_node._execute_single_tool(
        agent,
        "reminders.create",
        dict(title="Standup", due="in an hour", delivery=["telegram", "call"]),
    )

    assert result["status"] == "ok"
    assert result["data"]["delivery"] == [reminder_delivery.SPEAK]
    assert "telegram" in result["message"].lower()
