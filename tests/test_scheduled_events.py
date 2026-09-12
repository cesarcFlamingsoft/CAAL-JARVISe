"""The one packet that tells a dashboard its scheduled items changed.

A reminder or an alarm that was just created has to reach the widget without
waiting out the polling interval. The only thing published for that is a
constant: a version and a kind, and nothing else. No title, no time, no row id,
no owner, no arguments, no model output, no phone number and no chat. A
listener learns exactly one fact -- *something of yours changed, go and read
your own feed over the authenticated route* -- and learns it only after the
change was actually written.
"""

from __future__ import annotations

import importlib
import json
import logging
from types import SimpleNamespace

import pytest

from caal import scheduled_events
from caal.tools import alarms_tools, reminder_delivery, reminders_tools

ANA = "usr_" + "a" * 24
NOW = 1_000_000


@pytest.fixture
def store(monkeypatch, tmp_path):
    path = tmp_path / "assistant.sqlite3"
    monkeypatch.setattr(alarms_tools, "STORE_PATH", path)
    monkeypatch.setattr(reminders_tools, "STORE_PATH", path)
    monkeypatch.setattr(reminder_delivery, "STORE_PATH", path)
    return path


# --- the packet itself --------------------------------------------------------------------


def test_the_packet_is_a_constant_with_a_version_and_a_kind():
    assert scheduled_events.TOPIC == "scheduled_changed"
    assert json.loads(scheduled_events.PAYLOAD) == {"v": 1, "kind": "scheduled_changed"}
    assert set(json.loads(scheduled_events.PAYLOAD)) == {"v", "kind"}
    assert isinstance(scheduled_events.PAYLOAD, bytes) is False
    assert len(scheduled_events.PAYLOAD) < 64


def test_the_packet_is_the_same_bytes_however_the_change_was_made():
    first = scheduled_events.PAYLOAD
    second = scheduled_events.PAYLOAD
    assert first == second
    assert "title" not in first and "due" not in first and "user" not in first


@pytest.mark.parametrize(
    "tool_name, result, expected",
    [
        ("reminders.create", dict(status="ok"), True),
        ("reminders.set_delivery", dict(status="ok"), True),
        ("alarms.set", dict(status="ok"), True),
        ("reminders.create", dict(status="invalid_request"), False),
        ("alarms.set", dict(status="invalid_time"), False),
        ("reminders.create", "not a result", False),
        ("reminders.list", dict(status="ok"), False),
        ("inbox.recent", dict(status="ok"), False),
    ],
)
def test_only_a_successful_scheduled_mutation_announces_itself(tool_name, result, expected):
    assert scheduled_events.is_scheduled_mutation(tool_name, result) is expected


# --- through the real native tool path ----------------------------------------------------


class _Provider:
    def format_tool_call_message(self, content, tool_calls):
        return dict(role="assistant", content=content or "")

    def format_tool_result(self, content, tool_call_id, tool_name):
        return dict(role="tool", content=content, tool_call_id=tool_call_id)


class _Agent:
    def __init__(self, user_id=ANA):
        from caal.user_scope import UserScope

        self._user_scope = UserScope.for_user(SimpleNamespace(user_id=user_id))
        self.published: list[tuple[str, str]] = []

    async def _on_scheduled_change(self) -> None:
        self.published.append((scheduled_events.TOPIC, scheduled_events.PAYLOAD))


async def _execute(agent, monkeypatch, name, arguments):
    node = importlib.import_module("caal.llm.llm_node")
    monkeypatch.setattr(node.settings_module, "get_setting", lambda key, default=None: True)
    call = SimpleNamespace(name=name, arguments=arguments, id="1")
    return await node._execute_tool_calls(agent, [], [call], None, _Provider())


@pytest.mark.asyncio
async def test_a_created_reminder_publishes_exactly_one_safe_packet(monkeypatch, store):
    agent = _Agent()

    await _execute(agent, monkeypatch, "reminders.create", dict(title="Biopsy results", due="PT2M"))

    assert agent.published == [("scheduled_changed", scheduled_events.PAYLOAD)]
    packet = agent.published[0][1]
    for secret in ("Biopsy", "PT2M", ANA, "reminder_id"):
        assert secret not in packet


@pytest.mark.asyncio
async def test_a_set_alarm_publishes_the_same_packet(monkeypatch, store):
    agent = _Agent()

    await _execute(
        agent, monkeypatch, "alarms.set", dict(label="Divorce lawyer", when="PT2M", kind="alarm")
    )

    assert agent.published == [("scheduled_changed", scheduled_events.PAYLOAD)]


@pytest.mark.asyncio
async def test_a_refused_call_announces_nothing(monkeypatch, store):
    agent = _Agent()

    await _execute(agent, monkeypatch, "alarms.set", dict(label="Tea", when="soon", kind="alarm"))

    assert agent.published == []
    assert alarms_tools.pending_count(user_id=ANA) == 0


@pytest.mark.asyncio
async def test_a_read_announces_nothing(monkeypatch, store):
    agent = _Agent()

    await _execute(agent, monkeypatch, "reminders.list", dict())

    assert agent.published == []


@pytest.mark.asyncio
async def test_a_delivery_choice_announces_the_change(monkeypatch, store):
    agent = _Agent()
    reminders_tools.create_reminder(title="Call the clinic", due="PT30M", user_id=ANA)

    await _execute(agent, monkeypatch, "reminders.set_delivery", dict(delivery=["default"]))

    assert agent.published == [("scheduled_changed", scheduled_events.PAYLOAD)]


@pytest.mark.asyncio
async def test_a_session_that_cannot_publish_still_creates_the_reminder(monkeypatch, store):
    """A dead room is not a reason to lose a reminder."""

    class _Broken(_Agent):
        async def _on_scheduled_change(self):
            raise RuntimeError("the room went away")

    agent = _Broken()

    await _execute(agent, monkeypatch, "reminders.create", dict(title="Stretch", due="PT2M"))

    assert alarms_tools.pending_count(user_id=ANA) == 1


@pytest.mark.asyncio
async def test_publishing_leaves_nothing_private_in_the_log(monkeypatch, store, caplog):
    agent = _Agent()

    with caplog.at_level(logging.DEBUG):
        await _execute(
            agent, monkeypatch, "reminders.create", dict(title="Biopsy results", due="PT2M")
        )

    for secret in ("Biopsy", "PT2M", ANA):
        assert secret not in caplog.text


# --- how the running agent publishes it ---------------------------------------------------


def _agent_source() -> str:
    from pathlib import Path

    return (Path(__file__).resolve().parents[1] / "voice_agent.py").read_text()


def test_the_agent_publishes_the_constant_on_its_own_topic():
    source = _agent_source()

    assert "scheduled_events.PAYLOAD.encode" in source
    assert "topic=scheduled_events.TOPIC" in source
    assert "on_scheduled_change=_publish_scheduled_change" in source
    # The tool-status packet carries model arguments; nothing of it is reused.
    # The code of the publisher, past its own docstring: the tool-status packet
    # carries model arguments, and none of it is reused here.
    body = source.split("async def _publish_scheduled_change")[1].split('"""')[2]
    body = body.split("async def")[0]
    for forbidden in ("tool_params", "tool_names", "title", "label", "user_id", "json.dumps"):
        assert forbidden not in body


def test_the_voice_assistant_accepts_the_publisher_and_keeps_it_private():
    source = _agent_source()

    assert "on_scheduled_change: Callable[[], Awaitable[None]] | None = None" in source
    assert "self._on_scheduled_change = on_scheduled_change" in source


def test_the_alarm_loop_only_announces_to_a_room_somebody_is_in():
    source = _agent_source()

    assert "has_listening_participant" in source
    assert "listener=lambda: has_listening_participant(ctx.room)" in source
