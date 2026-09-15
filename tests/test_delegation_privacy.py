"""Private HA reads stay local when administrator delegation is restored."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from caal.background_task_session import capture_task_context
from caal.llm.context_barrier import (
    forget_private_answers,
    record_private_answer,
    sanitize_for_escalation,
)
from caal.llm.providers.base import LLMResponse, ToolCall
from caal.llm.providers.hermes_provider import HermesProvider

node = importlib.import_module("caal.llm.llm_node")
PRIVATE = "Bedroom occupancy: occupied; attic temperature: 28 degrees."
REQUEST = "Research Python packaging and implement the parser in the background."


@pytest.fixture(autouse=True)
def ledger():
    forget_private_answers()
    yield
    forget_private_answers()


class ChatMessage:
    def __init__(self, role, text):
        self.type = "message"
        self.role = role
        self.text_content = text
        self.content = [text]


def message(role, text):
    return ChatMessage(role, text)


@pytest.mark.parametrize("stream", [False, True])
def test_direct_hermes_payload_drops_ha_data_and_derived_answer(stream):
    record_private_answer(PRIVATE)
    messages = [
        {"role": "assistant", "content": PRIVATE},
        {"role": "tool", "name": "hass_assist", "content": PRIVATE},
        {"role": "user", "content": REQUEST},
    ]
    payload = HermesProvider(api_key="test-only")._build_payload(messages, stream=stream)
    assert PRIVATE not in json.dumps(payload)
    assert REQUEST in json.dumps(payload)
    assert messages[0]["content"] == PRIVATE


def test_background_snapshot_sanitizes_private_turns_before_flattening():
    record_private_answer(PRIVATE)
    session = SimpleNamespace(
        history=SimpleNamespace(
            items=[
                message("user", "What is the temperature?"),
                message("assistant", PRIVATE),
                message("assistant", "I can help with Python."),
                message("user", REQUEST),
            ]
        )
    )
    context = capture_task_context(session)
    assert PRIVATE not in context
    assert REQUEST in context
    assert "I can help with Python." in context


@pytest.mark.asyncio
@pytest.mark.parametrize("scheduled", [[], ["Your reminder is set."]])
async def test_exact_spoken_ha_answer_is_private_including_mixed_outcomes(monkeypatch, scheduled):
    calls = [ToolCall(id="ha-1", name="hass_assist", arguments={})]
    provider = SimpleNamespace(
        manages_own_tools=False,
        chat=AsyncMock(return_value=LLMResponse(content=None, tool_calls=calls)),
    )
    monkeypatch.setattr(
        node, "_discover_tools", AsyncMock(return_value=[{"function": {"name": "hass_assist"}}])
    )
    monkeypatch.setattr(node, "_invalid_tool_batch", lambda *args: False)
    monkeypatch.setattr(
        node,
        "_execute_tool_calls",
        AsyncMock(
            return_value=(
                [],
                node.ToolOutcomes(
                    hass=["**Bedroom occupancy:** occupied.", "Attic: 28 degrees."],
                    scheduled=scheduled,
                ),
            )
        ),
    )
    spoken = "".join(
        [
            chunk
            async for chunk in node.llm_node(SimpleNamespace(), SimpleNamespace(items=[]), provider)
        ]
    )
    assert "Bedroom occupancy" in spoken
    sanitized = sanitize_for_escalation(
        [{"role": "assistant", "content": spoken}, {"role": "user", "content": REQUEST}]
    )
    assert "Bedroom occupancy" not in json.dumps(sanitized)
    assert REQUEST in json.dumps(sanitized)


@pytest.mark.asyncio
async def test_generic_argument_generation_never_sees_private_history(monkeypatch):
    record_private_answer(PRIVATE)
    provider = SimpleNamespace(
        manages_own_tools=False,
        chat=AsyncMock(return_value=LLMResponse(content="Ready for research.", tool_calls=[])),
    )
    monkeypatch.setattr(
        node,
        "_discover_tools",
        AsyncMock(
            return_value=[
                {"function": {"name": "friday_assist"}},
                {"function": {"name": "hass_assist"}},
            ]
        ),
    )
    monkeypatch.setattr(node, "_invalid_tool_batch", lambda *args: False)
    context = SimpleNamespace(items=[message("assistant", PRIVATE), message("user", REQUEST)])
    spoken = "".join([chunk async for chunk in node.llm_node(SimpleNamespace(), context, provider)])
    sent = provider.chat.call_args.kwargs
    assert PRIVATE not in json.dumps(sent["messages"])
    assert REQUEST in json.dumps(sent["messages"])
    assert "friday_assist" in json.dumps(sent["tools"])
    assert spoken == "Ready for research."


@pytest.mark.asyncio
async def test_private_result_blocks_only_later_external_calls_in_same_batch(monkeypatch):
    executed = []

    async def execute(agent, name, arguments):
        executed.append(name)
        return PRIVATE if name == "hass_assist" else "done"

    monkeypatch.setattr(node, "_execute_single_tool", execute)
    provider = HermesProvider(api_key="test-only")
    calls = [
        ToolCall(id="ha-1", name="hass_assist", arguments={}),
        ToolCall(id="external-1", name="friday_assist", arguments={"request": "Continue"}),
        ToolCall(id="ha-2", name="hass_assist", arguments={}),
    ]
    await node._execute_tool_calls(SimpleNamespace(), [], calls, None, provider)
    assert executed == ["hass_assist", "hass_assist"]
    executed.clear()
    await node._execute_tool_calls(SimpleNamespace(), [], calls[1:2], None, provider)
    assert executed == ["friday_assist"], "a fresh ordinary delegation batch stays available"


@pytest.mark.parametrize("named_result", [False, True])
def test_private_derived_tool_call_and_its_result_never_cross_barrier(named_result):
    derived_call = {
        "id": "derived-1",
        "type": "function",
        "function": {"name": "friday_assist", "arguments": json.dumps({"request": PRIVATE})},
    }
    derived_result = {"role": "tool", "tool_call_id": "derived-1", "content": PRIVATE}
    if named_result:
        derived_result["name"] = "friday_assist"
    messages = [
        {"role": "tool", "name": "hass_assist", "content": PRIVATE},
        {"role": "assistant", "content": "Continuing from the read.", "tool_calls": [derived_call]},
        derived_result,
        {"role": "assistant", "content": PRIVATE},
        {"role": "user", "content": REQUEST},
    ]
    sanitized = sanitize_for_escalation(messages)
    assert PRIVATE not in json.dumps(sanitized)
    assert "derived-1" not in json.dumps(sanitized)
    assert REQUEST in json.dumps(sanitized)
    ordinary = [
        {"role": "user", "content": REQUEST},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "ordinary-1",
                    "type": "function",
                    "function": {
                        "name": "friday_assist",
                        "arguments": json.dumps({"request": REQUEST}),
                    },
                }
            ],
        },
        {"role": "tool", "name": "friday_assist", "tool_call_id": "ordinary-1", "content": "Done."},
    ]
    assert sanitize_for_escalation(ordinary) == ordinary
