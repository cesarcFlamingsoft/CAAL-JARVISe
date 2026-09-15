"""Tool availability is a runtime boundary, not a prompt preference."""

import importlib
from types import SimpleNamespace

import pytest

from caal.user_scope import UserScope

node = importlib.import_module("caal.llm.llm_node")


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [UserScope.anonymous(), UserScope("usr_" + "a" * 24, True)])
async def test_connected_sessions_do_not_offer_or_execute_legacy_calendar_reads(monkeypatch, scope):
    monkeypatch.setattr(node.settings_module, "get_setting", lambda key, default=None: True)
    from caal.tools import calendar_tools

    calls = []
    monkeypatch.setattr(calendar_tools, "list_calendar_events", lambda **kw: calls.append(kw))
    agent = SimpleNamespace(_user_scope=scope)
    tools = await node._discover_tools(agent)
    names = {tool["function"]["name"] for tool in tools}
    assert "schedule.upcoming" in names
    assert not names.intersection(
        {"calendar.list_events", "calendar.find_free_time", "email.search", "email.read"}
    )
    result = await node._execute_single_tool(
        agent, "calendar.list_events", {"source": "all", "start": "x", "end": "y"}
    )
    assert result["status"] == "unsupported_tool"
    assert calls == []


@pytest.mark.asyncio
async def test_unknown_call_is_corrected_before_any_batch_side_effect(monkeypatch):
    from livekit.agents.llm import ChatContext

    from caal.llm.providers.base import LLMProvider, LLMResponse, ToolCall
    from caal.tools import knowledge_tools, memory_tools

    monkeypatch.setattr(node.settings_module, "get_setting", lambda key, default=None: True)
    reads, writes = [], []
    monkeypatch.setattr(
        knowledge_tools,
        "upcoming_schedule",
        lambda **kw: reads.append(kw) or {"status": "ok", "data": {}},
    )
    monkeypatch.setattr(memory_tools, "remember", lambda **kw: writes.append(kw))

    class Provider(LLMProvider):
        provider_name = "test"
        model = "test"

        def __init__(self):
            self.calls = []

        async def chat(self, messages, tools=None, **kwargs):
            self.calls.append((list(messages), kwargs))
            if len(self.calls) == 1:
                return LLMResponse(
                    None,
                    [
                        ToolCall("a", "calendar.upcoming", {"day": "today"}),
                        ToolCall("b", "memory.remember", {"key": "preference", "value": "private"}),
                    ],
                )
            assert any(m.get("content") == "whats on my calendar for today" for m in messages)
            assert not any(m.get("role") == "tool" or m.get("tool_calls") for m in messages)
            return LLMResponse(None, [ToolCall("c", "schedule.upcoming", {"day": "today"})])

        async def chat_stream(self, messages, tools=None, **kwargs):
            yield "Read completed."

    provider = Provider()
    agent = SimpleNamespace(_user_scope=UserScope("usr_" + "a" * 24, True))
    ctx = ChatContext()
    ctx.add_message(role="user", content="whats on my calendar for today")
    answer = "".join([p async for p in node.llm_node(agent, ctx, provider, reasoning=True)])
    assert len(provider.calls) == 2
    assert len(reads) == 1
    assert reads[0]["user_id"] == agent._user_scope.user_id
    assert reads[0]["day"] == "today"
    assert writes == []
    assert provider.calls[1][1]["think"] is True
    assert answer == "Read completed."


@pytest.mark.asyncio
async def test_connected_read_does_not_silently_drop_an_account_filter(monkeypatch):
    from livekit.agents.llm import ChatContext

    from caal.llm.providers.base import LLMProvider, LLMResponse, ToolCall
    from caal.tools import knowledge_tools

    monkeypatch.setattr(node.settings_module, "get_setting", lambda key, default=None: True)
    reads = []
    monkeypatch.setattr(
        knowledge_tools,
        "upcoming_schedule",
        lambda **kw: reads.append(kw) or {"status": "ok", "data": {}},
    )

    class Provider(LLMProvider):
        provider_name = model = "test"
        calls = 0

        async def chat(self, messages, tools=None, **kwargs):
            self.calls += 1
            args = (
                {"day": "today", "source": "work"}
                if self.calls == 1
                else {"day": "today", "account": "work"}
            )
            return LLMResponse(None, [ToolCall("a", "schedule.upcoming", args)])

        async def chat_stream(self, messages, tools=None, **kwargs):
            yield "Done."

    provider = Provider()
    agent = SimpleNamespace(_user_scope=UserScope("usr_" + "a" * 24, True))
    ctx = ChatContext()
    ctx.add_message(role="user", content="What is on my work calendar today?")
    _ = [p async for p in node.llm_node(agent, ctx, provider)]
    assert len(reads) == 1
    assert reads[0].get("account") == "work"
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_direct_connected_dispatch_refuses_undeclared_scope(monkeypatch):
    from caal.tools import knowledge_tools

    monkeypatch.setattr(node.settings_module, "get_setting", lambda key, default=None: True)
    calls = []
    monkeypatch.setattr(
        knowledge_tools, "upcoming_schedule", lambda **kw: calls.append(kw) or {"status": "ok"}
    )
    agent = SimpleNamespace(_user_scope=UserScope("usr_" + "a" * 24, True))
    result = await node._execute_single_tool(
        agent, "schedule.upcoming", {"day": "today", "source": "work"}
    )
    assert calls == []
    assert result["status"] == "invalid_request"


@pytest.mark.asyncio
async def test_failed_correction_is_bounded_and_yes_please_keeps_scoped_request(monkeypatch):
    from livekit.agents.llm import ChatContext

    from caal.llm.providers.base import LLMProvider, LLMResponse, ToolCall
    from caal.tools import knowledge_tools, memory_tools

    monkeypatch.setattr(node.settings_module, "get_setting", lambda key, default=None: True)
    reads, writes = [], []
    monkeypatch.setattr(
        knowledge_tools,
        "upcoming_schedule",
        lambda **kw: reads.append(kw) or {"status": "ok", "data": {}},
    )
    monkeypatch.setattr(memory_tools, "remember", lambda **kw: writes.append(kw))

    class Provider(LLMProvider):
        provider_name = model = "test"
        calls = 0

        async def chat(self, messages, tools=None, **kwargs):
            self.calls += 1
            if self.calls <= 2:
                return LLMResponse(
                    None,
                    [
                        ToolCall("a", "calendar.upcoming", {"day": "tomorrow"}),
                        ToolCall("b", "memory.remember", {"key": "x", "value": "y"}),
                    ],
                )
            assert [m["content"] for m in messages if m["role"] == "user"] == [
                "Read my work calendar tomorrow.",
                "Yes please",
            ]
            return LLMResponse(
                None,
                [
                    ToolCall(
                        "c",
                        "schedule.upcoming",
                        {"day": "tomorrow", "account": "work", "user_id": "usr_" + "b" * 24},
                    )
                ],
            )

        async def chat_stream(self, messages, tools=None, **kwargs):
            yield "Read completed."

    provider = Provider()
    agent = SimpleNamespace(_user_scope=UserScope("usr_" + "a" * 24, True))
    ctx = ChatContext()
    ctx.add_message(role="user", content="Read my work calendar tomorrow.")
    answer = "".join([p async for p in node.llm_node(agent, ctx, provider)])
    assert provider.calls == 2
    assert reads == writes == []
    ctx.add_message(role="assistant", content=answer)
    ctx.add_message(role="user", content="Yes please")
    _ = [p async for p in node.llm_node(agent, ctx, provider)]
    assert len(reads) == 1
    assert reads[0] == {"day": "tomorrow", "account": "work", "user_id": agent._user_scope.user_id}
    assert writes == []


@pytest.mark.asyncio
async def test_legacy_session_retains_legacy_calendar_tools(monkeypatch):
    monkeypatch.setattr(node.settings_module, "get_setting", lambda key, default=None: True)
    tools = await node._discover_tools(SimpleNamespace(_user_scope=UserScope.legacy()))
    assert "calendar.list_events" in {t["function"]["name"] for t in tools}
