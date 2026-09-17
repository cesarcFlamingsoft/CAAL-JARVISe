"""Conversational weather shares the dashboard backend and privacy boundary."""

# ruff: noqa: F811
import asyncio
import importlib
import json
from types import SimpleNamespace

import httpx
from test_weather import (  # noqa: F401
    NOW,
    PARIS,
    FakeUpstream,
    _forecast_body,
    ana,
    bo,
    store,
    users,
)

from caal.llm.context_barrier import is_knowledge_tool, sanitize_for_escalation
from caal.tools.registry import create_default_registry
from caal.user_scope import UserScope
from caal.weather import WeatherClient

node = importlib.import_module("caal.llm.llm_node")


def run(coro):
    return asyncio.run(coro)


def test_catalog_and_private_barrier():
    tool = create_default_registry().get("weather.current")
    assert tool.user_scoped and not tool.requires_confirmation
    assert tool.parameters["properties"] == {}
    assert is_knowledge_tool(tool.name)
    assert node._keeps_contents_private(tool.name)
    cache = node.ToolDataCache()
    assert cache.add(tool.name, {"location": "Secret location"}) is False
    assert cache.get_context_message() is None
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "w", "function": {"name": tool.name, "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "w", "content": "Secret location"},
        {"role": "assistant", "content": "Secret location weather"},
    ]
    assert "Secret location" not in json.dumps(sanitize_for_escalation(messages))


def test_handler_shared_cache_scope_and_failure(monkeypatch, store, ana, bo):
    from caal.tools import weather_tools

    upstream = FakeUpstream()
    upstream.add(httpx.Response(200, json=_forecast_body()))
    clock = [NOW]
    client = WeatherClient(store, transport=upstream.transport, clock=lambda: clock[0])
    runtime = SimpleNamespace(client=client, now=lambda: clock[0])
    monkeypatch.setattr(weather_tools, "_runtime", lambda: runtime)
    store.set_city(ana, PARIS, now=NOW)
    agent = SimpleNamespace(
        _native_tool_registry=create_default_registry(), _user_scope=UserScope(ana, True)
    )
    first = run(node._execute_single_tool(agent, "weather.current", {"user_id": bo}))
    assert first["status"] == "ok"
    assert first["data"] == dict(generated_at=NOW, **run(client.current(ana)).view())
    assert len(upstream.forecast_calls) == 1
    assert ana not in json.dumps(first) and "latitude" not in json.dumps(first)
    invalid = run(node._execute_single_tool(agent, "weather.current", {"city": "Tokyo"}))
    assert invalid["status"] == "invalid_request"
    clock[0] += 3601
    assert run(weather_tools.current_weather(user_id=ana))["status"] == "stale"
    store.set_city(bo, PARIS, now=NOW)
    assert run(weather_tools.current_weather(user_id=bo))["status"] == "unavailable"
    for scope in (UserScope.anonymous(), UserScope.legacy()):
        agent._user_scope = scope
        denied = run(node._execute_single_tool(agent, "weather.current", {"user_id": ana}))
        assert denied["status"] == "unauthorized"
        assert "weather" in denied["message"].lower()


def test_expired_location_read_does_not_mutate(monkeypatch, store, ana):
    from caal.tools import weather_tools

    store.set_browser_location(ana, latitude=10, longitude=20, now=NOW)

    def row():
        with store._connect() as db:
            return tuple(db.execute("SELECT * FROM weather_preferences").fetchone())

    before = row()
    now = NOW + store.browser_ttl_seconds + 1
    client = WeatherClient(store, clock=lambda: now)
    monkeypatch.setattr(
        weather_tools, "_runtime", lambda: SimpleNamespace(client=client, now=lambda: now)
    )
    assert run(weather_tools.current_weather(user_id=ana))["status"] == "no_location"
    assert row() == before


def test_backend_exception_never_exposes_location(monkeypatch, ana):
    from caal.tools import weather_tools

    def broken():
        raise RuntimeError("secret coordinate and token")

    monkeypatch.setattr(weather_tools, "_runtime", broken)
    result = run(weather_tools.current_weather(user_id=ana))
    assert result["status"] == "unavailable"
    assert "secret" not in json.dumps(result)
    assert result["data"] == {}


def test_real_tool_loop_keeps_location_out_of_logs(monkeypatch, caplog, store, ana):
    import logging

    from caal.llm.providers.base import LLMResponse, ToolCall
    from caal.llm.providers.ollama_provider import OllamaProvider
    from caal.tools import weather_tools

    upstream = FakeUpstream()
    upstream.add(httpx.Response(200, json=_forecast_body()))
    store.set_city(ana, PARIS, now=NOW)
    client = WeatherClient(store, transport=upstream.transport, clock=lambda: NOW)
    monkeypatch.setattr(
        weather_tools, "_runtime", lambda: SimpleNamespace(client=client, now=lambda: NOW)
    )
    seen = []

    class Provider(OllamaProvider):
        async def chat(self, messages, tools=None, **kwargs):
            assert any(t["function"]["name"] == "weather.current" for t in tools)
            assert "weather.current" in messages[0]["content"]
            assert "follow-up" in messages[0]["content"]
            return LLMResponse(None, [ToolCall("w", "weather.current", {})])

        async def chat_stream(self, messages, **kwargs):
            result = json.loads(next(m["content"] for m in messages if m["role"] == "tool"))
            seen.append(result)
            yield f"Paris: {result['data']['observation']['temperature_c']} degrees."

    async def speak():
        agent = SimpleNamespace(_user_scope=UserScope(ana, True))
        return "".join(
            [
                part
                async for part in node.llm_node(
                    agent, SimpleNamespace(items=[]), provider=Provider()
                )
            ]
        )

    with caplog.at_level(logging.DEBUG):
        answer = run(speak())
    assert answer == "Paris: 7.4 degrees."
    assert seen[0]["status"] == "ok"
    assert "Paris" not in caplog.text
    assert ana not in caplog.text
