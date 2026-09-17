import asyncio
import importlib
import json
from datetime import datetime
from types import SimpleNamespace

from caal import settings
from caal.llm.providers.base import LLMResponse, ToolCall
from caal.llm.providers.ollama_provider import OllamaProvider
from caal.tools.registry import create_default_registry
from caal.user_scope import UserScope


def run(coro):
    return asyncio.run(coro)


def test_current_time_tool_uses_a_fresh_configured_zone(monkeypatch):
    from caal.tools import clock_tools

    monkeypatch.setenv("TIMEZONE", "America/Edmonton")
    tool = create_default_registry().get("time.current")
    assert tool.category == "time"
    assert tool.parameters["properties"] == {}
    assert not tool.requires_confirmation

    first = clock_tools.current_time_for_testing(datetime(2026, 9, 16, 20, 15, 0))
    second = clock_tools.current_time_for_testing(datetime(2026, 9, 16, 20, 16, 0))

    assert first["status"] == second["status"] == "ok"
    assert first["data"] == {
        "local_time": "2026-09-16T14:15:00-06:00",
        "timezone": "America/Edmonton",
        "timezone_abbreviation": "MDT",
        "location_source": "weather_city",
    }
    assert second["data"]["local_time"] == "2026-09-16T14:16:00-06:00"
    assert "fresh" in first["message"].lower()


def test_current_time_uses_the_verified_users_weather_timezone(monkeypatch):
    from caal.tools import clock_tools

    runtime = SimpleNamespace(
        now=lambda: 0,
        store=SimpleNamespace(
            preferences=lambda user_id, **kwargs: SimpleNamespace(
                resolved=SimpleNamespace(timezone="Asia/Tokyo")
            )
        ),
        client=SimpleNamespace(),
    )
    monkeypatch.setattr(clock_tools, "_runtime", lambda: runtime)

    result = run(clock_tools.current_time(user_id="usr_" + "a" * 24))

    assert result["status"] == "ok"
    assert result["data"]["timezone"] == "Asia/Tokyo"
    assert result["data"]["location_source"] == "weather_city"


def test_session_prompt_never_injects_a_current_time(monkeypatch):
    monkeypatch.setattr(
        settings, "load_prompt_content", lambda: "{{CURRENT_DATE_CONTEXT}} -- {{TIMEZONE}}"
    )

    prompt = settings.load_prompt_with_context("America/Edmonton", "Mountain Time")

    assert "The current time is" not in prompt
    assert "time.current" in prompt
    assert "Mountain Time" in prompt


def test_time_question_requires_a_fresh_native_tool_read(monkeypatch):
    from caal.tools import clock_tools

    runtime = SimpleNamespace(
        now=lambda: 0,
        store=SimpleNamespace(
            preferences=lambda user_id, **kwargs: SimpleNamespace(
                resolved=SimpleNamespace(timezone="Asia/Tokyo")
            )
        ),
        client=SimpleNamespace(),
    )
    monkeypatch.setattr(clock_tools, "_runtime", lambda: runtime)
    node = importlib.import_module("caal.llm.llm_node")
    observed = []

    class Provider(OllamaProvider):

        async def chat(self, messages, tools=None, **kwargs):
            observed.append((messages, tools))
            assert any(tool["function"]["name"] == "time.current" for tool in tools)
            assert "call time.current before answering" in messages[0]["content"]
            return LLMResponse(None, [ToolCall("clock", "time.current", {})])

        async def chat_stream(self, messages, **kwargs):
            result = json.loads(messages[-1]["content"])
            assert result["status"] == "ok"
            assert result["data"]["timezone"] == "Asia/Tokyo"
            assert result["data"]["location_source"] == "weather_city"
            yield "It is the fresh local time."

    async def speak():
        agent = SimpleNamespace(_user_scope=UserScope("usr_" + "a" * 24, True))
        return "".join(
            [
                part
                async for part in node.llm_node(
                    agent, SimpleNamespace(items=[]), provider=Provider()
                )
            ]
        )

    assert run(speak()) == "It is the fresh local time."
    assert len(observed) == 1
