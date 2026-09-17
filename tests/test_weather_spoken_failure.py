"""Failures must give an actionable bilingual reply even if narration is empty."""

import asyncio
from types import SimpleNamespace

import pytest

from caal.language_policy import LanguageSession
from caal.llm.llm_node import llm_node
from caal.llm.providers.base import LLMResponse, ToolCall
from caal.llm.providers.ollama_provider import OllamaProvider
from caal.tools import weather_tools
from caal.user_scope import UserScope


@pytest.mark.parametrize(
    "language,expected", [("en", "weather widget"), ("es", "widget del tiempo")]
)
@pytest.mark.parametrize("state", ["no_location", "unavailable", "unauthorized"])
def test_failure_is_spoken_without_relying_on_model(monkeypatch, language, expected, state):
    async def current_weather(**kwargs):
        return {"status": state, "data": {}}

    monkeypatch.setattr(weather_tools, "current_weather", current_weather)

    class Provider(OllamaProvider):
        async def chat(self, **kwargs):
            return LLMResponse(None, [ToolCall("w", "weather.current", {})])

        async def chat_stream(self, **kwargs):
            if False:
                yield ""

    async def run():
        agent = SimpleNamespace(
            _user_scope=UserScope("usr_" + "a" * 24, True),
            _language_session=LanguageSession(language),
        )
        return "".join(
            [part async for part in llm_node(agent, SimpleNamespace(items=[]), Provider())]
        )

    answer = asyncio.run(run())
    assert answer
    if state == "no_location":
        assert expected in answer
    elif state == "unavailable":
        assert ("try again" if language == "en" else "intentarlo") in answer
    else:
        assert ("sign in" if language == "en" else "sesión") in answer
