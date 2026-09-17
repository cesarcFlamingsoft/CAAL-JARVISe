"""Opt-in bounded inference through production adapter/catalog and synthetic backend."""
# ruff: noqa: F811

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from livekit.agents.llm import ChatContext
from test_weather import NOW, PARIS, FakeUpstream, _forecast_body, ana, store, users  # noqa: F401

from caal.language_policy import LanguageSession
from caal.llm.llm_node import llm_node
from caal.llm.providers.ollama_provider import OllamaProvider
from caal.tools import weather_tools
from caal.user_scope import UserScope
from caal.weather import WeatherClient

pytestmark = pytest.mark.skipif(
    os.getenv("CAAL_TEST_LOCAL_MODEL") != "1", reason="local inference opt-in"
)


@pytest.mark.parametrize(
    "language,question",
    [
        ("en", "What's the weather like outside?"),
        ("es", "¿Qué tiempo hace ahora?"),
    ],
)
@pytest.mark.parametrize("state", ["ok", "no_location", "unavailable"])
def test_natural_weather_uses_real_adapter_and_catalog(
    monkeypatch, store, ana, language, question, state
):  # noqa: F811
    upstream = FakeUpstream()
    if state != "no_location":
        store.set_city(ana, PARIS, now=NOW)
    upstream.add(
        httpx.Response(200, json=_forecast_body()) if state == "ok" else httpx.Response(503)
    )
    runtime = SimpleNamespace(
        client=WeatherClient(store, transport=upstream.transport, clock=lambda: NOW),
        now=lambda: NOW,
    )
    monkeypatch.setattr(weather_tools, "_runtime", lambda: runtime)
    calls = []
    original = weather_tools.current_weather

    async def record(**kwargs):
        assert kwargs == {"user_id": ana}
        result = await original(**kwargs)
        calls.append(result["status"])
        return result

    monkeypatch.setattr(weather_tools, "current_weather", record)
    provider = OllamaProvider(
        model=os.getenv("CAAL_TEST_OLLAMA_MODEL", "gemma4:e4b"),
        base_url=os.getenv("CAAL_TEST_OLLAMA_HOST", "http://10.0.0.64:11434"),
        temperature=0,
        num_ctx=32768,
    )
    # Bound the sync client too: asyncio cancellation cannot stop a worker thread.
    provider._client._client.timeout = httpx.Timeout(30, connect=2)
    ctx = ChatContext()
    ctx.add_message(role="system", content=Path("prompt/default.md").read_text())
    ctx.add_message(role="user", content=question)
    agent = SimpleNamespace(
        _user_scope=UserScope(ana, True), _language_session=LanguageSession(language)
    )

    async def run():
        return "".join([part async for part in llm_node(agent, ctx, provider)])

    answer = asyncio.run(asyncio.wait_for(run(), 45))
    assert calls == [state], (calls, answer)
    assert answer
    if state == "ok":
        assert any(
            value in answer.lower()
            for value in (
                "7.4",
                "7,4",
                "seven point four",
                "siete punto cuatro",
                "siete coma cuatro",
            )
        )
        assert not any(
            p in answer.lower() for p in ["no tengo acceso", "no access", "cannot access"]
        )
    else:
        assert answer == weather_tools.spoken_failure({"status": state}, language)
    print(
        json.dumps(
            {"language": language, "state": state, "calls": calls, "answer": answer},
            ensure_ascii=False,
        )
    )
