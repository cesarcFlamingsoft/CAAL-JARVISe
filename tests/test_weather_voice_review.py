"""Review regressions for dashboard parity and weather's private dispatch."""
import asyncio
import importlib
import json
from types import SimpleNamespace

import httpx
from test_weather import NOW, PARIS, FakeUpstream, _forecast_body, ana, store, users  # noqa: F401

from caal import weather_api
from caal.tools import weather_tools
from caal.tools.registry import create_default_registry
from caal.user_scope import UserScope
from caal.weather import WeatherClient

node = importlib.import_module("caal.llm.llm_node")


def test_weather_matches_dashboard_response_with_missing_measurements(monkeypatch, store, ana):  # noqa: F811
    body = _forecast_body()
    body["current"]["rain"] = None
    upstream = FakeUpstream()
    upstream.add(httpx.Response(200, json=body))
    store.set_city(ana, PARIS, now=NOW)
    runtime = SimpleNamespace(
        client=WeatherClient(store, transport=upstream.transport, clock=lambda: NOW),
        now=lambda: NOW,
    )
    monkeypatch.setattr(weather_tools, "_runtime", lambda: runtime)

    async def read():
        voice = await weather_tools.current_weather(user_id=ana)
        dashboard = await weather_api.current_weather(
            user=SimpleNamespace(profile=SimpleNamespace(user_id=ana)), runtime=runtime
        )
        return voice, dashboard.model_dump()

    voice, dashboard = asyncio.run(read())
    assert voice["status"] == "ok"
    assert voice["data"] == dashboard
    assert voice["data"]["observation"]["rain_mm"] is None
    assert "never none expected" in voice["message"]
    assert len(upstream.forecast_calls) == 1


def test_unverified_scope_and_location_overrides_never_open_weather(monkeypatch, ana):  # noqa: F811
    def forbidden():
        raise AssertionError("weather runtime must not be opened")

    monkeypatch.setattr(weather_tools, "_runtime", forbidden)
    for scope in (UserScope.anonymous(), UserScope.legacy(), None):
        agent = SimpleNamespace(_user_scope=scope, _native_tool_registry=create_default_registry())
        result = asyncio.run(node._execute_single_tool(agent, "weather.current", {"user_id": ana}))
        assert result["status"] == "unauthorized"
        assert result["data"] == {}
    agent._user_scope = UserScope(ana, True)
    for arguments in ({"city": "Tokyo"}, {"latitude": 12}, {"days": 7}):
        result = asyncio.run(node._execute_single_tool(agent, "weather.current", arguments))
        assert result["status"] == "invalid_request"
        assert "Tokyo" not in json.dumps(result)
