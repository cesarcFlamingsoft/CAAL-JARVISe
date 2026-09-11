"""A saved endpoint is the one JARVIS actually uses, not merely $OLLAMA_HOST.

The settings UI writes ``ollama_host``; if the runtime kept reading the
environment variable instead, saving would appear to work and change nothing.
These tests pin the two places where the saved value has to win: the provider
built for a new session, and the model list the settings UI reads.

They also pin that a saved value which is no longer acceptable (someone edited
settings.json by hand and put a public address in it) is passed over rather
than used or allowed to stop the assistant from starting.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from caal import webhooks
from caal.llm.providers import create_provider_from_settings
from caal.llm.providers.routed_provider import RoutedProvider


def _settings(**overrides):
    base = dict(
        llm_provider="routed",
        ollama_host="http://192.168.1.50:11434",
        ollama_model="llama3.2:3b",
    )
    base.update(overrides)
    return base


def test_a_new_session_uses_the_saved_endpoint(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://host.docker.internal:11434")
    provider = create_provider_from_settings(_settings())
    assert isinstance(provider, RoutedProvider)
    local = provider.primary
    assert local.provider_name == "ollama"
    assert local.base_url == "http://192.168.1.50:11434"
    assert local.model == "llama3.2:3b"


def test_a_pinned_ollama_provider_uses_the_saved_endpoint_too(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    provider = create_provider_from_settings(_settings(llm_provider="ollama"))
    assert provider.base_url == "http://192.168.1.50:11434"


def test_an_unacceptable_saved_endpoint_does_not_stop_a_session(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    provider = create_provider_from_settings(_settings(ollama_host="https://ollama.example.com"))
    assert provider.primary.base_url == "http://localhost:11434"


def test_the_model_list_route_reads_the_saved_endpoint(monkeypatch):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})

    monkeypatch.setenv("OLLAMA_HOST", "http://host.docker.internal:11434")
    monkeypatch.setattr(webhooks.settings_module, "load_settings", lambda: _settings())
    monkeypatch.setattr(webhooks, "MODEL_DISCOVERY_TRANSPORT", httpx.MockTransport(handler))
    with TestClient(webhooks.app) as client:
        response = client.get("/models")
    assert response.status_code == 200
    assert response.json()["models"] == ["qwen3:8b"]
    assert seen == ["http://192.168.1.50:11434/api/tags"]


def test_the_model_list_route_answers_an_unreachable_ollama_with_nothing(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(webhooks.settings_module, "load_settings", lambda: _settings())
    monkeypatch.setattr(webhooks, "MODEL_DISCOVERY_TRANSPORT", httpx.MockTransport(handler))
    with TestClient(webhooks.app) as client:
        response = client.get("/models")
    assert response.status_code == 200
    assert response.json()["models"] == []


@pytest.mark.parametrize(
    "host",
    [
        "https://ollama.example.com",
        "http://ollama.example.com:11434",
        "http://8.8.8.8:11434",
        "http://user:secret@localhost:11434",
    ],
)
def test_the_legacy_test_route_refuses_a_host_off_the_local_network(monkeypatch, host):
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never runs
        raise AssertionError("no request may be made to a non-local endpoint")

    monkeypatch.setattr(webhooks, "MODEL_DISCOVERY_TRANSPORT", httpx.MockTransport(handler))
    with TestClient(webhooks.app) as client:
        response = client.post("/setup/test-ollama", json={"host": host})
    body = response.json()
    assert body["success"] is False
    assert body["error"]
    assert "secret" not in response.text


def test_the_legacy_test_route_still_lists_models_for_a_local_host(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})

    monkeypatch.setattr(webhooks, "MODEL_DISCOVERY_TRANSPORT", httpx.MockTransport(handler))
    with TestClient(webhooks.app) as client:
        response = client.post("/setup/test-ollama", json={"host": "http://10.0.0.12:11434"})
    body = response.json()
    assert body["success"] is True
    assert body["models"] == ["qwen3:8b"]
