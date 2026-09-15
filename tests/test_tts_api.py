"""TTS settings inherit the signed-principal boundary without returning infrastructure secrets."""

import pytest
from fastapi.testclient import TestClient
from test_local_model_api import Harness

from caal import tts_api, user_api, webhooks


@pytest.fixture
def harness(tmp_path, monkeypatch):
    h = Harness(tmp_path)
    h.settings.update(tts_provider="kokoro", tts_voice_kokoro="am_adam")
    monkeypatch.setattr(tts_api.settings_module, "load_settings", lambda: dict(h.settings))
    monkeypatch.setattr(tts_api.settings_module, "save_settings", h._save)
    webhooks.app.dependency_overrides[user_api.get_runtime] = lambda: h.identity
    yield h
    webhooks.app.dependency_overrides.pop(user_api.get_runtime, None)


def test_anonymous_cannot_change_tts_and_member_can_save_personal_choice(harness):
    with TestClient(webhooks.app) as c:
        assert c.get("/users/me/tts").status_code == 401
        assert c.put("/users/me/tts", json={"provider": "kokoro"}).status_code == 401
        assert (
            c.put(
                "/users/me/tts", json={"provider": "kokoro"}, headers=harness.bearer(harness.member)
            ).status_code
            == 200
        )
        r = c.get("/users/me/tts", headers=harness.bearer(harness.member))
        assert r.status_code == 200
        assert r.json()["provider"] == "kokoro"
        assert "secret" not in r.text and "token" not in r.text
        assert r.headers["cache-control"] == "no-store"
    assert harness.saved == []


def test_admin_roundtrip_changes_only_personal_tts(harness):
    before = dict(harness.settings)
    with TestClient(webhooks.app) as c:
        for provider in ["piper", "kokoro"]:
            r = c.put(
                "/users/me/tts", json={"provider": provider}, headers=harness.bearer(harness.admin)
            )
            assert r.status_code == 200
            assert r.json()["provider"] == provider
            assert harness.settings == before
        r = c.put(
            "/users/me/tts", json={"provider": "qwen-trial"}, headers=harness.bearer(harness.admin)
        )
        assert r.status_code == 503
        r = c.put("/users/me/tts", json={"provider": "bad"}, headers=harness.bearer(harness.admin))
        assert r.status_code == 422
    assert harness.settings == before


def test_qwen_can_be_selected_only_after_authenticated_health_check(harness, monkeypatch):
    import httpx

    monkeypatch.setenv("CAAL_QWEN_TRIAL_TOKEN", "a" * 32)
    calls = []

    def handler(req):
        calls.append(req)
        assert str(req.url) == "http://host.docker.internal:18003/health"
        assert req.headers["authorization"] == "Bearer " + "a" * 32
        return httpx.Response(200, json={"status": "ok", "busy": False})

    monkeypatch.setattr(tts_api, "HEALTH_TRANSPORT", httpx.MockTransport(handler), raising=False)
    with TestClient(webhooks.app) as c:
        r = c.put(
            "/users/me/tts", json={"provider": "qwen-trial"}, headers=harness.bearer(harness.admin)
        )
        assert r.status_code == 200
        assert r.json()["provider"] == "qwen-trial"
        assert len(calls) == 1
        assert "a" * 32 not in r.text


def test_failed_trial_health_does_not_change_settings(harness, monkeypatch):
    import httpx

    monkeypatch.setenv("CAAL_QWEN_TRIAL_TOKEN", "a" * 32)
    monkeypatch.setattr(
        tts_api,
        "HEALTH_TRANSPORT",
        httpx.MockTransport(lambda req: httpx.Response(500, text="private error detail")),
    )
    before = dict(harness.settings)
    with TestClient(webhooks.app) as c:
        response = c.put(
            "/users/me/tts", json={"provider": "qwen-trial"}, headers=harness.bearer(harness.admin)
        )
    assert response.status_code == 503
    assert harness.settings == before
    assert "private error" not in response.text


def test_real_settings_file_roundtrip_preserves_unrelated_fields(tmp_path, monkeypatch):
    from caal import settings

    path = tmp_path / "settings.json"
    monkeypatch.setattr(settings, "SETTINGS_PATH", path)
    monkeypatch.setattr(settings, "_settings_cache", None)
    settings.save_settings(
        {"tts_provider": "kokoro", "tts_voice_kokoro": "am_adam", "ollama_model": "unchanged-model"}
    )
    settings.save_settings({"tts_provider": "qwen-trial"})
    stored = settings.load_settings()
    assert stored["tts_provider"] == "qwen-trial"
    assert stored["tts_voice_kokoro"] == "am_adam"
    assert stored["ollama_model"] == "unchanged-model"


def test_legacy_settings_cannot_bypass_authenticated_qwen_opt_in(harness):
    with TestClient(webhooks.app) as c:
        response = c.post("/settings", json={"settings": {"tts_provider": "qwen-trial"}})
    assert response.status_code == 403
    assert harness.saved == []


def test_legacy_setup_cannot_bypass_authenticated_qwen_opt_in(harness):
    with TestClient(webhooks.app) as c:
        response = c.post(
            "/setup/complete", json={"llm_provider": "ollama", "tts_provider": "qwen-trial"}
        )
    assert response.status_code == 403
    assert harness.saved == []
