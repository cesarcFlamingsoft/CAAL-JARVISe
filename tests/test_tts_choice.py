"""Owner-scoped TTS preferences, exercised through real signed API requests."""

import pytest
from fastapi.testclient import TestClient
from test_tts_api import harness as _harness

from caal import webhooks

# Reuse the isolated signed identity fixture.
harness = _harness


def test_member_choice_is_private_and_keeps_global_fallback(harness):
    before = dict(harness.settings)
    with TestClient(webhooks.app) as client:
        r = client.put(
            "/users/me/tts", headers=harness.bearer(harness.member), json={"provider": "piper"}
        )
        assert r.status_code == 200
        assert r.json()["provider"] == "piper"
        assert r.json()["source"] == "personal"
        assert (
            client.get("/users/me/tts", headers=harness.bearer(harness.admin)).json()["provider"]
            == "kokoro"
        )
        assert (
            client.get("/users/me/tts", headers=harness.bearer(harness.member)).json()["provider"]
            == "piper"
        )
    assert harness.settings == before


def test_voicebox_config_is_admin_only_encrypted_and_never_selects_it(harness, monkeypatch):
    import httpx

    from caal import tts_api

    assert hasattr(tts_api, "VOICEBOX_TRANSPORT")
    schema = {
        "paths": {
            "/generate/stream": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/GenerationRequest"}
                            }
                        }
                    }
                }
            },
            "/profiles/{profile_id}": {},
        },
        "components": {
            "schemas": {
                "GenerationRequest": {
                    "properties": {
                        "profile_id": {},
                        "text": {},
                        "engine": {},
                        "model_size": {},
                        "personality": {},
                    }
                }
            }
        },
    }

    def handle(req):
        assert req.headers["authorization"] == "Bearer test-only-credential"
        return httpx.Response(
            200,
            json=schema
            if req.url.path == "/openapi.json"
            else {"status": "ok", "model_loaded": False},
        )

    monkeypatch.setattr(tts_api, "VOICEBOX_TRANSPORT", httpx.MockTransport(handle))
    payload = {"endpoint": "http://127.0.0.1:8000", "credential": "test-only-credential"}
    with TestClient(webhooks.app) as c:
        assert (
            c.put(
                "/users/me/tts/voicebox", headers=harness.bearer(harness.member), json=payload
            ).status_code
            == 403
        )
        r = c.put("/users/me/tts/voicebox", headers=harness.bearer(harness.admin), json=payload)
        assert r.status_code == 200
        assert r.json()["endpoint"] == payload["endpoint"]
        assert r.json()["credential_configured"] is True
        assert "test-only-credential" not in r.text
        assert "test-only-credential" not in harness.store.path.read_bytes().decode(errors="ignore")
        assert (
            c.get("/users/me/tts/voicebox", headers=harness.bearer(harness.member)).status_code
            == 403
        )
        r = c.get("/users/me/tts", headers=harness.bearer(harness.member))
        assert r.json()["provider"] == "kokoro"
        assert payload["endpoint"] not in r.text
        assert (
            c.put(
                "/users/me/tts",
                headers=harness.bearer(harness.member),
                json={"provider": "voicebox"},
            ).status_code
            == 422
        )


def test_voicebox_unconfigured_never_selected_and_status_is_truthful(harness):
    with TestClient(webhooks.app) as c:
        r = c.get("/users/me/tts", headers=harness.bearer(harness.member))
        assert r.json().get("voicebox_status") == "not_configured"
        assert r.json()["can_configure"] is False
        r = c.put(
            "/users/me/tts",
            headers=harness.bearer(harness.member),
            json={
                "provider": "voicebox",
                "profile_id": "explicit-profile",
                "engine": "kokoro",
                "model_size": "1.7B",
            },
        )
        assert r.status_code == 503
        assert (
            c.get("/users/me/tts", headers=harness.bearer(harness.member)).json()["provider"]
            == "kokoro"
        )


def test_global_settings_cannot_select_voicebox(harness):
    with TestClient(webhooks.app) as c:
        assert (
            c.post("/settings", json={"settings": {"tts_provider": "voicebox"}}).status_code == 403
        )


def test_explicit_voicebox_profile_requires_matching_profile_and_cached_model(harness, monkeypatch):
    import httpx

    from caal import tts_api
    from caal.tts_store import TTSStore

    TTSStore(harness.identity).save_config(
        {"endpoint": "http://127.0.0.1:8000", "credential": "", "revision": "one"}
    )
    monkeypatch.setattr(tts_api, "verify_voicebox", _healthy)
    profile = {
        "id": "chosen",
        "voice_type": "preset",
        "preset_engine": "kokoro",
        "preset_voice_id": "am_adam",
        "language": "en",
    }

    def handle(req):
        if req.url.path == "/profiles/chosen":
            return httpx.Response(200, json=profile)
        if req.url.path == "/models/status":
            return httpx.Response(
                200, json={"models": [{"model_name": "kokoro", "downloaded": True}]}
            )
        raise AssertionError("No profile enumeration or other operations allowed")

    monkeypatch.setattr(tts_api, "VOICEBOX_TRANSPORT", httpx.MockTransport(handle))
    choice = {
        "provider": "voicebox",
        "profile_id": "chosen",
        "engine": "kokoro",
        "model_size": "1.7B",
    }
    with TestClient(webhooks.app) as c:
        profile["preset_engine"] = "qwen_custom_voice"
        assert (
            c.put("/users/me/tts", headers=harness.bearer(harness.member), json=choice).status_code
            == 422
        )
        profile["preset_engine"] = "kokoro"
        r = c.put("/users/me/tts", headers=harness.bearer(harness.member), json=choice)
        assert r.status_code == 200
        assert r.json()["profile_id"] == "chosen"
        assert TTSStore(harness.identity).preference(harness.member)["config_revision"] == "one"
        assert (
            c.get("/users/me/tts", headers=harness.bearer(harness.admin)).json()["profile_id"]
            is None
        )


async def _healthy(config):
    return None


@pytest.mark.asyncio
async def test_session_factory_uses_only_verified_owners_preference(harness, monkeypatch):
    from caal import tts_selection
    from caal.tts_store import TTSStore

    assert hasattr(tts_selection, "select_user_tts")
    store = TTSStore(harness.identity)
    store.save_config({"endpoint": "http://127.0.0.1:8000", "credential": "", "revision": "one"})
    store.save_preference(
        harness.member,
        {
            "provider": "voicebox",
            "profile_id": "chosen",
            "engine": "kokoro",
            "model_size": "1.7B",
            "config_revision": "one",
        },
    )

    class Existing:
        closed = False

        async def aclose(self):
            self.closed = True

    runtime = {"tts_provider": "kokoro", "tts_voice_kokoro": "am_adam"}
    kwargs = dict(
        kokoro_url="http://localhost:8001",
        speaches_url="http://localhost:8001",
        kokoro_model="kokoro",
    )
    for owner in [None, harness.admin]:
        default = Existing()
        assert (
            await tts_selection.select_user_tts(
                default, runtime, user_id=owner, identity=harness.identity, **kwargs
            )
            is default
        )
        assert not default.closed
    default = Existing()
    provider = await tts_selection.select_user_tts(
        default, runtime, user_id=harness.member, identity=harness.identity, **kwargs
    )
    assert provider._wrapped_tts.provider == "voicebox-native"
    assert provider._wrapped_tts.profile_id == "chosen"
    assert provider._wrapped_tts.fallback._opts.voice == "am_adam"
    assert default.closed
    await provider.aclose()
    assert runtime["tts_provider"] == "kokoro"
    store.save_config({"endpoint": "http://127.0.0.1:8000", "credential": "", "revision": "two"})
    provider = await tts_selection.select_user_tts(
        Existing(), runtime, user_id=harness.member, identity=harness.identity, **kwargs
    )
    assert provider.model == "kokoro"
    await provider.aclose()


def test_connection_change_is_reported_in_personal_readback(harness, monkeypatch):
    from caal import tts_api
    from caal.tts_store import TTSStore

    monkeypatch.setattr(tts_api, "verify_voicebox", _healthy)
    store = TTSStore(harness.identity)
    store.save_config({"endpoint": "http://127.0.0.1:8000", "credential": "", "revision": "new"})
    store.save_preference(
        harness.member,
        {
            "provider": "voicebox",
            "profile_id": "chosen",
            "engine": "kokoro",
            "model_size": "1.7B",
            "config_revision": "old",
        },
    )
    with TestClient(webhooks.app) as c:
        r = c.get("/users/me/tts", headers=harness.bearer(harness.member))
        assert r.json()["voicebox_status"] == "configuration_changed"


def test_tts_status_checks_are_rate_limited_per_owner(harness):
    from caal.internal_auth import RateLimiter

    harness.identity.mutation_limiter = RateLimiter(limit=1, window_seconds=60)
    with TestClient(webhooks.app) as c:
        assert c.get("/users/me/tts", headers=harness.bearer(harness.member)).status_code == 200
        assert c.get("/users/me/tts", headers=harness.bearer(harness.member)).status_code == 429
        assert c.get("/users/me/tts", headers=harness.bearer(harness.admin)).status_code == 200
