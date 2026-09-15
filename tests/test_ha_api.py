import pytest
from fastapi.testclient import TestClient
from test_local_model_api import Harness

from caal import user_api, webhooks


@pytest.fixture
def h(tmp_path):
    h = Harness(tmp_path)
    webhooks.app.dependency_overrides[user_api.get_runtime] = lambda: h.identity
    yield h
    webhooks.app.dependency_overrides.pop(user_api.get_runtime, None)


def test_admin_grant_roundtrip_and_member_cannot_grant(h):
    with TestClient(webhooks.app) as c:
        path = f"/admin/users/{h.member}/home-assistant"
        assert c.get("/users/me/home-assistant").status_code == 401
        assert (
            c.put(
                path, headers=h.bearer(h.member), json={"enabled": True, "connection_id": None}
            ).status_code
            == 403
        )
        assert (
            c.put(
                path, headers=h.bearer(h.admin), json={"enabled": True, "connection_id": None}
            ).status_code
            == 200
        )
        r = c.get("/users/me/home-assistant", headers=h.bearer(h.member))
        assert r.status_code == 200
        assert r.json()["enabled"] is True
        assert r.json()["status"] == "connection_required"
        assert r.headers["cache-control"] == "no-store"
        assert (
            c.put(
                path,
                headers=h.bearer(h.admin),
                json={"enabled": False, "connection_id": None, "ha_user_id": "fake"},
            ).status_code
            == 422
        )


def test_oauth_callback_binds_real_identity_and_never_returns_tokens(h, monkeypatch):
    from caal import ha_api
    from caal.ha_access import HAStore

    monkeypatch.setenv("CAAL_PUBLIC_ORIGIN", "https://jarvis.example.com")
    monkeypatch.setattr(
        ha_api.settings,
        "load_settings",
        lambda: {"hass_enabled": True, "hass_host": "http://127.0.0.1:8123"},
    )

    async def request(self, method, path, **kwargs):
        assert path == "/auth/token"
        return {
            "access_token": "private-access",
            "refresh_token": "private-refresh",
            "expires_in": 1800,
        }

    async def current_user(self, token):
        assert token == "private-access"
        return {"id": "actual-ha-identity", "name": "HA member", "is_admin": False}

    monkeypatch.setattr(ha_api.HAClient, "request", request)
    monkeypatch.setattr(ha_api.HAClient, "current_user", current_user)
    with TestClient(webhooks.app) as c:
        started = c.post("/users/me/home-assistant/authorize", headers=h.bearer(h.member))
        assert started.status_code == 200
        state = started.json()["state_id"]
        wrong = c.post(
            "/users/me/home-assistant/callback",
            headers=h.bearer(h.admin),
            json={"state": state, "code": "fixture-code"},
        )
        assert wrong.status_code == 400
        result = c.post(
            "/users/me/home-assistant/callback",
            headers=h.bearer(h.member),
            json={"state": state, "code": "fixture-code"},
        )
        assert result.status_code == 200
        assert "private" not in result.text and "fixture-code" not in result.text
        replay = c.post(
            "/users/me/home-assistant/callback",
            headers=h.bearer(h.member),
            json={"state": state, "code": "fixture-code"},
        )
        assert replay.status_code == 400
        own = c.get("/users/me/home-assistant", headers=h.bearer(h.member)).json()
        assert own["status"] == "denied"  # Linking never grants a member capability.
        assert own["connections"][0]["label"] == "HA member"
        assert "private" not in str(own)
        assert (
            HAStore(h.identity).connection(own["connections"][0]["id"])["provider_user"]["id"]
            == "actual-ha-identity"
        )
