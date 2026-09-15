import json
from types import SimpleNamespace

import pytest
from test_ha_access import h as ha_fixture

from caal.user_scope import UserScope

h = ha_fixture


@pytest.mark.asyncio
async def test_real_dispatch_uses_bound_connection_and_rechecks_revocation(h):
    from caal.ha_access import HAStore
    from caal.ha_assist import create_tools

    store = HAStore(h)
    cid = store.save_connection(
        h.member.user_id,
        {"id": "member-ha"},
        {"access_token": "member-token", "refresh_token": "refresh"},
        endpoint="http://127.0.0.1:8123",
        client_id="https://jarvis.example.com",
    )
    store.grant(h.admin.user_id, h.member.user_id, enabled=True, connection_id=cid)
    calls = []

    class Client:
        async def current_user(self, token):
            return {"id": "member-ha"}

        async def request(self, method, path, **kwargs):
            calls.append((method, path, kwargs))
            return [
                {
                    "entity_id": "light.office",
                    "state": "off",
                    "attributes": {"friendly_name": "Office"},
                }
            ]

    class Model:
        manages_own_tools = False

        async def chat(self, *args, **kwargs):
            return SimpleNamespace(
                content=json.dumps({"operation": "read", "entities": ["light.office"]})
            )

    definitions, tools = create_tools(
        scope=UserScope.for_user(h.member),
        identity=h,
        provider=Model(),
        settings_getter=lambda: {"hass_enabled": True, "hass_host": "http://127.0.0.1:8123"},
        client_factory=lambda endpoint: Client(),
    )
    assert definitions[0]["function"]["name"] == "hass_assist"
    assert "off" in await tools["hass_assist"]("Is the office light on?")
    assert calls[0][2]["token"] == "member-token"
    assert all("conversation_id" not in str(c) for c in calls)
    store.grant(h.admin.user_id, h.member.user_id, enabled=False, connection_id=None)
    assert "not granted" in await tools["hass_assist"]("Is it on?")
    assert len(calls) == 1
    with pytest.raises(TypeError):
        await tools["hass_assist"]("read", user_id=h.admin.user_id)


@pytest.mark.asyncio
async def test_security_actions_never_reach_provider(h):
    from caal.ha_assist import validate_plan

    for plan in [
        {"operation": "unlock", "entities": ["lock.front"]},
        {"operation": "turn_on", "entities": ["cover.garage"]},
        {"operation": "turn_on", "entities": []},
        {"operation": "read", "entities": ["light.office"], "user_id": "fake"},
    ]:
        with pytest.raises(ValueError):
            validate_plan(plan, {"light.office", "cover.garage", "lock.front"})


@pytest.mark.asyncio
async def test_voice_factory_without_verified_scope_does_not_use_global_token():
    from test_runtime_imports import _load_voice_agent_module

    voice = _load_voice_agent_module()
    _, tools = voice.create_hass_tools(
        "http://127.0.0.1:8123", "global-owner-token", "conversation.stale"
    )
    assert "not granted" in await tools["hass_assist"]("Check lights")


@pytest.mark.asyncio
async def test_revocation_during_planning_prevents_light_action(h):
    from caal.ha_access import HAStore
    from caal.ha_assist import create_tools

    store = HAStore(h)
    calls = []

    class Client:
        async def current_user(self, token):
            return {"id": "owner"}

        async def request(self, method, path, **kwargs):
            calls.append(method)
            assert method == "GET"
            return [{"entity_id": "light.office", "state": "off", "attributes": {}}]

    class Model:
        manages_own_tools = False

        async def chat(self, *args, **kwargs):
            store.grant(h.admin.user_id, h.admin.user_id, enabled=False, connection_id=None)
            return SimpleNamespace(
                content=json.dumps({"operation": "turn_on", "entities": ["light.office"]})
            )

    _, tools = create_tools(
        scope=UserScope.for_user(h.admin),
        identity=h,
        provider=Model(),
        settings_getter=lambda: {
            "hass_enabled": True,
            "hass_host": "http://127.0.0.1:8123",
            "hass_token": "owner-token",
        },
        client_factory=lambda _: Client(),
    )
    assert "authorization changed" in await tools["hass_assist"]("Turn on office")
    assert calls == ["GET"]


@pytest.mark.asyncio
async def test_provider_rejected_token_marks_connection_required(h):
    from caal.ha_access import HAStore
    from caal.ha_assist import create_tools

    store = HAStore(h)
    cid = store.save_connection(
        h.member.user_id,
        {"id": "ha-member"},
        {"access_token": "revoked", "refresh_token": "refresh"},
        endpoint="http://127.0.0.1:8123",
        client_id="https://jarvis.example.com",
    )
    store.grant(h.admin.user_id, h.member.user_id, enabled=True, connection_id=cid)
    scope = UserScope.for_user(h.member)

    class Client:
        async def current_user(self, token):
            raise PermissionError("ha_reconnect_required")

    _, tools = create_tools(
        scope=scope,
        identity=h,
        provider=None,
        settings_getter=lambda: {"hass_enabled": True, "hass_host": "http://127.0.0.1:8123"},
        client_factory=lambda _: Client(),
    )
    assert "expired" in await tools["hass_assist"]("Check light state")
    assert store.access(scope)["status"] == "connection_required"
