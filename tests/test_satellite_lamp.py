"""Real native dispatch with synthetic owner grants and a final fake HA transport."""

import asyncio
import json
import logging

import httpx
import pytest
from test_local_model_api import Harness
from test_satellite_rollout import LIVING, LIVING_DEVICE, catalog

from caal.ha_access import HAStore
from caal.ha_client import HAClient
from caal.llm.llm_node import _execute_single_tool
from caal.satellite import DEVICE, PILOT, SatelliteStore, restricted_agent
from caal.satellite_home import SatelliteHome


@pytest.fixture(autouse=True)
def capture_satellite_log(caplog):
    # Other integration tests install a non-propagating CAAL log handler.
    # Capture this logger directly without changing the application's setup.
    logger = logging.getLogger("caal.llm.llm_node")
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger=logger.name)
    try:
        yield
    finally:
        logger.removeHandler(caplog.handler)


@pytest.fixture
def lamp(tmp_path):
    h = Harness(tmp_path)
    store = SatelliteStore(h.identity)
    hs = HAStore(h.identity)
    cid = hs.save_connection(
        h.admin,
        {"id": "ha-owner", "is_admin": True},
        {"access_token": "fixture-token", "refresh_token": "fixture-refresh", "expires_in": 3600},
        endpoint="http://127.0.0.1:8123",
        client_id="https://fixture.invalid",
    )
    catalog(store, cid)
    principals = [
        store.authenticate(store.enroll(h.admin, satellite_id=e, connection_id=cid)["credential"])
        for e in (PILOT, LIVING)
    ]
    for p in principals:
        store.configure(h.admin, p.id, connection_id=cid, scope="states_and_lights")
    writes = []
    states = [
        {
            "entity_id": "light.bedroom_lamp",
            "state": "on",
            "attributes": {"friendly_name": "Bedroom lamp"},
        }
    ]

    def transport(request):
        if request.method == "GET":
            return httpx.Response(200, json=states)
        writes.append(json.loads(request.content))
        return httpx.Response(200, json=[])

    class Client(HAClient):
        async def current_user(self, token):
            assert token == "fixture-token"
            return {"id": "ha-owner", "is_admin": True}

        async def registry(self, token):
            return [
                {"entity_id": e, "device_id": d, "config_entry_id": "fixture"}
                for e, d in ((PILOT, DEVICE), (LIVING, LIVING_DEVICE))
            ]

    homes = [
        SatelliteHome(
            store,
            p,
            client_factory=lambda endpoint: Client(
                endpoint, transport=httpx.MockTransport(transport)
            ),
            settings_getter=lambda: {"hass_enabled": True, "hass_host": "http://127.0.0.1:8123"},
        )
        for p in principals
    ]
    return h, store, hs, cid, homes, writes, states


@pytest.mark.asyncio
async def test_unknown_light_is_resolvable_target_error_not_authorization(lamp, caplog):
    _, _, _, _, homes, writes, _ = lamp
    caplog.set_level(logging.INFO)
    agent = restricted_agent(home=homes[0])
    assert (await _execute_single_tool(agent, "home.states", {}))["status"] == "ok"
    result = await _execute_single_tool(
        agent,
        "home.light",
        {
            "operation": "turn_off",
            "entity_ids": ["light.fabricated_private_target"],
        },
    )
    assert result["status"] == "target_unavailable"
    assert result["reason_code"] == "light_not_available"
    assert "home.states" in result["message"] and "clarif" in result["message"]
    assert "permission" not in result["message"].lower()
    assert writes == []
    assert "reason_code=light_not_available" in caplog.text
    assert "fabricated_private_target" not in caplog.text
    retry = await _execute_single_tool(
        agent,
        "home.light",
        {
            "operation": "turn_off",
            "entity_ids": ["light.bedroom_lamp"],
        },
    )
    assert retry["status"] == "ok"
    assert writes == [{"entity_id": ["light.bedroom_lamp"]}]


@pytest.mark.asyncio
async def test_satellite_continues_read_then_light_and_resolves_failed_target(lamp):
    from livekit.agents import llm

    from caal.llm.llm_node import llm_node
    from caal.llm.providers.base import LLMProvider, LLMResponse, ToolCall

    _, _, _, _, homes, writes, _ = lamp

    class Model(LLMProvider):
        provider_name = "fixture"
        model = "fixture"

        async def chat(self, messages, tools=None, **kwargs):
            results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
            assert {t["function"]["name"] for t in tools} == {
                "home.states",
                "home.light",
                "satellite.capabilities",
            }
            if not results:
                return LLMResponse(None, [ToolCall("1", "home.states", {})])
            if len(results) == 1:
                assert results[-1]["data"]["states"][0]["entity_id"] == "light.bedroom_lamp"
                return LLMResponse(
                    None,
                    [
                        ToolCall(
                            "2",
                            "home.light",
                            {"operation": "turn_off", "entity_ids": ["light.invented"]},
                        )
                    ],
                )
            if len(results) == 2:
                assert results[-1]["status"] == "target_unavailable"
                return LLMResponse(None, [ToolCall("3", "home.states", {})])
            if len(results) == 3:
                return LLMResponse(
                    None,
                    [
                        ToolCall(
                            "4",
                            "home.light",
                            {
                                "operation": "turn_off",
                                "entity_ids": [results[-1]["data"]["states"][0]["entity_id"]],
                            },
                        )
                    ],
                )
            assert results[-1]["status"] == "ok"
            return LLMResponse("Home Assistant accepted the request.", [])

        async def chat_stream(self, *args, **kwargs):
            raise ValueError("unexpected_model_stream")
            yield "unreachable"

    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="Turn off the bedroom lamp.")
    reply = "".join([c async for c in llm_node(restricted_agent(home=homes[0]), ctx, Model())])
    assert reply == "Home Assistant accepted the light off request."
    assert writes == [{"entity_id": ["light.bedroom_lamp"]}]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["accepted", "timeout", "revoked_after_write"])
async def test_model_repeated_light_never_duplicates_an_accepted_or_uncertain_write(lamp, outcome):
    h, store, _, _, homes, writes, _ = lamp
    home = homes[0]
    factory = home.client_factory

    def client(endpoint):
        c = factory(endpoint)
        original = c.request

        async def request(method, path, **kwargs):
            result = await original(method, path, **kwargs)
            if method == "POST":
                if outcome == "timeout":
                    raise TimeoutError("private upstream URL")
                if outcome == "revoked_after_write":
                    store.revoke(h.admin, home.principal.id)
            return result

        c.request = request
        return c

    home.client_factory = client
    agent = restricted_agent(home=home)
    args = {"operation": "turn_off", "entity_ids": ["light.bedroom_lamp"]}
    first = await _execute_single_tool(agent, "home.light", args)
    second = await _execute_single_tool(agent, "home.light", args)
    assert len(writes) == 1
    assert first["status"] == ("ok" if outcome == "accepted" else "action_uncertain")
    if outcome != "accepted":
        assert "private upstream" not in str(first)
        assert second["status"] != "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        "ha_connection_changed",
        "ha_credentials_changed",
        "ha_reconnect_required",
        "ha_permission_denied",
        "satellite_grant_changed",
        "satellite_registry_changed",
        "satellite_revoked",
        "private URL token=secret",
    ],
)
async def test_permission_reason_is_allowlisted_and_content_free(lamp, caplog, reason):
    _, _, _, _, homes, _, _ = lamp
    caplog.set_level(logging.INFO)
    home = homes[0]

    async def denied(*args, **kwargs):
        raise PermissionError(reason)

    home._context = denied
    result = await _execute_single_tool(
        restricted_agent(home=home),
        "home.light",
        {"operation": "turn_off", "entity_ids": ["light.bedroom_lamp"]},
    )
    expected = reason if reason != "private URL token=secret" else "satellite_permission_denied"
    assert result["status"] == "unauthorized"
    assert result["reason_code"] == expected
    assert "reason_code=" + expected in caplog.text
    assert "private URL" not in str(result) + caplog.text


@pytest.mark.asyncio
async def test_invalid_operation_is_argument_error_not_permission_error(lamp):
    home = lamp[4][0]
    result = await _execute_single_tool(
        restricted_agent(home=home),
        "home.light",
        {"operation": "toggle", "entity_ids": ["light.bedroom_lamp"]},
    )
    assert result["status"] == "invalid_request"
    assert result["reason_code"] == "invalid_arguments"


@pytest.mark.asyncio
async def test_satellite_cannot_claim_success_after_a_refused_write(lamp):
    from livekit.agents import llm

    from caal.llm.llm_node import llm_node
    from caal.llm.providers.base import LLMProvider, LLMResponse, ToolCall

    class Model(LLMProvider):
        provider_name = "fixture"
        model = "fixture"

        async def chat(self, messages, tools=None, **kwargs):
            if any(m["role"] == "tool" for m in messages):
                return LLMResponse("The lamp has been turned off.", [])
            return LLMResponse(
                None,
                [
                    ToolCall(
                        "1",
                        "home.light",
                        {"operation": "turn_off", "entity_ids": ["light.fabricated"]},
                    )
                ],
            )

        async def chat_stream(self, *args, **kwargs):
            yield "The lamp has been turned off."

    ctx = llm.ChatContext()
    ctx.add_message(role="user", content="Turn off the lamp.")
    reply = "".join([x async for x in llm_node(restricted_agent(home=lamp[4][0]), ctx, Model())])
    assert "turned off" not in reply
    assert "Which lamp" in reply
    assert not lamp[5]


@pytest.mark.asyncio
async def test_both_real_grant_paths_remain_independent_under_revocation(lamp):
    h, store, _, _, homes, writes, _ = lamp
    agents = [restricted_agent(home=x) for x in homes]
    reads = await asyncio.gather(*[_execute_single_tool(a, "home.states", {}) for a in agents])
    assert all(r["status"] == "ok" for r in reads)
    store.revoke(h.admin, homes[0].principal.id)
    results = await asyncio.gather(
        *[
            _execute_single_tool(
                a,
                "home.light",
                {
                    "operation": "turn_off",
                    "entity_ids": ["light.bedroom_lamp"],
                },
            )
            for a in agents
        ]
    )
    assert results[0]["reason_code"] == "satellite_revoked"
    assert results[1]["status"] == "ok"
    assert len(writes) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["grant", "connection", "owner"])
async def test_changed_authority_during_state_read_prevents_any_write(lamp, change):
    h, store, hs, cid, homes, writes, _ = lamp
    home = homes[0]
    factory = home.client_factory

    def client(endpoint):
        c = factory(endpoint)
        original = c.request

        async def request(method, path, **kwargs):
            result = await original(method, path, **kwargs)
            if method == "GET":
                if change == "grant":
                    store.configure(h.admin, home.principal.id, connection_id=cid, scope="states")
                elif change == "connection":
                    hs.refresh_connection(
                        hs.connection(cid), {"access_token": "rotated", "expires_in": 3600}
                    )
                else:
                    hs.disconnect(h.admin, cid)
            return result

        c.request = request
        return c

    home.client_factory = client
    result = await _execute_single_tool(
        restricted_agent(home=home),
        "home.light",
        {
            "operation": "turn_off",
            "entity_ids": ["light.bedroom_lamp"],
        },
    )
    assert result["status"] == "unauthorized"
    assert result["reason_code"] in {"satellite_grant_changed", "ha_connection_changed"}
    assert not writes


@pytest.mark.asyncio
async def test_expired_connection_refresh_keeps_owner_identity_and_concurrent_cas(lamp):
    h, store, hs, cid, homes, _, _ = lamp
    hs.refresh_connection(hs.connection(cid), {"access_token": "expired", "expires_in": -10})
    arrived = asyncio.Event()
    waiting = 0
    tokens = []

    class RefreshClient(HAClient):
        async def current_user(self, token):
            assert token in tokens
            return {"id": "ha-owner"}

        async def request(self, method, path, **kwargs):
            nonlocal waiting
            assert (method, path) == ("POST", "/auth/token")
            assert kwargs["form"]["refresh_token"] == "fixture-refresh"
            waiting += 1
            token = "fresh-" + str(waiting)
            tokens.append(token)
            if waiting == 2:
                arrived.set()
            await arrived.wait()
            return {"access_token": token, "expires_in": 3600}

    from caal.satellite_home import connection_credentials

    async def refresh():
        return await connection_credentials(
            store, cid, client_factory=RefreshClient, settings_getter=homes[0].settings_getter
        )

    results = await asyncio.gather(refresh(), refresh(), return_exceptions=True)
    assert sum(isinstance(r, PermissionError) for r in results) == 1
    assert [r.args for r in results if isinstance(r, PermissionError)] == [
        ("ha_credentials_changed",)
    ]
    assert hs.connection(cid)["owner_id"] == h.admin
    assert hs.connection(cid)["access_token"] in tokens


@pytest.mark.asyncio
async def test_state_paging_exposes_exact_ids_and_rejects_switch_actuation(lamp):
    _, _, _, _, homes, writes, states = lamp
    states[:] = [
        {"entity_id": f"light.fixture_{i:03}", "state": "on", "attributes": {}} for i in range(34)
    ]
    states.append(
        {
            "entity_id": "switch.bedside_lamp",
            "state": "on",
            "attributes": {"friendly_name": "Bedside lamp"},
        }
    )
    agent = restricted_agent(home=homes[0])
    first = await _execute_single_tool(agent, "home.states", {})
    second = await _execute_single_tool(
        agent, "home.states", {"offset": first["data"]["next_offset"]}
    )
    assert len(first["data"]["states"]) == 32
    assert len(second["data"]["states"]) == 2 and second["data"]["next_offset"] is None
    assert (
        await _execute_single_tool(
            agent, "home.light", {"operation": "turn_off", "entity_ids": ["switch.bedside_lamp"]}
        )
    )["status"] == "invalid_request"
    assert not writes
