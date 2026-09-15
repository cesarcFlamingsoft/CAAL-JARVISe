from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_unsupported_delegation_refuses_before_transport(monkeypatch):
    from caal import ha_policy
    from caal.llm.providers.hermes_provider import HermesProvider

    monkeypatch.setattr(ha_policy, "restricted_runtime", lambda: True)
    provider = HermesProvider(api_key="fixture-only")

    async def forbidden():
        raise AssertionError("network must not be opened")

    monkeypatch.setattr(provider, "_get_client", forbidden)
    with pytest.raises(PermissionError):
        await provider.chat([{"role": "user", "content": "Ask HA to unlock"}])
    with pytest.raises(PermissionError):
        async for _ in provider.chat_stream([{"role": "user", "content": "Ask HA to unlock"}]):
            pass


@pytest.mark.asyncio
async def test_raw_home_assistant_and_friday_dispatch_are_refused(monkeypatch):
    from caal import ha_policy
    from caal.llm.llm_node import _execute_single_tool

    monkeypatch.setattr(ha_policy, "restricted_runtime", lambda: True)
    for name in ("home_assistant__unlock", "n8n__hass_control", "ask_friday"):
        result = await _execute_single_tool(SimpleNamespace(), name, {})
        assert result["status"] == "unsupported_tool"


def test_unscoped_durable_work_cannot_be_queued_for_old_worker(monkeypatch):
    from caal import background_tasks, ha_policy

    monkeypatch.setattr(ha_policy, "restricted_runtime", lambda: True)
    with pytest.raises(PermissionError):
        background_tasks.enqueue("Fixture task")
