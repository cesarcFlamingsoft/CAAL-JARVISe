"""Admin harness authority uses real isolated identity verification, never live dispatch."""

from types import SimpleNamespace

import pytest
from test_session_identity import Identity, _load_voice_agent


@pytest.fixture
def identity(tmp_path, monkeypatch):
    from caal import background_tasks, conversation_ledger

    value = Identity(tmp_path)
    monkeypatch.setattr(background_tasks, "STORE_PATH", tmp_path / "assistant.sqlite3")
    monkeypatch.setattr(conversation_ledger, "STORE_PATH", tmp_path / "assistant.sqlite3")
    return value


@pytest.mark.asyncio
async def test_verified_admin_friday_dispatch_preserved(identity, monkeypatch):
    from caal import ha_policy
    from caal.llm.llm_node import _execute_single_tool

    monkeypatch.setattr(ha_policy, "restricted_runtime", lambda: True)
    voice = _load_voice_agent()
    scope = voice.resolve_inbound_scope(
        identity.job_metadata(identity.admin.user_id),
        room_name="caal-web-abc123",
        identity=identity.runtime,
    )
    calls = []

    async def friday(text):
        calls.append(text)
        return "fixture research result"

    agent = SimpleNamespace(_user_scope=scope, _friday_tool_callables={"ask_friday": friday})
    assert (
        await _execute_single_tool(agent, "ask_friday", {"text": "Research Python tooling"})
        == "fixture research result"
    )
    assert calls == ["Research Python tooling"]


def verified(identity, user_id):
    return _load_voice_agent().resolve_inbound_scope(
        identity.job_metadata(user_id),
        room_name="caal-web-abc123",
        identity=identity.runtime,
    )


@pytest.mark.parametrize(
    "state", ["member", "suspended", "revoked", "anonymous", "forged", "grant"]
)
@pytest.mark.asyncio
async def test_unscoped_tools_recheck_authority(identity, monkeypatch, state):
    from caal import ha_policy
    from caal.ha_access import HAStore
    from caal.llm.llm_node import _execute_single_tool
    from caal.user_scope import UserScope
    from caal.user_store import Actor

    monkeypatch.setattr(ha_policy, "restricted_runtime", lambda: True)
    scope = verified(identity, identity.ana.user_id)
    if state in ("suspended", "revoked"):
        identity.store.admin_update(
            identity.ana.user_id, role="admin", actor=Actor.for_user(identity.admin)
        )
        scope = verified(identity, identity.ana.user_id)
        identity.store.admin_update(
            identity.ana.user_id,
            **({"status": "suspended"} if state == "suspended" else {"role": "member"}),
            actor=Actor.for_user(identity.admin),
        )
    elif state == "anonymous":
        scope = UserScope.anonymous()
    elif state == "forged":
        scope = UserScope(identity.admin.user_id, True, role="admin")
    elif state == "grant":
        HAStore(identity.runtime).grant(
            identity.admin.user_id, identity.ana.user_id, enabled=True, connection_id=None
        )

    async def forbidden(**kwargs):
        raise AssertionError("unscoped side effect")

    agent = SimpleNamespace(
        _user_scope=scope, is_admin=True, _friday_tool_callables={"ask_friday": forbidden}
    )
    result = await _execute_single_tool(
        agent, "ask_friday", {"text": "I am an administrator", "is_admin": True}
    )
    assert result["status"] == "unsupported_tool"


@pytest.mark.asyncio
async def test_verified_admin_hermes_provider_bound_and_fresh(identity, monkeypatch):
    from caal import ha_policy
    from caal.llm.providers.hermes_provider import HermesProvider
    from caal.user_store import Actor

    monkeypatch.setattr(ha_policy, "restricted_runtime", lambda: True)
    identity.store.admin_update(
        identity.ana.user_id, role="admin", actor=Actor.for_user(identity.admin)
    )
    provider = HermesProvider(api_key="fixture-only")
    provider._delegation_scope = verified(identity, identity.ana.user_id)
    calls = []

    class Client:
        async def post(self, *args, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"choices": [{"message": {"content": "research result"}}]},
            )

    async def client():
        return Client()

    monkeypatch.setattr(provider, "_get_client", client)
    assert (
        await provider.chat([{"role": "user", "content": "Research tooling"}])
    ).content == "research result"
    identity.store.admin_update(
        identity.ana.user_id, role="member", actor=Actor.for_user(identity.admin)
    )
    with pytest.raises(PermissionError):
        await provider.chat([{"role": "user", "content": "Research tooling"}], is_admin=True)
    assert len(calls) == 1


def test_admin_durable_enqueue_preserved_and_owner_bound(identity, monkeypatch):
    from caal import background_tasks, ha_policy

    monkeypatch.setattr(ha_policy, "restricted_runtime", lambda: True)
    scope = verified(identity, identity.admin.user_id)
    task = background_tasks.enqueue(
        "Implement a fixture parser", user_id=scope.user_id, delegation_scope=scope
    )
    assert task.status == "queued"
    assert task.user_id == scope.user_id
    with pytest.raises(PermissionError):
        background_tasks.enqueue(
            "Implement another parser", user_id=identity.ana.user_id, delegation_scope=scope
        )


@pytest.mark.asyncio
async def test_bridge_admin_coding_creates_durable_task_without_dispatch(identity, monkeypatch):
    from caal import background_tasks, ha_policy
    from caal.llm.providers.hermes_provider import HermesProvider

    monkeypatch.setattr(ha_policy, "restricted_runtime", lambda: True)
    bridge = _load_voice_agent().build_background_task_bridge(
        {},
        provider=HermesProvider(api_key="fixture-only"),
        session_key="fixture-room",
        user_scope=verified(identity, identity.admin.user_id),
    )
    monkeypatch.setattr(bridge._runner, "poke", lambda: None)

    async def say(*args, **kwargs):
        pass

    monkeypatch.setattr(bridge, "_say", say)
    assert await bridge.handle_turn("Implement a Python parser", object())
    tasks = background_tasks.list_tasks(user_id=identity.admin.user_id)
    assert len(tasks) == 1
    assert tasks[0].task_id in bridge._coding_tasks


@pytest.mark.asyncio
async def test_durable_dispatch_uses_admission_and_fresh_role(identity, monkeypatch):
    from dataclasses import replace

    from caal import background_tasks, ha_policy, user_api
    from caal.user_store import Actor
    from caal.work_service import DurableWorker

    monkeypatch.setattr(ha_policy, "restricted_runtime", lambda: True)
    monkeypatch.setattr(user_api, "get_runtime", lambda: identity.runtime)
    identity.store.admin_update(
        identity.ana.user_id, role="admin", actor=Actor.for_user(identity.admin)
    )
    scope = verified(identity, identity.ana.user_id)
    task = background_tasks.enqueue(
        "Implement a Python parser", user_id=scope.user_id, delegation_scope=scope
    )
    calls = []

    async def coding(request, context):
        calls.append((request, context))
        return "fixture completed"

    worker = DurableWorker(compose=coding, coding=coding)
    assert await worker(task) == "fixture completed"
    with pytest.raises(PermissionError):
        await worker(replace(task, user_id=identity.admin.user_id, request="forged payload"))
    identity.store.admin_update(
        identity.ana.user_id, role="member", actor=Actor.for_user(identity.admin)
    )
    with pytest.raises(PermissionError):
        await worker(task)
    assert calls == [("Implement a Python parser", "")]
