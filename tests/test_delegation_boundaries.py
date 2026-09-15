"""No ambient system authority or contextual-owner substitution."""

from types import SimpleNamespace

import pytest
from test_delegation_correction import identity, verified  # noqa: F401


@pytest.mark.parametrize("kind", ["missing", "legacy", "anonymous"])
def test_no_identity_runtime_does_not_authorize_unbound_delegation(monkeypatch, kind):
    from caal import ha_policy
    from caal.user_scope import UserScope

    monkeypatch.setattr(ha_policy, "restricted_runtime", lambda: False)
    scope = {"missing": None, "legacy": UserScope.legacy(), "anonymous": UserScope.anonymous()}[
        kind
    ]
    with pytest.raises(PermissionError):
        ha_policy.require_delegated_scope(scope)
    assert not ha_policy.tool_allowed(SimpleNamespace(_user_scope=scope), "ask_friday")


def test_contextual_administrator_cannot_enqueue_for_another_owner(identity, monkeypatch):  # noqa: F811
    from caal import background_tasks, ha_policy, user_api

    monkeypatch.setattr(ha_policy, "restricted_runtime", lambda: True)
    monkeypatch.setattr(user_api, "get_runtime", lambda: identity.runtime)
    scope = verified(identity, identity.admin.user_id)
    task = background_tasks.enqueue(
        "Implement a fixture parser", user_id=scope.user_id, delegation_scope=scope
    )
    with ha_policy.task_dispatch_scope(task):
        with pytest.raises(PermissionError):
            background_tasks.enqueue("Implement a second parser", user_id=identity.ana.user_id)
        with pytest.raises(PermissionError):
            background_tasks.enqueue("Implement an unowned parser")
    assert len(background_tasks.list_tasks()) == 1


def test_default_worker_context_without_runtime_requires_explicit_authority(monkeypatch, tmp_path):
    from caal import background_tasks, ha_policy

    monkeypatch.setattr(background_tasks, "STORE_PATH", tmp_path / "tasks.db")

    monkeypatch.setattr(ha_policy, "restricted_runtime", lambda: False)
    with pytest.raises(PermissionError):
        with ha_policy.task_dispatch_scope(SimpleNamespace(task_id="unadmitted", user_id=None)):
            pytest.fail("unbound system dispatch was authorized")


@pytest.mark.asyncio
async def test_child_dispatch_authority_expires_when_parent_dispatch_exits(identity, monkeypatch):  # noqa: F811
    import asyncio

    from caal import background_tasks, ha_policy, user_api

    monkeypatch.setattr(user_api, "get_runtime", lambda: identity.runtime)
    scope = verified(identity, identity.admin.user_id)
    task = background_tasks.enqueue(
        "Implement a fixture parser", user_id=scope.user_id, delegation_scope=scope
    )
    release = asyncio.Event()

    async def child():
        await release.wait()
        ha_policy.require_delegated_scope()

    with ha_policy.task_dispatch_scope(task):
        pending = asyncio.create_task(child())
    release.set()
    with pytest.raises(PermissionError):
        await pending


def test_dispatch_context_resets_after_exception(identity, monkeypatch):  # noqa: F811
    from caal import background_tasks, ha_policy, user_api

    monkeypatch.setattr(user_api, "get_runtime", lambda: identity.runtime)
    scope = verified(identity, identity.admin.user_id)
    task = background_tasks.enqueue(
        "Implement a fixture parser", user_id=scope.user_id, delegation_scope=scope
    )
    with pytest.raises(RuntimeError, match="fixture interruption"):
        with ha_policy.task_dispatch_scope(task):
            ha_policy.require_delegated_scope()
            raise RuntimeError("fixture interruption")
    with pytest.raises(PermissionError):
        ha_policy.require_delegated_scope()
    ha_policy.require_delegated_scope(scope)  # Independent session authority survives.


@pytest.mark.asyncio
async def test_stream_rechecks_demotion_after_client_initialization(identity, monkeypatch):  # noqa: F811
    from caal.llm.providers.hermes_provider import HermesProvider
    from caal.user_store import Actor

    identity.store.admin_update(
        identity.ana.user_id, role="admin", actor=Actor.for_user(identity.admin)
    )
    provider = HermesProvider(api_key="fixture-only")
    provider._delegation_scope = verified(identity, identity.ana.user_id)

    async def client():
        identity.store.admin_update(
            identity.ana.user_id, role="member", actor=Actor.for_user(identity.admin)
        )
        return SimpleNamespace(
            stream=lambda *args, **kwargs: pytest.fail("stream transport was reached")
        )

    monkeypatch.setattr(provider, "_get_client", client)
    with pytest.raises(PermissionError):
        async for _ in provider.chat_stream([{"role": "user", "content": "Research Python."}]):
            pytest.fail("stream emitted output")


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["suspended", "anonymous", "admissionless", "forged"])
async def test_durable_dispatch_rejects_invalid_owner_or_admission(identity, monkeypatch, state):  # noqa: F811
    from dataclasses import replace

    from caal import background_tasks, user_api
    from caal.user_store import Actor
    from caal.work_service import DurableWorker

    monkeypatch.setattr(user_api, "get_runtime", lambda: identity.runtime)
    identity.store.admin_update(
        identity.ana.user_id, role="admin", actor=Actor.for_user(identity.admin)
    )
    scope = verified(identity, identity.ana.user_id)
    task = background_tasks.enqueue(
        "Implement a fixture parser", user_id=scope.user_id, delegation_scope=scope
    )
    if state == "suspended":
        identity.store.admin_update(
            identity.ana.user_id, status="suspended", actor=Actor.for_user(identity.admin)
        )
    elif state == "anonymous":
        task = replace(task, user_id=None)
    elif state == "admissionless":
        with background_tasks._connect() as connection:
            connection.execute(
                "DELETE FROM delegation_admissions WHERE task_id = ?", (task.task_id,)
            )
    else:
        task = replace(task, request="Research unrelated instructions")

    async def forbidden(*args, **kwargs):
        pytest.fail("worker execution was reached")

    with pytest.raises(PermissionError):
        await DurableWorker(compose=forbidden, coding=forbidden)(task)
