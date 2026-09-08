"""Cross-user isolation of conversations, background work, and devices.

Every durable thing a session touches now carries the opaque id of the user
it belongs to. A phone leg cannot claim another user's conversation, a bridge
cannot see, cancel, or arm callbacks for another user's tasks, and a device
only ever learns about the other devices of its own user. Legacy sessions
(no identity configured) keep working under the unscoped ``NULL`` owner.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from caal import background_tasks, conversation_ledger, device_registry, user_api, webhooks
from caal.background_task_session import CALLBACK_NOTHING_RUNNING_REPLY, BackgroundTaskBridge
from caal.background_tasks import QUEUED, RUNNING, SUCCEEDED
from caal.internal_auth import AUDIENCE_BACKEND, mint_principal

ANA = "usr_" + "a" * 24
BO = "usr_" + "b" * 24
SECRET = "s" * 48
ENROLLMENT_TOKEN = "enrollment-secret"


@pytest.fixture(autouse=True)
def store(monkeypatch, tmp_path):
    path = tmp_path / "assistant.sqlite3"
    monkeypatch.setattr(conversation_ledger, "STORE_PATH", path)
    monkeypatch.setattr(background_tasks, "STORE_PATH", path)
    monkeypatch.setattr(device_registry, "STORE_PATH", path)
    return path


# --- conversation ledger -------------------------------------------------------------


def test_conversation_is_owned_by_its_user_and_only_that_user_may_continue_it() -> None:
    conversation_id = conversation_ledger.open_conversation(
        session_key="web-ana", user_id=ANA, now=1000
    )
    conversation_ledger.append_turn(conversation_id, "user", "plan Lisbon", now=1001)

    assert conversation_ledger.conversation_user_id(conversation_id) == ANA
    with pytest.raises(PermissionError):
        conversation_ledger.link_continuation(
            conversation_id, session_key="attempt-bo", user_id=BO, now=1002
        )
    with pytest.raises(PermissionError):
        conversation_ledger.link_continuation(conversation_id, session_key="attempt-anon", now=1002)

    conversation_ledger.link_continuation(
        conversation_id, session_key="attempt-ana", user_id=ANA, now=1003
    )
    assert (
        conversation_ledger.claim_continuation(
            conversation_id, session_key="attempt-ana", user_id=BO, now=1004
        )
        is None
    )
    context = conversation_ledger.claim_continuation(
        conversation_id, session_key="attempt-ana", user_id=ANA, now=1005
    )
    assert context is not None and [t.text for t in context.turns] == ["plan Lisbon"]


def test_legacy_conversations_stay_unscoped_and_refuse_a_user_bound_continuation() -> None:
    conversation_id = conversation_ledger.open_conversation(session_key="web-legacy", now=1000)

    assert conversation_ledger.conversation_user_id(conversation_id) is None
    conversation_ledger.link_continuation(conversation_id, session_key="attempt-1", now=1001)
    assert (
        conversation_ledger.claim_continuation(conversation_id, session_key="attempt-1", now=1002)
        is not None
    )
    with pytest.raises(PermissionError):
        conversation_ledger.link_continuation(
            conversation_id, session_key="attempt-2", user_id=ANA, now=1003
        )
    assert conversation_ledger.conversation_user_id("not-an-id") is None


@pytest.mark.parametrize("bad", ["", "usr_x", "ana", 42])
def test_ledger_rejects_malformed_user_ids(bad) -> None:
    with pytest.raises(ValueError):
        conversation_ledger.open_conversation(session_key="web", user_id=bad, now=1000)


# --- background tasks ------------------------------------------------------------------


def test_tasks_are_filed_under_their_user_and_hidden_from_everyone_else() -> None:
    ana_task = background_tasks.enqueue("find fares", session_key="conv-1", user_id=ANA)
    bo_task = background_tasks.enqueue("find hotels", session_key="conv-1", user_id=BO)
    legacy_task = background_tasks.enqueue("legacy work", session_key="conv-1")

    assert ana_task.user_id == ANA and legacy_task.user_id is None
    assert [t.task_id for t in background_tasks.list_tasks(session_key="conv-1", user_id=ANA)] == [
        ana_task.task_id
    ]
    assert [t.task_id for t in background_tasks.list_tasks(session_key="conv-1", user_id=None)] == [
        legacy_task.task_id
    ]
    # Unscoped listing (operator tooling) still sees everything.
    assert len(background_tasks.list_tasks(session_key="conv-1")) == 3
    assert ANA not in repr(ana_task) or "user" in repr(ana_task)

    for task in (ana_task, bo_task, legacy_task):
        background_tasks._mark_running(task.task_id)
        background_tasks._finish(task.task_id, SUCCEEDED, result="done")
    mine = background_tasks.pending_notifications(session_key="conv-1", user_id=ANA)
    assert [t.task_id for t in mine] == [ana_task.task_id]
    assert background_tasks.claim_next_notification("conv-1", "ana", user_id=ANA).task_id == (
        ana_task.task_id
    )
    assert background_tasks.claim_next_notification("conv-1", "ana", user_id=ANA) is None
    assert (
        background_tasks.claim_next_notification("conv-1", "legacy", user_id=None).task_id
        == legacy_task.task_id
    )


def test_callbacks_can_only_be_armed_for_the_owning_user() -> None:
    task = background_tasks.enqueue("find fares", session_key="conv-1", user_id=ANA)

    assert (
        background_tasks.arm_callback(task.task_id, None, session_key="conv-1", user_id=BO) is False
    )
    assert (
        background_tasks.arm_callback(task.task_id, "+17805558345", session_key="conv-1") is False
    )
    assert (
        background_tasks.arm_callback(task.task_id, None, session_key="conv-1", user_id=ANA) is True
    )
    assert background_tasks.callback_armed(task.task_id) is True

    background_tasks._mark_running(task.task_id)
    background_tasks._finish(task.task_id, SUCCEEDED, result="done")
    target = background_tasks.claim_callback_target(task.task_id, "runner")
    assert target is not None
    assert target.user_id == ANA and target.destination is None
    assert background_tasks.claim_callback_target(task.task_id, "runner") is None


def test_legacy_destination_callbacks_still_work_and_never_carry_a_user() -> None:
    task = background_tasks.enqueue("find fares", session_key="room-1")
    assert background_tasks.arm_callback(task.task_id, "+17805558345", session_key="room-1") is True
    background_tasks._mark_running(task.task_id)
    background_tasks._finish(task.task_id, SUCCEEDED, result="done")

    assert background_tasks.claim_callback(task.task_id, "runner") == "+17805558345"
    assert background_tasks.claim_callback(task.task_id, "runner") is None
    with pytest.raises(ValueError):
        background_tasks.arm_callback(task.task_id, None, session_key="room-1")  # nothing to dial


class _Execute:
    async def __call__(self, request: str, context: str) -> str:
        return "done"


class _Session:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def say(self, text: str) -> None:
        self.spoken.append(text)


@pytest.mark.asyncio
async def test_bridges_of_different_users_never_see_each_others_work() -> None:
    ana = BackgroundTaskBridge(execute=_Execute(), session_key="room-1", user_id=ANA)
    bo = BackgroundTaskBridge(execute=_Execute(), session_key="room-1", user_id=BO)
    await ana.start()
    await bo.start()
    try:
        ana_session = _Session()
        assert await ana.handle_turn("look into Lisbon fares in the background", ana_session)
        (task,) = background_tasks.list_tasks(statuses=[QUEUED, RUNNING, SUCCEEDED])
        assert task.user_id == ANA

        bo_session = _Session()
        assert bo.can_arm_callback is False
        assert await bo.arm_callback(None, bo_session, user_id=BO) is False
        assert bo_session.spoken == [CALLBACK_NOTHING_RUNNING_REPLY]
        await bo.handle_turn("cancel the background task", bo_session)
        assert bo_session.spoken[-1] != "Understood. I've stopped that background task."
        for _ in range(20):
            import asyncio

            await asyncio.sleep(0.01)
            if background_tasks.get_task(task.task_id).status == SUCCEEDED:
                break
        assert background_tasks.get_task(task.task_id).status == SUCCEEDED
        assert await bo.deliver_pending(bo_session) == 0
        assert await ana.deliver_pending(ana_session) == 1
    finally:
        await ana.close()
        await bo.close()
        await ana.abandon()
        await bo.abandon()


@pytest.mark.asyncio
async def test_shared_fallback_channel_never_receives_user_scoped_outcomes() -> None:
    sent: list[str] = []

    async def fallback(text: str) -> None:
        sent.append(text)

    legacy = background_tasks.enqueue("legacy", session_key="room-old")
    background_tasks._mark_running(legacy.task_id)
    background_tasks._finish(legacy.task_id, SUCCEEDED, result="old answer")
    scoped = background_tasks.enqueue("private", session_key="room-old", user_id=ANA)
    background_tasks._mark_running(scoped.task_id)
    background_tasks._finish(scoped.task_id, SUCCEEDED, result="ana's private answer")

    bridge = BackgroundTaskBridge(execute=_Execute(), session_key="room-1", fallback=fallback)
    await bridge.start()
    try:
        assert await bridge.flush_stale_to_fallback(min_age_seconds=0) == 1
        assert "old answer" in sent[0]
        assert not any("private answer" in text for text in sent)
        # Ana's own later session still gets her outcome.
        ana = BackgroundTaskBridge(execute=_Execute(), session_key="room-old", user_id=ANA)
        await ana.start()
        session = _Session()
        assert await ana.deliver_pending(session) == 1
        assert "private answer" in session.spoken[0]
        await ana.close()
        await ana.abandon()
    finally:
        await bridge.close()
        await bridge.abandon()


@pytest.mark.asyncio
async def test_settling_bridge_never_leaks_a_user_callback_outcome_to_shared_fallback() -> None:
    """A user's private task outcome must never reach the shared fallback channel.

    Background tasks are drained from one shared queue, so a session that never
    scheduled a task can still be the one that settles it. When that task carries
    a user-bound callback and the callback cannot be dialed (the user's number
    was cleared, the trunk is down, ...), the settling bridge must not spill the
    result onto its own shared fallback channel — that channel belongs to
    everyone, and the outcome belongs to one user.
    """
    sent: list[str] = []

    async def fallback(text: str) -> None:
        sent.append(text)

    async def failing_user_dialer(user_id: str, task_id: str) -> None:
        raise PermissionError("user has no approved callback number")

    # Ana schedules private work and arms a callback to her own (now missing) number.
    task = background_tasks.enqueue(
        "summarize my medical results", session_key="room-ana", user_id=ANA
    )
    assert (
        background_tasks.arm_callback(task.task_id, None, session_key="room-ana", user_id=ANA)
        is True
    )
    background_tasks._mark_running(task.task_id)
    background_tasks._finish(task.task_id, SUCCEEDED, result="Ana's private medical summary")

    # A different, anonymous session (no verified user) settles it off the shared
    # queue and does have a shared fallback channel.
    anon = BackgroundTaskBridge(
        execute=_Execute(),
        session_key="room-anon",
        user_id=None,
        fallback=fallback,
        dial_user_callback=failing_user_dialer,
    )
    await anon.start()
    try:
        await anon._on_settled(task.task_id)
    finally:
        await anon.close()
        await anon.abandon()

    assert not any("private medical summary" in text for text in sent), sent


# --- device registry -----------------------------------------------------------------


def test_devices_are_scoped_to_their_user() -> None:
    ana_phone = device_registry.register_device_session(
        device_id="ana-phone",
        room_name="room-a",
        label="Ana's phone",
        transport="mobile",
        user_id=ANA,
        now=1000,
    )
    device_registry.register_device_session(
        device_id="bo-laptop",
        room_name="room-b",
        label="Bo's laptop",
        transport="web",
        user_id=BO,
        now=1000,
    )
    device_registry.register_device_session(
        device_id="lan-kiosk", room_name="room-c", label="Kitchen", transport="web", now=1000
    )

    assert ana_phone.user_id == ANA
    assert [d.device_id for d in device_registry.list_active_sessions(now=1001, user_id=ANA)] == [
        "ana-phone"
    ]
    assert [d.device_id for d in device_registry.list_active_sessions(now=1001, user_id=None)] == [
        "lan-kiosk"
    ]
    assert len(device_registry.list_active_sessions(now=1001)) == 3

    handoff = device_registry.create_handoff(ana_phone.session_id, context={"s": "x"}, now=1002)
    assert handoff.user_id == ANA
    assert (
        device_registry.claim_handoff(
            handoff.handoff_id, device_id="bo-laptop", user_id=BO, now=1003
        )
        is None
    )
    assert (
        device_registry.claim_handoff(handoff.handoff_id, device_id="lan-kiosk", now=1003) is None
    )
    claimed = device_registry.claim_handoff(
        handoff.handoff_id, device_id="ana-tablet", user_id=ANA, now=1003
    )
    assert claimed is not None and claimed.context == {"s": "x"}


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv("CAAL_DEVICE_ENROLLMENT_TOKEN", ENROLLMENT_TOKEN)
    runtime = user_api.PrincipalOnlyRuntime(secret=SECRET, now=lambda: 1_700_000_000)
    webhooks.app.dependency_overrides[user_api.get_runtime] = lambda: runtime
    try:
        with TestClient(webhooks.app) as client:
            yield client
    finally:
        webhooks.app.dependency_overrides.pop(user_api.get_runtime, None)


def _principal(user_id: str) -> str:
    return mint_principal(
        secret=SECRET, subject=user_id, audience=AUDIENCE_BACKEND, now=1_700_000_000
    )


def _register(client, device_id, label, *, user_id=None, principal=None):
    headers = {"X-Caal-Device-Token": ENROLLMENT_TOKEN}
    if user_id is not None:
        headers["X-CAAL-Principal"] = principal or _principal(user_id)
    return client.post(
        "/devices/register",
        headers=headers,
        json={"device_id": device_id, "room_name": "room", "label": label, "transport": "web"},
    )


def test_device_api_binds_the_principal_and_lists_only_the_users_devices(api_client) -> None:
    ana = _register(api_client, "ana-1", "Ana laptop", user_id=ANA)
    bo = _register(api_client, "bo-1", "Bo laptop", user_id=BO)
    anon = _register(api_client, "anon-1", "Kitchen kiosk")
    assert ana.status_code == bo.status_code == anon.status_code == 200
    assert ANA not in ana.text and "principal" not in ana.text

    ana_view = api_client.get(
        "/devices/active", headers={"Authorization": f"Bearer {ana.json()['session_token']}"}
    )
    assert {d["label"] for d in ana_view.json()["devices"]} == {"Ana laptop"}
    anon_view = api_client.get(
        "/devices/active", headers={"Authorization": f"Bearer {anon.json()['session_token']}"}
    )
    assert {d["label"] for d in anon_view.json()["devices"]} == {"Kitchen kiosk"}

    staged = api_client.post(
        "/devices/handoff",
        headers={"Authorization": f"Bearer {ana.json()['session_token']}"},
        json={"context": {"topic": "deploy"}},
    ).json()["handoff_id"]
    stolen = api_client.post(
        "/devices/handoff/claim",
        headers={"Authorization": f"Bearer {bo.json()['session_token']}"},
        json={"handoff_id": staged},
    )
    assert stolen.status_code == 404
    ana_tablet = _register(api_client, "ana-2", "Ana tablet", user_id=ANA)
    claimed = api_client.post(
        "/devices/handoff/claim",
        headers={"Authorization": f"Bearer {ana_tablet.json()['session_token']}"},
        json={"handoff_id": staged},
    )
    assert claimed.status_code == 200 and claimed.json()["context"] == {"topic": "deploy"}


def test_device_api_rejects_a_bad_principal_instead_of_downgrading(api_client) -> None:
    forged = mint_principal(secret="x" * 48, subject=ANA, audience=AUDIENCE_BACKEND)
    assert (
        _register(api_client, "ana-1", "Ana laptop", user_id=ANA, principal=forged).status_code
        == 401
    )
    assert (
        _register(api_client, "ana-1", "Ana laptop", user_id=ANA, principal="garbage").status_code
        == 401
    )
    assert len(device_registry.list_active_sessions(now=1_700_000_000)) == 0
