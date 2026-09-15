from types import SimpleNamespace

import pytest

from caal.profile_crypto import KeyRing, generate_key_material
from caal.user_scope import UserScope
from caal.user_store import Actor, UserStore


@pytest.fixture
def h(tmp_path):
    key = KeyRing.from_env(generate_key_material())
    users = UserStore(tmp_path / "users.db", keyring=key)
    admin = users.create_user(
        email="admin@example.com", display_name="Admin", role="admin", actor=Actor.system()
    )
    member = users.create_user(
        email="member@example.com",
        display_name="Member",
        role="member",
        actor=Actor.for_user(admin),
    )
    return SimpleNamespace(
        store=users, config=SimpleNamespace(keyring=key), admin=admin, member=member
    )


def test_grant_revoke_is_live_and_does_not_invent_a_connection(h):
    from caal.ha_access import HAStore

    store = HAStore(h)
    scope = UserScope.for_user(h.member)
    assert store.access(scope)["status"] == "denied"
    store.grant(h.admin.user_id, h.member.user_id, enabled=True, connection_id=None)
    assert store.access(scope)["status"] == "connection_required"
    store.grant(h.admin.user_id, h.member.user_id, enabled=False, connection_id=None)
    assert store.access(scope)["status"] == "denied"
    assert store.access(UserScope.anonymous())["status"] == "denied"


def test_authenticated_connection_assignment_authority_and_encryption(h):
    from caal.ha_access import HAStore

    store = HAStore(h)
    a = store.save_connection(
        h.admin.user_id,
        {"id": "ha-admin", "name": "Admin HA", "is_admin": True},
        {"access_token": "private-A", "refresh_token": "refresh-A"},
        endpoint="http://127.0.0.1:8123",
        client_id="https://jarvis.example.com",
    )
    b = store.save_connection(
        h.member.user_id,
        {"id": "ha-member", "name": "Member HA", "is_admin": False},
        {"access_token": "private-B", "refresh_token": "refresh-B"},
        endpoint="http://127.0.0.1:8123",
        client_id="https://jarvis.example.com",
    )
    with pytest.raises(PermissionError):
        store.grant(h.member.user_id, h.member.user_id, enabled=True, connection_id=a)
    store.grant(h.admin.user_id, h.member.user_id, enabled=True, connection_id=b)
    scope = UserScope.for_user(h.member)
    assert store.access(scope)["status"] == "connected"
    assert store.credentials(scope)["access_token"] == "private-B"
    with h.store.connect() as db:
        assert "private-B" not in str(
            [tuple(r) for r in db.execute("SELECT * FROM ha_connections")]
        )
    store.disconnect(h.member.user_id, b)
    assert store.access(scope)["status"] == "connection_required"
    with pytest.raises(PermissionError):
        store.credentials(scope)


def test_arbitrary_identity_and_suspended_owner_cannot_supply_credentials(h):
    from caal.ha_access import HAStore

    store = HAStore(h)
    with pytest.raises(PermissionError):
        store.grant(
            h.admin.user_id, h.member.user_id, enabled=True, connection_id="arbitrary-ha-user-id"
        )
    assert store.access(UserScope(h.member.user_id, True, role="admin"))["status"] == "denied"


def test_oauth_state_is_one_use_and_bound_to_user_and_endpoint(h):
    from caal.ha_access import HAStore

    store = HAStore(h)
    state = store.start_state(
        h.member.user_id, "http://127.0.0.1:8123", "https://jarvis.example.com", now=100
    )
    with pytest.raises(PermissionError):
        store.consume_state(h.admin.user_id, state, now=101)
    value = store.consume_state(h.member.user_id, state, now=101)
    assert value["endpoint"] == "http://127.0.0.1:8123"
    with pytest.raises(PermissionError):
        store.consume_state(h.member.user_id, state, now=101)


def test_existing_admin_service_access_is_explicit_and_can_be_revoked(h):
    from caal.ha_access import HAStore

    store = HAStore(h)
    scope = UserScope.for_user(h.admin)
    assert store.access(scope)["service_account"] is True
    assert store.access(scope)["status"] == "service_account"
    store.grant(h.admin.user_id, h.admin.user_id, enabled=False, connection_id=None)
    assert store.access(scope)["status"] == "denied"


def test_refresh_cannot_resurrect_a_disconnected_connection(h):
    from caal.ha_access import HAStore

    store = HAStore(h)
    cid = store.save_connection(
        h.member.user_id,
        {"id": "ha-member"},
        {"access_token": "old", "refresh_token": "refresh"},
        endpoint="http://127.0.0.1:8123",
        client_id="https://jarvis.example.com",
    )
    store.grant(h.admin.user_id, h.member.user_id, enabled=True, connection_id=cid)
    credentials = store.credentials(UserScope.for_user(h.member))
    store.disconnect(h.member.user_id, cid)
    with pytest.raises(PermissionError):
        store.refresh_connection(credentials, {"access_token": "new", "expires_in": 1800})
    assert store.access(UserScope.for_user(h.member))["status"] == "connection_required"


def test_cross_user_assignment_and_ciphertext_swapping_fail_closed(h):
    from caal.ha_access import HAStore

    store = HAStore(h)
    third = h.store.create_user(
        email="third@example.com",
        display_name="Third",
        role="member",
        actor=Actor.for_user(h.admin),
    )
    cid = store.save_connection(
        third.user_id,
        {"id": "third-ha"},
        {"access_token": "third-token"},
        endpoint="http://127.0.0.1:8123",
        client_id="https://jarvis.example.com",
    )
    with pytest.raises(PermissionError):
        store.grant(h.admin.user_id, h.member.user_id, enabled=True, connection_id=cid)
    assert store.choices(h.member.user_id, h.member.user_id) == []
    with pytest.raises(PermissionError):
        store.choices(h.member.user_id, third.user_id)


def test_suspension_denies_an_existing_active_scope(h):
    from caal.ha_access import HAStore

    store = HAStore(h)
    scope = UserScope.for_user(h.member)
    store.grant(h.admin.user_id, h.member.user_id, enabled=True, connection_id=None)
    h.store.admin_update(h.member.user_id, status="suspended", actor=Actor.for_user(h.admin))
    assert store.access(scope)["status"] == "denied"
