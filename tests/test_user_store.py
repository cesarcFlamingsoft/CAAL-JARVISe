"""User profiles, bootstrap, callback numbers, and the audit trail.

The store is the single source of truth for who a verified email is, whether
they may act, and which phone JARVIS may dial for them. These tests pin the
security properties rather than the storage details: opaque ids, normalized
emails, deny-by-default resolution with a one-time bootstrap for the named
administrator, encrypted-at-rest callback numbers that never appear in views
or the audit trail, role and status guards that cannot lock every admin out,
and versioned migrations that upgrade an older database in place.
"""

from __future__ import annotations

import sqlite3

import pytest

from caal import profile_crypto
from caal.profile_crypto import KeyRing
from caal.user_store import (
    ACTIVE,
    ADMIN,
    MEMBER,
    SUSPENDED,
    Actor,
    CallbackNumberError,
    DuplicateUserError,
    LastAdminError,
    NotConfiguredError,
    UnknownUserError,
    UserStore,
    UserSuspendedError,
    is_valid_user_id,
    normalize_e164,
)

BOOTSTRAP = "cesarc@mexcantech.com"
NOW = 1_700_000_000
SYSTEM = Actor.system()


@pytest.fixture
def ring() -> KeyRing:
    return KeyRing.from_env(profile_crypto.generate_key_material(version=1))


@pytest.fixture
def store(tmp_path, ring) -> UserStore:
    return UserStore(tmp_path / "assistant.sqlite3", keyring=ring)


def _admin(store: UserStore) -> tuple:
    admin = store.resolve_identity(BOOTSTRAP, bootstrap_admin_email=BOOTSTRAP, now=NOW)
    return admin, Actor.for_user(admin)


# --- bootstrap and resolution ------------------------------------------------


def test_first_verified_bootstrap_email_becomes_the_only_admin(store) -> None:
    admin = store.resolve_identity(
        "CesarC@MexcanTech.com ", bootstrap_admin_email=BOOTSTRAP, now=NOW
    )

    assert is_valid_user_id(admin.user_id)
    assert admin.user_id.startswith("usr_")
    assert admin.email == BOOTSTRAP
    assert admin.role == ADMIN
    assert admin.status == ACTIVE
    assert admin.display_name
    assert admin.has_callback_number is False
    assert BOOTSTRAP not in repr(admin)
    assert [u.user_id for u in store.list_users()] == [admin.user_id]

    events = store.list_audit_events()
    assert [event.action for event in events] == ["user.bootstrap"]
    assert events[0].target_id == admin.user_id
    assert BOOTSTRAP not in repr(events[0]) and BOOTSTRAP not in str(events[0].detail)


def test_bootstrap_only_ever_happens_once_and_only_for_the_named_email(store) -> None:
    with pytest.raises(UnknownUserError):
        store.resolve_identity("someone@else.com", bootstrap_admin_email=BOOTSTRAP, now=NOW)
    assert store.list_users() == []

    first = store.resolve_identity(BOOTSTRAP, bootstrap_admin_email=BOOTSTRAP, now=NOW)
    again = store.resolve_identity(BOOTSTRAP, bootstrap_admin_email=BOOTSTRAP, now=NOW + 5)
    assert again.user_id == first.user_id
    assert [e.action for e in store.list_audit_events()] == ["user.bootstrap"]

    # A second identity that merely matches the bootstrap email later is still
    # the same single user, never a second admin.
    assert len(store.list_users()) == 1


def test_bootstrap_requires_a_configured_bootstrap_email(store) -> None:
    with pytest.raises(NotConfiguredError):
        store.resolve_identity(BOOTSTRAP, bootstrap_admin_email="", now=NOW)
    with pytest.raises(NotConfiguredError):
        store.resolve_identity(BOOTSTRAP, bootstrap_admin_email="not-an-email", now=NOW)


def test_unknown_and_suspended_identities_are_refused(store) -> None:
    _, actor = _admin(store)
    member = store.create_user(
        email="Ana@Example.com", display_name="Ana", role=MEMBER, actor=actor
    )

    resolved = store.resolve_identity("ana@example.com", bootstrap_admin_email=BOOTSTRAP, now=NOW)
    assert resolved.user_id == member.user_id
    assert resolved.last_seen_at == NOW

    store.admin_update(member.user_id, status=SUSPENDED, actor=actor)
    with pytest.raises(UserSuspendedError):
        store.resolve_identity("ana@example.com", bootstrap_admin_email=BOOTSTRAP, now=NOW)
    with pytest.raises(UnknownUserError):
        store.resolve_identity("nobody@example.com", bootstrap_admin_email=BOOTSTRAP, now=NOW)


@pytest.mark.parametrize("bad", ["", "cesar", "a@b", "a b@c.com", 42, None])
def test_resolution_rejects_malformed_emails_without_touching_the_store(store, bad) -> None:
    with pytest.raises(ValueError):
        store.resolve_identity(bad, bootstrap_admin_email=BOOTSTRAP, now=NOW)
    assert store.list_users() == []


# --- admin management ---------------------------------------------------------


def test_admin_creates_lists_edits_and_suspends_users(store) -> None:
    admin, actor = _admin(store)

    member = store.create_user(
        email="  Ana@Example.com", display_name="  Ana   Lima ", role=MEMBER, actor=actor
    )
    assert member.email == "ana@example.com"
    assert member.display_name == "Ana Lima"
    assert member.role == MEMBER and member.status == ACTIVE
    assert member.created_by == admin.user_id

    with pytest.raises(DuplicateUserError):
        store.create_user(email="ANA@example.com", display_name="Dup", role=MEMBER, actor=actor)

    renamed = store.admin_update(member.user_id, display_name="Ana L.", actor=actor)
    promoted = store.admin_update(member.user_id, role=ADMIN, actor=actor)
    suspended = store.admin_update(member.user_id, status=SUSPENDED, actor=actor)
    assert renamed.display_name == "Ana L."
    assert promoted.role == ADMIN
    assert suspended.status == SUSPENDED
    assert store.get_user(member.user_id).status == SUSPENDED

    listed = store.list_users()
    assert [u.user_id for u in listed] == [admin.user_id, member.user_id]
    assert [e.action for e in store.list_audit_events()] == [
        "user.bootstrap",
        "user.create",
        "user.update",
        "user.role",
        "user.status",
    ][::-1] or [e.action for e in store.list_audit_events(oldest_first=True)] == [
        "user.bootstrap",
        "user.create",
        "user.update",
        "user.role",
        "user.status",
    ]


def test_audit_events_are_bounded_redacted_and_carry_opaque_ids_only(store) -> None:
    admin, actor = _admin(store)
    member = store.create_user(
        email="ana@example.com", display_name="Ana", role=MEMBER, actor=actor
    )
    store.set_callback_number(member.user_id, "+17805558345", actor=actor)
    store.admin_update(member.user_id, display_name="Xiomara Quintanilla-Reyes", actor=actor)

    events = store.list_audit_events(limit=3)
    assert len(events) == 3
    rendered = repr(events) + " ".join(str(e.detail) for e in events)
    assert "7805558345" not in rendered
    assert "ana@example.com" not in rendered
    assert "Xiomara" not in rendered and "Quintanilla" not in rendered
    assert all(e.actor_id == admin.user_id for e in events)
    assert all(e.target_id == member.user_id for e in events)
    assert all(len(str(e.detail)) <= 512 for e in events)
    assert store.list_audit_events(limit=100_000).__len__() <= 500


def test_the_last_active_admin_cannot_be_demoted_suspended_or_locked_out(store) -> None:
    admin, actor = _admin(store)

    with pytest.raises(LastAdminError):
        store.admin_update(admin.user_id, role=MEMBER, actor=actor)
    with pytest.raises(LastAdminError):
        store.admin_update(admin.user_id, status=SUSPENDED, actor=actor)

    second = store.create_user(email="ops@example.com", display_name="Ops", role=ADMIN, actor=actor)
    store.admin_update(admin.user_id, role=MEMBER, actor=Actor.for_user(second))
    assert store.get_user(admin.user_id).role == MEMBER
    with pytest.raises(LastAdminError):
        store.admin_update(second.user_id, status=SUSPENDED, actor=Actor.for_user(second))


def test_only_admins_may_manage_and_members_may_only_edit_their_own_display_name(store) -> None:
    admin, actor = _admin(store)
    member = store.create_user(
        email="ana@example.com", display_name="Ana", role=MEMBER, actor=actor
    )
    member_actor = Actor.for_user(member)

    with pytest.raises(PermissionError):
        store.create_user(email="x@example.com", display_name="X", role=MEMBER, actor=member_actor)
    with pytest.raises(PermissionError):
        store.admin_update(admin.user_id, role=MEMBER, actor=member_actor)
    with pytest.raises(PermissionError):
        store.admin_update(member.user_id, role=ADMIN, actor=member_actor)
    with pytest.raises(PermissionError):
        store.set_callback_number(member.user_id, "+17805558345", actor=member_actor)

    updated = store.update_display_name(member.user_id, "Ana L", actor=member_actor)
    assert updated.display_name == "Ana L"
    with pytest.raises(PermissionError):
        store.update_display_name(admin.user_id, "Hacked", actor=member_actor)
    assert store.get_user(admin.user_id).display_name != "Hacked"


@pytest.mark.parametrize("bad_name", ["", "   ", "x" * 81, "bad\x00name", "line\nbreak"])
def test_display_names_are_validated(store, bad_name) -> None:
    admin, actor = _admin(store)
    with pytest.raises(ValueError):
        store.create_user(email="a@example.com", display_name=bad_name, role=MEMBER, actor=actor)
    with pytest.raises(ValueError):
        store.update_display_name(admin.user_id, bad_name, actor=actor)


def test_unknown_user_ids_and_malformed_ids_are_refused_uniformly(store) -> None:
    _, actor = _admin(store)
    for bad in ("usr_deadbeefdeadbeefdeadbeef", "", "'; DROP TABLE users; --", "usr_x", None):
        assert store.get_user(bad) is None
        with pytest.raises(UnknownUserError):
            store.admin_update(bad, display_name="x", actor=actor)
        with pytest.raises(UnknownUserError):
            store.set_callback_number(bad, "+17805558345", actor=actor)


# --- callback numbers -----------------------------------------------------------


def test_callback_number_is_encrypted_at_rest_and_never_rendered(store, tmp_path) -> None:
    admin, actor = _admin(store)
    member = store.create_user(
        email="ana@example.com", display_name="Ana", role=MEMBER, actor=actor
    )

    view = store.set_callback_number(member.user_id, " +1 (780) 555-8345 ", actor=actor)

    assert view.has_callback_number is True
    assert view.callback_number_updated_at is not None
    assert "7805558345" not in repr(view)
    assert "7805558345" not in str(view.public_view()) + str(view.admin_view())
    assert store.approved_callback_number(member.user_id) == "+17805558345"

    dump = "\n".join(sqlite3.connect(tmp_path / "assistant.sqlite3").iterdump())
    assert "7805558345" not in dump
    assert "+1" not in dump.replace("+1 ", "")


@pytest.mark.parametrize(
    "raw",
    ["", "780-555-8345", "+1", "+0123456789", "+178055583451234567", "+1780555834a", "911", None],
)
def test_callback_number_must_be_a_real_e164_number(store, raw) -> None:
    _, actor = _admin(store)
    member = store.create_user(
        email="ana@example.com", display_name="Ana", role=MEMBER, actor=actor
    )
    with pytest.raises(CallbackNumberError):
        store.set_callback_number(member.user_id, raw, actor=actor)
    assert store.get_user(member.user_id).has_callback_number is False


def test_normalize_e164_accepts_common_spellings() -> None:
    assert normalize_e164("+1 (780) 555-8345") == "+17805558345"
    assert normalize_e164("+44 20 7946 0958") == "+442079460958"
    with pytest.raises(CallbackNumberError):
        normalize_e164("(780) 555-8345")  # no country code: never guess


def test_cleared_or_suspended_users_have_no_dialable_number(store) -> None:
    _, actor = _admin(store)
    member = store.create_user(
        email="ana@example.com", display_name="Ana", role=MEMBER, actor=actor
    )
    store.set_callback_number(member.user_id, "+17805558345", actor=actor)

    store.admin_update(member.user_id, status=SUSPENDED, actor=actor)
    assert store.approved_callback_number(member.user_id) is None
    store.admin_update(member.user_id, status=ACTIVE, actor=actor)
    assert store.approved_callback_number(member.user_id) == "+17805558345"

    cleared = store.clear_callback_number(member.user_id, actor=actor)
    assert cleared.has_callback_number is False
    assert store.approved_callback_number(member.user_id) is None
    assert store.approved_callback_number("usr_deadbeefdeadbeefdeadbeef") is None
    assert [e.action for e in store.list_audit_events(oldest_first=True)][
        -1
    ] == "user.callback.clear"


def test_callback_number_lookup_by_caller_id_uses_the_blind_index(store, tmp_path) -> None:
    _, actor = _admin(store)
    ana = store.create_user(email="ana@example.com", display_name="Ana", role=MEMBER, actor=actor)
    bo = store.create_user(email="bo@example.com", display_name="Bo", role=MEMBER, actor=actor)
    store.set_callback_number(ana.user_id, "+17805558345", actor=actor)
    store.set_callback_number(bo.user_id, "+17805550000", actor=actor)

    assert store.find_user_by_callback_number("+1 780 555 8345").user_id == ana.user_id
    assert store.find_user_by_callback_number("+17805550000").user_id == bo.user_id
    assert store.find_user_by_callback_number("+17805559999") is None
    assert store.find_user_by_callback_number("garbage") is None

    store.admin_update(ana.user_id, status=SUSPENDED, actor=actor)
    assert store.find_user_by_callback_number("+17805558345") is None


def test_the_same_number_cannot_be_approved_for_two_users(store) -> None:
    _, actor = _admin(store)
    ana = store.create_user(email="ana@example.com", display_name="Ana", role=MEMBER, actor=actor)
    bo = store.create_user(email="bo@example.com", display_name="Bo", role=MEMBER, actor=actor)
    store.set_callback_number(ana.user_id, "+17805558345", actor=actor)

    with pytest.raises(CallbackNumberError):
        store.set_callback_number(bo.user_id, "+17805558345", actor=actor)
    assert store.get_user(bo.user_id).has_callback_number is False


def test_key_rotation_re_encrypts_numbers_under_the_active_key(tmp_path) -> None:
    old = KeyRing.from_env(profile_crypto.generate_key_material(version=1))
    store = UserStore(tmp_path / "assistant.sqlite3", keyring=old)
    _, actor = _admin(store)
    ana = store.create_user(email="ana@example.com", display_name="Ana", role=MEMBER, actor=actor)
    store.set_callback_number(ana.user_id, "+17805558345", actor=actor)

    rotated = KeyRing.from_env(
        old_material := ",".join(
            [
                _ring_entry(old, 1),
                profile_crypto.generate_key_material(version=2),
            ]
        )
    )
    assert old_material
    upgraded = UserStore(tmp_path / "assistant.sqlite3", keyring=rotated)
    assert upgraded.approved_callback_number(ana.user_id) == "+17805558345"
    assert upgraded.find_user_by_callback_number("+17805558345").user_id == ana.user_id

    assert upgraded.rotate_encryption() == 1
    assert upgraded.rotate_encryption() == 0
    assert upgraded.approved_callback_number(ana.user_id) == "+17805558345"
    assert upgraded.find_user_by_callback_number("+17805558345").user_id == ana.user_id

    retired = UserStore(
        tmp_path / "assistant.sqlite3",
        keyring=KeyRing.from_env(_ring_entry(rotated, 2)),
    )
    assert retired.approved_callback_number(ana.user_id) == "+17805558345"


def _ring_entry(ring: KeyRing, version: int) -> str:
    import base64

    raw = ring._keys[version]
    return f"v{version}:{base64.urlsafe_b64encode(raw).decode().rstrip('=')}"


def test_without_a_keyring_callback_numbers_fail_closed(tmp_path) -> None:
    store = UserStore(tmp_path / "assistant.sqlite3", keyring=None)
    _, actor = _admin(store)
    member = store.create_user(
        email="ana@example.com", display_name="Ana", role=MEMBER, actor=actor
    )

    with pytest.raises(NotConfiguredError):
        store.set_callback_number(member.user_id, "+17805558345", actor=actor)
    assert store.approved_callback_number(member.user_id) is None
    assert store.find_user_by_callback_number("+17805558345") is None


def test_wrong_key_material_never_yields_a_number(tmp_path) -> None:
    store = UserStore(
        tmp_path / "assistant.sqlite3",
        keyring=KeyRing.from_env(profile_crypto.generate_key_material(version=1)),
    )
    _, actor = _admin(store)
    member = store.create_user(
        email="ana@example.com", display_name="Ana", role=MEMBER, actor=actor
    )
    store.set_callback_number(member.user_id, "+17805558345", actor=actor)

    other = UserStore(
        tmp_path / "assistant.sqlite3",
        keyring=KeyRing.from_env(profile_crypto.generate_key_material(version=1)),
    )
    assert other.approved_callback_number(member.user_id) is None
    assert other.get_user(member.user_id).has_callback_number is True


# --- migrations -----------------------------------------------------------------


def test_schema_migrations_are_versioned_and_idempotent(tmp_path, ring) -> None:
    path = tmp_path / "assistant.sqlite3"
    first = UserStore(path, keyring=ring)
    assert first.schema_version() >= 1
    version = first.schema_version()

    second = UserStore(path, keyring=ring)
    assert second.schema_version() == version
    with sqlite3.connect(path) as connection:
        applied = connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
    assert applied == version


def test_store_coexists_with_the_other_tables_in_the_shared_database(tmp_path, ring) -> None:
    from caal import conversation_ledger, device_registry
    from caal.tools import memory_tools

    path = tmp_path / "assistant.sqlite3"
    conversation_ledger.STORE_PATH = path
    conversation_ledger.open_conversation(session_key="room-1")
    device_registry.STORE_PATH = path
    memory_tools.STORE_PATH = path

    store = UserStore(path, keyring=ring)
    admin = store.resolve_identity(BOOTSTRAP, bootstrap_admin_email=BOOTSTRAP, now=NOW)
    assert store.get_user(admin.user_id) is not None


def test_views_expose_only_safe_fields(store) -> None:
    admin, actor = _admin(store)
    member = store.create_user(
        email="ana@example.com", display_name="Ana", role=MEMBER, actor=actor
    )
    store.set_callback_number(member.user_id, "+17805558345", actor=actor)
    member = store.get_user(member.user_id)

    public = member.public_view()
    assert set(public) == {
        "user_id",
        "email",
        "display_name",
        "role",
        "status",
        "has_callback_number",
        "callback_number_updated_at",
        "created_at",
        "updated_at",
        "last_seen_at",
    }
    assert "7805558345" not in str(public)
    admin_view = member.admin_view()
    assert admin_view["created_by"] == admin.user_id
    assert "7805558345" not in str(admin_view)
