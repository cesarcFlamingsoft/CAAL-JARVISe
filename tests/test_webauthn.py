"""Passkey persistence and ceremony security tests."""

from __future__ import annotations

import base64
import json
from contextlib import closing
from types import SimpleNamespace

import pytest

from caal.local_auth import InvalidCredentialsError, LocalAuth
from caal.password_hash import Argon2Params
from caal.user_store import ADMIN, Actor, UserStore
from caal.webauthn_auth import CeremonyError, WebAuthnConfig, WebAuthnService

CHEAP = Argon2Params(memory_kib=64, iterations=1, lanes=1)
PASSWORD = "correct horse battery staple"
SESSION = "a" * 64
OTHER_SESSION = "b" * 64


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _user(store: UserStore, auth: LocalAuth):
    profile = store.create_user(
        email="passkey@example.com",
        display_name="Pass Key",
        role=ADMIN,
        actor=Actor.system(),
        now=1000,
    )
    auth.set_password(profile.user_id, PASSWORD, actor=Actor.system(), now=1000)
    return profile


def test_passkey_migration_is_minimal_and_idempotent(tmp_path) -> None:
    path = tmp_path / "assistant.sqlite3"
    first = UserStore(path, keyring=None)

    with closing(first.connect()) as connection:
        credential_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(webauthn_credentials)")
        }
        ceremony_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(webauthn_ceremonies)")
        }

    assert credential_columns == {
        "credential_id",
        "user_id",
        "public_key",
        "sign_count",
        "transports",
        "label",
        "created_at",
        "last_used_at",
    }
    assert ceremony_columns == {
        "ceremony_id",
        "purpose",
        "user_id",
        "challenge_hash",
        "session_binding_hash",
        "created_at",
        "expires_at",
        "consumed_at",
    }
    version = first.schema_version()
    assert UserStore(path, keyring=None).schema_version() == version


@pytest.mark.parametrize(
    ("raw", "origin", "rp_id"),
    [
        ("https://friday.example.com", "https://friday.example.com", "friday.example.com"),
        ("https://friday.example.com:443", "https://friday.example.com", "friday.example.com"),
        (
            "https://friday.example.com:8443",
            "https://friday.example.com:8443",
            "friday.example.com",
        ),
    ],
)
def test_webauthn_config_derives_exact_origin_and_rp_id(
    raw: str, origin: str, rp_id: str
) -> None:
    config = WebAuthnConfig.from_public_origin(raw)

    assert config.origin == origin
    assert config.rp_id == rp_id


@pytest.mark.parametrize(
    "raw",
    [
        "http://friday.example.com",
        "https://user@friday.example.com",
        "https://friday.example.com/path",
        "https://friday.example.com?query=yes",
        "not a URL",
    ],
)
def test_webauthn_config_rejects_unsafe_public_origins(raw: str) -> None:
    with pytest.raises(ValueError):
        WebAuthnConfig.from_public_origin(raw)


def test_registration_options_require_a_discoverable_platform_authenticator(tmp_path) -> None:
    store = UserStore(tmp_path / "assistant.sqlite3", keyring=None)
    auth = LocalAuth(store, params=CHEAP, clock=lambda: 1000)
    user = _user(store, auth)
    service = WebAuthnService(
        store,
        auth,
        WebAuthnConfig.from_public_origin("https://friday.example.com"),
        clock=lambda: 1000,
    )

    result = service.begin_registration(user.user_id, "My Mac", PASSWORD, SESSION)

    assert set(result) == {"ceremonyId", "publicKey"}
    options = result["publicKey"]
    assert options["rp"] == {"id": "friday.example.com", "name": "FRIDAY"}
    assert options["user"]["id"] == _b64(user.user_id.encode())
    assert options["authenticatorSelection"] == {
        "authenticatorAttachment": "platform",
        "residentKey": "required",
        "requireResidentKey": True,
        "userVerification": "required",
    }
    assert options["attestation"] == "none"
    assert options["timeout"] <= 300_000


def test_ceremony_is_purpose_bound_short_lived_and_atomically_consumed(tmp_path) -> None:
    store = UserStore(tmp_path / "assistant.sqlite3", keyring=None)
    auth = LocalAuth(store, params=CHEAP, clock=lambda: 1000)
    user = _user(store, auth)
    now = 1000
    service = WebAuthnService(
        store,
        auth,
        WebAuthnConfig.from_public_origin("https://friday.example.com"),
        clock=lambda: now,
    )
    begun = service.begin_registration(user.user_id, "Phone", PASSWORD, SESSION)
    challenge = begun["publicKey"]["challenge"]

    assert service.consume_ceremony(
        begun["ceremonyId"], purpose="register", challenge=challenge,
        user_id=user.user_id, session_binding=SESSION
    )
    assert not service.consume_ceremony(
        begun["ceremonyId"], purpose="register", challenge=challenge,
        user_id=user.user_id, session_binding=SESSION
    )

    other = service.begin_registration(user.user_id, "Phone", PASSWORD, SESSION)
    assert not service.consume_ceremony(
        other["ceremonyId"], purpose="authenticate", challenge=other["publicKey"]["challenge"]
    )
    expired = service.begin_registration(user.user_id, "Phone", PASSWORD, SESSION)
    now += 301
    assert not service.consume_ceremony(
        expired["ceremonyId"],
        purpose="register",
        challenge=expired["publicKey"]["challenge"],
        user_id=user.user_id,
        session_binding=SESSION,
    )


def test_registration_rejects_cross_user_and_non_platform_response_before_verification(
    tmp_path,
) -> None:
    store = UserStore(tmp_path / "assistant.sqlite3", keyring=None)
    auth = LocalAuth(store, params=CHEAP, clock=lambda: 1000)
    user = _user(store, auth)
    other = store.create_user(
        email="other@example.com",
        display_name="Other",
        role=ADMIN,
        actor=Actor.system(),
        now=1000,
    )
    auth.set_password(other.user_id, PASSWORD, actor=Actor.system(), now=1000)
    calls = 0

    def verifier(**kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("must not verify an invalid ceremony")

    service = WebAuthnService(
        store,
        auth,
        WebAuthnConfig.from_public_origin("https://friday.example.com"),
        clock=lambda: 1000,
        registration_verifier=verifier,
    )
    begun = service.begin_registration(user.user_id, "Laptop", PASSWORD, SESSION)
    client_data = _b64(
        json.dumps(
            {"type": "webauthn.create", "challenge": begun["publicKey"]["challenge"]}
        ).encode()
    )
    credential = {
        "id": _b64(b"credential"),
        "rawId": _b64(b"credential"),
        "type": "public-key",
        "authenticatorAttachment": "cross-platform",
        "response": {"clientDataJSON": client_data, "attestationObject": _b64(b"x")},
    }

    with pytest.raises(CeremonyError):
        service.finish_registration(
            other.user_id, begun["ceremonyId"], "Laptop", credential, SESSION
        )
    assert calls == 0


def _credential(challenge: str, *, credential_id: bytes = b"credential", create=True):
    client_data = _b64(
        json.dumps(
            {
                "type": "webauthn.create" if create else "webauthn.get",
                "challenge": challenge,
                "origin": "https://friday.example.com",
            }
        ).encode()
    )
    return {
        "id": _b64(credential_id),
        "rawId": _b64(credential_id),
        "type": "public-key",
        "authenticatorAttachment": "platform",
        "response": {
            "clientDataJSON": client_data,
            **(
                {"attestationObject": _b64(b"attestation"), "transports": ["internal"]}
                if create
                else {
                    "authenticatorData": _b64(b"authenticator"),
                    "signature": _b64(b"signature"),
                }
            ),
        },
    }


def test_stolen_authenticated_session_cannot_enroll_or_revoke_without_current_password(
    tmp_path,
) -> None:
    store = UserStore(tmp_path / "assistant.sqlite3", keyring=None)
    auth = LocalAuth(store, params=CHEAP, clock=lambda: 1000)
    user = _user(store, auth)
    service = WebAuthnService(
        store,
        auth,
        WebAuthnConfig.from_public_origin("https://friday.example.com"),
        clock=lambda: 1000,
        registration_verifier=lambda **_: SimpleNamespace(
            credential_id=b"credential", credential_public_key=b"public", sign_count=0
        ),
    )

    with pytest.raises(InvalidCredentialsError):
        service.begin_registration(user.user_id, "Phone", "wrong password", SESSION)

    begun = service.begin_registration(user.user_id, "Phone", PASSWORD, SESSION)
    credential = _credential(begun["publicKey"]["challenge"])
    with pytest.raises(CeremonyError, match="invalid_ceremony"):
        service.finish_registration(
            user.user_id, begun["ceremonyId"], "Phone", credential, OTHER_SESSION
        )

    begun = service.begin_registration(user.user_id, "Phone", PASSWORD, SESSION)
    service.finish_registration(
        user.user_id,
        begun["ceremonyId"],
        "Phone",
        _credential(begun["publicKey"]["challenge"]),
        SESSION,
    )
    key_id = service.list_credentials(user.user_id)[0]["id"]
    with pytest.raises(InvalidCredentialsError):
        service.revoke_credential(user.user_id, key_id, "wrong password")
    assert service.list_credentials(user.user_id)[0]["id"] == key_id
    assert service.revoke_credential(user.user_id, key_id, PASSWORD) is True


def test_registration_stores_public_material_and_lists_only_safe_metadata(tmp_path) -> None:
    store = UserStore(tmp_path / "assistant.sqlite3", keyring=None)
    auth = LocalAuth(store, params=CHEAP, clock=lambda: 1000)
    user = _user(store, auth)
    seen = {}

    def verifier(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            credential_id=b"credential", credential_public_key=b"public-cose-key", sign_count=7
        )

    service = WebAuthnService(
        store,
        auth,
        WebAuthnConfig.from_public_origin("https://friday.example.com"),
        clock=lambda: 1000,
        registration_verifier=verifier,
    )
    begun = service.begin_registration(user.user_id, "My Mac", PASSWORD, SESSION)

    created = service.finish_registration(
        user.user_id,
        begun["ceremonyId"],
        "My Mac",
        _credential(begun["publicKey"]["challenge"]),
        SESSION,
    )

    assert seen["expected_origin"] == "https://friday.example.com"
    assert seen["expected_rp_id"] == "friday.example.com"
    assert seen["require_user_verification"] is True
    assert created == {"label": "My Mac", "createdAt": 1000, "lastUsedAt": None}
    listed = service.list_credentials(user.user_id)
    assert len(listed) == 1
    assert listed[0]["id"].startswith("key_")
    assert {key: listed[0][key] for key in created} == created
    serialized = json.dumps(listed)
    assert "credential" not in serialized
    assert "public-cose-key" not in serialized
    assert "sign_count" not in serialized


def test_user_can_rename_and_revoke_only_their_own_passkey(tmp_path) -> None:
    store = UserStore(tmp_path / "assistant.sqlite3", keyring=None)
    auth = LocalAuth(store, params=CHEAP, clock=lambda: 1000)
    user = _user(store, auth)
    other = store.create_user(
        email="other@example.com",
        display_name="Other",
        role=ADMIN,
        actor=Actor.system(),
        now=1000,
    )
    auth.set_password(other.user_id, PASSWORD, actor=Actor.system(), now=1000)
    service = WebAuthnService(
        store,
        auth,
        WebAuthnConfig.from_public_origin("https://friday.example.com"),
        clock=lambda: 1000,
        registration_verifier=lambda **_: SimpleNamespace(
            credential_id=b"credential", credential_public_key=b"public", sign_count=0
        ),
    )
    begun = service.begin_registration(user.user_id, "Phone", PASSWORD, SESSION)
    service.finish_registration(
        user.user_id,
        begun["ceremonyId"],
        "Phone",
        _credential(begun["publicKey"]["challenge"]),
        SESSION,
    )
    key_id = service.list_credentials(user.user_id)[0]["id"]

    assert service.rename_credential(other.user_id, key_id, "Stolen") is False
    assert service.revoke_credential(other.user_id, key_id, PASSWORD) is False
    assert service.rename_credential(user.user_id, key_id, "Work Mac") is True
    assert service.list_credentials(user.user_id)[0]["label"] == "Work Mac"
    assert service.revoke_credential(user.user_id, key_id, PASSWORD) is True
    assert service.list_credentials(user.user_id) == []


def test_usernameless_authentication_verifies_handle_signature_and_counter_then_issues_session(
    tmp_path,
) -> None:
    store = UserStore(tmp_path / "assistant.sqlite3", keyring=None)
    auth = LocalAuth(store, params=CHEAP, clock=lambda: 1000)
    user = _user(store, auth)
    observed = {}
    service = WebAuthnService(
        store,
        auth,
        WebAuthnConfig.from_public_origin("https://friday.example.com"),
        clock=lambda: 1000,
        registration_verifier=lambda **_: SimpleNamespace(
            credential_id=b"credential", credential_public_key=b"public", sign_count=2
        ),
        authentication_verifier=lambda **kwargs: (
            observed.update(kwargs) or SimpleNamespace(new_sign_count=3)
        ),
    )
    registered = service.begin_registration(user.user_id, "Phone", PASSWORD, SESSION)
    service.finish_registration(
        user.user_id,
        registered["ceremonyId"],
        "Phone",
        _credential(registered["publicKey"]["challenge"]),
        SESSION,
    )
    begun = service.begin_authentication()
    assert "allowCredentials" not in begun["publicKey"]
    assertion = _credential(begun["publicKey"]["challenge"], create=False)
    assertion["response"]["userHandle"] = _b64(user.user_id.encode())

    result = service.finish_authentication(begun["ceremonyId"], assertion)

    assert observed["credential_public_key"] == b"public"
    assert observed["credential_current_sign_count"] == 2
    assert observed["require_user_verification"] is True
    assert result.user.user_id == user.user_id
    assert result.token and auth.verify_session(result.token) is not None
    assert service.list_credentials(user.user_id)[0]["lastUsedAt"] == 1000
    with closing(store.connect()) as connection:
        count = connection.execute(
            "SELECT sign_count FROM webauthn_credentials"
        ).fetchone()["sign_count"]
    assert count == 3


def test_authentication_generically_rejects_replay_wrong_handle_suspension_and_forced_change(
    tmp_path,
) -> None:
    store = UserStore(tmp_path / "assistant.sqlite3", keyring=None)
    auth = LocalAuth(store, params=CHEAP, clock=lambda: 1000)
    user = _user(store, auth)
    store.create_user(
        email="remaining-admin@example.com",
        display_name="Remaining Admin",
        role=ADMIN,
        actor=Actor.system(),
        now=1000,
    )
    service = WebAuthnService(
        store,
        auth,
        WebAuthnConfig.from_public_origin("https://friday.example.com"),
        clock=lambda: 1000,
        registration_verifier=lambda **_: SimpleNamespace(
            credential_id=b"credential", credential_public_key=b"public", sign_count=0
        ),
        authentication_verifier=lambda **_: SimpleNamespace(new_sign_count=1),
    )
    registered = service.begin_registration(user.user_id, "Phone", PASSWORD, SESSION)
    service.finish_registration(
        user.user_id,
        registered["ceremonyId"],
        "Phone",
        _credential(registered["publicKey"]["challenge"]),
        SESSION,
    )

    begun = service.begin_authentication()
    wrong = _credential(begun["publicKey"]["challenge"], create=False)
    wrong["response"]["userHandle"] = _b64(b"usr_" + b"f" * 24)
    with pytest.raises(CeremonyError, match="invalid_credentials"):
        service.finish_authentication(begun["ceremonyId"], wrong)
    with pytest.raises(CeremonyError, match="invalid_credentials"):
        service.finish_authentication(begun["ceremonyId"], wrong)

    store.admin_update(user.user_id, status="suspended", actor=Actor.system(), now=1001)
    suspended = service.begin_authentication()
    assertion = _credential(suspended["publicKey"]["challenge"], create=False)
    assertion["response"]["userHandle"] = _b64(user.user_id.encode())
    with pytest.raises(CeremonyError, match="invalid_credentials"):
        service.finish_authentication(suspended["ceremonyId"], assertion)

    store.admin_update(user.user_id, status="active", actor=Actor.system(), now=1002)
    auth.set_password(user.user_id, PASSWORD + " changed", actor=Actor.system(), must_change=True)
    forced = service.begin_authentication()
    assertion = _credential(forced["publicKey"]["challenge"], create=False)
    assertion["response"]["userHandle"] = _b64(user.user_id.encode())
    with pytest.raises(CeremonyError, match="invalid_credentials"):
        service.finish_authentication(forced["ceremonyId"], assertion)
