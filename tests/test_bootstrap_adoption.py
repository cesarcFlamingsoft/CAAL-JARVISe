"""The first administrator adopts the deployment's pre-multi-user memories, once.

Before identity existed, JARVIS kept one shared memory store for the single
person using it. When that person's email bootstraps as the first
administrator, those legacy memories become theirs; no other user ever sees
them, and the adoption is recorded in the audit trail as a count only.
"""

from __future__ import annotations

import json

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from caal import profile_crypto, user_api, webhooks
from caal.access_jwt import AccessConfig, AccessVerifier
from caal.internal_auth import AUDIENCE_IDENTITY, mint_principal
from caal.profile_crypto import KeyRing
from caal.security_config import MultiUserConfig
from caal.tools import memory_tools
from caal.user_api import IdentityRuntime
from caal.user_store import Actor, UserStore

SECRET = "s" * 48
TEAM = "https://example-team.cloudflareaccess.com"
AUD = "d1d09a2c79e964918d59077b9bb5b3a7b67ff76a04a80822eb8a3ce5f46354ac"
BOOTSTRAP = "cesarc@mexcantech.com"
NOW = 1_700_000_000

PRIVATE = rsa.generate_private_key(public_exponent=65537, key_size=2048)
JWK = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(PRIVATE.public_key()))
JWK.update({"kid": "kid-a", "alg": "RS256", "use": "sig"})


def test_store_reports_bootstrap_through_a_hook_exactly_once(tmp_path) -> None:
    store = UserStore(
        tmp_path / "assistant.sqlite3",
        keyring=KeyRing.from_env(profile_crypto.generate_key_material(version=1)),
    )
    seen: list[str] = []

    first = store.resolve_identity(
        BOOTSTRAP,
        bootstrap_admin_email=BOOTSTRAP,
        now=NOW,
        on_bootstrap=lambda p: seen.append(p.user_id),
    )
    store.resolve_identity(
        BOOTSTRAP,
        bootstrap_admin_email=BOOTSTRAP,
        now=NOW + 1,
        on_bootstrap=lambda p: seen.append("again"),
    )

    assert seen == [first.user_id]


def test_hook_failure_never_undoes_the_bootstrap(tmp_path) -> None:
    store = UserStore(tmp_path / "assistant.sqlite3", keyring=None)

    def broken(profile) -> None:
        raise RuntimeError("adoption exploded")

    admin = store.resolve_identity(
        BOOTSTRAP, bootstrap_admin_email=BOOTSTRAP, now=NOW, on_bootstrap=broken
    )

    assert store.get_user(admin.user_id) is not None
    assert [e.action for e in store.list_audit_events(oldest_first=True)] == ["user.bootstrap"]


def test_resolve_endpoint_adopts_legacy_memories_for_the_bootstrap_admin(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(memory_tools, "STORE_PATH", tmp_path / "assistant.sqlite3")
    memory_tools.remember("coffee", "flat white")  # saved before multi-user existed
    memory_tools.remember("timezone", "America/Edmonton")

    keyring = KeyRing.from_env(profile_crypto.generate_key_material(version=1))
    store = UserStore(tmp_path / "assistant.sqlite3", keyring=keyring)
    config = MultiUserConfig(
        internal_auth_secret=SECRET,
        keyring=keyring,
        bootstrap_admin_email=BOOTSTRAP,
        access=AccessConfig(team_domain=TEAM, audience=AUD),
        store_path=tmp_path / "assistant.sqlite3",
    )
    runtime = IdentityRuntime(
        config,
        store=store,
        access_verifier=AccessVerifier(
            config.access, fetch_jwks=lambda url: {"keys": [JWK]}, clock=lambda: NOW
        ),
        clock=lambda: NOW,
    )
    webhooks.app.dependency_overrides[user_api.get_runtime] = lambda: runtime
    try:
        with TestClient(webhooks.app) as client:
            cf_token = jwt.encode(
                {
                    "iss": TEAM,
                    "aud": [AUD],
                    "sub": "cf",
                    "email": BOOTSTRAP,
                    "type": "app",
                    "iat": NOW,
                    "exp": NOW + 600,
                },
                PRIVATE,
                algorithm="RS256",
                headers={"kid": "kid-a"},
            )
            assertion = mint_principal(
                secret=SECRET,
                subject="identity",
                audience=AUDIENCE_IDENTITY,
                claims={"email": BOOTSTRAP},
                now=NOW,
            )
            response = client.post(
                "/auth/resolve",
                headers={
                    "Authorization": f"Bearer {assertion}",
                    "Cf-Access-Jwt-Assertion": cf_token,
                },
            )
    finally:
        webhooks.app.dependency_overrides.pop(user_api.get_runtime, None)

    assert response.status_code == 200
    admin_id = response.json()["user_id"]
    assert memory_tools.recall("coffee", user_id=admin_id)["data"]["value"] == "flat white"
    assert memory_tools.recall()["data"]["memories"] == []  # legacy scope is now empty
    actions = [e.action for e in store.list_audit_events(oldest_first=True)]
    assert actions == ["user.bootstrap", "memory.adopt_legacy"]
    adoption = store.list_audit_events(oldest_first=True)[-1]
    assert adoption.detail == {"count": 2}
    assert "coffee" not in repr(adoption) and "flat white" not in str(adoption.detail)

    # Nobody else ever inherits legacy memories.
    member = store.create_user(
        email="ana@example.com",
        display_name="Ana",
        role="member",
        actor=Actor.for_user(store.get_user(admin_id)),
    )
    assert memory_tools.recall("coffee", user_id=member.user_id)["status"] == "not_found"


def test_record_audit_is_available_for_system_events(tmp_path) -> None:
    store = UserStore(tmp_path / "assistant.sqlite3", keyring=None)
    admin = store.resolve_identity(BOOTSTRAP, bootstrap_admin_email=BOOTSTRAP, now=NOW)

    store.record_audit(
        "memory.adopt_legacy", actor=Actor.system(), target_id=admin.user_id, detail={"count": 3}
    )

    latest = store.list_audit_events()[0]
    assert latest.action == "memory.adopt_legacy"
    assert latest.actor_id == "system" and latest.target_id == admin.user_id
    assert latest.detail == {"count": 3}
    with pytest.raises(ValueError):
        store.record_audit("", actor=Actor.system())
    with pytest.raises(ValueError):
        store.record_audit("x" * 200, actor=Actor.system())
