"""Backend identity, profile, and admin API behind the internal trust boundary.

The Next.js BFF is the only caller. It proves who it is with a short-lived
signed principal on every request and, for identity resolution, also hands
over the original Cloudflare Access assertion so the backend verifies the
identity independently. These tests exercise the real FastAPI app with a
throwaway store, key ring, RSA signing key and JWKS fetcher injected through
FastAPI's dependency override seam; nothing reads the process environment.
"""

from __future__ import annotations

import json

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from caal import profile_crypto, user_api, webhooks
from caal.access_jwt import AccessConfig, AccessVerifier
from caal.internal_auth import (
    AUDIENCE_AGENT,
    AUDIENCE_BACKEND,
    AUDIENCE_IDENTITY,
    RateLimiter,
    mint_principal,
)
from caal.profile_crypto import KeyRing
from caal.security_config import MultiUserConfig
from caal.user_api import IdentityRuntime
from caal.user_store import ADMIN, MEMBER, SUSPENDED, UserStore

SECRET = "s" * 48
TEAM = "https://example-team.cloudflareaccess.com"
AUD = "d1d09a2c79e964918d59077b9bb5b3a7b67ff76a04a80822eb8a3ce5f46354ac"
BOOTSTRAP = "cesarc@mexcantech.com"
NUMBER = "+17805558345"


def _keypair(kid: str):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
    jwk.update({"kid": kid, "alg": "RS256", "use": "sig"})
    return private, jwk


PRIVATE, JWK = _keypair("kid-a")


class Harness:
    """Everything a test needs to speak to the API as the BFF would."""

    def __init__(self, tmp_path, *, resolve_limit=100, mutation_limit=100) -> None:
        self.now = 1_700_000_000
        self.keyring = KeyRing.from_env(profile_crypto.generate_key_material(version=1))
        self.store = UserStore(tmp_path / "assistant.sqlite3", keyring=self.keyring)
        self.config = MultiUserConfig(
            internal_auth_secret=SECRET,
            keyring=self.keyring,
            bootstrap_admin_email=BOOTSTRAP,
            access=AccessConfig(team_domain=TEAM, audience=AUD),
            store_path=tmp_path / "assistant.sqlite3",
        )
        self.runtime = IdentityRuntime(
            self.config,
            store=self.store,
            access_verifier=AccessVerifier(
                self.config.access, fetch_jwks=lambda url: {"keys": [JWK]}, clock=lambda: self.now
            ),
            resolve_limiter=RateLimiter(limit=resolve_limit, window_seconds=60),
            mutation_limiter=RateLimiter(limit=mutation_limit, window_seconds=60),
            clock=lambda: self.now,
        )

    def cf_token(self, email: str, **claims) -> str:
        payload = {
            "iss": TEAM,
            "aud": [AUD],
            "sub": "cf-subject",
            "email": email,
            "type": "app",
            "iat": self.now,
            "exp": self.now + 600,
        }
        payload.update(claims)
        return jwt.encode(payload, PRIVATE, algorithm="RS256", headers={"kid": "kid-a"})

    def assertion(self, email: str, **overrides) -> str:
        params = {
            "secret": SECRET,
            "subject": "identity",
            "audience": AUDIENCE_IDENTITY,
            "claims": {"email": email},
            "now": self.now,
        }
        params.update(overrides)
        return mint_principal(**params)

    def principal(self, user_id: str, **overrides) -> str:
        params = {
            "secret": SECRET,
            "subject": user_id,
            "audience": AUDIENCE_BACKEND,
            "now": self.now,
        }
        params.update(overrides)
        return mint_principal(**params)

    def resolve(self, client: TestClient, email: str, **kwargs):
        return client.post(
            "/auth/resolve",
            headers={
                "Authorization": f"Bearer {self.assertion(email)}",
                "Cf-Access-Jwt-Assertion": self.cf_token(email),
                **kwargs.pop("headers", {}),
            },
            **kwargs,
        )

    def bearer(self, user_id: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.principal(user_id)}"}


@pytest.fixture
def harness(tmp_path):
    return Harness(tmp_path)


@pytest.fixture
def client(harness):
    webhooks.app.dependency_overrides[user_api.get_runtime] = lambda: harness.runtime
    try:
        with TestClient(webhooks.app) as test_client:
            yield test_client
    finally:
        webhooks.app.dependency_overrides.pop(user_api.get_runtime, None)


def _bootstrap_admin(harness, client) -> str:
    response = harness.resolve(client, BOOTSTRAP)
    assert response.status_code == 200, response.text
    return response.json()["user_id"]


def _create_member(harness, client, admin_id: str, email="ana@example.com") -> str:
    response = client.post(
        "/admin/users",
        headers=harness.bearer(admin_id),
        json={"email": email, "display_name": "Ana", "role": MEMBER},
    )
    assert response.status_code == 201, response.text
    return response.json()["user_id"]


# --- configuration gate -----------------------------------------------------------


def test_status_reports_configuration_and_everything_fails_closed_without_it(harness) -> None:
    webhooks.app.dependency_overrides[user_api.get_runtime] = lambda: None
    try:
        with TestClient(webhooks.app) as client:
            assert client.get("/identity/status").json() == {
                "configured": False,
                "password_login": False,
                "cloudflare_access": False,
            }
            assert harness.resolve(client, BOOTSTRAP).status_code == 503
            assert (
                client.get("/users/me", headers=harness.bearer("usr_" + "0" * 24)).status_code
                == 503
            )
            assert (
                client.get("/admin/users", headers=harness.bearer("usr_" + "0" * 24)).status_code
                == 503
            )
            assert client.get("/health").status_code == 200  # legacy endpoints untouched
    finally:
        webhooks.app.dependency_overrides.pop(user_api.get_runtime, None)


def test_status_is_true_when_configured(client) -> None:
    response = client.get("/identity/status")
    # This harness configures both providers, so both are advertised.
    assert response.json() == {
        "configured": True,
        "password_login": True,
        "cloudflare_access": True,
    }
    assert response.headers["cache-control"] == "no-store"


# --- identity resolution ------------------------------------------------------------


def test_resolve_bootstraps_the_named_admin_and_returns_opaque_fields_only(harness, client) -> None:
    response = harness.resolve(client, "CesarC@MexcanTech.com")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"user_id", "role", "status", "display_name"}
    assert body["role"] == ADMIN and body["status"] == "active"
    assert body["user_id"].startswith("usr_")
    assert "mexcantech" not in response.text
    assert response.headers["cache-control"] == "no-store"


def test_resolve_requires_both_a_bff_assertion_and_a_verified_access_token(harness, client) -> None:
    good_cf = harness.cf_token(BOOTSTRAP)
    good_assertion = harness.assertion(BOOTSTRAP)

    assert client.post("/auth/resolve").status_code == 401
    assert (
        client.post("/auth/resolve", headers={"Cf-Access-Jwt-Assertion": good_cf}).status_code
        == 401
    )
    assert (
        client.post(
            "/auth/resolve", headers={"Authorization": f"Bearer {good_assertion}"}
        ).status_code
        == 401
    )
    # The email header alone is worthless.
    assert (
        client.post(
            "/auth/resolve",
            headers={
                "Authorization": f"Bearer {good_assertion}",
                "Cf-Access-Authenticated-User-Email": BOOTSTRAP,
            },
        ).status_code
        == 401
    )
    # The two identities must agree.
    assert (
        client.post(
            "/auth/resolve",
            headers={
                "Authorization": f"Bearer {harness.assertion('other@example.com')}",
                "Cf-Access-Jwt-Assertion": good_cf,
            },
        ).status_code
        == 401
    )
    # The wrong kind of principal is not an identity assertion.
    assert (
        client.post(
            "/auth/resolve",
            headers={
                "Authorization": f"Bearer {harness.principal('usr_' + '0' * 24)}",
                "Cf-Access-Jwt-Assertion": good_cf,
            },
        ).status_code
        == 401
    )
    # A tampered Access token fails regardless of the assertion.
    assert (
        client.post(
            "/auth/resolve",
            headers={
                "Authorization": f"Bearer {good_assertion}",
                "Cf-Access-Jwt-Assertion": good_cf[:-3] + "abc",
            },
        ).status_code
        == 401
    )
    assert client.get("/admin/users", headers={}).status_code == 401
    assert len(harness.store.list_users()) == 0


def test_resolve_assertions_are_single_use(harness, client) -> None:
    headers = {
        "Authorization": f"Bearer {harness.assertion(BOOTSTRAP)}",
        "Cf-Access-Jwt-Assertion": harness.cf_token(BOOTSTRAP),
    }
    assert client.post("/auth/resolve", headers=headers).status_code == 200
    assert client.post("/auth/resolve", headers=headers).status_code == 401


def test_resolve_denies_unknown_and_suspended_identities(harness, client) -> None:
    admin_id = _bootstrap_admin(harness, client)

    denied = harness.resolve(client, "stranger@example.com")
    assert denied.status_code == 403
    assert denied.json()["detail"] == "no_account"

    member_id = _create_member(harness, client, admin_id)
    assert harness.resolve(client, "ana@example.com").json()["user_id"] == member_id
    client.patch(
        f"/admin/users/{member_id}", headers=harness.bearer(admin_id), json={"status": SUSPENDED}
    )
    suspended = harness.resolve(client, "ana@example.com")
    assert suspended.status_code == 403
    assert suspended.json()["detail"] == "suspended"


def test_resolve_is_rate_limited_per_client(tmp_path) -> None:
    harness = Harness(tmp_path, resolve_limit=2)
    webhooks.app.dependency_overrides[user_api.get_runtime] = lambda: harness.runtime
    try:
        with TestClient(webhooks.app) as client:
            assert harness.resolve(client, BOOTSTRAP).status_code == 200
            assert harness.resolve(client, BOOTSTRAP).status_code == 200
            limited = harness.resolve(client, BOOTSTRAP)
            assert limited.status_code == 429
            assert "retry-after" in limited.headers
    finally:
        webhooks.app.dependency_overrides.pop(user_api.get_runtime, None)


# --- principals ------------------------------------------------------------------------


def test_profile_requires_a_valid_backend_principal(harness, client) -> None:
    admin_id = _bootstrap_admin(harness, client)

    assert client.get("/users/me").status_code == 401
    assert client.get("/users/me", headers={"Authorization": "Token abc"}).status_code == 401
    assert client.get("/users/me", headers={"Authorization": "Bearer "}).status_code == 401
    assert client.get("/users/me", headers={"Authorization": "Bearer nope"}).status_code == 401
    for audience in (AUDIENCE_IDENTITY, AUDIENCE_AGENT):
        wrong = harness.principal(admin_id, audience=audience)
        assert (
            client.get("/users/me", headers={"Authorization": f"Bearer {wrong}"}).status_code == 401
        )
    expired = harness.principal(admin_id, now=harness.now - 3600)
    assert (
        client.get("/users/me", headers={"Authorization": f"Bearer {expired}"}).status_code == 401
    )
    forged = mint_principal(secret="x" * 48, subject=admin_id, audience=AUDIENCE_BACKEND)
    assert client.get("/users/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401
    unknown = harness.principal("usr_" + "f" * 24)
    assert (
        client.get("/users/me", headers={"Authorization": f"Bearer {unknown}"}).status_code == 401
    )

    ok = client.get("/users/me", headers=harness.bearer(admin_id))
    assert ok.status_code == 200


def test_backend_principals_are_single_use(harness, client) -> None:
    admin_id = _bootstrap_admin(harness, client)
    headers = harness.bearer(admin_id)

    assert client.get("/users/me", headers=headers).status_code == 200
    assert client.get("/users/me", headers=headers).status_code == 401


def test_suspended_users_are_refused_even_with_a_valid_principal(harness, client) -> None:
    admin_id = _bootstrap_admin(harness, client)
    member_id = _create_member(harness, client, admin_id)
    client.patch(
        f"/admin/users/{member_id}", headers=harness.bearer(admin_id), json={"status": SUSPENDED}
    )

    assert client.get("/users/me", headers=harness.bearer(member_id)).status_code == 403


# --- self-service profile ----------------------------------------------------------------


def test_users_see_and_edit_only_their_own_safe_fields(harness, client) -> None:
    admin_id = _bootstrap_admin(harness, client)
    member_id = _create_member(harness, client, admin_id)

    me = client.get("/users/me", headers=harness.bearer(member_id))
    assert me.status_code == 200
    body = me.json()
    assert body["user_id"] == member_id
    assert body["role"] == MEMBER
    assert body["has_callback_number"] is False
    assert "callback_number" not in body
    assert "created_by" not in body

    renamed = client.patch(
        "/users/me", headers=harness.bearer(member_id), json={"display_name": "  Ana  Lima "}
    )
    assert renamed.status_code == 200
    assert renamed.json()["display_name"] == "Ana Lima"

    for forbidden in (
        {"role": ADMIN},
        {"status": "active"},
        {"email": "x@y.com"},
        {"user_id": admin_id},
    ):
        response = client.patch("/users/me", headers=harness.bearer(member_id), json=forbidden)
        assert response.status_code == 422, forbidden
    assert client.get("/users/me", headers=harness.bearer(member_id)).json()["role"] == MEMBER
    assert (
        client.patch(
            "/users/me", headers=harness.bearer(member_id), json={"display_name": ""}
        ).status_code
        == 422
    )


# --- admin -----------------------------------------------------------------------------


def test_members_cannot_reach_any_admin_route(harness, client) -> None:
    admin_id = _bootstrap_admin(harness, client)
    member_id = _create_member(harness, client, admin_id)
    other_id = _create_member(harness, client, admin_id, email="bo@example.com")

    attempts = [
        ("GET", "/admin/users", None),
        ("POST", "/admin/users", {"email": "x@example.com", "display_name": "X", "role": MEMBER}),
        ("GET", f"/admin/users/{other_id}", None),
        ("PATCH", f"/admin/users/{other_id}", {"status": SUSPENDED}),
        ("PATCH", f"/admin/users/{member_id}", {"role": ADMIN}),
        ("PUT", f"/admin/users/{member_id}/callback-number", {"number": NUMBER}),
        ("DELETE", f"/admin/users/{other_id}/callback-number", None),
        ("GET", "/admin/audit", None),
    ]
    for method, path, body in attempts:
        response = client.request(method, path, headers=harness.bearer(member_id), json=body)
        assert response.status_code == 403, (method, path, response.text)
    assert client.get("/users/me", headers=harness.bearer(member_id)).json()["role"] == MEMBER
    assert len(harness.store.list_users()) == 3


def test_admin_manages_users_roles_status_and_callback_numbers(harness, client) -> None:
    admin_id = _bootstrap_admin(harness, client)
    member_id = _create_member(harness, client, admin_id)

    listed = client.get("/admin/users", headers=harness.bearer(admin_id))
    assert listed.status_code == 200
    assert [u["user_id"] for u in listed.json()["users"]] == [admin_id, member_id]
    assert listed.json()["users"][1]["email"] == "ana@example.com"

    one = client.get(f"/admin/users/{member_id}", headers=harness.bearer(admin_id))
    assert one.status_code == 200 and one.json()["created_by"] == admin_id

    promoted = client.patch(
        f"/admin/users/{member_id}",
        headers=harness.bearer(admin_id),
        json={"role": ADMIN, "display_name": "Ana Admin"},
    )
    assert promoted.status_code == 200
    assert promoted.json()["role"] == ADMIN and promoted.json()["display_name"] == "Ana Admin"

    approved = client.put(
        f"/admin/users/{member_id}/callback-number",
        headers=harness.bearer(admin_id),
        json={"number": " +1 (780) 555-8345 "},
    )
    assert approved.status_code == 200
    assert approved.json()["has_callback_number"] is True
    assert "7805558345" not in approved.text
    assert harness.store.approved_callback_number(member_id) == NUMBER

    cleared = client.delete(
        f"/admin/users/{member_id}/callback-number", headers=harness.bearer(admin_id)
    )
    assert cleared.status_code == 200
    assert cleared.json()["has_callback_number"] is False

    suspended = client.patch(
        f"/admin/users/{member_id}", headers=harness.bearer(admin_id), json={"status": SUSPENDED}
    )
    assert suspended.status_code == 200 and suspended.json()["status"] == SUSPENDED

    audit = client.get("/admin/audit?limit=50", headers=harness.bearer(admin_id))
    assert audit.status_code == 200
    events = audit.json()["events"]
    actions = [event["action"] for event in events]
    assert actions[0] == "user.status"
    assert "user.bootstrap" in actions and "user.callback.set" in actions
    assert "7805558345" not in audit.text
    assert "ana@example.com" not in audit.text
    assert "mexcantech" not in audit.text
    assert all(
        set(event) >= {"event_id", "occurred_at", "actor_id", "action", "target_id"}
        for event in events
    )


def test_admin_validation_and_conflicts(harness, client) -> None:
    admin_id = _bootstrap_admin(harness, client)
    member_id = _create_member(harness, client, admin_id)
    headers = harness.bearer

    invalid = [
        {"email": "not-an-email", "display_name": "X", "role": MEMBER},
        {"email": "x@example.com", "display_name": "", "role": MEMBER},
        {"email": "x@example.com", "display_name": "X", "role": "root"},
        {"email": "x@example.com", "display_name": "X", "role": MEMBER, "status": SUSPENDED},
        {"email": "x@example.com", "display_name": "x" * 500, "role": MEMBER},
    ]
    for body in invalid:
        assert (
            client.post("/admin/users", headers=headers(admin_id), json=body).status_code == 422
        ), body
    duplicate = client.post(
        "/admin/users",
        headers=headers(admin_id),
        json={"email": "ANA@example.com", "display_name": "Dup", "role": MEMBER},
    )
    assert duplicate.status_code == 409

    assert client.get("/admin/users/usr_" + "0" * 24, headers=headers(admin_id)).status_code == 404
    assert client.get("/admin/users/not-an-id", headers=headers(admin_id)).status_code == 404
    assert (
        client.patch(
            "/admin/users/usr_" + "0" * 24, headers=headers(admin_id), json={"role": ADMIN}
        ).status_code
        == 404
    )
    assert (
        client.patch(
            f"/admin/users/{member_id}", headers=headers(admin_id), json={"role": "root"}
        ).status_code
        == 422
    )
    assert (
        client.patch(
            f"/admin/users/{member_id}", headers=headers(admin_id), json={"email": "z@z.com"}
        ).status_code
        == 422
    )
    last_admin = client.patch(
        f"/admin/users/{admin_id}", headers=headers(admin_id), json={"role": MEMBER}
    )
    assert last_admin.status_code == 409

    for bad_number in ("780-555-8345", "", "+1", "call me"):
        response = client.put(
            f"/admin/users/{member_id}/callback-number",
            headers=headers(admin_id),
            json={"number": bad_number},
        )
        assert response.status_code == 422, bad_number
    other_id = _create_member(harness, client, admin_id, email="bo@example.com")
    client.put(
        f"/admin/users/{member_id}/callback-number",
        headers=headers(admin_id),
        json={"number": NUMBER},
    )
    clash = client.put(
        f"/admin/users/{other_id}/callback-number",
        headers=headers(admin_id),
        json={"number": NUMBER},
    )
    assert clash.status_code == 409
    assert "7805558345" not in clash.text


def test_admin_mutations_are_rate_limited(tmp_path) -> None:
    harness = Harness(tmp_path, mutation_limit=2)
    webhooks.app.dependency_overrides[user_api.get_runtime] = lambda: harness.runtime
    try:
        with TestClient(webhooks.app) as client:
            admin_id = _bootstrap_admin(harness, client)
            body = {"display_name": "A", "role": MEMBER}
            first = client.post(
                "/admin/users", headers=harness.bearer(admin_id), json={"email": "a@x.com", **body}
            )
            second = client.post(
                "/admin/users", headers=harness.bearer(admin_id), json={"email": "b@x.com", **body}
            )
            third = client.post(
                "/admin/users", headers=harness.bearer(admin_id), json={"email": "c@x.com", **body}
            )
            assert (first.status_code, second.status_code, third.status_code) == (201, 201, 429)
            # Reads are not throttled by the mutation budget.
            assert client.get("/admin/users", headers=harness.bearer(admin_id)).status_code == 200
    finally:
        webhooks.app.dependency_overrides.pop(user_api.get_runtime, None)


def test_every_identity_response_is_uncacheable(harness, client) -> None:
    admin_id = _bootstrap_admin(harness, client)
    responses = [
        client.get("/identity/status"),
        client.get("/users/me"),
        client.get("/users/me", headers=harness.bearer(admin_id)),
        client.get("/admin/users", headers=harness.bearer(admin_id)),
        client.get("/admin/users/nope", headers=harness.bearer(admin_id)),
        harness.resolve(client, "stranger@example.com"),
    ]
    for response in responses:
        assert response.headers.get("cache-control") == "no-store", response.url
        assert response.headers.get("x-content-type-options") == "nosniff", response.url


def test_callback_numbers_never_appear_in_any_response_or_log(harness, client, caplog) -> None:
    admin_id = _bootstrap_admin(harness, client)
    member_id = _create_member(harness, client, admin_id)
    with caplog.at_level("DEBUG"):
        set_response = client.put(
            f"/admin/users/{member_id}/callback-number",
            headers=harness.bearer(admin_id),
            json={"number": NUMBER},
        )
        listed = client.get("/admin/users", headers=harness.bearer(admin_id))
        me = client.get("/users/me", headers=harness.bearer(member_id))
        audit = client.get("/admin/audit", headers=harness.bearer(admin_id))
    for response in (set_response, listed, me, audit):
        assert "7805558345" not in response.text
    assert "7805558345" not in caplog.text
    assert "ana@example.com" not in caplog.text


def test_identity_routes_never_carry_permissive_cors_headers(harness, client) -> None:
    """The app-wide CORS policy (any origin, with credentials) must not reach identity routes."""
    evil = {"Origin": "https://evil.example"}
    admin_id = _bootstrap_admin(harness, client)

    for response in (
        client.get("/identity/status", headers=evil),
        client.get("/users/me", headers={**evil, **harness.bearer(admin_id)}),
        client.options(
            "/admin/users",
            headers={**evil, "Access-Control-Request-Method": "POST"},
        ),
    ):
        assert "access-control-allow-origin" not in {k.lower() for k in response.headers}
        assert "access-control-allow-credentials" not in {k.lower() for k in response.headers}

    # Legacy endpoints keep their existing behaviour for the LAN frontend.
    legacy = client.get("/setup/status", headers=evil)
    assert legacy.headers.get("access-control-allow-origin") == "*"
