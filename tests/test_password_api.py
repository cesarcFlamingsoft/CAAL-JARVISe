"""The backend's standalone sign-in API, as the Next.js BFF actually calls it.

Every route here sits behind the internal trust boundary: the BFF proves
itself with a short-lived single-use signed principal, and the *user's*
identity comes from a password check plus an opaque server-side session, with
no Cloudflare Access anywhere in the picture.

These tests drive the real FastAPI app with a throwaway store, so the
assertions are about wire behaviour: status codes, response shapes, what is
absent from a body, and what a second attempt is allowed to do.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from caal import profile_crypto, user_api, webhooks
from caal.internal_auth import AUDIENCE_BACKEND, AUDIENCE_IDENTITY, RateLimiter, mint_principal
from caal.local_auth import LocalAuth, SessionPolicy
from caal.password_hash import Argon2Params
from caal.profile_crypto import KeyRing
from caal.security_config import MultiUserConfig
from caal.user_api import IdentityRuntime
from caal.user_store import ADMIN, MEMBER, SUSPENDED, Actor, UserStore

SECRET = "s" * 48
ADMIN_EMAIL = "cesarc@mexcantech.com"
MEMBER_EMAIL = "ana@example.com"
PASSWORD = "a perfectly fine passphrase"
NEW_PASSWORD = "an entirely different one"
CHEAP = Argon2Params(memory_kib=64, iterations=1, lanes=1)


class Harness:
    def __init__(self, tmp_path, *, login_limit=100, password_login=True) -> None:
        self.now = 1_700_000_000
        keyring = KeyRing.from_env(profile_crypto.generate_key_material(version=1))
        self.store = UserStore(tmp_path / "assistant.sqlite3", keyring=keyring)
        self.config = MultiUserConfig(
            internal_auth_secret=SECRET,
            keyring=keyring,
            bootstrap_admin_email=ADMIN_EMAIL,
            access=None,
            store_path=tmp_path / "assistant.sqlite3",
            password_login=password_login,
            session_policy=SessionPolicy(),
        )
        self.auth = LocalAuth(
            self.store, params=CHEAP, policy=SessionPolicy(), clock=lambda: self.now
        )
        self.runtime = IdentityRuntime(
            self.config,
            store=self.store,
            access_verifier=None,
            local_auth=self.auth,
            login_limiter=RateLimiter(limit=login_limit, window_seconds=60),
            clock=lambda: self.now,
        )

    def identity_headers(self, client_key: str = "ip:1.2.3.4") -> dict[str, str]:
        token = mint_principal(
            secret=SECRET,
            subject="identity",
            audience=AUDIENCE_IDENTITY,
            claims={"client": client_key},
            now=self.now,
        )
        return {"Authorization": f"Bearer {token}"}

    def bearer(self, user_id: str) -> dict[str, str]:
        token = mint_principal(
            secret=SECRET, subject=user_id, audience=AUDIENCE_BACKEND, now=self.now
        )
        return {"Authorization": f"Bearer {token}"}

    def make_user(self, email: str, *, role: str = MEMBER, password: str | None = PASSWORD):
        profile = self.store.create_user(
            email=email, display_name=email.split("@")[0], role=role, actor=Actor.system()
        )
        if password is not None:
            self.auth.set_password(profile.user_id, password, actor=Actor.system())
        return profile

    def login(self, client: TestClient, email: str, password: str, **kwargs):
        return client.post(
            "/auth/login",
            headers={**self.identity_headers(kwargs.pop("client_key", "ip:1.2.3.4"))},
            json={"email": email, "password": password},
            **kwargs,
        )


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


# --- status ------------------------------------------------------------------------


def test_status_advertises_the_available_sign_in_methods(harness, client) -> None:
    body = client.get("/identity/status").json()

    assert body["configured"] is True
    assert body["password_login"] is True
    assert body["cloudflare_access"] is False


def test_status_reveals_nothing_else(harness, client) -> None:
    body = client.get("/identity/status").json()

    assert set(body) == {"configured", "password_login", "cloudflare_access"}


# --- sign in -----------------------------------------------------------------------


def test_login_returns_a_session_and_the_user(harness, client) -> None:
    profile = harness.make_user(MEMBER_EMAIL)

    response = harness.login(client, MEMBER_EMAIL, PASSWORD)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["user_id"] == profile.user_id
    assert body["role"] == MEMBER
    assert body["must_change_password"] is False
    assert body["session_token"]
    assert body["expires_at"] > harness.now


def test_login_response_never_carries_the_email_or_a_hash(harness, client) -> None:
    harness.make_user(MEMBER_EMAIL)

    text = harness.login(client, MEMBER_EMAIL, PASSWORD).text

    assert MEMBER_EMAIL not in text
    assert "argon2" not in text
    assert PASSWORD not in text


def test_login_requires_the_internal_principal(harness, client) -> None:
    harness.make_user(MEMBER_EMAIL)

    naked = client.post("/auth/login", json={"email": MEMBER_EMAIL, "password": PASSWORD})
    assert naked.status_code == 401

    wrong_audience = client.post(
        "/auth/login",
        headers={
            "Authorization": "Bearer "
            + mint_principal(
                secret=SECRET,
                subject="identity",
                audience=AUDIENCE_BACKEND,
                now=harness.now,
            )
        },
        json={"email": MEMBER_EMAIL, "password": PASSWORD},
    )
    assert wrong_audience.status_code == 401


def test_an_internal_principal_is_single_use(harness, client) -> None:
    harness.make_user(MEMBER_EMAIL)
    headers = harness.identity_headers()

    body = {"email": MEMBER_EMAIL, "password": PASSWORD}
    first = client.post("/auth/login", headers=headers, json=body)
    replay = client.post("/auth/login", headers=headers, json=body)

    assert first.status_code == 200
    assert replay.status_code == 401


@pytest.mark.parametrize(
    "email, password",
    [
        (MEMBER_EMAIL, "the wrong password entirely"),
        ("nobody@example.com", PASSWORD),
        ("not-an-email", PASSWORD),
    ],
)
def test_every_refusal_looks_the_same(harness, client, email, password) -> None:
    harness.make_user(MEMBER_EMAIL)

    response = harness.login(client, email, password)

    assert response.status_code == 401
    assert response.json()["detail"] == "invalid_credentials"


def test_a_suspended_user_cannot_sign_in_and_is_not_told_why(harness, client) -> None:
    profile = harness.make_user(MEMBER_EMAIL)
    harness.store.admin_update(profile.user_id, status=SUSPENDED, actor=Actor.system())

    response = harness.login(client, MEMBER_EMAIL, PASSWORD)

    assert response.status_code == 401
    assert response.json()["detail"] == "invalid_credentials"


def test_a_lockout_is_reported_with_retry_after(harness, client) -> None:
    harness.make_user(MEMBER_EMAIL)
    for _ in range(SessionPolicy().max_failed_attempts):
        harness.login(client, MEMBER_EMAIL, "wrong one")

    response = harness.login(client, MEMBER_EMAIL, PASSWORD)

    assert response.status_code == 429
    assert response.json()["detail"] == "locked"
    assert int(response.headers["Retry-After"]) > 0


def test_login_is_rate_limited_per_client_not_per_backend_connection(tmp_path) -> None:
    harness = Harness(tmp_path, login_limit=3)
    webhooks.app.dependency_overrides[user_api.get_runtime] = lambda: harness.runtime
    try:
        with TestClient(webhooks.app) as client:
            harness.make_user(MEMBER_EMAIL)
            for _ in range(3):
                harness.login(client, MEMBER_EMAIL, "wrong", client_key="ip:9.9.9.9")

            blocked = harness.login(client, MEMBER_EMAIL, PASSWORD, client_key="ip:9.9.9.9")
            other = harness.login(client, MEMBER_EMAIL, PASSWORD, client_key="ip:8.8.8.8")

            assert blocked.status_code == 429
            assert other.status_code == 200
    finally:
        webhooks.app.dependency_overrides.pop(user_api.get_runtime, None)


def test_login_is_refused_outright_when_password_sign_in_is_disabled(tmp_path) -> None:
    harness = Harness(tmp_path, password_login=False)
    webhooks.app.dependency_overrides[user_api.get_runtime] = lambda: harness.runtime
    try:
        with TestClient(webhooks.app) as client:
            harness.make_user(MEMBER_EMAIL)
            assert harness.login(client, MEMBER_EMAIL, PASSWORD).status_code == 404
            assert client.get("/identity/status").json()["password_login"] is False
    finally:
        webhooks.app.dependency_overrides.pop(user_api.get_runtime, None)


@pytest.mark.parametrize(
    "body",
    [
        {"email": MEMBER_EMAIL},
        {"password": PASSWORD},
        {"email": MEMBER_EMAIL, "password": PASSWORD, "role": "admin"},
        {"email": MEMBER_EMAIL, "password": "x" * 5000},
        {"email": "x" * 500, "password": PASSWORD},
    ],
)
def test_malformed_login_bodies_are_refused_by_schema(harness, client, body) -> None:
    assert harness.login and client.post(
        "/auth/login", headers=harness.identity_headers(), json=body
    ).status_code in (401, 422)


# --- session verification ----------------------------------------------------------


def test_a_session_token_resolves_to_its_user(harness, client) -> None:
    profile = harness.make_user(MEMBER_EMAIL)
    token = harness.login(client, MEMBER_EMAIL, PASSWORD).json()["session_token"]

    response = client.post(
        "/auth/session", headers=harness.identity_headers(), json={"session_token": token}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == profile.user_id
    assert body["display_name"] == "ana"
    assert body["must_change_password"] is False
    assert "email" not in body


@pytest.mark.parametrize("token", ["", "nope", "a" * 400])
def test_a_bad_session_token_is_unauthorized(harness, client, token) -> None:
    response = client.post(
        "/auth/session", headers=harness.identity_headers(), json={"session_token": token}
    )

    assert response.status_code in (401, 422)


def test_a_revoked_session_stops_working_immediately(harness, client) -> None:
    harness.make_user(MEMBER_EMAIL)
    token = harness.login(client, MEMBER_EMAIL, PASSWORD).json()["session_token"]

    logout = client.post(
        "/auth/logout", headers=harness.identity_headers(), json={"session_token": token}
    )
    assert logout.status_code == 204

    after = client.post(
        "/auth/session", headers=harness.identity_headers(), json={"session_token": token}
    )
    assert after.status_code == 401


def test_logging_out_an_unknown_token_still_succeeds(harness, client) -> None:
    response = client.post(
        "/auth/logout", headers=harness.identity_headers(), json={"session_token": "unknown"}
    )

    assert response.status_code == 204


def test_suspending_a_user_invalidates_their_session(harness, client) -> None:
    profile = harness.make_user(MEMBER_EMAIL)
    token = harness.login(client, MEMBER_EMAIL, PASSWORD).json()["session_token"]

    harness.store.admin_update(profile.user_id, status=SUSPENDED, actor=Actor.system())

    response = client.post(
        "/auth/session", headers=harness.identity_headers(), json={"session_token": token}
    )
    assert response.status_code == 401


# --- changing a password -----------------------------------------------------------


def test_a_user_may_change_their_own_password(harness, client) -> None:
    profile = harness.make_user(MEMBER_EMAIL)
    token = harness.login(client, MEMBER_EMAIL, PASSWORD).json()["session_token"]

    response = client.post(
        "/auth/password",
        headers=harness.bearer(profile.user_id),
        json={
            "current_password": PASSWORD,
            "new_password": NEW_PASSWORD,
            "keep_session_token": token,
        },
    )

    assert response.status_code == 204
    assert harness.login(client, MEMBER_EMAIL, PASSWORD).status_code == 401
    assert harness.login(client, MEMBER_EMAIL, NEW_PASSWORD).status_code == 200


def test_changing_a_password_keeps_the_calling_session_alive(harness, client) -> None:
    profile = harness.make_user(MEMBER_EMAIL)
    token = harness.login(client, MEMBER_EMAIL, PASSWORD).json()["session_token"]
    doomed = harness.login(client, MEMBER_EMAIL, PASSWORD).json()["session_token"]

    client.post(
        "/auth/password",
        headers=harness.bearer(profile.user_id),
        json={
            "current_password": PASSWORD,
            "new_password": NEW_PASSWORD,
            "keep_session_token": token,
        },
    )

    kept = client.post(
        "/auth/session", headers=harness.identity_headers(), json={"session_token": token}
    )
    killed = client.post(
        "/auth/session", headers=harness.identity_headers(), json={"session_token": doomed}
    )
    assert kept.status_code == 200
    assert killed.status_code == 401


def test_the_wrong_current_password_is_refused(harness, client) -> None:
    profile = harness.make_user(MEMBER_EMAIL)

    response = client.post(
        "/auth/password",
        headers=harness.bearer(profile.user_id),
        json={"current_password": "not it", "new_password": NEW_PASSWORD},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "invalid_credentials"


def test_a_weak_or_reused_new_password_is_refused_with_a_reason(harness, client) -> None:
    profile = harness.make_user(MEMBER_EMAIL)

    weak = client.post(
        "/auth/password",
        headers=harness.bearer(profile.user_id),
        json={"current_password": PASSWORD, "new_password": "short"},
    )
    reused = client.post(
        "/auth/password",
        headers=harness.bearer(profile.user_id),
        json={"current_password": PASSWORD, "new_password": PASSWORD},
    )

    assert weak.status_code == 422
    assert reused.status_code == 422
    assert reused.json()["detail"] == "password_policy"


def test_changing_a_password_needs_a_backend_principal(harness, client) -> None:
    harness.make_user(MEMBER_EMAIL)

    response = client.post(
        "/auth/password",
        json={"current_password": PASSWORD, "new_password": NEW_PASSWORD},
    )

    assert response.status_code == 401


def test_one_user_cannot_change_another_users_password(harness, client) -> None:
    harness.make_user(MEMBER_EMAIL)
    attacker = harness.make_user("mallory@example.com")

    # The principal names the attacker, so the route can only ever act on them.
    response = client.post(
        "/auth/password",
        headers=harness.bearer(attacker.user_id),
        json={"current_password": PASSWORD, "new_password": NEW_PASSWORD},
    )

    assert response.status_code == 204
    assert harness.login(client, MEMBER_EMAIL, PASSWORD).status_code == 200


# --- forced change -----------------------------------------------------------------


def test_a_forced_change_is_advertised_on_login_and_on_the_session(harness, client) -> None:
    profile = harness.make_user(MEMBER_EMAIL, password=None)
    harness.auth.set_password(
        profile.user_id, PASSWORD, actor=Actor.system(), must_change=True
    )

    login = harness.login(client, MEMBER_EMAIL, PASSWORD).json()
    session = client.post(
        "/auth/session",
        headers=harness.identity_headers(),
        json={"session_token": login["session_token"]},
    ).json()

    assert login["must_change_password"] is True
    assert session["must_change_password"] is True


# --- administrative reset ----------------------------------------------------------


def test_an_admin_can_issue_a_one_time_password(harness, client) -> None:
    admin = harness.make_user(ADMIN_EMAIL, role=ADMIN)
    target = harness.make_user(MEMBER_EMAIL)

    response = client.post(
        f"/admin/users/{target.user_id}/password", headers=harness.bearer(admin.user_id)
    )

    assert response.status_code == 200
    issued = response.json()["one_time_password"]
    assert len(issued) >= 20
    assert response.headers["cache-control"] == "no-store"
    login = harness.login(client, MEMBER_EMAIL, issued)
    assert login.status_code == 200
    assert login.json()["must_change_password"] is True


def test_a_member_cannot_issue_a_password_for_anyone(harness, client) -> None:
    member = harness.make_user(MEMBER_EMAIL)
    victim = harness.make_user("victim@example.com")

    response = client.post(
        f"/admin/users/{victim.user_id}/password", headers=harness.bearer(member.user_id)
    )

    assert response.status_code == 403


def test_resetting_a_password_for_an_unknown_user_is_a_404(harness, client) -> None:
    admin = harness.make_user(ADMIN_EMAIL, role=ADMIN)

    for bogus in ["usr_" + "0" * 24, "not-a-user-id", "../../etc/passwd"]:
        response = client.post(
            f"/admin/users/{bogus}/password", headers=harness.bearer(admin.user_id)
        )
        assert response.status_code == 404


def test_an_admin_creating_a_user_can_get_them_a_one_time_password(harness, client) -> None:
    admin = harness.make_user(ADMIN_EMAIL, role=ADMIN)

    created = client.post(
        "/admin/users",
        headers=harness.bearer(admin.user_id),
        json={
            "email": "new@example.com",
            "display_name": "New",
            "role": MEMBER,
            "with_password": True,
        },
    )

    assert created.status_code == 201
    issued = created.json()["one_time_password"]
    assert issued
    login = harness.login(client, "new@example.com", issued)
    assert login.status_code == 200
    assert login.json()["must_change_password"] is True


def test_creating_a_user_without_a_password_issues_none(harness, client) -> None:
    admin = harness.make_user(ADMIN_EMAIL, role=ADMIN)

    created = client.post(
        "/admin/users",
        headers=harness.bearer(admin.user_id),
        json={"email": "new@example.com", "display_name": "New", "role": MEMBER},
    )

    assert created.status_code == 201
    assert created.json()["one_time_password"] is None


# --- caching -----------------------------------------------------------------------


def test_no_authentication_response_is_ever_cacheable(harness, client) -> None:
    harness.make_user(MEMBER_EMAIL)
    responses = [
        harness.login(client, MEMBER_EMAIL, PASSWORD),
        harness.login(client, MEMBER_EMAIL, "wrong"),
        client.get("/identity/status"),
    ]

    for response in responses:
        assert response.headers["cache-control"] == "no-store"


def test_suspending_a_user_through_the_admin_api_tears_down_their_sessions(
    harness, client
) -> None:
    admin = harness.make_user(ADMIN_EMAIL, role=ADMIN)
    target = harness.make_user(MEMBER_EMAIL)
    token = harness.login(client, MEMBER_EMAIL, PASSWORD).json()["session_token"]

    response = client.patch(
        f"/admin/users/{target.user_id}",
        headers=harness.bearer(admin.user_id),
        json={"status": SUSPENDED},
    )

    assert response.status_code == 200
    assert harness.auth.verify_session(token) is None


def test_a_forced_change_session_is_still_a_real_session_for_changing_the_password(
    harness, client
) -> None:
    """The flag must gate what a session may do, not whether it exists."""
    profile = harness.make_user(MEMBER_EMAIL, password=None)
    harness.auth.set_password(profile.user_id, PASSWORD, actor=Actor.system(), must_change=True)
    token = harness.login(client, MEMBER_EMAIL, PASSWORD).json()["session_token"]

    changed = client.post(
        "/auth/password",
        headers=harness.bearer(profile.user_id),
        json={
            "current_password": PASSWORD,
            "new_password": NEW_PASSWORD,
            "keep_session_token": token,
        },
    )

    assert changed.status_code == 204
    session = client.post(
        "/auth/session", headers=harness.identity_headers(), json={"session_token": token}
    )
    assert session.status_code == 200
    assert session.json()["must_change_password"] is False
