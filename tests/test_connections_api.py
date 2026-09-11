"""HTTP contract for a user's own provider connections.

``/users/me/connections`` sits behind the same internal trust boundary as the
rest of the identity API: the Next.js BFF proves itself with a single-use
signed principal, the backend loads the user from its own database, and every
row read or written is scoped to that user. The OAuth ``state`` is opaque,
signed, one-time, expiry-bound and user-bound; provider tokens are encrypted
at rest and never appear in a response or a log line.

No test here talks to a real provider: the token exchange is an injected
collaborator, and the production runtime has none until a transport is added.
"""

from __future__ import annotations

import logging
from contextlib import closing
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from caal import connections_api, profile_crypto, user_api, webhooks
from caal.connections_api import ConnectionsRuntime
from caal.internal_auth import AUDIENCE_AGENT, AUDIENCE_BACKEND, RateLimiter, mint_principal
from caal.oauth_providers import (
    ENV_PUBLIC_ORIGIN,
    OAUTH_CALLBACK_PATH,
    TokenExchangeError,
    TokenGrant,
    load_provider_registry,
)
from caal.profile_crypto import KeyRing
from caal.security_config import MultiUserConfig
from caal.user_api import IdentityRuntime
from caal.user_store import MEMBER, SUSPENDED, Actor, UserStore

SECRET = "s" * 48
BOOTSTRAP = "cesarc@mexcantech.com"
NOW = 1_700_000_000
ORIGIN = "https://jarvis.example.com"
GOOGLE_SECRET = "google-client-secret-value"
GOOGLE_ENV = {
    "CAAL_OAUTH_GOOGLE_CLIENT_ID": "google-client-id.apps.googleusercontent.com",
    "CAAL_OAUTH_GOOGLE_CLIENT_SECRET": GOOGLE_SECRET,
    ENV_PUBLIC_ORIGIN: ORIGIN,
}
ACCESS = "ya29.ACCESS-TOKEN-PLAINTEXT"
REFRESH = "1//REFRESH-TOKEN-PLAINTEXT"
CODE = "4/0AX4XfWh-AUTHORIZATION-CODE"
_UNSET = object()


class FakeExchanger:
    """Stands in for the provider's token endpoint. Records what it was asked."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.error: Exception | None = None
        self.grant = TokenGrant(
            access_token=ACCESS,
            refresh_token=REFRESH,
            expires_in=3600,
            scopes=("openid", "email", "https://www.googleapis.com/auth/calendar.readonly"),
            provider_account_id="google-sub-ana",
            account_label="ana@gmail.com",
        )

    async def exchange(self, config, *, code, redirect_uri, code_verifier):
        self.calls.append(
            {
                "provider": config.provider,
                "code": code,
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            }
        )
        if self.error is not None:
            raise self.error
        return self.grant


class Harness:
    def __init__(self, tmp_path, *, mutation_limit=100, exchanger=_UNSET, env=None) -> None:
        self.now = NOW
        self.keyring = KeyRing.from_env(profile_crypto.generate_key_material(version=1))
        self.store = UserStore(tmp_path / "assistant.sqlite3", keyring=self.keyring)
        self.config = MultiUserConfig(
            internal_auth_secret=SECRET,
            keyring=self.keyring,
            bootstrap_admin_email=BOOTSTRAP,
            store_path=tmp_path / "assistant.sqlite3",
        )
        self.runtime = IdentityRuntime(
            self.config,
            store=self.store,
            mutation_limiter=RateLimiter(limit=mutation_limit, window_seconds=60),
            clock=lambda: self.now,
        )
        self.exchanger = FakeExchanger() if exchanger is _UNSET else exchanger
        self.registry = load_provider_registry(GOOGLE_ENV if env is None else env)
        self.connections = ConnectionsRuntime(
            self.runtime, providers=self.registry, exchanger=self.exchanger
        )
        self.ana = self.user("ana@example.com")
        self.bo = self.user("bo@example.com")

    def user(self, email: str, role: str = MEMBER) -> str:
        return self.store.create_user(
            email=email,
            display_name=email.split("@")[0],
            role=role,
            actor=Actor.system(),
            now=self.now,
        ).user_id

    def principal(self, user_id: str, **overrides) -> str:
        params = {
            "secret": SECRET,
            "subject": user_id,
            "audience": AUDIENCE_BACKEND,
            "now": self.now,
        }
        params.update(overrides)
        return mint_principal(**params)

    def bearer(self, user_id: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.principal(user_id)}"}

    def rows(self, table: str):
        with closing(self.store.connect()) as connection:
            return connection.execute(f"SELECT * FROM {table}").fetchall()


def _install(harness: Harness):
    webhooks.app.dependency_overrides[user_api.get_runtime] = lambda: harness.runtime
    webhooks.app.dependency_overrides[connections_api.get_connections_runtime] = (
        lambda: harness.connections
    )


def _uninstall() -> None:
    webhooks.app.dependency_overrides.pop(user_api.get_runtime, None)
    webhooks.app.dependency_overrides.pop(connections_api.get_connections_runtime, None)


_SLICE_LOGGERS = (
    "caal.connections_api",
    "caal.provider_connections",
    "caal.oauth_providers",
    "caal.oauth_exchange",
    "httpx",
    "httpcore",
)


@pytest.fixture(autouse=True)
def capture_slice_logs(caplog):
    """See every record the slice emits, even after ``voice_agent`` stops propagation.

    ``voice_agent.py`` (imported by other test modules) sets ``propagate =
    False`` on the ``caal`` logger, which would make every "nothing leaked
    into the log" assertion here vacuous in a full run.
    """
    targets = [logging.getLogger(name) for name in _SLICE_LOGGERS]
    previous = [(target.level, target.propagate) for target in targets]
    for target in targets:
        target.setLevel(logging.DEBUG)
        target.addHandler(caplog.handler)
    try:
        yield
    finally:
        for target, (level, propagate) in zip(targets, previous):
            target.removeHandler(caplog.handler)
            target.setLevel(level)
            target.propagate = propagate


@pytest.fixture
def harness(tmp_path):
    return Harness(tmp_path)


@pytest.fixture
def client(harness):
    _install(harness)
    try:
        with TestClient(webhooks.app) as test_client:
            yield test_client
    finally:
        _uninstall()


def _authorize(harness, client, user_id: str, provider: str = "google"):
    return client.post(
        f"/users/me/connections/{provider}/authorize", headers=harness.bearer(user_id)
    )


def _state_of(response) -> str:
    assert response.status_code == 200, response.text
    query = parse_qs(urlsplit(response.json()["authorization_url"]).query, strict_parsing=True)
    return query["state"][0]


def _callback(harness, client, user_id: str, state: str, code: str = CODE, **extra):
    return client.post(
        "/users/me/connections/callback",
        headers=harness.bearer(user_id),
        json={"state": state, "code": code, **extra},
    )


def _connect(harness, client, user_id: str, provider: str = "google") -> dict:
    state = _state_of(_authorize(harness, client, user_id, provider))
    response = _callback(harness, client, user_id, state)
    assert response.status_code == 200, response.text
    return response.json()


# --- configuration gate and authentication -------------------------------------------


def test_everything_fails_closed_without_multi_user_configuration(harness) -> None:
    webhooks.app.dependency_overrides[user_api.get_runtime] = lambda: None
    try:
        with TestClient(webhooks.app) as client:
            headers = harness.bearer(harness.ana)
            assert client.get("/users/me/connections", headers=headers).status_code == 503
            assert _authorize(harness, client, harness.ana).status_code == 503
            assert _callback(harness, client, harness.ana, "state").status_code == 503
            assert (
                client.delete("/users/me/connections/con_" + "0" * 24, headers=headers).status_code
                == 503
            )
    finally:
        _uninstall()


def test_every_route_requires_a_valid_single_use_backend_principal(harness, client) -> None:
    connection_id = "con_" + "0" * 24
    attempts = [
        ("GET", "/users/me/connections", None),
        ("POST", "/users/me/connections/google/authorize", None),
        ("POST", "/users/me/connections/callback", {"state": "s" * 40, "code": "c"}),
        ("DELETE", f"/users/me/connections/{connection_id}", None),
    ]
    for method, path, body in attempts:
        assert client.request(method, path, json=body).status_code == 401, path
        assert (
            client.request(
                method, path, headers={"Authorization": "Bearer nope"}, json=body
            ).status_code
            == 401
        ), path
        wrong_audience = harness.principal(harness.ana, audience=AUDIENCE_AGENT)
        assert (
            client.request(
                method, path, headers={"Authorization": f"Bearer {wrong_audience}"}, json=body
            ).status_code
            == 401
        ), path
        forged = mint_principal(secret="x" * 48, subject=harness.ana, audience=AUDIENCE_BACKEND)
        assert (
            client.request(
                method, path, headers={"Authorization": f"Bearer {forged}"}, json=body
            ).status_code
            == 401
        ), path
        unknown = harness.bearer("usr_" + "f" * 24)
        assert client.request(method, path, headers=unknown, json=body).status_code == 401, path

    # A principal is single use.
    headers = harness.bearer(harness.ana)
    assert client.get("/users/me/connections", headers=headers).status_code == 200
    assert client.get("/users/me/connections", headers=headers).status_code == 401

    # Suspended users are refused even with a valid principal.
    harness.store.admin_update(harness.bo, status=SUSPENDED, actor=Actor.system(), now=harness.now)
    assert (
        client.get("/users/me/connections", headers=harness.bearer(harness.bo)).status_code == 403
    )
    assert _authorize(harness, client, harness.bo).status_code == 403


# --- listing --------------------------------------------------------------------------


def test_listing_starts_empty_and_reports_which_providers_are_configured(harness, client) -> None:
    response = client.get("/users/me/connections", headers=harness.bearer(harness.ana))

    assert response.status_code == 200
    assert response.json() == {
        "connections": [],
        "providers": [
            {"provider": "google", "display_name": "Google", "configured": True},
            {"provider": "microsoft", "display_name": "Microsoft", "configured": False},
            {"provider": "zoho", "display_name": "Zoho", "configured": False},
        ],
        "token_exchange_available": True,
    }
    assert GOOGLE_SECRET not in response.text
    assert "client" not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_connection_routes_never_carry_permissive_cors_headers(harness, client) -> None:
    evil = {"Origin": "https://evil.example"}
    responses = [
        client.get("/users/me/connections", headers={**evil, **harness.bearer(harness.ana)}),
        client.options(
            "/users/me/connections/google/authorize",
            headers={**evil, "Access-Control-Request-Method": "POST"},
        ),
        client.get("/users/me/connections"),
    ]
    for response in responses:
        names = {name.lower() for name in response.headers}
        assert "access-control-allow-origin" not in names, response.url
        assert "access-control-allow-credentials" not in names, response.url
        assert response.headers.get("cache-control") == "no-store", response.url


# --- authorization initiation ------------------------------------------------------------


def test_authorize_is_explicit_when_the_provider_is_not_configured(harness, client) -> None:
    response = _authorize(harness, client, harness.ana, "zoho")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "provider_not_configured",
        "status": "configuration_needed",
        "provider": "zoho",
        "missing": [
            "CAAL_OAUTH_ZOHO_CLIENT_ID",
            "CAAL_OAUTH_ZOHO_CLIENT_SECRET",
        ],
    }
    assert harness.rows("oauth_states") == []
    assert response.headers["cache-control"] == "no-store"


def test_authorize_reports_a_missing_public_origin_by_name(tmp_path) -> None:
    harness = Harness(tmp_path, env={k: v for k, v in GOOGLE_ENV.items() if k != ENV_PUBLIC_ORIGIN})
    _install(harness)
    try:
        with TestClient(webhooks.app) as client:
            response = _authorize(harness, client, harness.ana, "google")
            assert response.status_code == 503
            assert response.json()["status"] == "configuration_needed"
            assert response.json()["missing"] == [ENV_PUBLIC_ORIGIN]
    finally:
        _uninstall()


def test_authorize_returns_an_opaque_state_bound_authorization_url(harness, client) -> None:
    response = _authorize(harness, client, harness.ana)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"provider", "authorization_url", "expires_at"}
    assert body["provider"] == "google"
    assert body["expires_at"] == harness.now + connections_api.STATE_TTL_SECONDS

    parts = urlsplit(body["authorization_url"])
    assert parts.scheme == "https" and parts.netloc == "accounts.google.com"
    query = parse_qs(parts.query, strict_parsing=True)
    assert query["client_id"] == [GOOGLE_ENV["CAAL_OAUTH_GOOGLE_CLIENT_ID"]]
    assert query["redirect_uri"] == [ORIGIN + OAUTH_CALLBACK_PATH]
    assert query["response_type"] == ["code"]
    assert query["code_challenge_method"] == ["S256"]
    assert len(query["code_challenge"][0]) == 43
    state = query["state"][0]
    assert len(state) >= 40
    assert harness.ana not in state and "google" not in state
    assert GOOGLE_SECRET not in response.text
    assert "client_secret" not in response.text

    (row,) = harness.rows("oauth_states")
    assert row["user_id"] == harness.ana and row["provider"] == "google"
    assert row["consumed_at"] is None
    assert state not in str(tuple(row))  # the state itself is never stored verbatim

    actions = [event.action for event in harness.store.list_audit_events(limit=10)]
    assert actions[0] == "connection.authorize"
    event = harness.store.list_audit_events(limit=1)[0]
    assert event.target_id == harness.ana and event.detail == {"provider": "google"}


def test_unknown_providers_are_not_found_and_bodies_are_strict(harness, client) -> None:
    for provider in ("apple", "GOOGLE", "google%20", "settings"):
        response = client.post(
            f"/users/me/connections/{provider}/authorize", headers=harness.bearer(harness.ana)
        )
        assert response.status_code == 404, provider
        assert response.json()["detail"] == "unknown_provider"

    # No route accepts a password or an app password, ever.
    for body in ({"password": "hunter2"}, {"app_password": "abcd"}, {"username": "ana"}):
        assert (
            client.post(
                "/users/me/connections/google/authorize",
                headers=harness.bearer(harness.ana),
                json=body,
            ).status_code
            == 422
        ), body
        assert (
            client.post(
                "/users/me/connections/callback",
                headers=harness.bearer(harness.ana),
                json={"state": "s" * 40, "code": "c", **body},
            ).status_code
            == 422
        ), body
    for body in ({"state": "s" * 40}, {"code": "c"}, {"state": "", "code": "c"}, {}):
        assert (
            client.post(
                "/users/me/connections/callback", headers=harness.bearer(harness.ana), json=body
            ).status_code
            == 422
        ), body
    assert harness.rows("oauth_states") == []
    assert harness.rows("provider_connections") == []


# --- callback ---------------------------------------------------------------------------------


def test_callback_completes_the_connection_and_never_returns_tokens(
    harness, client, caplog
) -> None:
    with caplog.at_level(logging.DEBUG):
        authorize = _authorize(harness, client, harness.ana)
        state = _state_of(authorize)
        response = _callback(harness, client, harness.ana, state)

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {
        "connection_id",
        "provider",
        "status",
        "account_label",
        "user_label",
        "aliases",
        "scopes",
        "token_expires_at",
        "has_refresh_token",
        "connected_at",
        "updated_at",
    }
    assert body["provider"] == "google" and body["status"] == "connected"
    assert body["account_label"] == "ana@gmail.com"
    assert body["has_refresh_token"] is True
    assert body["token_expires_at"] == harness.now + 3600
    assert body["scopes"] == list(harness.exchanger.grant.scopes)
    assert body["connection_id"].startswith("con_")
    assert "access_token" not in body and "refresh_token" not in body
    for secret in (ACCESS, REFRESH, CODE, state, GOOGLE_SECRET):
        assert secret not in response.text, secret

    # The exchange was asked with exactly what the provider expects, and PKCE.
    (call,) = harness.exchanger.calls
    assert call["provider"] == "google" and call["code"] == CODE
    assert call["redirect_uri"] == ORIGIN + OAUTH_CALLBACK_PATH
    assert 43 <= len(call["code_verifier"]) <= 128

    # Tokens are ciphertext at rest, and the state row is consumed.
    (row,) = harness.rows("provider_connections")
    for column in row.keys():
        assert ACCESS not in str(row[column]) and REFRESH not in str(row[column]), column
    assert row["access_token_enc"].startswith("enc:v1:")
    assert harness.rows("oauth_states")[0]["consumed_at"] == harness.now

    listed = client.get("/users/me/connections", headers=harness.bearer(harness.ana))
    assert [c["connection_id"] for c in listed.json()["connections"]] == [body["connection_id"]]
    assert ACCESS not in listed.text and REFRESH not in listed.text

    for secret in (ACCESS, REFRESH, CODE, state, GOOGLE_SECRET, "ana@gmail.com", "ana@example.com"):
        assert secret not in caplog.text, secret

    actions = [event.action for event in harness.store.list_audit_events(limit=10)]
    assert actions[:2] == ["connection.connect", "connection.authorize"]
    assert ACCESS not in str(harness.store.list_audit_events(limit=10))


def test_callback_state_is_single_use_expiry_bound_and_user_bound(harness, client) -> None:
    # Replay.
    state = _state_of(_authorize(harness, client, harness.ana))
    assert _callback(harness, client, harness.ana, state).status_code == 200
    replayed = _callback(harness, client, harness.ana, state)
    assert replayed.status_code == 400 and replayed.json()["detail"] == "invalid_state"

    # Expiry.
    state = _state_of(_authorize(harness, client, harness.ana))
    harness.now += connections_api.STATE_TTL_SECONDS
    expired = _callback(harness, client, harness.ana, state)
    assert expired.status_code == 400 and expired.json()["detail"] == "invalid_state"

    # Another user's state: refused, and burned for the real owner as well.
    state = _state_of(_authorize(harness, client, harness.ana))
    stolen = _callback(harness, client, harness.bo, state)
    assert stolen.status_code == 400 and stolen.json()["detail"] == "invalid_state"
    burned = _callback(harness, client, harness.ana, state)
    assert burned.status_code == 400 and burned.json()["detail"] == "invalid_state"

    # Tampered and unknown states.
    state = _state_of(_authorize(harness, client, harness.ana))
    state_id, signature = state.split(".")
    tampered = state_id + "." + ("A" if signature[-1] != "A" else "B") + signature[:-1]
    for bad in (tampered, "unknown." + signature, "x" * 60, state_id):
        response = _callback(harness, client, harness.ana, bad)
        assert response.status_code == 400 and response.json()["detail"] == "invalid_state", bad

    # Only the one genuine, first-use callback produced a connection; nothing
    # was minted for bo at all.
    assert len(harness.exchanger.calls) == 1
    assert [row["user_id"] for row in harness.rows("provider_connections")] == [harness.ana]
    assert (
        client.get("/users/me/connections", headers=harness.bearer(harness.bo)).json()[
            "connections"
        ]
        == []
    )


def test_callback_without_a_token_exchange_transport_is_an_explicit_error(tmp_path) -> None:
    harness = Harness(tmp_path, exchanger=None)
    _install(harness)
    try:
        with TestClient(webhooks.app) as client:
            listed = client.get("/users/me/connections", headers=harness.bearer(harness.ana))
            assert listed.json()["token_exchange_available"] is False
            state = _state_of(_authorize(harness, client, harness.ana))
            response = _callback(harness, client, harness.ana, state)
            assert response.status_code == 503
            assert response.json()["detail"] == "token_exchange_unavailable"
            # The state was still consumed: a retry needs a fresh authorization.
            assert (
                _callback(harness, client, harness.ana, state).json()["detail"] == "invalid_state"
            )
            assert harness.rows("provider_connections") == []
    finally:
        _uninstall()


def test_exchange_failures_are_reported_without_leaking_details(harness, client, caplog) -> None:
    harness.exchanger.error = TokenExchangeError("provider said: invalid_grant sk-live-LEAK")
    state = _state_of(_authorize(harness, client, harness.ana))
    with caplog.at_level(logging.DEBUG):
        response = _callback(harness, client, harness.ana, state)

    assert response.status_code == 502
    assert response.json() == {
        "detail": "token_exchange_failed",
        "reason": "provider_refused",
        "provider": "google",
        "settings": [],
    }
    assert "LEAK" not in response.text and "LEAK" not in caplog.text
    assert CODE not in caplog.text and state not in caplog.text
    assert harness.rows("provider_connections") == []
    assert _callback(harness, client, harness.ana, state).json()["detail"] == "invalid_state"


# --- disconnect ------------------------------------------------------------------------------


def test_disconnect_revokes_only_the_owners_connection(harness, client) -> None:
    mine = _connect(harness, client, harness.ana)["connection_id"]
    theirs = _connect(harness, client, harness.bo)["connection_id"]

    stolen = client.delete(f"/users/me/connections/{mine}", headers=harness.bearer(harness.bo))
    assert stolen.status_code == 404 and stolen.json()["detail"] == "not_found"
    assert harness.connections.store.credentials(harness.ana, mine) is not None

    for bad in ("con_" + "z" * 24, "usr_" + "0" * 24, "nope", "con_" + "0" * 23):
        assert (
            client.delete(
                f"/users/me/connections/{bad}", headers=harness.bearer(harness.ana)
            ).status_code
            == 404
        ), bad

    revoked = client.delete(f"/users/me/connections/{mine}", headers=harness.bearer(harness.ana))
    assert revoked.status_code == 204 and revoked.content == b""
    assert (
        client.get("/users/me/connections", headers=harness.bearer(harness.ana)).json()[
            "connections"
        ]
        == []
    )
    assert harness.connections.store.credentials(harness.ana, mine) is None
    again = client.delete(f"/users/me/connections/{mine}", headers=harness.bearer(harness.ana))
    assert again.status_code == 404

    # Bo's connection is untouched.
    listed = client.get("/users/me/connections", headers=harness.bearer(harness.bo)).json()
    assert [c["connection_id"] for c in listed["connections"]] == [theirs]
    assert harness.connections.store.credentials(harness.bo, theirs).access_token == ACCESS

    actions = [event.action for event in harness.store.list_audit_events(limit=10)]
    assert actions[0] == "connection.revoke"
    assert harness.store.list_audit_events(limit=1)[0].target_id == harness.ana


# --- throttling -------------------------------------------------------------------------------


def test_mutations_share_the_per_user_mutation_budget(tmp_path) -> None:
    harness = Harness(tmp_path, mutation_limit=2)
    _install(harness)
    try:
        with TestClient(webhooks.app) as client:
            first = _authorize(harness, client, harness.ana)
            second = _authorize(harness, client, harness.ana)
            third = _authorize(harness, client, harness.ana)
            assert (first.status_code, second.status_code, third.status_code) == (200, 200, 429)
            assert "retry-after" in third.headers
            # Reads are not throttled by the mutation budget, and other users
            # have their own budget.
            assert (
                client.get("/users/me/connections", headers=harness.bearer(harness.ana)).status_code
                == 200
            )
            assert _authorize(harness, client, harness.bo).status_code == 200
    finally:
        _uninstall()


# --- several accounts from one provider, identity and bounded failures --------------------


def _grant(sub: str, label: str, access: str = ACCESS) -> TokenGrant:
    return TokenGrant(
        access_token=access,
        refresh_token=REFRESH,
        expires_in=3600,
        scopes=("openid", "email", "https://www.googleapis.com/auth/gmail.readonly"),
        provider_account_id=sub,
        account_label=label,
    )


def test_two_google_accounts_connect_side_by_side_and_reconnect_in_place(
    harness, client, caplog
) -> None:
    with caplog.at_level(logging.DEBUG):
        harness.exchanger.grant = _grant("sub-personal", "cesar.personal@gmail.com")
        personal = _connect(harness, client, harness.ana)
        harness.now += 10
        harness.exchanger.grant = _grant("sub-work", "cesar.work@gmail.com", access="ya29.WORK")
        work = _connect(harness, client, harness.ana)

    assert personal["connection_id"] != work["connection_id"]
    assert personal["account_label"] == "cesar.personal@gmail.com"
    assert work["account_label"] == "cesar.work@gmail.com"
    assert "provider_account_id" not in personal and "sub-personal" not in str(personal)

    listed = client.get("/users/me/connections", headers=harness.bearer(harness.ana)).json()
    assert [c["connection_id"] for c in listed["connections"]] == [
        personal["connection_id"],
        work["connection_id"],
    ]
    assert [c["provider"] for c in listed["connections"]] == ["google", "google"]
    assert "sub-personal" not in str(listed) and "sub-work" not in str(listed)

    # The personal account again: same connection, new tokens, work untouched.
    harness.now += 10
    harness.exchanger.grant = _grant("sub-personal", "cesar.personal@gmail.com", access="ya29.P2")
    again = _connect(harness, client, harness.ana)
    assert again["connection_id"] == personal["connection_id"]
    assert again["updated_at"] == harness.now
    store = harness.connections.store
    assert store.credentials(harness.ana, personal["connection_id"]).access_token == "ya29.P2"
    assert store.credentials(harness.ana, work["connection_id"]).access_token == "ya29.WORK"
    assert len(harness.rows("provider_connections")) == 2

    # Disconnecting one leaves the other.
    gone = client.delete(
        f"/users/me/connections/{work['connection_id']}", headers=harness.bearer(harness.ana)
    )
    assert gone.status_code == 204
    remaining = client.get("/users/me/connections", headers=harness.bearer(harness.ana)).json()
    assert [c["connection_id"] for c in remaining["connections"]] == [personal["connection_id"]]

    for hidden in (
        "cesar.personal@gmail.com",
        "cesar.work@gmail.com",
        "sub-personal",
        "sub-work",
        ACCESS,
        "ya29.WORK",
        REFRESH,
        CODE,
    ):
        assert hidden not in caplog.text, hidden


def test_the_same_provider_account_is_isolated_per_user(harness, client) -> None:
    harness.exchanger.grant = _grant("sub-shared", "shared@gmail.com")
    mine = _connect(harness, client, harness.ana)
    harness.exchanger.grant = _grant("sub-shared", "shared@gmail.com", access="ya29.BO")
    theirs = _connect(harness, client, harness.bo)

    assert mine["connection_id"] != theirs["connection_id"]
    store = harness.connections.store
    assert store.credentials(harness.ana, mine["connection_id"]).access_token == ACCESS
    assert store.credentials(harness.bo, theirs["connection_id"]).access_token == "ya29.BO"
    ana_list = client.get("/users/me/connections", headers=harness.bearer(harness.ana)).json()
    bo_list = client.get("/users/me/connections", headers=harness.bearer(harness.bo)).json()
    assert [c["connection_id"] for c in ana_list["connections"]] == [mine["connection_id"]]
    assert [c["connection_id"] for c in bo_list["connections"]] == [theirs["connection_id"]]
    assert (
        client.delete(
            f"/users/me/connections/{theirs['connection_id']}", headers=harness.bearer(harness.ana)
        ).status_code
        == 404
    )


def test_a_grant_without_a_resolved_identity_is_refused_and_nothing_is_stored(
    harness, client, caplog
) -> None:
    harness.exchanger.grant = TokenGrant(
        access_token=ACCESS, refresh_token=REFRESH, expires_in=3600, account_label="ana@gmail.com"
    )
    state = _state_of(_authorize(harness, client, harness.ana))
    with caplog.at_level(logging.DEBUG):
        response = _callback(harness, client, harness.ana, state)
    assert response.status_code == 502
    assert response.json() == {
        "detail": "token_exchange_failed",
        "reason": "identity_unavailable",
        "provider": "google",
        "settings": ["CAAL_OAUTH_GOOGLE_SCOPES"],
    }
    assert harness.rows("provider_connections") == []
    assert ACCESS not in caplog.text and "ana@gmail.com" not in caplog.text


def test_classified_exchange_failures_are_bounded_and_actionable(harness, client, caplog) -> None:
    harness.exchanger.error = TokenExchangeError(
        f"provider said scope {CODE} is bad sk-live-LEAK",
        reason="insufficient_scope",
        settings=("CAAL_OAUTH_GOOGLE_SCOPES",),
    )
    state = _state_of(_authorize(harness, client, harness.ana))
    with caplog.at_level(logging.DEBUG):
        response = _callback(harness, client, harness.ana, state)
    assert response.status_code == 502
    assert response.json() == {
        "detail": "token_exchange_failed",
        "reason": "insufficient_scope",
        "provider": "google",
        "settings": ["CAAL_OAUTH_GOOGLE_SCOPES"],
    }
    assert "LEAK" not in response.text and "LEAK" not in caplog.text and CODE not in caplog.text
    assert "insufficient_scope" in caplog.text  # the reason is loggable; the wording is not

    harness.exchanger.error = RuntimeError(f"socket exploded while sending {CODE}")
    state = _state_of(_authorize(harness, client, harness.ana))
    with caplog.at_level(logging.DEBUG):
        response = _callback(harness, client, harness.ana, state)
    assert response.status_code == 502
    assert response.json()["reason"] == "transport"
    assert CODE not in caplog.text and "exploded" not in caplog.text
    assert harness.rows("provider_connections") == []


ZOHO_ENV = {
    "CAAL_OAUTH_ZOHO_CLIENT_ID": "zoho-client-id",
    "CAAL_OAUTH_ZOHO_CLIENT_SECRET": "zoho-client-secret",
    ENV_PUBLIC_ORIGIN: ORIGIN,
}


def test_zoho_callbacks_from_another_data_center_are_refused_before_any_exchange(tmp_path) -> None:
    harness = Harness(tmp_path, env={**GOOGLE_ENV, **ZOHO_ENV})
    harness.exchanger.grant = _grant("zuid-1", "cesar@zohomail.com")
    _install(harness)
    try:
        with TestClient(webhooks.app) as client:
            # The configured accounts server is accounts.zoho.com; a callback
            # that says the user lives in the EU data center cannot be redeemed there.
            state = _state_of(_authorize(harness, client, harness.ana, "zoho"))
            response = _callback(harness, client, harness.ana, state, location="eu")
            assert response.status_code == 502
            assert response.json() == {
                "detail": "token_exchange_failed",
                "reason": "datacenter_mismatch",
                "provider": "zoho",
                "settings": ["CAAL_OAUTH_ZOHO_ACCOUNTS_DOMAIN"],
            }
            assert harness.exchanger.calls == []
            # The state is spent either way.
            replay = _callback(harness, client, harness.ana, state)
            assert replay.json()["detail"] == "invalid_state"

            # An unknown data center code is refused the same way.
            state = _state_of(_authorize(harness, client, harness.ana, "zoho"))
            unknown = _callback(harness, client, harness.ana, state, location="zz")
            assert unknown.json()["reason"] == "datacenter_mismatch"
            assert harness.exchanger.calls == []

            # The matching data center, or no location at all, proceeds.
            state = _state_of(_authorize(harness, client, harness.ana, "zoho"))
            ok = _callback(harness, client, harness.ana, state, location="us")
            assert ok.status_code == 200, ok.text
            state = _state_of(_authorize(harness, client, harness.ana, "zoho"))
            assert _callback(harness, client, harness.ana, state).status_code == 200
            assert len(harness.exchanger.calls) == 2

            # A location is only meaningful for Zoho; Google ignores it.
            harness.exchanger.grant = _grant("sub-ana", "ana@gmail.com")
            state = _state_of(_authorize(harness, client, harness.ana, "google"))
            assert _callback(harness, client, harness.ana, state, location="eu").status_code == 200

            # Malformed locations are refused at the edge.
            for bad in ("EU", "usa", "u", "e-u", "", 7):
                state = _state_of(_authorize(harness, client, harness.ana, "zoho"))
                refused = _callback(harness, client, harness.ana, state, location=bad)
                assert refused.status_code == 422, bad
    finally:
        _uninstall()


def test_production_runtime_gets_a_real_exchanger_only_when_a_provider_is_configured(
    tmp_path, monkeypatch
) -> None:
    from caal.oauth_exchange import HttpTokenExchanger
    from caal.oauth_providers import all_env_names

    harness = Harness(tmp_path)
    for name in all_env_names():
        monkeypatch.delenv(name, raising=False)
    connections_api.reset_connections_runtime()
    try:
        bare = connections_api.get_connections_runtime(harness.runtime)
        assert bare is not None and bare.exchanger is None
        assert bare.providers.configured_providers == ()
        assert bare.token_exchange_available is False

        connections_api.reset_connections_runtime()
        for name, value in GOOGLE_ENV.items():
            monkeypatch.setenv(name, value)
        wired = connections_api.get_connections_runtime(harness.runtime)
        assert isinstance(wired.exchanger, HttpTokenExchanger)
        assert wired.providers.configured_providers == ("google",)
        assert wired.token_exchange_available is True
        # Built once per identity runtime.
        assert connections_api.get_connections_runtime(harness.runtime) is wired
        assert connections_api.get_connections_runtime(None) is None
    finally:
        connections_api.reset_connections_runtime()


# --- naming one of your own accounts -------------------------------------------------


def _rename(harness, client, user_id: str, connection_id: str, **body):
    return client.patch(
        f"/users/me/connections/{connection_id}",
        headers=harness.bearer(user_id),
        json=body,
    )


def test_naming_a_connection_is_owner_only_and_never_touches_the_grant(
    harness, client, caplog
) -> None:
    mine = _connect(harness, client, harness.ana)
    assert mine["user_label"] is None and mine["aliases"] == []

    with caplog.at_level(logging.DEBUG):
        response = _rename(
            harness,
            client,
            harness.ana,
            mine["connection_id"],
            user_label="  Work  ",
            aliases=["Office", "office", " day job "],
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["user_label"] == "Work"
    assert body["aliases"] == ["Office", "day job"]
    # The provider decided this, and naming does not touch it.
    assert body["account_label"] == "ana@gmail.com"
    assert body["has_refresh_token"] is True
    assert ACCESS not in response.text and REFRESH not in response.text
    # What the user typed is never written to a log line.
    for hidden in ("Work", "Office", "day job", harness.ana):
        assert hidden not in caplog.text, hidden

    listed = client.get("/users/me/connections", headers=harness.bearer(harness.ana)).json()
    assert listed["connections"][0]["user_label"] == "Work"
    assert listed["connections"][0]["aliases"] == ["Office", "day job"]

    # Bo has nothing of his own here, and cannot rename Ana's connection.
    bo_list = client.get("/users/me/connections", headers=harness.bearer(harness.bo)).json()
    assert bo_list["connections"] == []
    denied = _rename(harness, client, harness.bo, mine["connection_id"], user_label="mine now")
    assert denied.status_code == 404 and denied.json()["detail"] == "not_found"
    still = client.get("/users/me/connections", headers=harness.bearer(harness.ana)).json()
    assert still["connections"][0]["user_label"] == "Work"


def test_naming_refuses_unknown_ids_bad_payloads_and_unauthenticated_callers(
    harness, client
) -> None:
    mine = _connect(harness, client, harness.ana)
    connection_id = mine["connection_id"]

    unknown = _rename(harness, client, harness.ana, "con_" + "f" * 24, user_label="ghost")
    assert unknown.status_code == 404
    assert _rename(harness, client, harness.ana, "not-an-id", user_label="ghost").status_code == 404

    for bad in (
        dict(user_label="x" * 200),
        dict(user_label="line\nbreak"),
        dict(user_label="-"),
        dict(aliases=["x" * 200]),
        dict(aliases=[f"n{index}" for index in range(9)]),
        dict(aliases="work"),
        dict(aliases=[7]),
        dict(account_label="ana@evil.example"),
    ):
        refused = _rename(harness, client, harness.ana, connection_id, **bad)
        assert refused.status_code == 422, (bad, refused.text)
        assert ACCESS not in refused.text

    # An empty body changes nothing rather than clearing anything by accident.
    empty = client.patch(
        f"/users/me/connections/{connection_id}",
        headers=harness.bearer(harness.ana),
        json=dict(),
    )
    assert empty.status_code == 422 and empty.json()["detail"] == "no_change"

    anonymous = client.patch(f"/users/me/connections/{connection_id}", json=dict(user_label="work"))
    assert anonymous.status_code == 401
    agent = client.patch(
        f"/users/me/connections/{connection_id}",
        headers=dict(
            Authorization="Bearer " + harness.principal(harness.ana, audience=AUDIENCE_AGENT)
        ),
        json=dict(user_label="work"),
    )
    assert agent.status_code in (401, 403)
    unchanged = client.get("/users/me/connections", headers=harness.bearer(harness.ana)).json()
    assert unchanged["connections"][0]["user_label"] is None


def test_naming_shares_the_per_user_mutation_budget(tmp_path) -> None:
    harness = Harness(tmp_path, mutation_limit=2)
    _install(harness)
    try:
        with TestClient(webhooks.app) as client:
            connection = _connect(harness, client, harness.ana)
            limited = _rename(
                harness, client, harness.ana, connection["connection_id"], user_label="work"
            )
            assert limited.status_code == 429
    finally:
        _uninstall()


def test_naming_a_revoked_connection_is_not_found(harness, client) -> None:
    connection = _connect(harness, client, harness.ana)
    removed = client.delete(
        f"/users/me/connections/{connection['connection_id']}",
        headers=harness.bearer(harness.ana),
    )
    assert removed.status_code == 204
    gone = _rename(harness, client, harness.ana, connection["connection_id"], user_label="work")
    assert gone.status_code == 404 and gone.json()["detail"] == "not_found"
