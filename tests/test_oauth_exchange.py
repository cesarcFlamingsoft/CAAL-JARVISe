"""The production token exchange: a bounded HTTPS transport plus identity resolution.

No test here reaches a network. The exchanger is driven through an injected
``httpx`` transport that plays the provider, records every request it saw,
and can misbehave on demand. Pinned properties:

* the authorization code is redeemed with a single POST of form fields to the
  provider's token endpoint, over https only, never following a redirect;
* the response is read within a byte budget and a wall-clock budget;
* who the account belongs to is resolved by one bounded call to the
  provider's own identity endpoint (Google userinfo, Microsoft Graph ``/me``,
  Zoho's accounts server) with the access token just issued, and the result
  is a stable account id plus a label. An ``id_token`` in the token response
  is never consulted: its signature is not verified here, so a forged one
  with perfect-looking claims must not be able to choose the account;
* a refusal, a timeout, an oversized or malformed body, or an unresolvable
  identity each become one :class:`TokenExchangeError` with a bounded reason;
* nothing a test can observe -- the error text, ``repr`` of the grant, the
  log -- ever contains the code, a token, the client secret or an email.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from caal.oauth_exchange import (
    CONNECT_TIMEOUT_SECONDS,
    MAX_REQUESTS_PER_EXCHANGE,
    MAX_RESPONSE_BYTES,
    TIMEOUT_SECONDS,
    HttpTokenExchanger,
)
from caal.oauth_providers import (
    EXCHANGE_REASONS,
    ProviderConfig,
    TokenExchangeError,
    load_provider_registry,
)

ORIGIN = "https://jarvis.example.com"
CLIENT_SECRET = "client-secret-PLAINTEXT-value"
ENV = {
    "CAAL_OAUTH_GOOGLE_CLIENT_ID": "google-client-id.apps.googleusercontent.com",
    "CAAL_OAUTH_GOOGLE_CLIENT_SECRET": CLIENT_SECRET,
    "CAAL_OAUTH_MICROSOFT_CLIENT_ID": "ms-client-id",
    "CAAL_OAUTH_MICROSOFT_CLIENT_SECRET": CLIENT_SECRET,
    "CAAL_OAUTH_ZOHO_CLIENT_ID": "zoho-client-id",
    "CAAL_OAUTH_ZOHO_CLIENT_SECRET": CLIENT_SECRET,
    "CAAL_PUBLIC_ORIGIN": ORIGIN,
}
CODE = "4/0AX4XfWh-AUTHORIZATION-CODE"
VERIFIER = "v" * 64
ACCESS = "ya29.ACCESS-TOKEN-PLAINTEXT"
REFRESH = "1//REFRESH-TOKEN-PLAINTEXT"
EMAIL = "cesar.personal@gmail.com"
NOW = 1_700_000_000
# 2100-01-01: an ID token with this expiry is unexpired by any real clock, so no
# expiry check can be what keeps it from being trusted.
FAR_FUTURE = 4_102_444_800
SECRETS = (CODE, VERIFIER, ACCESS, REFRESH, CLIENT_SECRET, EMAIL)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def jwt(claims: dict) -> str:
    header = _b64(json.dumps({"alg": "RS256", "typ": "JWT", "kid": "k1"}).encode())
    payload = _b64(json.dumps(claims).encode())
    return f"{header}.{payload}.{_b64(b'not-a-real-signature')}"


def google_id_token(**overrides) -> str:
    claims = {
        "iss": "https://accounts.google.com",
        "aud": ENV["CAAL_OAUTH_GOOGLE_CLIENT_ID"],
        "sub": "108234567890123456789",
        "email": EMAIL,
        "email_verified": True,
        "iat": NOW,
        "exp": FAR_FUTURE,
    }
    claims.update(overrides)
    return jwt(claims)


ATTACKER_ID = "999999999999999999999"
ATTACKER_OID = "00000000-0000-0000-0000-00000000dead"
ATTACKER_EMAIL = "attacker@example.net"
MICROSOFT_ISSUER = "https://login.microsoftonline.com/9188040d-6c67-4c5b-b112-36a304b66dad/v2.0"


def forged_id_token(config: ProviderConfig) -> str:
    """An ID token whose issuer, audience and expiry all look right for ``config``
    but whose signature is nobody's. Nothing in the exchanger can tell it from a
    real one, so the only safe thing is to never consult it."""
    issuer = MICROSOFT_ISSUER if config.provider == "microsoft" else "https://accounts.google.com"
    return jwt(
        {
            "iss": issuer,
            "aud": config.client_id,
            "sub": ATTACKER_ID,
            "oid": ATTACKER_OID,
            "email": ATTACKER_EMAIL,
            "preferred_username": ATTACKER_EMAIL,
            "email_verified": True,
            "iat": NOW,
            "exp": FAR_FUTURE,
        }
    )


class Provider:
    """A scripted provider behind ``httpx.MockTransport``."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.token_response: httpx.Response | Exception = httpx.Response(
            200,
            json={
                "access_token": ACCESS,
                "refresh_token": REFRESH,
                "expires_in": 3599,
                "token_type": "Bearer",
                "scope": "openid email https://www.googleapis.com/auth/gmail.readonly",
                "id_token": google_id_token(),
            },
        )
        self.identity_response: httpx.Response | Exception = httpx.Response(
            200, json={"sub": "108234567890123456789", "email": EMAIL}
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.token_response if request.method == "POST" else self.identity_response
        if isinstance(answer, Exception):
            raise answer
        return answer

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def form(self, index: int = 0) -> dict[str, list[str]]:
        return parse_qs(self.requests[index].content.decode("utf-8"), strict_parsing=True)


def run(coro):
    return asyncio.run(coro)


def exchange(provider: Provider, config: ProviderConfig, *, code_verifier=VERIFIER, **kw):
    exchanger = HttpTokenExchanger(transport=provider.transport, **kw)
    return run(
        exchanger.exchange(
            config, code=CODE, redirect_uri=config.redirect_uri, code_verifier=code_verifier
        )
    )


@pytest.fixture(autouse=True)
def capture_transport_logs(caplog):
    """Attach caplog to the transport loggers directly: ``voice_agent`` (imported
    elsewhere in the suite) stops ``caal`` records propagating to the root."""
    targets = [
        logging.getLogger(name)
        for name in ("caal.oauth_exchange", "caal.oauth_providers", "httpx", "httpcore")
    ]
    previous = [target.level for target in targets]
    for target in targets:
        target.setLevel(logging.DEBUG)
        target.addHandler(caplog.handler)
    try:
        yield
    finally:
        for target, level in zip(targets, previous):
            target.removeHandler(caplog.handler)
            target.setLevel(level)


@pytest.fixture
def registry():
    return load_provider_registry(ENV)


@pytest.fixture
def google(registry) -> ProviderConfig:
    return registry.get("google")


@pytest.fixture
def microsoft(registry) -> ProviderConfig:
    return registry.get("microsoft")


@pytest.fixture
def zoho(registry) -> ProviderConfig:
    return registry.get("zoho")


def assert_no_secret(text: str, *, extra: tuple[str, ...] = ()) -> None:
    for secret in (*SECRETS, *extra):
        assert secret not in text, secret


# --- the exchange request ----------------------------------------------------------------


def test_google_exchange_posts_the_code_once_then_asks_userinfo_who_the_account_is(
    google, caplog
) -> None:
    provider = Provider()  # its token response carries an ID token; that is not consulted
    with caplog.at_level(logging.DEBUG):
        grant = exchange(provider, google)

    token, identity = provider.requests
    assert token.method == "POST"
    assert str(token.url) == "https://oauth2.googleapis.com/token"
    assert token.headers["content-type"] == "application/x-www-form-urlencoded"
    assert "authorization" not in {name.lower() for name in token.headers}
    form = provider.form()
    assert form["grant_type"] == ["authorization_code"]
    assert form["code"] == [CODE]
    assert form["client_id"] == [google.client_id]
    assert form["client_secret"] == [CLIENT_SECRET]
    assert form["redirect_uri"] == [google.redirect_uri]
    assert form["code_verifier"] == [VERIFIER]

    assert identity.method == "GET"
    assert str(identity.url) == "https://openidconnect.googleapis.com/v1/userinfo"
    assert identity.headers["authorization"] == f"Bearer {ACCESS}"
    assert identity.content == b""

    assert grant.access_token == ACCESS
    assert grant.refresh_token == REFRESH
    assert grant.expires_in == 3599
    assert grant.scopes == (
        "openid",
        "email",
        "https://www.googleapis.com/auth/gmail.readonly",
    )
    assert grant.provider_account_id == "108234567890123456789"
    assert grant.account_label == EMAIL
    assert_no_secret(repr(grant) + str(grant), extra=(grant.provider_account_id,))
    assert_no_secret(caplog.text, extra=("id_token",))


def test_google_identity_is_resolved_the_same_way_when_no_id_token_is_returned(google) -> None:
    provider = Provider()
    provider.token_response = httpx.Response(
        200, json={"access_token": ACCESS, "token_type": "Bearer", "expires_in": 3600}
    )
    grant = exchange(provider, google)

    assert [request.method for request in provider.requests] == ["POST", "GET"]
    assert str(provider.requests[1].url) == "https://openidconnect.googleapis.com/v1/userinfo"
    assert grant.provider_account_id == "108234567890123456789"
    assert grant.account_label == EMAIL
    assert grant.refresh_token is None
    # No scope echoed by the provider: the configured request is what was granted.
    assert grant.scopes == google.scopes


def test_microsoft_exchange_uses_pkce_and_names_the_account_from_graph(microsoft) -> None:
    provider = Provider()
    provider.token_response = httpx.Response(
        200,
        json={
            "token_type": "Bearer",
            "scope": "openid email Mail.Read Calendars.Read",
            "expires_in": 4800,
            "access_token": ACCESS,
            "refresh_token": REFRESH,
            "id_token": forged_id_token(microsoft),
        },
    )
    provider.identity_response = httpx.Response(
        200,
        json={
            "@odata.context": "https://graph.microsoft.com/v1.0/$metadata#users/$entity",
            "id": "00000000-0000-0000-66f3-3332eca7ea81",
            "displayName": "Cesar C",
            "mail": "cesar@outlook.com",
            "userPrincipalName": "cesar_outlook.com#EXT#@contoso.onmicrosoft.com",
        },
    )
    grant = exchange(provider, microsoft)

    token, identity = provider.requests
    assert str(token.url) == "https://login.microsoftonline.com/common/oauth2/v2.0/token"
    assert provider.form()["code_verifier"] == [VERIFIER]
    assert str(identity.url) == "https://graph.microsoft.com/v1.0/me"
    assert identity.headers["authorization"] == f"Bearer {ACCESS}"
    assert grant.provider_account_id == "00000000-0000-0000-66f3-3332eca7ea81"
    assert grant.account_label == "cesar@outlook.com"
    assert grant.scopes == ("openid", "email", "Mail.Read", "Calendars.Read")
    assert grant.expires_in == 4800

    # Without a ``mail`` attribute Graph still names the account; the UPN labels it.
    provider = Provider()
    provider.token_response = httpx.Response(
        200, json={"access_token": ACCESS, "token_type": "Bearer", "expires_in": 3600}
    )
    provider.identity_response = httpx.Response(
        200,
        json={"id": "00000000-0000-0000-66f3-3332eca7ea81", "userPrincipalName": "c@contoso.com"},
    )
    grant = exchange(provider, microsoft)
    assert grant.provider_account_id == "00000000-0000-0000-66f3-3332eca7ea81"
    assert grant.account_label == "c@contoso.com"


def test_zoho_exchange_sends_no_verifier_and_resolves_identity_from_the_accounts_server(
    zoho, caplog
) -> None:
    provider = Provider()
    provider.token_response = httpx.Response(
        200,
        json={
            "access_token": ACCESS,
            "refresh_token": REFRESH,
            "api_domain": "https://www.zohoapis.com",
            "token_type": "Bearer",
            "expires_in": 3600,
        },
    )
    provider.identity_response = httpx.Response(
        200,
        json={
            "First_Name": "Cesar",
            "Email": "cesar@zohomail.com",
            "Last_Name": "C",
            "Display_Name": "Cesar C",
            "ZUID": 12345678,
        },
    )
    with caplog.at_level(logging.DEBUG):
        grant = exchange(provider, zoho, code_verifier=None)

    token, identity = provider.requests
    assert str(token.url) == "https://accounts.zoho.com/oauth/v2/token"
    assert "code_verifier" not in provider.form()
    assert provider.form()["client_secret"] == [CLIENT_SECRET]
    assert str(identity.url) == "https://accounts.zoho.com/oauth/user/info"
    assert identity.headers["authorization"] == f"Zoho-oauthtoken {ACCESS}"
    assert grant.provider_account_id == "12345678"
    assert grant.account_label == "cesar@zohomail.com"
    # Zoho does not echo scopes on this response; the configured request stands.
    assert grant.scopes == zoho.scopes
    assert_no_secret(caplog.text, extra=("cesar@zohomail.com",))


def test_zoho_identity_follows_an_operator_configured_accounts_domain() -> None:
    registry = load_provider_registry(
        {**ENV, "CAAL_OAUTH_ZOHO_ACCOUNTS_DOMAIN": "https://accounts.zoho.eu"}
    )
    provider = Provider()
    provider.token_response = httpx.Response(
        200, json={"access_token": ACCESS, "token_type": "Bearer", "expires_in": 3600}
    )
    provider.identity_response = httpx.Response(200, json={"ZUID": 7, "Email": "c@z.eu"})
    exchange(provider, registry.get("zoho"), code_verifier=None)
    assert [str(r.url) for r in provider.requests] == [
        "https://accounts.zoho.eu/oauth/v2/token",
        "https://accounts.zoho.eu/oauth/user/info",
    ]


# --- refusals and bounded reasons ------------------------------------------------------------


def _refused(provider: Provider, config: ProviderConfig, **kw) -> TokenExchangeError:
    with pytest.raises(TokenExchangeError) as info:
        exchange(provider, config, **kw)
    error = info.value
    assert error.reason in EXCHANGE_REASONS
    assert_no_secret(str(error) + repr(error))
    return error


def test_a_provider_refusal_is_a_bounded_error_without_the_providers_wording(google) -> None:
    provider = Provider()
    provider.token_response = httpx.Response(
        400,
        json={
            "error": "invalid_grant",
            "error_description": f"Bad Request: code {CODE} secret {CLIENT_SECRET} LEAK",
        },
    )
    error = _refused(provider, google)
    assert error.reason == "provider_refused"
    assert error.provider_code == "invalid_grant"
    assert "LEAK" not in str(error)
    assert CLIENT_SECRET not in error.provider_code
    assert len(provider.requests) == 1


def test_zoho_reports_errors_with_a_200_and_they_are_still_refusals(zoho) -> None:
    provider = Provider()
    provider.token_response = httpx.Response(200, json={"error": "invalid_code"})
    error = _refused(provider, zoho, code_verifier=None)
    assert error.reason == "provider_refused"
    assert len(provider.requests) == 1


def test_an_unsupported_scope_is_an_actionable_failure_naming_the_setting(zoho, google) -> None:
    provider = Provider()
    provider.token_response = httpx.Response(200, json={"error": "invalid_scope"})
    error = _refused(provider, zoho, code_verifier=None)
    assert error.reason == "insufficient_scope"
    assert error.settings == ("CAAL_OAUTH_ZOHO_SCOPES",)

    # A grant that echoes scopes without the identity scope, and whose account
    # the provider then will not name, is a scope problem for the operator --
    # not a vague identity failure.
    provider = Provider()
    provider.token_response = httpx.Response(
        200,
        json={
            "access_token": ACCESS,
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "https://www.googleapis.com/auth/gmail.readonly",
        },
    )
    provider.identity_response = httpx.Response(401, json={"error": "invalid_token"})
    error = _refused(provider, google)
    assert error.reason == "insufficient_scope"
    assert error.settings == ("CAAL_OAUTH_GOOGLE_SCOPES",)
    assert len(provider.requests) == 2

    # Google echoes the OpenID shorthand in its long form; that is not a
    # missing scope, and a resolved identity is never second-guessed.
    provider = Provider()
    provider.token_response = httpx.Response(
        200,
        json={
            "access_token": ACCESS,
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "openid https://www.googleapis.com/auth/userinfo.email",
        },
    )
    grant = exchange(provider, google)
    assert grant.provider_account_id == "108234567890123456789"
    assert grant.scopes == ("openid", "https://www.googleapis.com/auth/userinfo.email")


def test_an_identity_endpoint_that_does_not_name_the_account_is_a_refusal(zoho) -> None:
    base = {"access_token": ACCESS, "token_type": "Bearer", "expires_in": 3600}
    for answer in (
        httpx.Response(401, json={"error": "invalid_token"}),
        httpx.Response(200, json={"Email": "x@y.z"}),
        httpx.Response(200, json={"ZUID": "has space", "Email": "x@y.z"}),
        httpx.Response(200, content=b"<html>not json</html>"),
        httpx.Response(302, headers={"location": "https://evil.example/"}),
        httpx.ReadTimeout("read timed out"),
    ):
        provider = Provider()
        provider.token_response = httpx.Response(200, json=base)
        provider.identity_response = answer
        error = _refused(provider, zoho, code_verifier=None)
        assert error.reason == "identity_unavailable"
        assert error.settings == ("CAAL_OAUTH_ZOHO_SCOPES",)
        assert len(provider.requests) == 2


# --- an ID token never decides the account ---------------------------------------------------


GENUINE = {
    "google": (
        {"sub": "108234567890123456789", "email": EMAIL},
        "108234567890123456789",
        EMAIL,
    ),
    "microsoft": (
        {"id": "00000000-0000-0000-66f3-3332eca7ea81", "mail": "cesar@outlook.com"},
        "00000000-0000-0000-66f3-3332eca7ea81",
        "cesar@outlook.com",
    ),
    "zoho": (
        {"ZUID": 12345678, "Email": "cesar@zohomail.com"},
        "12345678",
        "cesar@zohomail.com",
    ),
}


@pytest.mark.parametrize("name", ["google", "microsoft", "zoho"])
def test_an_id_token_in_the_token_response_never_decides_the_account(
    registry, name, caplog
) -> None:
    """The exchanger has no key to check an ID token's signature, so a token
    endpoint answer carrying one -- forged or genuine -- must not be able to
    say which account was linked. Only the provider's identity endpoint,
    asked with the access token just issued, can."""
    config = registry.get(name)
    verifier = VERIFIER if config.pkce else None
    identity_json, genuine_id, genuine_label = GENUINE[name]
    base = {"access_token": ACCESS, "token_type": "Bearer", "expires_in": 3600}

    # A perfectly-shaped token naming an attacker: the provider's answer wins.
    for id_token in (forged_id_token(config), "not.a.jwt.at.all", 12345):
        provider = Provider()
        provider.token_response = httpx.Response(200, json={**base, "id_token": id_token})
        provider.identity_response = httpx.Response(200, json=identity_json)
        with caplog.at_level(logging.DEBUG):
            grant = exchange(provider, config, code_verifier=verifier)

        token, identity = provider.requests
        assert identity.method == "GET"
        assert str(identity.url) == config.identity_endpoint
        assert identity.headers["authorization"].split(" ", 1)[1] == ACCESS
        assert grant.provider_account_id == genuine_id
        assert grant.account_label == genuine_label
        assert grant.provider_account_id != ATTACKER_ID
        assert grant.provider_account_id != ATTACKER_OID
        assert grant.account_label != ATTACKER_EMAIL
        assert_no_secret(caplog.text, extra=("id_token", ATTACKER_ID, ATTACKER_OID, ATTACKER_EMAIL))

    # When the provider will not name the account, the token cannot stand in.
    for answer in (
        httpx.Response(401, json={"error": "invalid_token"}),
        httpx.Response(200, json={}),
        httpx.ConnectError("connection refused"),
    ):
        provider = Provider()
        provider.token_response = httpx.Response(
            200, json={**base, "id_token": forged_id_token(config)}
        )
        provider.identity_response = answer
        error = _refused(provider, config, code_verifier=verifier)
        assert error.reason == "identity_unavailable"
        assert error.settings == (config.env_scopes,)
        assert len(provider.requests) == 2
        assert_no_secret(str(error) + repr(error), extra=(ATTACKER_ID, ATTACKER_EMAIL))


def test_the_exchanger_has_no_id_token_machinery_left() -> None:
    """Nothing to trust, nothing to get wrong: no claim parsing, no clock."""
    from caal import oauth_exchange

    assert not any("id_token" in name.lower() for name in dir(HttpTokenExchanger))
    assert "clock" not in HttpTokenExchanger.__init__.__code__.co_varnames
    with pytest.raises(TypeError):
        HttpTokenExchanger(clock=lambda: NOW)  # type: ignore[call-arg]
    assert "base64" not in vars(oauth_exchange)
    assert not any("ISSUER" in name for name in vars(oauth_exchange))


def test_malformed_token_responses_are_refused(google) -> None:
    for answer in (
        httpx.Response(200, content=b"<html>login</html>"),
        httpx.Response(200, json=["not", "an", "object"]),
        httpx.Response(200, json={"token_type": "Bearer"}),  # no access_token
        httpx.Response(200, json={"access_token": "", "token_type": "Bearer"}),
        httpx.Response(200, json={"access_token": "x\ny", "token_type": "Bearer"}),
        httpx.Response(200, json={"access_token": ACCESS, "token_type": "MAC"}),
        httpx.Response(302, headers={"location": "https://evil.example/"}),
        httpx.Response(500, content=b"oops"),
    ):
        provider = Provider()
        provider.token_response = answer
        error = _refused(provider, google)
        assert error.reason in ("malformed_response", "provider_refused"), answer.status_code
        assert len(provider.requests) == 1, "a redirect or error must never be followed"


# --- transport bounds ------------------------------------------------------------------------


def test_transport_budgets_are_fixed_and_small() -> None:
    assert 0 < CONNECT_TIMEOUT_SECONDS <= 10
    assert CONNECT_TIMEOUT_SECONDS <= TIMEOUT_SECONDS <= 30
    assert 1024 <= MAX_RESPONSE_BYTES <= 256 * 1024
    assert MAX_REQUESTS_PER_EXCHANGE == 2
    for bad in (0, -1, 31, 3600):
        with pytest.raises(ValueError):
            HttpTokenExchanger(timeout_seconds=bad)
    for bad in (0, 512, 10 * 1024 * 1024):
        with pytest.raises(ValueError):
            HttpTokenExchanger(max_response_bytes=bad)
    exchanger = HttpTokenExchanger()
    assert exchanger.timeout_seconds == TIMEOUT_SECONDS
    assert exchanger.max_response_bytes == MAX_RESPONSE_BYTES


def test_a_timeout_or_transport_failure_is_reported_as_such(google) -> None:
    for failure in (
        httpx.ReadTimeout("read timed out"),
        httpx.ConnectTimeout("connect timed out"),
        httpx.ConnectError("connection refused"),
        httpx.RemoteProtocolError("server disconnected"),
    ):
        provider = Provider()
        provider.token_response = failure
        error = _refused(provider, google)
        assert error.reason == "transport", type(failure).__name__


def test_an_oversized_body_is_cut_off_and_refused(google) -> None:
    provider = Provider()
    padding = "p" * (MAX_RESPONSE_BYTES + 1)
    provider.token_response = httpx.Response(
        200, json={"access_token": ACCESS, "token_type": "Bearer", "padding": padding}
    )
    error = _refused(provider, google)
    assert error.reason == "malformed_response"

    provider = Provider()
    provider.token_response = httpx.Response(
        200,
        headers={"content-length": str(MAX_RESPONSE_BYTES * 4)},
        json={"access_token": ACCESS, "token_type": "Bearer"},
    )
    error = _refused(provider, google)
    assert error.reason == "malformed_response"


def test_only_https_endpoints_are_ever_contacted(google) -> None:
    provider = Provider()
    plain = ProviderConfig(
        provider="google",
        display_name="Google",
        client_id=google.client_id,
        client_secret=google.client_secret,
        scopes=google.scopes,
        identity_scopes=google.identity_scopes,
        authorization_endpoint=google.authorization_endpoint,
        token_endpoint="http://oauth2.googleapis.com/token",
        identity_endpoint=google.identity_endpoint,
        redirect_uri=google.redirect_uri,
        pkce=True,
    )
    error = _refused(provider, plain)
    assert error.reason == "transport"
    assert provider.requests == []

    # The identity call is held to the same rule, and no ID token can excuse it.
    provider = Provider()
    provider.token_response = httpx.Response(
        200,
        json={
            "access_token": ACCESS,
            "token_type": "Bearer",
            "expires_in": 3600,
            "id_token": google_id_token(),
        },
    )
    plain_identity = ProviderConfig(
        provider="google",
        display_name="Google",
        client_id=google.client_id,
        client_secret=google.client_secret,
        scopes=google.scopes,
        identity_scopes=google.identity_scopes,
        authorization_endpoint=google.authorization_endpoint,
        token_endpoint=google.token_endpoint,
        identity_endpoint="http://openidconnect.googleapis.com/v1/userinfo",
        redirect_uri=google.redirect_uri,
        pkce=True,
    )
    error = _refused(provider, plain_identity)
    assert error.reason == "identity_unavailable"
    assert len(provider.requests) == 1


def test_the_exchange_never_makes_more_than_two_requests(google, zoho) -> None:
    for config, verifier in ((google, VERIFIER), (zoho, None)):
        provider = Provider()
        provider.token_response = httpx.Response(
            200, json={"access_token": ACCESS, "token_type": "Bearer", "expires_in": 3600}
        )
        provider.identity_response = httpx.Response(
            200, json={"sub": "1", "email": "a@b.c", "ZUID": 1, "Email": "a@b.c"}
        )
        exchange(provider, config, code_verifier=verifier)
        assert len(provider.requests) <= MAX_REQUESTS_PER_EXCHANGE
        assert all(urlsplit(str(r.url)).scheme == "https" for r in provider.requests)


def test_inputs_are_validated_before_anything_is_sent(google) -> None:
    provider = Provider()
    exchanger = HttpTokenExchanger(transport=provider.transport)
    for code in ("", "has space", "x" * 5000, None):
        with pytest.raises((TokenExchangeError, ValueError)):
            run(
                exchanger.exchange(
                    google, code=code, redirect_uri=google.redirect_uri, code_verifier=VERIFIER
                )
            )
    with pytest.raises((TokenExchangeError, ValueError)):
        run(
            exchanger.exchange(
                google, code=CODE, redirect_uri="https://evil.example/cb", code_verifier=VERIFIER
            )
        )
    assert provider.requests == []
