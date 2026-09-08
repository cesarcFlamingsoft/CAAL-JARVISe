"""Cryptographic verification of Cloudflare Access application tokens.

The public portal sits behind Cloudflare Access, which attaches a signed JWT
(``Cf-Access-Jwt-Assertion``) to every authenticated request. A bare email
header is worthless as identity; only a token that verifies against the team's
published keys, for this application's audience, is. These tests use a
throwaway RSA key and an injected JWKS fetcher so the verifier's behaviour is
pinned without any network access: fixed algorithm, issuer and audience
binding, lifetime checks, key-id rotation with bounded refreshes, and refusal
of service tokens and meta tokens that carry no verified email.
"""

from __future__ import annotations

import json

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from caal.access_jwt import (
    AccessConfig,
    AccessConfigError,
    AccessTokenError,
    AccessVerifier,
    normalize_email,
)

TEAM = "https://example-team.cloudflareaccess.com"
AUD = "d1d09a2c79e964918d59077b9bb5b3a7b67ff76a04a80822eb8a3ce5f46354ac"
NOW = 1_800_000_000


def _keypair(kid: str):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
    jwk.update({"kid": kid, "alg": "RS256", "use": "sig"})
    return private, jwk


PRIVATE_A, JWK_A = _keypair("kid-a")
PRIVATE_B, JWK_B = _keypair("kid-b")


class FakeJwks:
    """Injected JWKS source: counts fetches and can rotate or fail on demand."""

    def __init__(self, *jwks) -> None:
        self.keys = list(jwks)
        self.calls = 0
        self.fail = False

    def __call__(self, url: str) -> dict:
        self.calls += 1
        assert url == f"{TEAM}/cdn-cgi/access/certs"
        if self.fail:
            raise ConnectionError("jwks unavailable")
        return {"keys": list(self.keys), "public_cert": {}, "public_certs": []}


def _token(private=PRIVATE_A, kid="kid-a", *, now=NOW, ttl=600, algorithm="RS256", **claims):
    payload = {
        "iss": TEAM,
        "aud": [AUD],
        "sub": "8f1a5b1e-0c1c-4b4b-9d70-0a0b0c0d0e0f",
        "email": "Cesar@MexcanTech.com",
        "type": "app",
        "iat": now,
        "nbf": now,
        "exp": now + ttl,
        "identity_nonce": "abc123",
        "country": "CA",
    }
    payload.update(claims)
    payload = {key: value for key, value in payload.items() if value is not None}
    return jwt.encode(payload, private, algorithm=algorithm, headers={"kid": kid})


def _hand_rolled_hs256(payload: dict, key: bytes, *, kid: str) -> str:
    import base64
    import hashlib
    import hmac

    def segment(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    header = segment(json.dumps({"alg": "HS256", "typ": "JWT", "kid": kid}).encode())
    body = segment(json.dumps(payload).encode())
    signature = hmac.new(key, f"{header}.{body}".encode(), hashlib.sha256).digest()
    return f"{header}.{body}.{segment(signature)}"


def _verifier(jwks: FakeJwks, **config) -> AccessVerifier:
    return AccessVerifier(
        AccessConfig(team_domain=TEAM, audience=AUD, **config), fetch_jwks=jwks, clock=lambda: NOW
    )


def test_valid_application_token_yields_a_normalized_identity() -> None:
    jwks = FakeJwks(JWK_A)
    verifier = _verifier(jwks)

    identity = verifier.verify(_token())

    assert identity.email == "cesar@mexcantech.com"
    assert identity.subject == "8f1a5b1e-0c1c-4b4b-9d70-0a0b0c0d0e0f"
    assert identity.expires_at == NOW + 600
    assert identity.issued_at == NOW
    assert "mexcantech" not in repr(identity)
    assert jwks.calls == 1


@pytest.mark.parametrize(
    "claims",
    [
        {"iss": "https://other-team.cloudflareaccess.com"},
        {"iss": None},
        {"aud": ["0" * 64]},
        {"aud": None},
        {"aud": [AUD, "0" * 64], "iss": "https://evil.example"},
        {"exp": NOW - 120},  # beyond the clock-skew leeway
        {"nbf": NOW + 300},
        {"iat": NOW + 300},
        {"email": None},
        {"email": ""},
        {"email": "not-an-email"},
        {"email": 42},
        {"type": "meta"},
        {"sub": None},
    ],
)
def test_claim_violations_are_rejected(claims: dict) -> None:
    verifier = _verifier(FakeJwks(JWK_A))

    with pytest.raises(AccessTokenError):
        verifier.verify(_token(**claims))


def test_audience_may_be_a_bare_string_and_extra_audiences_are_tolerated() -> None:
    verifier = _verifier(FakeJwks(JWK_A))

    assert verifier.verify(_token(aud=AUD)).email == "cesar@mexcantech.com"
    assert verifier.verify(_token(aud=[AUD, "1" * 64])).email == "cesar@mexcantech.com"


def test_service_tokens_without_an_email_are_not_user_identities() -> None:
    verifier = _verifier(FakeJwks(JWK_A))

    with pytest.raises(AccessTokenError):
        verifier.verify(_token(email=None, common_name="ci-service-token"))


def test_signature_key_and_algorithm_are_pinned() -> None:
    verifier = _verifier(FakeJwks(JWK_A))
    payload = {
        "iss": TEAM,
        "aud": [AUD],
        "sub": "s",
        "email": "a@b.co",
        "type": "app",
        "iat": NOW,
        "exp": NOW + 60,
    }
    public_pem = PRIVATE_A.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    # PyJWT refuses to HMAC with a public key, so build the classic
    # algorithm-confusion token by hand: HS256 keyed with the RSA public key.
    confused = _hand_rolled_hs256(payload, public_pem, kid="kid-a")
    unsigned = jwt.encode(payload, None, algorithm="none", headers={"kid": "kid-a"})
    other_key = _token(PRIVATE_B, "kid-a")  # right kid, wrong key
    no_kid = jwt.encode(payload, PRIVATE_A, algorithm="RS256")

    for bad in (confused, unsigned, other_key, no_kid, "", "a.b.c", "x" * 5000):
        with pytest.raises(AccessTokenError):
            verifier.verify(bad)


def test_unknown_kid_triggers_one_refresh_and_rotation_is_picked_up() -> None:
    jwks = FakeJwks(JWK_A)
    verifier = _verifier(jwks, refresh_cooldown_seconds=0)
    verifier.verify(_token())
    assert jwks.calls == 1

    with pytest.raises(AccessTokenError):
        verifier.verify(_token(PRIVATE_B, "kid-b"))
    assert jwks.calls == 2  # refreshed once, still unknown

    jwks.keys = [JWK_A, JWK_B]
    assert verifier.verify(_token(PRIVATE_B, "kid-b")).email == "cesar@mexcantech.com"
    assert jwks.calls == 3
    # Both keys are now cached; no further fetches for either.
    verifier.verify(_token())
    verifier.verify(_token(PRIVATE_B, "kid-b"))
    assert jwks.calls == 3


def test_refreshes_for_unknown_kids_are_rate_limited() -> None:
    jwks = FakeJwks(JWK_A)
    verifier = _verifier(jwks, refresh_cooldown_seconds=60)
    verifier.verify(_token())

    for _ in range(5):
        with pytest.raises(AccessTokenError):
            verifier.verify(_token(PRIVATE_B, "kid-b"))

    assert jwks.calls == 2  # exactly one refresh inside the cooldown window


def test_cached_keys_expire_after_the_configured_ttl() -> None:
    jwks = FakeJwks(JWK_A)
    clock = {"now": NOW}
    verifier = AccessVerifier(
        AccessConfig(team_domain=TEAM, audience=AUD, jwks_ttl_seconds=100),
        fetch_jwks=jwks,
        clock=lambda: clock["now"],
    )
    verifier.verify(_token())
    clock["now"] = NOW + 50
    verifier.verify(_token(now=NOW + 50))
    assert jwks.calls == 1
    clock["now"] = NOW + 150
    verifier.verify(_token(now=NOW + 150))
    assert jwks.calls == 2


def test_jwks_fetch_failure_fails_closed_but_keeps_serving_cached_keys() -> None:
    jwks = FakeJwks(JWK_A)
    verifier = _verifier(jwks, refresh_cooldown_seconds=0)
    jwks.fail = True
    with pytest.raises(AccessTokenError):
        verifier.verify(_token())

    jwks.fail = False
    verifier.verify(_token())
    jwks.fail = True
    # A cached key keeps working while the origin is unreachable...
    verifier.verify(_token())
    # ...but an unknown key is still refused rather than trusted.
    with pytest.raises(AccessTokenError):
        verifier.verify(_token(PRIVATE_B, "kid-b"))


def test_malformed_jwks_documents_are_ignored_safely() -> None:
    class Broken:
        def __call__(self, url: str) -> dict:
            return {"keys": [{"kty": "RSA", "kid": "kid-a", "n": "!!", "e": "AQAB"}, "junk"]}

    verifier = AccessVerifier(
        AccessConfig(team_domain=TEAM, audience=AUD), fetch_jwks=Broken(), clock=lambda: NOW
    )
    with pytest.raises(AccessTokenError):
        verifier.verify(_token())


@pytest.mark.parametrize(
    "team_domain, audience",
    [
        ("http://example-team.cloudflareaccess.com", AUD),  # must be https
        ("https://example-team.cloudflareaccess.com/", AUD),  # no trailing slash
        ("https://example.com", AUD),  # not a Cloudflare Access team domain
        ("example-team.cloudflareaccess.com", AUD),
        ("", AUD),
        (TEAM, ""),
        (TEAM, "not-hex"),
        (TEAM, "abc"),
        (TEAM, "A" * 64),  # AUD tags are lower-case hex
    ],
)
def test_config_rejects_unsafe_values(team_domain: str, audience: str) -> None:
    with pytest.raises(AccessConfigError):
        AccessConfig(team_domain=team_domain, audience=audience)


def test_config_exposes_the_jwks_url_and_issuer() -> None:
    config = AccessConfig(team_domain=TEAM, audience=AUD)

    assert config.jwks_url == f"{TEAM}/cdn-cgi/access/certs"
    assert config.issuer == TEAM


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Cesar@MexcanTech.com", "cesar@mexcantech.com"),
        ("  cesar@mexcantech.com \n", "cesar@mexcantech.com"),
    ],
)
def test_email_normalization(raw: str, expected: str) -> None:
    assert normalize_email(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "cesar", "cesar@", "@mexcantech.com", "a@b", "a b@c.com", "a@b.com\x00", "a" * 300],
)
def test_email_normalization_rejects_unusable_values(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_email(raw)


def test_verifier_uses_the_shared_keyset_by_default_without_touching_the_network() -> None:
    verifier = AccessVerifier(AccessConfig(team_domain=TEAM, audience=AUD))

    assert verifier.config.audience == AUD
    assert verifier.cached_key_ids == ()
