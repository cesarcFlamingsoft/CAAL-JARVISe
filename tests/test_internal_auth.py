"""Short-lived signed principals between the Next.js BFF and the CAAL backend.

The BFF is the only component that sees a Cloudflare Access assertion. What it
passes downstream is a compact, HMAC-signed principal naming an opaque user id
and nothing else. These tests pin the verification rules: fixed algorithm,
required claims, issuer and audience binding, bounded lifetime, single-use
nonces for HTTP calls, constant-time secret comparison, and a secret strength
floor so a weak or missing secret can never mint anything.
"""

from __future__ import annotations

import json
import time

import jwt
import pytest

from caal import internal_auth
from caal.internal_auth import (
    AUDIENCE_AGENT,
    AUDIENCE_BACKEND,
    AUDIENCE_IDENTITY,
    ISSUER_BFF,
    MAX_TTL_SECONDS,
    MIN_SECRET_LENGTH,
    InMemoryNonceStore,
    PrincipalError,
    RateLimiter,
    SqliteNonceStore,
    mint_principal,
    verify_principal,
)

SECRET = "s" * 48
OTHER_SECRET = "t" * 48
# A fixed instant in the past, so tokens minted for it also satisfy PyJWT's own
# wall-clock checks when a test decodes them directly.
NOW = 1_700_000_000


def _mint(**overrides):
    params = {
        "secret": SECRET,
        "subject": "usr_0123456789abcdef01234567",
        "audience": AUDIENCE_BACKEND,
        "now": NOW,
    }
    params.update(overrides)
    return mint_principal(**params)


def test_mint_and_verify_round_trip_carries_only_opaque_claims() -> None:
    token = _mint(claims={"role": "member"})

    principal = verify_principal(token, secret=SECRET, audience=AUDIENCE_BACKEND, now=NOW + 5)

    assert principal.subject == "usr_0123456789abcdef01234567"
    assert principal.audience == AUDIENCE_BACKEND
    assert principal.issuer == ISSUER_BFF
    assert principal.claims == {"role": "member"}
    assert principal.expires_at - principal.issued_at == 60
    assert len(principal.jti) >= 16
    assert token not in repr(principal)


def test_token_is_a_standard_hs256_jwt_the_bff_can_produce_with_jose() -> None:
    token = _mint()

    header = jwt.get_unverified_header(token)
    # Wire-format check only: the fixed instant is in the past, so skip expiry.
    payload = jwt.decode(
        token,
        SECRET,
        algorithms=["HS256"],
        audience=AUDIENCE_BACKEND,
        options={"verify_exp": False},
    )

    assert header == {"alg": "HS256", "typ": "JWT"}
    assert payload["iss"] == ISSUER_BFF
    assert payload["aud"] == AUDIENCE_BACKEND
    assert payload["sub"] == "usr_0123456789abcdef01234567"
    assert payload["iat"] == NOW
    assert payload["exp"] == NOW + 60
    assert isinstance(payload["jti"], str)


@pytest.mark.parametrize(
    "mutation",
    [
        {"secret": OTHER_SECRET},
        {"audience": AUDIENCE_AGENT},
        {"issuer": "someone-else"},
    ],
)
def test_wrong_secret_audience_or_issuer_is_rejected(mutation: dict) -> None:
    token = _mint(**mutation)

    with pytest.raises(PrincipalError):
        verify_principal(token, secret=SECRET, audience=AUDIENCE_BACKEND, now=NOW + 1)


def test_expired_and_not_yet_valid_tokens_are_rejected_with_small_leeway() -> None:
    token = _mint(ttl_seconds=30)

    assert verify_principal(token, secret=SECRET, audience=AUDIENCE_BACKEND, now=NOW + 33)
    with pytest.raises(PrincipalError):
        verify_principal(token, secret=SECRET, audience=AUDIENCE_BACKEND, now=NOW + 40)
    with pytest.raises(PrincipalError):
        verify_principal(token, secret=SECRET, audience=AUDIENCE_BACKEND, now=NOW - 30)


def test_lifetime_is_bounded_even_if_the_issuer_asks_for_more() -> None:
    with pytest.raises(PrincipalError):
        _mint(ttl_seconds=MAX_TTL_SECONDS + 1)
    with pytest.raises(PrincipalError):
        _mint(ttl_seconds=0)

    # A token forged with a long lifetime by a holder of the secret is still refused.
    long_lived = jwt.encode(
        {
            "iss": ISSUER_BFF,
            "aud": AUDIENCE_BACKEND,
            "sub": "usr_x",
            "iat": NOW,
            "exp": NOW + MAX_TTL_SECONDS * 10,
            "jti": "j" * 22,
        },
        SECRET,
        algorithm="HS256",
    )
    with pytest.raises(PrincipalError):
        verify_principal(long_lived, secret=SECRET, audience=AUDIENCE_BACKEND, now=NOW + 1)


@pytest.mark.parametrize("missing", ["sub", "jti", "iat", "exp", "iss", "aud"])
def test_every_required_claim_must_be_present(missing: str) -> None:
    payload = {
        "iss": ISSUER_BFF,
        "aud": AUDIENCE_BACKEND,
        "sub": "usr_x",
        "iat": NOW,
        "exp": NOW + 60,
        "jti": "j" * 22,
    }
    del payload[missing]
    token = jwt.encode(payload, SECRET, algorithm="HS256")

    with pytest.raises(PrincipalError):
        verify_principal(token, secret=SECRET, audience=AUDIENCE_BACKEND, now=NOW + 1)


def test_algorithm_confusion_and_unsigned_tokens_are_rejected() -> None:
    payload = {
        "iss": ISSUER_BFF,
        "aud": AUDIENCE_BACKEND,
        "sub": "usr_x",
        "iat": NOW,
        "exp": NOW + 60,
        "jti": "j" * 22,
    }
    unsigned = jwt.encode(payload, None, algorithm="none")
    hs512 = jwt.encode(payload, SECRET, algorithm="HS512")

    for token in (unsigned, hs512, "", "not.a.jwt", "a.b"):
        with pytest.raises(PrincipalError):
            verify_principal(token, secret=SECRET, audience=AUDIENCE_BACKEND, now=NOW + 1)


def test_weak_or_missing_secrets_can_neither_mint_nor_verify() -> None:
    token = _mint()
    for weak in ("", "short", "x" * (MIN_SECRET_LENGTH - 1), None):
        with pytest.raises(PrincipalError):
            mint_principal(secret=weak, subject="usr_x", audience=AUDIENCE_BACKEND, now=NOW)
        with pytest.raises(PrincipalError):
            verify_principal(token, secret=weak, audience=AUDIENCE_BACKEND, now=NOW + 1)


def test_subject_and_audience_are_validated_at_mint_time() -> None:
    with pytest.raises(PrincipalError):
        _mint(subject="")
    with pytest.raises(PrincipalError):
        _mint(subject="x" * 300)
    with pytest.raises(PrincipalError):
        _mint(audience="")
    with pytest.raises(PrincipalError):
        _mint(claims={"sub": "override"})  # reserved claims cannot be smuggled


def test_nonce_store_makes_http_principals_single_use() -> None:
    store = InMemoryNonceStore()
    token = _mint()

    verify_principal(
        token, secret=SECRET, audience=AUDIENCE_BACKEND, now=NOW + 1, nonce_store=store
    )
    with pytest.raises(PrincipalError):
        verify_principal(
            token, secret=SECRET, audience=AUDIENCE_BACKEND, now=NOW + 2, nonce_store=store
        )
    # A different token from the same issuer is fine.
    verify_principal(
        _mint(), secret=SECRET, audience=AUDIENCE_BACKEND, now=NOW + 2, nonce_store=store
    )


def test_sqlite_nonce_store_is_shared_and_forgets_expired_entries(tmp_path) -> None:
    path = tmp_path / "assistant.sqlite3"
    store = SqliteNonceStore(path)
    sibling = SqliteNonceStore(path)  # another process on the same file

    assert store.consume("jti-1", expires_at=NOW + 60, now=NOW) is True
    assert sibling.consume("jti-1", expires_at=NOW + 60, now=NOW + 1) is False
    # Once the token itself could no longer verify, its nonce need not be remembered.
    assert store.consume("jti-1", expires_at=NOW + 60, now=NOW + 3600) is True
    with pytest.raises(ValueError):
        store.consume("", expires_at=NOW + 60, now=NOW)


def test_agent_audience_tokens_bind_to_the_room_they_were_minted_for() -> None:
    token = _mint(audience=AUDIENCE_AGENT, ttl_seconds=300, claims={"room": "caal-web-abc"})

    principal = verify_principal(
        token, secret=SECRET, audience=AUDIENCE_AGENT, now=NOW + 10, room="caal-web-abc"
    )

    assert principal.claims["room"] == "caal-web-abc"
    with pytest.raises(PrincipalError):
        verify_principal(
            token, secret=SECRET, audience=AUDIENCE_AGENT, now=NOW + 10, room="another-room"
        )
    # A token without a room claim can never satisfy a room-bound check.
    with pytest.raises(PrincipalError):
        verify_principal(
            _mint(audience=AUDIENCE_AGENT),
            secret=SECRET,
            audience=AUDIENCE_AGENT,
            now=NOW + 10,
            room="caal-web-abc",
        )


def test_identity_assertions_carry_an_email_only_for_resolution() -> None:
    token = _mint(
        audience=AUDIENCE_IDENTITY, subject="identity", claims={"email": "Cesar@Example.com"}
    )

    principal = verify_principal(token, secret=SECRET, audience=AUDIENCE_IDENTITY, now=NOW + 1)

    assert principal.claims["email"] == "Cesar@Example.com"
    assert "Cesar" not in repr(principal)


def test_verify_uses_the_real_clock_by_default() -> None:
    token = mint_principal(secret=SECRET, subject="usr_x", audience=AUDIENCE_BACKEND)

    principal = verify_principal(token, secret=SECRET, audience=AUDIENCE_BACKEND)

    assert abs(principal.issued_at - time.time()) < 5


def test_rate_limiter_allows_a_burst_then_refuses_until_the_window_moves() -> None:
    limiter = RateLimiter(limit=3, window_seconds=60)

    assert [limiter.allow("ip-1", now=NOW + i) for i in range(4)] == [True, True, True, False]
    assert limiter.allow("ip-2", now=NOW) is True
    assert limiter.allow("ip-1", now=NOW + 61) is True
    assert limiter.retry_after("ip-1", now=NOW + 61) >= 0


def test_rate_limiter_bounds_its_memory() -> None:
    limiter = RateLimiter(limit=1, window_seconds=60, max_keys=10)
    for index in range(50):
        limiter.allow(f"key-{index}", now=NOW)
    assert limiter.tracked_keys <= 10


def test_module_exposes_no_secret_defaults() -> None:
    source = json.dumps(
        {name: getattr(internal_auth, name) for name in dir(internal_auth) if name.isupper()},
        default=str,
    )
    assert SECRET not in source
    assert "devkey" not in source and "secret" not in source.lower().replace("min_secret", "")
