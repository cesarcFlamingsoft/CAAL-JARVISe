"""Authenticated, versioned encryption for sensitive profile fields.

The approved callback number is the only phone number JARVIS ever dials for a
user, so it is stored encrypted at rest under an operator-provided key that
never lives in the repository. These tests pin the properties the rest of the
system relies on: AES-GCM with versioned key material from the environment,
ciphertext bound to the owning row through associated data, a keyed blind
index so a caller-id lookup never needs plaintext in the database, and
fail-closed behaviour for anything malformed.
"""

from __future__ import annotations

import base64
import os

import pytest

from caal import profile_crypto
from caal.profile_crypto import DecryptionError, KeyRing, KeyRingError


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


KEY_V1 = _b64(bytes(range(32)))
KEY_V2 = _b64(bytes(range(32, 64)))


def test_keyring_parses_versioned_keys_and_picks_the_highest_version_as_active() -> None:
    ring = KeyRing.from_env(f"v1:{KEY_V1},v2:{KEY_V2}")

    assert ring.active_version == 2
    assert ring.versions == (1, 2)


def test_keyring_accepts_a_single_key_with_surrounding_whitespace() -> None:
    ring = KeyRing.from_env(f"  v3:{KEY_V1} ")

    assert ring.active_version == 3
    assert ring.versions == (3,)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        KEY_V1,  # no version prefix
        "v0:" + KEY_V1,  # versions start at 1
        "vx:" + KEY_V1,
        "v1:" + _b64(b"short"),
        "v1:" + _b64(bytes(64)),  # wrong length
        "v1:not*base64*at*all",
        f"v1:{KEY_V1},v1:{KEY_V2}",  # duplicate version
        f"v1:{KEY_V1},garbage",
    ],
)
def test_keyring_rejects_malformed_key_material(value: str) -> None:
    with pytest.raises(KeyRingError):
        KeyRing.from_env(value)


def test_keyring_never_reveals_key_bytes_in_repr_or_str() -> None:
    ring = KeyRing.from_env(f"v1:{KEY_V1}")

    for rendered in (repr(ring), str(ring)):
        assert KEY_V1 not in rendered
        assert "00010203" not in rendered


def test_encrypt_round_trips_and_binds_ciphertext_to_its_associated_data() -> None:
    ring = KeyRing.from_env(f"v1:{KEY_V1}")

    token = ring.encrypt("+17805558345", aad="callback:usr_a")

    assert token.startswith("enc:v1:")
    assert "+17805558345" not in token
    assert "7805558345" not in token
    assert ring.decrypt(token, aad="callback:usr_a") == "+17805558345"
    with pytest.raises(DecryptionError):
        ring.decrypt(token, aad="callback:usr_b")


def test_encrypt_uses_a_fresh_nonce_every_time() -> None:
    ring = KeyRing.from_env(f"v1:{KEY_V1}")

    first = ring.encrypt("+17805558345", aad="x")
    second = ring.encrypt("+17805558345", aad="x")

    assert first != second
    assert ring.decrypt(first, aad="x") == ring.decrypt(second, aad="x")


def test_tampered_or_malformed_ciphertext_fails_closed() -> None:
    ring = KeyRing.from_env(f"v1:{KEY_V1}")
    token = ring.encrypt("+17805558345", aad="x")
    prefix, version, nonce, body = token.split(":")
    flipped = body[:-2] + ("A" if body[-2] != "A" else "B") + body[-1]

    for bad in (
        f"{prefix}:{version}:{nonce}:{flipped}",
        f"{prefix}:v9:{nonce}:{body}",  # unknown key version
        f"{prefix}:{version}:{nonce}",  # missing body
        "plain:+17805558345",
        "",
        token + "x",
    ):
        with pytest.raises(DecryptionError):
            ring.decrypt(bad, aad="x")


def test_older_key_versions_still_decrypt_and_new_writes_use_the_active_key() -> None:
    old_ring = KeyRing.from_env(f"v1:{KEY_V1}")
    token_v1 = old_ring.encrypt("+17805558345", aad="x")

    rotated = KeyRing.from_env(f"v1:{KEY_V1},v2:{KEY_V2}")

    assert rotated.decrypt(token_v1, aad="x") == "+17805558345"
    assert rotated.encrypt("+17805558345", aad="x").startswith("enc:v2:")
    assert rotated.version_of(token_v1) == 1
    assert rotated.needs_rotation(token_v1) is True
    assert rotated.needs_rotation(rotated.encrypt("+1", aad="x")) is False

    retired = KeyRing.from_env(f"v2:{KEY_V2}")
    with pytest.raises(DecryptionError):
        retired.decrypt(token_v1, aad="x")


def test_blind_index_is_deterministic_keyed_and_purpose_separated() -> None:
    ring = KeyRing.from_env(f"v1:{KEY_V1}")
    other = KeyRing.from_env(f"v1:{KEY_V2}")

    index = ring.blind_index("+17805558345", purpose="callback-number")

    assert index == ring.blind_index("+17805558345", purpose="callback-number")
    assert index != ring.blind_index("+17805558346", purpose="callback-number")
    assert index != ring.blind_index("+17805558345", purpose="other")
    assert index != other.blind_index("+17805558345", purpose="callback-number")
    assert "7805558345" not in index
    assert len(index) >= 32


def test_blind_index_candidates_cover_every_key_version_for_lookups() -> None:
    ring = KeyRing.from_env(f"v1:{KEY_V1},v2:{KEY_V2}")

    candidates = ring.blind_index_candidates("+17805558345", purpose="callback-number")

    assert len(candidates) == 2
    assert len(set(candidates)) == 2
    assert ring.blind_index("+17805558345", purpose="callback-number") in candidates
    assert (
        KeyRing.from_env(f"v1:{KEY_V1}").blind_index("+17805558345", purpose="callback-number")
        in candidates
    )


def test_generate_key_material_is_a_valid_active_key() -> None:
    generated = profile_crypto.generate_key_material(version=4)

    ring = KeyRing.from_env(generated)

    assert ring.active_version == 4
    raw = base64.urlsafe_b64decode(generated.split(":", 1)[1] + "==")
    assert len(raw) == 32
    assert raw != bytes(32)
    assert generated != profile_crypto.generate_key_material(version=4)


def test_secrets_are_compared_and_derived_without_leaking_through_os_environ(monkeypatch) -> None:
    """The ring is built from a value, never by reading the process environment itself."""
    monkeypatch.setenv("CAAL_PROFILE_ENCRYPTION_KEYS", f"v1:{KEY_V1}")

    ring = KeyRing.from_env(os.environ["CAAL_PROFILE_ENCRYPTION_KEYS"])

    assert ring.active_version == 1
