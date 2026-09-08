"""Adaptive password hashing for the standalone (Cloudflare-free) sign-in.

These tests pin the properties that matter for credential storage: the stored
form is salted and self-describing, verification is total (a corrupt row fails
rather than raising), the cost is upgradeable without invalidating old hashes,
and a generated bootstrap password is unguessable.

Cost parameters here are deliberately tiny so the suite stays fast; production
cost comes from :data:`caal.password_hash.DEFAULT_PARAMS`.
"""

from __future__ import annotations

import pytest

from caal.password_hash import (
    DEFAULT_PARAMS,
    MAX_PASSWORD_LENGTH,
    MIN_PASSWORD_LENGTH,
    Argon2Params,
    PasswordPolicyError,
    ScryptParams,
    argon2_available,
    dummy_verify,
    generate_password,
    hash_password,
    is_password_hash,
    validate_password,
    verify_password,
)

CHEAP = Argon2Params(memory_kib=64, iterations=1, lanes=1)
CHEAP_SCRYPT = ScryptParams(log_n=12, r=8, p=1)
PASSWORD = "correct horse battery staple"


def test_argon2id_is_the_algorithm_actually_in_use() -> None:
    assert argon2_available(), "this build must offer Argon2id, not silently downgrade"
    assert hash_password(PASSWORD, params=CHEAP).startswith("$argon2id$v=19$")


def test_hash_is_salted_so_equal_passwords_do_not_collide() -> None:
    first = hash_password(PASSWORD, params=CHEAP)
    second = hash_password(PASSWORD, params=CHEAP)

    assert first != second
    assert verify_password(PASSWORD, first).ok
    assert verify_password(PASSWORD, second).ok


def test_hash_never_contains_the_password() -> None:
    encoded = hash_password(PASSWORD, params=CHEAP)

    assert PASSWORD not in encoded
    assert "horse" not in encoded


def test_wrong_password_is_refused() -> None:
    encoded = hash_password(PASSWORD, params=CHEAP)

    assert not verify_password("correct horse battery stapl", encoded).ok
    assert not verify_password("", encoded).ok
    assert not verify_password(PASSWORD + " ", encoded).ok


def test_unicode_password_matches_across_normalization_forms() -> None:
    # U+00E9 and "e" + U+0301 are the same password to a person.
    composed = "passwordcaf\u00e9!!"
    decomposed = "passwordcafe\u0301!!"
    assert composed != decomposed
    encoded = hash_password(composed, params=CHEAP)

    assert verify_password(decomposed, encoded).ok


@pytest.mark.parametrize(
    "corrupt",
    [
        "",
        "not-a-hash",
        "$argon2id$v=19$m=64,t=1,p=1$short$short",
        "$argon2id$v=13$m=64,t=1,p=1$YWJjZGVmZ2hpamtsbW5vcA$YWJjZGVmZ2hpamtsbW5vcA",
        "$md5$deadbeef",
        "$argon2id$v=19$m=64,t=1,p=1$!!!!!!!!$YWJjZGVmZ2hpamtsbW5vcA",
        None,
        12345,
        {"hash": "x"},
    ],
)
def test_corrupt_or_unknown_hash_fails_closed_without_raising(corrupt: object) -> None:
    assert verify_password(PASSWORD, corrupt).ok is False


def test_truncated_hash_of_a_real_encoding_still_fails() -> None:
    encoded = hash_password(PASSWORD, params=CHEAP)

    assert not verify_password(PASSWORD, encoded[:-4]).ok
    assert not verify_password(PASSWORD, encoded.replace("$argon2id$", "$argon2i$")).ok


def test_cost_can_be_raised_and_old_hashes_still_verify_but_ask_for_rehash() -> None:
    old = hash_password(PASSWORD, params=CHEAP)
    stronger = Argon2Params(memory_kib=128, iterations=2, lanes=1)

    result = verify_password(PASSWORD, old, params=stronger)

    assert result.ok
    assert result.needs_rehash
    assert not verify_password(PASSWORD, old, params=CHEAP).needs_rehash


def test_scrypt_hashes_verify_and_are_marked_for_upgrade_to_argon2() -> None:
    encoded = hash_password(PASSWORD, params=CHEAP_SCRYPT)

    assert encoded.startswith("$scrypt$ln=12,")
    result = verify_password(PASSWORD, encoded)
    assert result.ok
    assert result.needs_rehash is argon2_available()
    assert not verify_password("wrong password entirely", encoded).ok


def test_dummy_verify_always_fails_and_costs_something() -> None:
    assert dummy_verify("anything", params=CHEAP).ok is False
    assert dummy_verify(params=CHEAP).ok is False


class TestPolicy:
    def test_short_passwords_are_refused(self) -> None:
        with pytest.raises(PasswordPolicyError):
            validate_password("a" * (MIN_PASSWORD_LENGTH - 1))

    def test_absurdly_long_passwords_are_refused_before_hashing(self) -> None:
        with pytest.raises(PasswordPolicyError):
            validate_password("a" * (MAX_PASSWORD_LENGTH + 1))
        encoded = hash_password(PASSWORD, params=CHEAP)
        assert not verify_password("a" * (MAX_PASSWORD_LENGTH + 1), encoded).ok

    def test_low_variety_passwords_are_refused(self) -> None:
        with pytest.raises(PasswordPolicyError):
            validate_password("aaaaaaaaaaaaaaaaaaaa")

    def test_surrounding_whitespace_and_control_characters_are_refused(self) -> None:
        with pytest.raises(PasswordPolicyError):
            validate_password("  a good long password  ")
        with pytest.raises(PasswordPolicyError):
            validate_password("a good long\x00password")

    def test_non_text_is_refused(self) -> None:
        with pytest.raises(PasswordPolicyError):
            validate_password(None)
        with pytest.raises(PasswordPolicyError):
            validate_password(b"a long enough byte string")

    def test_a_reasonable_passphrase_is_accepted(self) -> None:
        assert validate_password(PASSWORD) == PASSWORD


class TestGeneratedPassword:
    def test_generated_password_satisfies_the_policy_and_is_long(self) -> None:
        password = generate_password()

        assert len(password) == 24
        assert validate_password(password) == password

    def test_generated_passwords_do_not_repeat(self) -> None:
        assert len({generate_password() for _ in range(50)}) == 50

    def test_generated_password_avoids_visually_ambiguous_characters(self) -> None:
        for _ in range(20):
            assert not set(generate_password()) & set("O0Il1o")

    def test_absurd_lengths_are_refused(self) -> None:
        with pytest.raises(PasswordPolicyError):
            generate_password(length=4)
        with pytest.raises(PasswordPolicyError):
            generate_password(length=MAX_PASSWORD_LENGTH + 1)


class TestIsPasswordHash:
    def test_recognizes_our_own_encodings(self) -> None:
        assert is_password_hash(hash_password(PASSWORD, params=CHEAP))
        assert is_password_hash(hash_password(PASSWORD, params=CHEAP_SCRYPT))

    def test_rejects_anything_else(self) -> None:
        for value in ["", "hunter2", "$argon2id$", None, 7, "$2b$12$abcdefghijklmnopqrstuv"]:
            assert not is_password_hash(value)


def test_production_defaults_meet_owasp_guidance() -> None:
    # OWASP's Argon2id floor is m=19456 KiB, t=2, p=1.
    assert DEFAULT_PARAMS.memory_kib >= 19456
    assert DEFAULT_PARAMS.iterations >= 2
    assert DEFAULT_PARAMS.lanes >= 1
