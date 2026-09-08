"""Adaptive password hashing for CAAL's standalone (Cloudflare-free) sign-in.

Passwords are stored only as a self-describing hash string in the PHC format
the reference Argon2 implementation uses::

    $argon2id$v=19$m=65536,t=3,p=2$<salt>$<hash>

Argon2id is the primary algorithm -- memory-hard, side-channel resistant, and
OWASP's first choice. It comes from ``cryptography``'s OpenSSL binding, so no
new dependency is introduced. Where the linked OpenSSL is too old to offer
Argon2, the module falls back to :func:`hashlib.scrypt`, also memory-hard and
also an OWASP-approved choice, under its own ``$scrypt$`` prefix; such a hash
verifies normally and asks to be upgraded on the next successful sign-in.
Both encodings carry their own parameters, so raising the cost later never
invalidates a stored hash.

Verification is total: a corrupt, truncated, or unknown-algorithm hash is a
failed check, never an exception. A row damaged in the database therefore
cannot become an authentication bypass, nor a 500 that distinguishes one
account from another.

The plaintext never leaves this module's arguments. Nothing here logs, and no
exception message quotes a password, a hash, or a salt.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import re
import secrets
import string
import unicodedata
from dataclasses import dataclass

__all__ = [
    "DEFAULT_PARAMS",
    "DEFAULT_SCRYPT_PARAMS",
    "MAX_PASSWORD_LENGTH",
    "MIN_PASSWORD_LENGTH",
    "Argon2Params",
    "PasswordHashError",
    "PasswordPolicyError",
    "ScryptParams",
    "VerifyResult",
    "argon2_available",
    "dummy_verify",
    "generate_password",
    "hash_password",
    "is_password_hash",
    "normalize_password",
    "validate_password",
    "verify_password",
]

# Long enough that a stolen hash is not worth grinding, short enough that a
# person can hold it in a password manager. The bootstrap password this module
# generates is far above the floor.
MIN_PASSWORD_LENGTH = 12
# Bounded so a request body can never become a memory-hard denial of service.
MAX_PASSWORD_LENGTH = 256

_SALT_BYTES = 16
_HASH_BYTES = 32
_ARGON2_VERSION = 19


class PasswordHashError(ValueError):
    """A hash string could not be produced or parsed. Never echoes the value."""


class PasswordPolicyError(ValueError):
    """The proposed password does not meet policy. The message names the rule only."""


@dataclass(frozen=True)
class Argon2Params:
    """Argon2id cost. ``memory_kib`` is the total across all lanes."""

    memory_kib: int = 65536  # 64 MiB
    iterations: int = 3
    lanes: int = 2

    def __post_init__(self) -> None:
        if not 8 * self.lanes <= self.memory_kib <= 4_194_304:
            raise PasswordHashError("Argon2 memory cost is out of bounds")
        if not 1 <= self.iterations <= 32:
            raise PasswordHashError("Argon2 iteration count is out of bounds")
        if not 1 <= self.lanes <= 16:
            raise PasswordHashError("Argon2 lane count is out of bounds")

    @property
    def label(self) -> str:
        return f"m={self.memory_kib},t={self.iterations},p={self.lanes}"


@dataclass(frozen=True)
class ScryptParams:
    """scrypt cost. ``log_n`` is the base-2 logarithm of the CPU/memory cost."""

    log_n: int = 16  # N = 65536 -> 64 MiB at r=8, p=1
    r: int = 8
    p: int = 1

    def __post_init__(self) -> None:
        if not 12 <= self.log_n <= 22:
            raise PasswordHashError("scrypt cost is out of bounds")
        if not 1 <= self.r <= 32 or not 1 <= self.p <= 16:
            raise PasswordHashError("scrypt block or parallelism is out of bounds")

    @property
    def label(self) -> str:
        return f"ln={self.log_n},r={self.r},p={self.p}"

    @property
    def maxmem(self) -> int:
        # OpenSSL refuses to allocate past maxmem; 128*N*r*p is what it needs.
        return 128 * (1 << self.log_n) * self.r * self.p + (1 << 20)


DEFAULT_PARAMS = Argon2Params()
DEFAULT_SCRYPT_PARAMS = ScryptParams()

_ARGON2_ENCODED = re.compile(
    r"^\$argon2id\$v=(?P<v>\d{1,3})\$m=(?P<m>\d{1,8}),t=(?P<t>\d{1,3}),p=(?P<p>\d{1,3})"
    r"\$(?P<salt>[A-Za-z0-9+/]{11,88})\$(?P<hash>[A-Za-z0-9+/]{22,88})$"
)
_SCRYPT_ENCODED = re.compile(
    r"^\$scrypt\$ln=(?P<ln>\d{1,2}),r=(?P<r>\d{1,3}),p=(?P<p>\d{1,3})"
    r"\$(?P<salt>[A-Za-z0-9+/]{11,88})\$(?P<hash>[A-Za-z0-9+/]{22,88})$"
)


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    padded = text + "=" * (-len(text) % 4)
    try:
        return base64.b64decode(padded.encode("ascii"), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PasswordHashError("Password hash is not valid base64") from exc


try:  # pragma: no cover - the except branch needs an OpenSSL without Argon2
    from cryptography.hazmat.primitives.kdf.argon2 import Argon2id as _Argon2id

    def _argon2id_derive_impl(
        password: bytes, salt: bytes, params: Argon2Params, length: int
    ) -> bytes:
        return _Argon2id(
            salt=salt,
            length=length,
            iterations=params.iterations,
            lanes=params.lanes,
            memory_cost=params.memory_kib,
        ).derive(password)

    # Prove the *linked* OpenSSL really implements it; importing does not.
    _argon2id_derive_impl(b"probe", b"0123456789abcdef", Argon2Params(8, 1, 1), 32)
    _argon2id_derive = _argon2id_derive_impl
except Exception:  # pragma: no cover - only on an OpenSSL without Argon2
    _argon2id_derive = None  # type: ignore[assignment]


def argon2_available() -> bool:
    """Whether this build can actually compute Argon2id, not merely import it."""
    return _argon2id_derive is not None


def normalize_password(raw: object) -> bytes:
    """NFKC-normalize and UTF-8 encode, or raise :class:`PasswordPolicyError`.

    Normalizing means a password typed through a different keyboard layout or
    input method still matches; without it, an invisible encoding difference
    could lock a user out of their own account.
    """
    if not isinstance(raw, str):
        raise PasswordPolicyError("Password must be text")
    if len(raw) > MAX_PASSWORD_LENGTH:
        raise PasswordPolicyError(f"Password must be at most {MAX_PASSWORD_LENGTH} characters")
    normalized = unicodedata.normalize("NFKC", raw)
    if "\x00" in normalized:
        raise PasswordPolicyError("Password must not contain a null character")
    return normalized.encode("utf-8")


def validate_password(raw: object) -> str:
    """Apply the policy to a *new* password and return its normalized form.

    Length is the control that matters. Composition rules ("one digit, one
    symbol") push people towards predictable substitutions, so the only extra
    rules here reject inputs that stay trivially guessable at any length.
    """
    if not isinstance(raw, str):
        raise PasswordPolicyError("Password must be text")
    if len(raw) > MAX_PASSWORD_LENGTH:
        raise PasswordPolicyError(f"Password must be at most {MAX_PASSWORD_LENGTH} characters")
    normalized = unicodedata.normalize("NFKC", raw)
    if len(normalized) < MIN_PASSWORD_LENGTH:
        raise PasswordPolicyError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
    if len(normalized) > MAX_PASSWORD_LENGTH:
        raise PasswordPolicyError(f"Password must be at most {MAX_PASSWORD_LENGTH} characters")
    if normalized.strip() != normalized:
        raise PasswordPolicyError("Password must not start or end with whitespace")
    if any(unicodedata.category(char) in ("Cc", "Cf") for char in normalized):
        raise PasswordPolicyError("Password must not contain control characters")
    if len(set(normalized)) < 5:
        raise PasswordPolicyError("Password must use at least five distinct characters")
    return normalized


def hash_password(password: object, *, params: Argon2Params | ScryptParams | None = None) -> str:
    """Hash a password under a fresh random salt; return a self-describing string."""
    material = normalize_password(password)
    salt = secrets.token_bytes(_SALT_BYTES)
    if isinstance(params, ScryptParams):
        return _encode_scrypt(material, salt, params)
    if _argon2id_derive is None:  # pragma: no cover - only without Argon2
        return _encode_scrypt(material, salt, DEFAULT_SCRYPT_PARAMS)
    argon = params or DEFAULT_PARAMS
    digest = _argon2id_derive(material, salt, argon, _HASH_BYTES)
    return f"$argon2id$v={_ARGON2_VERSION}${argon.label}${_b64(salt)}${_b64(digest)}"


def _encode_scrypt(material: bytes, salt: bytes, params: ScryptParams) -> str:
    digest = hashlib.scrypt(
        material,
        salt=salt,
        n=1 << params.log_n,
        r=params.r,
        p=params.p,
        maxmem=params.maxmem,
        dklen=_HASH_BYTES,
    )
    return f"$scrypt${params.label}${_b64(salt)}${_b64(digest)}"


@dataclass(frozen=True)
class VerifyResult:
    """Whether the password matched, and whether the stored hash is now stale."""

    ok: bool
    needs_rehash: bool = False

    def __bool__(self) -> bool:
        return self.ok


def verify_password(
    password: object, encoded: object, *, params: Argon2Params | None = None
) -> VerifyResult:
    """Check a password against a stored hash. Never raises for bad input."""
    if not isinstance(encoded, str) or not encoded:
        return VerifyResult(False)
    try:
        material = normalize_password(password)
    except PasswordPolicyError:
        return VerifyResult(False)

    argon = _ARGON2_ENCODED.match(encoded)
    if argon is not None:
        return _verify_argon2(material, argon, params or DEFAULT_PARAMS)
    scrypt = _SCRYPT_ENCODED.match(encoded)
    if scrypt is not None:
        return _verify_scrypt(material, scrypt)
    return VerifyResult(False)


def _verify_argon2(material: bytes, match: re.Match[str], preferred: Argon2Params) -> VerifyResult:
    if int(match.group("v")) != _ARGON2_VERSION or _argon2id_derive is None:
        return VerifyResult(False)
    try:
        stored = Argon2Params(
            memory_kib=int(match.group("m")),
            iterations=int(match.group("t")),
            lanes=int(match.group("p")),
        )
        salt = _unb64(match.group("salt"))
        expected = _unb64(match.group("hash"))
    except (PasswordHashError, ValueError):
        return VerifyResult(False)
    if len(salt) < 8 or len(expected) < 16:
        return VerifyResult(False)
    try:
        actual = _argon2id_derive(material, salt, stored, len(expected))
    except Exception:
        return VerifyResult(False)
    if not hmac.compare_digest(actual, expected):
        return VerifyResult(False)
    return VerifyResult(True, needs_rehash=stored != preferred)


def _verify_scrypt(material: bytes, match: re.Match[str]) -> VerifyResult:
    try:
        stored = ScryptParams(
            log_n=int(match.group("ln")), r=int(match.group("r")), p=int(match.group("p"))
        )
        salt = _unb64(match.group("salt"))
        expected = _unb64(match.group("hash"))
    except (PasswordHashError, ValueError):
        return VerifyResult(False)
    if len(salt) < 8 or len(expected) < 16:
        return VerifyResult(False)
    try:
        actual = hashlib.scrypt(
            material,
            salt=salt,
            n=1 << stored.log_n,
            r=stored.r,
            p=stored.p,
            maxmem=stored.maxmem,
            dklen=len(expected),
        )
    except Exception:
        return VerifyResult(False)
    if not hmac.compare_digest(actual, expected):
        return VerifyResult(False)
    # scrypt is acceptable, but Argon2id is preferred wherever it exists.
    return VerifyResult(True, needs_rehash=argon2_available() or stored != DEFAULT_SCRYPT_PARAMS)


_DUMMY_HASH: dict[object, str] = {}


def dummy_verify(password: object = "", *, params: Argon2Params | None = None) -> VerifyResult:
    """Burn the work a real verification would, and always fail.

    Called when no account matches, so "no such user" and "wrong password" cost
    the same and cannot be told apart by response time.
    """
    key = params or DEFAULT_PARAMS
    encoded = _DUMMY_HASH.get(key)
    if encoded is None:
        encoded = hash_password(secrets.token_urlsafe(32), params=key)
        _DUMMY_HASH[key] = encoded
    verify_password(password, encoded, params=params)
    return VerifyResult(False)


def is_password_hash(value: object) -> bool:
    """Whether ``value`` is one of our encodings. Used to validate configuration."""
    return isinstance(value, str) and (
        _ARGON2_ENCODED.match(value) is not None or _SCRYPT_ENCODED.match(value) is not None
    )


# An unambiguous alphabet: no O/0, I/l/1, so a password read aloud or copied
# out of a terminal survives the trip.
_ALPHABET = (
    "".join(c for c in string.ascii_uppercase if c not in "IO")
    + "".join(c for c in string.ascii_lowercase if c not in "lo")
    + "23456789"
    + "-_=+.@#%"
)


def generate_password(*, length: int = 24) -> str:
    """A cryptographically random one-time password.

    24 characters over this 66-symbol alphabet is about 145 bits of entropy:
    unguessable, and never derived from anything in the source tree.
    """
    if not MIN_PASSWORD_LENGTH <= int(length) <= MAX_PASSWORD_LENGTH:
        raise PasswordPolicyError("Generated password length is out of bounds")
    while True:
        candidate = "".join(secrets.choice(_ALPHABET) for _ in range(int(length)))
        try:
            return validate_password(candidate)
        except PasswordPolicyError:  # pragma: no cover - astronomically unlikely
            continue
