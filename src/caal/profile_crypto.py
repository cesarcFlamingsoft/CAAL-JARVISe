"""Authenticated encryption for sensitive profile fields.

Key material comes from the operator's environment as a small, versioned
ring (``v1:<base64url 32 bytes>[,v2:...]``). Values are encrypted with
AES-256-GCM under the highest version and can be decrypted with any version
still in the ring, which is what makes rotation a two-step, zero-downtime
operation: add the new key, re-encrypt, retire the old key.

Every ciphertext is bound to associated data supplied by the caller (the
owning row, for example), so a value copied between rows fails to decrypt
rather than silently changing whose phone JARVIS would dial. A keyed blind
index lets the store look a value up (caller-id to user) without ever holding
the plaintext in the database or comparing it in the clear.

Nothing in this module logs, and the ring never renders its key bytes.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass, field

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

__all__ = [
    "DecryptionError",
    "KeyRing",
    "KeyRingError",
    "generate_key_material",
]

KEY_BYTES = 32
_NONCE_BYTES = 12
_TOKEN_PREFIX = "enc"
_KEY_ENTRY = re.compile(r"^v(?P<version>\d+):(?P<key>[A-Za-z0-9_-]+={0,2})$")
_INDEX_DOMAIN = b"caal.profile_crypto.blind_index"


class KeyRingError(ValueError):
    """The configured key material cannot be used. The message never echoes it."""


class DecryptionError(ValueError):
    """The value could not be authenticated and decrypted."""


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(text: str) -> bytes:
    padded = text + "=" * (-len(text) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    except (binascii.Error, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("not base64url") from exc


def generate_key_material(*, version: int = 1) -> str:
    """Return a fresh ``vN:<key>`` entry suitable for the environment."""
    if int(version) < 1:
        raise KeyRingError("Key versions start at 1")
    return f"v{int(version)}:{_b64encode(secrets.token_bytes(KEY_BYTES))}"


@dataclass(frozen=True)
class KeyRing:
    """Versioned AES-256-GCM keys. Never prints its material."""

    _keys: dict[int, bytes] = field(repr=False)

    def __repr__(self) -> str:
        return f"KeyRing(versions={self.versions}, active={self.active_version})"

    __str__ = __repr__

    @classmethod
    def from_env(cls, value: str) -> KeyRing:
        """Parse ``v1:<base64url>[,v2:<base64url>]``; raise :class:`KeyRingError` otherwise."""
        if not isinstance(value, str) or not value.strip():
            raise KeyRingError("Profile encryption keys are not configured")
        keys: dict[int, bytes] = {}
        for entry in value.split(","):
            match = _KEY_ENTRY.fullmatch(entry.strip())
            if match is None:
                raise KeyRingError("Profile encryption key entry is malformed")
            version = int(match.group("version"))
            if version < 1:
                raise KeyRingError("Profile encryption key versions start at 1")
            if version in keys:
                raise KeyRingError("Profile encryption key versions must be unique")
            try:
                raw = _b64decode(match.group("key"))
            except ValueError as exc:
                raise KeyRingError("Profile encryption key is not base64url") from exc
            if len(raw) != KEY_BYTES:
                raise KeyRingError("Profile encryption keys must be exactly 32 bytes")
            keys[version] = raw
        return cls(_keys=keys)

    @property
    def versions(self) -> tuple[int, ...]:
        return tuple(sorted(self._keys))

    @property
    def active_version(self) -> int:
        return max(self._keys)

    # --- encryption -----------------------------------------------------------

    def encrypt(self, plaintext: str, *, aad: str) -> str:
        """Encrypt under the active key, bound to ``aad``."""
        if not isinstance(plaintext, str):
            raise TypeError("plaintext must be text")
        version = self.active_version
        nonce = secrets.token_bytes(_NONCE_BYTES)
        ciphertext = AESGCM(self._keys[version]).encrypt(
            nonce, plaintext.encode("utf-8"), self._aad(version, aad)
        )
        return f"{_TOKEN_PREFIX}:v{version}:{_b64encode(nonce)}:{_b64encode(ciphertext)}"

    def decrypt(self, token: str, *, aad: str) -> str:
        """Authenticate and decrypt ``token``; raise :class:`DecryptionError` on any failure."""
        version, nonce, ciphertext = self._parse(token)
        key = self._keys.get(version)
        if key is None:
            raise DecryptionError("No key for this ciphertext version")
        try:
            plaintext = AESGCM(key).decrypt(nonce, ciphertext, self._aad(version, aad))
        except InvalidTag as exc:
            raise DecryptionError("Ciphertext failed authentication") from exc
        try:
            return plaintext.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DecryptionError("Ciphertext did not decode to text") from exc

    def version_of(self, token: str) -> int:
        """The key version a token was written under (parses only the header)."""
        return self._parse(token)[0]

    def needs_rotation(self, token: str) -> bool:
        """Whether ``token`` was written under something other than the active key."""
        return self.version_of(token) != self.active_version

    @staticmethod
    def _aad(version: int, aad: str) -> bytes:
        if not isinstance(aad, str) or not aad:
            raise ValueError("associated data is required")
        return f"{_TOKEN_PREFIX}:v{version}:{aad}".encode("utf-8")

    @staticmethod
    def _parse(token: str) -> tuple[int, bytes, bytes]:
        if not isinstance(token, str):
            raise DecryptionError("Ciphertext must be text")
        parts = token.split(":")
        if len(parts) != 4 or parts[0] != _TOKEN_PREFIX:
            raise DecryptionError("Ciphertext has an unknown shape")
        match = re.fullmatch(r"v(\d+)", parts[1])
        if match is None:
            raise DecryptionError("Ciphertext has an unknown version marker")
        try:
            nonce = _b64decode(parts[2])
            ciphertext = _b64decode(parts[3])
        except ValueError as exc:
            raise DecryptionError("Ciphertext is not base64url") from exc
        if len(nonce) != _NONCE_BYTES or len(ciphertext) < 16:
            raise DecryptionError("Ciphertext is truncated")
        return int(match.group(1)), nonce, ciphertext

    # --- blind index ----------------------------------------------------------

    def blind_index(self, value: str, *, purpose: str, version: int | None = None) -> str:
        """Keyed, purpose-separated digest of ``value`` for equality lookups."""
        chosen = self.active_version if version is None else int(version)
        key = self._keys.get(chosen)
        if key is None:
            raise KeyRingError("No key for the requested index version")
        if not isinstance(purpose, str) or not purpose:
            raise ValueError("purpose is required")
        index_key = hashlib.pbkdf2_hmac(
            "sha256", key, _INDEX_DOMAIN + purpose.encode("utf-8"), 1, dklen=32
        )
        digest = hmac.new(index_key, value.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"v{chosen}:{digest}"

    def blind_index_candidates(self, value: str, *, purpose: str) -> tuple[str, ...]:
        """The index under every known key version, for lookups across a rotation."""
        return tuple(self.blind_index(value, purpose=purpose, version=v) for v in self.versions)
