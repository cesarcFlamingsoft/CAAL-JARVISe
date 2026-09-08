"""Local keypad-PIN authentication for telephone participants.

The PIN is never persisted in plaintext. Store only the output of ``hash_pin``
in a secret environment variable such as ``CAAL_CALL_PIN_HASH``.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import secrets
from enum import Enum


class GateState(str, Enum):
    """Result of accepting a single DTMF digit."""

    COLLECTING = "collecting"
    RETRY = "retry"
    GRANTED = "granted"
    LOCKED = "locked"
    INVALID_INPUT = "invalid_input"


def hash_pin(pin: str, *, salt: bytes | None = None) -> str:
    """Return a salted scrypt encoding suitable for local secret storage."""
    if not pin.isdigit() or not 4 <= len(pin) <= 12:
        raise ValueError("Call PIN must contain 4 to 12 digits.")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(pin.encode(), salt=salt, n=2**14, r=8, p=1)
    return "scrypt:16384:8:1:{}:{}".format(
        base64.urlsafe_b64encode(salt).decode(),
        base64.urlsafe_b64encode(digest).decode(),
    )


def verify_pin(pin: str, encoded_hash: str) -> bool:
    """Verify a PIN against an encoding created by :func:`hash_pin`."""
    try:
        algorithm, n, r, p, salt_text, digest_text = encoded_hash.split(":", 5)
        if algorithm != "scrypt":
            return False
        salt = base64.urlsafe_b64decode(salt_text.encode())
        expected = base64.urlsafe_b64decode(digest_text.encode())
        actual = hashlib.scrypt(
            pin.encode(), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected)
        )
    except (TypeError, ValueError):
        return False
    return hmac.compare_digest(actual, expected)


class CallAccessGate:
    """Collect DTMF PIN digits and enforce a bounded authentication attempt count."""

    def __init__(self, pin_hash: str, *, max_attempts: int = 3) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one.")
        self._pin_hash = pin_hash
        self._max_attempts = max_attempts
        self._attempts = 0
        self._digits = ""
        self._granted = False
        self._locked = False

    @property
    def is_granted(self) -> bool:
        return self._granted

    @property
    def attempts_remaining(self) -> int:
        return max(0, self._max_attempts - self._attempts)

    def accept_digit(self, digit: str) -> GateState:
        """Accept one DTMF character; ``#`` submits and ``*`` clears input.

        Submitting runs scrypt on the calling thread. Callers on an event loop
        should use :meth:`accept_digit_async` instead.
        """
        submitted = self._collect(digit)
        if isinstance(submitted, GateState):
            return submitted
        return self._resolve(bool(submitted) and verify_pin(submitted, self._pin_hash))

    async def accept_digit_async(self, digit: str) -> GateState:
        """Accept one DTMF character, verifying the PIN off the event loop.

        scrypt is deliberately CPU-hard, so verifying it inline would stall the
        audio event loop for every submission.
        """
        submitted = self._collect(digit)
        if isinstance(submitted, GateState):
            return submitted
        if not submitted:
            return self._resolve(False)
        valid = await asyncio.to_thread(verify_pin, submitted, self._pin_hash)
        return self._resolve(valid)

    def _collect(self, digit: str) -> GateState | str:
        """Apply one digit; return the submitted PIN, or a terminal state."""
        if self._locked:
            return GateState.LOCKED
        if self._granted:
            return GateState.GRANTED
        if digit == "*":
            self._digits = ""
            return GateState.COLLECTING
        if digit == "#":
            submitted, self._digits = self._digits, ""
            return submitted
        if len(digit) != 1 or not digit.isdigit():
            return GateState.INVALID_INPUT
        self._digits += digit
        return GateState.COLLECTING

    def _resolve(self, valid: bool) -> GateState:
        """Record the outcome of one submitted PIN."""
        if valid:
            self._granted = True
            return GateState.GRANTED
        self._attempts += 1
        if self._attempts >= self._max_attempts:
            self._locked = True
            return GateState.LOCKED
        return GateState.RETRY
