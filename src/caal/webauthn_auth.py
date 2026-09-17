"""WebAuthn passkeys bound to CAAL's opaque users and server-side sessions.

Authenticators verify a device-local user gesture. CAAL never receives or
stores biometric samples, templates, or attestation certificates.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from webauthn import (
    generate_authentication_options,
    generate_registration_options,
    options_to_json,
    verify_authentication_response,
    verify_registration_response,
)
from webauthn.helpers.structs import (
    AttestationConveyancePreference,
    AuthenticatorAttachment,
    AuthenticatorSelectionCriteria,
    AuthenticatorTransport,
    PublicKeyCredentialDescriptor,
    ResidentKeyRequirement,
    UserVerificationRequirement,
)

from .local_auth import LocalAuth
from .user_scope import is_valid_user_id
from .user_store import ACTIVE, UserStore

_TRANSPORTS = frozenset(item.value for item in AuthenticatorTransport)
_MAX_LABEL = 64
_SESSION_BINDING_LENGTH = 64


class CeremonyError(Exception):
    """A bounded WebAuthn refusal that never contains credential material."""


def _b64decode(value: str) -> bytes:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise CeremonyError("invalid_ceremony")
    try:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (ValueError, TypeError) as exc:
        raise CeremonyError("invalid_ceremony") from exc


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _challenge_from_credential(credential: object) -> tuple[str, bytes]:
    if not isinstance(credential, dict) or credential.get("type") != "public-key":
        raise CeremonyError("invalid_ceremony")
    response = credential.get("response")
    if not isinstance(response, dict):
        raise CeremonyError("invalid_ceremony")
    client_data = response.get("clientDataJSON")
    if not isinstance(client_data, str):
        raise CeremonyError("invalid_ceremony")
    try:
        decoded = json.loads(_b64decode(client_data))
        challenge = decoded["challenge"]
    except (KeyError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CeremonyError("invalid_ceremony") from exc
    if not isinstance(challenge, str):
        raise CeremonyError("invalid_ceremony")
    return challenge, _b64decode(challenge)


def _label(value: object) -> str:
    if not isinstance(value, str) or not value.isprintable():
        raise CeremonyError("invalid_label")
    normalized = " ".join(value.split())
    if not normalized or len(normalized) > _MAX_LABEL:
        raise CeremonyError("invalid_label")
    return normalized


def _binding_hash(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SESSION_BINDING_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CeremonyError("invalid_ceremony")
    return hashlib.sha256(value.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class WebAuthnConfig:
    """Exact relying-party values derived from one trusted public origin."""

    origin: str
    rp_id: str
    rp_name: str = "FRIDAY"
    challenge_ttl_seconds: int = 300

    @classmethod
    def from_public_origin(cls, raw: object) -> "WebAuthnConfig":
        if not isinstance(raw, str) or not raw or raw != raw.strip():
            raise ValueError("A canonical HTTPS public origin is required")
        try:
            parsed = urlsplit(raw)
            port = parsed.port
        except ValueError as exc:
            raise ValueError("The public origin is invalid") from exc
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("A canonical HTTPS public origin is required")
        host = parsed.hostname.lower()
        authority = host if port in (None, 443) else f"{host}:{port}"
        return cls(origin=f"https://{authority}", rp_id=host)


class WebAuthnService:
    """Create and verify short-lived WebAuthn ceremonies over the shared store."""

    def __init__(
        self,
        store: UserStore,
        local_auth: LocalAuth,
        config: WebAuthnConfig,
        *,
        clock: Callable[[], float] = time.time,
        registration_verifier: Callable[..., Any] = verify_registration_response,
        authentication_verifier: Callable[..., Any] = verify_authentication_response,
    ) -> None:
        self._store = store
        self._local_auth = local_auth
        self.config = config
        self._clock = clock
        self._registration_verifier = registration_verifier
        self._authentication_verifier = authentication_verifier

    def _now(self) -> int:
        return int(self._clock())

    def _new_ceremony(
        self,
        purpose: str,
        challenge: bytes,
        user_id: str | None,
        session_binding_hash: str | None = None,
    ) -> str:
        ceremony_id = secrets.token_urlsafe(32)
        now = self._now()
        digest = hashlib.sha256(challenge).hexdigest()
        with closing(self._store.connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "DELETE FROM webauthn_ceremonies WHERE expires_at <= ?", (now,)
                )
                connection.execute(
                    "INSERT INTO webauthn_ceremonies "
                    "(ceremony_id, purpose, user_id, challenge_hash, session_binding_hash, "
                    "created_at, expires_at, consumed_at) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                    (
                        ceremony_id,
                        purpose,
                        user_id,
                        digest,
                        session_binding_hash,
                        now,
                        now + self.config.challenge_ttl_seconds,
                    ),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return ceremony_id

    def consume_ceremony(
        self,
        ceremony_id: object,
        *,
        purpose: str,
        challenge: object,
        user_id: str | None = None,
        session_binding: object = None,
    ) -> bool:
        if (
            not isinstance(ceremony_id, str)
            or len(ceremony_id) > 128
            or purpose not in ("register", "authenticate")
            or not isinstance(challenge, str)
        ):
            return False
        try:
            digest = hashlib.sha256(_b64decode(challenge)).hexdigest()
            binding_hash = _binding_hash(session_binding) if purpose == "register" else None
        except CeremonyError:
            return False
        now = self._now()
        with closing(self._store.connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    "UPDATE webauthn_ceremonies SET consumed_at = ? "
                    "WHERE ceremony_id = ? AND purpose = ? AND challenge_hash = ? "
                    "AND expires_at > ? AND consumed_at IS NULL "
                    "AND ((user_id IS NULL AND ? IS NULL) OR user_id = ?) "
                    "AND ((session_binding_hash IS NULL AND ? IS NULL) "
                    "OR session_binding_hash = ?)",
                    (
                        now,
                        ceremony_id,
                        purpose,
                        digest,
                        now,
                        user_id,
                        user_id,
                        binding_hash,
                        binding_hash,
                    ),
                )
                connection.execute("COMMIT")
                return cursor.rowcount == 1
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def begin_registration(
        self,
        user_id: object,
        label: object,
        current_password: object,
        session_binding: object,
    ) -> dict[str, Any]:
        if not is_valid_user_id(user_id):
            raise CeremonyError("invalid_ceremony")
        profile = self._store.get_user(user_id)
        if (
            profile is None
            or profile.status != ACTIVE
            or not self._local_auth.has_password(user_id)
        ):
            raise CeremonyError("invalid_ceremony")
        _label(label)
        binding_hash = _binding_hash(session_binding)
        self._local_auth.verify_current_password(
            user_id, current_password, action="auth.passkey.add", now=self._now()
        )
        challenge = secrets.token_bytes(32)
        descriptors: list[PublicKeyCredentialDescriptor] = []
        with closing(self._store.connect()) as connection:
            rows = connection.execute(
                "SELECT credential_id, transports FROM webauthn_credentials WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        for row in rows:
            transports = [
                AuthenticatorTransport(item)
                for item in json.loads(row["transports"])
                if item in _TRANSPORTS
            ]
            descriptors.append(
                PublicKeyCredentialDescriptor(
                    id=_b64decode(row["credential_id"]), transports=transports
                )
            )
        options = generate_registration_options(
            rp_id=self.config.rp_id,
            rp_name=self.config.rp_name,
            user_id=str(user_id).encode("utf-8"),
            user_name=str(user_id),
            user_display_name=profile.display_name,
            challenge=challenge,
            timeout=self.config.challenge_ttl_seconds * 1000,
            attestation=AttestationConveyancePreference.NONE,
            authenticator_selection=AuthenticatorSelectionCriteria(
                authenticator_attachment=AuthenticatorAttachment.PLATFORM,
                resident_key=ResidentKeyRequirement.REQUIRED,
                require_resident_key=True,
                user_verification=UserVerificationRequirement.REQUIRED,
            ),
            exclude_credentials=descriptors,
        )
        ceremony_id = self._new_ceremony(
            "register", challenge, str(user_id), session_binding_hash=binding_hash
        )
        return {"ceremonyId": ceremony_id, "publicKey": json.loads(options_to_json(options))}

    def finish_registration(
        self,
        user_id: object,
        ceremony_id: object,
        label: object,
        credential: object,
        session_binding: object,
    ) -> dict[str, Any]:
        if not is_valid_user_id(user_id) or not isinstance(credential, dict):
            raise CeremonyError("invalid_ceremony")
        normalized_label = _label(label)
        if credential.get("authenticatorAttachment") != "platform":
            raise CeremonyError("invalid_ceremony")
        challenge_text, challenge = _challenge_from_credential(credential)
        if not self.consume_ceremony(
            ceremony_id,
            purpose="register",
            challenge=challenge_text,
            user_id=str(user_id),
            session_binding=session_binding,
        ):
            raise CeremonyError("invalid_ceremony")
        try:
            verified = self._registration_verifier(
                credential=credential,
                expected_challenge=challenge,
                expected_rp_id=self.config.rp_id,
                expected_origin=self.config.origin,
                require_user_verification=True,
            )
        except Exception as exc:
            raise CeremonyError("invalid_ceremony") from exc
        transports_raw = credential.get("response", {}).get("transports", [])
        transports = (
            sorted({item for item in transports_raw if item in _TRANSPORTS})
            if isinstance(transports_raw, list)
            else []
        )
        credential_id = _b64encode(bytes(verified.credential_id))
        now = self._now()
        try:
            with closing(self._store.connect()) as connection:
                connection.execute(
                    "INSERT INTO webauthn_credentials "
                    "(credential_id, user_id, public_key, sign_count, transports, label, "
                    "created_at, last_used_at) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                    (
                        credential_id,
                        user_id,
                        bytes(verified.credential_public_key),
                        int(verified.sign_count),
                        json.dumps(transports, separators=(",", ":")),
                        normalized_label,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise CeremonyError("invalid_ceremony") from exc
        self._store.record_audit(
            "auth.passkey.add",
            actor=self._actor(user_id),
            target_id=str(user_id),
            detail={},
            outcome="ok",
            now=now,
        )
        return {"label": normalized_label, "createdAt": now, "lastUsedAt": None}

    def _actor(self, user_id: object):
        from .user_store import Actor

        profile = self._store.get_user(user_id)
        return Actor.for_user(profile) if profile is not None else Actor.system()

    @staticmethod
    def _management_id(credential_id: str) -> str:
        return "key_" + hashlib.sha256(credential_id.encode("ascii")).hexdigest()[:24]

    def list_credentials(self, user_id: object) -> list[dict[str, Any]]:
        if not is_valid_user_id(user_id):
            return []
        with closing(self._store.connect()) as connection:
            rows = connection.execute(
                "SELECT credential_id, label, created_at, last_used_at "
                "FROM webauthn_credentials WHERE user_id = ? ORDER BY created_at, rowid",
                (user_id,),
            ).fetchall()
        return [
            {
                "id": self._management_id(row["credential_id"]),
                "label": row["label"],
                "createdAt": int(row["created_at"]),
                "lastUsedAt": (
                    int(row["last_used_at"]) if row["last_used_at"] is not None else None
                ),
            }
            for row in rows
        ]

    def _credential_id_for_owner(self, user_id: object, management_id: object) -> str | None:
        if (
            not is_valid_user_id(user_id)
            or not isinstance(management_id, str)
            or len(management_id) != 28
        ):
            return None
        with closing(self._store.connect()) as connection:
            rows = connection.execute(
                "SELECT credential_id FROM webauthn_credentials WHERE user_id = ?", (user_id,)
            ).fetchall()
        return next(
            (
                row["credential_id"]
                for row in rows
                if secrets.compare_digest(self._management_id(row["credential_id"]), management_id)
            ),
            None,
        )

    def rename_credential(self, user_id: object, management_id: object, label: object) -> bool:
        credential_id = self._credential_id_for_owner(user_id, management_id)
        normalized = _label(label)
        if credential_id is None:
            return False
        with closing(self._store.connect()) as connection:
            changed = connection.execute(
                "UPDATE webauthn_credentials SET label = ? "
                "WHERE credential_id = ? AND user_id = ?",
                (normalized, credential_id, user_id),
            ).rowcount
        if changed:
            self._store.record_audit(
                "auth.passkey.rename",
                actor=self._actor(user_id),
                target_id=str(user_id),
                detail={},
                now=self._now(),
            )
        return changed == 1

    def revoke_credential(
        self, user_id: object, management_id: object, current_password: object
    ) -> bool:
        self._local_auth.verify_current_password(
            user_id, current_password, action="auth.passkey.revoke", now=self._now()
        )
        credential_id = self._credential_id_for_owner(user_id, management_id)
        if credential_id is None:
            return False
        with closing(self._store.connect()) as connection:
            changed = connection.execute(
                "DELETE FROM webauthn_credentials WHERE credential_id = ? AND user_id = ?",
                (credential_id, user_id),
            ).rowcount
        if changed:
            self._store.record_audit(
                "auth.passkey.revoke",
                actor=self._actor(user_id),
                target_id=str(user_id),
                detail={},
                now=self._now(),
            )
        return changed == 1

    def begin_authentication(self) -> dict[str, Any]:
        challenge = secrets.token_bytes(32)
        options = generate_authentication_options(
            rp_id=self.config.rp_id,
            challenge=challenge,
            timeout=self.config.challenge_ttl_seconds * 1000,
            user_verification=UserVerificationRequirement.REQUIRED,
        )
        ceremony_id = self._new_ceremony("authenticate", challenge, None)
        public_key = json.loads(options_to_json(options))
        # An absent allowCredentials member is the interoperable signal for a
        # discoverable, usernameless ceremony. The library serializes None as
        # an empty array, so remove it at the JSON boundary.
        if public_key.get("allowCredentials") == []:
            public_key.pop("allowCredentials")
        return {"ceremonyId": ceremony_id, "publicKey": public_key}

    def finish_authentication(self, ceremony_id: object, credential: object):
        refusal = CeremonyError("invalid_credentials")
        try:
            if not isinstance(credential, dict):
                raise refusal
            challenge_text, challenge = _challenge_from_credential(credential)
            if not self.consume_ceremony(
                ceremony_id,
                purpose="authenticate",
                challenge=challenge_text,
                user_id=None,
            ):
                raise refusal
            credential_id = credential.get("id")
            if (
                not isinstance(credential_id, str)
                or credential.get("rawId") != credential_id
                or len(credential_id) > 2048
            ):
                raise refusal
            with closing(self._store.connect()) as connection:
                row = connection.execute(
                    "SELECT * FROM webauthn_credentials WHERE credential_id = ?",
                    (credential_id,),
                ).fetchone()
            if row is None:
                raise refusal
            response = credential.get("response")
            user_handle = response.get("userHandle") if isinstance(response, dict) else None
            if not isinstance(user_handle, str) or not secrets.compare_digest(
                _b64decode(user_handle), row["user_id"].encode("utf-8")
            ):
                raise refusal
            profile = self._store.get_user(row["user_id"])
            if profile is None or profile.status != ACTIVE:
                raise refusal
            verified = self._authentication_verifier(
                credential=credential,
                expected_challenge=challenge,
                expected_rp_id=self.config.rp_id,
                expected_origin=self.config.origin,
                credential_public_key=bytes(row["public_key"]),
                credential_current_sign_count=int(row["sign_count"]),
                require_user_verification=True,
            )
            old_count = int(row["sign_count"])
            new_count = int(verified.new_sign_count)
            if (old_count != 0 or new_count != 0) and new_count <= old_count:
                raise refusal
            now = self._now()
            with closing(self._store.connect()) as connection:
                changed = connection.execute(
                    "UPDATE webauthn_credentials SET sign_count = ?, last_used_at = ? "
                    "WHERE credential_id = ? AND user_id = ? AND sign_count = ?",
                    (new_count, now, credential_id, row["user_id"], old_count),
                ).rowcount
            if changed != 1:
                raise refusal
            result = self._local_auth.issue_passkey_session(row["user_id"], now=now)
            if not result.ok:
                raise refusal
            return result
        except CeremonyError:
            raise
        except Exception as exc:
            raise refusal from exc
