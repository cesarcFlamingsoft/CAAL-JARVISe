"""A half-configured multi-user deployment locks sessions instead of degrading.

If an operator has *tried* to configure multi-user identity but something is
missing or invalid, the voice agent must not quietly fall back to the shared
single-user behaviour (where memory is shared and the allowlist phone is
dialed). Every session runs anonymous until the configuration validates.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from caal import profile_crypto, user_api
from caal.internal_auth import AUDIENCE_AGENT, mint_principal
from caal.security_config import (
    ENV_ACCESS_AUD,
    ENV_ACCESS_TEAM_DOMAIN,
    ENV_BOOTSTRAP_ADMIN_EMAIL,
    ENV_INTERNAL_AUTH_SECRET,
    ENV_PROFILE_ENCRYPTION_KEYS,
    REQUIRED_ENV,
)
from caal.user_scope import UserScope

SECRET = "s" * 48
ROOM = "caal-web-abc"
USER = "usr_" + "a" * 24


def _load_voice_agent():
    module_path = Path(__file__).parents[1] / "voice_agent.py"
    spec = importlib.util.spec_from_file_location("voice_agent_lockout_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clear(monkeypatch) -> None:
    for name in REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    user_api.reset_runtime()


def test_no_configuration_means_legacy_sessions(monkeypatch) -> None:
    _clear(monkeypatch)
    voice_agent = _load_voice_agent()

    identity = voice_agent.load_identity_runtime()

    assert identity is None
    assert voice_agent.resolve_inbound_scope("", room_name=ROOM, identity=identity) == (
        UserScope.legacy()
    )


def test_partial_configuration_locks_every_session_anonymous(monkeypatch, tmp_path, caplog) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv(ENV_INTERNAL_AUTH_SECRET, SECRET)
    monkeypatch.setenv(ENV_PROFILE_ENCRYPTION_KEYS, profile_crypto.generate_key_material())
    monkeypatch.setenv(ENV_BOOTSTRAP_ADMIN_EMAIL, "cesarc@mexcantech.com")
    monkeypatch.setenv(ENV_ACCESS_TEAM_DOMAIN, "https://flamingsoftinc.cloudflareaccess.com")
    # The audience tag is missing: attempted, but not valid.
    monkeypatch.delenv(ENV_ACCESS_AUD, raising=False)
    monkeypatch.setenv("CAAL_DATA_DIR", str(tmp_path))
    voice_agent = _load_voice_agent()

    # The agent logger does not propagate to the root logger; listen to it directly.
    voice_agent.logger.addHandler(caplog.handler)
    try:
        with caplog.at_level("ERROR"):
            identity = voice_agent.load_identity_runtime()
    finally:
        voice_agent.logger.removeHandler(caplog.handler)

    assert isinstance(identity, user_api.LockedIdentityRuntime)
    assert "SECURITY CONFIGURATION ERROR" in caplog.text
    assert ENV_ACCESS_AUD in caplog.text
    assert SECRET not in caplog.text

    # Even a principal signed with the real secret is refused: nothing can be
    # verified against a locked runtime, and no user store is consulted.
    metadata = json.dumps(
        {
            "caal_principal": mint_principal(
                secret=SECRET, subject=USER, audience=AUDIENCE_AGENT, claims={"room": ROOM}
            )
        }
    )
    assert voice_agent.resolve_inbound_scope(metadata, room_name=ROOM, identity=identity) == (
        UserScope.anonymous()
    )
    sip_kind = voice_agent.rtc.ParticipantKind.PARTICIPANT_KIND_SIP
    room = SimpleNamespace(
        remote_participants={
            "sip": SimpleNamespace(kind=sip_kind, attributes={"sip.phoneNumber": "+17805558345"})
        }
    )
    assert voice_agent.resolve_sip_caller_scope(room, identity=identity) == UserScope.anonymous()
    assert (
        voice_agent.build_phone_handoff_controller(
            SimpleNamespace(job=SimpleNamespace(metadata=""), api=None),
            user_scope=UserScope.anonymous(),
            identity=identity,
        )
        is None
    )
    assert (
        voice_agent.build_user_callback_dialer(SimpleNamespace(api=None), identity=identity) is None
    )
    # The HTTP identity API fails closed too.
    assert user_api.get_runtime() is None
    user_api.reset_runtime()
