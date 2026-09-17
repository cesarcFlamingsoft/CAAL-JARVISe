"""Where a company-private session actually comes from.

The entry request is not something the browser, a participant or a model can
assert. It is a claim inside the same short-lived, room-bound ``caal-agent``
principal that names the user, signed by the BFF with the internal secret and
delivered as LiveKit job metadata that no participant can read or forge. The
worker reads it from the *verified* token or not at all.

These tests drive the real ``voice_agent`` entry point against a real identity
runtime and a real minted principal. Nothing is monkeypatched into place.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("isolated_harness_components")

from caal import background_tasks, conversation_ledger, profile_crypto
from caal.access_jwt import AccessConfig, AccessVerifier
from caal.internal_auth import AUDIENCE_AGENT, AUDIENCE_BACKEND, mint_principal
from caal.profile_crypto import KeyRing
from caal.security_config import MultiUserConfig
from caal.user_api import IdentityRuntime
from caal.user_store import MEMBER, Actor, UserStore

SECRET = "s" * 48
TEAM = "https://example-team.cloudflareaccess.com"
AUD = "d1d09a2c79e964918d59077b9bb5b3a7b67ff76a04a80822eb8a3ce5f46354ac"
BOOTSTRAP = "cesarc@mexcantech.com"
ROOM = "caal-web-abc123"
NOW = 1_700_000_000

_voice_agent = None


def _load_voice_agent():
    global _voice_agent
    if _voice_agent is None:
        module_path = Path(__file__).parents[1] / "voice_agent.py"
        spec = importlib.util.spec_from_file_location("voice_agent_company_entry_test", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _voice_agent = module
    return _voice_agent


class Identity:
    def __init__(self, tmp_path) -> None:
        self.keyring = KeyRing.from_env(profile_crypto.generate_key_material(version=1))
        self.store = UserStore(tmp_path / "assistant.sqlite3", keyring=self.keyring)
        self.config = MultiUserConfig(
            internal_auth_secret=SECRET,
            keyring=self.keyring,
            bootstrap_admin_email=BOOTSTRAP,
            access=AccessConfig(team_domain=TEAM, audience=AUD),
            store_path=tmp_path / "assistant.sqlite3",
        )
        self.runtime = IdentityRuntime(
            self.config,
            store=self.store,
            access_verifier=AccessVerifier(
                self.config.access, fetch_jwks=lambda url: {"keys": []}, clock=lambda: NOW
            ),
            clock=lambda: NOW,
        )
        self.admin = self.store.resolve_identity(
            BOOTSTRAP, bootstrap_admin_email=BOOTSTRAP, now=NOW
        )
        self.ana = self.store.create_user(
            email="ana@example.com",
            display_name="Ana",
            role=MEMBER,
            actor=Actor.for_user(self.admin),
            now=NOW,
        )

    def metadata(self, user_id: str, *, room: str = ROOM, claims=None, **overrides) -> str:
        params = {
            "secret": SECRET,
            "subject": user_id,
            "audience": AUDIENCE_AGENT,
            "ttl_seconds": 300,
            "claims": {"room": room, **(claims or {})},
            "now": NOW,
        }
        params.update(overrides)
        return json.dumps({"caal_principal": mint_principal(**params)})


@pytest.fixture
def identity(tmp_path, monkeypatch):
    monkeypatch.setattr(background_tasks, "STORE_PATH", tmp_path / "assistant.sqlite3")
    monkeypatch.setattr(conversation_ledger, "STORE_PATH", tmp_path / "assistant.sqlite3")
    return Identity(tmp_path)


def _resolve(identity, metadata: str):
    return _load_voice_agent().resolve_inbound_session(
        metadata, room_name=ROOM, identity=identity.runtime
    )


def test_a_session_without_the_claim_is_an_ordinary_session(identity) -> None:
    scope, requested = _resolve(identity, identity.metadata(identity.admin.user_id))
    assert scope.user_id == identity.admin.user_id
    assert requested is False


def test_the_signed_claim_is_what_opens_a_company_session(identity) -> None:
    scope, requested = _resolve(
        identity, identity.metadata(identity.admin.user_id, claims={"company_private": True})
    )
    assert scope.user_id == identity.admin.user_id
    assert requested is True


def test_the_claim_must_be_exactly_true(identity) -> None:
    for value in ("true", 1, "yes", None, {}):
        _, requested = _resolve(
            identity, identity.metadata(identity.admin.user_id, claims={"company_private": value})
        )
        assert requested is False, value


def test_an_unsigned_claim_beside_the_principal_is_ignored(identity) -> None:
    """Job metadata is not the trust boundary; the token inside it is."""
    raw = json.loads(identity.metadata(identity.admin.user_id))
    raw["company_private"] = True
    _, requested = _resolve(identity, json.dumps(raw))
    assert requested is False


def test_a_claim_in_a_token_for_another_room_buys_nothing(identity) -> None:
    metadata = identity.metadata(
        identity.admin.user_id, room="caal-web-somewhere-else", claims={"company_private": True}
    )
    scope, requested = _resolve(identity, metadata)
    assert scope.user_id is None
    assert requested is False


def test_a_claim_in_a_token_for_the_wrong_audience_buys_nothing(identity) -> None:
    metadata = identity.metadata(
        identity.admin.user_id, audience=AUDIENCE_BACKEND, claims={"company_private": True}
    )
    scope, requested = _resolve(identity, metadata)
    assert scope.user_id is None
    assert requested is False


def test_a_claim_signed_with_the_wrong_secret_buys_nothing(identity) -> None:
    metadata = identity.metadata(
        identity.admin.user_id, secret="x" * 48, claims={"company_private": True}
    )
    scope, requested = _resolve(identity, metadata)
    assert scope.user_id is None
    assert requested is False


def test_the_existing_scope_resolver_is_unchanged(identity) -> None:
    voice_agent = _load_voice_agent()
    scope = voice_agent.resolve_inbound_scope(
        identity.metadata(identity.ana.user_id), room_name=ROOM, identity=identity.runtime
    )
    assert scope.user_id == identity.ana.user_id
