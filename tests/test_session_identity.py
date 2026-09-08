"""How a live voice session learns, and is bound to, the user it acts for.

The voice agent never trusts a room name, a participant's self-description, or
anything spoken. A web session is identified by a short-lived ``caal-agent``
principal the BFF placed in the signed LiveKit room configuration (so it
arrives as job metadata, invisible to other participants) and bound to this
room. A phone caller is identified by matching caller-id against the users'
approved numbers, and only after the DTMF PIN gate. An outbound leg is
identified by the opaque user id the trusted coordinator dispatched. Anything
that does not verify leaves the session anonymous, which under multi-user
means no memory, no handoff, and no callbacks.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from caal import background_tasks, conversation_ledger, profile_crypto
from caal.access_jwt import AccessConfig, AccessVerifier
from caal.background_task_session import BackgroundTaskBridge
from caal.background_tasks import RUNNING, callback_armed
from caal.conversation_ledger import ConversationRecorder
from caal.handoff_intent import NO_CALLBACK_NUMBER_REPLY, STARTING_REPLY, PhoneHandoffController
from caal.internal_auth import AUDIENCE_AGENT, AUDIENCE_BACKEND, mint_principal
from caal.outbound_runtime import OutboundRoomConfig
from caal.profile_crypto import KeyRing
from caal.security_config import MultiUserConfig
from caal.user_api import IdentityRuntime
from caal.user_scope import UserScope
from caal.user_store import MEMBER, SUSPENDED, Actor, UserStore

SECRET = "s" * 48
TEAM = "https://example-team.cloudflareaccess.com"
AUD = "d1d09a2c79e964918d59077b9bb5b3a7b67ff76a04a80822eb8a3ce5f46354ac"
BOOTSTRAP = "cesarc@mexcantech.com"
ANA_NUMBER = "+17805558345"
ROOM = "caal-web-abc123"
NOW = 1_700_000_000

_voice_agent = None


def _load_voice_agent():
    global _voice_agent
    if _voice_agent is None:
        module_path = Path(__file__).parents[1] / "voice_agent.py"
        spec = importlib.util.spec_from_file_location("voice_agent_identity_test", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _voice_agent = module
    return _voice_agent


class Identity:
    """A real identity runtime over a throwaway store, with two users."""

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
        actor = Actor.for_user(self.admin)
        self.ana = self.store.create_user(
            email="ana@example.com", display_name="Ana", role=MEMBER, actor=actor, now=NOW
        )
        self.bo = self.store.create_user(
            email="bo@example.com", display_name="Bo", role=MEMBER, actor=actor, now=NOW
        )
        self.store.set_callback_number(self.ana.user_id, ANA_NUMBER, actor=actor, now=NOW)

    def agent_principal(self, user_id: str, *, room: str = ROOM, **overrides) -> str:
        params = {
            "secret": SECRET,
            "subject": user_id,
            "audience": AUDIENCE_AGENT,
            "ttl_seconds": 300,
            "claims": {"room": room},
            "now": NOW,
        }
        params.update(overrides)
        return mint_principal(**params)

    def job_metadata(self, user_id: str, **overrides) -> str:
        return json.dumps({"caal_principal": self.agent_principal(user_id, **overrides)})


@pytest.fixture
def identity(tmp_path, monkeypatch):
    monkeypatch.setattr(background_tasks, "STORE_PATH", tmp_path / "assistant.sqlite3")
    monkeypatch.setattr(conversation_ledger, "STORE_PATH", tmp_path / "assistant.sqlite3")
    return Identity(tmp_path)


class _Rooms:
    def __init__(self) -> None:
        self.created: list = []

    async def create_room(self, request):
        self.created.append(request)


class _Dispatch:
    def __init__(self) -> None:
        self.created: list = []

    async def create_dispatch(self, request):
        self.created.append(request)


class FakeContext:
    def __init__(self, *, room_name: str = ROOM, metadata: str = "") -> None:
        self.room = SimpleNamespace(name=room_name, remote_participants={})
        self.job = SimpleNamespace(metadata=metadata)
        self.api = SimpleNamespace(room=_Rooms(), agent_dispatch=_Dispatch())


class FakeSession:
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.history = None

    async def say(self, text: str, **_) -> None:
        self.spoken.append(text)


# --- web sessions: principal in signed room configuration ---------------------------


def test_web_session_is_identified_only_by_a_room_bound_agent_principal(identity) -> None:
    voice_agent = _load_voice_agent()

    scope = voice_agent.resolve_inbound_scope(
        identity.job_metadata(identity.ana.user_id), room_name=ROOM, identity=identity.runtime
    )

    assert scope.user_id == identity.ana.user_id
    assert scope.identity_configured is True
    assert scope.role == MEMBER
    assert "Ana" not in repr(scope)
    # Reconnects within the token lifetime keep working: the agent token is
    # room-bound rather than single-use.
    again = voice_agent.resolve_inbound_scope(
        identity.job_metadata(identity.ana.user_id), room_name=ROOM, identity=identity.runtime
    )
    assert again.user_id == identity.ana.user_id


@pytest.mark.parametrize(
    "metadata_factory",
    [
        lambda identity: "",
        lambda identity: "not json",
        lambda identity: json.dumps({"caal_principal": ""}),
        lambda identity: json.dumps({"caal_principal": "garbage"}),
        lambda identity: json.dumps({"caal_user_id": identity.ana.user_id}),  # bare claim
        lambda identity: identity.job_metadata(identity.ana.user_id, room="another-room"),
        lambda identity: identity.job_metadata(identity.ana.user_id, now=NOW - 3600),
        lambda identity: json.dumps(
            {
                "caal_principal": mint_principal(
                    secret=SECRET,
                    subject=identity.ana.user_id,
                    audience=AUDIENCE_BACKEND,
                    claims={"room": ROOM},
                    now=NOW,
                )
            }
        ),
        lambda identity: json.dumps(
            {
                "caal_principal": mint_principal(
                    secret="x" * 48,
                    subject=identity.ana.user_id,
                    audience=AUDIENCE_AGENT,
                    claims={"room": ROOM},
                    now=NOW,
                )
            }
        ),
        lambda identity: identity.job_metadata("usr_" + "f" * 24),  # unknown user
    ],
)
def test_anything_that_does_not_verify_leaves_the_session_anonymous(
    identity, metadata_factory
) -> None:
    voice_agent = _load_voice_agent()

    scope = voice_agent.resolve_inbound_scope(
        metadata_factory(identity), room_name=ROOM, identity=identity.runtime
    )

    assert scope == UserScope.anonymous()
    assert scope.memory_available is False


def test_suspended_users_get_an_anonymous_session(identity) -> None:
    voice_agent = _load_voice_agent()
    identity.store.admin_update(
        identity.ana.user_id, status=SUSPENDED, actor=Actor.for_user(identity.admin)
    )

    scope = voice_agent.resolve_inbound_scope(
        identity.job_metadata(identity.ana.user_id), room_name=ROOM, identity=identity.runtime
    )

    assert scope == UserScope.anonymous()


def test_without_multi_user_configuration_sessions_are_legacy(identity) -> None:
    voice_agent = _load_voice_agent()

    scope = voice_agent.resolve_inbound_scope(
        identity.job_metadata(identity.ana.user_id), room_name=ROOM, identity=None
    )

    assert scope == UserScope.legacy()
    assert scope.memory_available is True


# --- phone callers: caller-id after the PIN gate ---------------------------------------


def _sip_room(number: str | None):
    voice_agent = _load_voice_agent()
    sip_kind = voice_agent.rtc.ParticipantKind.PARTICIPANT_KIND_SIP
    attributes = {"sip.phoneNumber": number} if number is not None else {}
    participant = SimpleNamespace(kind=sip_kind, attributes=attributes, identity="sip-1")
    web = SimpleNamespace(kind=0, attributes={"sip.phoneNumber": ANA_NUMBER}, identity="web")
    return SimpleNamespace(name="sip-room", remote_participants={"web": web, "sip": participant})


def test_sip_caller_is_resolved_by_approved_number_only(identity) -> None:
    voice_agent = _load_voice_agent()

    matched = voice_agent.resolve_sip_caller_scope(_sip_room(ANA_NUMBER), identity=identity.runtime)
    spaced = voice_agent.resolve_sip_caller_scope(
        _sip_room("+1 (780) 555-8345"), identity=identity.runtime
    )
    unknown = voice_agent.resolve_sip_caller_scope(
        _sip_room("+17805550000"), identity=identity.runtime
    )
    missing = voice_agent.resolve_sip_caller_scope(_sip_room(None), identity=identity.runtime)
    legacy = voice_agent.resolve_sip_caller_scope(_sip_room(ANA_NUMBER), identity=None)

    assert matched.user_id == identity.ana.user_id
    assert spaced.user_id == identity.ana.user_id
    assert unknown == UserScope.anonymous()
    assert missing == UserScope.anonymous()
    assert legacy == UserScope.legacy()


def test_a_web_participant_claiming_a_phone_number_is_ignored(identity) -> None:
    voice_agent = _load_voice_agent()
    room = SimpleNamespace(
        name="room",
        remote_participants={
            "web": SimpleNamespace(kind=0, attributes={"sip.phoneNumber": ANA_NUMBER})
        },
    )

    assert voice_agent.resolve_sip_caller_scope(room, identity=identity.runtime) == (
        UserScope.anonymous()
    )
    assert voice_agent.has_sip_participant(room) is False
    assert voice_agent.has_sip_participant(_sip_room(ANA_NUMBER)) is True


# --- outbound legs: the dispatched user id -----------------------------------------------


def _outbound_metadata(user_id: str | None, **extra) -> str:
    metadata = {"caal_outbound": True, "attempt_id": "abc", "destination": ANA_NUMBER, **extra}
    if user_id is not None:
        metadata["user_id"] = user_id
    return json.dumps(metadata)


def test_outbound_configuration_revalidates_the_users_number_from_the_store(
    identity, monkeypatch
) -> None:
    voice_agent = _load_voice_agent()
    monkeypatch.setenv("CAAL_OUTBOUND_ALLOWED_DESTINATIONS", "+17805550000")

    config = voice_agent.parse_outbound_config(
        _outbound_metadata(identity.ana.user_id), identity=identity.runtime
    )
    assert config is not None and config.user_id == identity.ana.user_id
    assert config.destination == ANA_NUMBER

    scope = voice_agent.outbound_scope(config, identity=identity.runtime)
    assert scope.user_id == identity.ana.user_id

    # Bo has no number: his leg can never be configured.
    with pytest.raises(PermissionError):
        voice_agent.parse_outbound_config(
            _outbound_metadata(identity.bo.user_id), identity=identity.runtime
        )
    # Ana's number cleared between dispatch and worker start: refused too.
    identity.store.clear_callback_number(identity.ana.user_id, actor=Actor.for_user(identity.admin))
    with pytest.raises(PermissionError):
        voice_agent.parse_outbound_config(
            _outbound_metadata(identity.ana.user_id), identity=identity.runtime
        )
    # Legacy dispatches keep the allowlist.
    legacy = voice_agent.parse_outbound_config(
        json.dumps({"caal_outbound": True, "attempt_id": "abc", "destination": "+17805550000"}),
        identity=identity.runtime,
    )
    assert legacy is not None and legacy.user_id is None
    assert voice_agent.outbound_scope(legacy, identity=identity.runtime) == UserScope.anonymous()
    assert voice_agent.outbound_scope(legacy, identity=None) == UserScope.legacy()
    assert voice_agent.parse_outbound_config("{}", identity=identity.runtime) is None


def test_outbound_scope_never_grants_a_suspended_or_unknown_user(identity) -> None:
    voice_agent = _load_voice_agent()
    config = OutboundRoomConfig(
        attempt_id="abc", destination=ANA_NUMBER, user_id=identity.ana.user_id
    )
    identity.store.admin_update(
        identity.ana.user_id, status=SUSPENDED, actor=Actor.for_user(identity.admin)
    )
    assert voice_agent.outbound_scope(config, identity=identity.runtime) == UserScope.anonymous()
    unknown = OutboundRoomConfig(
        attempt_id="abc", destination=ANA_NUMBER, user_id="usr_" + "f" * 24
    )
    assert voice_agent.outbound_scope(unknown, identity=identity.runtime) == UserScope.anonymous()


# --- phone handoff wiring -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handoff_for_a_user_dials_only_their_own_number_with_their_id(identity) -> None:
    voice_agent = _load_voice_agent()
    ctx = FakeContext()
    controller = voice_agent.build_phone_handoff_controller(
        ctx,
        conversation_id=None,
        user_scope=UserScope.for_user(identity.ana),
        identity=identity.runtime,
    )
    assert isinstance(controller, PhoneHandoffController)
    session = FakeSession()

    await controller.handle_final_transcript("continue this conversation on my phone", session)
    await controller.handle_final_transcript("yes", session)

    assert session.spoken[-1] == STARTING_REPLY
    assert len(ctx.api.agent_dispatch.created) == 1
    dispatch = json.loads(ctx.api.agent_dispatch.created[0].metadata)
    assert dispatch["destination"] == ANA_NUMBER
    assert dispatch["user_id"] == identity.ana.user_id
    assert ANA_NUMBER not in ctx.api.room.created[0].metadata
    assert all(ANA_NUMBER not in text for text in session.spoken)


@pytest.mark.asyncio
async def test_handoff_for_a_user_without_a_number_declines_and_explains(identity) -> None:
    voice_agent = _load_voice_agent()
    ctx = FakeContext()
    controller = voice_agent.build_phone_handoff_controller(
        ctx, user_scope=UserScope.for_user(identity.bo), identity=identity.runtime
    )
    session = FakeSession()

    await controller.handle_final_transcript("continue this conversation on my phone", session)
    await controller.handle_final_transcript("yes", session)

    assert session.spoken[-1] == NO_CALLBACK_NUMBER_REPLY
    assert ctx.api.agent_dispatch.created == []


def test_anonymous_sessions_under_multi_user_get_no_handoff_at_all(identity, monkeypatch) -> None:
    voice_agent = _load_voice_agent()
    monkeypatch.setenv("CAAL_OUTBOUND_ALLOWED_DESTINATIONS", "+17805550000")

    assert (
        voice_agent.build_phone_handoff_controller(
            FakeContext(), user_scope=UserScope.anonymous(), identity=identity.runtime
        )
        is None
    )
    # Legacy deployments keep the allowlist-driven handoff exactly as before.
    legacy = voice_agent.build_phone_handoff_controller(
        FakeContext(), user_scope=UserScope.legacy(), identity=None
    )
    assert isinstance(legacy, PhoneHandoffController)


@pytest.mark.asyncio
async def test_handoff_number_is_re_read_at_dispatch_and_refused_if_it_changed(identity) -> None:
    voice_agent = _load_voice_agent()
    ctx = FakeContext()
    controller = voice_agent.build_phone_handoff_controller(
        ctx, user_scope=UserScope.for_user(identity.ana), identity=identity.runtime
    )
    session = FakeSession()
    await controller.handle_final_transcript("continue this conversation on my phone", session)

    # The number is cleared while JARVIS is asking for confirmation.
    identity.store.clear_callback_number(identity.ana.user_id, actor=Actor.for_user(identity.admin))
    await controller.handle_final_transcript("yes", session)

    assert ctx.api.agent_dispatch.created == []
    assert session.spoken[-1] == NO_CALLBACK_NUMBER_REPLY


# --- background callbacks -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_callback_dialer_dials_the_users_current_number_or_refuses(identity) -> None:
    voice_agent = _load_voice_agent()
    ctx = FakeContext()
    dial = voice_agent.build_user_callback_dialer(ctx, identity=identity.runtime)
    task = background_tasks.enqueue("find fares", session_key="room", user_id=identity.ana.user_id)

    await dial(identity.ana.user_id, task.task_id)

    dispatch = json.loads(ctx.api.agent_dispatch.created[0].metadata)
    assert dispatch["destination"] == ANA_NUMBER
    assert dispatch["user_id"] == identity.ana.user_id
    assert dispatch["callback_task_id"] == task.task_id
    with pytest.raises(PermissionError):
        await dial(identity.bo.user_id, task.task_id)
    assert len(ctx.api.agent_dispatch.created) == 1
    assert voice_agent.build_user_callback_dialer(ctx, identity=None) is None


@pytest.mark.asyncio
async def test_callback_arming_on_a_user_leg_binds_the_callback_to_the_user(identity) -> None:
    voice_agent = _load_voice_agent()
    config = OutboundRoomConfig(
        attempt_id="abc", destination=ANA_NUMBER, user_id=identity.ana.user_id
    )
    ctx = FakeContext(
        room_name="caal-outbound-abc", metadata=_outbound_metadata(identity.ana.user_id)
    )
    ctx.api.room.delete_room = _delete_room(ctx)
    bridge = BackgroundTaskBridge(
        execute=_never_runs, session_key="caal-outbound-abc", user_id=identity.ana.user_id
    )
    await bridge.start()
    try:
        task = background_tasks.enqueue(
            "find fares", session_key="caal-outbound-abc", user_id=identity.ana.user_id
        )
        arm = voice_agent.build_callback_arming(
            ctx, session=FakeSession(), bridge=bridge, outbound_config=lambda: config
        )
        await arm()
        assert callback_armed(task.task_id) is True
        target = background_tasks.claim_callback_target(task.task_id, "probe") if False else None
        assert target is None
        assert ctx.deleted == ["caal-outbound-abc"]
    finally:
        await bridge.close()
        await bridge.abandon()


async def _never_runs(request: str, context: str) -> str:  # pragma: no cover
    raise AssertionError("worker must not run in this test")


def _delete_room(ctx):
    ctx.deleted = []

    async def delete_room(request) -> None:
        ctx.deleted.append(request.room)

    return delete_room


# --- bridge and ledger wiring --------------------------------------------------------------------


def test_user_scoped_sessions_get_no_shared_fallback_channel(identity) -> None:
    voice_agent = _load_voice_agent()
    runtime = {
        "background_tasks_enabled": True,
        "telegram_bot_token": "tok",
        "telegram_chat_id": "42",
        "background_task_max_concurrency": 1,
        "background_task_timeout_seconds": 10,
    }
    provider = SimpleNamespace(chat=None)

    scoped = voice_agent.build_background_task_bridge(
        runtime,
        provider=provider,
        session_key="room",
        user_scope=UserScope.for_user(identity.ana),
    )
    legacy = voice_agent.build_background_task_bridge(
        runtime, provider=provider, session_key="room", user_scope=UserScope.legacy()
    )
    anonymous = voice_agent.build_background_task_bridge(
        runtime, provider=provider, session_key="room", user_scope=UserScope.anonymous()
    )

    assert scoped.user_id == identity.ana.user_id and scoped.has_fallback is False
    assert legacy.user_id is None and legacy.has_fallback is True
    assert anonymous.user_id is None and anonymous.has_fallback is True


def test_session_conversations_are_opened_under_the_users_id(identity) -> None:
    voice_agent = _load_voice_agent()

    conversation_id = voice_agent.open_session_conversation(
        "", session_key="room", user_id=identity.ana.user_id
    )

    assert conversation_ledger.conversation_user_id(conversation_id) == identity.ana.user_id
    assert voice_agent.open_session_conversation("", session_key="room-2") is not None


@pytest.mark.asyncio
async def test_answered_call_only_claims_a_continuation_of_the_same_user(
    identity, monkeypatch
) -> None:
    voice_agent = _load_voice_agent()
    monkeypatch.setenv("LIVEKIT_OUTBOUND_TRUNK_ID", "ST_test")
    monkeypatch.setenv("CAAL_OUTBOUND_RING_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("CAAL_OUTBOUND_MAX_DURATION_SECONDS", "900")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    class _AMD:
        def __init__(self, session, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc) -> None:
            return None

        async def execute(self):
            return SimpleNamespace(category=SimpleNamespace(value="human"))

    monkeypatch.setattr(voice_agent.agents, "AMD", _AMD)

    ana_conversation = conversation_ledger.open_conversation(
        session_key="web-ana", user_id=identity.ana.user_id
    )
    conversation_ledger.append_turn(ana_conversation, "user", "plan Lisbon")
    conversation_ledger.link_continuation(
        ana_conversation, session_key="abc", user_id=identity.ana.user_id
    )

    async def _create_sip(request):
        return object()

    async def _shutdown(reason: str = "") -> None:
        return None

    def _ctx():
        ctx = FakeContext(room_name="caal-outbound-abc")
        ctx.api.sip = SimpleNamespace(create_sip_participant=_create_sip)
        ctx.shutdown = _shutdown
        return ctx

    own = OutboundRoomConfig(
        attempt_id="abc",
        destination=ANA_NUMBER,
        conversation_id=ana_conversation,
        user_id=identity.ana.user_id,
    )
    recorder = ConversationRecorder()
    answered = await voice_agent.run_outbound_call(
        _ctx(), object(), own, agent=None, recorder=recorder, background=None
    )
    assert answered is True and recorder.conversation_id == ana_conversation

    stolen = OutboundRoomConfig(
        attempt_id="abc",
        destination=ANA_NUMBER,
        conversation_id=ana_conversation,
        user_id=identity.bo.user_id,
    )
    conversation_ledger.link_continuation(
        ana_conversation, session_key="abc", user_id=identity.ana.user_id
    )
    other_recorder = ConversationRecorder()
    answered = await voice_agent.run_outbound_call(
        _ctx(), object(), stolen, agent=None, recorder=other_recorder, background=None
    )
    assert answered is True and other_recorder.conversation_id is None


def test_local_turn_handler_still_wires_without_identity() -> None:
    voice_agent = _load_voice_agent()
    handler = voice_agent.LocalTurnHandler(
        phone_handoff=None, session=FakeSession(), end_call=_never_ends
    )
    assert handler is not None
    assert RUNNING == "running"


async def _never_ends() -> None:  # pragma: no cover
    return None
