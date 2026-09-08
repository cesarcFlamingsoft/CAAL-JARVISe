"""Regression coverage for "hang up and call me back" after a phone handoff.

The reported failure: a long task is scheduled from the web session, the
conversation is handed off to the phone, and the caller's explicit callback
command is met with silence. The web session's bridge kept its work under the
web room name, the phone leg's bridge searched under its own room name, found
nothing, and neither armed a callback nor hung up.

Background work now belongs to the logical conversation (the private ledger
id both legs share), and a phone leg adopts that ownership the moment a human
answers. A session with no ledger conversation keeps its room name as the
owner key. The raw conversation id itself is never stored in the task queue,
logged, or rendered; a derived opaque key stands in for it.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from caal import background_tasks, conversation_ledger
from caal.background_task_session import (
    BACKGROUND_ACK_REPLY,
    CALLBACK_ARMED_REPLY,
    CALLBACK_NOTHING_RUNNING_REPLY,
    BackgroundTaskBridge,
)
from caal.background_tasks import RUNNING, callback_armed, list_tasks
from caal.conversation_ledger import ConversationRecorder
from caal.outbound_runtime import OutboundRoomConfig

APPROVED = "+17805558345"
WEB_ROOM = "room-web-1"
PHONE_ROOM = "caal-outbound-abc"
ATTEMPT_ID = "abc"
BACKGROUND_REQUEST = (
    "research flight prices to Lisbon in the background and let me know when it's done"
)
CALLBACK_REQUEST = "hang up and call me back when you're done"

_voice_agent = None


def _load_voice_agent():
    """Load voice_agent.py once; it is a script, not an importable package."""
    global _voice_agent
    if _voice_agent is None:
        module_path = Path(__file__).parents[1] / "voice_agent.py"
        spec = importlib.util.spec_from_file_location(
            "voice_agent_callback_handoff_test", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _voice_agent = module
    return _voice_agent


# --- fakes -------------------------------------------------------------------


class FakeSession:
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.history = SimpleNamespace(items=[])

    async def say(self, text: str) -> None:
        self.spoken.append(text)


class FakeExecute:
    """Injected worker: holds the task open until released."""

    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.gate.set()

    async def __call__(self, request: str, context: str) -> str:
        await self.gate.wait()
        return "Cheapest fare is 180 euros on Tuesday."


class FakeDial:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, destination: str, task_id: str) -> None:
        self.calls.append((destination, task_id))


class FakeEndCall:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self) -> None:
        self.calls += 1


class FakeArm:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self) -> None:
        self.calls += 1


class FakeRoomService:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def delete_room(self, request) -> None:
        self.deleted.append(request.room)


class FakeSip:
    async def create_sip_participant(self, request):
        return object()


class FakeContext:
    def __init__(self, *, room_name: str, metadata: str) -> None:
        self.room = SimpleNamespace(name=room_name)
        self.job = SimpleNamespace(metadata=metadata)
        self.api = SimpleNamespace(room=FakeRoomService(), sip=FakeSip())
        self.shutdown_reasons: list[str] = []

    async def shutdown(self, reason: str = "") -> None:
        self.shutdown_reasons.append(reason)


class _FakeAMDResult:
    def __init__(self, category: str) -> None:
        self.category = SimpleNamespace(value=category)


class _FakeAMDFactory:
    def __init__(self, category: str) -> None:
        self.category = category

    def __call__(self, session, **kwargs):
        factory = self

        class _Detector:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc) -> None:
                return None

            async def execute(self):
                return _FakeAMDResult(factory.category)

        return _Detector()


class TypedMessage:
    def __init__(self, text: str) -> None:
        self.text_content = text


# --- fixtures and helpers ----------------------------------------------------


@pytest.fixture
def store(monkeypatch, tmp_path):
    """One shared SQLite file for the task queue and the ledger, as in production."""
    path = tmp_path / "assistant.sqlite3"
    monkeypatch.setattr(background_tasks, "STORE_PATH", path)
    monkeypatch.setattr(conversation_ledger, "STORE_PATH", path)
    return path


@pytest.fixture
def outbound_env(monkeypatch):
    monkeypatch.setenv("LIVEKIT_OUTBOUND_TRUNK_ID", "ST_test")
    monkeypatch.setenv("CAAL_OUTBOUND_ALLOWED_DESTINATIONS", APPROVED)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("CAAL_OUTBOUND_RING_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("CAAL_OUTBOUND_MAX_DURATION_SECONDS", raising=False)


def _dispatch_metadata(conversation_id: str | None) -> str:
    metadata = {"caal_outbound": True, "attempt_id": ATTEMPT_ID, "destination": APPROVED}
    if conversation_id is not None:
        metadata["conversation_id"] = conversation_id
    return json.dumps(metadata)


def _outbound_config(conversation_id: str | None) -> OutboundRoomConfig:
    config = OutboundRoomConfig.from_dispatch_metadata(
        _dispatch_metadata(conversation_id), allowed_destinations=APPROVED
    )
    assert config is not None
    return config


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0.01)


async def _teardown(*bridges: BackgroundTaskBridge) -> None:
    for bridge in bridges:
        await bridge.close()
    await _settle()
    for bridge in bridges:
        await bridge.abandon()


async def _run_turn(handler, text: str, reached_llm: list[str]) -> None:
    """Mimic the LiveKit turn: a StopResponse means Hermes never sees the text."""
    voice_agent = _load_voice_agent()
    assistant = SimpleNamespace(_turn_consumed=handler.turn_consumed, _sync_return_context=None)
    try:
        await voice_agent.VoiceAssistant.on_user_turn_completed(
            assistant, SimpleNamespace(), TypedMessage(text)
        )
    except voice_agent.StopResponse:
        return
    reached_llm.append(text)


async def _web_session_with_running_task(conversation_id: str, *, dial: FakeDial | None = None):
    """The origin web session schedules work that is still running."""
    execute = FakeExecute()
    execute.gate.clear()
    bridge = BackgroundTaskBridge(
        execute=execute,
        session_key=WEB_ROOM,
        conversation_id=conversation_id,
        dial_callback=dial,
        max_concurrency=1,
    )
    session = FakeSession()
    await bridge.start()
    assert await bridge.handle_turn(BACKGROUND_REQUEST, session) is True
    await _settle()
    (task,) = list_tasks(statuses=[RUNNING])
    return bridge, execute, session, task


def _phone_leg(voice_agent, bridge: BackgroundTaskBridge, config: OutboundRoomConfig):
    """The outbound phone leg's turn handler, wired exactly as the entrypoint wires it."""
    session = FakeSession()
    ctx = FakeContext(room_name=PHONE_ROOM, metadata=_dispatch_metadata(config.conversation_id))
    arm = voice_agent.build_callback_arming(
        ctx, session=session, bridge=bridge, outbound_config=lambda: config
    )
    assert arm is not None
    handler = voice_agent.LocalTurnHandler(
        phone_handoff=None,
        session=session,
        end_call=FakeEndCall(),
        background=bridge,
        arm_callback_and_end_call=arm,
    )
    return handler, session, ctx


def _raw_dump(path) -> str:
    connection = sqlite3.connect(path)
    try:
        return "\n".join(connection.iterdump())
    finally:
        connection.close()


# --- the reported failure ------------------------------------------------------


@pytest.mark.asyncio
async def test_phone_leg_arms_callback_for_the_task_the_web_session_scheduled(store) -> None:
    voice_agent = _load_voice_agent()
    conversation_id = conversation_ledger.open_conversation(session_key=WEB_ROOM)
    web_dial = FakeDial()
    web_bridge, web_execute, web_session, task = await _web_session_with_running_task(
        conversation_id, dial=web_dial
    )

    # The handoff dispatches the phone leg with only the opaque conversation id.
    # Its bridge is keyed by its own room until a human answers and it adopts
    # the conversation, which is what run_outbound_call does.
    config = _outbound_config(conversation_id)
    phone_bridge = BackgroundTaskBridge(execute=FakeExecute(), session_key=PHONE_ROOM)
    phone_bridge.bind_conversation(config.conversation_id)
    await phone_bridge.start()
    handler, phone_session, ctx = _phone_leg(voice_agent, phone_bridge, config)
    reached_llm: list[str] = []
    try:
        handler.on_final_transcript(CALLBACK_REQUEST)
        await _run_turn(handler, CALLBACK_REQUEST, reached_llm)
        await _settle()

        assert reached_llm == []
        assert phone_session.spoken == [CALLBACK_ARMED_REPLY]
        assert callback_armed(task.task_id) is True
        # Only the phone room ends; the web session and its work carry on.
        assert ctx.api.room.deleted == [PHONE_ROOM]
        assert web_session.spoken == [BACKGROUND_ACK_REPLY]
        assert list_tasks(statuses=[RUNNING])[0].task_id == task.task_id

        # The task settles on the web session's runner, which places the callback.
        web_execute.gate.set()
        await _settle()
        assert web_dial.calls == [(APPROVED, task.task_id)]
        assert callback_armed(task.task_id) is False
    finally:
        web_execute.gate.set()
        await _teardown(web_bridge, phone_bridge)


@pytest.mark.asyncio
async def test_phone_pdf_request_uploads_then_automatically_calls_back(store, tmp_path) -> None:
    """A deliverable requested on an active phone leg needs no second callback turn."""
    from caal.document_work import DocumentWorker

    class Compose:
        async def __call__(self, request: str, context: str) -> str:
            return "Title: JARVIS\nA local-first assistant."

    uploads: list[bytes] = []
    dial = FakeDial()

    async def deliver(*, filename: str, content: bytes, caption: str) -> None:
        assert filename.endswith(".pdf")
        uploads.append(content)

    config = _outbound_config(None)
    bridge = BackgroundTaskBridge(
        execute=DocumentWorker(compose=Compose(), deliver=deliver, artifact_dir=tmp_path / "art"),
        session_key=PHONE_ROOM,
        dial_callback=dial,
        max_concurrency=1,
    )
    await bridge.start()
    handler, session, ctx = _phone_leg(voice_agent := _load_voice_agent(), bridge, config)
    reached_llm: list[str] = []
    try:
        request = "Create a short PDF about yourself and send it to me on Telegram."
        handler.on_final_transcript(request)
        await _run_turn(handler, request, reached_llm)
        await _settle()

        (task,) = list_tasks(session_key=PHONE_ROOM)
        assert reached_llm == []
        assert ctx.api.room.deleted == [PHONE_ROOM]
        assert uploads and uploads[0].startswith(b"%PDF-")
        assert dial.calls == [(APPROVED, task.task_id)]
        assert "would you like me to call" not in " ".join(session.spoken).lower()
    finally:
        await _teardown(bridge)


@pytest.mark.asyncio
async def test_unrelated_sessions_cannot_arm_another_conversations_task(store) -> None:
    voice_agent = _load_voice_agent()
    conversation_id = conversation_ledger.open_conversation(session_key=WEB_ROOM)
    other_conversation_id = conversation_ledger.open_conversation(session_key="room-web-2")
    web_bridge, web_execute, _, task = await _web_session_with_running_task(conversation_id)

    strangers = {
        # Another web conversation of its own.
        "other conversation": BackgroundTaskBridge(
            execute=FakeExecute(), session_key="room-web-2", conversation_id=other_conversation_id
        ),
        # A fresh outbound call that continues nothing.
        "fresh phone call": BackgroundTaskBridge(execute=FakeExecute(), session_key=PHONE_ROOM),
        # A session reusing the origin's room name without its conversation.
        "same room name": BackgroundTaskBridge(execute=FakeExecute(), session_key=WEB_ROOM),
    }
    try:
        for name, bridge in strangers.items():
            await bridge.start()
            assert bridge.can_arm_callback is False, name
            session = FakeSession()
            assert await bridge.arm_callback(APPROVED, session) is False, name
            assert session.spoken == [CALLBACK_NOTHING_RUNNING_REPLY], name

        # The phone leg of a different conversation gets the same answer, and
        # a literal callback command there keeps the line open.
        config = _outbound_config(other_conversation_id)
        phone_bridge = strangers["other conversation"]
        handler, phone_session, ctx = _phone_leg(voice_agent, phone_bridge, config)
        reached_llm: list[str] = []
        handler.on_final_transcript(CALLBACK_REQUEST)
        await _run_turn(handler, CALLBACK_REQUEST, reached_llm)
        await _settle()
        assert reached_llm == []
        assert phone_session.spoken == [CALLBACK_NOTHING_RUNNING_REPLY]
        assert ctx.api.room.deleted == []

        # The store refuses any key that is not the owner's.
        for key in (WEB_ROOM, PHONE_ROOM, conversation_id, other_conversation_id):
            assert background_tasks.arm_callback(task.task_id, APPROVED, session_key=key) is False
        assert callback_armed(task.task_id) is False
    finally:
        web_execute.gate.set()
        await _teardown(web_bridge, *strangers.values())


@pytest.mark.asyncio
async def test_conversation_id_never_reaches_logs_store_or_snapshots(store, caplog) -> None:
    voice_agent = _load_voice_agent()
    captured = [
        logging.getLogger("caal.background_tasks"),
        logging.getLogger("caal.background_task_session"),
        logging.getLogger("voice-agent"),
    ]
    for target in captured:
        target.setLevel(logging.DEBUG)
        target.addHandler(caplog.handler)
    conversation_id = conversation_ledger.open_conversation(session_key=WEB_ROOM)
    try:
        web_bridge, web_execute, _, task = await _web_session_with_running_task(conversation_id)
        config = _outbound_config(conversation_id)
        phone_bridge = BackgroundTaskBridge(execute=FakeExecute(), session_key=PHONE_ROOM)
        phone_bridge.bind_conversation(config.conversation_id)
        await phone_bridge.start()
        handler, _, _ = _phone_leg(voice_agent, phone_bridge, config)
        handler.on_final_transcript(CALLBACK_REQUEST)
        await _run_turn(handler, CALLBACK_REQUEST, [])
        await _settle()
        web_execute.gate.set()
        await _teardown(web_bridge, phone_bridge)
    finally:
        for target in captured:
            target.removeHandler(caplog.handler)

    assert caplog.text
    assert conversation_id not in caplog.text
    for snapshot in list_tasks():
        assert conversation_id not in repr(snapshot)
        assert snapshot.session_key != conversation_id
    assert phone_bridge.owner_key == web_bridge.owner_key
    assert conversation_id not in phone_bridge.owner_key
    # The queue's own tables never hold the ledger id (the ledger's do, by design).
    dump = _raw_dump(store)
    queue_rows = [line for line in dump.splitlines() if "background_" in line]
    assert queue_rows
    assert all(conversation_id not in line for line in queue_rows)


# --- ownership adoption on the outbound leg ----------------------------------


async def _answer_outbound_call(monkeypatch, *, category: str, config, bridge):
    voice_agent = _load_voice_agent()
    monkeypatch.setattr(voice_agent.agents, "AMD", _FakeAMDFactory(category))
    ctx = FakeContext(room_name=PHONE_ROOM, metadata=_dispatch_metadata(config.conversation_id))
    recorder = ConversationRecorder()
    answered = await voice_agent.run_outbound_call(
        ctx, object(), config, agent=None, recorder=recorder, background=bridge
    )
    return answered, recorder


@pytest.mark.asyncio
async def test_human_answer_adopts_the_conversations_background_work(
    monkeypatch, outbound_env, store
) -> None:
    conversation_id = conversation_ledger.open_conversation(session_key=WEB_ROOM)
    conversation_ledger.append_turn(conversation_id, "user", "let's plan Lisbon")
    conversation_ledger.link_continuation(conversation_id, session_key=ATTEMPT_ID)
    web_bridge, web_execute, _, task = await _web_session_with_running_task(conversation_id)
    phone_bridge = BackgroundTaskBridge(execute=FakeExecute(), session_key=PHONE_ROOM)
    try:
        assert phone_bridge.can_arm_callback is False
        answered, recorder = await _answer_outbound_call(
            monkeypatch,
            category="human",
            config=_outbound_config(conversation_id),
            bridge=phone_bridge,
        )
        assert answered is True
        assert recorder.conversation_id == conversation_id
        assert phone_bridge.can_arm_callback is True
        assert phone_bridge.owner_key == web_bridge.owner_key
    finally:
        web_execute.gate.set()
        await _teardown(web_bridge, phone_bridge)


@pytest.mark.parametrize("category", ["machine-vm", "machine-ivr", "uncertain"])
@pytest.mark.asyncio
async def test_non_human_answers_adopt_nothing(monkeypatch, outbound_env, store, category) -> None:
    conversation_id = conversation_ledger.open_conversation(session_key=WEB_ROOM)
    conversation_ledger.link_continuation(conversation_id, session_key=ATTEMPT_ID)
    web_bridge, web_execute, _, _ = await _web_session_with_running_task(conversation_id)
    phone_bridge = BackgroundTaskBridge(execute=FakeExecute(), session_key=PHONE_ROOM)
    try:
        answered, _ = await _answer_outbound_call(
            monkeypatch,
            category=category,
            config=_outbound_config(conversation_id),
            bridge=phone_bridge,
        )
        assert answered is False
        assert phone_bridge.can_arm_callback is False
        assert phone_bridge.owner_key != web_bridge.owner_key
    finally:
        web_execute.gate.set()
        await _teardown(web_bridge, phone_bridge)


@pytest.mark.asyncio
async def test_a_call_that_continues_nothing_keeps_its_room_as_owner(
    monkeypatch, outbound_env, store
) -> None:
    phone_bridge = BackgroundTaskBridge(execute=FakeExecute(), session_key=PHONE_ROOM)
    answered, _ = await _answer_outbound_call(
        monkeypatch, category="human", config=_outbound_config(None), bridge=phone_bridge
    )
    assert answered is True
    assert phone_bridge.owner_key == PHONE_ROOM


def test_owner_key_is_stable_opaque_and_falls_back_to_the_room() -> None:
    from caal.background_task_session import owner_key_for

    conversation_id = conversation_ledger.new_conversation_id()
    key = owner_key_for(conversation_id, room_key=WEB_ROOM)
    assert key == owner_key_for(conversation_id, room_key=PHONE_ROOM)
    assert key != owner_key_for(conversation_ledger.new_conversation_id(), room_key=WEB_ROOM)
    assert conversation_id not in key and WEB_ROOM not in key
    assert owner_key_for(None, room_key=WEB_ROOM) == WEB_ROOM
    with pytest.raises(ValueError):
        owner_key_for("not a ledger id!", room_key=WEB_ROOM)
    with pytest.raises(ValueError):
        owner_key_for(None, room_key="")


def test_callback_dialer_is_available_to_every_session(monkeypatch, outbound_env) -> None:
    """The callback is dialed by whichever session's runner settles the task."""
    voice_agent = _load_voice_agent()
    web_ctx = FakeContext(room_name=WEB_ROOM, metadata="{}")
    phone_ctx = FakeContext(room_name=PHONE_ROOM, metadata=_dispatch_metadata(None))
    assert voice_agent.build_background_callback_dialer(web_ctx) is not None
    assert voice_agent.build_background_callback_dialer(phone_ctx) is not None
    monkeypatch.setenv("CAAL_OUTBOUND_ALLOWED_DESTINATIONS", "not-a-number")
    assert voice_agent.build_background_callback_dialer(web_ctx) is None


def test_callback_arming_is_only_built_for_an_outbound_leg() -> None:
    voice_agent = _load_voice_agent()
    bridge = BackgroundTaskBridge(execute=FakeExecute(), session_key=WEB_ROOM)
    web_ctx = FakeContext(room_name=WEB_ROOM, metadata="{}")
    phone_ctx = FakeContext(room_name=PHONE_ROOM, metadata=_dispatch_metadata(None))
    config_source = lambda: _outbound_config(None)  # noqa: E731
    assert (
        voice_agent.build_callback_arming(
            web_ctx, session=FakeSession(), bridge=bridge, outbound_config=config_source
        )
        is None
    )
    assert (
        voice_agent.build_callback_arming(
            phone_ctx, session=FakeSession(), bridge=None, outbound_config=config_source
        )
        is None
    )
    assert (
        voice_agent.build_callback_arming(
            phone_ctx, session=FakeSession(), bridge=bridge, outbound_config=config_source
        )
        is not None
    )


# --- exact voice commands consume their turn --------------------------------


def _command_handler(session: FakeSession):
    voice_agent = _load_voice_agent()
    end_call, arm = FakeEndCall(), FakeArm()
    handler = voice_agent.LocalTurnHandler(
        phone_handoff=None, session=session, end_call=end_call, arm_callback_and_end_call=arm
    )
    return handler, end_call, arm


@pytest.mark.parametrize(("text", "expected"), [("hang up", "end"), (CALLBACK_REQUEST, "arm")])
@pytest.mark.asyncio
async def test_spoken_exact_command_never_falls_through_to_hermes(text, expected) -> None:
    session = FakeSession()
    handler, end_call, arm = _command_handler(session)
    reached_llm: list[str] = []

    handler.on_final_transcript(text)
    await _run_turn(handler, text, reached_llm)
    await asyncio.sleep(0)

    assert reached_llm == []
    assert (end_call.calls, arm.calls) == ((1, 0) if expected == "end" else (0, 1))
    # The next turn is ordinary conversation again.
    await _run_turn(handler, "what time is it", reached_llm)
    assert reached_llm == ["what time is it"]
    assert (end_call.calls, arm.calls) == ((1, 0) if expected == "end" else (0, 1))


@pytest.mark.parametrize(("text", "expected"), [("hang up", "end"), (CALLBACK_REQUEST, "arm")])
@pytest.mark.asyncio
async def test_typed_exact_command_never_falls_through_to_hermes(text, expected) -> None:
    session = FakeSession()
    handler, end_call, arm = _command_handler(session)
    reached_llm: list[str] = []

    await _run_turn(handler, text, reached_llm)
    await asyncio.sleep(0)

    assert reached_llm == []
    assert (end_call.calls, arm.calls) == ((1, 0) if expected == "end" else (0, 1))


@pytest.mark.asyncio
async def test_late_final_transcript_with_a_callback_command_is_consumed_once() -> None:
    """The turn commits before the slow STT final; the gate still lets the command win."""
    session = FakeSession()
    handler, end_call, arm = _command_handler(session)
    reached_llm: list[str] = []

    handler.on_speech_transcript_started()
    committed_turn = asyncio.create_task(_run_turn(handler, "", reached_llm))
    await asyncio.sleep(0)
    handler.on_final_transcript(CALLBACK_REQUEST)
    await committed_turn
    await asyncio.sleep(0)

    assert reached_llm == []
    assert (end_call.calls, arm.calls) == (0, 1)
    assert session.spoken == []


@pytest.mark.asyncio
async def test_callback_command_without_an_armer_still_reaches_hermes() -> None:
    voice_agent = _load_voice_agent()
    session, end_call = FakeSession(), FakeEndCall()
    handler = voice_agent.LocalTurnHandler(phone_handoff=None, session=session, end_call=end_call)
    reached_llm: list[str] = []

    handler.on_final_transcript(CALLBACK_REQUEST)
    await _run_turn(handler, CALLBACK_REQUEST, reached_llm)

    assert reached_llm == [CALLBACK_REQUEST]
    assert end_call.calls == 0
