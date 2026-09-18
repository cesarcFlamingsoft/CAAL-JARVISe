"""Ephemeral, explicit-only browser commands; descriptions never enter a model."""

import asyncio
import json
from pathlib import Path

import pytest

from caal.visual_bridge import VisualBridge, explicit_visual_question


@pytest.mark.parametrize(
    "text",
    [
        "What is this I am handling?",
        "What am I holding?",
        "Can you describe what you see?",
        "What's in my hand?",
        "Analyze the current camera view.",
    ],
)
def test_explicit_questions(text):
    assert explicit_visual_question(text)


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        "I am handling a problem",
        "What is this?",
        "look up cameras",
        "Don't analyze the camera view",
        "Explain what is this I am handling in Python",
        "What am I holding tomorrow?",
        "Read my private contract",
        "Vision is open",
    ],
)
def test_unrelated_or_ambiguous_text(text):
    assert not explicit_visual_question(text)


def setup_bridge():
    commands, answers = [], []

    async def send(packet, participant):
        commands.append((packet, participant))

    async def speak(text):
        answers.append(text)

    bridge = VisualBridge(user="usr_test", room="caal-web-test", send=send, speak=speak)
    ready = dict(action="vision.ready", user="usr_test", room="caal-web-test", epoch="a" * 32)
    bridge.receive(json.dumps(ready).encode(), "user-test")
    return bridge, commands, answers


@pytest.mark.asyncio
async def test_one_shot_bound_result_and_no_normal_chat_command():
    bridge, commands, answers = setup_bridge()
    assert not await bridge.handle("hello")
    task = asyncio.create_task(bridge.handle("What is this I am handling?"))
    await asyncio.sleep(0)
    command, participant = commands[0]
    assert participant == "user-test"
    assert command["action"] == "vision.analyze"
    assert "prompt" not in command
    reply = {**command, "action": "vision.result", "description": "A blue mug."}
    bridge.receive(json.dumps(reply).encode(), "wrong-user")
    assert not task.done()
    bridge.receive(json.dumps(reply).encode(), participant)
    assert await task
    bridge.receive(json.dumps(reply).encode(), participant)
    assert answers == ["A blue mug."]
    assert bridge.pending is None


@pytest.mark.asyncio
async def test_company_closed_stale_and_cancel():
    bridge, commands, answers = setup_bridge()
    assert await bridge.handle("What am I holding?", company=True)
    assert commands == []
    task = asyncio.create_task(bridge.handle("What am I holding?"))
    await asyncio.sleep(0)
    packet, participant = commands[0]
    bridge.cancel()
    bridge.receive(
        json.dumps({**packet, "action": "vision.result", "description": "secret"}).encode(),
        participant,
    )
    assert await task
    assert "secret" not in answers
    bridge.close()
    await bridge.handle("What am I holding?")
    assert [packet["action"] for packet, _ in commands] == ["vision.analyze", "vision.cancel"]


def test_source_boundaries():
    root = Path(__file__).parents[1]
    source = (root / "src/caal/visual_bridge.py").read_text()
    for forbidden in (
        "logger",
        "generate_reply",
        "chat_ctx",
        "conversation_ledger",
        "hermes",
        "ollama",
        "memory",
        "images",
    ):
        assert forbidden not in source.lower()
    agent = (root / "voice_agent.py").read_text()
    assert "visual=visual_bridge" in agent
    assert "visual_bridge.receive" in agent
    assert "visual_bridge.close()" in agent
    for path in (root / "src/caal/llm/agent_tools.py", root / "prompt/default.md"):
        text = path.read_text().lower()
        for forbidden in ("vision.analyze", "visual_bridge", "gemma4", "visual_speech"):
            assert forbidden not in text


@pytest.mark.asyncio
async def test_close_cancels_private_speech_already_in_progress():
    bridge, commands, _ = setup_bridge()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def speak(text):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    bridge._speak = speak
    task = asyncio.create_task(bridge.handle("What am I holding?"))
    await asyncio.sleep(0)
    packet, participant = commands[0]
    bridge.receive(
        json.dumps({**packet, "action": "vision.result", "description": "private"}).encode(),
        participant,
    )
    await started.wait()
    bridge.close()
    await asyncio.sleep(0)
    try:
        assert cancelled.is_set()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("spoken", [False, True])
async def test_actual_local_turn_hook_consumes_visual_without_model_or_duplicate(spoken):
    from test_typed_chat_handoff import (
        FakeEndCall,
        FakeSession,
        _load_voice_agent,
        _run_turn,
    )

    voice_agent = _load_voice_agent()
    bridge, commands, answers = setup_bridge()
    handler = voice_agent.LocalTurnHandler(
        phone_handoff=None,
        session=FakeSession(),
        end_call=FakeEndCall(),
        visual=bridge,
    )
    text = "What is this I am handling?"
    if spoken:
        handler.on_final_transcript(text)
    reached_model = []
    turn = asyncio.create_task(_run_turn(handler, text, reached_model))
    for _ in range(10):
        await asyncio.sleep(0)
        if commands:
            break
    command, participant = commands[0]
    bridge.receive(
        json.dumps({**command, "action": "vision.result", "description": "A mug."}).encode(),
        participant,
    )
    await turn
    assert reached_model == []
    assert answers == ["A mug."]
    assert len(commands) == 1
    assert not await handler.turn_consumed("hello")


@pytest.mark.asyncio
async def test_invalid_results_never_resolve_pending_turn():
    bridge, commands, answers = setup_bridge()
    task = asyncio.create_task(bridge.handle("What am I holding?"))
    await asyncio.sleep(0)
    command, participant = commands[0]
    reply = {**command, "action": "vision.result", "description": "A mug."}
    for patch in [
        {"user": "wrong"},
        {"room": "wrong"},
        {"epoch": "b" * 32},
        {"seq": 0},
        {"description": "x" * 1201},
        {"description": "secret\n"},
        {"extra": True},
    ]:
        bridge.receive(json.dumps({**reply, **patch}).encode(), participant)
    bridge.receive(b"bad json", participant)
    assert answers == []
    assert not task.done()
    bridge.close()
    assert await task


@pytest.mark.asyncio
async def test_participant_disconnect_clears_pending_and_rejects_late_result():
    bridge, commands, answers = setup_bridge()
    task = asyncio.create_task(bridge.handle("What am I holding?"))
    await asyncio.sleep(0)
    command, participant = commands[0]
    bridge.disconnect(participant)
    bridge.receive(
        json.dumps({**command, "action": "vision.result", "description": "secret"}).encode(),
        participant,
    )
    assert await task
    assert bridge.pending is None
    assert answers == []


@pytest.mark.parametrize(
    "text",
    [
        "Friday, what can you see?",
        "Can you tell me what I'm holding?",
        "What does this look like?",
        "What is in front of me?",
        "What am I holding in my hand?",
        "What am I holding on my hand?",
        "Friday, can you see what I have in my hand?",
    ],
)
def test_natural_explicit_variations_need_no_special_phrase(text):
    assert explicit_visual_question(text)
