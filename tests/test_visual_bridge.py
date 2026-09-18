"""Ephemeral, explicit-only browser commands; descriptions never enter a model."""

import ast
import asyncio
import json
from pathlib import Path

import pytest

from caal.visual_bridge import VisualBridge


def setup_bridge():
    commands, answers = [], []

    async def send(packet, participant):
        commands.append((packet, participant))

    bridge = VisualBridge(user="usr_test", room="caal-web-test", send=send)
    ready = dict(action="vision.ready", user="usr_test", room="caal-web-test", epoch="a" * 32)
    bridge.receive(json.dumps(ready).encode(), "user-test")
    return bridge, commands, answers


@pytest.mark.asyncio
async def test_one_shot_bound_result_returns_to_the_tool_without_speaking():
    bridge, commands, answers = setup_bridge()
    assert bridge.available()
    task = asyncio.create_task(bridge.analyze())
    await asyncio.sleep(0)
    command, participant = commands[0]
    assert participant == "user-test"
    assert command["action"] == "vision.analyze"
    assert "prompt" not in command
    reply = {**command, "action": "vision.result", "description": "A blue mug."}
    bridge.receive(json.dumps(reply).encode(), "wrong-user")
    assert not task.done()
    bridge.receive(json.dumps(reply).encode(), participant)
    assert await task == "A blue mug."
    assert bridge.last_observation() == "A blue mug."
    bridge.receive(json.dumps(reply).encode(), participant)
    assert answers == []
    assert bridge.pending is None


@pytest.mark.asyncio
async def test_closed_or_unbound_camera_is_not_available_to_the_tool():
    bridge = VisualBridge(user=None, room="caal-web-test", send=lambda *_: None)
    assert not bridge.available()
    with pytest.raises(ValueError, match="vision_unavailable"):
        await bridge.analyze()
    bridge, _, _ = setup_bridge()
    bridge.close()
    assert not bridge.available()


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
async def test_close_cancels_an_in_progress_analysis():
    bridge, commands, _ = setup_bridge()
    task = asyncio.create_task(bridge.analyze())
    await asyncio.sleep(0)
    command, participant = commands[0]
    bridge.close()
    bridge.receive(
        json.dumps({**command, "action": "vision.result", "description": "private"}).encode(),
        participant,
    )
    with pytest.raises(ValueError, match="vision_unavailable"):
        await task


@pytest.mark.asyncio
async def test_visual_intent_reaches_the_llm_which_can_choose_the_camera_tool():
    from test_typed_chat_handoff import FakeEndCall, FakeSession, _load_voice_agent

    voice_agent = _load_voice_agent()
    bridge, commands, answers = setup_bridge()
    handler = voice_agent.LocalTurnHandler(
        phone_handoff=None,
        session=FakeSession(),
        end_call=FakeEndCall(),
        visual=bridge,
    )

    assert await handler.turn_consumed("What is this I am handling?") is False
    assert commands == []
    assert answers == []


@pytest.mark.asyncio
async def test_camera_tool_analysis_returns_a_bound_result_without_speaking_it():
    bridge, commands, answers = setup_bridge()
    task = asyncio.create_task(bridge.analyze())
    await asyncio.sleep(0)
    command, participant = commands[0]
    bridge.receive(
        json.dumps({**command, "action": "vision.result", "description": "A blue mug."}).encode(),
        participant,
    )
    assert await task == "A blue mug."
    assert answers == []


@pytest.mark.asyncio
async def test_invalid_results_never_resolve_pending_turn():
    bridge, commands, answers = setup_bridge()
    task = asyncio.create_task(bridge.analyze())
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
    with pytest.raises(ValueError, match="vision_unavailable"):
        await task


@pytest.mark.asyncio
async def test_participant_disconnect_clears_pending_and_rejects_late_result():
    bridge, commands, answers = setup_bridge()
    task = asyncio.create_task(bridge.analyze())
    await asyncio.sleep(0)
    command, participant = commands[0]
    bridge.disconnect(participant)
    bridge.receive(
        json.dumps({**command, "action": "vision.result", "description": "secret"}).encode(),
        participant,
    )
    with pytest.raises(ValueError, match="vision_unavailable"):
        await task
    assert bridge.pending is None
    assert answers == []


@pytest.mark.asyncio
async def test_camera_tool_is_discovered_after_the_camera_opens_even_if_cache_was_cold(monkeypatch):
    from importlib import import_module
    from types import SimpleNamespace

    llm_node = import_module("caal.llm.llm_node")

    monkeypatch.setattr(llm_node.settings_module, "get_setting", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        llm_node,
        "_tool_available",
        lambda agent, name: name == "web_search"
        or (name == "analyze_camera_view" and agent._visual_bridge.available()),
    )

    from livekit.agents import function_tool

    @function_tool
    async def analyze_camera_view(self):
        """Inspect the live camera."""

    @function_tool
    async def web_search(self, query: str):
        """Search the web."""

    class View:
        def __init__(self):
            self.live = False

        def available(self):
            return self.live

    view = View()
    agent = SimpleNamespace(
        _tools=[
            SimpleNamespace(__func__=analyze_camera_view),
            SimpleNamespace(__func__=web_search),
        ],
        _visual_bridge=view,
        _llm_tools_cache=None,
    )
    cold_tools = await llm_node._discover_tools(agent)
    assert [tool["function"]["name"] for tool in cold_tools] == ["web_search"]
    view.live = True
    tools = await llm_node._discover_tools(agent)
    assert [tool["function"]["name"] for tool in tools] == ["analyze_camera_view", "web_search"]


def test_entrypoint_keeps_json_as_the_module_serializer_for_visual_send():
    """A conditional local import makes json unbound in the earlier sender closure."""
    tree = ast.parse(Path("voice_agent.py").read_text())
    entry = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "entrypoint"
    )

    class DirectScopeImports(ast.NodeVisitor):
        names: set[str]

        def __init__(self):
            self.names = set()

        def visit_FunctionDef(self, node):
            return

        def visit_AsyncFunctionDef(self, node):
            return

        def visit_Import(self, node):
            self.names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)

        def visit_ImportFrom(self, node):
            self.names.update(alias.asname or alias.name for alias in node.names)

    finder = DirectScopeImports()
    for statement in entry.body:
        finder.visit(statement)
    assert "json" not in finder.names
