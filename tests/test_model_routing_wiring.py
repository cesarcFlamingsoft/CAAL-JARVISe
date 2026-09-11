"""The runtime actually wires the routing: local main model, Hermes for code.

These are the seams between the pieces: the settings a deployment reads, the
coding worker built from them, and the background bridge that queues a coding
turn for the Hermes agent runtime instead of for a chat model or a local
process.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from caal import background_tasks
from caal.background_task_session import CODING_ACK_REPLY
from caal.coding_delegation import HermesCodingDelegate

BRIDGE_RUNTIME = dict(background_tasks_enabled=True, work_router_enabled=False)


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setattr(background_tasks, "STORE_PATH", tmp_path / "assistant.sqlite3")
    yield


@pytest.fixture(scope="module")
def voice_agent():
    module_path = Path(__file__).parents[1] / "voice_agent.py"
    spec = importlib.util.spec_from_file_location("voice_agent_routing", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StubSession:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def say(self, text: str, **_: object) -> None:
        self.spoken.append(text)


class StubHermes:
    """The agent runtime. Nothing in these tests may actually call it."""

    provider_name = "hermes"
    manages_own_tools = True

    async def chat(self, messages, **_: object):
        raise AssertionError("no test here should reach the agent runtime")


class StubLocal:
    provider_name = "ollama"
    manages_own_tools = False

    async def chat(self, messages, **_: object):
        raise AssertionError("a coding turn must never reach the local chat model")


def routed(escalation):
    from caal.llm.providers import RoutedProvider

    return RoutedProvider(primary=StubLocal(), escalation=escalation)


# --- settings -----------------------------------------------------------------


def test_the_runtime_defaults_to_the_local_model_with_coding_delegation(
    voice_agent, monkeypatch
) -> None:
    monkeypatch.setattr(voice_agent.settings_module, "load_settings", dict)
    monkeypatch.setattr(voice_agent.settings_module, "load_user_settings", dict)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    runtime = voice_agent.get_runtime_settings()

    assert runtime["llm_provider"] == "routed"
    assert runtime["coding_delegation_enabled"] is True
    assert runtime["coding_delegation_timeout_seconds"] == 900


def test_the_runtime_names_no_workspace_command_or_cli(voice_agent, monkeypatch) -> None:
    """Coding runs in Hermes, so no path or command is configurable here at all."""
    monkeypatch.setattr(voice_agent.settings_module, "load_settings", dict)
    monkeypatch.setattr(voice_agent.settings_module, "load_user_settings", dict)

    runtime = voice_agent.get_runtime_settings()

    assert not [key for key in runtime if "workdir" in key or "cli_path" in key]


# --- the coding worker --------------------------------------------------------


def test_the_coding_worker_is_the_hermes_delegate(voice_agent) -> None:
    worker = voice_agent.build_coding_delegate(dict(), provider=routed(StubHermes()))

    assert isinstance(worker, HermesCodingDelegate)


def test_without_hermes_there_is_no_coding_worker(voice_agent) -> None:
    assert voice_agent.build_coding_delegate(dict(), provider=routed(None)) is None


# --- the bridge ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_session_bridge_queues_a_coding_turn_for_hermes(voice_agent, store) -> None:
    bridge = voice_agent.build_background_task_bridge(
        dict(BRIDGE_RUNTIME), provider=routed(StubHermes()), session_key="room-wiring"
    )
    assert bridge is not None
    session = StubSession()

    # The runner is never started, so nothing executes: this asserts the routing
    # decision and the spoken acknowledgement, not a real Hermes turn.
    outcome = await bridge.process_turn("fix the retry bug in the ollama provider", session)

    assert outcome.consumed is True
    assert outcome.scheduled is True
    assert session.spoken == [CODING_ACK_REPLY]


@pytest.mark.asyncio
async def test_without_hermes_a_coding_turn_is_left_to_the_ordinary_path(
    voice_agent, store
) -> None:
    bridge = voice_agent.build_background_task_bridge(
        dict(BRIDGE_RUNTIME), provider=routed(None), session_key="room-wiring-2"
    )
    assert bridge is not None
    session = StubSession()

    outcome = await bridge.process_turn("fix the retry bug in the ollama provider", session)

    assert outcome.consumed is False
    assert session.spoken == []


# --- which provider each job runs on -----------------------------------------


def test_the_turn_classifier_always_runs_on_the_local_model(voice_agent) -> None:
    """Routing the classifier could turn a one-word label into a full agent turn."""
    from caal.llm.providers import RoutedProvider

    local, harness = object(), object()
    provider = RoutedProvider(primary=local, escalation=harness)

    class CAALLLMLike:
        provider_instance = provider

    assert voice_agent.work_router_provider(provider) is local
    assert voice_agent.work_router_provider(CAALLLMLike()) is local


def test_background_work_runs_on_the_agent_harness_when_there_is_one(voice_agent) -> None:
    """Queued work is the long multi-step kind, so it goes to Hermes directly."""
    from caal.llm.providers import RoutedProvider

    local, harness = object(), object()
    assert (
        voice_agent.background_worker_provider(RoutedProvider(primary=local, escalation=harness))
        is harness
    )
    only_local = RoutedProvider(primary=local, escalation=None)
    assert voice_agent.background_worker_provider(only_local) is only_local
