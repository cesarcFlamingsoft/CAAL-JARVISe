"""Where the cross-language binding is *read*, and when it stops existing.

`test_company_expansion_runtime.py` pins which provider may translate and that
the binding is session-scoped. This file pins the two things an independent
probe found were still wrong once the feature met the real SDK:

* **a task that already existed when the session bound** -- the typed RoomIO
  ingress reader, started by the room's own `_listen_task` before
  `bind_session_expansion` ran -- carries a context copied *before* the bind,
  so it saw no translator at all. The repair is a verified, session-owned
  expander reference on the agent that the LLM/tool entry re-binds into
  whatever task it actually runs in, the same shape `company_privacy.begin_turn`
  already uses. It is **not** a process-global: the reference is read from the
  agent of that session and from nowhere else;
* **release did not invalidate an inherited handle.** A child task created
  while bound kept a live expander object after the parent released it, so the
  integration report's "no binding or cached question outlives the session"
  was not established. A released expander is closed: its cache is dropped and
  it translates nothing ever again, for any identity.

Nothing here reaches a network, a model, or a document.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caal.company import expansion_runtime, query_expansion  # noqa: E402

from test_company_expansion_runtime import FakeLocal  # noqa: E402

OWNER = "user-owner"
STRANGER = "user-stranger"


class _Scope:
    def __init__(self, user_id: str | None) -> None:
        self.user_id = user_id


class _Agent:
    """Only what the binding reads off a session's agent."""

    def __init__(self, user_id: str | None = OWNER) -> None:
        self._user_scope = _Scope(user_id)


@pytest.fixture(autouse=True)
def _unbound():
    query_expansion.reset()
    yield
    query_expansion.reset()


# --- the task that already existed when the session bound ---------------------------------


@pytest.mark.asyncio
async def test_a_task_created_before_the_bind_expands_once_the_entry_rebinds():
    """The reproduced defect: typed ingress starts before the session binds."""
    agent = _Agent()
    local = FakeLocal()
    ingress_ready = asyncio.Event()
    seen: dict[str, object] = {}

    async def pre_existing_ingress_task() -> None:
        # This task's context was copied before `bind_session_expansion` ran.
        await ingress_ready.wait()
        assert query_expansion.enabled() is False, "the ContextVar cannot reach here by itself"
        # The production LLM/tool entry re-binds from the agent it belongs to.
        expansion_runtime.rebind_session_expansion(agent)
        seen["enabled"] = query_expansion.enabled()
        seen["variants"] = await query_expansion.expand("plazo de preaviso", user_id=OWNER)

    task = asyncio.create_task(pre_existing_ingress_task())
    await asyncio.sleep(0)
    bound = expansion_runtime.bind_session_expansion(
        local, owner=agent, authorize=lambda uid: uid == OWNER
    )
    assert bound is not None
    ingress_ready.set()
    await task

    assert seen["enabled"] is True
    assert "notice period" in seen["variants"]
    assert len(local.calls) == 1


@pytest.mark.asyncio
async def test_the_rebind_is_the_session_own_expander_not_a_shared_one():
    agent = _Agent()
    expander = expansion_runtime.bind_session_expansion(FakeLocal(), owner=agent)
    query_expansion.reset()

    async def elsewhere() -> object:
        expansion_runtime.rebind_session_expansion(agent)
        return query_expansion.get_expander()

    assert await asyncio.create_task(elsewhere()) is expander


@pytest.mark.asyncio
async def test_an_agent_that_never_bound_rebinds_nothing():
    agent = _Agent()
    assert expansion_runtime.rebind_session_expansion(agent) is None
    assert query_expansion.enabled() is False


@pytest.mark.asyncio
async def test_a_rebind_never_carries_a_binding_across_a_different_session_agent():
    """The reference is verified against the session it was minted for."""
    owner_agent = _Agent(OWNER)
    expansion_runtime.bind_session_expansion(FakeLocal(), owner=owner_agent)
    # The same agent object, now naming a different verified user: a session's
    # identity changing under a binding must invalidate it, not reuse it.
    owner_agent._user_scope = _Scope(STRANGER)
    assert expansion_runtime.rebind_session_expansion(owner_agent) is None
    assert query_expansion.enabled() is False


@pytest.mark.asyncio
async def test_a_rebind_clears_an_inherited_binding_from_another_session():
    """An entry that rebinds must fail closed, not leave a foreign expander in place."""
    a_agent, b_agent = _Agent(OWNER), _Agent(OWNER)
    a_local = FakeLocal(reply='{"en": "session a", "es": "sesion a"}')
    expansion_runtime.bind_session_expansion(a_local, owner=a_agent)
    # B bound nothing (its endpoint is refused), and runs in a context that
    # inherited A's binding.
    expansion_runtime.bind_session_expansion(
        FakeLocal(base_url="http://8.8.8.8:11434"), owner=b_agent
    )
    query_expansion.bind_session(  # pretend the inherited context still has A's
        query_expansion.provider_translator(a_local)
    )
    assert expansion_runtime.rebind_session_expansion(b_agent) is None
    assert query_expansion.enabled() is False
    assert await query_expansion.expand("una pregunta", user_id=OWNER) == ["una pregunta"]
    assert a_local.calls == []


@pytest.mark.asyncio
async def test_two_live_sessions_rebinding_in_the_same_process_stay_separate():
    a_agent, b_agent = _Agent(OWNER), _Agent(OWNER)
    a_local = FakeLocal(reply='{"en": "session a", "es": "sesion a"}')
    b_local = FakeLocal(reply='{"en": "session b", "es": "sesion b"}')
    expansion_runtime.bind_session_expansion(a_local, owner=a_agent)
    expansion_runtime.bind_session_expansion(b_local, owner=b_agent)

    async def turn(agent: _Agent) -> list[str]:
        expansion_runtime.rebind_session_expansion(agent)
        return await query_expansion.expand("la misma pregunta", user_id=OWNER)

    a_variants, b_variants = await asyncio.gather(
        asyncio.create_task(turn(a_agent)), asyncio.create_task(turn(b_agent))
    )
    assert "session a" in a_variants and "session b" in b_variants
    assert len(a_local.calls) == 1 and len(b_local.calls) == 1


@pytest.mark.asyncio
async def test_a_rebind_does_not_bypass_the_per_turn_authorization():
    agent = _Agent()
    local = FakeLocal()
    allowed = {"now": False}
    expansion_runtime.bind_session_expansion(
        local, owner=agent, authorize=lambda _uid: allowed["now"]
    )

    async def turn() -> list[str]:
        expansion_runtime.rebind_session_expansion(agent)
        return await query_expansion.expand("plazo de preaviso", user_id=OWNER)

    assert await asyncio.create_task(turn()) == ["plazo de preaviso"]
    assert local.calls == []
    allowed["now"] = True
    assert "notice period" in await asyncio.create_task(turn())
    assert len(local.calls) == 1


# --- release closes the expander, for every task that ever saw it -------------------------


@pytest.mark.asyncio
async def test_release_invalidates_a_handle_a_child_task_already_inherited():
    agent = _Agent()
    local = FakeLocal()
    expander = expansion_runtime.bind_session_expansion(
        local, owner=agent, authorize=lambda uid: uid == OWNER
    )
    assert expander is not None
    ready = asyncio.Event()
    after: dict[str, object] = {}

    async def inheriting_child() -> None:
        await ready.wait()
        after["same_object"] = query_expansion.get_expander() is expander
        after["enabled"] = query_expansion.enabled()
        # Its own identity, which was authorized a moment ago.
        after["variants"] = await query_expansion.expand("plazo de preaviso", user_id=OWNER)

    child = asyncio.create_task(inheriting_child())
    await asyncio.sleep(0)
    expansion_runtime.release(agent)
    ready.set()
    await child

    assert after["enabled"] is False
    assert after["variants"] == ["plazo de preaviso"]
    assert local.calls == [], "a released session still reached the translator"


@pytest.mark.asyncio
async def test_release_drops_the_questions_that_were_cached():
    agent = _Agent()
    local = FakeLocal()
    expander = expansion_runtime.bind_session_expansion(local, owner=agent)
    assert expander is not None
    await query_expansion.expand("plazo de preaviso", user_id=OWNER)
    assert len(local.calls) == 1
    expansion_runtime.release(agent)
    assert expander.closed is True
    assert expander._cache == {}


@pytest.mark.asyncio
async def test_release_cannot_be_undone_by_rebinding_the_same_agent():
    agent = _Agent()
    local = FakeLocal()
    expansion_runtime.bind_session_expansion(local, owner=agent)
    expansion_runtime.release(agent)
    assert expansion_runtime.rebind_session_expansion(agent) is None
    assert await query_expansion.expand("plazo de preaviso", user_id=OWNER) == ["plazo de preaviso"]
    assert local.calls == []


@pytest.mark.asyncio
async def test_releasing_one_session_leaves_another_live_session_working():
    a_agent, b_agent = _Agent(OWNER), _Agent(OWNER)
    a_local = FakeLocal(reply='{"en": "session a", "es": "sesion a"}')
    b_local = FakeLocal(reply='{"en": "session b", "es": "sesion b"}')
    expansion_runtime.bind_session_expansion(a_local, owner=a_agent)
    expansion_runtime.bind_session_expansion(b_local, owner=b_agent)

    expansion_runtime.release(a_agent)

    async def b_turn() -> list[str]:
        expansion_runtime.rebind_session_expansion(b_agent)
        return await query_expansion.expand("una pregunta", user_id=OWNER)

    assert "session b" in await asyncio.create_task(b_turn())
    assert len(b_local.calls) == 1
    assert a_local.calls == []


@pytest.mark.asyncio
async def test_release_without_an_owner_still_unbinds_this_context():
    """The legacy call shape stays exactly what it was."""
    expansion_runtime.bind_session_expansion(FakeLocal())
    assert query_expansion.enabled() is True
    expansion_runtime.release()
    assert query_expansion.enabled() is False


# --- the production entry points actually call it ------------------------------------------


def _voice_agent_source() -> str:
    return (Path(__file__).resolve().parents[1] / "voice_agent.py").read_text()


def test_the_session_binds_with_its_own_agent_as_the_owner():
    source = _voice_agent_source()
    assert "owner=assistant" in source
    assert "company_expansion.release(assistant)" in source


def test_the_llm_node_rebinds_the_session_expander_before_it_generates():
    import ast

    tree = ast.parse(_voice_agent_source())
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef | ast.FunctionDef) and n.name == "llm_node"
    )
    called = {
        ast.unparse(child.func)
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
    }
    assert any("rebind_session_expansion" in name for name in called)


def test_the_local_command_chain_rebinds_it_too():
    import ast

    tree = ast.parse(_voice_agent_source())
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_handle_local_commands"
    )
    called = {
        ast.unparse(child.func)
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
    }
    assert any("rebind_session_expansion" in name for name in called)
