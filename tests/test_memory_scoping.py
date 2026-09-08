"""Per-user scoping of JARVIS's explicit memory.

The existing ``memory.remember`` / ``memory.recall`` tools are the memory
path; they gain a user scope rather than being replaced. A memory saved for
one user is invisible to every other user, the LLM cannot name a user scope
through tool arguments, an unidentified session under multi-user cannot touch
memory at all, and the legacy single-user rows survive the schema upgrade and
are adopted by the bootstrap administrator exactly once.
"""

from __future__ import annotations

import importlib
import sqlite3
from types import SimpleNamespace

import pytest

from caal.tools import memory_tools
from caal.tools.registry import create_default_registry
from caal.user_scope import (
    MEMORY_UNAVAILABLE_REPLY,
    UserScope,
    scoped_tool_arguments,
)

# ``caal.llm`` re-exports the ``llm_node`` *function*; load the module itself.
llm_node_module = importlib.import_module("caal.llm.llm_node")

ANA = "usr_" + "a" * 24
BO = "usr_" + "b" * 24


@pytest.fixture(autouse=True)
def store(monkeypatch, tmp_path):
    path = tmp_path / "assistant.sqlite3"
    monkeypatch.setattr(memory_tools, "STORE_PATH", path)
    return path


# --- store scoping -----------------------------------------------------------------


def test_memories_are_private_to_the_user_that_saved_them() -> None:
    memory_tools.remember("coffee", "oat milk latte", user_id=ANA)
    memory_tools.remember("coffee", "black, no sugar", user_id=BO)

    assert memory_tools.recall("coffee", user_id=ANA)["data"]["value"] == "oat milk latte"
    assert memory_tools.recall("coffee", user_id=BO)["data"]["value"] == "black, no sugar"
    assert memory_tools.recall("coffee", user_id="usr_" + "c" * 24)["status"] == "not_found"
    assert [m["key"] for m in memory_tools.recall(user_id=ANA)["data"]["memories"]] == ["coffee"]
    assert memory_tools.recall(user_id="usr_" + "c" * 24)["data"]["memories"] == []


def test_legacy_single_user_scope_is_separate_from_every_user() -> None:
    memory_tools.remember("timezone", "America/Edmonton")  # legacy: no user

    assert memory_tools.recall("timezone")["data"]["value"] == "America/Edmonton"
    assert memory_tools.recall("timezone", user_id=ANA)["status"] == "not_found"
    assert memory_tools.recall(user_id=ANA)["data"]["memories"] == []


@pytest.mark.parametrize("bad", ["", "usr_x", "ana", "usr_" + "A" * 24, 42, "' OR 1=1 --"])
def test_malformed_user_scopes_are_rejected_before_any_query(bad) -> None:
    with pytest.raises(ValueError):
        memory_tools.remember("k", "v", user_id=bad)
    with pytest.raises(ValueError):
        memory_tools.recall("k", user_id=bad)


def test_keys_and_values_are_bounded_and_plain_text() -> None:
    with pytest.raises(ValueError):
        memory_tools.remember("k" * 200, "v", user_id=ANA)
    with pytest.raises(ValueError):
        memory_tools.remember("k", "v" * 5000, user_id=ANA)
    with pytest.raises(ValueError):
        memory_tools.remember("bad\x00key", "v", user_id=ANA)
    with pytest.raises(ValueError):
        memory_tools.remember("k", "", user_id=ANA)


def test_existing_legacy_table_is_upgraded_in_place_and_rows_survive(store) -> None:
    with sqlite3.connect(store) as connection:
        connection.execute(
            "CREATE TABLE preferences (key TEXT PRIMARY KEY, value TEXT NOT NULL, "
            "updated_at INTEGER NOT NULL DEFAULT (unixepoch()))"
        )
        connection.execute("INSERT INTO preferences (key, value) VALUES ('coffee', 'flat white')")

    assert memory_tools.recall("coffee")["data"]["value"] == "flat white"
    assert memory_tools.recall("coffee", user_id=ANA)["status"] == "not_found"
    memory_tools.remember("coffee", "cortado", user_id=ANA)
    assert memory_tools.recall("coffee")["data"]["value"] == "flat white"

    with sqlite3.connect(store) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(preferences)")}
    assert {"user_id", "key", "value", "updated_at"} <= columns


def test_legacy_memories_can_be_adopted_by_exactly_one_user_once() -> None:
    memory_tools.remember("coffee", "flat white")
    memory_tools.remember("timezone", "America/Edmonton")
    memory_tools.remember("coffee", "cortado", user_id=ANA)  # would clash on key

    adopted = memory_tools.adopt_legacy_memories(ANA)

    assert adopted == 2
    assert memory_tools.recall("timezone", user_id=ANA)["data"]["value"] == "America/Edmonton"
    # The user's own newer value wins over the legacy one on a key clash.
    assert memory_tools.recall("coffee", user_id=ANA)["data"]["value"] == "cortado"
    assert memory_tools.recall()["data"]["memories"] == []
    assert memory_tools.adopt_legacy_memories(ANA) == 0
    assert memory_tools.adopt_legacy_memories(BO) == 0
    with pytest.raises(ValueError):
        memory_tools.adopt_legacy_memories("nope")


def test_tool_results_never_carry_the_user_id() -> None:
    saved = memory_tools.remember("coffee", "oat milk latte", user_id=ANA)
    recalled = memory_tools.recall("coffee", user_id=ANA)
    listed = memory_tools.recall(user_id=ANA)

    for result in (saved, recalled, listed):
        assert ANA not in str(result)


# --- scope object ----------------------------------------------------------------------


def test_user_scope_states() -> None:
    legacy = UserScope.legacy()
    anonymous = UserScope.anonymous()
    ana = UserScope.for_user(SimpleNamespace(user_id=ANA, display_name="Ana", role="member"))

    assert legacy.user_id is None and legacy.identity_configured is False
    assert legacy.memory_available is True
    assert anonymous.user_id is None and anonymous.identity_configured is True
    assert anonymous.memory_available is False
    assert ana.user_id == ANA and ana.identity_configured is True and ana.memory_available
    assert ana.role == "member"
    assert "Ana" not in repr(ana)


def test_registry_marks_memory_tools_as_user_scoped_and_nothing_else() -> None:
    registry = create_default_registry()

    scoped = {tool.name for tool in registry.list() if tool.user_scoped}

    assert scoped == {"memory.remember", "memory.recall"}
    for name in ("memory.remember", "memory.recall"):
        assert "user_id" not in registry.get(name).parameters["properties"]


def test_scoped_arguments_strip_llm_supplied_user_ids_and_bind_the_session_user() -> None:
    registry = create_default_registry()
    tool = registry.get("memory.remember")
    llm_args = {"key": "coffee", "value": "latte", "user_id": BO}  # attempt to write as Bo

    bound = scoped_tool_arguments(tool, llm_args, UserScope.for_user(SimpleNamespace(user_id=ANA)))

    assert bound == {"key": "coffee", "value": "latte", "user_id": ANA}
    assert scoped_tool_arguments(tool, llm_args, UserScope.legacy()) == {
        "key": "coffee",
        "value": "latte",
        "user_id": None,
    }
    assert scoped_tool_arguments(tool, llm_args, UserScope.anonymous()) is None
    # Unscoped tools are passed through untouched.
    plain = registry.get("alarms.set")
    assert scoped_tool_arguments(plain, {"label": "x"}, UserScope.anonymous()) == {"label": "x"}


# --- dispatch through the LLM node -----------------------------------------------------


class _Agent:
    def __init__(self, scope: UserScope | None) -> None:
        if scope is not None:
            self._user_scope = scope


@pytest.mark.asyncio
async def test_llm_dispatch_binds_memory_to_the_session_user_only(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_node_module.settings_module, "get_setting", lambda key, default=None: True
    )
    ana_agent = _Agent(UserScope.for_user(SimpleNamespace(user_id=ANA)))
    bo_agent = _Agent(UserScope.for_user(SimpleNamespace(user_id=BO)))

    await llm_node_module._execute_single_tool(
        ana_agent, "memory.remember", {"key": "coffee", "value": "latte", "user_id": BO}
    )

    assert memory_tools.recall("coffee", user_id=ANA)["data"]["value"] == "latte"
    assert memory_tools.recall("coffee", user_id=BO)["status"] == "not_found"
    recalled = await llm_node_module._execute_single_tool(
        bo_agent, "memory.recall", {"key": "coffee", "user_id": ANA}
    )
    assert recalled["status"] == "not_found"


@pytest.mark.asyncio
async def test_llm_dispatch_refuses_memory_for_unidentified_sessions_under_multi_user(
    monkeypatch, store
) -> None:
    monkeypatch.setattr(
        llm_node_module.settings_module, "get_setting", lambda key, default=None: True
    )
    anonymous = _Agent(UserScope.anonymous())

    result = await llm_node_module._execute_single_tool(
        anonymous, "memory.remember", {"key": "coffee", "value": "latte"}
    )

    assert result["status"] == "unauthorized"
    assert result["message"] == MEMORY_UNAVAILABLE_REPLY
    with sqlite3.connect(store) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
    assert "preferences" not in tables  # never even opened the store


@pytest.mark.asyncio
async def test_llm_dispatch_without_a_scope_keeps_legacy_single_user_memory(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_node_module.settings_module, "get_setting", lambda key, default=None: True
    )

    await llm_node_module._execute_single_tool(
        _Agent(None), "memory.remember", {"key": "coffee", "value": "latte"}
    )

    assert memory_tools.recall("coffee")["data"]["value"] == "latte"
    assert memory_tools.recall("coffee", user_id=ANA)["status"] == "not_found"
