from __future__ import annotations


def test_memory_tools_are_registered_for_the_llm():
    from caal.tools.registry import create_default_registry

    registry = create_default_registry()

    assert registry.get("memory.remember").handler.__name__ == "remember"
    assert registry.get("memory.recall").handler.__name__ == "recall"


def test_preference_is_persisted_and_recalled(monkeypatch, tmp_path):
    from caal.tools import memory_tools

    monkeypatch.setattr(memory_tools, "STORE_PATH", tmp_path / "assistant.sqlite3")

    saved = memory_tools.remember("preferred coffee", "oat milk latte")
    recalled = memory_tools.recall("preferred coffee")

    assert saved["message"] == "I'll remember that preferred coffee is oat milk latte."
    assert recalled["message"] == "preferred coffee is oat milk latte."
    assert recalled["data"] == {"key": "preferred coffee", "value": "oat milk latte"}


def test_remember_updates_existing_preference_and_lists_memories(monkeypatch, tmp_path):
    from caal.tools import memory_tools

    monkeypatch.setattr(memory_tools, "STORE_PATH", tmp_path / "assistant.sqlite3")
    memory_tools.remember("timezone", "America/Edmonton")
    memory_tools.remember("timezone", "America/Calgary")

    memories = memory_tools.recall()

    assert memories["message"] == "I remember 1 preference."
    assert memories["data"]["memories"] == [
        {"key": "timezone", "value": "America/Calgary"}
    ]
