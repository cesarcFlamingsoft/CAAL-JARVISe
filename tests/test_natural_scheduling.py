"""Recovering the schedule a natural voice request actually contained.

Cesar says "remind me in one minute to stretch". The local model calls
``reminders.create`` with a title and no ``due``, and until this existed the
reminder became a silent list item: stored, never announced, and no delivery
question asked. Nothing was wrong with the store; the schedule was simply
dropped between the words and the arguments.

The recovery reads the current user turn, which lives only in the memory of
the turn that is executing. It is never stored, never cached, never logged and
never sent to the escalation model, and it only ever yields a time the user
actually said in a form the bounded parser already accepts.
"""

from __future__ import annotations

import importlib
import logging
from types import SimpleNamespace

import pytest

from caal.tools import alarms_tools, natural_schedule, reminder_delivery, reminders_tools

ANA = "usr_" + "a" * 24
NOW = 1_000_000

UTTERANCE = "remind me in one minute to stretch"


@pytest.fixture
def store(monkeypatch, tmp_path):
    path = tmp_path / "assistant.sqlite3"
    monkeypatch.setattr(alarms_tools, "STORE_PATH", path)
    monkeypatch.setattr(reminders_tools, "STORE_PATH", path)
    monkeypatch.setattr(reminder_delivery, "STORE_PATH", path)
    return path


# --- the phrase orders a person actually uses --------------------------------------------


@pytest.mark.parametrize(
    "turn, title, seconds",
    [
        ("remind me in one minute to stretch", "stretch", 60),
        ("remind me to stretch in one minute", "stretch", 60),
        ("remind me in 2 hours to call the clinic", "call the clinic", 7200),
        ("hey, remind me to take the bread out in 10 minutes", "take the bread out", 600),
        ("can you remind me in an hour to water the plants", "water the plants", 3600),
    ],
)
def test_a_natural_request_recovers_its_own_time_and_words(turn, title, seconds):
    """Both orders, with the title missing entirely."""
    recovered = natural_schedule.recover(dict(), turn)

    assert recovered["due"] == natural_schedule.iso_duration(seconds)
    assert recovered["title"] == title


@pytest.mark.parametrize(
    "turn, seconds",
    [
        ("remind me in one minute to stretch", 60),
        ("remind me to stretch in one minute", 60),
    ],
)
def test_a_model_title_is_kept_and_only_the_time_is_recovered(turn, seconds):
    recovered = natural_schedule.recover(dict(title="stretch"), turn)

    assert recovered["title"] == "stretch"
    assert recovered["due"] == natural_schedule.iso_duration(seconds)


def test_a_title_that_is_only_time_wording_is_replaced_by_the_request():
    recovered = natural_schedule.recover(dict(title="in one minute"), UTTERANCE)

    assert recovered["title"] == "stretch"
    assert recovered["due"] == natural_schedule.iso_duration(60)


def test_a_time_the_model_already_gave_is_never_second_guessed():
    recovered = natural_schedule.recover(dict(title="stretch", due="PT5M"), UTTERANCE)

    assert recovered["due"] == "PT5M"


@pytest.mark.parametrize(
    "turn",
    [
        "remind me to call mom later",
        "remind me to call mom in a bit",
        "remind me to call mom in a few minutes",
        "remind me to call mom tomorrow",
        "remind me to buy milk",
    ],
)
def test_a_vague_request_is_never_turned_into_a_time(turn):
    recovered = natural_schedule.recover(dict(title="call mom"), turn)

    assert recovered.get("due") is None


def test_two_different_times_in_one_breath_are_not_guessed_between():
    recovered = natural_schedule.recover(
        dict(title="call the clinic"), "remind me in 1 minute to call in 2 hours"
    )

    assert recovered.get("due") is None


def test_a_vague_request_becomes_an_undated_item_that_asks_for_a_clear_time(store):
    created = reminders_tools.create_reminder(title="Call mom", user_id=ANA, now=NOW)

    assert created["data"]["timed"] is False
    assert "will not alert you" in created["message"]
    assert "clear time" in created["message"].lower()


# --- the turn text lives only in this turn ------------------------------------------------


def test_the_current_turn_is_empty_outside_its_own_scope():
    assert natural_schedule.current_turn() == ""
    with natural_schedule.user_turn(UTTERANCE):
        assert natural_schedule.current_turn() == UTTERANCE
    assert natural_schedule.current_turn() == ""


def test_the_turn_is_released_even_when_the_turn_fails():
    with pytest.raises(RuntimeError):
        with natural_schedule.user_turn(UTTERANCE):
            raise RuntimeError("the room went away")
    assert natural_schedule.current_turn() == ""


def test_nothing_of_the_request_reaches_the_log(store, caplog):
    with caplog.at_level(logging.DEBUG):
        with natural_schedule.user_turn("remind me in one minute to see the divorce lawyer"):
            recovered = natural_schedule.recover(dict(), natural_schedule.current_turn())
            reminders_tools.create_reminder(user_id=ANA, now=NOW, **recovered)

    text = caplog.text
    for secret in ("divorce", "lawyer", "one minute", "PT1M", ANA):
        assert secret not in text


# --- the real native tool path ------------------------------------------------------------


class _Provider:
    """A local provider: it calls one tool, then streams a reply."""

    manages_own_tools = False

    def __init__(self, tool_calls):
        self._tool_calls = tool_calls
        self.turn_during_stream = None

    async def chat(self, messages, tools=None):
        return SimpleNamespace(tool_calls=self._tool_calls, content=None)

    async def chat_stream(self, messages, tools=None):
        self.turn_during_stream = natural_schedule.current_turn()
        yield "done"

    def format_tool_call_message(self, content, tool_calls):
        return dict(role="assistant", content=content or "")

    def format_tool_result(self, content, tool_call_id, tool_name):
        return dict(role="tool", content=content, tool_call_id=tool_call_id)


class _Hermes(_Provider):
    manages_own_tools = True


class ChatMessage:
    def __init__(self, role, text):
        self.role = role
        self.text_content = text
        self.id = "m1"


class _Agent:
    def __init__(self, user_id=ANA):
        from caal.user_scope import UserScope

        self._user_scope = UserScope.for_user(SimpleNamespace(user_id=user_id))
        self.published = []

    async def _on_scheduled_change(self):
        self.published.append(True)


def _chat_ctx(turn: str):
    return SimpleNamespace(
        items=[
            ChatMessage("system", "You are JARVIS."),
            ChatMessage("user", turn),
        ]
    )


async def _run(agent, provider, turn, cache=None):
    node = importlib.import_module("caal.llm.llm_node")
    chunks = []
    async for chunk in node.llm_node(agent, _chat_ctx(turn), provider, tool_data_cache=cache):
        chunks.append(chunk)
    return chunks


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "turn",
    ["remind me in one minute to stretch", "remind me to stretch in one minute"],
)
async def test_the_native_tool_path_recovers_the_schedule_for_both_orders(monkeypatch, store, turn):
    node = importlib.import_module("caal.llm.llm_node")
    monkeypatch.setattr(node.settings_module, "get_setting", lambda key, default=None: True)
    call = SimpleNamespace(name="reminders.create", arguments=dict(title="stretch"), id="1")

    await _run(_Agent(), _Provider([call]), turn)

    listed = reminders_tools.list_reminders(user_id=ANA)["data"]["reminders"]
    assert [item["title"] for item in listed] == ["stretch"]
    assert listed[0]["timed"] is True
    assert alarms_tools.pending_count(user_id=ANA) == 1


@pytest.mark.asyncio
async def test_the_turn_never_outlives_the_tool_execution(monkeypatch, store):
    """A created reminder is spoken by its own handler, so no stream is reached.

    The guarantee is stronger than it was: the recovered turn used to have to
    be out of scope by the time the follow-up stream ran, and now there is no
    follow-up stream at all. See :mod:`caal.llm.scheduled_reply`.
    """
    node = importlib.import_module("caal.llm.llm_node")
    monkeypatch.setattr(node.settings_module, "get_setting", lambda key, default=None: True)
    call = SimpleNamespace(name="reminders.create", arguments=dict(title="stretch"), id="1")
    provider = _Provider([call])

    await _run(_Agent(), provider, UTTERANCE)

    assert provider.turn_during_stream is None
    assert natural_schedule.current_turn() == ""
    assert natural_schedule.current_turn() == ""


@pytest.mark.asyncio
async def test_an_escalated_turn_never_enters_the_recovery_scope(monkeypatch, store):
    """Hermes runs its own tool loop: nothing here puts the turn anywhere it can see."""
    provider = _Hermes([])

    await _run(_Agent(), provider, UTTERANCE)

    assert provider.turn_during_stream == ""
    assert natural_schedule.current_turn() == ""


@pytest.mark.asyncio
async def test_the_recovered_request_is_kept_out_of_the_log(monkeypatch, store, caplog):
    node = importlib.import_module("caal.llm.llm_node")
    monkeypatch.setattr(node.settings_module, "get_setting", lambda key, default=None: True)
    call = SimpleNamespace(name="reminders.create", arguments=dict(), id="1")

    with caplog.at_level(logging.DEBUG):
        await _run(_Agent(), _Provider([call]), "remind me in one minute to see the oncologist")

    text = caplog.text
    for secret in ("oncologist", "one minute", "PT1M", ANA):
        assert secret not in text


@pytest.mark.asyncio
async def test_the_recovered_request_is_kept_out_of_the_tool_data_cache(monkeypatch, store):
    node = importlib.import_module("caal.llm.llm_node")
    monkeypatch.setattr(node.settings_module, "get_setting", lambda key, default=None: True)
    cache = node.ToolDataCache()
    call = SimpleNamespace(name="reminders.create", arguments=dict(), id="1")

    await _run(_Agent(), _Provider([call]), "remind me in one minute to see the oncologist", cache)

    context = cache.get_context_message() or ""
    assert "remind me in one minute" not in context
