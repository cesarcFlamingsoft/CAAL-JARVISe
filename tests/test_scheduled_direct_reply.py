"""Speaking a scheduled outcome without asking a model to describe it.

A reminder that was actually created is the end of the work, not the middle of
it. Until this existed the runtime stored the reminder, armed it, and then
asked the local model to narrate what had just happened from a tool-call
history. The effective local model returns zero chunks for that particular
shape of continuation, the routed provider then fell through to the escalation
runtime, and the person heard "no model available" about a reminder that was
already set and already armed.

The handler already writes the one sentence worth saying -- including the
delivery-choice question -- so that sentence is spoken as it is. No follow-up
stream, no escalation, no second description of the same fact, and no private
scheduling data handed to anything that did not already have it.
"""

from __future__ import annotations

import importlib
import logging
from types import SimpleNamespace

import pytest

from caal import scheduled_events
from caal.tools import alarms_tools, reminder_delivery, reminders_tools

ANA = "usr_" + "a" * 24


@pytest.fixture
def store(monkeypatch, tmp_path):
    path = tmp_path / "assistant.sqlite3"
    monkeypatch.setattr(alarms_tools, "STORE_PATH", path)
    monkeypatch.setattr(reminders_tools, "STORE_PATH", path)
    monkeypatch.setattr(reminder_delivery, "STORE_PATH", path)
    return path


class _Provider:
    """A local provider whose follow-up stream is exactly the failure we saw.

    ``chat_stream`` raises, so any test that still reaches it fails loudly
    rather than quietly yielding nothing.
    """

    manages_own_tools = False

    def __init__(self, tool_calls, chunks=("model said something",)):
        self._tool_calls = tool_calls
        self._chunks = chunks
        self.stream_calls = 0
        self.streamed_messages: list[list[dict]] = []

    async def chat(self, messages, tools=None):
        return SimpleNamespace(tool_calls=self._tool_calls, content=None)

    async def chat_stream(self, messages, tools=None):
        self.stream_calls += 1
        self.streamed_messages.append(list(messages))
        raise RuntimeError("the local model returned no chunks for a tool history")
        yield ""  # pragma: no cover - unreachable, keeps this an async generator

    def format_tool_call_message(self, content, tool_calls):
        return dict(role="assistant", content=content or "")

    def format_tool_result(self, content, tool_call_id, tool_name):
        return dict(role="tool", content=content, tool_call_id=tool_call_id)


class _StreamingProvider(_Provider):
    """The ordinary path: a tool result the model is still asked to describe."""

    async def chat_stream(self, messages, tools=None):
        self.stream_calls += 1
        self.streamed_messages.append(list(messages))
        for chunk in self._chunks:
            yield chunk


class ChatMessage:
    def __init__(self, role, text):
        self.role = role
        self.text_content = text
        self.id = "m1"


class _Agent:
    def __init__(self, user_id=ANA):
        from caal.user_scope import UserScope

        self._user_scope = UserScope.for_user(SimpleNamespace(user_id=user_id))
        self.published: list[str] = []

    async def _on_scheduled_change(self) -> None:
        self.published.append(scheduled_events.PAYLOAD)


def _call(name, arguments, call_id="1"):
    return SimpleNamespace(name=name, arguments=arguments, id=call_id)


def _chat_ctx(turn: str):
    return SimpleNamespace(
        items=[ChatMessage("system", "You are JARVIS."), ChatMessage("user", turn)]
    )


async def _run(agent, provider, turn="", cache=None, monkeypatch=None):
    node = importlib.import_module("caal.llm.llm_node")
    if monkeypatch is not None:
        monkeypatch.setattr(node.settings_module, "get_setting", lambda key, default=None: True)
    chunks = []
    async for chunk in node.llm_node(agent, _chat_ctx(turn), provider, tool_data_cache=cache):
        chunks.append(chunk)
    return chunks


# --- a successful mutation speaks its own handler message ----------------------------------


@pytest.mark.asyncio
async def test_a_created_reminder_is_spoken_without_asking_the_model(monkeypatch, store):
    agent = _Agent()
    provider = _Provider([_call("reminders.create", dict(title="stretch", due="PT2M"))])

    chunks = await _run(
        agent, provider, "remind me in two minutes to stretch", monkeypatch=monkeypatch
    )

    spoken = "".join(chunks)
    assert "stretch" in spoken
    # The delivery question the handler already wrote is what the person hears.
    assert "?" in spoken
    assert provider.stream_calls == 0
    assert agent.published == [scheduled_events.PAYLOAD]
    assert reminders_tools.list_reminders(user_id=ANA)["data"]["reminders"][0]["timed"] is True


@pytest.mark.asyncio
async def test_the_spoken_reply_is_the_handler_message(monkeypatch, store):
    agent = _Agent()
    provider = _Provider([_call("reminders.create", dict(title="stretch", due="PT2M"))])

    chunks = await _run(agent, provider, monkeypatch=monkeypatch)

    expected = reminders_tools.create_reminder(title="stretch", due="PT2M", user_id=ANA)["message"]
    assert "".join(chunks).split(".")[0] == expected.split(".")[0]


@pytest.mark.asyncio
async def test_a_set_alarm_is_spoken_directly(monkeypatch, store):
    agent = _Agent()
    provider = _Provider([_call("alarms.set", dict(label="bread", when="PT5M", kind="timer"))])

    chunks = await _run(agent, provider, monkeypatch=monkeypatch)

    spoken = "".join(chunks)
    assert "bread" in spoken and "Timer set" in spoken
    assert provider.stream_calls == 0
    assert agent.published == [scheduled_events.PAYLOAD]


@pytest.mark.asyncio
async def test_a_delivery_choice_is_spoken_directly(monkeypatch, store):
    reminders_tools.create_reminder(title="call the clinic", due="PT30M", user_id=ANA)
    agent = _Agent()
    provider = _Provider([_call("reminders.set_delivery", dict(delivery=["default"]))])

    chunks = await _run(agent, provider, monkeypatch=monkeypatch)

    assert "".join(chunks).strip() != ""
    assert provider.stream_calls == 0
    assert agent.published == [scheduled_events.PAYLOAD]


# --- a refusal is spoken too, and changes nothing -------------------------------------------


@pytest.mark.asyncio
async def test_a_time_that_cannot_be_read_is_refused_out_loud(monkeypatch, store):
    agent = _Agent()
    provider = _Provider([_call("alarms.set", dict(label="tea", when="soon", kind="alarm"))])

    chunks = await _run(agent, provider, monkeypatch=monkeypatch)

    assert "".join(chunks).strip() != ""
    assert provider.stream_calls == 0
    assert agent.published == []
    assert alarms_tools.pending_count(user_id=ANA) == 0


@pytest.mark.asyncio
async def test_a_missing_argument_is_spoken_without_the_model_instructions(monkeypatch, store):
    agent = _Agent()
    provider = _Provider([_call("reminders.create", dict())])

    chunks = await _run(agent, provider, "please do that thing", monkeypatch=monkeypatch)

    spoken = "".join(chunks).lower()
    assert spoken.strip() != ""
    for instruction in ("call the tool again", "ask the user", "tell the user"):
        assert instruction not in spoken
    assert provider.stream_calls == 0
    assert agent.published == []


@pytest.mark.asyncio
async def test_a_handler_error_is_spoken_as_an_internal_failure(monkeypatch, store):
    reply = importlib.import_module("caal.llm.scheduled_reply")

    def _broken(**kwargs):
        raise RuntimeError("the store is gone")

    monkeypatch.setattr(reminders_tools, "create_reminder", _broken)
    agent = _Agent()
    provider = _Provider([_call("reminders.create", dict(title="stretch", due="PT2M"))])

    chunks = await _run(agent, provider, monkeypatch=monkeypatch)

    spoken = "".join(chunks)
    assert spoken.strip() == reply.INTERNAL_ERROR_REPLY
    # Never the generic routing apology, and never a claim that anything was sent.
    assert "model" not in spoken.lower()
    for claim in ("I will call", "Telegram", "when it comes due"):
        assert claim not in spoken
    assert provider.stream_calls == 0
    assert agent.published == []


# --- everything else still goes through the model -------------------------------------------


@pytest.mark.asyncio
async def test_a_generic_native_tool_still_gets_its_model_follow_up(monkeypatch, store):
    agent = _Agent()
    provider = _StreamingProvider([_call("memory.remember", dict(key="colour", value="blue"))])

    chunks = await _run(agent, provider, monkeypatch=monkeypatch)

    assert "".join(chunks) == "model said something"
    assert provider.stream_calls == 1


@pytest.mark.asyncio
async def test_a_connected_account_tool_still_gets_its_model_follow_up(monkeypatch, store):
    agent = _Agent()
    provider = _StreamingProvider([_call("inbox.recent", dict(limit=1))])

    chunks = await _run(agent, provider, monkeypatch=monkeypatch)

    assert "".join(chunks) == "model said something"
    assert provider.stream_calls == 1


# --- more than one call in a turn -----------------------------------------------------------


@pytest.mark.asyncio
async def test_two_scheduled_calls_are_spoken_in_call_order(monkeypatch, store):
    agent = _Agent()
    provider = _Provider(
        [
            _call("alarms.set", dict(label="bread", when="PT5M", kind="timer"), "1"),
            _call("reminders.create", dict(title="stretch", due="PT9M"), "2"),
        ]
    )

    spoken = "".join(await _run(agent, provider, monkeypatch=monkeypatch))

    assert spoken.index("bread") < spoken.index("stretch")
    assert provider.stream_calls == 0
    assert agent.published == [scheduled_events.PAYLOAD, scheduled_events.PAYLOAD]


@pytest.mark.asyncio
async def test_a_mixed_turn_keeps_the_scheduled_reply_and_never_streams(monkeypatch, store):
    """Bounded on purpose: a turn with private scheduling in it is not narrated.

    The alternative would be handing the model a history that contains the
    reminder of this person, which is exactly what the direct path exists to
    avoid.
    """
    agent = _Agent()
    provider = _Provider(
        [
            _call("reminders.create", dict(title="biopsy results", due="PT2M"), "1"),
            _call("memory.remember", dict(key="colour", value="blue"), "2"),
        ]
    )

    spoken = "".join(await _run(agent, provider, monkeypatch=monkeypatch))

    assert "biopsy results" in spoken
    assert provider.stream_calls == 0
    assert provider.streamed_messages == []
    assert agent.published == [scheduled_events.PAYLOAD]


# --- nothing of it is kept anywhere ----------------------------------------------------------


@pytest.mark.asyncio
async def test_a_scheduled_result_never_reaches_the_tool_data_cache(monkeypatch, store):
    node = importlib.import_module("caal.llm.llm_node")
    cache = node.ToolDataCache()
    agent = _Agent()
    provider = _Provider([_call("reminders.create", dict(title="biopsy results", due="PT2M"))])

    await _run(agent, provider, cache=cache, monkeypatch=monkeypatch)

    context = cache.get_context_message() or ""
    for secret in ("biopsy", "reminders.create", "PT2M"):
        assert secret not in context


@pytest.mark.asyncio
async def test_the_direct_reply_leaves_nothing_private_in_the_log(monkeypatch, store, caplog):
    agent = _Agent()
    provider = _Provider([_call("reminders.create", dict(title="biopsy results", due="PT2M"))])

    with caplog.at_level(logging.DEBUG):
        await _run(
            agent,
            provider,
            "remind me in two minutes about biopsy results",
            monkeypatch=monkeypatch,
        )

    for secret in ("biopsy", "PT2M", ANA):
        assert secret not in caplog.text


@pytest.mark.asyncio
async def test_nothing_of_a_scheduled_turn_is_handed_to_the_provider(monkeypatch, store):
    """The escalation runtime only ever sees what a stream hands it, and there is none."""
    agent = _Agent()
    provider = _Provider([_call("reminders.create", dict(title="biopsy results", due="PT2M"))])

    await _run(agent, provider, monkeypatch=monkeypatch)

    assert provider.streamed_messages == []


# --- the mapping itself ------------------------------------------------------------------------


def test_only_the_scheduled_tools_have_a_direct_outcome():
    reply = importlib.import_module("caal.llm.scheduled_reply")

    ok = dict(status="ok", message="Reminder set.", data=dict())
    assert reply.spoken_outcome("reminders.create", ok) == "Reminder set."
    assert reply.spoken_outcome("alarms.set", ok) == "Reminder set."
    assert reply.spoken_outcome("reminders.set_delivery", ok) == "Reminder set."
    assert reply.spoken_outcome("reminders.list", ok) is None
    assert reply.spoken_outcome("inbox.recent", ok) is None
    assert reply.spoken_outcome("memory.remember", ok) is None
    assert reply.spoken_outcome("reminders.create", "not a result") is None


def test_the_scheduled_tools_are_the_ones_the_event_already_names():
    reply = importlib.import_module("caal.llm.scheduled_reply")

    assert reply.SCHEDULED_TOOLS == scheduled_events.SCHEDULED_TOOLS


def test_an_error_status_never_says_what_went_wrong():
    reply = importlib.import_module("caal.llm.scheduled_reply")

    spoken = reply.spoken_outcome(
        "reminders.create", dict(status="error", message="sqlite3.OperationalError: no such table")
    )

    assert spoken == reply.INTERNAL_ERROR_REPLY
    assert "sqlite" not in spoken.lower()
