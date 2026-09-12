"""Understanding a delivery answer that is said naturally, without widening it.

The delivery question -- "say it here, send it to your Telegram, call you, or
nothing?" -- gets answered in whatever words the person happens to use. The
offline whitelist in :mod:`caal.tools.delivery_answer` reads the short exact
forms and nothing else, so "make sure it rings me as well" falls through it and
lands on a model that has repeatedly read those words as a request to place a
phone call now.

So a second, narrow layer sits behind the whitelist: the *local* model, asked
one classification-only question, for one signed-in owner, only while that
owner has a real open delivery question. These tests pin what it is allowed to
see, what it is allowed to return, and -- mostly -- everything it is never
allowed to touch:

* the offline gate runs first and is the safety boundary. A question, a number,
  a hypothetical, somebody else, a calendar or task request, a contradiction,
  or a bare "sure" never reaches a model at all;
* the payload is two fixed messages: the shared system prompt and the bounded
  current turn. No title, no row id, no due time, no owner, no destination, no
  history, no tool schemas, no cached provider data;
* the reply is read as strict top-level JSON in a bounded enum and validated
  through the same channel parser the tool uses. Prose, extra fields, an
  invented channel, a destination, or a contradictory pair decide nothing;
* an unreachable, slow or unusable local model fails closed to ordinary
  routing: nothing is spoken, nothing is written, and Hermes is never asked.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from caal.delivery_router import DeliveryAnswerHandler
from caal.tools import alarms_tools, reminder_delivery, reminders_tools
from caal.tools.delivery_semantics import (
    DELIVERY_SEMANTIC_SYSTEM_PROMPT,
    MAX_SEMANTIC_INPUT_CHARS,
    SemanticDeliveryReader,
    askable,
    parse_delivery_reply,
)
from caal.user_scope import UserScope
from caal.work_router import provider_classifier

ANA = "usr_" + "a" * 24
BO = "usr_" + "b" * 24
NOW = 1_000_000
SOON = "PT30M"

SPEAK = reminder_delivery.SPEAK
TELEGRAM = reminder_delivery.TELEGRAM
CALL = reminder_delivery.CALL
ALL = reminder_delivery.ALL
DEFAULT = reminder_delivery.DEFAULT
NONE = reminder_delivery.NONE

TITLE = "Take the blue folder to the clinic"

SPEAK_AND_CALL = json.dumps(dict(delivery=[SPEAK, CALL]))


# --- the doubles ------------------------------------------------------------------------------


class FakeLocalModel:
    """The local provider, recording exactly what it was asked and offered."""

    def __init__(self, reply: str = SPEAK_AND_CALL, *, error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[tuple[Any, Any]] = []

    async def chat(self, messages: Any, tools: Any = None, **_: Any) -> Any:
        self.calls.append((messages, tools))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=self.reply, tool_calls=[])


class RefusingModel(FakeLocalModel):
    """A local model whose answer would be wrong if it were ever consulted."""

    def __init__(self) -> None:
        super().__init__(json.dumps(dict(delivery=[CALL])))


class FakeSession:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def say(self, text: str, **_: Any) -> None:
        self.spoken.append(text)


class Announcer:
    def __init__(self) -> None:
        self.published = 0

    async def __call__(self) -> None:
        self.published += 1


@pytest.fixture
def store(monkeypatch, tmp_path):
    path = tmp_path / "assistant.sqlite3"
    monkeypatch.setattr(alarms_tools, "STORE_PATH", path)
    monkeypatch.setattr(reminders_tools, "STORE_PATH", path)
    monkeypatch.setattr(reminder_delivery, "STORE_PATH", path)
    return path


@pytest.fixture
def everything_available(monkeypatch):
    monkeypatch.setattr(reminder_delivery, "telegram_owner", lambda: ANA)
    monkeypatch.setattr(reminder_delivery, "telegram_configured", lambda: True)
    monkeypatch.setattr(
        reminder_delivery,
        "resolve_callback_number",
        lambda user_id: "+15551230000" if user_id == ANA else None,
    )


@pytest.fixture
def not_dry_run(monkeypatch):
    monkeypatch.delenv("CAAL_CALLBACK_DISPATCH_DRY_RUN", raising=False)


def run(coroutine):
    return asyncio.run(coroutine)


def _reader(model: FakeLocalModel) -> SemanticDeliveryReader:
    return SemanticDeliveryReader(classify=provider_classifier(model))


def _handler(
    model: FakeLocalModel,
    user_id: str | None = ANA,
    *,
    announce: Any = None,
    now: int = NOW,
) -> DeliveryAnswerHandler:
    scope = (
        UserScope(user_id=user_id, identity_configured=True) if user_id else UserScope.anonymous()
    )
    return DeliveryAnswerHandler(
        scope=scope, announce=announce, now=lambda: now, semantic=_reader(model)
    )


def _reminder_id(path, user_id: str = ANA) -> str:
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT id FROM reminders WHERE user_id = ? ORDER BY created_at DESC, rowid DESC",
            (user_id,),
        ).fetchall()
    finally:
        connection.close()
    assert rows, "expected a reminder for this owner"
    return rows[0][0]


def _ask(user_id: str = ANA) -> None:
    created = reminders_tools.create_reminder(title=TITLE, due=SOON, user_id=user_id, now=NOW)
    assert created["data"]["delivery_pending"] is True


# --- the offline gate decides what a model may ever see ---------------------------------------


NATURAL = (
    "i want it to call me too while saying it here",
    "make sure it rings me as well",
    "keep the spoken reminder and add a call",
    "send it to telegram too and keep saying it here",
    "i would like a call as well as saying it here",
    "both a call and telegram please",
)


@pytest.mark.parametrize("text", NATURAL)
def test_a_natural_delivery_answer_may_be_read_by_the_local_model(text) -> None:
    assert askable(text) is not None


NEVER_ASKED = (
    "",
    "   ",
    None,
    "call me in five minutes",
    "call me in 5 minutes",
    "call me at 4",
    "call my wife",
    "call her",
    "call them instead",
    "call the clinic",
    "call John",
    "should you call me?",
    "what are my options",
    "do you want to call me",
    "would you call me",
    "maybe call me",
    "if it is urgent call me",
    "call me when the research is done",
    "hang up and call me back when you are done",
    "put a call on my calendar",
    "schedule a call for tomorrow",
    "add a call to my tasks",
    "remind me about my call",
    "call me about the dentist appointment tomorrow",
    "call me later",
    "tell me a joke",
    "call me and also book the room",
    "nothing and call me",
    "leave it silent but ring me",
    "sure",
    "yes please",
    "ok thanks",
    "yeah",
    "i want it",
)


@pytest.mark.parametrize("text", NEVER_ASKED)
def test_anything_conservative_safeguards_catch_never_reaches_a_model(text) -> None:
    assert askable(text) is None


def test_a_long_turn_is_never_shown_to_a_model() -> None:
    assert askable("please call me " * 40) is None


def test_the_text_shown_to_a_model_is_bounded_and_plain() -> None:
    shown = askable("Make sure it RINGS me, as well!")
    assert shown == "make sure it rings me as well"
    assert len(shown) <= MAX_SEMANTIC_INPUT_CHARS


# --- the reply is read strictly, or not at all ------------------------------------------------

OBJECT = json.dumps(dict(delivery=[CALL]))
FENCE = "```"


@pytest.mark.parametrize(
    "reply, expected",
    [
        (OBJECT, (CALL,)),
        (json.dumps(dict(delivery=[SPEAK, CALL])), (SPEAK, CALL)),
        (json.dumps(dict(delivery=[CALL, SPEAK])), (SPEAK, CALL)),
        (json.dumps(dict(delivery=[TELEGRAM, SPEAK])), (SPEAK, TELEGRAM)),
        (json.dumps(dict(delivery=[ALL])), (ALL,)),
        (json.dumps(dict(delivery=[DEFAULT])), (DEFAULT,)),
        (json.dumps(dict(delivery=[NONE])), (NONE,)),
        (json.dumps(dict(delivery=CALL)), (CALL,)),
        ("  " + OBJECT + "  ", (CALL,)),
    ],
)
def test_a_strict_reply_selects_the_bounded_channels_it_names(reply, expected) -> None:
    assert parse_delivery_reply(reply) == expected


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "   ",
        None,
        123,
        "call",
        "call me",
        "They want a call.",
        "Sure! " + OBJECT,
        OBJECT + " -- hope that helps",
        FENCE + "json\n" + OBJECT + "\n" + FENCE,
        json.dumps(dict(delivery=[CALL], reason="they said so")),
        json.dumps(dict(delivery=[CALL], confidence=0.9)),
        json.dumps(dict(route="work")),
        json.dumps(dict(channels=[CALL])),
        json.dumps(dict(delivery="unclear")),
        json.dumps(dict(delivery=[])),
        json.dumps(dict(delivery=["sms"])),
        json.dumps(dict(delivery=["email"])),
        json.dumps(dict(delivery=["+15551230000"])),
        json.dumps(dict(delivery=["call", "+15551230000"])),
        json.dumps(dict(delivery=[NONE, CALL])),
        json.dumps(dict(delivery=[ALL, CALL])),
        json.dumps(dict(delivery=[DEFAULT, TELEGRAM])),
        json.dumps(dict(delivery=[SPEAK, TELEGRAM, CALL, SPEAK])),
        json.dumps(dict(delivery=[[CALL]])),
        json.dumps(dict(delivery=dict(channel=CALL))),
        json.dumps([CALL]),
        json.dumps(CALL),
        chr(123),
    ],
)
def test_a_reply_that_is_not_exactly_that_shape_decides_nothing(reply) -> None:
    assert parse_delivery_reply(reply) is None


def test_a_reply_longer_than_the_bound_decides_nothing() -> None:
    assert parse_delivery_reply(OBJECT + " " * 5_000) is None


# --- what the model is shown --------------------------------------------------------------


def test_the_payload_is_the_shared_prompt_and_the_current_turn_and_nothing_else() -> None:
    model = FakeLocalModel()

    assert run(_reader(model).read("make sure it rings me as well")) == (SPEAK, CALL)

    assert len(model.calls) == 1
    messages, tools = model.calls[0]
    assert tools is None, "a classification call must never carry tool schemas"
    assert messages == [
        dict(role="system", content=DELIVERY_SEMANTIC_SYSTEM_PROMPT),
        dict(role="user", content="make sure it rings me as well"),
    ]


def test_the_payload_carries_nothing_that_belongs_to_the_owner(
    store, everything_available, not_dry_run
) -> None:
    _ask()
    model = FakeLocalModel()
    session = FakeSession()

    assert run(_handler(model).handle("keep the spoken reminder and add a call", session)) is True

    sent = json.dumps(model.calls[0][0]).lower()
    for secret in (
        TITLE.lower(),
        _reminder_id(store).lower(),
        ANA.lower(),
        "+15551230000",
        "reminder_id",
        "user_id",
        "function",
        "parameters",
    ):
        assert secret not in sent


def test_the_prompt_never_names_a_destination_or_a_tool() -> None:
    lowered = DELIVERY_SEMANTIC_SYSTEM_PROMPT.lower()
    for forbidden in ("reminders.set_delivery", "tool", "http", "chat id", "phone number"):
        assert forbidden not in lowered


def _voice_agent(name: str):
    module_path = Path(__file__).parents[1] / "voice_agent.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_classifier_runs_on_the_local_model_and_never_on_hermes() -> None:
    voice_agent = _voice_agent("voice_agent_for_semantics")
    primary = FakeLocalModel()
    escalation = RefusingModel()
    routed = SimpleNamespace(
        provider_instance=SimpleNamespace(primary=primary, escalation=escalation)
    )

    handler = voice_agent.build_delivery_answer_handler(
        UserScope(user_id=ANA, identity_configured=True), provider=routed
    )

    assert handler is not None
    assert voice_agent.work_router_provider(routed) is primary
    assert run(handler.semantic.read("make sure it rings me as well")) == (SPEAK, CALL)
    assert len(primary.calls) == 1
    assert escalation.calls == []


def test_without_a_provider_the_route_is_the_offline_whitelist_alone() -> None:
    voice_agent = _voice_agent("voice_agent_no_provider")

    handler = voice_agent.build_delivery_answer_handler(
        UserScope(user_id=ANA, identity_configured=True)
    )

    assert handler is not None
    assert handler.semantic is None or handler.semantic.enabled is False


# --- the route, end to end ----------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "i want it to call me too while saying it here",
        "make sure it rings me as well",
        "keep the spoken reminder and add a call",
        "i would like a call as well as saying it here",
    ],
)
def test_a_natural_answer_settles_the_reminder_and_announces_once(
    store, everything_available, not_dry_run, text
) -> None:
    _ask()
    model = FakeLocalModel()
    session = FakeSession()
    announce = Announcer()

    consumed = run(_handler(model, announce=announce).handle(text, session))

    assert consumed is True
    assert reminder_delivery.channels_of(_reminder_id(store)) == (SPEAK, CALL)
    assert announce.published == 1
    assert len(model.calls) == 1
    assert session.spoken
    assert reminder_delivery.awaiting_reminder(ANA, NOW) is None


def test_a_natural_answer_can_choose_telegram_beside_the_spoken_one(
    store, everything_available, not_dry_run
) -> None:
    _ask()
    model = FakeLocalModel(json.dumps(dict(delivery=[SPEAK, TELEGRAM])))
    session = FakeSession()

    consumed = run(
        _handler(model).handle("send it to telegram too and keep saying it here", session)
    )

    assert consumed is True
    assert reminder_delivery.channels_of(_reminder_id(store)) == (SPEAK, TELEGRAM)


def test_the_exact_whitelist_still_answers_without_consulting_any_model(
    store, everything_available, not_dry_run
) -> None:
    _ask()
    model = RefusingModel()
    session = FakeSession()

    assert run(_handler(model).handle("say it here and also call me", session)) is True
    assert reminder_delivery.channels_of(_reminder_id(store)) == (SPEAK, CALL)
    assert model.calls == [], "the offline whitelist must not pay for a model call"


class RefusingBackground:
    """Background work that fails the test if an answered turn ever reaches it."""

    def has_deliverable_file_request(self, _text: str) -> bool:
        return False

    async def process_turn(self, *_args, **_kwargs):
        raise AssertionError("a delivery answer must never start background work")


async def _never_end() -> None:
    raise AssertionError("a delivery answer must not end the call")


def _local_turn(voice_agent, session, handler):
    return voice_agent.LocalTurnHandler(
        phone_handoff=None,
        session=session,
        end_call=_never_end,
        background=RefusingBackground(),
        delivery=handler,
    )


def test_a_spoken_natural_answer_is_claimed_before_any_other_route(
    store, everything_available, not_dry_run
) -> None:
    voice_agent = _voice_agent("voice_agent_semantic_speech")
    _ask()
    model = FakeLocalModel()
    session = FakeSession()
    text = "make sure it rings me as well"
    local = _local_turn(voice_agent, session, _handler(model))

    async def turn() -> bool:
        local.on_final_transcript(text)
        return await local.turn_consumed(text)

    assert run(turn()) is True
    assert reminder_delivery.channels_of(_reminder_id(store)) == (SPEAK, CALL)


def test_typed_chat_gets_the_same_natural_route(store, everything_available, not_dry_run) -> None:
    voice_agent = _voice_agent("voice_agent_semantic_typed")
    _ask()
    model = FakeLocalModel()
    session = FakeSession()
    local = _local_turn(voice_agent, session, _handler(model))

    assert run(local.turn_consumed("keep the spoken reminder and add a call")) is True
    assert reminder_delivery.channels_of(_reminder_id(store)) == (SPEAK, CALL)


# --- and everything that must still fall through --------------------------------------------


@pytest.mark.parametrize("text", NEVER_ASKED[3:])
def test_a_turn_the_gate_refuses_is_left_to_ordinary_routing(
    store, everything_available, not_dry_run, text
) -> None:
    _ask()
    model = RefusingModel()
    session = FakeSession()
    announce = Announcer()

    assert run(_handler(model, announce=announce).handle(text, session)) is False
    assert model.calls == []
    assert reminder_delivery.channels_of(_reminder_id(store)) == (SPEAK,)
    assert announce.published == 0
    assert session.spoken == []


@pytest.mark.parametrize(
    "reply",
    [
        "They clearly want a call.",
        "Sure! " + json.dumps(dict(delivery=[CALL])),
        json.dumps(dict(delivery=[CALL], reason="obvious")),
        json.dumps(dict(delivery=["sms"])),
        json.dumps(dict(delivery=["+15551230000"])),
        json.dumps(dict(delivery=[NONE, CALL])),
        json.dumps(dict(delivery="unclear")),
        "",
    ],
)
def test_an_unusable_model_answer_changes_nothing(
    store, everything_available, not_dry_run, reply
) -> None:
    _ask()
    model = FakeLocalModel(reply)
    session = FakeSession()
    announce = Announcer()

    assert (
        run(_handler(model, announce=announce).handle("make sure it rings me as well", session))
        is False
    )
    assert reminder_delivery.channels_of(_reminder_id(store)) == (SPEAK,)
    assert announce.published == 0
    assert session.spoken == []


def test_an_unreachable_local_model_fails_closed_and_says_nothing(
    store, everything_available, not_dry_run
) -> None:
    _ask()
    model = FakeLocalModel(error=ConnectionError("http://192.168.1.50:11434 refused"))
    session = FakeSession()
    announce = Announcer()

    assert (
        run(_handler(model, announce=announce).handle("make sure it rings me as well", session))
        is False
    )
    assert reminder_delivery.channels_of(_reminder_id(store)) == (SPEAK,)
    assert announce.published == 0
    assert session.spoken == []


def test_a_slow_local_model_is_abandoned_rather_than_waited_on(
    store, everything_available, not_dry_run
) -> None:
    _ask()
    session = FakeSession()

    async def _never_answers(_messages):
        await asyncio.sleep(30)
        return json.dumps(dict(delivery=[CALL]))

    handler = DeliveryAnswerHandler(
        scope=UserScope(user_id=ANA, identity_configured=True),
        now=lambda: NOW,
        semantic=SemanticDeliveryReader(classify=_never_answers, timeout_seconds=0.1),
    )

    assert run(handler.handle("make sure it rings me as well", session)) is False
    assert reminder_delivery.channels_of(_reminder_id(store)) == (SPEAK,)
    assert session.spoken == []


def test_with_no_open_question_no_model_is_asked_at_all(
    store, everything_available, not_dry_run
) -> None:
    reminders_tools.create_reminder(title=TITLE, due=SOON, delivery=[SPEAK], user_id=ANA, now=NOW)
    model = RefusingModel()
    session = FakeSession()

    assert run(_handler(model).handle("make sure it rings me as well", session)) is False
    assert model.calls == []
    assert reminder_delivery.channels_of(_reminder_id(store)) == (SPEAK,)


def test_another_owner_question_is_never_answered_semantically(
    store, everything_available, not_dry_run
) -> None:
    _ask()
    model = RefusingModel()
    session = FakeSession()

    assert run(_handler(model, BO).handle("make sure it rings me as well", session)) is False
    assert model.calls == []
    assert reminder_delivery.channels_of(_reminder_id(store)) == (SPEAK,)


def test_an_anonymous_session_never_reaches_the_model(
    store, everything_available, not_dry_run
) -> None:
    _ask()
    model = RefusingModel()
    session = FakeSession()

    assert run(_handler(model, None).handle("make sure it rings me as well", session)) is False
    assert model.calls == []


def test_an_expired_question_is_never_answered_semantically(
    store, everything_available, not_dry_run
) -> None:
    _ask()
    model = RefusingModel()
    session = FakeSession()

    handler = _handler(model, now=NOW + 3_600)

    assert run(handler.handle("make sure it rings me as well", session)) is False
    assert model.calls == []
    assert reminder_delivery.channels_of(_reminder_id(store)) == (SPEAK,)


def test_a_superseded_question_is_never_answered_semantically(
    store, everything_available, not_dry_run
) -> None:
    _ask()
    reminders_tools.create_reminder(
        title="Second thing", due="PT45M", delivery=[SPEAK], user_id=ANA, now=NOW
    )
    model = RefusingModel()
    session = FakeSession()

    assert run(_handler(model).handle("make sure it rings me as well", session)) is False
    assert model.calls == []


def test_the_same_natural_answer_twice_only_changes_anything_once(
    store, everything_available, not_dry_run
) -> None:
    _ask()
    model = FakeLocalModel()
    session = FakeSession()
    announce = Announcer()
    handler = _handler(model, announce=announce)
    text = "make sure it rings me as well"

    assert run(handler.handle(text, session)) is True
    assert run(handler.handle(text, session)) is False
    assert announce.published == 1
    assert reminder_delivery.channels_of(_reminder_id(store)) == (SPEAK, CALL)


def test_the_semantic_route_logs_nothing_that_belongs_to_anybody(
    store, everything_available, not_dry_run, caplog
) -> None:
    _ask()
    model = FakeLocalModel()
    session = FakeSession()
    utterance = "i want it to call me too while saying it here"

    with caplog.at_level(logging.DEBUG):
        assert run(_handler(model).handle(utterance, session)) is True

    written = " ".join(record.getMessage() for record in caplog.records).lower()
    for secret in (
        utterance,
        "rings me",
        TITLE.lower(),
        _reminder_id(store).lower(),
        ANA.lower(),
        SPEAK_AND_CALL.lower(),
    ):
        assert secret not in written
