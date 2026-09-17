"""Stage 3: the fixed outcome strings a Spanish session hears.

Stage 2 left every *reply* CAAL says itself in English (STAGE2.md §6.2): the
model answered in Spanish, but "Calling your phone now." did not. These are not
model output -- they are constants spoken by handler code, so no directive can
reach them.

The rule this pins, and the reason it is narrow:

* Only an **exact, whole, known** English constant is localized. The lookup is a
  vetted table keyed by the English string itself, so an arbitrary sentence --
  a tool result, a model answer, a composed message -- passes through untouched.
  There is no translator here and nothing is sent anywhere to be translated.
* A localized pair says the **same thing**: same question, same risk, same
  refusal. A confirmation question stays a question, and a refusal stays a
  refusal. It changes the language of an outcome, never the policy of one.
* The English side is **byte-identical** to what it was, so an English session
  is unchanged. That is asserted directly, on the bytes.
* Localization happens where a reply is spoken. It cannot reach the point where
  an action is authorized: a Spanish session needs exactly the same pending
  confirmation as an English one, which the handler-path tests below exercise
  for real.

No phone call, email, reminder, light or device is touched by any test here:
every seam is a fixture and the dial seam records instead of dialing.
"""

from __future__ import annotations

import asyncio
import hashlib

import pytest

from caal.background_task_session import (
    BACKGROUND_CONTROL_REPLIES,
    CALLBACK_ARMED_REPLY,
    LONG_WORK_ACK_REPLY,
)
from caal.end_call_intent import END_CALL_CONTROL_REPLIES
from caal.handoff_intent import (
    HANDOFF_CONTROL_REPLIES,
    NO_CALLBACK_NUMBER_REPLY,
    PhoneHandoffController,
)
from caal.language_policy import EN, ES, LanguageSession
from caal.reply_localization import (
    END_CALL_REPLY,
    begin_turn,
    current_language,
    localize,
    spanish_pairs,
)

ALL_FIXED = (
    *HANDOFF_CONTROL_REPLIES,
    *END_CALL_CONTROL_REPLIES,
    *BACKGROUND_CONTROL_REPLIES,
    END_CALL_REPLY,
)

#: The phrasings these paths actually recognise today. Both are English: the
#: *inputs* that reach these handlers are unchanged by this work, and only the
#: Spanish confirmation and denial vocabulary added in stage 1 is understood
#: (see ``test_a_spanish_phrasing_of_the_request_is_still_not_recognised``).
HANDOFF_REQUEST = "continue this conversation on my phone"
STATUS_REQUEST = "background task status"
CANCEL_REQUEST = "cancel the background task"


@pytest.fixture(autouse=True)
def _english_by_default():
    """Every test starts in English, and leaves the turn language as it found it."""
    from caal import reply_localization

    token = reply_localization.reply_language.set(EN)
    yield
    reply_localization.reply_language.reset(token)


# --- 1. The English side does not move ---------------------------------------


def test_the_english_constants_are_byte_identical():
    """One hash over every fixed reply. A change to any byte fails here.

    This is the backwards-compatibility guarantee for every English session:
    localization added a table beside these strings and did not touch them.
    """
    digest = hashlib.sha256("\x00".join(ALL_FIXED).encode()).hexdigest()
    assert digest == "b60befb6d2fbda77f153f7376dbee524c8863ca2d625f488872658f12335dd5c"


def test_an_english_turn_gets_the_english_constant_unchanged():
    for text in ALL_FIXED:
        assert localize(text) is text


def test_the_default_turn_language_is_english():
    assert current_language() == EN


# --- 2. The Spanish table is complete, and only a table ----------------------


def test_every_fixed_reply_has_a_vetted_spanish_pair():
    missing = [text for text in ALL_FIXED if text not in spanish_pairs()]
    assert missing == []


def test_no_pair_is_empty_or_a_copy_of_the_english():
    for english, spanish in spanish_pairs().items():
        assert spanish.strip()
        assert spanish != english


def test_a_question_stays_a_question_and_a_statement_stays_a_statement():
    """The risk an outcome carries is in its shape; the pair keeps it."""
    for english, spanish in spanish_pairs().items():
        assert english.rstrip().endswith("?") == spanish.rstrip().endswith("?")


def test_an_arbitrary_message_is_never_translated():
    """No general translation: anything not in the table is spoken as given."""
    from caal import reply_localization

    reply_localization.reply_language.set(ES)
    for text in (
        "You have 3 unread emails from Dana Whitfield.",
        "Understood. I'll work on that in the background.",  # near-miss, not exact
        LONG_WORK_ACK_REPLY + " ",  # trailing byte differs
        "",
    ):
        assert localize(text) is text


def test_a_spanish_turn_gets_the_pair():
    from caal import reply_localization

    reply_localization.reply_language.set(ES)
    assert localize(CALLBACK_ARMED_REPLY) == spanish_pairs()[CALLBACK_ARMED_REPLY]
    assert localize(CALLBACK_ARMED_REPLY) != CALLBACK_ARMED_REPLY


# --- 3. The turn language comes from the session, not from a message ---------


class _Agent:
    def __init__(self, language: str) -> None:
        self._language_session = LanguageSession()
        self._language_session.current = language


def test_begin_turn_reads_the_session_language():
    assert begin_turn(_Agent(ES)) == ES
    assert current_language() == ES
    assert begin_turn(_Agent(EN)) == EN
    assert current_language() == EN


def test_begin_turn_on_an_agent_without_a_session_is_english():
    assert begin_turn(object()) == EN
    assert current_language() == EN


def test_begin_turn_never_raises_on_a_broken_session():
    class Broken:
        @property
        def _language_session(self):
            raise RuntimeError("no")

    assert begin_turn(Broken()) == EN


# --- 4. The handler paths, for real ------------------------------------------


class _Session:
    """A speaker that records. Nothing here reaches TTS or a room."""

    def __init__(self) -> None:
        self.said: list[str] = []

    async def say(self, text, **kwargs):
        self.said.append(text)


def _controller(dialed: list[tuple], *, number: str | None = "+15550001111"):
    async def start_call(destination, **kwargs):
        dialed.append((destination, kwargs))

    return PhoneHandoffController(
        start_call=start_call,
        allowed_destinations=number or "",
        user_id="user-1",
        destination_resolver=(lambda: number),
    )


def test_a_spanish_session_hears_the_confirmation_question_in_spanish_and_is_not_called():
    """The safety property: the question is localized, the policy is not."""
    from caal import reply_localization

    reply_localization.reply_language.set(ES)
    dialed: list[tuple] = []
    controller = _controller(dialed)
    session = _Session()

    consumed = asyncio.run(controller.handle_final_transcript(HANDOFF_REQUEST, session))

    assert consumed is True
    # A pending confirmation exists -- this is the route, not a turn the
    # handler happened to ignore.
    assert controller.awaiting_confirmation is True
    assert dialed == []
    spoken = session.said[-1]
    assert spoken in spanish_pairs().values()
    assert spoken.rstrip().endswith("?")


def test_confirming_in_spanish_then_dials_and_says_so_in_spanish():
    from caal import reply_localization
    from caal.handoff_intent import STARTING_REPLY

    reply_localization.reply_language.set(ES)
    dialed: list[tuple] = []
    controller = _controller(dialed)
    session = _Session()

    asyncio.run(controller.handle_final_transcript(HANDOFF_REQUEST, session))
    assert dialed == []
    asyncio.run(controller.handle_final_transcript("sí", session))

    assert len(dialed) == 1
    assert dialed[0][0] == "+15550001111"
    assert session.said[-1] == spanish_pairs()[STARTING_REPLY]


def test_the_same_english_flow_is_byte_for_byte_what_it_was():
    from caal.handoff_intent import STARTING_REPLY

    dialed: list[tuple] = []
    controller = _controller(dialed)
    session = _Session()
    asyncio.run(controller.handle_final_transcript(HANDOFF_REQUEST, session))
    assert dialed == []
    assert session.said[-1] in HANDOFF_CONTROL_REPLIES
    asyncio.run(controller.handle_final_transcript("yes", session))
    assert len(dialed) == 1
    assert session.said[-1] == STARTING_REPLY


def test_a_refusal_stays_a_refusal_in_spanish_and_still_does_not_dial():
    """No approved number: the Spanish session is refused, not called."""
    from caal import reply_localization

    reply_localization.reply_language.set(ES)
    dialed: list[tuple] = []
    controller = _controller(dialed, number=None)
    session = _Session()

    asyncio.run(controller.handle_final_transcript(HANDOFF_REQUEST, session))
    asyncio.run(controller.handle_final_transcript("sí", session))

    assert dialed == []
    assert session.said[-1] == spanish_pairs()[NO_CALLBACK_NUMBER_REPLY]


def test_a_spanish_denial_still_denies():
    from caal import reply_localization
    from caal.handoff_intent import CANCELLED_REPLY

    reply_localization.reply_language.set(ES)
    dialed: list[tuple] = []
    controller = _controller(dialed)
    session = _Session()
    asyncio.run(controller.handle_final_transcript(HANDOFF_REQUEST, session))
    asyncio.run(controller.handle_final_transcript("no", session))
    assert dialed == []
    assert controller.awaiting_confirmation is False
    assert session.said[-1] == spanish_pairs()[CANCELLED_REPLY]


def test_a_spanish_phrasing_of_the_request_is_still_not_recognised():
    """An honest gap, pinned rather than papered over.

    This work localizes what CAAL *says*. It does not widen what CAAL
    *understands*: the handoff request itself is still matched in English only,
    so "sigamos por teléfono" reaches no handoff route at all and the turn goes
    to the model like any other. Only the confirmation and denial vocabulary
    accepts Spanish, which stage 1 added.

    A Spanish session therefore gets Spanish outcome strings, not a Spanish
    intent surface. Recognising a Spanish request is a separate piece of work,
    and it is a safety-bearing one: it decides when a call may be offered.
    """
    from caal.handoff_intent import HandoffIntent, classify_handoff_intent

    assert classify_handoff_intent("sigamos por teléfono") is HandoffIntent.NONE
    assert classify_handoff_intent("llámame al teléfono") is HandoffIntent.NONE
    # What *is* understood in Spanish, and only while a question is pending:
    from caal.handoff_intent import confirmation_given, denial_given

    assert confirmation_given("sí") is True
    assert denial_given("no") is True


def test_the_spanish_request_phrase_reaches_no_handoff_and_dials_nothing():
    """The same gap, through the real controller: no route, no call, no reply."""
    from caal import reply_localization

    reply_localization.reply_language.set(ES)
    dialed: list[tuple] = []
    controller = _controller(dialed)
    session = _Session()

    consumed = asyncio.run(controller.handle_final_transcript("sigamos por teléfono", session))

    assert consumed is False
    assert controller.awaiting_confirmation is False
    assert dialed == []
    assert session.said == []


# --- 5. The turn language is per-task state, not shared ----------------------


def test_two_concurrent_turns_do_not_share_a_reply_language():
    """Two sessions speaking at once must not localize each other's replies.

    The language is a context variable set per turn, so a task started with its
    own context keeps its own value. This drives both turns through the real
    controller concurrently and checks what each one was actually told.
    """
    import contextvars

    from caal import reply_localization
    from caal.handoff_intent import ASK_CONFIRMATION_REPLY

    async def one_turn(language, session, ready, go):
        reply_localization.reply_language.set(language)
        dialed: list[tuple] = []
        controller = _controller(dialed)
        # Interleave the two turns around the await, so a shared variable would
        # show up as the wrong language rather than as luck.
        ready.set()
        await go.wait()
        await controller.handle_final_transcript(HANDOFF_REQUEST, session)
        return dialed

    async def scenario():
        spanish, english = _Session(), _Session()
        ready_es, ready_en = asyncio.Event(), asyncio.Event()
        go = asyncio.Event()
        tasks = [
            asyncio.create_task(
                one_turn(ES, spanish, ready_es, go), context=contextvars.copy_context()
            ),
            asyncio.create_task(
                one_turn(EN, english, ready_en, go), context=contextvars.copy_context()
            ),
        ]
        await ready_es.wait()
        await ready_en.wait()
        go.set()
        dialed = await asyncio.gather(*tasks)
        return spanish, english, dialed

    spanish, english, dialed = asyncio.run(scenario())

    assert spanish.said == [spanish_pairs()[ASK_CONFIRMATION_REPLY]]
    assert english.said == [ASK_CONFIRMATION_REPLY]
    assert dialed == [[], []]
    # And the ambient turn language is untouched by either of them.
    assert current_language() == EN


def test_the_language_is_read_when_the_turn_starts_not_when_a_reply_is_composed():
    """``begin_turn`` is the only reader of session state; a later change to the
    session does not retroactively re-language a turn already under way."""
    from caal import reply_localization

    agent = _Agent(ES)
    begin_turn(agent)
    agent._language_session.current = EN  # the *next* turn's language
    assert reply_localization.current_language() == ES


# --- 6. Scheduling and error acknowledgements, through the bridge ------------


@pytest.fixture
def store(monkeypatch, tmp_path):
    """The durable-work queue, in a temporary file. No production data dir."""
    from caal import background_tasks

    monkeypatch.setattr(background_tasks, "STORE_PATH", tmp_path / "assistant.sqlite3")
    return tmp_path


def _bridge(reply: str = "conversation"):
    from caal.background_task_session import BackgroundTaskBridge
    from caal.work_router import SemanticWorkRouter

    scheduled: list[str] = []

    async def classify(messages):
        return reply

    async def execute(request):  # pragma: no cover - the queue is not run here
        raise AssertionError("no worker should run in this test")

    bridge = BackgroundTaskBridge(
        execute=execute,
        session_key="room-es",
        work_router=SemanticWorkRouter(classify=classify, timeout_seconds=1.0),
    )
    return bridge, scheduled


def test_a_spanish_session_hears_the_status_reply_in_spanish(store):
    from caal import reply_localization
    from caal.background_task_session import BACKGROUND_STATUS_IDLE_REPLY

    reply_localization.reply_language.set(ES)
    bridge, _ = _bridge()
    session = _Session()
    outcome = asyncio.run(bridge.process_turn(STATUS_REQUEST, session))
    assert outcome.consumed is True
    assert session.said == [spanish_pairs()[BACKGROUND_STATUS_IDLE_REPLY]]


def test_the_same_status_turn_in_english_is_unchanged(store):
    from caal.background_task_session import BACKGROUND_STATUS_IDLE_REPLY

    bridge, _ = _bridge()
    session = _Session()
    asyncio.run(bridge.process_turn(STATUS_REQUEST, session))
    assert session.said == [BACKGROUND_STATUS_IDLE_REPLY]


def test_a_spanish_session_hears_nothing_to_cancel_in_spanish(store):
    from caal import reply_localization
    from caal.background_task_session import BACKGROUND_NOTHING_TO_CANCEL_REPLY

    reply_localization.reply_language.set(ES)
    bridge, _ = _bridge()
    session = _Session()
    outcome = asyncio.run(bridge.process_turn(CANCEL_REQUEST, session))
    assert outcome.consumed is True
    assert session.said == [spanish_pairs()[BACKGROUND_NOTHING_TO_CANCEL_REPLY]]
