"""Migration contracts: visible FRIDAY identity and both spoken aliases."""

import json
from pathlib import Path

import pytest

from caal.background_tasks import (
    background_cancel_requested,
    background_status_requested,
    long_running_work_inferred,
)
from caal.call_termination import callback_requested, end_call_requested
from caal.end_call_intent import EndCallIntent, classify_end_call_intent
from caal.handoff_intent import confirmation_given, denial_given, handoff_requested
from caal.knowledge_router import plan_knowledge_turn
from caal.tools.delivery_answer import read_delivery_answer
from caal.tools.delivery_semantics import askable
from caal.work_router import RouteSource, deterministic_route

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["FRIDAY", "Jarvis"])
@pytest.mark.parametrize(
    ("reader", "phrase"),
    [
        (end_call_requested, "{name}, hang up now"),
        (end_call_requested, "{name}, end the call"),
        (end_call_requested, "{name}, disconnect"),
        (end_call_requested, "goodbye {name}"),
        (callback_requested, "{name}, hang up and call me back when you're done"),
        (handoff_requested, "{name}, continue this conversation on my phone"),
        (confirmation_given, "{name}, yes"),
        (confirmation_given, "hey {name}, yes"),
        (denial_given, "{name}, no"),
        (background_cancel_requested, "{name}, cancel the background task"),
        (background_status_requested, "{name}, is the report ready yet"),
        (long_running_work_inferred, "{name}, prepare a PDF with meeting notes"),
    ],
)
def test_command_aliases(name, reader, phrase):
    assert reader(phrase.format(name=name))


@pytest.mark.parametrize("name", ["friday", "jarvis"])
def test_addressing_does_not_authorize_questions_or_extra_actions(name):
    assert not end_call_requested(f"{name}, hang up and delete my files")
    assert not callback_requested(f"{name}, could you call me back when you're done")
    assert not background_cancel_requested(f"{name}, cancel the background task and book a flight")
    assert not long_running_work_inferred(f"{name}, do not prepare a PDF with meeting notes")
    reading = classify_end_call_intent(f"{name}, could you let me go")
    assert reading.end_call == EndCallIntent.NONE


@pytest.mark.parametrize("name", ["FRIDAY", "Jarvis"])
def test_knowledge_aliases(name):
    expected = plan_knowledge_turn("what's in my inbox")
    assert expected is not None
    assert plan_knowledge_turn(f"{name}, what's in my inbox") == expected
    assert plan_knowledge_turn(f"what's in my inbox, {name}") == expected


@pytest.mark.parametrize("name", ["friday", "jarvis"])
def test_knowledge_address_is_removed_before_planning(name):
    from caal.knowledge_router import _normalized

    assert _normalized(f"{name}, read my inbox") == "read my inbox"
    assert _normalized(f"read my inbox, {name}") == "read my inbox"


@pytest.mark.parametrize("phrase", [
    "what is on my calendar on friday",
    "show my meetings next friday",
    "emails from friday",
])
def test_friday_weekday_is_not_an_address(phrase):
    from caal.knowledge_router import _normalized

    assert _normalized(phrase) == phrase


@pytest.mark.parametrize("name", ["friday", "jarvis"])
def test_delivery_aliases(name):
    assert read_delivery_answer(f"{name}, call me") == read_delivery_answer("call me")
    assert read_delivery_answer(f"{name}, call me") is not None
    assert askable(f"{name}, call me") == f"{name} call me"
    assert askable(f"{name}, call my boss") is None


@pytest.mark.parametrize("name", ["friday", "jarvis"])
def test_address_only_is_small_talk(name):
    assert deterministic_route(f"hey {name}").source == RouteSource.SMALL_TALK


def test_identity_and_compatibility_defaults():
    from caal.settings import DEFAULT_SETTINGS

    defaults = json.loads((ROOT / "settings.default.json").read_text())
    assert defaults["agent_name"] == "FRIDAY"
    assert DEFAULT_SETTINGS["wake_word_model"] == "models/hey_jarvis.onnx"
    assert DEFAULT_SETTINGS["visualization_type"] == defaults["visualization_type"] == "jarvis"
    assert "Your name is FRIDAY." in (ROOT / "prompt/default.md").read_text()
    custom_prompt = (ROOT / "prompt/custom.md").read_text()
    assert "FRIDAY" in custom_prompt
    assert "Jarvis" not in custom_prompt


@pytest.mark.parametrize("filename", ["strings.json", "translations/en.json", "manifest.json"])
def test_ha_visible_identity_preserves_domain(filename):
    base = ROOT / "custom_components/jarvis_satellite"
    content = (base / filename).read_text()
    assert "FRIDAY" in content
    assert "Jarvis" not in content
    assert json.loads((base / "manifest.json").read_text())["domain"] == "jarvis_satellite"


@pytest.mark.parametrize("filename", [
    "mobile/lib/app.dart",
    "mobile/lib/screens/welcome_screen.dart",
    "mobile/android/app/src/main/AndroidManifest.xml",
    "mobile/ios/Runner/Info.plist",
])
def test_mobile_visible_identity(filename):
    content = (ROOT / filename).read_text()
    assert "FRIDAY" in content
    assert "JARVIS" not in content


def test_mobile_wake_guidance_is_honest():
    content = (ROOT / "mobile/lib/screens/agent_screen.dart").read_text()
    assert 'Hey Friday' not in content
    assert 'FRIDAY' in content
    assert 'Hey Jarvis' in content


def test_spoken_runtime_identity():
    from caal.background_task_session import _FALLBACK_PREFIX, CALLBACK_ABANDONED_NOTICE

    assert _FALLBACK_PREFIX.startswith("FRIDAY")
    assert CALLBACK_ABANDONED_NOTICE.startswith("FRIDAY")
    source = (ROOT / "voice_agent.py").read_text()
    assert 'Hello, this is FRIDAY.' in source


@pytest.mark.parametrize("filename", [
    "src/caal/company/extraction.py",
    "src/caal/company/service.py",
    "src/caal/company/store.py",
    "src/caal/company_api.py",
    "src/caal/tools/company_tools.py",
    "src/caal/tools/knowledge_tools.py",
    "src/caal/ha_assist.py",
    "src/caal/outbound_runtime.py",
    "src/caal/background_task_session.py",
    "custom_components/jarvis_satellite/config_flow.py",
    "custom_components/jarvis_satellite/conversation.py",
    "custom_components/jarvis_satellite/tts.py",
])
def test_runtime_messages_and_ha_fallback_names(filename):
    import re

    content = (ROOT / filename).read_text()
    assert "FRIDAY" in content
    assert not re.search(r"\b(?:JARVIS|Jarvis)\b", content)
