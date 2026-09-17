"""Stage 2: the REAL local model, asked in English and in Spanish.

FINAL-STAGE1 §3.2 called this the largest untested assumption in the change: the
reply-language directive was written and injected, but nothing proved the model
obeys it, picks the same tools, or leaves identifiers alone.

What this does and does not do:

* It talks to the real configured Ollama endpoint with the real `gemma4:e4b`
  model, the real tool catalog from `create_default_registry()`, and the real
  directive from `caal.language_policy.reply_directive`.
* It never executes a tool handler, so no email, calendar, reminder, alarm or
  home-control action can happen. Only the model's *choice* is inspected.
* Every tool result it feeds back is a hand-written synthetic fixture. No real
  company document, mailbox, calendar or contact is read or sent anywhere, and
  nothing leaves the local network: the endpoint is the LAN Ollama host.
* It skips, rather than fails, when the endpoint is unreachable, so it can live
  in the normal suite.

Language models are not deterministic. Assertions are therefore about the
contract that matters — same tool, right language, identifiers untouched — with
a small retry budget, not about exact wording.
"""

import asyncio
import json
import os
import urllib.request

import pytest

from caal.language_policy import ES, reply_directive
from caal.llm.llm_node import _with_directive
from caal.llm.providers.ollama_provider import OllamaProvider
from caal.tools.registry import create_default_registry

HOST = os.environ.get("CAAL_TEST_OLLAMA_HOST", "http://10.0.0.64:11434")
MODEL = os.environ.get("CAAL_TEST_OLLAMA_MODEL", "gemma4:e4b")
ATTEMPTS = 3


#: Opt-in only. Without it this module makes no network call at all — not even a
#: reachability probe — so an ordinary `pytest tests/` collection never touches
#: the LAN. Run with CAAL_TEST_LOCAL_MODEL=1 to exercise the real model.
OPT_IN = os.environ.get("CAAL_TEST_LOCAL_MODEL") == "1"


def _reachable():
    if not OPT_IN:
        return False
    try:
        with urllib.request.urlopen(f"{HOST}/api/tags", timeout=5) as response:
            names = {m["name"] for m in json.loads(response.read())["models"]}
        return MODEL in names
    except Exception:  # noqa: BLE001 - an unreachable LAN host is a skip, not a failure
        return False


pytestmark = pytest.mark.skipif(
    not _reachable(),
    reason=(
        f"set CAAL_TEST_LOCAL_MODEL=1 and make {MODEL} reachable at {HOST}"
        if not OPT_IN
        else f"local model {MODEL} not reachable at {HOST}"
    ),
)

READ_ONLY = {
    "email.search",
    "email.read",
    "inbox.recent",
    "inbox.search",
    "inbox.read_summary",
    "calendar.list_events",
    "calendar.find_free_time",
    "schedule.upcoming",
    "schedule.next",
    "schedule.find_event",
    "reminders.list",
    "memory.recall",
    "company.search",
    "company.read",
}

SYSTEM = (
    "You are JARVIS, a voice assistant. Use the provided tools to answer questions about "
    "the user's email, calendar, reminders and company documents. Call exactly one tool "
    "when the user asks for information you do not already have."
)


def catalog():
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in create_default_registry().list()
    ]


def provider():
    return OllamaProvider(model=MODEL, base_url=HOST, think=False, temperature=0.0, num_ctx=32768)


def ask(utterance, language, tools=None, extra=None):
    """One real turn. Returns the LLMResponse."""
    messages = [{"role": "system", "content": SYSTEM}, *(extra or [])]
    if utterance is not None:
        messages.append({"role": "user", "content": utterance})
    directive = reply_directive(language)
    if directive:
        messages = _with_directive(messages, directive)
    return asyncio.run(provider().chat(messages=messages, tools=tools))


def tool_names(response):
    return [call.name for call in response.tool_calls]


def first_tool(utterance, language, *, attempts=ATTEMPTS):
    """The tool the model picks, retried a little against sampling noise."""
    seen = []
    for _ in range(attempts):
        names = tool_names(ask(utterance, language, tools=catalog()))
        if names:
            return names[0], seen
        seen.append(names)
    return None, seen


# --- 1. The same question in either language picks the same read-only tool ---

PAIRS = [
    ("what is on my calendar today?", "¿qué tengo en el calendario hoy?"),
    ("read me my most recent emails", "léeme mis correos más recientes"),
    ("what is my next meeting?", "¿cuál es mi próxima reunión?"),
]


@pytest.mark.parametrize(("english", "spanish"), PAIRS)
def test_spanish_selects_the_same_read_only_tool_as_english(english, spanish):
    """Parity, which is what the Spanish directive can be held responsible for.

    Whether this model calls a tool *at all* for a given phrasing is a property
    of the model and the prompt, not of the language directive: measured 5/5,
    `what is on my calendar today?` selects nothing in English while the Spanish
    form selects `schedule.upcoming`
    (`reports/bilingual/evidence/stage2-local-model-tool-choice.json`). So the
    assertion is the one this stage is actually about — Spanish must never be
    worse than the English baseline, and must never reach a different tool.
    """
    en_tool, _ = first_tool(english, "en")
    es_tool, es_misses = first_tool(spanish, ES)
    if en_tool is None:
        # English baseline selects nothing; Spanish may only match or do better.
        assert es_tool is None or es_tool in READ_ONLY, es_tool
        return
    assert en_tool in READ_ONLY, en_tool
    assert es_tool is not None, f"English chose {en_tool} but Spanish chose nothing: {es_misses}"
    assert es_tool == en_tool, f"{english!r} -> {en_tool}, {spanish!r} -> {es_tool}"


def test_the_spanish_directive_does_not_translate_the_tool_names_it_calls():
    names = create_default_registry().names()
    for _, spanish in PAIRS:
        response = ask(spanish, ES, tools=catalog())
        for call in response.tool_calls:
            assert call.name in names, call.name


# --- 2. The post-tool follow-up: right language, identifiers verbatim --------

#: Entirely invented. No real person, address, domain or document.
FIXTURE = {
    "messages": [
        {
            "id": "MSG-8841-Z",
            "from": "dana.whitfield@northgate-labs.example",
            "subject": "Northgate Q3 rollout checklist",
            "received": "2026-09-15T14:05:00Z",
            "link": "https://mail.example.invalid/m/MSG-8841-Z",
            "snippet": "The staging cutover is scheduled for Thursday at 09:00.",
        }
    ]
}
IDENTIFIERS = [
    "MSG-8841-Z",
    "dana.whitfield@northgate-labs.example",
    "https://mail.example.invalid/m/MSG-8841-Z",
    "Northgate",
]


ASKS = {"es": "léeme mi correo más reciente", "en": "read me my most recent email"}


def followup(language):
    """The spoken answer the model produces after a read-only tool result."""
    extra = [
        {"role": "user", "content": ASKS[language]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "inbox.recent", "arguments": {"limit": 1}}}
            ],
        },
        {"role": "tool", "name": "inbox.recent", "content": json.dumps(FIXTURE)},
    ]
    return ask(None, language, extra=extra)


def _spanish_enough(text):
    from caal.language_policy import _looks_spanish, is_evidential

    return is_evidential(text) and _looks_spanish(text)


def test_the_spanish_followup_answers_in_spanish():
    failures = []
    for _ in range(ATTEMPTS):
        content = (followup(ES).content or "").strip()
        if _spanish_enough(content):
            return
        failures.append(content[:200])
    pytest.fail(f"no Spanish reply in {ATTEMPTS} attempts: {failures}")


def test_the_spanish_followup_keeps_identifiers_and_addresses_verbatim():
    """At least one attempt must repeat every identifier it chooses to mention."""
    best = None
    for _ in range(ATTEMPTS):
        content = followup(ES).content or ""
        mangled = [
            value for value in IDENTIFIERS if _mentions(content, value) and value not in content
        ]
        if not mangled:
            return
        best = (mangled, content[:400])
    pytest.fail(f"identifiers altered: {best}")


def _mentions(content, value):
    """Whether the reply tried to say this identifier at all (in any casing)."""
    if "@" in value:
        return "northgate-labs" in content.lower() or "dana" in content.lower()
    if value.startswith("http"):
        return "mail.example" in content.lower()
    return value.lower() in content.lower()


def test_the_english_followup_is_unchanged_by_this_work():
    content = (followup("en").content or "").strip()
    assert content
    assert not _spanish_enough(content)


# --- 3. Confirmation safety survives the Spanish directive -------------------

CONFIRMED = {
    name
    for name, tool in ((t.name, t) for t in create_default_registry().list())
    if tool.requires_confirmation
}


def test_a_spanish_request_to_send_email_never_reaches_an_unconfirmed_tool():
    """No handler runs here. Only the *selection* is inspected.

    Stated as a negative invariant so it asserts something in every outcome,
    including the measured one: this model selects no tool at all for this ask,
    5/5, in both languages
    (`reports/bilingual/evidence/stage2-local-model-tool-choice.json`). That
    makes this weak evidence about `email.send` specifically — it is evidence
    that the Spanish directive does not *create* an unconfirmed mutation, not
    evidence that the confirmation flow was exercised.
    """
    registry = create_default_registry()
    selected = []
    for _ in range(ATTEMPTS):
        names = tool_names(
            ask(
                "manda un correo a dana.whitfield@northgate-labs.example "
                "diciendo que llego tarde",
                ES,
                tools=catalog(),
            )
        )
        selected.extend(names)
        for name in names:
            tool = registry.get(name)
            assert tool is not None, name
            if name not in READ_ONLY:
                assert tool.requires_confirmation, (
                    f"Spanish request selected the unconfirmed mutating tool {name}"
                )
    # Recorded so the report cannot overstate what this run proved.
    assert all(registry.get(name) is not None for name in selected)


def test_the_confirmation_flags_are_language_independent():
    """The safety contract is data, not prose, so Spanish cannot weaken it."""
    assert "email.send" in CONFIRMED
    assert {"calendar.create_event", "calendar.update_event", "calendar.delete_event"} <= CONFIRMED
    for name in READ_ONLY:
        tool = create_default_registry().get(name)
        if tool is not None:
            assert not tool.requires_confirmation, name
