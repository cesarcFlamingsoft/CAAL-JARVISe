"""Asking the company library about a person nobody registered as a subject.

The library keeps two different things and the first repair conflated them:

* an **employee subject** is a record the owner created deliberately, with a
  display name, aliases and an identifier. Documents are attached to it, and a
  search scoped to one is scoped to exactly that person;
* a **name in the text** of an uploaded document is not a record of anybody. It
  is a word in a passage.

``company.search`` used to resolve the ``person`` argument against the subject
registry and stop there: a library with no subjects answered every question
about a named person with "nobody in the company library goes by that name",
while the name -- and, in the report that prompted this, an address next to it
-- was sitting in the uploaded text the whole time. That sentence was not a
refusal to guess; it was false.

So this is the boundary these tests pin:

* a name that resolves to **one** registered subject still scopes the search to
  that subject and to nothing else. An explicit filter never silently widens;
* a name that resolves to **two** registered subjects still returns no passage
  at all. Picking one of two people is the guess this library never makes;
* a name that resolves to **no** registered subject falls back to searching the
  document text for it, and says plainly, in the sentence it speaks, that it
  has not identified a person -- the passages merely mention the name. If the
  passages name more than one person beginning that way, the answer says so
  rather than deciding;
* a name that is in no registry *and* in no passage is told the truth about
  both, not just the registry.

The second half of the report is the prompt: a private session was answering
straightforward questions about the owner's own documents with a generic
privacy refusal, because nothing in its instructions told it that a company
library was open to it. A company-private session now carries one system
directive that says so.

Every fixture here is synthetic, and every name and address in it is invented.
"""

from __future__ import annotations

import importlib
import socket
import threading
import time
from types import SimpleNamespace

import pytest

from caal import company_privacy, profile_crypto
from caal.company import mcp_server
from caal.company.client import CompanyMcpClient
from caal.company.config import CompanyConfig
from caal.company.principal import CompanyAuthorizer, UserFacts
from caal.company.service import CompanyLibrary
from caal.internal_auth import InMemoryNonceStore
from caal.tools import company_tools
from caal.tools.registry import create_default_registry

OWNER = "usr_" + "a1" * 12
SECRET = "x" * 48
BEARER = "b" * 48

# One uploaded document that mentions a person by name, with an address beside
# the name, and no employee subject anywhere in the library for them.
DIRECTORY = (
    "FIXTURE Team Directory\n\n"
    "Section 2. Platform team\n"
    "Quill Marlow leads the platform team and is reachable at "
    "quill.marlow@fixture.example for anything about the build pipeline.\n"
)
# A second document that names a *different* person whose name begins the same
# way, so a one-word question about "Quill" is genuinely ambiguous in the text.
ROSTER = (
    "FIXTURE Support Roster\n\n"
    "Section 1. Weekend cover\n"
    "Quill Ashby covers the weekend support rotation and is reachable at "
    "quill.ashby@fixture.example.\n"
)
# A long passage whose useful clause sits past what is ever spoken aloud: the
# index quotes twenty-four tokens around a match, and these are long ones.
LONG_NOTE = (
    "FIXTURE Facilities Note\n\n"
    "Section 9. Building access\n"
    "Badgereader verification documentation supplements environmental "
    "accessibility requirements, maintenance responsibilities, housekeeping "
    "expectations, reconfiguration restrictions, identification procedures, "
    "authorisation thresholds, notification obligations, decommissioning "
    "instructions, reinstatement conditions, contingency arrangements, "
    "correspondence particulars, administrative appendices; afterwards "
    "Tamsin Okoro holds the master badge.\n"
)

_USERS = {OWNER: UserFacts(user_id=OWNER, role="admin", status="active")}


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture()
def config(tmp_path):
    return CompanyConfig.from_env(
        {
            "CAAL_COMPANY_LIBRARY_DIR": str(tmp_path / "library"),
            "CAAL_COMPANY_LIBRARY_KEYS": profile_crypto.generate_key_material(version=1),
            "CAAL_COMPANY_NAME": "FIXTURE Org",
            "CAAL_COMPANY_MCP_HOST": "127.0.0.1",
            "CAAL_COMPANY_MCP_PORT": str(_free_port()),
            "CAAL_COMPANY_MCP_TOKEN": BEARER,
            "CAAL_INTERNAL_AUTH_SECRET": SECRET,
            "CAAL_COMPANY_OWNER_USER_ID": OWNER,
            "CAAL_COMPANY_ROLE": "owner",
        }
    )


@pytest.fixture()
def library(config):
    service = CompanyLibrary(config)
    service.ingest(
        owner=OWNER,
        filename="FIXTURE-team-directory.txt",
        data=DIRECTORY.encode(),
        title="FIXTURE Team Directory",
        classification="general",
        status="current",
    )
    try:
        yield service
    finally:
        service.close()


@pytest.fixture()
def service(config, library):
    import uvicorn

    app = mcp_server.build_app(
        library=library,
        authorizer=CompanyAuthorizer(
            config=config, resolve_user=_USERS.get, nonce_store=InMemoryNonceStore()
        ),
        config=config,
    )
    running = uvicorn.Server(
        uvicorn.Config(
            app, host=config.mcp_host, port=config.mcp_port, log_level="warning", access_log=False
        )
    )
    thread = threading.Thread(target=running.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not running.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert running.started
    try:
        yield config
    finally:
        running.should_exit = True
        thread.join(timeout=10)


@pytest.fixture()
def tools(service):
    client = CompanyMcpClient(config=service)
    company_tools.configure(lambda: client)
    try:
        yield company_tools
    finally:
        company_tools.reset()


# --- a name nobody registered ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unregistered_name_is_looked_for_in_the_document_text(tools):
    """The report, in fixture form: the name is in an uploaded file, not in a registry."""
    result = await tools.search(
        user_id=OWNER, query="what is the email address of Quill Marlow", person="Quill Marlow"
    )
    assert result["status"] == "ok"
    assert result["data"]["citations"], "a passage that mentions the name is evidence"
    assert "quill.marlow@fixture.example" in result["message"]
    assert "FIXTURE Team Directory" in result["message"]


@pytest.mark.asyncio
async def test_the_answer_says_the_person_was_never_identified(tools):
    result = await tools.search(
        user_id=OWNER, query="email address of Quill Marlow", person="Quill Marlow"
    )
    identity = result["data"]["person_identity"]
    assert identity["verified"] is False
    assert identity["resolution"] == "text_mention"
    assert "registered" in result["message"].lower()
    assert "mention" in result["message"].lower()


@pytest.mark.asyncio
async def test_a_name_in_no_registry_and_no_passage_is_told_the_truth_about_both(tools):
    result = await tools.search(
        user_id=OWNER, query="job title", person="Fixture Person Nobody"
    )
    assert result["status"] == "no_person_match"
    assert result["data"]["citations"] == []
    spoken = result["message"].lower()
    assert "registered" in spoken
    assert "mention" in spoken or "text" in spoken


@pytest.mark.asyncio
async def test_more_than_one_person_in_the_text_is_flagged_not_decided(library, tools):
    library.ingest(
        owner=OWNER,
        filename="FIXTURE-support-roster.txt",
        data=ROSTER.encode(),
        title="FIXTURE Support Roster",
        classification="general",
        status="current",
    )
    result = await tools.search(user_id=OWNER, query="email address", person="Quill")
    identity = result["data"]["person_identity"]
    assert identity["verified"] is False
    assert len(identity["variants"]) == 2
    assert "more than one" in result["message"].lower()


# --- what an explicit registered filter still means ------------------------------------------


@pytest.mark.asyncio
async def test_a_registered_name_still_scopes_to_that_subject_and_nothing_else(library, tools):
    """An explicit subject filter must never quietly widen into a text search."""
    person = library.add_subject(owner=OWNER, display_name="Tamsin Okoro")
    library.ingest(
        owner=OWNER,
        filename="FIXTURE-offer-tamsin.txt",
        data=b"FIXTURE Offer Letter. Tamsin Okoro holds the title Staff Engineer.",
        title="FIXTURE Offer Letter Tamsin",
        classification="hr",
        status="current",
        subjects=(person.subject_id,),
    )
    library.ingest(
        owner=OWNER,
        filename="FIXTURE-unattached-note.txt",
        data=b"FIXTURE Unattached Note. Tamsin Okoro was mentioned at the title review.",
        title="FIXTURE Unattached Note",
        classification="general",
        status="current",
    )
    result = await tools.search(user_id=OWNER, query="title", person="Tamsin Okoro")
    assert result["status"] == "ok"
    titles = {citation["title"] for citation in result["data"]["citations"]}
    assert titles == {"FIXTURE Offer Letter Tamsin"}
    assert result["data"]["person_identity"]["verified"] is True
    assert result["data"]["person_identity"]["resolution"] == "registered_subject"


@pytest.mark.asyncio
async def test_two_registered_people_with_one_name_still_return_no_passage(library, tools):
    library.add_subject(owner=OWNER, display_name="Quill Marlow")
    library.add_subject(owner=OWNER, display_name="Quill Marlow")
    result = await tools.search(user_id=OWNER, query="email address", person="Quill Marlow")
    assert result["status"] == "ambiguous_person"
    assert result["data"]["citations"] == []


# --- nothing is cut out of an answer without saying so ---------------------------------------


@pytest.mark.asyncio
async def test_a_passage_longer_than_the_spoken_limit_says_it_was_cut(library, tools):
    library.ingest(
        owner=OWNER,
        filename="FIXTURE-facilities-note.txt",
        data=LONG_NOTE.encode(),
        title="FIXTURE Facilities Note",
        classification="general",
        status="current",
    )
    result = await tools.search(user_id=OWNER, query="badgereader verification documentation")
    assert result["status"] == "ok"
    snippet = result["data"]["citations"][0]["snippet"]
    assert len(snippet) > company_tools.MAX_SPOKEN_SNIPPET
    assert company_tools.TRUNCATION_NOTICE in result["message"]


# --- what a company-private session is told ---------------------------------------------------


class _Scope:
    def __init__(self, user_id: str | None = OWNER) -> None:
        self.user_id = user_id
        self.identity_configured = True
        self.memory_available = True


@pytest.fixture(autouse=True)
def clean_privacy_state(monkeypatch):
    company_privacy.reset()
    monkeypatch.delenv("CAAL_COMPANY_PRIVATE_MODE", raising=False)
    yield
    company_privacy.reset()


@pytest.fixture()
def owner_library(monkeypatch):
    from caal.company import runtime

    class _Config:
        owner_user_id = OWNER

    monkeypatch.setattr(runtime, "get_config", lambda: _Config())
    return _Config()


def _private_agent():
    agent = SimpleNamespace(_user_scope=_Scope(), _native_tool_registry=create_default_registry())
    company_privacy.bind_session(agent, requested=True)
    return agent


def test_a_private_session_carries_a_directive_about_its_own_library(owner_library):
    agent = _private_agent()
    company_privacy.begin_turn(agent, "what is the email of Quill")
    directive = company_privacy.private_session_directive(agent)
    assert directive is not None
    lowered = directive.lower()
    assert "company.search" in lowered
    # It must not refuse on privacy grounds, and must not ask to be allowed.
    assert "permission" in lowered
    assert "refuse" in lowered
    # And it must not claim the library has no record of anyone without looking.
    assert "no record" in lowered or "nobody" in lowered


def test_an_ordinary_owner_session_carries_no_such_directive(owner_library):
    agent = SimpleNamespace(_user_scope=_Scope())
    company_privacy.begin_turn(agent, "what time is it")
    assert company_privacy.private_session_directive(agent) is None


def test_a_session_that_is_not_the_owners_carries_no_such_directive(owner_library):
    agent = SimpleNamespace(_user_scope=_Scope("usr_" + "d4" * 12))
    company_privacy.bind_session(agent, requested=True)
    company_privacy.begin_turn(agent, "what is the email of Quill")
    assert company_privacy.private_session_directive(agent) is None


class ChatMessage:
    def __init__(self, role: str, text: str) -> None:
        self.role = role
        self.text_content = text


class _CapturingLocalModel:
    """The local model, remembering the system prompt it was actually given."""

    provider_name = "fixture-local"
    model = "fixture-local-model"
    manages_own_tools = False
    supports_think = False

    def __init__(self) -> None:
        self.system: list[str] = []

    async def chat(self, messages, tools=None, **_):
        self.system.extend(m["content"] for m in messages if m.get("role") == "system")
        return SimpleNamespace(content=None, tool_calls=[])

    async def chat_stream(self, messages, tools=None, **_):
        self.system.extend(m["content"] for m in messages if m.get("role") == "system")
        yield "FIXTURE local answer."

    def format_tool_call_message(self, content, tool_calls):
        return dict(role="assistant", content=content or "")

    def format_tool_result(self, content, tool_call_id, tool_name):
        return dict(role="tool", content=content, tool_call_id=tool_call_id)


@pytest.mark.asyncio
async def test_the_directive_reaches_the_model_that_chooses_the_tools(owner_library):
    agent = _private_agent()
    chat_ctx = SimpleNamespace(
        items=[
            ChatMessage("system", "FIXTURE base prompt."),
            ChatMessage("user", "what is the email of Quill"),
        ]
    )
    provider = _CapturingLocalModel()
    node = importlib.import_module("caal.llm.llm_node")
    async for _ in node.llm_node(agent, chat_ctx, provider=provider):
        pass
    joined = "\n".join(provider.system)
    assert "FIXTURE base prompt." in joined, "the session directive adds, never replaces"
    assert company_privacy.PRIVATE_SESSION_DIRECTIVE in joined


@pytest.mark.asyncio
async def test_an_ordinary_session_prompt_is_left_alone(owner_library):
    agent = SimpleNamespace(_user_scope=_Scope(), _native_tool_registry=create_default_registry())
    chat_ctx = SimpleNamespace(
        items=[
            ChatMessage("system", "FIXTURE base prompt."),
            ChatMessage("user", "what time is it"),
        ]
    )
    provider = _CapturingLocalModel()
    node = importlib.import_module("caal.llm.llm_node")
    async for _ in node.llm_node(agent, chat_ctx, provider=provider):
        pass
    joined = "\n".join(provider.system)
    assert company_privacy.PRIVATE_SESSION_DIRECTIVE not in joined


# --- what the model is actually told about the person argument -------------------------------
#
# Found by replaying the reported turns against the real deployed runtime (the
# production directive, the real private-session catalogue, the real adapter and
# the configured local model). Both turns behaved, and they differed in exactly
# one thing that mattered:
#
# * the first turn called ``company.search`` with **only** ``query``. Isolated
#   afterwards with the model taken out of the loop, a query-only search for
#   that question returns passages from other documents and **no** passage
#   carrying the answer -- keyword ranking puts it out of reach;
# * the follow-up called it with ``person`` as well. That is the unregistered
#   name path: ``resolution: text_mention``, ``verified: false``, and the
#   passage carrying the answer comes back and is quoted in the spoken message.
#
# So the retrieval was never broken and the registry was never the problem. The
# model simply was not told to fill ``person`` in, because the schema advertised
# it as a rare optional *employee* filter -- and the person asked about is
# exactly the kind of person who is not a registered employee. These tests hold
# the advertised wording to what the run showed it has to say.


def _person_schema():
    tool = create_default_registry().get("company.search")
    return tool.parameters["properties"]["person"]


def test_the_person_argument_is_advertised_for_any_named_person(owner_library):
    """"Optional employee name" is why the first turn omitted it.

    The name in an uploaded directory or roster is usually *not* a registered
    employee, so a description scoped to employees reads as "not this one".
    """
    description = _person_schema()["description"].lower()
    assert "employee" not in description or "not" in description, description
    assert "name" in description


def test_the_person_argument_says_to_fill_it_in_whenever_a_name_was_asked_about(owner_library):
    """The retrieval that works has to be the one the model reaches for first."""
    description = _person_schema()["description"].lower()
    assert "whenever" in description or "always" in description, description
    # Naming somebody is the trigger, and it is stated as such.
    assert "named" in description or "names" in description


def test_the_person_argument_still_refuses_to_pick_between_two_people(owner_library):
    """The ambiguity guard is part of the advertised contract and stays there."""
    description = _person_schema()["description"].lower()
    assert "do not pick one" in description


def test_the_search_description_sends_a_named_person_question_to_that_argument(owner_library):
    """The tool description and the argument description must not disagree."""
    tool = create_default_registry().get("company.search")
    assert "person" in tool.description.lower()
