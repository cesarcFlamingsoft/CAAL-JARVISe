"""The wiring of cross-language company search into a real session lifecycle.

`test_company_crosslanguage.py` pins what the expansion *does* with an injected
translator. This file pins the thing that was missing: which provider that
translator is allowed to be, when it is bound, when it is released, and who is
allowed to reach it.

Five properties, each one a way the feature could be wrong in production while
every offline expansion test still passed:

* the translator is the **directly configured local model** -- a routed
  provider is unwrapped to its primary and never called itself, and a
  non-local provider binds nothing at all;
* the endpoint is re-validated against the local-only allowance, and anything
  else **fails closed** to the search that existed before;
* the binding is **session-scoped**: two sessions do not share one, and
  neither one's teardown unbinds the other;
* a session that is **not the library owner** is refused *before* the question
  is composed, so nothing leaves;
* the bounds asked of the API (`think=False`, `num_predict`) are really sent,
  and an option a provider does not declare is not sent at all.

Nothing here reaches a network, a model, or a document.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from caal.company import expansion_runtime, query_expansion  # noqa: E402
from caal.tools import company_tools  # noqa: E402

OWNER = "user-owner"
STRANGER = "user-stranger"
REPLY = '{"en": "notice period", "es": "plazo de preaviso"}'


# --- stand-ins for the provider objects the runtime actually passes around ----------------


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLocal:
    """The shape of ``OllamaProvider``: declares what its ``chat`` reads."""

    provider_name = "ollama"
    supported_chat_options = frozenset({"think", "num_predict"})

    def __init__(self, base_url: str | None = "http://127.0.0.1:11434", reply: str = REPLY):
        self.base_url = base_url
        self._reply = reply
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools, "kwargs": dict(kwargs)})
        return FakeResponse(self._reply)


class FakeUndeclared(FakeLocal):
    """A local provider that does not say which options it reads."""

    supported_chat_options = None  # type: ignore[assignment]


class FakeCloud:
    """Hermes, Groq -- anything that is not the local model."""

    provider_name = "hermes"
    base_url = "https://api.example.invalid"
    supported_chat_options = frozenset({"think"})

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        raise AssertionError("a company question was sent to a non-local provider")


class FakeRouted:
    """``RoutedProvider``: may escalate, so it must never be the translator."""

    provider_name = "routed"

    def __init__(self, primary: Any, escalation: Any) -> None:
        self.primary = primary
        self.escalation = escalation
        self.calls = 0

    async def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        raise AssertionError("a company question was sent through the routed provider")


class FakeCAALLLM:
    """The LiveKit wrapper the entrypoint holds; the runtime passes *this*."""

    def __init__(self, provider_instance: Any) -> None:
        self.provider_instance = provider_instance

    async def chat(self, *args, **kwargs):
        raise AssertionError("a company question was sent through the LiveKit wrapper")


@pytest.fixture(autouse=True)
def _unbound():
    query_expansion.reset()
    yield
    query_expansion.reset()


# --- which provider may translate ---------------------------------------------------------


def test_the_routed_provider_is_unwrapped_to_its_local_primary():
    local = FakeLocal()
    routed = FakeRouted(primary=local, escalation=FakeCloud())
    assert expansion_runtime.local_provider(FakeCAALLLM(routed)) is local


def test_a_bare_local_provider_is_itself():
    local = FakeLocal()
    assert expansion_runtime.local_provider(local) is local


def test_a_non_local_provider_binds_nothing():
    assert expansion_runtime.local_translator(FakeCloud()) is None
    assert expansion_runtime.local_translator(FakeCAALLLM(FakeCloud())) is None


def test_no_provider_at_all_binds_nothing():
    assert expansion_runtime.local_provider(None) is None
    assert expansion_runtime.local_translator(None) is None


@pytest.mark.asyncio
async def test_the_routed_provider_and_the_escalation_are_never_called():
    local = FakeLocal()
    cloud = FakeCloud()
    routed = FakeRouted(primary=local, escalation=cloud)
    with expansion_runtime.session_expansion(FakeCAALLLM(routed)):
        variants = await query_expansion.expand("plazo de preaviso", user_id=OWNER)
    # Both stand-ins raise if called; these assertions say so out loud anyway.
    assert routed.calls == 0
    assert cloud.calls == 0
    assert len(local.calls) == 1
    assert variants[0] == "plazo de preaviso"
    assert "notice period" in variants


# --- the endpoint is re-validated, and fails closed ---------------------------------------


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://8.8.8.8:11434",  # public
        "https://localhost:11434",  # not plain http
        "http://user:pw@127.0.0.1:11434",  # credentials
        "http://127.0.0.1:11434/v1/chat",  # a path
        "http://example.com:11434",  # a name someone else controls
        "http://169.254.169.254:80",  # cloud metadata
        "not a url",
    ],
)
def test_an_unapproved_endpoint_binds_nothing(endpoint):
    assert expansion_runtime.local_translator(FakeLocal(base_url=endpoint)) is None


@pytest.mark.parametrize(
    "endpoint",
    ["http://localhost:11434", "http://host.docker.internal:11434", "http://10.0.0.64:11434"],
)
def test_an_approved_local_endpoint_binds(endpoint):
    assert expansion_runtime.local_translator(FakeLocal(base_url=endpoint)) is not None


def test_no_base_url_falls_back_to_the_configured_endpoint_and_is_validated():
    assert (
        expansion_runtime.local_translator(
            FakeLocal(base_url=None), settings={"ollama_host": "http://192.168.1.50:11434"}
        )
        is not None
    )
    # A refused saved endpoint is passed over by `configured_endpoint`, which
    # falls back to the local default -- still local, so still bound.
    assert (
        expansion_runtime.local_translator(
            FakeLocal(base_url=None), settings={"ollama_host": "https://evil.example:443"}
        )
        is not None
    )


@pytest.mark.asyncio
async def test_a_refused_endpoint_leaves_the_search_exactly_as_it_was():
    with expansion_runtime.session_expansion(FakeLocal(base_url="http://8.8.8.8:11434")):
        assert query_expansion.enabled() is False
        assert await query_expansion.expand("plazo de preaviso", user_id=OWNER) == [
            "plazo de preaviso"
        ]


# --- the session lifecycle ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_binding_is_released_when_the_session_ends():
    with expansion_runtime.session_expansion(FakeLocal()):
        assert query_expansion.enabled() is True
    assert query_expansion.enabled() is False
    assert query_expansion.get_expander() is None


@pytest.mark.asyncio
async def test_a_failed_session_still_releases_its_binding():
    with pytest.raises(RuntimeError):
        with expansion_runtime.session_expansion(FakeLocal()):
            raise RuntimeError("the session fell over")
    assert query_expansion.enabled() is False


@pytest.mark.asyncio
async def test_two_sessions_do_not_share_or_unbind_each_other():
    """The reason this is a ContextVar and not a module global.

    Session A binds a local model, session B binds nothing (its endpoint is
    refused). Each must see its own binding, and B's teardown -- which with a
    process-global would call `reset()` on the one and only expander -- must
    leave A's alone while A is still in the middle of a turn.
    """
    a_local = FakeLocal(reply='{"en": "session a", "es": "sesion a"}')
    started_b = asyncio.Event()
    finished_b = asyncio.Event()

    async def session_a() -> list[str]:
        with expansion_runtime.session_expansion(a_local):
            assert query_expansion.enabled() is True
            started_b.set()
            await finished_b.wait()  # B has bound, expanded and torn down
            assert query_expansion.enabled() is True, "session B unbound session A"
            return await query_expansion.expand("a question", user_id=OWNER)

    async def session_b() -> bool:
        await started_b.wait()
        with expansion_runtime.session_expansion(FakeLocal(base_url="http://8.8.8.8:11434")):
            seen = query_expansion.enabled()
        finished_b.set()
        return seen

    a_result, b_enabled = await asyncio.gather(session_a(), session_b())
    assert b_enabled is False, "session B saw session A's translator"
    assert "session a" in a_result
    assert a_local.calls, "session A's own provider was not the one that answered it"


@pytest.mark.asyncio
async def test_two_sessions_do_not_share_a_cached_question():
    first = FakeLocal(reply='{"en": "first", "es": "primero"}')
    second = FakeLocal(reply='{"en": "second", "es": "segundo"}')

    async def run(provider: FakeLocal) -> list[str]:
        with expansion_runtime.session_expansion(provider):
            return await query_expansion.expand("la misma pregunta", user_id=OWNER)

    assert "first" in await run(first)
    assert "second" in await run(second)
    assert len(second.calls) == 1, "the second session reused the first session's cache"


# --- authorization, before anything is composed or sent -----------------------------------


@pytest.mark.asyncio
async def test_a_session_that_is_not_the_owner_is_refused_before_the_translator():
    local = FakeLocal()
    with expansion_runtime.session_expansion(local, authorize=lambda uid: uid == OWNER):
        assert await query_expansion.expand("plazo de preaviso", user_id=STRANGER) == [
            "plazo de preaviso"
        ]
    assert local.calls == [], "an unauthorized question was sent to the translator"


@pytest.mark.asyncio
async def test_an_anonymous_session_is_refused_before_the_translator():
    local = FakeLocal()
    with expansion_runtime.session_expansion(local, authorize=lambda uid: uid == OWNER):
        assert await query_expansion.expand("plazo de preaviso", user_id=None) == [
            "plazo de preaviso"
        ]
    assert local.calls == []


@pytest.mark.asyncio
async def test_the_owner_is_allowed_through_the_same_gate():
    local = FakeLocal()
    with expansion_runtime.session_expansion(local, authorize=lambda uid: uid == OWNER):
        variants = await query_expansion.expand("plazo de preaviso", user_id=OWNER)
    assert "notice period" in variants
    assert len(local.calls) == 1


@pytest.mark.asyncio
async def test_a_broken_authorization_is_a_closed_one():
    local = FakeLocal()

    def explode(_uid):
        raise RuntimeError("the privacy state could not be read")

    with expansion_runtime.session_expansion(local, authorize=explode):
        assert await query_expansion.expand("plazo de preaviso", user_id=OWNER) == [
            "plazo de preaviso"
        ]
    assert local.calls == []


@pytest.mark.asyncio
async def test_authorization_is_re_asked_per_call_not_cached_at_bind_time():
    local = FakeLocal()
    allowed = {"now": False}
    with expansion_runtime.session_expansion(local, authorize=lambda _uid: allowed["now"]):
        assert await query_expansion.expand("plazo de preaviso", user_id=OWNER) == [
            "plazo de preaviso"
        ]
        allowed["now"] = True  # the session engaged company mode mid-conversation
        assert "notice period" in await query_expansion.expand("plazo de preaviso", user_id=OWNER)
    assert len(local.calls) == 1


# --- the bounds asked of the API ----------------------------------------------------------


@pytest.mark.asyncio
async def test_think_false_and_a_token_cap_are_actually_sent():
    local = FakeLocal()
    with expansion_runtime.session_expansion(local):
        await query_expansion.expand("plazo de preaviso", user_id=OWNER)
    kwargs = local.calls[0]["kwargs"]
    assert kwargs["think"] is False
    assert kwargs["num_predict"] == query_expansion.MAX_TRANSLATION_TOKENS
    assert local.calls[0]["tools"] is None, "the translator must never be offered tools"


@pytest.mark.asyncio
async def test_an_option_a_provider_does_not_declare_is_not_sent():
    """The alternative -- passing it into ``**kwargs`` and hoping -- is a bound
    that looks real in the code and does nothing at the API."""
    local = FakeUndeclared()
    with expansion_runtime.session_expansion(local):
        await query_expansion.expand("plazo de preaviso", user_id=OWNER)
    assert local.calls[0]["kwargs"] == {}


def test_the_real_local_provider_declares_the_options_the_translator_needs():
    from caal.llm.providers.ollama_provider import OllamaProvider

    assert {"think", "num_predict"} <= set(
        query_expansion.supported_chat_options(OllamaProvider)
    )


def test_the_real_local_provider_puts_the_token_cap_in_its_request_options():
    from caal.llm.providers.ollama_provider import OllamaProvider

    provider = OllamaProvider.__new__(OllamaProvider)
    provider._temperature, provider._top_p = 0.7, 0.8
    provider._top_k, provider._num_ctx = 20, 8192
    assert "num_predict" not in provider._get_options()
    assert provider._get_options(160)["num_predict"] == 160


# --- the production configuration, not a probe's ------------------------------------------


@pytest.mark.asyncio
async def test_a_bound_session_uses_the_production_timeout_by_default():
    with expansion_runtime.session_expansion(FakeLocal()) as expander:
        assert expander is not None
        assert expander.timeout_seconds == query_expansion.DEFAULT_TIMEOUT_SECONDS == 4.0


@pytest.mark.asyncio
async def test_a_translator_slower_than_the_production_timeout_costs_only_the_expansion():
    class Slow(FakeLocal):
        async def chat(self, messages, tools=None, **kwargs):
            await asyncio.sleep(0.5)
            return FakeResponse(REPLY)

    with expansion_runtime.session_expansion(Slow(), timeout_seconds=0.05) as expander:
        assert await query_expansion.expand("plazo de preaviso", user_id=OWNER) == [
            "plazo de preaviso"
        ]
        assert expander is not None
        assert expander.last_latency_seconds is not None


@pytest.mark.asyncio
async def test_an_abandoned_turn_cancels_the_translation_rather_than_swallowing_it():
    class Hanging(FakeLocal):
        async def chat(self, messages, tools=None, **kwargs):
            await asyncio.sleep(30)
            raise AssertionError("unreachable")

    with expansion_runtime.session_expansion(Hanging(), timeout_seconds=10.0):
        task = asyncio.create_task(query_expansion.expand("plazo de preaviso", user_id=OWNER))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# --- through the real tool seam -----------------------------------------------------------


class RecordingClient:
    """Answers `company_search` and records the arguments dict it was given."""

    def __init__(self) -> None:
        self.searches: list[dict[str, Any]] = []

    async def call(self, tool: str, arguments: dict[str, Any], subject: str | None = None):
        assert tool == "company_search"
        self.searches.append(dict(arguments))
        return {"status": "ok", "results": [{"document_id": arguments["query"], "snippet": "x"}]}


@pytest.mark.asyncio
async def test_the_wired_seam_preserves_every_filter_and_only_replaces_the_query():
    client = RecordingClient()
    arguments = {
        "query": "plazo de preaviso de acme_2024 con ana@example.com",
        "classification": "contract",
        "subject_user_id": "subject-42",
        "an_undeclared_filter": "kept",
    }
    local = FakeLocal(reply='{"en": "notice period", "es": "plazo de preaviso"}')
    with expansion_runtime.session_expansion(local, authorize=lambda uid: uid == OWNER):
        results, status = await company_tools._cross_language(
            client,
            OWNER,
            arguments,
            arguments["query"],
            None,
            [{"document_id": "original", "snippet": "y"}],
            "ok",
        )

    assert client.searches, "the expansion never reached the search service"
    for sent in client.searches:
        assert sent["classification"] == "contract"
        assert sent["subject_user_id"] == "subject-42"
        assert sent["an_undeclared_filter"] == "kept"
        # Identifiers are not things to translate.
        assert "acme_2024" in sent["query"] and "ana@example.com" in sent["query"]
    assert results[0]["document_id"] == "original", "the original's top hit stopped being first"
    assert status == "ok"


@pytest.mark.asyncio
async def test_the_wired_seam_sends_nothing_for_a_session_that_is_not_the_owner():
    client = RecordingClient()
    local = FakeLocal()
    arguments = {"query": "plazo de preaviso", "classification": "contract"}
    with expansion_runtime.session_expansion(local, authorize=lambda uid: uid == OWNER):
        results, status = await company_tools._cross_language(
            client,
            STRANGER,
            arguments,
            arguments["query"],
            None,
            [{"document_id": "original", "snippet": "y"}],
            "ok",
        )
    assert local.calls == [], "a stranger's question was sent to the translator"
    assert client.searches == [], "a stranger's session ran an extra search"
    assert results == [{"document_id": "original", "snippet": "y"}]
    assert status == "ok"


@pytest.mark.asyncio
async def test_an_unbound_session_reaches_the_seam_and_does_nothing():
    client = RecordingClient()
    original = [{"document_id": "original", "snippet": "y"}]
    results, status = await company_tools._cross_language(
        client, OWNER, {"query": "q"}, "q", None, original, "ok"
    )
    assert client.searches == []
    assert results is original and status == "ok"
