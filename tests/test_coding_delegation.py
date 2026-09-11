"""A coding request is delegated to Hermes, never run inside the CAAL container.

The CAAL agent image carries neither Claude Code nor a checkout, so the only
way a coding request can reach code is the same one Hermes itself uses: the
Hermes agent runtime, told server-side to use its Claude Code capability at
its own default model and medium effort.

These tests pin that contract: what Hermes is sent, what is deliberately not
sent, what counts as a finished job, and what never reaches a log.
"""

from __future__ import annotations

import asyncio
import inspect
import logging

import pytest

from caal.coding_delegation import (
    CLAUDE_CODE_EFFORT,
    CODING_DELEGATION_CONTRACT,
    COMPLETION_MARKER,
    INCOMPLETE_MARKER,
    MAX_CODING_ANSWER_CHARS,
    MAX_CODING_REQUEST_CHARS,
    HermesCodingDelegate,
    build_coding_delegate,
)

REQUEST = "fix the retry bug in the ollama provider"
DONE = "I fixed the retry and the tests pass. " + COMPLETION_MARKER


class Answer:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls: list[object] = []


class FakeHermes:
    """A stand-in for the Hermes agent runtime."""

    provider_name = "hermes"
    manages_own_tools = True

    def __init__(self, content: str = DONE, *, fail: Exception | None = None) -> None:
        self._content = content
        self._fail = fail
        self.calls: list[list[dict[str, object]]] = []
        self.kwargs: list[dict[str, object]] = []

    async def chat(self, messages, **kwargs):
        self.calls.append(messages)
        self.kwargs.append(kwargs)
        if self._fail is not None:
            raise self._fail
        return Answer(self._content)

    @property
    def last_system(self) -> str:
        return str(self.calls[-1][0]["content"])

    @property
    def last_user(self) -> str:
        return str(self.calls[-1][-1]["content"])


def delegate(provider, **kwargs) -> HermesCodingDelegate:
    return HermesCodingDelegate(provider, **kwargs)


# --- the delegation contract --------------------------------------------------


@pytest.mark.asyncio
async def test_a_coding_request_goes_to_the_hermes_runtime() -> None:
    hermes = FakeHermes()

    answer = await delegate(hermes)(REQUEST, "")

    assert len(hermes.calls) == 1
    assert hermes.last_user == REQUEST
    assert answer.startswith("I fixed the retry")


@pytest.mark.asyncio
async def test_hermes_is_told_to_use_claude_code_at_default_model_medium_effort() -> None:
    hermes = FakeHermes()

    await delegate(hermes)(REQUEST, "")

    system = hermes.last_system.lower()
    assert "claude code" in system
    assert CLAUDE_CODE_EFFORT in system
    assert "default model" in system
    # The policy is stated server-side, in the system frame, never by the user.
    assert CODING_DELEGATION_CONTRACT in hermes.last_system
    assert hermes.calls[-1][0]["role"] == "system"
    assert hermes.calls[-1][-1]["role"] == "user"


@pytest.mark.asyncio
async def test_no_model_override_is_ever_sent_to_hermes() -> None:
    hermes = FakeHermes()

    await delegate(hermes)(REQUEST, "")

    assert all("model" not in kwargs for kwargs in hermes.kwargs)
    assert all(kwargs.get("tools") is None for kwargs in hermes.kwargs)


@pytest.mark.asyncio
async def test_the_transcript_is_never_forwarded_to_a_coding_job() -> None:
    hermes = FakeHermes()
    transcript = "User: my api key is sk-live-secret\nAssistant: understood"

    await delegate(hermes)(REQUEST, transcript)

    sent = " ".join(str(message["content"]) for message in hermes.calls[-1])
    assert "sk-live-secret" not in sent
    assert "Recent conversation" not in sent


@pytest.mark.asyncio
async def test_the_request_is_redacted_and_bounded() -> None:
    hermes = FakeHermes()

    await delegate(hermes)("fix the bug, api_key=sk-live-abcdefghijklmno " + "x" * 5000, "")

    assert "sk-live-abcdefghijklmno" not in hermes.last_user
    assert len(hermes.last_user) <= MAX_CODING_REQUEST_CHARS


def test_no_shell_command_or_working_path_is_ever_accepted() -> None:
    """The delegate takes a request and nothing else: no cwd, no argv, no CLI."""
    parameters = inspect.signature(HermesCodingDelegate.__init__).parameters
    assert "workdir" not in parameters
    assert "workdirs" not in parameters
    assert "cli_path" not in parameters
    call_parameters = inspect.signature(HermesCodingDelegate.__call__).parameters
    assert list(call_parameters) == ["self", "request", "context"]


def test_nothing_here_shells_out_to_a_claude_cli() -> None:
    """The container has no Claude Code and no checkout: a subprocess is a dead end."""
    import caal.coding_delegation as module

    source = inspect.getsource(module)
    assert "create_subprocess" not in source
    assert "shutil" not in source
    assert "subprocess" not in source


# --- verified completion ------------------------------------------------------


@pytest.mark.asyncio
async def test_a_verified_job_reports_its_answer_without_the_marker() -> None:
    hermes = FakeHermes("Fixed the retry loop and reran the suite. " + COMPLETION_MARKER)

    answer = await delegate(hermes)(REQUEST, "")

    assert answer == "Fixed the retry loop and reran the suite."


@pytest.mark.asyncio
async def test_an_unverified_answer_is_never_reported_as_complete() -> None:
    hermes = FakeHermes("I had a look and I think that would probably work.")

    with pytest.raises(RuntimeError):
        await delegate(hermes)(REQUEST, "")


@pytest.mark.asyncio
async def test_an_explicitly_incomplete_job_fails_rather_than_claiming_success() -> None:
    hermes = FakeHermes("I could not open the repository. " + INCOMPLETE_MARKER)

    with pytest.raises(RuntimeError):
        await delegate(hermes)(REQUEST, "")


@pytest.mark.asyncio
async def test_a_job_marked_both_ways_is_not_a_finished_job() -> None:
    hermes = FakeHermes("Partly done. " + COMPLETION_MARKER + " " + INCOMPLETE_MARKER)

    with pytest.raises(RuntimeError):
        await delegate(hermes)(REQUEST, "")


@pytest.mark.asyncio
async def test_an_empty_answer_is_not_a_finished_job() -> None:
    hermes = FakeHermes("   " + COMPLETION_MARKER + "  ")

    with pytest.raises(RuntimeError):
        await delegate(hermes)(REQUEST, "")


@pytest.mark.asyncio
async def test_an_unreachable_hermes_fails_loudly_without_upstream_detail() -> None:
    hermes = FakeHermes(fail=RuntimeError("connect to 10.0.0.4:8642 refused"))

    with pytest.raises(RuntimeError) as raised:
        await delegate(hermes)(REQUEST, "")

    assert "10.0.0.4" not in str(raised.value)


@pytest.mark.asyncio
async def test_a_job_past_its_budget_is_not_a_finished_job() -> None:
    class Slow:
        provider_name = "hermes"

        async def chat(self, messages, **_):
            await asyncio.sleep(10)

    with pytest.raises(RuntimeError):
        await delegate(Slow(), timeout_seconds=0.01)(REQUEST, "")


@pytest.mark.asyncio
async def test_a_cancelled_job_propagates_rather_than_being_swallowed() -> None:
    class Hanging:
        provider_name = "hermes"

        async def chat(self, messages, **_):
            await asyncio.sleep(10)

    task = asyncio.create_task(delegate(Hanging())(REQUEST, ""))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_the_answer_is_redacted_and_bounded() -> None:
    hermes = FakeHermes("token=sk-live-abcdefghijklmno " + "y" * 9000 + " " + COMPLETION_MARKER)

    answer = await delegate(hermes)(REQUEST, "")

    assert "sk-live-abcdefghijklmno" not in answer
    assert len(answer) <= MAX_CODING_ANSWER_CHARS


# --- logs ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nothing_about_the_request_or_the_answer_reaches_the_log(caplog) -> None:
    hermes = FakeHermes("I renamed the parser in secretmodule.py. " + COMPLETION_MARKER)
    caplog.set_level(logging.DEBUG)

    await delegate(hermes)("refactor the billing parser in secretmodule.py", "")

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "billing" not in logged
    assert "secretmodule" not in logged
    assert CODING_DELEGATION_CONTRACT not in logged


@pytest.mark.asyncio
async def test_a_failure_never_logs_the_upstream_error_text(caplog) -> None:
    hermes = FakeHermes(fail=RuntimeError("bearer sk-hermes-token rejected by 10.0.0.4"))
    caplog.set_level(logging.DEBUG)

    with pytest.raises(RuntimeError):
        await delegate(hermes)(REQUEST, "")

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "sk-hermes-token" not in logged
    assert "10.0.0.4" not in logged


# --- building it from the runtime ---------------------------------------------


class Routed:
    provider_name = "routed"

    def __init__(self, escalation=None) -> None:
        self.escalation = escalation


def test_a_hermes_escalation_builds_the_delegate() -> None:
    built = build_coding_delegate(dict(), provider=Routed(escalation=FakeHermes()))

    assert isinstance(built, HermesCodingDelegate)


def test_without_hermes_there_is_no_coding_delegate() -> None:
    """No agent runtime means coding turns stay on the ordinary path."""
    assert build_coding_delegate(dict(), provider=Routed(escalation=None)) is None


def test_a_disabled_deployment_builds_nothing() -> None:
    runtime = dict(coding_delegation_enabled=False)
    assert build_coding_delegate(runtime, provider=Routed(escalation=FakeHermes())) is None


def test_a_plain_hermes_provider_is_used_directly() -> None:
    assert isinstance(build_coding_delegate(dict(), provider=FakeHermes()), HermesCodingDelegate)


def test_a_livekit_wrapped_provider_is_unwrapped() -> None:
    class CAALLLMLike:
        provider_instance = Routed(escalation=FakeHermes())

    assert isinstance(build_coding_delegate(dict(), provider=CAALLLMLike()), HermesCodingDelegate)


# --- the transport bound ------------------------------------------------------


class TimedHermes(FakeHermes):
    """A runtime that lets one request raise its own transport bound."""

    accepts_request_timeout = True


@pytest.mark.asyncio
async def test_the_job_budget_reaches_a_runtime_that_can_take_it() -> None:
    """A voice-sized transport bound would cut a real coding job off long before."""
    hermes = TimedHermes()

    await delegate(hermes, timeout_seconds=900)(REQUEST, "")

    assert hermes.kwargs[-1]["request_timeout"] == 900
    assert "model" not in hermes.kwargs[-1]


@pytest.mark.asyncio
async def test_a_runtime_without_that_capability_is_called_plainly() -> None:
    hermes = FakeHermes()

    await delegate(hermes, timeout_seconds=900)(REQUEST, "")

    assert "request_timeout" not in hermes.kwargs[-1]


def test_the_real_hermes_provider_accepts_a_longer_bound_for_one_request() -> None:
    from caal.llm.providers import HermesProvider

    provider = HermesProvider(api_key="test-key")

    assert provider.accepts_request_timeout is True
    assert "request_timeout" in inspect.signature(provider.chat).parameters
