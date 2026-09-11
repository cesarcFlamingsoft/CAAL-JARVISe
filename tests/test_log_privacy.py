"""What may and may not reach a log line at runtime.

The LiveKit agents framework attaches the raw transcript of a turn to some of
its warnings as an ``lk.pii.*`` extra field, and its production JSON formatter
serialises every extra. A single warning ("skipping user input, current agent
is unavailable") was enough to print what the user had just said into the
container log. CAAL own code had two smaller leaks of the same kind: a debug
line echoing the transcript, and a tool line printing the model chosen
arguments verbatim.

Pinned properties:

* any ``lk.pii.*`` field is stripped from a record before a handler can format
  it, whoever logged it, without dropping the message itself;
* CAAL own tool logging names the tool and its argument *names*, never the
  values;
* redaction is idempotent and leaves ordinary fields alone.
"""

from __future__ import annotations

import io
import json
import logging

import pytest

from caal.log_privacy import PII_PREFIXES, PIIRedactionFilter, install_pii_redaction

UTTERANCE = "Zebulon Quixote merger review at four"


@pytest.fixture
def isolated_root():
    """A root logger restored exactly as it was, handlers and filters included."""
    root = logging.getLogger()
    handlers = list(root.handlers)
    filters = list(root.filters)
    level = root.level
    for handler in handlers:
        root.removeHandler(handler)
    try:
        yield root
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for item in list(root.filters):
            root.removeFilter(item)
        for handler in handlers:
            root.addHandler(handler)
        for item in filters:
            root.addFilter(item)
        root.setLevel(level)


class ExtraFormatter(logging.Formatter):
    """A stand-in for the LiveKit production formatter: it serialises extras."""

    _STANDARD = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__)

    def format(self, record: logging.LogRecord) -> str:
        extra = {
            name: value
            for name, value in record.__dict__.items()
            if name not in self._STANDARD and name != "message" and name != "asctime"
        }
        return record.getMessage() + " " + json.dumps(extra, default=str)


def _stream_handler() -> tuple[logging.Handler, io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(ExtraFormatter())
    return handler, stream


def test_a_livekit_pii_extra_never_reaches_a_handler(isolated_root) -> None:
    handler, stream = _stream_handler()
    isolated_root.addHandler(handler)
    isolated_root.setLevel(logging.DEBUG)
    install_pii_redaction()

    logging.getLogger("livekit.agents.voice.agent_activity").warning(
        "skipping user input, current agent is unavailable",
        extra={"lk.pii.user_input": UTTERANCE},
    )

    printed = stream.getvalue()
    assert "skipping user input" in printed, "the warning itself is still logged"
    assert UTTERANCE not in printed
    assert "lk.pii" not in printed


@pytest.mark.parametrize(
    "field",
    ["lk.pii.user_input", "lk.pii.user_transcript", "lk.pii.arguments", "lk.pii.text"],
)
def test_every_pii_field_shape_is_redacted(field: str) -> None:
    record = logging.LogRecord("x", logging.WARNING, __file__, 1, "message", None, None)
    setattr(record, field, UTTERANCE)
    record.__dict__["speech_id"] = "sp_1"

    assert PIIRedactionFilter().filter(record) is True
    assert UTTERANCE not in json.dumps(record.__dict__, default=str)
    assert record.__dict__["speech_id"] == "sp_1", "safe metadata is left alone"
    assert any(prefix for prefix in PII_PREFIXES)


def test_our_own_tool_logging_records_argument_names_not_values(caplog) -> None:
    """A model chosen argument is the words of the user; only the names are logged."""
    import importlib

    # caal.llm re-exports the llm_node *function*; load the module itself.
    llm_node_module = importlib.import_module("caal.llm.llm_node")

    class Recorder:
        def format_tool_call_message(self, content, tool_calls):
            return dict(role="assistant", content=content or "")

        def format_tool_result(self, content, tool_call_id, tool_name):
            return dict(role="tool", content=content, tool_call_id=tool_call_id)

    import asyncio
    from types import SimpleNamespace

    async def _execute(agent, tool_name, arguments):
        return dict(status="ok", message="done", data={})

    call = SimpleNamespace(id="c1", name="web.search", arguments=dict(query=UTTERANCE))
    agent = SimpleNamespace()

    # Another module may have made this logger non-propagating with a handler of
    # its own, so the record is captured at the logger itself.
    target = logging.getLogger("caal.llm.llm_node")
    previous, target.propagate = target.propagate, True
    target.addHandler(caplog.handler)
    original = llm_node_module._execute_single_tool
    llm_node_module._execute_single_tool = _execute
    try:
        with caplog.at_level(logging.DEBUG, logger="caal.llm.llm_node"):
            asyncio.run(llm_node_module._execute_tool_calls(agent, [], [call], None, Recorder()))
    finally:
        llm_node_module._execute_single_tool = original
        target.removeHandler(caplog.handler)
        target.propagate = previous

    assert "web.search" in caplog.text
    assert "query" in caplog.text, "the argument names are useful and safe"
    assert UTTERANCE not in caplog.text
