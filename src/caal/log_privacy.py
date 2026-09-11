"""Keep what the user said, and what the model passed around, out of the log.

The LiveKit agents framework attaches the raw transcript of a turn -- and, in
other places, tool arguments, tool output and message text -- to some of its
own log records as ``lk.pii.*`` extra fields. Its production formatter
serialises every extra field it finds, so a single framework warning was
enough to print a user utterance into the container log:

    WARN livekit.agents skipping user input, current agent is unavailable
         {"lk.pii.user_input": "..."}

The framework offers no switch for this, so CAAL strips those fields itself,
process-wide and before any handler can format them. The message survives; the
private field does not. Two paths are covered, because a record can reach a
handler by either:

* :func:`install_pii_redaction` wraps ``logging.Logger.makeRecord``, which is
  where the standard library attaches ``extra=`` fields to a record. This
  catches every logger in the process, including ones created later, and
  including the handlers LiveKit installs on the root logger in its worker
  processes.
* :class:`PIIRedactionFilter` does the same job for a handler or logger that
  is configured directly, and for records built by hand.

CAAL's own code does not log utterances or argument values at all; this module
is about the code CAAL does not own.
"""

from __future__ import annotations

import logging
import threading

__all__ = ["PII_PREFIXES", "PIIRedactionFilter", "install_pii_redaction", "redact_pii_fields"]

# Every private field the framework emits is namespaced. Matching the prefix
# rather than a list of names means a new field in a future version is redacted
# on arrival rather than after someone notices it in a log.
PII_PREFIXES = ("lk.pii.",)

_lock = threading.Lock()
_installed = False


def redact_pii_fields(record: logging.LogRecord) -> logging.LogRecord:
    """Drop every private extra field from ``record``; safe metadata is untouched.

    Idempotent, and cheap on the overwhelming majority of records, which carry
    no such field at all.
    """
    private = [name for name in record.__dict__ if name.startswith(PII_PREFIXES)]
    for name in private:
        del record.__dict__[name]
    return record


class PIIRedactionFilter(logging.Filter):
    """A filter that redacts rather than drops: the message is still logged."""

    def filter(self, record: logging.LogRecord) -> bool:
        redact_pii_fields(record)
        return True


def install_pii_redaction() -> None:
    """Redact private extras for every logger in this process, once.

    Wrapping ``makeRecord`` is what makes this hold for loggers and handlers
    CAAL never sees -- the ones the LiveKit worker configures for itself.
    """
    global _installed
    with _lock:
        if _installed:
            return
        original = logging.Logger.makeRecord

        def make_record(self, *args, **kwargs):
            return redact_pii_fields(original(self, *args, **kwargs))

        make_record.__doc__ = original.__doc__
        logging.Logger.makeRecord = make_record  # type: ignore[method-assign]
        logging.getLogger().addFilter(PIIRedactionFilter())
        _installed = True
