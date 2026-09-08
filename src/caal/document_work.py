"""Background work that has to hand back a file, not a paragraph.

"Create a short PDF about yourself and send it to me on Telegram" is not a
question with a spoken answer. Running it through the ordinary background
worker produces prose, stores the prose as the task's result, and reports
success -- so the user is told a document was sent when no file was ever
created. This module closes that gap for the one shape of request the
assistant can honestly satisfy today: a PDF, delivered on Telegram.

The reading is deterministic and offline, in the same spirit as the other
local classifiers: a request only becomes document work when it names a PDF
*and* asks for it to be delivered somewhere. Anything else -- including a PDF
nobody asked to have sent -- falls through to the ordinary worker untouched,
so existing background behaviour is unchanged.

Once a request is document work it either produces a real artifact and
delivers it, or it fails. Every failure path raises
:class:`DocumentWorkError` with a short sentence written here, never an
upstream exception's text, so the task settles quickly with an outcome that
is safe to speak and carries no host, token, or transport detail. Nothing in
this module logs the request, the document, the caption, or the destination.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from .pdf_document import (
    ARTIFACT_TTL_SECONDS,
    MAX_TITLE_CHARS,
    discard_artifact,
    purge_artifacts,
    render_pdf,
    store_artifact,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DELIVERY_TIMEOUT_SECONDS",
    "DELIVERED_FILENAME",
    "DOCUMENT_INSTRUCTIONS",
    "SUPPORTED_CHANNELS",
    "DeliverDocument",
    "DocumentRequest",
    "DocumentWorkError",
    "DocumentWorker",
    "detect_document_request",
]


class DocumentWorkError(RuntimeError):
    """A document task that could not be completed, with a sentence for the user."""


# Deliver one finished document. Bounded and keyword-only so a caller cannot
# accidentally pass the bytes and the name the wrong way round.
DeliverDocument = Callable[..., Awaitable[None]]
# The ordinary background worker: (request, context) -> spoken-style answer.
Compose = Callable[[str, str], Awaitable[str]]

PDF_KIND = "pdf"
TELEGRAM_CHANNEL = "telegram"
SUPPORTED_CHANNELS = frozenset({TELEGRAM_CHANNEL})

# A document is delivered under a fixed, neutral name. A filename travels
# through notification previews, download folders, and chat history; deriving
# it from the request or the task id would leak both into all of them.
DELIVERED_FILENAME = "jarvis-document.pdf"

DEFAULT_DELIVERY_TIMEOUT_SECONDS = 60.0
MAX_CLASSIFIER_CHARS = 2_000
MAX_CAPTION_CHARS = 180

DOCUMENT_INSTRUCTIONS = (
    "Write the contents of a short document for the user. Begin with a single "
    "line of the exact form 'Title: <a short title>' and then write the body as "
    "plain paragraphs of prose. Use no markdown, no headings, no bullet lists, "
    "no links, and no code. Keep it under 400 words. Write only the document "
    "itself; do not describe it, introduce it, or comment on it."
)

_DEFAULT_TITLE = "Document"
_TITLE_LINE = re.compile(r"^\s*title\s*[:\-]\s*(?P<title>.+?)\s*$", re.IGNORECASE)

# What is being asked for.
_PDF_CUE = re.compile(r"\bpdfs?\b")

# Where it goes has to be read as a *destination*, not as a topic: "write a
# PDF explaining how people send files on Telegram" names Telegram and asks
# for nothing to be delivered there. So a channel only counts inside a whole
# delivery clause -- a hand-it-over verb, optionally the thing being handed
# over and who it is for, then the channel. Everything between the verb and
# the channel must be one of these closed sets, so an arbitrary noun ("files
# on Telegram") breaks the clause instead of completing it.
_VERB = (
    r"(?:send|sends|sending|share|shares|sharing|deliver|delivers|delivering|"
    r"forward|forwards|drop|post|shoot|put)"
)
_OBJECT = (
    r"(?:it|this|that|them|these|those|me|us|a copy|the copy|"
    r"the pdf|the document|the doc|the file|the report|the summary)"
)
_RECIPIENT = r"(?:to\s+(?:me|us|my\s+[a-z]+)\s+|over\s+|out\s+|across\s+|along\s+)"
_PREPOSITION = r"(?:on|via|through|over|to|by|in|using)"


def _delivery_clause(channel: str) -> re.Pattern[str]:
    return re.compile(
        rf"\b{_VERB}\s+(?:{_OBJECT}\s+)*{_RECIPIENT}*"
        rf"{_PREPOSITION}\s+(?:my\s+|the\s+|a\s+)?{channel}\b"
    )


# A verb that names its own channel needs no preposition ("email it to me").
_CHANNEL_CUES: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    (TELEGRAM_CHANNEL, (_delivery_clause("telegram"),)),
    ("whatsapp", (_delivery_clause(r"whats\s?app"),)),
    ("signal", (_delivery_clause("signal"),)),
    ("slack", (_delivery_clause("slack"),)),
    (
        "email",
        (
            _delivery_clause(r"e-?mail"),
            re.compile(rf"\be-?mails?\s+(?:{_OBJECT}\s+)*{_RECIPIENT}*(?:to\s+me|to\s+us)?\b"),
        ),
    ),
    (
        "sms",
        (
            _delivery_clause(r"(?:sms|text\s+message)"),
            re.compile(rf"\btexts?\s+(?:{_OBJECT}\s+)*{_RECIPIENT}*(?:to\s+me|to\s+us)?\b"),
        ),
    ),
)


@dataclass(frozen=True)
class DocumentRequest:
    """A request read as "make this document and send it there".

    Holds no request text: only the kind of artifact and the named channel, so
    it is safe to keep, compare, and log.
    """

    kind: str
    channel: str

    @property
    def deliverable(self) -> bool:
        """Whether this deployment can actually deliver to the named channel."""
        return self.channel in SUPPORTED_CHANNELS


def _normalized(text: object) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text[:MAX_CLASSIFIER_CHARS].split()).lower().replace("’", "'")


def detect_document_request(text: object) -> DocumentRequest | None:
    """Read a turn as document work, or ``None`` for ordinary background work.

    Deterministic, bounded, and deliberately narrow: a PDF that nobody asked
    to have delivered stays ordinary work, so this can only ever add the
    document path to requests that were previously answered with a false
    promise.
    """
    normalized = _normalized(text)
    if not normalized or not _PDF_CUE.search(normalized):
        return None
    for channel, cues in _CHANNEL_CUES:
        if any(cue.search(normalized) for cue in cues):
            return DocumentRequest(kind=PDF_KIND, channel=channel)
    return None


def split_title(composed: str) -> tuple[str, str]:
    """Separate the model's ``Title:`` line from the body it wrote.

    A model that ignored the instruction still yields a document: the body is
    used as written and the title falls back to a neutral one.
    """
    lines = str(composed).strip().splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        match = _TITLE_LINE.match(line)
        if match is None:
            break
        title = " ".join(match.group("title").split())[:MAX_TITLE_CHARS]
        return (title or _DEFAULT_TITLE), "\n".join(lines[index + 1 :]).strip()
    return _DEFAULT_TITLE, "\n".join(lines).strip()


class DocumentWorker:
    """Wrap the ordinary background worker so document requests produce a file.

    ``compose`` is the existing bounded worker: it is what talks to the
    configured LLM, and it stays the only thing that does. ``deliver`` hands a
    finished document to its channel; without one, a delivery request fails
    immediately and says so rather than pretending.
    """

    def __init__(
        self,
        *,
        compose: Compose,
        deliver: DeliverDocument | None = None,
        artifact_dir: Path | str | None = None,
        fallback: Compose | None = None,
        delivery_timeout_seconds: float = DEFAULT_DELIVERY_TIMEOUT_SECONDS,
        artifact_ttl_seconds: float = ARTIFACT_TTL_SECONDS,
    ) -> None:
        self._compose = compose
        self._deliver = deliver
        self._artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self._fallback = fallback or compose
        self._delivery_timeout = max(1.0, float(delivery_timeout_seconds))
        self._artifact_ttl = float(artifact_ttl_seconds)

    async def __call__(self, request: str, context: str) -> str:
        document = detect_document_request(request)
        if document is None:
            return await self._fallback(request, context)
        return await self._produce(document, request, context)

    def is_deliverable_request(self, request: object) -> bool:
        """Whether this worker can create and deliver the requested artifact now."""
        document = detect_document_request(request)
        return bool(document and document.deliverable and self._deliver is not None and self._artifact_dir)

    async def _produce(self, document: DocumentRequest, request: str, context: str) -> str:
        if not document.deliverable:
            raise DocumentWorkError(
                "I can only send documents on Telegram at the moment, so I could not "
                "deliver that one."
            )
        if self._deliver is None or self._artifact_dir is None:
            raise DocumentWorkError(
                "Telegram document delivery is not set up for this session, so I could "
                "not send the document."
            )
        # Reclaim anything an earlier task was killed before it could clean up.
        purge_artifacts(self._artifact_dir, max_age_seconds=self._artifact_ttl)

        composed = await self._compose(f"{DOCUMENT_INSTRUCTIONS}\n\n{request}", context)
        if not isinstance(composed, str) or not composed.strip():
            raise DocumentWorkError("I could not write the contents of the document.")
        title, body = split_title(composed)
        if not body:
            title, body = _DEFAULT_TITLE, composed.strip()

        try:
            content = render_pdf(title=title, body=body)
        except Exception as exc:
            raise DocumentWorkError("I could not turn the document into a PDF.") from exc

        path = None
        try:
            path = store_artifact(content, directory=self._artifact_dir)
        except Exception as exc:
            raise DocumentWorkError("I could not save the PDF I had written.") from exc
        try:
            await asyncio.wait_for(
                self._deliver(
                    filename=DELIVERED_FILENAME,
                    content=content,
                    caption=_caption(title),
                ),
                self._delivery_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise DocumentWorkError(
                "The PDF was ready but Telegram did not accept it in time."
            ) from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise DocumentWorkError(
                "The PDF was ready but Telegram would not accept it."
            ) from exc
        finally:
            # The artifact existed so the document was real and recoverable
            # while it was in flight; once it is delivered it is the user's copy.
            discard_artifact(path)
        logger.info("Delivered a document of %d bytes on %s", len(content), document.channel)
        if title and title != _DEFAULT_TITLE:
            return f"I've put together a PDF titled {title} and sent it to you on Telegram."
        return "I've put together the PDF and sent it to you on Telegram."


def _caption(title: str) -> str:
    label = " ".join(str(title).split())[:MAX_CAPTION_CHARS]
    return label or _DEFAULT_TITLE
