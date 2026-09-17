"""Local, bounded text extraction from an uploaded company document.

Four formats, all parsed on this machine. Nothing here contacts a network,
resolves an external relationship, follows a link, downloads a model, or runs
anything the file asks it to run.

``.txt`` / ``.md``
    decoded in process: there is no structure here to be hostile with.
    Markdown headings become section locations so a citation can say where.

``.pdf``
    parsed by **pypdf**, an established local pure-Python PDF library, one
    page at a time through the document's real page tree. A citation that says
    "page 7" therefore means the seventh page of the file, not the seventh
    content stream that happened to contain text. The previous release read
    text-showing operators off content streams with a regular expression; it
    missed ordinary ``[(...)] TJ`` arrays, hex strings and CID encodings, and
    it numbered pages by stream, so it is gone.

``.docx``
    exactly one member of the zip -- ``word/document.xml`` -- read through
    :mod:`zipfile` with its declared size checked *before* it is inflated.
    ``vbaProject.bin`` is never read and never executed; relationships,
    external targets and embedded objects are never followed.

A file's extension and its magic bytes must agree, so a ``.docx`` renamed to
``.pdf`` is refused rather than misparsed. PDF and DOCX are parsed in a
**subprocess** under a wall clock, an address-space limit and a CPU limit,
because they are the two formats where a malicious file gets to drive a
parser; the bytes go over a pipe, so no temporary file is written.

**Nothing is dropped quietly.** A paragraph longer than the block bound is
split into further blocks that keep its location, rather than truncated. A
document with more blocks than the bound is refused with ``too_large`` rather
than silently shortened. Every result carries how much was read -- pages with
and without text, block and character counts, and whether the extraction is
:attr:`Extracted.complete` -- so a caller can say what it actually has.

Failure is always explicit. A password-protected PDF is ``password_required``;
a PDF whose pages hold no extractable text is ``needs_ocr``; a PDF where only
some pages hold text is ``ok`` with ``complete=False`` and a warning naming
the pages that were not read. None of them is stored as a successful, empty,
silently unsearchable document, and no release here promises OCR.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

__all__ = [
    "EXTRACT_TIMEOUT_SECONDS",
    "MAX_BLOCKS",
    "MAX_BLOCK_CHARS",
    "MAX_INFLATED_BYTES",
    "MAX_PDF_PAGES",
    "MAX_RATIO",
    "MAX_SOURCE_BYTES",
    "STATUSES",
    "Block",
    "Extracted",
    "extract",
    "apply_worker_limits",
    "extract_in_process",
    "extraction_capabilities",
    "pdf_parser_available",
    "supported_extensions",
]

MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_INFLATED_BYTES = 16 * 1024 * 1024
MAX_RATIO = 50
MAX_PDF_PAGES = 300
MAX_BLOCK_CHARS = 20_000
MAX_BLOCKS = 5_000
EXTRACT_TIMEOUT_SECONDS = 20
# What a parser subprocess may take of this machine before it is stopped.
WORKER_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
WORKER_CPU_SECONDS = 25

STATUSES = (
    "ok",
    "password_required",
    "needs_ocr",
    "unsupported_type",
    "too_large",
    "malformed",
    "timeout",
    "unavailable",
)

_EXTENSIONS = {".txt": "text", ".md": "text", ".pdf": "pdf", ".docx": "docx"}
_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_DOCUMENT_PART = "word/document.xml"


def supported_extensions() -> tuple[str, ...]:
    return tuple(sorted(_EXTENSIONS))


def pdf_parser_available() -> bool:
    """Whether the local PDF library is installed in *this* interpreter."""
    try:
        import pypdf  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure is "no PDF support"
        return False
    return True


def extraction_capabilities() -> dict[str, object]:
    """What this deployment can and cannot read. Surfaced, not assumed."""
    available = pdf_parser_available()
    return {
        "extensions": list(supported_extensions()),
        "pdf_parser": "pypdf" if available else None,
        "pdf_supported": available,
        "ocr": False,
        "note": (
            "Text PDFs are read with pypdf, page by page. Scanned pages and "
            "password-protected PDFs are not read: they are reported, never indexed as "
            "if they had worked. There is no OCR in this release."
            if available
            else "PDF support is not installed in this runtime; .pdf uploads are refused."
        ),
    }


@dataclass(frozen=True)
class Block:
    """One extracted passage and where in the document it came from."""

    text: str
    location: str


@dataclass(frozen=True)
class Extracted:
    """The outcome of reading one file, and how much of it was actually read."""

    status: str
    blocks: tuple[Block, ...] = ()
    reason: str | None = None
    page_count: int | None = None
    parser: str | None = None
    pages_with_text: int | None = None
    pages_without_text: int | None = None
    complete: bool = True
    warnings: tuple[str, ...] = field(default=())


def _failed(status: str, reason: str | None = None, **extra: object) -> Extracted:
    return Extracted(status=status, blocks=(), reason=reason or status, complete=False, **extra)


def _blocks_or_overflow(blocks: list[Block]) -> tuple[tuple[Block, ...], Extracted | None]:
    """Split long passages, keep their locations, and refuse an overflow out loud.

    The previous release cut every block at the character bound and threw away
    every block past the block bound, and still reported ``ok``. A clause past
    either bound simply was not in the library, and nothing said so.
    """
    split: list[Block] = []
    for block in blocks:
        text = block.text
        if not text.strip():
            continue
        if len(text) <= MAX_BLOCK_CHARS:
            split.append(block)
            continue
        start = 0
        part = 1
        while start < len(text):
            piece = text[start : start + MAX_BLOCK_CHARS]
            if start + MAX_BLOCK_CHARS < len(text):
                cut = piece.rfind(" ")
                if cut > MAX_BLOCK_CHARS // 2:
                    piece = piece[:cut]
            if piece.strip():
                split.append(Block(piece, f"{block.location} part {part}"))
                part += 1
            start += len(piece)
        if len(split) > MAX_BLOCKS:
            break
    if len(split) > MAX_BLOCKS:
        return (), _failed(
            "too_large",
            f"this document holds more than {MAX_BLOCKS} passages and was not indexed",
        )
    return tuple(split), None


def _kind(filename: str, data: bytes) -> str | None:
    """The format, only when the extension and the magic bytes agree."""
    _, _, extension = str(filename or "").lower().rpartition(".")
    kind = _EXTENSIONS.get(f".{extension}")
    if kind is None:
        return None
    if kind == "pdf":
        return kind if data[:5] == b"%PDF-" else None
    if kind == "docx":
        return kind if data[:4] == b"PK\x03\x04" else None
    # Text must actually be text: a NUL byte means somebody renamed a binary.
    if b"\x00" in data[:4096]:
        return None
    return kind


# --- text and markdown ------------------------------------------------------------------------


def _extract_text(data: bytes, *, markdown: bool) -> Extracted:
    try:
        body = data.decode("utf-8")
    except UnicodeDecodeError:
        return _failed("malformed", "not UTF-8 text", parser="text")
    blocks: list[Block] = []
    section = 0
    paragraph = 0
    for raw in re.split(r"\n\s*\n", body.replace("\r\n", "\n")):
        chunk = raw.strip()
        if not chunk:
            continue
        heading = markdown and chunk.lstrip().startswith("#")
        if heading:
            section += 1
            chunk = "\n".join(line.lstrip("# ").strip() for line in chunk.splitlines())
        if markdown and section:
            location = f"section {section}"
        else:
            paragraph += 1
            location = f"paragraph {paragraph}"
        blocks.append(Block(chunk, location))
    bounded, overflow = _blocks_or_overflow(blocks)
    if overflow is not None:
        return overflow
    if not bounded:
        return _failed("malformed", "the file holds no text", parser="text")
    return Extracted(
        status="ok",
        blocks=bounded,
        parser="text",
        complete=True,
    )


# --- docx -------------------------------------------------------------------------------------


def _extract_docx(data: bytes) -> Extracted:
    try:
        archive = zipfile.ZipFile(_BytesReader(data))
    except (zipfile.BadZipFile, OSError):
        return _failed("malformed", "not a readable .docx container", parser="docx")
    with archive:
        try:
            entry = archive.getinfo(_DOCUMENT_PART)
        except KeyError:
            return _failed("malformed", "the .docx has no document part", parser="docx")
        # Checked from the directory, before a single byte is inflated.
        if entry.file_size > MAX_INFLATED_BYTES:
            return _failed("too_large", "the document part is too large to read", parser="docx")
        if entry.compress_size and entry.file_size / max(entry.compress_size, 1) > MAX_RATIO:
            return _failed(
                "too_large", "the document part is compressed past the safe ratio", parser="docx"
            )
        try:
            with archive.open(entry) as handle:
                payload = handle.read(MAX_INFLATED_BYTES + 1)
        except (zipfile.BadZipFile, OSError, RuntimeError):
            return _failed("malformed", "the document part could not be read", parser="docx")
    if len(payload) > MAX_INFLATED_BYTES:
        return _failed("too_large", "the document part is too large to read", parser="docx")
    try:
        # No DTD is resolved and no external entity is fetched: ElementTree's
        # parser does not expand them, and nothing here enables it.
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError:
        return _failed("malformed", "the document part is not valid XML", parser="docx")
    blocks: list[Block] = []
    for index, paragraph in enumerate(root.iter(f"{_WORD_NS}p"), start=1):
        text = "".join(node.text or "" for node in paragraph.iter(f"{_WORD_NS}t")).strip()
        if text:
            blocks.append(Block(text, f"paragraph {index}"))
    if not blocks:
        return _failed("needs_ocr", "the document holds no extractable text", parser="docx")
    bounded, overflow = _blocks_or_overflow(blocks)
    if overflow is not None:
        return overflow
    return Extracted(status="ok", blocks=bounded, parser="docx", complete=True)


class _BytesReader:
    """A seekable in-memory file, so a .docx never needs a path on disk."""

    def __init__(self, data: bytes) -> None:
        import io

        self._buffer = io.BytesIO(data)

    def __getattr__(self, name: str):  # pragma: no cover - plain delegation
        return getattr(self._buffer, name)


# --- pdf --------------------------------------------------------------------------------------


def _extract_pdf(data: bytes) -> Extracted:
    """Read a text PDF with pypdf, page by page, and say what it could not read."""
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except Exception:  # noqa: BLE001 - an absent parser is an outage, not a bad file
        logger.error("The local PDF parser is not installed in this runtime")
        return _failed(
            "unavailable",
            "PDF support is not installed in this FRIDAY runtime",
            parser=None,
        )

    import io

    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
    except PdfReadError:
        return _failed("malformed", "this PDF could not be parsed", parser="pypdf")
    except Exception:  # noqa: BLE001 - a hostile file has many ways to fail
        return _failed("malformed", "this PDF could not be parsed", parser="pypdf")

    if getattr(reader, "is_encrypted", False):
        # An empty user password is still a document the owner meant to
        # protect; it is refused rather than quietly opened.
        return _failed(
            "password_required", "the PDF is password protected", parser="pypdf"
        )

    try:
        pages = list(reader.pages)
    except Exception:  # noqa: BLE001
        return _failed("malformed", "the PDF page tree could not be read", parser="pypdf")

    total = len(pages)
    if total == 0:
        return _failed("malformed", "the PDF has no pages", parser="pypdf")
    if total > MAX_PDF_PAGES:
        return _failed(
            "too_large",
            f"the PDF has more than {MAX_PDF_PAGES} pages",
            parser="pypdf",
            page_count=total,
        )

    blocks: list[Block] = []
    empty_pages: list[int] = []
    unreadable_pages: list[int] = []
    for number, page in enumerate(pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - one bad page must not lose the rest
            unreadable_pages.append(number)
            continue
        cleaned = "\n".join(line.rstrip() for line in text.splitlines())
        if not cleaned.strip():
            empty_pages.append(number)
            continue
        for paragraph in re.split(r"\n\s*\n", cleaned):
            body = paragraph.strip()
            if body:
                blocks.append(Block(body, f"page {number}"))

    if not blocks:
        return _failed(
            "needs_ocr",
            "the PDF holds no extractable text; it looks like a scan and there is no OCR here",
            parser="pypdf",
            page_count=total,
            pages_with_text=0,
            pages_without_text=total,
        )

    bounded, overflow = _blocks_or_overflow(blocks)
    if overflow is not None:
        return overflow

    read = total - len(empty_pages) - len(unreadable_pages)
    warnings: list[str] = []
    if empty_pages:
        warnings.append(
            "no text on "
            + _page_list(empty_pages)
            + "; those pages are images or scans and were not indexed"
        )
    if unreadable_pages:
        warnings.append("could not read " + _page_list(unreadable_pages))
    return Extracted(
        status="ok",
        blocks=bounded,
        parser="pypdf",
        page_count=total,
        pages_with_text=read,
        pages_without_text=len(empty_pages) + len(unreadable_pages),
        complete=not warnings,
        warnings=tuple(warnings),
    )


def _page_list(numbers: list[int], limit: int = 12) -> str:
    shown = ", ".join(str(number) for number in numbers[:limit])
    more = "" if len(numbers) <= limit else f" and {len(numbers) - limit} more"
    return f"page{'s' if len(numbers) != 1 else ''} {shown}{more}"


# --- the boundary ------------------------------------------------------------------------------


def extract_in_process(filename: str, data: bytes) -> Extracted:
    """Parse without a subprocess. The worker calls this; callers want :func:`extract`."""
    if len(data) > MAX_SOURCE_BYTES:
        return _failed("too_large", "the file is larger than this library accepts")
    kind = _kind(filename, data)
    if kind is None:
        return _failed("unsupported_type", "this file type is not supported")
    if kind == "text":
        return _extract_text(data, markdown=str(filename).lower().endswith(".md"))
    if kind == "docx":
        return _extract_docx(data)
    return _extract_pdf(data)


def extract(filename: str, data: bytes, *, timeout: float | None = None) -> Extracted:
    """Read ``data`` as ``filename``. PDF and DOCX are parsed under a wall clock.

    The bytes reach the subprocess over a pipe: nothing is written to a path,
    and a parser that hangs on a hostile file is killed rather than waited on.
    """
    if len(data) > MAX_SOURCE_BYTES:
        return _failed("too_large", "the file is larger than this library accepts")
    kind = _kind(filename, data)
    if kind is None:
        return _failed("unsupported_type", "this file type is not supported")
    if kind == "text":
        return _extract_text(data, markdown=str(filename).lower().endswith(".md"))
    return _run_worker(filename, data, timeout=timeout or EXTRACT_TIMEOUT_SECONDS)


def apply_worker_limits() -> None:
    """Cap what a parser may take of this machine.

    Called by the worker itself, at the top of its own ``main`` rather than
    through ``preexec_fn``: the backend that starts it is multithreaded, and a
    hook that runs between fork and exec there is a deadlock waiting to
    happen. The limits are in place before a single byte is parsed either way.
    """
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        ceiling = WORKER_ADDRESS_SPACE_BYTES
        if hard != resource.RLIM_INFINITY:
            ceiling = min(ceiling, hard)
        resource.setrlimit(resource.RLIMIT_AS, (ceiling, hard))
    except Exception:  # noqa: BLE001 - a platform without the limit still has the clock
        logger.debug("The extraction worker could not set an address-space limit")
    try:
        import resource

        _, hard = resource.getrlimit(resource.RLIMIT_CPU)
        ceiling = WORKER_CPU_SECONDS
        if hard != resource.RLIM_INFINITY:
            ceiling = min(ceiling, hard)
        resource.setrlimit(resource.RLIMIT_CPU, (ceiling, hard))
    except Exception:  # noqa: BLE001
        logger.debug("The extraction worker could not set a CPU limit")


def _run_worker(filename: str, data: bytes, *, timeout: float) -> Extracted:
    import json

    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
    # A parser subprocess has no business reading the library, so it is not
    # told where the library is or what opens it.
    for name in (
        "CAAL_COMPANY_LIBRARY_KEYS",
        "CAAL_COMPANY_LIBRARY_DIR",
        "CAAL_COMPANY_MCP_TOKEN",
        "CAAL_INTERNAL_AUTH_SECRET",
        "CAAL_PROFILE_ENCRYPTION_KEYS",
    ):
        environment.pop(name, None)
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell, bytes over a pipe
            [sys.executable, "-m", "caal.company.extract_worker", os.path.basename(filename)],
            input=data,
            capture_output=True,
            timeout=timeout,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _failed("timeout", "reading this file took too long and was stopped")
    except OSError:
        logger.error("The company extraction worker could not be started")
        return _failed("unavailable", "the document reader could not be started")
    if completed.returncode != 0:
        return _failed("malformed", "this file could not be read")
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return _failed("malformed", "this file could not be read")
    status = payload.get("status")
    if status not in STATUSES:
        return _failed("malformed", "this file could not be read")
    blocks = tuple(
        Block(str(item.get("text", "")), str(item.get("location", "")))
        for item in payload.get("blocks", [])
        if isinstance(item, dict)
    )

    def _count(name: str) -> int | None:
        value = payload.get(name)
        return int(value) if isinstance(value, int) else None

    warnings = tuple(
        str(item) for item in payload.get("warnings", []) if isinstance(item, str)
    )
    return Extracted(
        status=status,
        blocks=blocks,
        reason=payload.get("reason"),
        page_count=_count("page_count"),
        parser=payload.get("parser") if isinstance(payload.get("parser"), str) else None,
        pages_with_text=_count("pages_with_text"),
        pages_without_text=_count("pages_without_text"),
        complete=bool(payload.get("complete", False)),
        warnings=warnings,
    )
