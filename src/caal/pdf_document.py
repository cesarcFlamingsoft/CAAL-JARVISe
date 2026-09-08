"""Build a short text PDF with the standard library, and hold it safely on disk.

A background task that promises the user a PDF has to produce a file that a
PDF reader will actually open. Pulling in a rendering library for a page of
prose would add an unbounded dependency to an offline-first deployment, so
this module writes the small subset of PDF 1.4 that a text document needs:
one catalogue, one page tree, one Helvetica font, and one content stream per
page. The output is a complete, self-contained file with a real cross
reference table.

Everything is bounded before it reaches the page: the title, the body, the
number of lines on a page, and the number of pages. A model that answers with
a novel yields a truncated document, never an unbounded write.

Artifacts live in a private directory (``0o700``) as files nobody else can
read (``0o600``), under opaque random names: a filename is visible in far more
places than a file's contents, so it carries nothing about the request, the
task, or the user. They are meant to be short lived -- written, delivered, and
discarded -- and :func:`purge_artifacts` reclaims whatever a crash left behind.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_BODY_CHARS",
    "MAX_PAGES",
    "MAX_TITLE_CHARS",
    "artifact_ttl_reached",
    "discard_artifact",
    "purge_artifacts",
    "render_pdf",
    "store_artifact",
]

# US Letter, in PDF points.
_PAGE_WIDTH = 612
_PAGE_HEIGHT = 792
_MARGIN = 72
_TITLE_SIZE = 16
_BODY_SIZE = 11
_LEADING = 15

# Helvetica averages a little under half an em; 88 characters of body text fit
# the printable width with room to spare, and a long unbroken word is hard
# wrapped rather than allowed to run off the page.
_MAX_LINE_CHARS = 88
_LINES_PER_PAGE = (_PAGE_HEIGHT - 2 * _MARGIN) // _LEADING

MAX_TITLE_CHARS = 120
MAX_BODY_CHARS = 20_000
MAX_PAGES = 20

_TRUNCATION_MARK = "..."
_DEFAULT_TITLE = "Document"

# Artifacts older than this belong to a process that never came back.
ARTIFACT_TTL_SECONDS = 3_600


def _latin1(text: str) -> str:
    """PDF's WinAnsi encoding is Latin-1; anything else becomes a plain question mark."""
    return text.encode("latin-1", "replace").decode("latin-1")


def _escape(text: str) -> str:
    """Escape the three characters that end or nest a PDF string literal."""
    for character, replacement in (("\\", r"\\"), ("(", r"\("), (")", r"\)")):
        text = text.replace(character, replacement)
    return text


def _bound(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - len(_TRUNCATION_MARK)] + _TRUNCATION_MARK


def _wrap(paragraph: str) -> list[str]:
    """Greedy word wrap, hard breaking any single word wider than the page."""
    lines: list[str] = []
    current = ""
    for word in paragraph.split():
        while len(word) > _MAX_LINE_CHARS:
            if current:
                lines.append(current)
                current = ""
            lines.append(word[:_MAX_LINE_CHARS])
            word = word[_MAX_LINE_CHARS:]
        candidate = f"{current} {word}".strip()
        if len(candidate) > _MAX_LINE_CHARS:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def _layout(body: str) -> list[list[str]]:
    """Wrapped body lines split into pages, blank lines kept between paragraphs."""
    lines: list[str] = []
    for paragraph in _bound(body.strip(), MAX_BODY_CHARS).splitlines():
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(_wrap(paragraph.strip()))
    pages: list[list[str]] = []
    # The first page gives two of its lines to the title and the gap under it.
    budget = _LINES_PER_PAGE - 2
    while lines and len(pages) < MAX_PAGES:
        pages.append(lines[:budget])
        lines = lines[budget:]
        budget = _LINES_PER_PAGE
    return pages or [[]]


def _content_stream(lines: list[str], *, title: str | None) -> bytes:
    """One page's text drawing operators."""
    parts: list[str] = ["BT", f"1 0 0 1 {_MARGIN} {_PAGE_HEIGHT - _MARGIN} Tm", f"{_LEADING} TL"]
    if title is not None:
        parts += [f"/F1 {_TITLE_SIZE} Tf", f"({_escape(title)}) Tj", "T*", "T*"]
    parts.append(f"/F1 {_BODY_SIZE} Tf")
    for line in lines:
        parts += [f"({_escape(line)}) Tj", "T*"]
    parts.append("ET")
    return "\n".join(parts).encode("latin-1", "replace")


def render_pdf(*, title: str, body: str) -> bytes:
    """Return a complete PDF 1.4 file holding ``title`` and ``body`` as plain text.

    Bounded on every axis: the title and body are truncated, and the document
    never exceeds ``MAX_PAGES`` pages, so an unbounded answer cannot become an
    unbounded file.
    """
    heading = _latin1(" ".join(str(title).split())) or _DEFAULT_TITLE
    heading = _bound(heading, MAX_TITLE_CHARS)
    pages = _layout(_latin1(str(body)))

    # Object 1 catalogue, 2 page tree, 3 font, then a page and a stream each.
    page_ids = [4 + 2 * index for index in range(len(pages))]
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            "<< /Type /Pages /Kids ["
            + " ".join(f"{page_id} 0 R" for page_id in page_ids)
            + f"] /Count {len(pages)} >>"
        ).encode("latin-1"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    for index, (page_id, lines) in enumerate(zip(page_ids, pages)):
        stream = _content_stream(lines, title=heading if index == 0 else None)
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_PAGE_WIDTH} {_PAGE_HEIGHT}] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>"
            ).encode("latin-1")
        )
        objects.append(
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream
            + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for number, payload in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("ascii") + payload + b"\nendobj\n"
    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("ascii")
    out += f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    return bytes(out)


def store_artifact(content: bytes, *, directory: Path, suffix: str = ".pdf") -> Path:
    """Write ``content`` under an opaque name only this process's user can read.

    The directory is created private and the file is opened ``O_EXCL`` with
    ``0o600``, so the artifact is never briefly world readable and never
    silently overwrites an existing one. The name carries no request text, no
    task id, and no user id.
    """
    if not isinstance(content, (bytes, bytearray)) or not content:
        raise ValueError("an artifact needs content")
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        target_dir.chmod(0o700)
    except OSError:
        logger.warning("Could not tighten permissions on the artifact directory")
    path = target_dir / f"{secrets.token_hex(16)}{suffix}"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
    except BaseException:
        discard_artifact(path)
        raise
    return path


def discard_artifact(path: Path | None) -> bool:
    """Remove one artifact, best effort. Returns whether it is gone."""
    if path is None:
        return False
    try:
        Path(path).unlink()
    except FileNotFoundError:
        return True
    except OSError:
        logger.warning("Could not remove a document artifact")
        return False
    return True


def artifact_ttl_reached(path: Path, *, max_age_seconds: float, now: float | None = None) -> bool:
    try:
        modified = Path(path).stat().st_mtime
    except OSError:
        return False
    return (time.time() if now is None else now) - modified > max_age_seconds


def purge_artifacts(
    directory: Path, *, max_age_seconds: float = ARTIFACT_TTL_SECONDS, now: float | None = None
) -> int:
    """Reclaim artifacts a crashed or killed task left behind. Counts only."""
    target_dir = Path(directory)
    if not target_dir.is_dir():
        return 0
    removed = 0
    try:
        entries = list(target_dir.iterdir())
    except OSError:
        logger.warning("Could not list the artifact directory")
        return 0
    for entry in entries:
        if entry.is_file() and artifact_ttl_reached(
            entry, max_age_seconds=max_age_seconds, now=now
        ):
            removed += discard_artifact(entry)
    if removed:
        logger.info("Reclaimed %d stale document artifact(s)", removed)
    return removed
