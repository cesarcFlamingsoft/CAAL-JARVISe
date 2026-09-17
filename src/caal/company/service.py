"""The company library: what may go in, what comes out, and who may ask.

One library, one owner, and the owner is **provisioned, never claimed**. It is
the ``usr_...`` id in ``CAAL_COMPANY_OWNER_USER_ID``, chosen by the operator
before anything is uploaded and bound into the library the first time the
writer opens it. After that it is immutable: a library bound to one id refuses
to open against a configuration naming another, and a library with no
configured owner is ``unconfigured`` -- it holds nothing, answers nothing, and
says which of those two it is. Nobody becomes the owner by being first to
write, and no user is ever selected by name.

Every read and every write names an owner and is refused when it is not *the*
owner. Before any of that happens the store is refreshed, so a second process
decides authorization and answers questions from what the writer has actually
published, never from a catalogue it loaded minutes ago.

What comes out is always the same shape: a bounded, cited passage inside an
envelope that says ``treat_as: quoted_document_text``. Nothing in this module
returns a judgement, a ranking, a score about a person, or a summary it wrote
itself. It returns what a document says and where the document says it, or it
returns an explicit non-answer:

``no_match``              nothing in the library matches
``no_subject_match``      no employee subject goes by that name
``ambiguous_subject``     more than one does, and no content is returned
``conflicting_versions``  two versions of one document are both current
``no_current_version``    the document has only history or drafts
``extraction_failed``     the version is on file but was never readable
``forbidden``             the caller does not own this library
``unconfigured``          no owner has been provisioned for this library

Metadata is never guessed. A classification, a status and an effective date
come from the person who uploaded the file or they are absent. Uploading a
new version does not demote the old one; only an explicit status change does.

**Version policy, one rule in both directions.** A search and a read both mean
*the current version* unless the caller asks for something else in so many
words. Drafts are excluded from search unless ``include_drafts`` is passed;
history is reachable only by naming a ``version_id``; and whenever a passage
does not come from a current version, its status travels with it so the answer
can say so out loud.

Retrieval is lexical BM25 over SQLite FTS5. It is keyword matching, it is
called that everywhere it is surfaced, and it is not semantic search.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .config import ROLE_OWNER, CompanyConfig
from .extraction import Block, Extracted, extract
from .store import CompanyStore, SourceRemovalError

logger = logging.getLogger(__name__)

__all__ = [
    "CLASSIFICATIONS",
    "STATUSES",
    "CompanyLibrary",
    "CompanyOwnerMismatchError",
    "IngestResult",
    "Subject",
]

CLASSIFICATIONS: tuple[str, ...] = ("policy", "hr", "contract", "general")
STATUSES: tuple[str, ...] = ("current", "superseded", "draft")
# A superseded version is history: it is kept, and it is never a default
# answer. A draft is not in force yet, so it is not one either.
_DEFAULT_STATUSES = ("current",)
_WITH_DRAFTS = ("current", "draft")

_EFFECTIVE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'._-]*")
_MAX_TITLE = 200
_MAX_LABEL = 60
_MAX_ALIASES = 8
_MAX_NAME = 120


class CompanyOwnerMismatchError(RuntimeError):
    """This library is bound to a different owner than the configuration names."""


@dataclass(frozen=True)
class IngestResult:
    """What happened to one uploaded file. ``reason`` is set unless indexed."""

    status: str
    document_id: str | None = None
    version_id: str | None = None
    chunk_count: int = 0
    reason: str | None = None
    page_count: int | None = None
    coverage: dict[str, Any] | None = None


@dataclass(frozen=True)
class Subject:
    """An employee subject: an identifier the owner assigns, never a name."""

    subject_id: str
    display_name: str
    aliases: tuple[str, ...] = field(default=())


def _identifier(prefix: str, *, width: int = 24) -> str:
    return f"{prefix}_{secrets.token_hex(width // 2)}"


def _clean(value: object, *, limit: int, name: str, required: bool = True) -> str | None:
    if value is None or value == "":
        if required:
            raise ValueError(f"{name} is required")
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    text = value.strip()
    if not text and required:
        raise ValueError(f"{name} is required")
    if len(text) > limit:
        raise ValueError(f"{name} is too long")
    return text or None


class CompanyLibrary:
    """The whole library. One writer per deployment; safe across threads."""

    # Bounds. They are attributes so a deployment can tighten them, and so a
    # test can prove the refusal rather than fill the disk to see it.
    MAX_DOCUMENTS = 200
    MAX_CHUNKS = 20_000
    MAX_CHUNK_CHARS = 1_200
    MAX_QUERY_CHARS = 300
    MAX_RESULTS = 10
    MAX_SNIPPET_CHARS = 600
    MAX_EXCERPT_CHARS = 4_000
    MAX_PAGE_SIZE = 25
    MAX_PEOPLE_PAGE = 50
    MAX_VERSIONS_PER_DOCUMENT = 20
    MAX_EXCERPT_LOCATIONS = 40

    def __init__(self, config: CompanyConfig, *, role: str | None = None) -> None:
        self._config = config
        self._role = role or config.role
        self._store = CompanyStore(config, role=self._role)
        self._lock = threading.RLock()
        try:
            self._bind_owner()
        except BaseException:
            self._store.close()
            raise

    # --- lifecycle ------------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._store.close()

    def flush(self) -> None:
        """Republish the index. Every mutation already has, so this is a belt.

        Kept because "is what I just did actually on disk?" is a fair question
        for a test to ask directly. A reader does nothing: readers never write.
        """
        if not self.writable:
            return
        with self._lock:
            self._store.publish()

    @property
    def role(self) -> str:
        return self._role

    @property
    def writable(self) -> bool:
        return self._role == ROLE_OWNER

    @property
    def generation(self) -> int:
        return self._store.generation

    @property
    def company_name(self) -> str | None:
        return self._config.company_name

    @property
    def configured(self) -> bool:
        """Whether an owner has been provisioned for this library."""
        return self._config.owner_user_id is not None

    # --- ownership ------------------------------------------------------------------------

    def owner_id(self) -> str | None:
        """The bound owner, refreshed from disk first. ``None`` when unbound."""
        self._refresh()
        return self._stored_owner()

    def _stored_owner(self) -> str | None:
        row = self._store.connection.execute(
            "SELECT value FROM library WHERE key = 'owner_id'"
        ).fetchone()
        return row["value"] if row is not None else None

    def _refresh(self) -> None:
        """Pick up anything the writer published since the last call."""
        self._store.refresh()

    def _bind_owner(self) -> None:
        """Bind the provisioned owner once, and refuse to rebind afterwards."""
        wanted = self._config.owner_user_id
        stored = self._stored_owner()
        if stored is not None and wanted is not None and stored != wanted:
            raise CompanyOwnerMismatchError(
                "This company library is bound to a different provisioned owner"
            )
        if stored is not None or wanted is None:
            return
        if not self.writable:
            # A reader never writes, so it cannot bind. It simply has nothing
            # to serve until the writer has opened once.
            return
        with self._lock, self._store.mutation():
            self._store.connection.execute(
                "INSERT OR REPLACE INTO library (key, value) VALUES ('owner_id', ?)",
                (wanted,),
            )
        logger.info("The company library was bound to its provisioned owner")

    def _may_read(self, owner: object) -> bool:
        self._refresh()
        existing = self._stored_owner()
        return existing is not None and isinstance(owner, str) and owner == existing

    def _require_owner(self, owner: object) -> str:
        if not self.configured:
            raise PermissionError("unconfigured")
        self._refresh()
        existing = self._stored_owner()
        if existing is None or not isinstance(owner, str) or owner != existing:
            raise PermissionError("This company library belongs to another owner")
        return existing

    @staticmethod
    def _forbidden() -> dict[str, Any]:
        return {
            "status": "forbidden",
            "message": "This company library belongs to another account.",
            "results": [],
            "documents": [],
        }

    def _unconfigured(self, message: str) -> dict[str, Any]:
        return {
            "status": "unconfigured",
            "message": message,
            "results": [],
            "documents": [],
            "subjects": [],
        }

    def _refused(self, owner: object) -> dict[str, Any] | None:
        """The refusal for this caller, or ``None`` when they may proceed."""
        if not self.configured:
            return self._unconfigured(
                "No owner has been provisioned for this company library, so it holds "
                "nothing and answers nothing."
            )
        self._refresh()
        if self._stored_owner() is None:
            # Configured, but the single writer has not opened the library
            # even once. A reader has nothing to serve and does not invent it.
            return self._unconfigured(
                "The company library has not been initialised by its owner process yet."
            )
        if not self._may_read(owner):
            return self._forbidden()
        return None

    # --- subjects -------------------------------------------------------------------------

    def add_subject(
        self, *, owner: str, display_name: str, aliases: tuple[str, ...] | list[str] = ()
    ) -> Subject:
        """Create an employee subject. Two people with one name stay two subjects."""
        name = _clean(display_name, limit=_MAX_NAME, name="display_name")
        cleaned = tuple(
            text
            for text in (
                _clean(alias, limit=_MAX_NAME, name="alias", required=False)
                for alias in list(aliases)[:_MAX_ALIASES]
            )
            if text
        )
        with self._lock:
            self._require_owner(owner)
            subject = Subject(_identifier("emp", width=16), name or "", cleaned)
            with self._store.mutation():
                self._store.connection.execute(
                    "INSERT INTO subjects (subject_id, display_name, aliases, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        subject.subject_id,
                        subject.display_name,
                        json.dumps(cleaned),
                        int(time.time()),
                    ),
                )
        return subject

    def list_subjects(
        self, *, owner: str, limit: int | None = None, cursor: int = 0
    ) -> dict[str, Any]:
        refusal = self._refused(owner)
        if refusal is not None:
            return refusal
        page = max(1, min(int(limit or self.MAX_PEOPLE_PAGE), self.MAX_PEOPLE_PAGE))
        offset = max(0, int(cursor or 0))
        total = int(
            self._store.connection.execute("SELECT COUNT(*) AS n FROM subjects").fetchone()["n"]
        )
        rows = self._store.connection.execute(
            "SELECT subject_id, display_name, aliases FROM subjects "
            "ORDER BY display_name, subject_id LIMIT ? OFFSET ?",
            (page + 1, offset),
        ).fetchall()
        more = len(rows) > page
        return {
            "status": "ok",
            "subjects": [
                {
                    "subject_id": row["subject_id"],
                    "display_name": row["display_name"],
                    "aliases": json.loads(row["aliases"] or "[]"),
                }
                for row in rows[:page]
            ],
            "subject_count": total,
            "next_cursor": offset + page if more else None,
        }

    def resolve_subject(self, *, owner: str, name: str) -> dict[str, Any]:
        """Map a spoken name to one subject, or refuse to guess between people."""
        refusal = self._refused(owner)
        if refusal is not None:
            return refusal
        wanted = (name or "").strip().casefold()
        if not wanted:
            return {"status": "no_subject_match", "candidates": []}
        matches = []
        for row in self._store.connection.execute(
            "SELECT subject_id, display_name, aliases FROM subjects"
        ).fetchall():
            names = [row["display_name"], *json.loads(row["aliases"] or "[]")]
            if any(wanted == text.strip().casefold() for text in names if text):
                matches.append(
                    {"subject_id": row["subject_id"], "display_name": row["display_name"]}
                )
            if len(matches) > self.MAX_PEOPLE_PAGE:
                break
        if not matches:
            return {
                "status": "no_subject_match",
                "candidates": [],
                "message": "Nobody in the company library goes by that name.",
            }
        if len(matches) > 1:
            # Deliberately no documents: two people share this name, and
            # picking one of them would be the worst possible guess.
            return {
                "status": "ambiguous_subject",
                "candidates": matches[: self.MAX_PEOPLE_PAGE],
                "message": (
                    f"{len(matches)} people in the company library go by that name. "
                    "Say which one is meant."
                ),
            }
        return {"status": "ok", **matches[0], "candidates": matches}

    # --- ingest ---------------------------------------------------------------------------

    def ingest(
        self,
        *,
        owner: str,
        filename: str,
        data: bytes,
        title: str,
        classification: str,
        status: str,
        effective_date: str | None = None,
        version_label: str | None = None,
        subjects: tuple[str, ...] | list[str] = (),
        document_id: str | None = None,
    ) -> IngestResult:
        """Add one file, as a new document or a new version of an existing one."""
        heading = _clean(title, limit=_MAX_TITLE, name="title")
        label = _clean(version_label, limit=_MAX_LABEL, name="version_label", required=False)
        if classification not in CLASSIFICATIONS:
            raise ValueError(f"classification must be one of {', '.join(CLASSIFICATIONS)}")
        if status not in STATUSES:
            raise ValueError(f"status must be one of {', '.join(STATUSES)}")
        if effective_date is not None and not _EFFECTIVE_DATE.fullmatch(str(effective_date)):
            raise ValueError("effective_date must be an ISO date, YYYY-MM-DD")
        if not isinstance(data, (bytes, bytearray)) or not data:
            raise ValueError("the upload is empty")

        digest = hashlib.sha256(bytes(data)).hexdigest()
        with self._lock:
            keeper = self._require_owner(owner)
            connection = self._store.connection

            if document_id is not None:
                existing = connection.execute(
                    "SELECT document_id FROM documents WHERE document_id = ?", (document_id,)
                ).fetchone()
                if existing is None:
                    return IngestResult(status="not_found", reason="no such document")
                duplicate = connection.execute(
                    "SELECT version_id FROM versions WHERE document_id = ? AND content_sha = ?",
                    (document_id, digest),
                ).fetchone()
                if duplicate is not None:
                    return IngestResult(
                        status="duplicate",
                        document_id=document_id,
                        version_id=duplicate["version_id"],
                        reason="these exact bytes are already a version of this document",
                    )
                versions = int(
                    connection.execute(
                        "SELECT COUNT(*) AS n FROM versions WHERE document_id = ?",
                        (document_id,),
                    ).fetchone()["n"]
                )
                if versions >= self.MAX_VERSIONS_PER_DOCUMENT:
                    return IngestResult(
                        status="capacity_exceeded",
                        reason=(
                            f"a document holds at most {self.MAX_VERSIONS_PER_DOCUMENT} versions"
                        ),
                    )
            else:
                count = connection.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
                if count >= self.MAX_DOCUMENTS:
                    return IngestResult(
                        status="capacity_exceeded",
                        reason=f"this library holds at most {self.MAX_DOCUMENTS} documents",
                    )

            known = {
                row["subject_id"]
                for row in connection.execute("SELECT subject_id FROM subjects")
            }
            wanted = [s for s in list(subjects)[: _MAX_ALIASES * 4] if isinstance(s, str)]
            unknown = [s for s in wanted if s not in known]
            if unknown:
                raise ValueError("an employee subject id is not in this library")

        # Parsing happens outside the lock: it is a bounded subprocess, and a
        # slow file must not block a search.
        parsed = extract(filename, bytes(data))
        chunks = _chunk(parsed.blocks, limit=self.MAX_CHUNK_CHARS) if parsed.status == "ok" else []
        coverage = _coverage(parsed, len(chunks))

        with self._lock:
            connection = self._store.connection
            if parsed.status == "ok":
                indexed = connection.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
                if indexed + len(chunks) > self.MAX_CHUNKS:
                    return IngestResult(
                        status="capacity_exceeded",
                        reason=f"this library holds at most {self.MAX_CHUNKS} passages",
                    )
            now = int(time.time())
            doc = document_id or _identifier("cdoc")
            version_id = _identifier("cver")
            ingest_status = "indexed" if parsed.status == "ok" else "failed"
            # One durable, all-or-nothing change. If the encrypted original
            # cannot be written, or the index cannot be published, nothing
            # below is searchable and nothing can persist it afterwards.
            with self._store.mutation() as mutation:
                if document_id is None:
                    connection.execute(
                        "INSERT INTO documents (document_id, title, created_at) VALUES (?, ?, ?)",
                        (doc, heading, now),
                    )
                connection.execute(
                    "INSERT INTO versions (version_id, document_id, classification, status, "
                    "effective_date, version_label, filename, content_sha, byte_count, "
                    "ingest_status, reason, page_count, chunk_count, coverage, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        version_id,
                        doc,
                        classification,
                        status,
                        effective_date,
                        label,
                        str(filename)[:_MAX_TITLE],
                        digest,
                        len(data),
                        ingest_status,
                        None if parsed.status == "ok" else parsed.status,
                        parsed.page_count,
                        len(chunks),
                        json.dumps(coverage),
                        now,
                    ),
                )
                for subject_id in dict.fromkeys(wanted):
                    connection.execute(
                        "INSERT OR IGNORE INTO version_subjects (version_id, subject_id) "
                        "VALUES (?, ?)",
                        (version_id, subject_id),
                    )
                for ordinal, (text, location) in enumerate(chunks, start=1):
                    chunk_id = _identifier("cchk")
                    connection.execute(
                        "INSERT INTO chunks (chunk_id, version_id, ordinal, location, body) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (chunk_id, version_id, ordinal, location, text),
                    )
                    connection.execute(
                        "INSERT INTO chunk_fts (body, chunk_id) VALUES (?, ?)", (text, chunk_id)
                    )
                # The source is kept whatever the extractor made of it: the
                # owner uploaded it, and a file we could not read is still
                # their file. It is written before the catalogue commits.
                mutation.add_source(
                    bytes(data), owner=keeper, document_id=doc, version_id=version_id
                )

        if parsed.status != "ok":
            logger.info("A company document could not be read (%s)", parsed.status)
            return IngestResult(
                status="failed",
                document_id=doc,
                version_id=version_id,
                reason=parsed.status,
                page_count=parsed.page_count,
                coverage=coverage,
            )
        return IngestResult(
            status="indexed",
            document_id=doc,
            version_id=version_id,
            chunk_count=len(chunks),
            page_count=parsed.page_count,
            coverage=coverage,
        )

    def set_version_status(self, *, owner: str, version_id: str, status: str) -> dict[str, Any]:
        """The only way a version changes status. Nothing is demoted implicitly."""
        if status not in STATUSES:
            raise ValueError(f"status must be one of {', '.join(STATUSES)}")
        with self._lock:
            refusal = self._refused(owner)
            if refusal is not None:
                return refusal
            present = self._store.connection.execute(
                "SELECT version_id FROM versions WHERE version_id = ?", (version_id,)
            ).fetchone()
            if present is None:
                return {"status": "not_found"}
            with self._store.mutation():
                self._store.connection.execute(
                    "UPDATE versions SET status = ? WHERE version_id = ?", (status, version_id)
                )
        return {"status": "updated", "version_id": version_id, "version_status": status}

    # --- deletion -------------------------------------------------------------------------

    def delete_version(self, *, owner: str, version_id: str) -> dict[str, Any]:
        with self._lock:
            refusal = self._refused(owner)
            if refusal is not None:
                return refusal
            row = self._store.connection.execute(
                "SELECT document_id FROM versions WHERE version_id = ?", (version_id,)
            ).fetchone()
            if row is None:
                return {"status": "not_found"}
            try:
                with self._store.mutation() as mutation:
                    self._purge_versions([version_id], mutation)
                    remaining = self._store.connection.execute(
                        "SELECT COUNT(*) AS n FROM versions WHERE document_id = ?",
                        (row["document_id"],),
                    ).fetchone()["n"]
                    if remaining == 0:
                        self._store.connection.execute(
                            "DELETE FROM documents WHERE document_id = ?", (row["document_id"],)
                        )
            except SourceRemovalError:
                return _partial_deletion({"status": "deleted", "version_id": version_id})
        return {
            "status": "deleted",
            "version_id": version_id,
            "sources_removed": True,
            "note": "Removed from the library index and its encrypted source store.",
        }

    def delete_document(self, *, owner: str, document_id: str) -> dict[str, Any]:
        """Remove a document, every version of it, its passages and its originals."""
        with self._lock:
            refusal = self._refused(owner)
            if refusal is not None:
                return refusal
            existing = self._store.connection.execute(
                "SELECT document_id FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
            if existing is None:
                return {"status": "not_found"}
            versions = [
                row["version_id"]
                for row in self._store.connection.execute(
                    "SELECT version_id FROM versions WHERE document_id = ?", (document_id,)
                )
            ]
            outcome = {
                "status": "deleted",
                "document_id": document_id,
                "versions_removed": len(versions),
            }
            try:
                with self._store.mutation() as mutation:
                    self._purge_versions(versions, mutation)
                    self._store.connection.execute(
                        "DELETE FROM documents WHERE document_id = ?", (document_id,)
                    )
            except SourceRemovalError:
                return _partial_deletion(outcome)
        return {
            **outcome,
            "sources_removed": True,
            # Said plainly rather than overclaimed: the library no longer holds
            # it. This is not an erasure of the underlying disk.
            "note": "Removed from the library index and its encrypted source store.",
        }

    def _purge_versions(self, version_ids: list[str], mutation: Any) -> None:
        connection = self._store.connection
        for version_id in version_ids:
            chunk_ids = [
                row["chunk_id"]
                for row in connection.execute(
                    "SELECT chunk_id FROM chunks WHERE version_id = ?", (version_id,)
                )
            ]
            for chunk_id in chunk_ids:
                connection.execute("DELETE FROM chunk_fts WHERE chunk_id = ?", (chunk_id,))
            connection.execute("DELETE FROM chunks WHERE version_id = ?", (version_id,))
            connection.execute("DELETE FROM version_subjects WHERE version_id = ?", (version_id,))
            connection.execute("DELETE FROM versions WHERE version_id = ?", (version_id,))
            mutation.remove_source(version_id)

    # --- reading --------------------------------------------------------------------------

    def list_documents(
        self,
        *,
        owner: str,
        classification: str | None = None,
        status: str | None = None,
        limit: int | None = None,
        cursor: int = 0,
    ) -> dict[str, Any]:
        refusal = self._refused(owner)
        if refusal is not None:
            return refusal
        page = max(1, min(int(limit or self.MAX_PAGE_SIZE), self.MAX_PAGE_SIZE))
        offset = max(0, int(cursor or 0))
        clauses, params = [], []
        if classification is not None:
            if classification not in CLASSIFICATIONS:
                raise ValueError("unknown classification")
            clauses.append("v.classification = ?")
            params.append(classification)
        if status is not None:
            if status not in STATUSES:
                raise ValueError("unknown status")
            clauses.append("v.status = ?")
            params.append(status)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._store.connection.execute(
            "SELECT d.document_id, d.title, d.created_at FROM documents d "
            "WHERE d.document_id IN (SELECT v.document_id FROM versions v" + where + ") "
            "ORDER BY d.created_at DESC, d.document_id LIMIT ? OFFSET ?",
            (*params, page + 1, offset),
        ).fetchall()
        more = len(rows) > page
        documents = []
        for row in rows[:page]:
            versions, total = self._versions_of(row["document_id"])
            documents.append(
                {
                    "document_id": row["document_id"],
                    "title": row["title"],
                    "created_at": row["created_at"],
                    "version_count": total,
                    "versions_truncated": total > len(versions),
                    "versions": [self._version_view(version) for version in versions],
                }
            )
        return {
            "status": "ok",
            "documents": documents,
            "next_cursor": offset + page if more else None,
            "company_name": self._config.company_name,
        }

    def _versions_of(self, document_id: str) -> tuple[list[Any], int]:
        """The newest bounded page of versions, and how many there are in all."""
        total = int(
            self._store.connection.execute(
                "SELECT COUNT(*) AS n FROM versions WHERE document_id = ?", (document_id,)
            ).fetchone()["n"]
        )
        rows = self._store.connection.execute(
            "SELECT * FROM versions WHERE document_id = ? "
            "ORDER BY created_at DESC, version_id LIMIT ?",
            (document_id, self.MAX_VERSIONS_PER_DOCUMENT),
        ).fetchall()
        return list(rows), total

    def _version_view(self, row: Any) -> dict[str, Any]:
        subjects = [
            item["subject_id"]
            for item in self._store.connection.execute(
                "SELECT subject_id FROM version_subjects WHERE version_id = ? LIMIT ?",
                (row["version_id"], _MAX_ALIASES * 4),
            )
        ]
        return {
            "version_id": row["version_id"],
            "classification": row["classification"],
            "status": row["status"],
            "effective_date": row["effective_date"],
            "version_label": row["version_label"],
            "filename": row["filename"],
            "ingest_status": row["ingest_status"],
            "reason": row["reason"],
            "page_count": row["page_count"],
            "chunk_count": row["chunk_count"],
            "byte_count": row["byte_count"],
            "coverage": _decode_coverage(row["coverage"] if "coverage" in row.keys() else None),
            "created_at": row["created_at"],
            "subjects": subjects,
        }

    def search(
        self,
        *,
        owner: str,
        query: str,
        classification: str | None = None,
        subject: str | None = None,
        limit: int | None = None,
        include_drafts: bool = False,
    ) -> dict[str, Any]:
        """Lexical BM25 lookup over current versions. Cited passages or a non-answer."""
        refusal = self._refused(owner)
        if refusal is not None:
            return refusal
        wanted = max(1, min(int(limit or self.MAX_RESULTS), self.MAX_RESULTS))
        expression = _match_expression(str(query or "")[: self.MAX_QUERY_CHARS])
        statuses = _WITH_DRAFTS if include_drafts else _DEFAULT_STATUSES
        envelope: dict[str, Any] = {
            "status": "no_match",
            "results": [],
            "conflicts": [],
            "retrieval": "lexical_bm25",
            "version_policy": "drafts_included" if include_drafts else "current_only",
            "treat_as": "quoted_document_text",
            "company_name": self._config.company_name,
        }
        if expression is None:
            envelope["message"] = "That question has no searchable words in it."
            return envelope

        clauses = [
            "v.ingest_status = 'indexed'",
            f"v.status IN ({','.join('?' * len(statuses))})",
        ]
        params: list[Any] = [expression, *statuses]
        if classification is not None:
            if classification not in CLASSIFICATIONS:
                raise ValueError("unknown classification")
            clauses.append("v.classification = ?")
            params.append(classification)
        if subject is not None:
            clauses.append(
                "EXISTS (SELECT 1 FROM version_subjects vs "
                "WHERE vs.version_id = v.version_id AND vs.subject_id = ?)"
            )
            params.append(subject)
        try:
            rows = self._store.connection.execute(
                "SELECT c.chunk_id, c.location, c.body, v.version_id, v.document_id, "
                "v.classification, v.status, v.effective_date, v.version_label, d.title, "
                "snippet(chunk_fts, 0, '', '', ' ...', 24) AS excerpt "
                "FROM chunk_fts f "
                "JOIN chunks c ON c.chunk_id = f.chunk_id "
                "JOIN versions v ON v.version_id = c.version_id "
                "JOIN documents d ON d.document_id = v.document_id "
                "WHERE chunk_fts MATCH ? AND " + " AND ".join(clauses) + " "
                "ORDER BY bm25(chunk_fts) LIMIT ?",
                (*params, wanted),
            ).fetchall()
        except Exception:  # noqa: BLE001 - a malformed MATCH is a miss, never a 500
            logger.info("A company search expression could not be evaluated")
            return envelope

        results = [
            {
                "document_id": row["document_id"],
                "version_id": row["version_id"],
                "chunk_id": row["chunk_id"],
                "title": row["title"],
                "location": row["location"],
                "classification": row["classification"],
                "version_status": row["status"],
                "effective_date": row["effective_date"],
                "version_label": row["version_label"],
                "snippet": (row["excerpt"] or row["body"])[: self.MAX_SNIPPET_CHARS],
            }
            for row in rows
        ]
        envelope["results"] = results
        if not results:
            envelope["message"] = "Nothing in the company library matches that."
            return envelope

        conflicts = self._conflicts({row["document_id"] for row in rows})
        envelope["conflicts"] = conflicts
        envelope["status"] = "conflicting_versions" if conflicts else "ok"
        if conflicts:
            envelope["message"] = (
                "More than one version of a matching document is marked current. "
                "Say which version is meant, or set one of them superseded."
            )
        return envelope

    def _conflicts(self, document_ids: set[str]) -> list[dict[str, Any]]:
        found = []
        for document_id in sorted(document_ids):
            rows = self._current_versions(document_id)
            if len(rows) > 1:
                found.append(
                    {
                        "document_id": document_id,
                        "current_versions": [
                            {
                                "version_id": row["version_id"],
                                "version_label": row["version_label"],
                                "effective_date": row["effective_date"],
                            }
                            for row in rows
                        ],
                    }
                )
        return found

    def _current_versions(self, document_id: str) -> list[Any]:
        return self._store.connection.execute(
            "SELECT version_id, version_label, effective_date FROM versions "
            "WHERE document_id = ? AND status = 'current' AND ingest_status = 'indexed' "
            "ORDER BY created_at DESC, version_id LIMIT ?",
            (document_id, self.MAX_VERSIONS_PER_DOCUMENT),
        ).fetchall()

    def fetch(
        self,
        *,
        owner: str,
        document_id: str,
        version_id: str | None = None,
        chunk_id: str | None = None,
    ) -> dict[str, Any]:
        """A bounded excerpt of one document with its metadata. Never the whole file.

        Without a ``version_id`` this means *the* current version and nothing
        else: two of them is a conflict the caller must resolve, none of them
        is history or a draft the caller must ask for by name.
        """
        refusal = self._refused(owner)
        if refusal is not None:
            return refusal
        connection = self._store.connection
        if version_id is not None:
            row = connection.execute(
                "SELECT v.*, d.title FROM versions v JOIN documents d "
                "ON d.document_id = v.document_id WHERE v.version_id = ? AND v.document_id = ?",
                (version_id, document_id),
            ).fetchone()
        else:
            document = connection.execute(
                "SELECT document_id, title FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
            if document is None:
                return _not_found()
            current = self._current_versions(document_id)
            if len(current) > 1:
                # The same refusal search gives. A default read must not be the
                # quiet way around a conflict the owner has not resolved.
                return {
                    "status": "conflicting_versions",
                    "document_id": document_id,
                    "title": document["title"],
                    "message": (
                        "Two versions of that document are both marked current. Name the "
                        "version to read, or set one of them superseded."
                    ),
                    "conflicts": self._conflicts({document_id}),
                    "treat_as": "quoted_document_text",
                    "company_name": self._config.company_name,
                }
            if not current:
                versions, total = self._versions_of(document_id)
                return {
                    "status": "no_current_version",
                    "document_id": document_id,
                    "title": document["title"],
                    "message": (
                        "That document has no current, readable version. Drafts and "
                        "superseded history are only read when a version is named."
                    ),
                    "version_count": total,
                    "versions": [self._version_view(version) for version in versions],
                    "treat_as": "quoted_document_text",
                    "company_name": self._config.company_name,
                }
            row = connection.execute(
                "SELECT v.*, d.title FROM versions v JOIN documents d "
                "ON d.document_id = v.document_id WHERE v.version_id = ?",
                (current[0]["version_id"],),
            ).fetchone()
        if row is None:
            return _not_found()
        if row["ingest_status"] != "indexed":
            # A stored file nobody could read is not an empty document. Saying
            # "ok" with no excerpt would be a successful-looking nothing.
            return {
                "status": "extraction_failed",
                "document_id": row["document_id"],
                "version_id": row["version_id"],
                "title": row["title"],
                "ingest_status": row["ingest_status"],
                "reason": row["reason"],
                "message": (
                    "That version is on file but its text could not be read, so there is "
                    "nothing to quote from it."
                ),
                "coverage": _decode_coverage(row["coverage"]),
                "treat_as": "quoted_document_text",
                "company_name": self._config.company_name,
            }
        if chunk_id is not None:
            chunks = connection.execute(
                "SELECT location, body FROM chunks WHERE version_id = ? AND chunk_id = ?",
                (row["version_id"], chunk_id),
            ).fetchall()
        else:
            chunks = connection.execute(
                "SELECT location, body FROM chunks WHERE version_id = ? ORDER BY ordinal LIMIT ?",
                (row["version_id"], self.MAX_EXCERPT_LOCATIONS * 4),
            ).fetchall()
        excerpt = "\n\n".join(f"[{c['location']}] {c['body']}" for c in chunks)
        truncated = len(excerpt) > self.MAX_EXCERPT_CHARS
        excerpt = excerpt[: self.MAX_EXCERPT_CHARS]
        # Only the locations the excerpt actually reaches: a citation list
        # longer than the text it cites is a claim the excerpt does not support.
        locations: list[str] = []
        consumed = 0
        for chunk in chunks:
            consumed += len(chunk["location"]) + len(chunk["body"]) + 5
            if consumed > len(excerpt) + 5 or len(locations) >= self.MAX_EXCERPT_LOCATIONS:
                break
            locations.append(chunk["location"])
        return {
            "status": "ok",
            "document_id": row["document_id"],
            "version_id": row["version_id"],
            "title": row["title"],
            "classification": row["classification"],
            "version_status": row["status"],
            "effective_date": row["effective_date"],
            "version_label": row["version_label"],
            "ingest_status": row["ingest_status"],
            "reason": row["reason"],
            "coverage": _decode_coverage(row["coverage"]),
            "requested_version": version_id is not None,
            "locations": locations,
            "excerpt": excerpt,
            "truncated": truncated,
            "treat_as": "quoted_document_text",
            "company_name": self._config.company_name,
        }

    def status(self, *, owner: str) -> dict[str, Any]:
        refusal = self._refused(owner)
        if refusal is not None:
            return refusal
        connection = self._store.connection

        def count(sql: str, *params: Any) -> int:
            return int(connection.execute(sql, params).fetchone()["n"])

        conflicts = self._conflicts(
            {row["document_id"] for row in connection.execute("SELECT document_id FROM documents")}
        )
        from .extraction import extraction_capabilities

        return {
            "status": "ok",
            "company_name": self._config.company_name,
            "claimed": self._stored_owner() is not None,
            "generation": self._store.generation,
            "role": self._role,
            "document_count": count("SELECT COUNT(*) AS n FROM documents"),
            "indexed_versions": count(
                "SELECT COUNT(*) AS n FROM versions WHERE ingest_status = 'indexed'"
            ),
            "failed_versions": count(
                "SELECT COUNT(*) AS n FROM versions WHERE ingest_status = 'failed'"
            ),
            "passage_count": count("SELECT COUNT(*) AS n FROM chunks"),
            "subject_count": count("SELECT COUNT(*) AS n FROM subjects"),
            "conflicting_documents": [item["document_id"] for item in conflicts],
            "classifications": list(CLASSIFICATIONS),
            "retrieval": "lexical_bm25",
            "version_policy": (
                "Search and read both mean the current version. Drafts are searched only "
                "when asked for, and superseded history only when a version is named."
            ),
            "retrieval_note": (
                "Keyword matching over the document text. It is not semantic search: a word "
                "the document does not contain will not be found."
            ),
            "extraction": extraction_capabilities(),
            "capacity": {
                "documents": self.MAX_DOCUMENTS,
                "passages": self.MAX_CHUNKS,
                "versions_per_document": self.MAX_VERSIONS_PER_DOCUMENT,
            },
        }


# --- helpers ---------------------------------------------------------------------------------


def _not_found() -> dict[str, Any]:
    return {
        "status": "not_found",
        "message": "No document in the company library has that identifier.",
    }


def _partial_deletion(outcome: dict[str, Any]) -> dict[str, Any]:
    """The catalogue entry is gone and its original is not. Said, not swallowed."""
    logger.error("A company deletion committed but an encrypted original survived")
    return {
        **outcome,
        "status": "deleted_with_errors",
        "sources_removed": False,
        "message": (
            "The document was removed from the library, but at least one encrypted "
            "original could not be deleted from disk. It is no longer searchable or "
            "readable through FRIDAY; an operator must remove the file."
        ),
    }


def _coverage(parsed: Extracted, chunk_count: int) -> dict[str, Any]:
    """What the extractor actually got, in numbers the owner can check."""
    return {
        "status": parsed.status,
        "parser": parsed.parser,
        "pages_total": parsed.page_count,
        "pages_with_text": parsed.pages_with_text,
        "pages_without_text": parsed.pages_without_text,
        "blocks": len(parsed.blocks),
        "chunks": chunk_count,
        "characters": sum(len(block.text) for block in parsed.blocks),
        "complete": bool(parsed.complete),
        "warnings": list(parsed.warnings),
    }


def _decode_coverage(raw: object) -> dict[str, Any] | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        loaded = json.loads(raw)
    except ValueError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _chunk(blocks: tuple[Block, ...], *, limit: int) -> list[tuple[str, str]]:
    """Group extracted blocks into bounded passages that keep their location."""
    chunks: list[tuple[str, str]] = []
    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        while len(text) > limit:
            cut = text.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            chunks.append((text[:cut].strip(), block.location))
            text = text[cut:].strip()
        if text:
            chunks.append((text, block.location))
    return chunks


def _match_expression(query: str) -> str | None:
    """Turn free text into a safe FTS5 expression, or ``None`` if there is nothing to match.

    The model and the user both write this string, so it never reaches FTS5
    as syntax: only word characters survive, each one quoted.
    """
    tokens = [token for token in _TOKEN.findall(query or "") if len(token) > 1][:16]
    if not tokens:
        return None
    return " OR ".join('"' + token.replace('"', "") + '"' for token in tokens)
