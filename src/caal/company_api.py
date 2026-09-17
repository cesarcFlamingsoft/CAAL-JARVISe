"""The owner's management surface for the company document library.

``GET  /admin/company/status``               what is in the library, what it can hold,
                                             which formats it takes and what retrieval it does
``GET  /admin/company/documents``            the catalogue, paginated, with version metadata
``POST /admin/company/documents``            upload one file: the **raw bytes** as the body and
                                             every piece of metadata -- including the filename --
                                             in one bounded base64url JSON envelope in the
                                             ``X-CAAL-Company-Metadata`` header. Deliberately not
                                             multipart and deliberately not a query string; see
                                             :func:`_upload_metadata` and :func:`_bounded_body`.
``PATCH /admin/company/versions/{id}``       change one version's status, explicitly
``DELETE /admin/company/documents/{id}``     remove a document, its versions and its originals
``POST /admin/company/search``               the same lookup the model does, so the owner can
                                             see exactly what FRIDAY would find. A POST with the
                                             query in the body: a read, but never in a URL, because
                                             a query string is what every access log records. See
                                             :func:`preview_search`
``GET|POST /admin/company/people``           employee subjects: identifiers the owner assigns

This is the **only** way a byte enters or leaves the library. The model's MCP
surface is read only and has no counterpart to any route here.

Every route sits behind the identity boundary of :mod:`caal.user_api`: a
single-use ``caal-backend`` principal from the BFF names the user, the user is
loaded from the database on every call, and administrator is decided there --
never from a header or a body. Ownership is decided once more, by the library
itself, so an administrator of this deployment who does not own this library
is refused with ``403``.

Nothing here returns a filesystem path, a key, the MCP service bearer, or the
contents of a file beyond the bounded excerpts the library already publishes.
Nothing here logs a title, a query or a passage.

**No route here takes a sensitive value in the URL.** Titles, filenames,
subject ids and search queries travel in a header envelope or a request body;
the only query parameters left are the catalogue's own enumerated filters and a
numeric cursor. ``tests/test_company_search_transport.py`` enumerates the
router and fails if that stops being true.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import re
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .company.extraction import MAX_SOURCE_BYTES, extraction_capabilities, supported_extensions
from .company.runtime import get_library
from .company.service import CLASSIFICATIONS, STATUSES
from .user_api import CurrentUser, require_admin, throttle_admin_mutation

logger = logging.getLogger(__name__)

__all__ = ["router"]

_UNCONFIGURED = (
    "The company library is not configured on this FRIDAY backend. An operator sets "
    "CAAL_COMPANY_LIBRARY_KEYS, CAAL_COMPANY_LIBRARY_DIR and CAAL_COMPANY_OWNER_USER_ID "
    "to enable it."
)
_READ_ONLY = (
    "This FRIDAY backend is not the writer for the company library. Exactly one process "
    "holds that role (CAAL_COMPANY_ROLE=owner); an operator must point the management "
    "plane at it."
)
_UPLOAD_CONTENT_TYPE = "application/octet-stream"

#: Where every piece of upload metadata travels. Named in
#: :data:`caal.log_privacy.SENSITIVE_HEADERS`, because its value is a document
#: title, a filename and a list of employee subject ids.
METADATA_HEADER = "X-CAAL-Company-Metadata"
#: Bound on the raw header, checked before it is decoded.
MAX_METADATA_HEADER_BYTES = 2048
#: Bound on the decoded JSON, checked before it is parsed.
MAX_METADATA_JSON_BYTES = 1536

_SUBJECT_ID = re.compile(r"^emp_[a-f0-9]{16}$")
_DOCUMENT_ID = re.compile(r"^cdoc_[a-f0-9]{24}$")
_EFFECTIVE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

Classification = Literal["policy", "hr", "contract", "general"]
VersionStatus = Literal["current", "superseded", "draft"]

router = APIRouter(prefix="/admin/company", tags=["company"])


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VersionStatusRequest(_Strict):
    status: VersionStatus


class UploadMetadata(_Strict):
    """Everything about an upload except its bytes. Strict: an unknown field is a refusal.

    ``extra="forbid"`` is the point of the model. A field this version does not
    know is a caller that thinks it is asking for something, and quietly
    dropping it would mean storing a document under metadata nobody chose.
    """

    filename: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=200)
    classification: Classification
    status: VersionStatus
    effective_date: str | None = Field(default=None, max_length=10)
    version_label: str | None = Field(default=None, max_length=60)
    document_id: str | None = Field(default=None, max_length=40)
    subjects: list[str] = Field(default_factory=list, max_length=32)


class SearchRequest(_Strict):
    """A search, in the body. Strict, bounded, and never in a URL.

    ``subject`` is an employee subject id, which names a person; ``query`` is
    the question itself. Both are exactly what an access log must not hold, so
    neither has a query-parameter form any more.
    """

    query: str = Field(min_length=1, max_length=300)
    classification: Classification | None = None
    subject: str | None = Field(default=None, max_length=64)


class SubjectRequest(_Strict):
    display_name: str = Field(min_length=1, max_length=120)
    aliases: list[str] = Field(default_factory=list, max_length=8)


def _library():
    """The in-process library, or a 503 that says how to turn it on."""
    library = get_library()
    if library is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_UNCONFIGURED
        )
    return library


def _owned(payload: dict[str, Any]) -> dict[str, Any]:
    """Turn the library's own refusal into the HTTP one. Never says who owns it."""
    if payload.get("status") == "forbidden":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")
    if payload.get("status") == "unconfigured":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(payload.get("message") or _UNCONFIGURED),
        )
    return payload


# --- status and catalogue ------------------------------------------------------------------


@router.get("/status")
async def company_status(user: CurrentUser = Depends(require_admin)) -> dict[str, Any]:
    library = _library()
    report = _owned(library.status(owner=user.profile.user_id))
    capabilities = extraction_capabilities()
    accepted = [
        extension
        for extension in supported_extensions()
        if extension != ".pdf" or capabilities["pdf_supported"]
    ]
    return {
        **report,
        # What the upload form must tell the user before they pick a file.
        "accepted_extensions": accepted,
        "max_upload_bytes": MAX_SOURCE_BYTES,
        "classification_options": list(CLASSIFICATIONS),
        "status_options": list(STATUSES),
        "writable": bool(getattr(library, "writable", False)),
        "unsupported_note": (
            "Scanned pages and password-protected PDFs cannot be read: they are stored and "
            "reported as needing OCR or a password, never indexed as if they had worked. "
            "A PDF where only some pages hold text is indexed with a warning naming the "
            "pages that were not read."
        ),
    }


@router.get("/documents")
async def list_documents(
    user: CurrentUser = Depends(require_admin),
    classification: Classification | None = None,
    version_status: VersionStatus | None = Query(default=None, alias="status"),
    cursor: int = Query(default=0, ge=0, le=100_000),
) -> dict[str, Any]:
    library = _library()
    try:
        return _owned(
            library.list_documents(
                owner=user.profile.user_id,
                classification=classification,
                status=version_status,
                cursor=cursor,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/search")
async def preview_search(
    body: SearchRequest,
    user: CurrentUser = Depends(require_admin),
) -> dict[str, Any]:
    """Exactly the lookup the model does, so the owner can see what FRIDAY would find.

    A **POST**, and the query is in the body. Not because searching changes
    anything -- it does not, and this route is deliberately behind the plain
    read dependency rather than the mutation throttle -- but because
    ``?q=what+is+FIXTURE+Person's+severance`` is the confidential half of the
    question, and a query string is recorded by every access log in the path.
    uvicorn logs the request line including the query; so does a reverse proxy
    in combined format; so does the browser's own history. A request body is
    recorded by none of them by default.

    Redacting it in the log formatter instead was considered and rejected: the
    formatter would have to know which paths are sensitive, it does nothing
    about the proxy or the browser, and the leak would sit one new route away
    from coming back. Not putting it in a URL is the version that holds.

    Nothing about the query is echoed into an error message or a log line.
    """
    library = _library()
    return _owned(
        library.search(
            owner=user.profile.user_id,
            query=body.query,
            classification=body.classification,
            subject=body.subject,
        )
    )


# --- upload --------------------------------------------------------------------------------


async def _bounded_body(request: Request) -> bytes:
    """Read the whole request body into memory, bounded, before anything parses it.

    Deliberately **not** multipart. Starlette's multipart parser spools a part
    past one mebibyte into a ``SpooledTemporaryFile``, which writes the
    plaintext of an HR document or a contract to whatever temporary filesystem
    the container has. That is the opposite of the claim this library makes, so
    the contract here is a raw binary body with the metadata in one validated
    header envelope, and the body never leaves this process's memory.

    A declared ``Content-Length`` is checked before a byte is read. A missing
    or understated one is caught anyway, because the stream is counted as it
    arrives and abandoned the moment it passes the bound.

    What this does *not* claim: that the bytes cannot reach a disk at all. They
    are in process memory, and process memory can be swapped or captured in a
    core dump. That is stated, not designed around.
    """
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if content_type != _UPLOAD_CONTENT_TYPE:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"upload the file as a raw {_UPLOAD_CONTENT_TYPE} body",
        )
    declared_text = request.headers.get("content-length")
    declared: int | None = None
    if declared_text is not None:
        if not declared_text.isdigit():
            raise HTTPException(status_code=422, detail="content-length is not a number")
        declared = int(declared_text)
        if declared > MAX_SOURCE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"the maximum upload is {MAX_SOURCE_BYTES} bytes",
            )
    buffer = bytearray()
    async for chunk in request.stream():
        buffer.extend(chunk)
        if len(buffer) > MAX_SOURCE_BYTES:
            # Stop reading rather than finish accumulating something oversized:
            # a lying Content-Length must not buy an unbounded allocation.
            raise HTTPException(
                status_code=413,
                detail=f"the maximum upload is {MAX_SOURCE_BYTES} bytes",
            )
    if declared is not None and len(buffer) != declared:
        raise HTTPException(status_code=422, detail="the body did not match its content-length")
    return bytes(buffer)


def _upload_metadata(request: Request) -> UploadMetadata:
    """Decode and validate the metadata envelope. Called **before** the body is read.

    The envelope is base64url of a compact JSON object. It is bounded twice --
    once on the header text, once on the decoded JSON -- so a caller cannot
    make this allocate, and it is validated strictly, so an unknown field is a
    422 rather than a silently different document.

    It is a header rather than query parameters because a query string is
    recorded by every access log in the path, and a document title, a filename
    and a list of employee subject ids are precisely what must not be.

    No part of the envelope is ever echoed into an error message or a log.
    """
    raw = request.headers.get(METADATA_HEADER)
    if not raw:
        raise HTTPException(status_code=422, detail=f"{METADATA_HEADER} is required")
    if len(raw) > MAX_METADATA_HEADER_BYTES:
        raise HTTPException(status_code=422, detail=f"{METADATA_HEADER} is too large")
    try:
        decoded = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"{METADATA_HEADER} is not base64url"
        ) from exc
    if len(decoded) > MAX_METADATA_JSON_BYTES:
        raise HTTPException(status_code=422, detail=f"{METADATA_HEADER} is too large")
    try:
        payload = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=422, detail=f"{METADATA_HEADER} is not a JSON object"
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail=f"{METADATA_HEADER} is not a JSON object")
    try:
        metadata = UploadMetadata.model_validate(payload)
    except ValidationError as exc:
        # The field names are ours; the values are the owner's and never print.
        unknown = sorted(
            {str(error["loc"][-1]) for error in exc.errors() if error["type"] == "extra_forbidden"}
        )
        if unknown:
            raise HTTPException(
                status_code=422, detail=f"unexpected metadata fields: {', '.join(unknown)}"
            ) from exc
        fields = sorted({str(error["loc"][-1]) for error in exc.errors()})
        raise HTTPException(
            status_code=422, detail=f"invalid metadata fields: {', '.join(fields)}"
        ) from exc

    if not metadata.title.strip() or not metadata.filename.strip():
        raise HTTPException(status_code=422, detail="invalid metadata fields: title, filename")
    if metadata.effective_date and not _EFFECTIVE_DATE.match(metadata.effective_date):
        raise HTTPException(status_code=422, detail="invalid metadata fields: effective_date")
    if metadata.document_id and not _DOCUMENT_ID.match(metadata.document_id):
        raise HTTPException(status_code=422, detail="invalid metadata fields: document_id")
    if any(not _SUBJECT_ID.match(subject) for subject in metadata.subjects):
        raise HTTPException(status_code=422, detail="invalid metadata fields: subjects")
    return metadata


@router.post("/documents", status_code=status.HTTP_201_CREATED)
async def upload_document(
    request: Request,
    user: CurrentUser = Depends(throttle_admin_mutation),
) -> dict[str, Any]:
    """Add one file. The metadata is the owner's; nothing about it is inferred.

    The file is the raw request body; everything else, filename included, is in
    the ``X-CAAL-Company-Metadata`` envelope, fully validated before a byte of
    the body is read.
    """
    library = _library()
    if not getattr(library, "writable", False):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_READ_ONLY)
    metadata = _upload_metadata(request)
    filename = metadata.filename.strip()
    extension = "." + filename.lower().rpartition(".")[2]
    if extension not in supported_extensions():
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"accepted formats: {', '.join(supported_extensions())}",
        )
    if extension == ".pdf" and not extraction_capabilities()["pdf_supported"]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="PDF support is not installed in this FRIDAY runtime",
        )

    data = await _bounded_body(request)
    if not data:
        raise HTTPException(status_code=422, detail="the upload is empty")

    try:
        result = library.ingest(
            owner=user.profile.user_id,
            filename=filename,
            data=data,
            title=metadata.title.strip(),
            classification=metadata.classification,
            status=metadata.status,
            effective_date=(metadata.effective_date or None),
            version_label=(metadata.version_label or None),
            subjects=tuple(metadata.subjects),
            document_id=(metadata.document_id or None),
        )
    except PermissionError as exc:
        if str(exc) == "unconfigured":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_UNCONFIGURED
            ) from exc
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if result.status == "not_found":
        raise HTTPException(status_code=404, detail="no such document")
    if result.status == "capacity_exceeded":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=result.reason)
    return {
        "status": result.status,
        "document_id": result.document_id,
        "version_id": result.version_id,
        # "indexed" or "failed", said the same way the library says it.
        "ingest_status": "indexed" if result.status == "indexed" else result.status,
        "chunk_count": result.chunk_count,
        "page_count": result.page_count,
        "reason": result.reason,
        # What was actually read out of the file, so the owner is never left to
        # infer coverage from a nonzero passage count.
        "coverage": result.coverage,
    }


# --- versions and deletion -----------------------------------------------------------------


@router.patch("/versions/{version_id}")
async def set_version_status(
    version_id: str,
    body: VersionStatusRequest,
    user: CurrentUser = Depends(throttle_admin_mutation),
) -> dict[str, Any]:
    library = _library()
    result = _owned(
        library.set_version_status(
            owner=user.profile.user_id, version_id=version_id, status=body.status
        )
    )
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="no such version")
    return result


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str, user: CurrentUser = Depends(throttle_admin_mutation)
) -> dict[str, Any]:
    library = _library()
    result = _owned(library.delete_document(owner=user.profile.user_id, document_id=document_id))
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="no such document")
    return result


@router.delete("/versions/{version_id}")
async def delete_version(
    version_id: str, user: CurrentUser = Depends(throttle_admin_mutation)
) -> dict[str, Any]:
    library = _library()
    result = _owned(library.delete_version(owner=user.profile.user_id, version_id=version_id))
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="no such version")
    return result


# --- employee subjects ----------------------------------------------------------------------


@router.get("/people")
async def list_people(user: CurrentUser = Depends(require_admin)) -> dict[str, Any]:
    library = _library()
    return _owned(library.list_subjects(owner=user.profile.user_id))


@router.post("/people", status_code=status.HTTP_201_CREATED)
async def add_person(
    body: SubjectRequest, user: CurrentUser = Depends(throttle_admin_mutation)
) -> dict[str, Any]:
    """Create an employee subject. Two people with one name stay two subjects."""
    library = _library()
    try:
        subject = library.add_subject(
            owner=user.profile.user_id,
            display_name=body.display_name,
            aliases=tuple(body.aliases),
        )
    except PermissionError as exc:
        if str(exc) == "unconfigured":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_UNCONFIGURED
            ) from exc
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "subject_id": subject.subject_id,
        "display_name": subject.display_name,
        "aliases": list(subject.aliases),
    }
