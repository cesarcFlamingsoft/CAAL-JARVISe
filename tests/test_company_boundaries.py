"""The two boundaries the review found holes in: tool arguments and the upload body.

**Arguments.** A company tool call whose scope arguments the model misspelled
used to have those arguments silently dropped, which turned "what do we have on
file for FIXPERSON-A" into an unrestricted search of everything the owner can
see. The review reproduced it:

    argument_error None
    bound_keys ['query', 'user_id']

**The upload body.** The management API parsed multipart before it bounded
anything, and Starlette's multipart parser spools a part past one mebibyte into
a ``SpooledTemporaryFile`` -- the plaintext of an HR document on a temporary
filesystem, before the library had encrypted a byte of it. The contract is now
a raw binary body, and these tests hold the parser out of the path rather than
taking its absence on trust.

Every fixture is synthetic.
"""

from __future__ import annotations

import base64
import json

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from caal import company_api, profile_crypto
from caal.company import runtime as company_runtime
from caal.company.config import CompanyConfig
from caal.company.extraction import MAX_SOURCE_BYTES
from caal.company.service import CompanyLibrary
from caal.tools import create_default_registry
from caal.user_api import CurrentUser, require_admin, require_user

OWNER = "usr_" + "a1" * 12
MEBIBYTE = 1024 * 1024


# --- B7: undeclared arguments -----------------------------------------------------------------


def _company_tool(name: str):
    return create_default_registry().get(name)


def test_an_undeclared_company_filter_is_rejected_before_it_can_be_dropped():
    """The review's reproduction: subject and document_id used to just vanish."""
    import importlib

    node = importlib.import_module("caal.llm.llm_node")
    tool = _company_tool("company.search")
    arguments = {
        "query": "salary",
        "subject": "emp_fixture",
        "document_id": "cdoc_fixture",
    }

    refusal = node._connected_argument_error(tool, arguments)
    assert refusal is not None
    assert refusal["status"] == "invalid_request"
    assert "employee" in refusal["message"]


def test_a_declared_company_argument_is_accepted():
    import importlib

    node = importlib.import_module("caal.llm.llm_node")
    tool = _company_tool("company.search")
    assert node._connected_argument_error(tool, {"query": "notice", "person": "FIXPERSON-A"}) is None


def test_the_model_cannot_choose_whose_library_is_read():
    """`user_id` is stripped and replaced with the session's verified scope."""
    from caal.user_scope import UserScope, scoped_tool_arguments

    tool = _company_tool("company.search")
    bound = scoped_tool_arguments(
        tool, {"query": "notice", "user_id": "usr_" + "ff" * 12}, UserScope(user_id=OWNER, identity_configured=True)
    )
    assert bound == {"query": "notice", "user_id": OWNER}


def test_the_refusal_happens_before_the_arguments_are_bound():
    """Order matters: dropping first is what widened the read."""
    import importlib
    import inspect

    node = importlib.import_module("caal.llm.llm_node")
    source = inspect.getsource(node._execute_single_tool)
    assert source.index("_connected_argument_error") < source.index("scoped_tool_arguments")


# --- B8: the upload body ----------------------------------------------------------------------


class _Profile:
    def __init__(self, user_id: str = OWNER) -> None:
        self.user_id = user_id
        self.role = "admin"
        self.status = "active"
        self.display_name = "FIXTURE Admin"


@pytest.fixture()
def library(tmp_path):
    config = CompanyConfig.from_env(
        {
            "CAAL_COMPANY_LIBRARY_DIR": str(tmp_path / "library"),
            "CAAL_COMPANY_LIBRARY_KEYS": profile_crypto.generate_key_material(version=1),
            "CAAL_COMPANY_OWNER_USER_ID": OWNER,
            "CAAL_COMPANY_ROLE": "owner",
        }
    )
    service = CompanyLibrary(config)
    company_runtime.reset()
    try:
        yield service
    finally:
        service.close()
        company_runtime.reset()


@pytest.fixture()
def client(library, monkeypatch):
    monkeypatch.setattr(company_api, "get_library", lambda: library)
    app = FastAPI()
    app.include_router(company_api.router)

    async def _current() -> CurrentUser:
        return CurrentUser(profile=_Profile())

    app.dependency_overrides[require_user] = _current
    app.dependency_overrides[require_admin] = _current
    app.dependency_overrides[company_api.throttle_admin_mutation] = _current
    with TestClient(app) as running:
        yield running


@pytest.fixture()
def no_multipart(monkeypatch):
    """Make any multipart parse an immediate, loud failure."""
    import starlette.formparsers

    class _Refuser:
        def __init__(self, *args, **kwargs):
            raise AssertionError("the company upload path parsed multipart")

    monkeypatch.setattr(starlette.formparsers, "MultiPartParser", _Refuser)
    monkeypatch.setattr(starlette.requests, "MultiPartParser", _Refuser, raising=False)
    return _Refuser


def _envelope(**fields) -> str:
    """Encode the upload metadata the way the current contract carries it.

    One bounded base64url JSON object in ``X-CAAL-Company-Metadata`` -- the
    filename included. Nothing about an upload travels in the query string any
    more, because a query string is what every access log in the path records.
    """
    payload = json.dumps(fields, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _metadata(*, name="FIXTURE-policy.txt", **fields) -> dict[str, str]:
    metadata = {
        "filename": name,
        "title": "FIXTURE Remote Work Policy",
        "classification": "policy",
        "status": "current",
    }
    metadata.update(fields)
    return {company_api.METADATA_HEADER: _envelope(**metadata)}


def _post(client, data: bytes, *, name="FIXTURE-policy.txt", headers=None, **fields):
    sent = {"Content-Type": "application/octet-stream", **_metadata(name=name, **fields)}
    sent.update(headers or {})
    return client.post("/admin/company/documents", content=data, headers=sent)


def test_an_upload_larger_than_the_spool_threshold_never_parses_multipart(
    client, no_multipart
):
    """Well past 1 MiB: exactly the size that used to land on a temporary disk."""
    paragraph = "FIXTURE clause. " * 26  # ~416 bytes, well under the block bound
    body = "\n\n".join(f"{index}. {paragraph}" for index in range(3_000)).encode("utf-8")
    assert len(body) > MEBIBYTE
    response = _post(client, body)
    assert response.status_code == 201
    assert response.json()["ingest_status"] == "indexed"


def test_an_upload_at_the_maximum_is_accepted_and_one_byte_more_is_not(client, no_multipart):
    filler = b"FIXTURE padding line.\n"
    at_bound = (filler * ((MAX_SOURCE_BYTES // len(filler)) + 1))[:MAX_SOURCE_BYTES]
    assert len(at_bound) == MAX_SOURCE_BYTES
    assert _post(client, at_bound, name="FIXTURE-max.txt").status_code == 201
    assert _post(client, at_bound + b"x", name="FIXTURE-over.txt").status_code == 413


def test_a_declared_length_past_the_bound_is_refused_before_the_body_is_read(client):
    response = client.post(
        "/admin/company/documents",
        content=b"FIXTURE small body",
        headers={
            "Content-Type": "application/octet-stream",
            **_metadata(name="FIXTURE.txt", title="FIXTURE"),
            "Content-Length": str(MAX_SOURCE_BYTES + 1),
        },
    )
    assert response.status_code == 413


def test_a_body_that_does_not_match_its_declared_length_is_refused(client):
    response = client.post(
        "/admin/company/documents",
        content=b"FIXTURE small body",
        headers={
            "Content-Type": "application/octet-stream",
            **_metadata(name="FIXTURE.txt", title="FIXTURE"),
            "Content-Length": "4",
        },
    )
    assert response.status_code == 422


def test_an_oversized_chunked_body_with_no_declared_length_is_still_refused(client):
    """No Content-Length at all: the stream is counted as it arrives."""

    def _chunks():
        sent = 0
        block = b"F" * 65_536
        while sent <= MAX_SOURCE_BYTES:
            yield block
            sent += len(block)

    response = client.post(
        "/admin/company/documents",
        content=_chunks(),
        headers={
            "Content-Type": "application/octet-stream",
            **_metadata(name="FIXTURE-stream.txt", title="FIXTURE"),
        },
    )
    assert response.status_code == 413


def test_a_multipart_upload_is_refused_rather_than_parsed(client, no_multipart):
    response = client.post(
        "/admin/company/documents",
        files={"file": ("FIXTURE.txt", b"FIXTURE body", "text/plain")},
        headers=_metadata(name="FIXTURE.txt", title="FIXTURE"),
    )
    assert response.status_code == 415


def test_an_upload_with_no_metadata_envelope_is_refused(client):
    """The filename lives in the envelope now, so a missing envelope is a refusal."""
    response = client.post(
        "/admin/company/documents",
        content=b"FIXTURE body",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 422
    assert company_api.METADATA_HEADER in response.json()["detail"]


def test_an_envelope_with_no_filename_is_refused(client):
    """Same boundary as before, moved: nothing is stored under a guessed name."""
    response = client.post(
        "/admin/company/documents",
        content=b"FIXTURE body",
        headers={
            "Content-Type": "application/octet-stream",
            company_api.METADATA_HEADER: _envelope(
                title="FIXTURE", classification="policy", status="current"
            ),
        },
    )
    assert response.status_code == 422
    assert "filename" in response.json()["detail"]


def test_no_part_of_an_upload_travels_in_the_query_string(client, library):
    """The envelope is the whole contract: query parameters buy nothing."""
    response = client.post(
        "/admin/company/documents",
        params={
            "title": "FIXTURE query title",
            "classification": "policy",
            "status": "current",
            "filename": "FIXTURE-query.txt",
        },
        content=b"FIXTURE clause: thirty days written notice.",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 422
    assert library.status(owner=OWNER)["document_count"] == 0


def test_metadata_is_still_validated_before_the_body_is_touched(client):
    assert _post(client, b"FIXTURE body", classification="confidential").status_code == 422
    assert _post(client, b"FIXTURE body", status="approved").status_code == 422
    assert _post(client, b"FIXTURE body", effective_date="last tuesday").status_code == 422


def test_an_upload_leaves_no_readable_bytes_in_the_library_directory(client, library):
    canary = "zzqx-fixture-upload-canary-8814"
    body = (f"FIXTURE clause. {canary}\n\n" * 40_000).encode("utf-8")
    assert len(body) > MEBIBYTE
    assert _post(client, body, name="FIXTURE-big.txt").status_code == 201

    needle = canary.encode("utf-8")
    for path in library._config.data_dir.rglob("*"):
        if path.is_file():
            assert needle not in path.read_bytes(), path
