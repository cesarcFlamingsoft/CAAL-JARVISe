"""The management plane: the only way a document enters or leaves the library.

Pinned properties of :mod:`caal.company_api`:

* every route is administrator-only and behind the existing identity boundary;
  a member is refused and so is an administrator who does not own the library;
* upload is the *management* surface, not a model tool: it takes a file and
  the metadata the owner chose, and it never guesses a classification, a
  status or an effective date;
* an unreadable file is reported honestly (``needs_ocr``,
  ``password_required``) and is not counted as indexed;
* deletion removes the document, its versions, its passages and its encrypted
  originals, and says what it did without claiming disk erasure;
* nothing in the responses is a file path, a key, or the service bearer.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from caal import profile_crypto
from caal.company import runtime as company_runtime
from caal.company.config import CompanyConfig
from caal.company.service import CompanyLibrary
from caal.user_api import CurrentUser, require_admin, require_user

OWNER = "usr_" + "a1" * 12
OTHER_ADMIN = "usr_" + "c3" * 12

POLICY = (
    b"FIXTURE Remote Work Policy\n\nSection 1. Eligibility\n"
    b"Fixture employees may work remotely up to three days each week.\n"
)
def _blank_pdf() -> bytes:
    """A valid one-page PDF with no text on it: what a scan looks like."""
    import io

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


SCAN = _blank_pdf()


class _Profile:
    def __init__(self, user_id: str, role: str = "admin") -> None:
        self.user_id = user_id
        self.role = role
        self.status = "active"
        self.display_name = "FIXTURE Admin"


@pytest.fixture()
def library(tmp_path):
    config = CompanyConfig.from_env(
        {
            "CAAL_COMPANY_LIBRARY_DIR": str(tmp_path / "library"),
            "CAAL_COMPANY_LIBRARY_KEYS": profile_crypto.generate_key_material(version=1),
            "CAAL_COMPANY_NAME": "FIXTURE Org",
            # The owner is provisioned before anything is uploaded, never claimed.
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
    from caal import company_api

    monkeypatch.setattr(company_api, "get_library", lambda: library)
    app = FastAPI()
    app.include_router(company_api.router)
    caller = {"user": _Profile(OWNER)}

    def _current() -> CurrentUser:
        return CurrentUser(profile=caller["user"])

    app.dependency_overrides[require_user] = _current
    app.dependency_overrides[require_admin] = _current
    app.dependency_overrides[company_api.throttle_admin_mutation] = _current
    with TestClient(app) as running:
        running.caller = caller  # type: ignore[attr-defined]
        yield running


def _upload(client, *, data=POLICY, name="FIXTURE-remote-work-policy.txt", **fields):
    """The raw-binary upload contract: bytes in the body, metadata in one header envelope.

    Deliberately not multipart: Starlette's multipart parser spools a part past
    one mebibyte to a temporary file, which would put the plaintext of an HR
    document on disk before the library ever encrypted it. And deliberately not
    a query string: that is the part of a request every access log records.
    See ``tests/test_company_upload_contract.py`` for the contract itself.
    """
    import base64
    import json

    from caal.company_api import METADATA_HEADER

    payload = {
        "filename": name,
        "title": "FIXTURE Remote Work Policy",
        "classification": "policy",
        "status": "current",
        "effective_date": "2026-01-01",
    }
    payload.update(fields)
    if isinstance(payload.get("subjects"), str):
        payload["subjects"] = [
            item.strip() for item in payload["subjects"].split(",") if item.strip()
        ]
    payload = {key: value for key, value in payload.items() if value is not None}
    envelope = (
        base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        .decode("ascii")
        .rstrip("=")
    )
    return client.post(
        "/admin/company/documents",
        content=data,
        headers={"Content-Type": "application/octet-stream", METADATA_HEADER: envelope},
    )


# --- the empty library ------------------------------------------------------------------------


def test_a_fresh_library_reports_itself_empty_and_names_its_limits(client):
    response = client.get("/admin/company/status")
    assert response.status_code == 200
    body = response.json()
    assert body["document_count"] == 0
    assert body["retrieval"] == "lexical_bm25"
    assert set(body["accepted_extensions"]) == {".txt", ".md", ".pdf", ".docx"}
    assert body["max_upload_bytes"] > 0
    assert body["company_name"] == "FIXTURE Org"


def test_the_document_list_of_a_fresh_library_is_empty(client):
    assert client.get("/admin/company/documents").json()["documents"] == []


# --- upload ------------------------------------------------------------------------------------


def test_an_upload_is_indexed_and_reported(client):
    response = _upload(client)
    assert response.status_code == 201
    body = response.json()
    assert body["ingest_status"] == "indexed"
    assert body["document_id"].startswith("cdoc_")
    assert body["chunk_count"] >= 1


def test_metadata_is_never_guessed(client):
    assert _upload(client, classification="confidential").status_code == 422
    assert _upload(client, status="approved").status_code == 422
    assert _upload(client, effective_date="last tuesday").status_code == 422


def test_an_unreadable_file_is_reported_not_faked(client):
    response = _upload(client, data=SCAN, name="FIXTURE-scan.pdf", classification="general")
    assert response.status_code == 201
    body = response.json()
    assert body["ingest_status"] == "failed"
    assert body["reason"] == "needs_ocr"
    assert body["chunk_count"] == 0
    assert client.get("/admin/company/status").json()["failed_versions"] == 1


def test_an_unsupported_file_type_is_refused(client):
    response = _upload(client, data=b"MZ\x90\x00binary", name="FIXTURE-tool.exe")
    assert response.status_code == 415


def test_an_oversized_upload_is_refused(client):
    from caal.company.extraction import MAX_SOURCE_BYTES

    response = _upload(client, data=b"x" * (MAX_SOURCE_BYTES + 1), name="FIXTURE-big.txt")
    assert response.status_code == 413


# --- versions and search preview ---------------------------------------------------------------


def test_a_version_status_change_is_explicit(client):
    version_id = _upload(client).json()["version_id"]
    response = client.patch(
        f"/admin/company/versions/{version_id}", json={"status": "superseded"}
    )
    assert response.status_code == 200
    assert client.post("/admin/company/search", json={"query": "remotely"}).json()["results"] == []


def test_the_owner_can_preview_the_search_the_model_will_do(client):
    _upload(client)
    body = client.post(
        "/admin/company/search", json={"query": "remote work three days"}
    ).json()
    assert body["results"][0]["title"] == "FIXTURE Remote Work Policy"
    assert body["results"][0]["location"]
    assert body["retrieval"] == "lexical_bm25"


def test_two_current_versions_are_surfaced_to_the_owner(client):
    first = _upload(client).json()
    second = _upload(
        client,
        data=b"FIXTURE policy v2. Remotely four days each week.",
        name="FIXTURE-v2.txt",
        effective_date=None,
        document_id=first["document_id"],
        version_label="v2",
    )
    assert second.status_code == 201
    body = client.post("/admin/company/search", json={"query": "remotely"}).json()
    assert body["status"] == "conflicting_versions"


# --- people -------------------------------------------------------------------------------------


def test_two_people_with_one_name_are_two_subjects(client):
    first = client.post("/admin/company/people", json={"display_name": "Fixture Person A"})
    second = client.post("/admin/company/people", json={"display_name": "Fixture Person A"})
    assert first.status_code == 201 and second.status_code == 201
    assert first.json()["subject_id"] != second.json()["subject_id"]
    listed = client.get("/admin/company/people").json()["subjects"]
    assert len(listed) == 2


# --- deletion -----------------------------------------------------------------------------------


def test_deletion_removes_the_document_and_says_what_it_did(client):
    document_id = _upload(client).json()["document_id"]
    response = client.delete(f"/admin/company/documents/{document_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "deleted"
    assert "eras" not in body["note"].lower()  # no forensic-erasure claim
    assert client.get("/admin/company/documents").json()["documents"] == []


def test_deleting_something_that_is_not_there_is_a_clean_404(client):
    assert client.delete("/admin/company/documents/cdoc_" + "0" * 24).status_code == 404


# --- authorization --------------------------------------------------------------------------------


def test_an_administrator_who_does_not_own_the_library_is_refused(client):
    _upload(client)
    client.caller["user"] = _Profile(OTHER_ADMIN)
    assert client.get("/admin/company/documents").status_code == 403
    assert client.post("/admin/company/search", json={"query": "remote"}).status_code == 403
    assert _upload(client).status_code == 403
    assert client.delete("/admin/company/documents/cdoc_" + "0" * 24).status_code == 403


def test_nothing_in_a_response_leaks_a_path_or_a_key(client):
    _upload(client)
    # The search is a POST now: its query is the confidential half of the
    # question, and a query string is what every access log records.
    for path, body in (
        ("/admin/company/status", None),
        ("/admin/company/documents", None),
        ("/admin/company/search", {"query": "remote"}),
    ):
        rendered = (
            client.post(path, json=body).text
            if body is not None
            else client.get(path).text
        )
        assert "/tmp" not in rendered
        assert "enc:" not in rendered
        assert "Bearer" not in rendered
