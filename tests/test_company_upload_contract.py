"""The upload contract: no company detail in a URL, no plaintext spool, bounded first.

Two separate problems, both real.

**The URL.** The first repair moved the upload off multipart and put the
metadata in the query string. That fixed the spool and created a new leak: a
query string is the part of a request that everything records. Next.js logs it,
uvicorn's access log logs it, a reverse proxy logs it, and browser history
keeps it. ``?title=FIXTURE%20Person%20severance%20agreement&subjects=emp_...``
is the confidential part of the document written into an access log in
plaintext -- which is exactly what this library exists to prevent. So the
metadata now travels as one bounded, validated, base64url JSON envelope in a
request **header**, and the filename travels inside it rather than in a header
of its own, because a filename says what a document is.

**The body.** It is still raw ``application/octet-stream``, counted as it
arrives, refused before a byte is allocated past the bound, and never handed to
a multipart parser -- so nothing spools an HR document to a temporary file in
cleartext.

The order matters and is asserted: the envelope is validated *before* the body
is read, so an oversized upload with bad metadata is rejected on the metadata
without the eight mebibytes ever being accepted.
"""

from __future__ import annotations

import base64
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from caal import profile_crypto
from caal.company import runtime as company_runtime
from caal.company.config import CompanyConfig
from caal.company.service import CompanyLibrary
from caal.user_api import CurrentUser, require_admin, require_user

OWNER = "usr_" + "a1" * 12
CANARY = "FIXTURE_SECRET_zzqx7731"

POLICY = (
    b"FIXTURE Remote Work Policy\n\nSection 1. Eligibility\n"
    b"Fixture employees may work remotely up to three days each week.\n"
)

# The kind of metadata that must never appear in a URL.
SENSITIVE_TITLE = f"FIXTURE Person severance agreement {CANARY}"


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

    def _current() -> CurrentUser:
        return CurrentUser(profile=_Profile(OWNER))

    app.dependency_overrides[require_user] = _current
    app.dependency_overrides[require_admin] = _current
    app.dependency_overrides[company_api.throttle_admin_mutation] = _current
    with TestClient(app) as running:
        yield running


def _envelope(**fields) -> str:
    payload = {
        "filename": "FIXTURE-remote-work-policy.txt",
        "title": "FIXTURE Remote Work Policy",
        "classification": "policy",
        "status": "current",
    }
    payload.update(fields)
    payload = {key: value for key, value in payload.items() if value is not None}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _post(client, *, data=POLICY, envelope=None, headers=None, **fields):
    from caal.company_api import METADATA_HEADER

    sent = {
        "Content-Type": "application/octet-stream",
        METADATA_HEADER: _envelope(**fields) if envelope is None else envelope,
    }
    sent.update(headers or {})
    return client.post("/admin/company/documents", content=data, headers=sent)


# --- nothing confidential in the URL ----------------------------------------------------------


def test_a_successful_upload_puts_nothing_in_the_query_string(client):
    response = _post(client, title=SENSITIVE_TITLE)
    assert response.status_code == 201, response.text
    request_url = str(response.request.url)
    assert "?" not in request_url
    assert CANARY not in request_url
    assert "FIXTURE" not in request_url


def test_the_filename_is_inside_the_envelope_not_a_header_of_its_own(client):
    response = _post(client, filename=f"{CANARY}-termination.txt")
    assert response.status_code == 201, response.text
    for name, value in response.request.headers.items():
        if name.lower() == "x-caal-company-metadata":
            continue
        assert CANARY not in str(value), name


def test_metadata_in_the_query_string_is_ignored_entirely(client):
    """A caller that still sends the old contract is refused, not half-honoured."""
    response = client.post(
        "/admin/company/documents",
        params={"title": SENSITIVE_TITLE, "classification": "hr", "status": "current"},
        content=POLICY,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 422
    assert CANARY not in response.text


def test_the_metadata_header_name_is_declared_for_log_suppression(client):
    from caal.company_api import METADATA_HEADER
    from caal.log_privacy import SENSITIVE_HEADERS

    assert METADATA_HEADER.lower() in {name.lower() for name in SENSITIVE_HEADERS}


# --- the envelope is validated, strictly, first -------------------------------------------------


def test_an_envelope_with_an_unknown_field_is_refused(client):
    response = _post(client, secret_flag=True)
    assert response.status_code == 422
    assert "unexpected" in response.text.lower() or "unknown" in response.text.lower()


@pytest.mark.parametrize(
    "fields",
    [
        {"classification": "confidential"},
        {"status": "archived"},
        {"effective_date": "January 2026"},
        {"title": ""},
        {"filename": ""},
        {"title": "x" * 201},
        {"version_label": "y" * 61},
        {"document_id": "not-a-document-id"},
        {"subjects": ["not-a-subject"]},
        {"subjects": "emp_" + "a" * 16},
        {"filename": "policy.exe"},
    ],
)
def test_a_malformed_envelope_field_is_refused(client, fields):
    response = _post(client, **fields)
    assert response.status_code in (415, 422), response.text


def test_an_envelope_that_is_not_base64_is_refused(client):
    assert _post(client, envelope="not base64 at all!!").status_code == 422


def test_an_envelope_that_is_not_json_is_refused(client):
    raw = base64.urlsafe_b64encode(b"just some bytes").decode().rstrip("=")
    assert _post(client, envelope=raw).status_code == 422


def test_a_missing_envelope_is_refused(client):
    response = client.post(
        "/admin/company/documents",
        content=POLICY,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 422


def test_an_oversized_envelope_is_refused_by_its_own_bound(client):
    from caal.company_api import MAX_METADATA_HEADER_BYTES

    padded = _envelope(version_label="v" * 60)
    stuffed = padded + "A" * MAX_METADATA_HEADER_BYTES
    assert _post(client, envelope=stuffed).status_code == 422


# --- the body is bounded before it is allocated -------------------------------------------------


def test_the_envelope_is_rejected_before_the_body_is_read(client):
    """An unreadable envelope must not buy the right to stream eight mebibytes."""
    consumed = {"bytes": 0}

    def _stream():
        # A generator body: whatever the server reads, it reads from here.
        for _ in range(64):
            consumed["bytes"] += 65536
            yield b"\0" * 65536

    from caal.company_api import METADATA_HEADER

    response = client.post(
        "/admin/company/documents",
        content=_stream(),
        headers={"Content-Type": "application/octet-stream", METADATA_HEADER: "!!!not base64"},
    )
    assert response.status_code == 422
    assert consumed["bytes"] == 0


def test_a_body_past_the_bound_is_refused(client):
    from caal.company.extraction import MAX_SOURCE_BYTES

    response = _post(client, data=b"\0" * (MAX_SOURCE_BYTES + 1))
    assert response.status_code == 413


def test_a_body_that_lies_about_its_length_is_still_bounded(client):
    from caal.company.extraction import MAX_SOURCE_BYTES
    from caal.company_api import METADATA_HEADER

    def _stream():
        for _ in range((MAX_SOURCE_BYTES // 65536) + 4):
            yield b"\0" * 65536

    response = client.post(
        "/admin/company/documents",
        content=_stream(),
        headers={
            "Content-Type": "application/octet-stream",
            METADATA_HEADER: _envelope(),
            "Content-Length": "10",
        },
    )
    assert response.status_code in (413, 422)


def test_a_body_smaller_than_its_declared_length_is_refused(client):
    from caal.company_api import METADATA_HEADER

    response = client.post(
        "/admin/company/documents",
        content=POLICY,
        headers={
            "Content-Type": "application/octet-stream",
            METADATA_HEADER: _envelope(),
            "Content-Length": str(len(POLICY) + 100),
        },
    )
    assert response.status_code == 422


def test_a_declared_length_past_the_bound_is_refused_before_reading(client):
    from caal.company.extraction import MAX_SOURCE_BYTES
    from caal.company_api import METADATA_HEADER

    consumed = {"bytes": 0}

    def _stream():
        consumed["bytes"] += 1
        yield b"\0"

    response = client.post(
        "/admin/company/documents",
        content=_stream(),
        headers={
            "Content-Type": "application/octet-stream",
            METADATA_HEADER: _envelope(),
            "Content-Length": str(MAX_SOURCE_BYTES + 1),
        },
    )
    assert response.status_code == 413
    assert consumed["bytes"] == 0


def test_multipart_is_refused_outright(client):
    from caal.company_api import METADATA_HEADER

    response = client.post(
        "/admin/company/documents",
        files={"file": ("policy.txt", POLICY, "text/plain")},
        headers={METADATA_HEADER: _envelope()},
    )
    assert response.status_code == 415


def test_an_empty_body_is_refused(client):
    assert _post(client, data=b"").status_code == 422


# --- the round trip still works ------------------------------------------------------------------


def test_the_envelope_carries_every_field_the_owner_chose(client):
    person = client.post(
        "/admin/company/people", json={"display_name": "FIXTURE Person", "aliases": ["FIXPERSON-A"]}
    )
    assert person.status_code == 201
    subject_id = person.json()["subject_id"]

    response = _post(
        client,
        title="FIXTURE Employment Agreement",
        classification="hr",
        status="draft",
        effective_date="2026-03-01",
        version_label="Rev B",
        subjects=[subject_id],
    )
    assert response.status_code == 201, response.text

    listed = client.get("/admin/company/documents").json()["documents"]
    version = listed[0]["versions"][0]
    assert version["classification"] == "hr"
    assert version["status"] == "draft"
    assert version["effective_date"] == "2026-03-01"
    assert version["version_label"] == "Rev B"
    assert version["subjects"] == [subject_id]


def test_a_second_version_of_the_same_document_still_works(client):
    first = _post(client)
    assert first.status_code == 201
    document_id = first.json()["document_id"]

    second = _post(client, document_id=document_id, version_label="Rev 2", data=POLICY + b"More.\n")
    assert second.status_code == 201, second.text
    assert second.json()["document_id"] == document_id


def test_the_caller_is_told_what_was_actually_extracted(client):
    body = _post(client).json()
    assert body["coverage"] is not None
    assert body["ingest_status"] == "indexed"
