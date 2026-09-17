"""The owner's search query must not be written into an access log.

``GET /admin/company/search?q=...`` put the confidential half of a company
question into the request line. uvicorn's access logger records the request
line -- method, path *and* query string -- and so does every reverse proxy in
front of it and the browser's own history. An independent probe
(`reports/company-knowledge/verify-query-log.txt`) ran a canary query through
the real access-log formatter and the real redactor and found the canary intact
in the record:

    127.0.0.1:1 - "GET /admin/company/search?q=FIXTURE_CONFIDENTIAL_QUERY_992 HTTP/1.1" 200

Redacting it in the formatter was considered and rejected: it would have to
know which paths are sensitive, it does not cover the proxy or the browser
history, and it leaves the leak one route away from returning. So the query
travels in a bounded POST body instead, which no access log records by default.

The route keeps **read** semantics: it is not rate-limited as a mutation, it
changes nothing, and it sits behind the same identity and ownership checks it
always did. What changes is the transport.

Every fixture is synthetic.
"""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from caal import company_api, log_privacy, profile_crypto
from caal.company import runtime as company_runtime
from caal.company.config import CompanyConfig
from caal.company.service import CompanyLibrary
from caal.user_api import CurrentUser, require_admin, require_user

OWNER = "usr_" + "a1" * 12
CANARY = "FIXTURE_CONFIDENTIAL_QUERY_992"


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
def indexed(client, library):
    library.ingest(
        owner=OWNER,
        filename="FIXTURE-policy.txt",
        data=(
            f"FIXTURE clause. The {CANARY} allowance is thirty days of written notice.\n\n"
            "FIXTURE second clause about notice periods and written notice.\n"
        ).encode("utf-8"),
        title="FIXTURE Notice Policy",
        classification="policy",
        status="current",
    )
    return library


# --- the transport ------------------------------------------------------------------------------


def test_the_search_route_takes_a_post_body(client, indexed):
    response = client.post("/admin/company/search", json={"query": CANARY})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_the_query_string_form_is_gone(client, indexed):
    """Not merely deprecated: a GET must not be an alternative way in."""
    response = client.get(f"/admin/company/search?q={CANARY}")
    assert response.status_code == 405


def test_a_query_in_the_url_of_the_post_buys_nothing(client, indexed):
    """The body is the whole contract, so a URL query is not read at all."""
    response = client.post(f"/admin/company/search?q={CANARY}", json={"query": "notice"})
    assert response.status_code == 200
    results = response.json()["results"]
    assert results  # answered from the body's query, not the URL's


def test_the_body_is_strict(client, indexed):
    assert client.post("/admin/company/search", json={}).status_code == 422
    assert client.post("/admin/company/search", json={"query": ""}).status_code == 422
    assert client.post("/admin/company/search", json={"q": CANARY}).status_code == 422
    assert (
        client.post("/admin/company/search", json={"query": "x", "extra": 1}).status_code == 422
    )
    assert client.post("/admin/company/search", json={"query": "x" * 301}).status_code == 422


def test_the_filters_travel_in_the_body_too(client, indexed):
    response = client.post(
        "/admin/company/search", json={"query": "notice", "classification": "policy"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    refused = client.post(
        "/admin/company/search", json={"query": "notice", "classification": "confidential"}
    )
    assert refused.status_code == 422


def test_a_subject_filter_is_validated_and_never_in_a_url(client, indexed):
    response = client.post(
        "/admin/company/search", json={"query": "notice", "subject": "emp_" + "a" * 16}
    )
    assert response.status_code == 200


def test_search_is_a_read_and_is_not_throttled_as_a_mutation(client, indexed):
    """Ten searches in a row: a read limit belongs to the BFF, not to this route."""
    for _ in range(10):
        assert client.post("/admin/company/search", json={"query": "notice"}).status_code == 200


def test_search_changes_nothing(client, indexed):
    before = indexed.status(owner=OWNER)
    client.post("/admin/company/search", json={"query": CANARY})
    assert indexed.status(owner=OWNER) == before


def test_no_company_route_accepts_a_sensitive_query_parameter():
    """Enumerated from the router itself, so a new route cannot quietly add one."""
    sensitive = {"q", "query", "title", "filename", "name", "subject", "subjects", "text"}
    for route in company_api.router.routes:
        dependant = getattr(route, "dependant", None)
        if dependant is None:
            continue
        named = {param.name for param in dependant.query_params}
        assert not (named & sensitive), (route.path, named & sensitive)


# --- the access log -----------------------------------------------------------------------------


def _access_record(method: str, target: str) -> str:
    """One record from uvicorn's real access formatter, through the real redactor."""
    from uvicorn.logging import AccessFormatter

    logger = logging.getLogger("uvicorn.access.fixture")
    record = logger.makeRecord(
        "uvicorn.access",
        logging.INFO,
        "(none)",
        0,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:1", method, target, "1.1", 200),
        None,
    )
    log_privacy.redact_pii_fields(record)
    return AccessFormatter(use_colors=False).format(record)


def test_the_canary_survived_the_old_get_request_line():
    """The probe's finding, kept as a test so the regression is visible."""
    assert CANARY in _access_record("GET", f"/admin/company/search?q={CANARY}")


def test_the_new_request_line_carries_no_query_at_all():
    record = _access_record("POST", "/admin/company/search")
    assert CANARY not in record
    assert "?" not in record


def test_the_query_never_reaches_a_log_or_an_error_through_the_route(client, indexed, caplog):
    with caplog.at_level(logging.DEBUG):
        ok = client.post("/admin/company/search", json={"query": CANARY})
        refused = client.post("/admin/company/search", json={"query": CANARY, "bogus": 1})
    assert ok.status_code == 200
    assert refused.status_code == 422
    assert CANARY not in refused.text
    assert CANARY not in caplog.text


def test_the_body_of_a_search_is_not_a_logged_header(client, indexed):
    """Nothing declares the body loggable, and the metadata header stays redacted."""
    assert "x-caal-company-metadata" in log_privacy.SENSITIVE_HEADERS
    safe = log_privacy.redact_headers(
        {"X-CAAL-Company-Metadata": CANARY, "Content-Type": "application/json"}
    )
    assert safe["X-CAAL-Company-Metadata"] == "<redacted>"
    assert safe["Content-Type"] == "application/json"
