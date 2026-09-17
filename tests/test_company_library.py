"""The local company knowledge library: extraction, encryption, index, lookup.

Pinned properties of :mod:`caal.company`:

* nothing readable reaches the disk -- originals and the full-text index are
  AES-256-GCM under a dedicated key ring, bound to the owner and the exact
  object, and a fixture phrase never appears in any byte under the data dir;
* extraction is local and bounded, and a file it cannot read fails honestly
  (``password_required``, ``needs_ocr``) instead of being indexed empty;
* a version never silently becomes current, two current versions of one
  series are surfaced as a conflict, and a superseded version stays out of
  the default search;
* an employee subject is an identifier, never a name: two people with one
  name stay two subjects and an ambiguous lookup returns no content;
* ingest, replacement and deletion are atomic, and the library comes back
  identical after the process that held it goes away.

Every fixture here is unmistakably synthetic and lives in a temp directory
with its own key ring and its own owner id.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from caal import profile_crypto
from caal.company import extraction
from caal.company.config import CompanyConfig, CompanyConfigError
from caal.company.service import CompanyLibrary

OWNER = "usr_" + "a1" * 12
OTHER = "usr_" + "b2" * 12

# A phrase that exists nowhere else, used to prove it is never on disk.
SECRET_PHRASE = "zzqx-fixture-canary-phrase-7731"


# --- fixtures -------------------------------------------------------------------------------


def _keys() -> str:
    return profile_crypto.generate_key_material(version=1)


@pytest.fixture()
def config(tmp_path):
    return CompanyConfig.from_env(
        {
            "CAAL_COMPANY_LIBRARY_DIR": str(tmp_path / "library"),
            "CAAL_COMPANY_LIBRARY_KEYS": _keys(),
            "CAAL_COMPANY_NAME": "FIXTURE Org",
            "CAAL_COMPANY_OWNER_USER_ID": OWNER,
            "CAAL_COMPANY_ROLE": "owner",
        }
    )


@pytest.fixture()
def library(config):
    service = CompanyLibrary(config)
    try:
        yield service
    finally:
        service.close()


def _docx(paragraphs: list[str], *, extra: dict[str, bytes] | None = None) -> bytes:
    """A minimal real .docx: the one member the extractor is allowed to read."""
    body = "".join(
        "<w:p><w:r><w:t>" + text.replace("&", "&amp;").replace("<", "&lt;") + "</w:t></w:r></w:p>"
        for text in paragraphs
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    ).encode("utf-8")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", document)
        for name, payload in (extra or {}).items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def _pdf(title: str, body: str) -> bytes:
    """A real text PDF, built by the renderer this repo already ships."""
    from caal.pdf_document import render_pdf

    return render_pdf(title=title, body=body)


def _blank_pdf(pages: int = 1) -> bytes:
    """A valid PDF with real pages and no text on any of them."""
    import io

    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _ingest(library, **overrides):
    payload = dict(
        owner=OWNER,
        filename="FIXTURE-remote-work-policy.txt",
        data=(
            "FIXTURE Remote Work Policy\n\n"
            "Section 1. Eligibility\n"
            "Fixture employees may work remotely up to three days each week.\n\n"
            "Section 2. Equipment\n"
            f"The company provides a laptop. Canary {SECRET_PHRASE} appears here.\n"
        ).encode("utf-8"),
        title="FIXTURE Remote Work Policy",
        classification="policy",
        status="current",
        effective_date="2026-01-01",
    )
    payload.update(overrides)
    return library.ingest(**payload)


# --- configuration --------------------------------------------------------------------------


def test_config_requires_its_own_key_ring(tmp_path):
    with pytest.raises(CompanyConfigError):
        CompanyConfig.from_env({"CAAL_COMPANY_LIBRARY_DIR": str(tmp_path)})


def test_config_rejects_malformed_key_material(tmp_path):
    with pytest.raises(CompanyConfigError):
        CompanyConfig.from_env(
            {"CAAL_COMPANY_LIBRARY_DIR": str(tmp_path), "CAAL_COMPANY_LIBRARY_KEYS": "not-a-ring"}
        )


def test_config_company_name_is_optional(tmp_path):
    config = CompanyConfig.from_env(
        {"CAAL_COMPANY_LIBRARY_DIR": str(tmp_path), "CAAL_COMPANY_LIBRARY_KEYS": _keys()}
    )
    assert config.company_name is None


def test_data_dir_is_private(config):
    assert config.data_dir.exists()
    assert config.data_dir.stat().st_mode & 0o777 == 0o700


# --- extraction -----------------------------------------------------------------------------


def test_text_extraction_keeps_paragraph_locations():
    result = extraction.extract("FIXTURE-notes.txt", b"First line.\n\nSecond line.\n")
    assert result.status == "ok"
    assert [block.text for block in result.blocks] == ["First line.", "Second line."]
    assert result.blocks[1].location == "paragraph 2"


def test_markdown_extraction_uses_headings_as_sections():
    body = b"# FIXTURE Handbook\n\nIntro text.\n\n## Leave\n\nLeave text.\n"
    result = extraction.extract("FIXTURE-handbook.md", body)
    assert result.status == "ok"
    assert any(block.location.startswith("section") for block in result.blocks)
    assert any("Leave text." in block.text for block in result.blocks)


def test_docx_extraction_reads_only_the_document_part():
    data = _docx(
        ["FIXTURE clause one.", "FIXTURE clause two."],
        extra={"word/vbaProject.bin": b"\x00macro payload", "word/embeddings/x.bin": b"\x00"},
    )
    result = extraction.extract("FIXTURE-contract.docx", data)
    assert result.status == "ok"
    assert [block.text for block in result.blocks] == [
        "FIXTURE clause one.",
        "FIXTURE clause two.",
    ]


def test_pdf_extraction_reports_pages():
    result = extraction.extract("FIXTURE-policy.pdf", _pdf("FIXTURE Policy", "Payload sentence."))
    assert result.status == "ok"
    assert result.page_count == 1
    assert any("Payload sentence." in block.text for block in result.blocks)
    assert result.blocks[0].location.startswith("page ")


def test_encrypted_pdf_is_refused_not_indexed():
    """A really encrypted PDF, written by pypdf, not a byte-patched string."""
    import io

    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(_pdf("FIXTURE", "text"))))
    writer.encrypt("FIXTURE-password")
    buffer = io.BytesIO()
    writer.write(buffer)

    result = extraction.extract("FIXTURE-locked.pdf", buffer.getvalue())
    assert result.status == "password_required"
    assert result.blocks == ()


def test_pdf_without_text_asks_for_ocr():
    """A structurally valid PDF whose only page carries no text at all."""
    import io

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)

    result = extraction.extract("FIXTURE-scan.pdf", buffer.getvalue())
    assert result.status == "needs_ocr"
    assert result.blocks == ()
    assert result.pages_without_text == 1


def test_a_pdf_that_is_not_a_pdf_is_malformed_not_a_scan():
    broken = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n"
    result = extraction.extract("FIXTURE-broken.pdf", broken)
    assert result.status == "malformed"
    assert result.blocks == ()


def test_extension_and_magic_must_agree():
    result = extraction.extract("FIXTURE-fake.pdf", _docx(["not a pdf"]))
    assert result.status == "unsupported_type"


def test_unknown_extension_is_refused():
    result = extraction.extract("FIXTURE-macro.docm", _docx(["x"]))
    assert result.status == "unsupported_type"


def test_oversized_upload_is_refused():
    result = extraction.extract("FIXTURE-big.txt", b"x" * (extraction.MAX_SOURCE_BYTES + 1))
    assert result.status == "too_large"


def test_docx_decompression_ratio_is_bounded():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", b"\x00" * (extraction.MAX_INFLATED_BYTES + 1024))
    result = extraction.extract("FIXTURE-bomb.docx", buffer.getvalue())
    assert result.status in {"too_large", "malformed"}
    assert result.blocks == ()


# --- ingest, encryption at rest ---------------------------------------------------------------


def test_ingest_indexes_and_reports_honestly(library):
    result = _ingest(library)
    assert result.status == "indexed"
    assert result.document_id.startswith("cdoc_")
    assert result.version_id.startswith("cver_")
    assert result.chunk_count >= 1


def test_nothing_readable_reaches_the_disk(library, config):
    _ingest(library)
    library.flush()
    seen = b""
    for path in sorted(config.data_dir.rglob("*")):
        if path.is_file():
            seen += path.read_bytes()
    assert seen, "the library wrote nothing at all"
    assert SECRET_PHRASE.encode() not in seen
    assert b"Remote Work Policy" not in seen


def test_a_failed_extraction_is_stored_but_not_searchable(library):
    result = library.ingest(
        owner=OWNER,
        filename="FIXTURE-scan.pdf",
        data=_blank_pdf(),
        title="FIXTURE Scanned Handbook",
        classification="general",
        status="current",
    )
    assert result.status == "failed"
    assert result.reason == "needs_ocr"
    assert library.search(owner=OWNER, query="handbook")["results"] == []
    listed = library.list_documents(owner=OWNER)["documents"]
    assert listed[0]["versions"][0]["ingest_status"] == "failed"


def test_identical_bytes_are_deduplicated(library):
    first = _ingest(library)
    again = _ingest(library, document_id=first.document_id)
    assert again.status == "duplicate"
    assert again.version_id == first.version_id


# --- search and citation ------------------------------------------------------------------------


def test_search_returns_cited_bounded_snippets(library):
    ingested = _ingest(library)
    found = library.search(owner=OWNER, query="remotely three days")
    assert found["results"], "the policy phrase was not retrieved"
    hit = found["results"][0]
    assert hit["document_id"] == ingested.document_id
    assert hit["version_id"] == ingested.version_id
    assert hit["chunk_id"].startswith("cchk_")
    assert hit["location"]
    assert hit["classification"] == "policy"
    assert len(hit["snippet"]) <= library.MAX_SNIPPET_CHARS
    assert found["retrieval"] == "lexical_bm25"


def test_search_results_are_labelled_as_quoted_evidence(library):
    _ingest(library)
    found = library.search(owner=OWNER, query="equipment laptop")
    assert found["treat_as"] == "quoted_document_text"


def test_query_length_and_result_count_are_bounded(library):
    _ingest(library)
    found = library.search(owner=OWNER, query="x" * 5000, limit=999)
    assert found["status"] in {"ok", "no_match"}
    assert len(found["results"]) <= library.MAX_RESULTS


def test_search_without_a_match_says_so(library):
    _ingest(library)
    found = library.search(owner=OWNER, query="quarterly dividend reconciliation")
    assert found["status"] == "no_match"
    assert found["results"] == []


def test_fetch_returns_a_bounded_excerpt_with_its_metadata(library):
    ingested = _ingest(library)
    excerpt = library.fetch(owner=OWNER, document_id=ingested.document_id)
    assert excerpt["status"] == "ok"
    assert excerpt["title"] == "FIXTURE Remote Work Policy"
    assert excerpt["classification"] == "policy"
    assert excerpt["effective_date"] == "2026-01-01"
    assert len(excerpt["excerpt"]) <= library.MAX_EXCERPT_CHARS
    assert excerpt["treat_as"] == "quoted_document_text"


def test_fetch_of_an_unknown_document_is_a_clean_miss(library):
    excerpt = library.fetch(owner=OWNER, document_id="cdoc_" + "0" * 24)
    assert excerpt["status"] == "not_found"


# --- ownership -----------------------------------------------------------------------------------


def test_the_library_belongs_to_the_first_owner(library):
    _ingest(library)
    assert library.owner_id() == OWNER


def test_another_owner_sees_nothing_and_cannot_write(library):
    _ingest(library)
    assert library.search(owner=OTHER, query="remotely")["status"] == "forbidden"
    assert library.list_documents(owner=OTHER)["status"] == "forbidden"
    with pytest.raises(PermissionError):
        _ingest(library, owner=OTHER)


# --- versions ------------------------------------------------------------------------------------


def test_a_new_version_does_not_silently_supersede_the_old(library):
    first = _ingest(library)
    second = _ingest(
        library,
        document_id=first.document_id,
        data=b"FIXTURE Remote Work Policy revision two. Four days each week.",
        status="draft",
        version_label="v2 draft",
    )
    assert second.document_id == first.document_id
    versions = library.list_documents(owner=OWNER)["documents"][0]["versions"]
    statuses = {version["version_id"]: version["status"] for version in versions}
    assert statuses[first.version_id] == "current"
    assert statuses[second.version_id] == "draft"


def test_two_current_versions_are_surfaced_as_a_conflict(library):
    first = _ingest(library)
    _ingest(
        library,
        document_id=first.document_id,
        data=b"FIXTURE Remote Work Policy revision two. Employees may work remotely four days.",
        status="current",
        version_label="v2",
    )
    found = library.search(owner=OWNER, query="remotely")
    assert found["status"] == "conflicting_versions"
    assert found["conflicts"]


def test_a_superseded_version_stays_out_of_the_default_search(library):
    first = _ingest(library)
    library.set_version_status(
        owner=OWNER, version_id=first.version_id, status="superseded"
    )
    found = library.search(owner=OWNER, query="remotely three days")
    assert found["status"] == "no_match"
    assert found["results"] == []


def test_status_and_classification_are_never_guessed(library):
    with pytest.raises(ValueError):
        _ingest(library, classification="confidential")
    with pytest.raises(ValueError):
        _ingest(library, status="approved")


# --- employee subjects ----------------------------------------------------------------------------


def test_two_people_with_one_name_stay_two_subjects(library):
    first = library.add_subject(
        owner=OWNER, display_name="Fixture Person A", aliases=("A. Fixture",)
    )
    second = library.add_subject(owner=OWNER, display_name="Fixture Person A")
    assert first.subject_id != second.subject_id
    resolved = library.resolve_subject(owner=OWNER, name="Fixture Person A")
    assert resolved["status"] == "ambiguous_subject"
    assert len(resolved["candidates"]) == 2
    assert "documents" not in resolved


def test_an_unknown_person_is_a_clean_no_match(library):
    library.add_subject(owner=OWNER, display_name="Fixture Person A")
    resolved = library.resolve_subject(owner=OWNER, name="Fixture Person Q")
    assert resolved["status"] == "no_subject_match"


def test_a_subject_scoped_search_only_returns_that_subject(library):
    person = library.add_subject(owner=OWNER, display_name="Fixture Person B")
    other = library.add_subject(owner=OWNER, display_name="Fixture Person C")
    _ingest(
        library,
        filename="FIXTURE-offer-b.txt",
        data=b"FIXTURE offer letter. Fixture Person B holds the title Staff Engineer.",
        title="FIXTURE Offer Letter B",
        classification="hr",
        subjects=(person.subject_id,),
    )
    hits = library.search(owner=OWNER, query="Staff Engineer", subject=person.subject_id)
    assert hits["results"]
    assert library.search(owner=OWNER, query="Staff Engineer", subject=other.subject_id)[
        "results"
    ] == []


# --- deletion -------------------------------------------------------------------------------------


def test_deleting_a_document_removes_its_index_and_its_source(library, config):
    ingested = _ingest(library)
    library.flush()
    removed = library.delete_document(owner=OWNER, document_id=ingested.document_id)
    assert removed["status"] == "deleted"
    library.flush()
    assert library.search(owner=OWNER, query="remotely three days")["results"] == []
    assert library.fetch(owner=OWNER, document_id=ingested.document_id)["status"] == "not_found"
    sources = list((config.data_dir / "sources").glob("*"))
    assert sources == []


def test_delete_is_refused_for_another_owner(library):
    ingested = _ingest(library)
    assert (
        library.delete_document(owner=OTHER, document_id=ingested.document_id)["status"]
        == "forbidden"
    )


# --- persistence ---------------------------------------------------------------------------------


def test_the_library_survives_a_restart(config):
    first = CompanyLibrary(config)
    ingested = _ingest(first)
    first.close()

    second = CompanyLibrary(config)
    try:
        found = second.search(owner=OWNER, query="remotely three days")
        assert found["results"], "the index did not come back"
        assert found["results"][0]["version_id"] == ingested.version_id
        assert second.owner_id() == OWNER
    finally:
        second.close()


def test_a_snapshot_written_under_another_key_does_not_open(config, tmp_path):
    first = CompanyLibrary(config)
    _ingest(first)
    first.close()

    foreign = CompanyConfig.from_env(
        {
            "CAAL_COMPANY_LIBRARY_DIR": str(config.data_dir),
            "CAAL_COMPANY_LIBRARY_KEYS": _keys(),
        }
    )
    with pytest.raises(profile_crypto.DecryptionError):
        CompanyLibrary(foreign)


# --- status and bounds ----------------------------------------------------------------------------


def test_status_reports_counts_and_names_the_retrieval_it_actually_does(library):
    _ingest(library)
    report = library.status(owner=OWNER)
    assert report["document_count"] == 1
    assert report["indexed_versions"] == 1
    assert report["failed_versions"] == 0
    assert report["retrieval"] == "lexical_bm25"
    assert report["company_name"] == "FIXTURE Org"


def test_capacity_is_bounded_and_refuses_rather_than_dropping_text(library, monkeypatch):
    monkeypatch.setattr(library, "MAX_DOCUMENTS", 1, raising=False)
    _ingest(library)
    overflow = _ingest(library, filename="FIXTURE-second.txt", title="FIXTURE Second")
    assert overflow.status == "capacity_exceeded"
