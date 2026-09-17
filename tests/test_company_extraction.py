"""What the company library can actually read out of a document, and what it cannot.

The previous release read PDFs by matching parenthesised strings that were
immediately followed by a text operator, and numbered "pages" by counting
content streams that happened to contain text. Every fixture in the first half
of this file is something that shape of parser gets wrong while still
reporting ``ok``: an ordinary ``[(...)] TJ`` array, a hex string, two content
streams on one page, a blank page in the middle of the document, and a CID
font. A partially readable document that reports success is worse than one
that refuses, because the answer that comes out of it is a confident quotation
from a document with a clause missing.

The second half is about not dropping text quietly at all: a paragraph past
the block bound is split, not truncated, and a document past the block bound
is refused out loud rather than shortened into something that says less than
the original.

Every fixture is synthetic and built here. Nothing is read from disk.
"""

from __future__ import annotations

import io

import pytest

from caal.company import extraction

pypdf = pytest.importorskip("pypdf")


# --- building real PDFs -------------------------------------------------------------------------


def _simple_font(writer):
    from pypdf.generic import DictionaryObject, NameObject

    font = DictionaryObject()
    font.update(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    fonts = DictionaryObject()
    fonts.update({NameObject("/F1"): writer._add_object(font)})
    resources = DictionaryObject()
    resources.update({NameObject("/Font"): fonts})
    return resources


def _page(writer, *streams: bytes, font=None):
    """One real page whose content streams are exactly what is given."""
    from pypdf.generic import ArrayObject, DecodedStreamObject, NameObject

    page = writer.add_blank_page(width=300, height=200)
    page[NameObject("/Resources")] = font if font is not None else _simple_font(writer)
    references = []
    for stream in streams:
        obj = DecodedStreamObject()
        obj.set_data(stream)
        references.append(writer._add_object(obj))
    page[NameObject("/Contents")] = (
        ArrayObject(references) if len(references) > 1 else references[0]
    )
    return page


def _written(writer) -> bytes:
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _cid_font(writer):
    """A Type0 / Identity-H font with a ToUnicode CMap: real CID text."""
    from pypdf.generic import (
        ArrayObject,
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
        NumberObject,
        TextStringObject,
    )

    cmap = DecodedStreamObject()
    cmap.set_data(
        b"/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
        b"/CMapName /FIXTURE-UCS2 def\n/CMapType 2 def\n"
        b"1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
        b"3 beginbfchar\n<0001> <0043>\n<0002> <00C9>\n<0003> <2014>\nendbfchar\n"
        b"endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend"
    )
    descriptor = DictionaryObject()
    descriptor.update(
        {
            NameObject("/Type"): NameObject("/FontDescriptor"),
            NameObject("/FontName"): NameObject("/FIXTUREFont"),
            NameObject("/Flags"): NumberObject(4),
            NameObject("/ItalicAngle"): NumberObject(0),
            NameObject("/Ascent"): NumberObject(800),
            NameObject("/Descent"): NumberObject(-200),
            NameObject("/CapHeight"): NumberObject(700),
            NameObject("/StemV"): NumberObject(80),
            NameObject("/FontBBox"): ArrayObject(
                [NumberObject(0), NumberObject(-200), NumberObject(1000), NumberObject(900)]
            ),
        }
    )
    system = DictionaryObject()
    system.update(
        {
            NameObject("/Registry"): TextStringObject("Adobe"),
            NameObject("/Ordering"): TextStringObject("Identity"),
            NameObject("/Supplement"): NumberObject(0),
        }
    )
    descendant = DictionaryObject()
    descendant.update(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/CIDFontType2"),
            NameObject("/BaseFont"): NameObject("/FIXTUREFont"),
            NameObject("/CIDSystemInfo"): system,
            NameObject("/FontDescriptor"): writer._add_object(descriptor),
            NameObject("/DW"): NumberObject(1000),
        }
    )
    font = DictionaryObject()
    font.update(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type0"),
            NameObject("/BaseFont"): NameObject("/FIXTUREFont"),
            NameObject("/Encoding"): NameObject("/Identity-H"),
            NameObject("/DescendantFonts"): ArrayObject([writer._add_object(descendant)]),
            NameObject("/ToUnicode"): writer._add_object(cmap),
        }
    )
    fonts = DictionaryObject()
    fonts.update({NameObject("/F1"): writer._add_object(font)})
    resources = DictionaryObject()
    resources.update({NameObject("/Font"): fonts})
    return resources


# --- the PDF shapes the old parser got wrong ------------------------------------------------------


def test_a_tj_array_clause_is_extracted():
    """`[(...)] TJ` is how nearly every real PDF shows text. It must not be missed."""
    writer = pypdf.PdfWriter()
    _page(
        writer,
        b"BT /F1 12 Tf 20 150 Td "
        b"[(FIXTURE termination clause) -200 (thirty days written notice)] TJ ET",
    )
    result = extraction.extract_in_process("FIXTURE-contract.pdf", _written(writer))
    assert result.status == "ok"
    assert result.parser == "pypdf"
    body = " ".join(block.text for block in result.blocks)
    assert "FIXTURE termination clause" in body
    assert "thirty days written notice" in body


def test_a_hex_string_and_a_second_content_stream_are_both_read():
    writer = pypdf.PdfWriter()
    _page(
        writer,
        b"BT /F1 12 Tf 20 150 Td <46495854555245204845582043> Tj ET",
        b"BT /F1 12 Tf 20 120 Td (second content stream, same page) Tj ET",
    )
    result = extraction.extract_in_process("FIXTURE-hex.pdf", _written(writer))
    assert result.status == "ok"
    body = " ".join(block.text for block in result.blocks)
    assert "FIXTURE HEX C" in body
    assert "second content stream, same page" in body
    # One page, not two: streams are not pages.
    assert result.page_count == 1
    assert {block.location for block in result.blocks} == {"page 1"}


def test_a_blank_page_does_not_shift_the_citations_after_it():
    writer = pypdf.PdfWriter()
    _page(writer, b"BT /F1 12 Tf 20 150 Td (FIXTURE first page clause) Tj ET")
    writer.add_blank_page(width=300, height=200)
    _page(writer, b"BT /F1 12 Tf 20 150 Td (FIXTURE third page clause) Tj ET")

    result = extraction.extract_in_process("FIXTURE-gappy.pdf", _written(writer))
    assert result.status == "ok"
    located = {block.location: block.text for block in result.blocks}
    assert "FIXTURE first page clause" in located["page 1"]
    # The old parser numbered by nonempty stream and called this "page 2".
    assert "FIXTURE third page clause" in located["page 3"]
    assert result.page_count == 3


def test_a_partly_scanned_pdf_says_which_pages_it_could_not_read():
    writer = pypdf.PdfWriter()
    _page(writer, b"BT /F1 12 Tf 20 150 Td (FIXTURE readable clause) Tj ET")
    writer.add_blank_page(width=300, height=200)

    result = extraction.extract_in_process("FIXTURE-mixed.pdf", _written(writer))
    assert result.status == "ok"
    assert result.complete is False
    assert result.pages_with_text == 1
    assert result.pages_without_text == 1
    assert any("page 2" in warning for warning in result.warnings)


def test_a_pdf_with_no_text_at_all_is_refused_rather_than_indexed_empty():
    writer = pypdf.PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=300, height=200)

    result = extraction.extract_in_process("FIXTURE-scan.pdf", _written(writer))
    assert result.status == "needs_ocr"
    assert result.blocks == ()
    assert result.pages_with_text == 0
    assert result.pages_without_text == 3


def test_a_cid_font_with_a_tounicode_map_produces_real_unicode():
    writer = pypdf.PdfWriter()
    _page(writer, b"BT /F1 12 Tf 20 150 Td <000100020003> Tj ET", font=_cid_font(writer))

    result = extraction.extract_in_process("FIXTURE-cid.pdf", _written(writer))
    assert result.status == "ok"
    assert "CÉ—" in " ".join(block.text for block in result.blocks)


def test_a_file_that_is_not_really_a_pdf_is_malformed_not_a_scan():
    broken = b"%PDF-1.7\nthis is not a pdf at all\n%%EOF\n"
    result = extraction.extract_in_process("FIXTURE-broken.pdf", broken)
    assert result.status == "malformed"
    assert result.blocks == ()


def test_the_runtime_says_what_it_can_read_and_does_not_promise_ocr():
    capabilities = extraction.extraction_capabilities()
    assert capabilities["pdf_supported"] is True
    assert capabilities["pdf_parser"] == "pypdf"
    assert capabilities["ocr"] is False


# --- nothing is dropped quietly -------------------------------------------------------------------


def test_a_paragraph_past_the_block_bound_is_split_not_truncated():
    """The exact review reproduction: the trailing clause must survive."""
    body = ("A" * (extraction.MAX_BLOCK_CHARS + 1) + " OMITTED_FIXTURE_CLAUSE").encode("utf-8")
    result = extraction.extract_in_process("FIXTURE-long.txt", body)
    assert result.status == "ok"
    joined = " ".join(block.text for block in result.blocks)
    assert "OMITTED_FIXTURE_CLAUSE" in joined
    # Split, and the pieces still say where they came from.
    assert len(result.blocks) > 1
    assert all(block.location.startswith("paragraph 1") for block in result.blocks)


def test_a_document_past_the_block_bound_is_refused_rather_than_shortened():
    body = "\n\n".join(f"FIXTURE clause {index}" for index in range(extraction.MAX_BLOCKS + 2))
    result = extraction.extract_in_process("FIXTURE-huge.txt", body.encode("utf-8"))
    assert result.status == "too_large"
    assert result.blocks == ()
    assert "passages" in (result.reason or "")


def test_a_long_docx_paragraph_is_split_not_truncated():
    import random
    import zipfile

    # Deliberately not a run of one character: that compresses far past the
    # zip-bomb ratio guard, which would refuse the file for a different and
    # entirely correct reason and prove nothing about splitting.
    noise = random.Random(20260915)
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    text = (
        "".join(noise.choice(alphabet) for _ in range(extraction.MAX_BLOCK_CHARS + 1))
        + "OMITTED_FIXTURE_DOCX_CLAUSE"
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"
    ).encode("utf-8")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", document)

    result = extraction.extract_in_process("FIXTURE-long.docx", buffer.getvalue())
    assert result.status == "ok"
    assert "OMITTED_FIXTURE_DOCX_CLAUSE" in " ".join(block.text for block in result.blocks)


# --- what the library records about coverage ------------------------------------------------------


def test_the_library_records_what_it_actually_read(tmp_path):
    from caal import profile_crypto
    from caal.company.config import CompanyConfig
    from caal.company.service import CompanyLibrary

    owner = "usr_" + "a1" * 12
    config = CompanyConfig.from_env(
        {
            "CAAL_COMPANY_LIBRARY_DIR": str(tmp_path / "library"),
            "CAAL_COMPANY_LIBRARY_KEYS": profile_crypto.generate_key_material(version=1),
            "CAAL_COMPANY_OWNER_USER_ID": owner,
            "CAAL_COMPANY_ROLE": "owner",
        }
    )
    library = CompanyLibrary(config)
    try:
        writer = pypdf.PdfWriter()
        _page(writer, b"BT /F1 12 Tf 20 150 Td (FIXTURE readable clause) Tj ET")
        writer.add_blank_page(width=300, height=200)

        result = library.ingest(
            owner=owner,
            filename="FIXTURE-mixed.pdf",
            data=_written(writer),
            title="FIXTURE Mixed Handbook",
            classification="policy",
            status="current",
        )
        assert result.status == "indexed"
        assert result.coverage["complete"] is False
        assert result.coverage["pages_without_text"] == 1
        assert result.coverage["parser"] == "pypdf"

        listed = library.list_documents(owner=owner)["documents"][0]["versions"][0]
        assert listed["coverage"]["pages_with_text"] == 1
        assert listed["coverage"]["warnings"]
    finally:
        library.close()
