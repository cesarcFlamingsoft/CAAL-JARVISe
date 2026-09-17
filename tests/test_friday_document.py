"""The delivered download has a branded, non-private filename."""

import pytest

from caal.document_work import DocumentWorker


@pytest.mark.asyncio
async def test_delivered_pdf_filename(tmp_path):
    uploads = []

    async def compose(request, context):
        return "Title: FRIDAY\nA local voice assistant."

    async def deliver(**payload):
        uploads.append(payload)

    worker = DocumentWorker(compose=compose, deliver=deliver, artifact_dir=tmp_path)
    await worker("Create a PDF about yourself and send it to me on Telegram", "")
    assert len(uploads) == 1
    assert uploads[0]["filename"] == "friday-document.pdf"
    assert uploads[0]["content"].startswith(b"%PDF-")
    assert list(tmp_path.iterdir()) == []
