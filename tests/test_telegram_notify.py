from __future__ import annotations

import pytest

from caal import telegram_notify
from caal.telegram_notify import TelegramCallNotifier


class _Response:
    def raise_for_status(self) -> None:
        return None


class _Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def post(self, url: str, *, json: dict) -> _Response:
        self.calls.append((url, json))
        return _Response()


@pytest.mark.asyncio
async def test_notifier_sends_safe_no_answer_message_with_decision_options() -> None:
    client = _Client()
    notifier = TelegramCallNotifier(token="secret", chat_id="123", client=client)

    await notifier.notify_unanswered("machine-vm")

    assert client.calls[0][0].endswith("/botsecret/sendMessage")
    payload = client.calls[0][1]
    assert payload["chat_id"] == "123"
    assert "voicemail" in payload["text"].lower()
    assert "another number" in payload["text"].lower()
    assert "retry later" in payload["text"].lower()
    assert "continue through chat" in payload["text"].lower()
    assert "+1" not in payload["text"]


class _DocumentClient:
    """Records multipart uploads the way ``httpx.AsyncClient.post`` receives them."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, dict]] = []

    async def post(self, url: str, *, data: dict, files: dict) -> _Response:
        self.calls.append((url, data, files))
        return _Response()


def _notifier(client: object) -> TelegramCallNotifier:
    return TelegramCallNotifier(token="secret", chat_id="123", client=client)


@pytest.mark.asyncio
async def test_notifier_uploads_a_document_as_a_bounded_multipart_request() -> None:
    client = _DocumentClient()

    await _notifier(client).send_document(
        filename="jarvis-document.pdf", content=b"%PDF-1.4 body", caption="About JARVIS"
    )

    url, data, files = client.calls[0]
    assert url.endswith("/botsecret/sendDocument")
    assert data["chat_id"] == "123"
    assert data["caption"] == "About JARVIS"
    name, content, content_type = files["document"]
    assert (name, content, content_type) == (
        "jarvis-document.pdf",
        b"%PDF-1.4 body",
        "application/pdf",
    )


@pytest.mark.asyncio
async def test_notifier_bounds_the_caption_and_keeps_the_document_whole() -> None:
    client = _DocumentClient()

    await _notifier(client).send_document(
        filename="jarvis-document.pdf", content=b"%PDF-1.4", caption="x" * 5_000
    )

    _, data, files = client.calls[0]
    assert len(data["caption"]) <= telegram_notify.MAX_CAPTION_CHARS
    assert files["document"][1] == b"%PDF-1.4"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("jarvis-document.pdf", b""),
        ("../../etc/passwd", b"%PDF-1.4"),
        ("nested/jarvis.pdf", b"%PDF-1.4"),
        ("", b"%PDF-1.4"),
        ("jarvis-document.pdf", b"x" * (telegram_notify.MAX_DOCUMENT_BYTES + 1)),
    ],
)
async def test_notifier_refuses_an_unsafe_or_unbounded_document(filename, content) -> None:
    client = _DocumentClient()

    with pytest.raises(ValueError):
        await _notifier(client).send_document(
            filename=filename, content=content, caption="About JARVIS"
        )

    assert client.calls == []


@pytest.mark.asyncio
async def test_notifier_surfaces_a_rejected_upload() -> None:
    class _Rejecting(_DocumentClient):
        async def post(self, url: str, *, data: dict, files: dict):
            class _Bad:
                def raise_for_status(self) -> None:
                    raise RuntimeError("400 Bad Request")

            return _Bad()

    with pytest.raises(RuntimeError):
        await _notifier(_Rejecting()).send_document(
            filename="jarvis-document.pdf", content=b"%PDF-1.4", caption="About JARVIS"
        )
