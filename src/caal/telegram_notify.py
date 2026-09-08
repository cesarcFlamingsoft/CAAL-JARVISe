"""Telegram delivery for outbound-call failures and finished documents.

This uses the already authorized Hermes Telegram bot strictly for outbound
notifications. Decisions remain ordinary Telegram replies, so CAAL does not
compete with Hermes for Telegram update polling.

:meth:`TelegramCallNotifier.send_document` is the one way a background task
hands a finished file to the user. It is bounded on every axis a caller could
get wrong: the payload is capped well under Telegram's own limit, the caption
is truncated, and the filename must be a plain basename, so a name assembled
from anything but a constant can never walk out of its directory or arrive as
a path.
"""

from __future__ import annotations

import re
from typing import Protocol

# Telegram caps a message at 4096 characters; stay comfortably below it.
MAX_TEXT_CHARS = 3500
# Telegram caps a document caption at 1024 characters.
MAX_CAPTION_CHARS = 1000
# Bots may upload 50 MB. A short text document is a few kilobytes; anything
# approaching this cap is a bug upstream, not a document.
MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
_TRUNCATION_MARK = "…"

# A document name is a plain basename: letters, digits, dot, dash, underscore.
_SAFE_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


class _TelegramClient(Protocol):
    async def post(self, url: str, **kwargs: object) -> object: ...


class TelegramCallNotifier:
    """Send concise, destination-redacted call fallback prompts."""

    def __init__(self, *, token: str, chat_id: str, client: _TelegramClient) -> None:
        if not token or not chat_id:
            raise ValueError("Telegram token and chat ID are required")
        self._token = token
        self._chat_id = chat_id
        self._client = client

    async def notify_unanswered(self, reason: str) -> None:
        reason_text = {
            "machine-vm": "The call reached voicemail, so I hung up without leaving a message.",
            "machine-ivr": (
                "The call reached an automated menu, so I hung up without leaving a message."
            ),
            "machine-unavailable": "The call could not accept a voicemail, so I hung up.",
        }.get(reason, "The call was not answered, so I hung up without leaving a message.")
        await self._send(
            f"{reason_text}\n\n"
            "Reply with one of these: another number, retry later, or continue through chat."
        )

    async def notify_text(self, text: str) -> None:
        """Send one plain, bounded message. Callers redact before calling."""
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Telegram notification text must not be empty")
        if len(cleaned) > MAX_TEXT_CHARS:
            cleaned = cleaned[: MAX_TEXT_CHARS - len(_TRUNCATION_MARK)] + _TRUNCATION_MARK
        await self._send(cleaned)

    async def send_document(self, *, filename: str, content: bytes, caption: str = "") -> None:
        """Upload one bounded document without exposing a filesystem path.

        ``content`` stays in memory: callers may retain a private temporary
        artifact for crash recovery, but Telegram never receives that path.
        """
        if not isinstance(filename, str) or _SAFE_FILENAME.fullmatch(filename) is None:
            raise ValueError("Telegram document filename must be a safe basename")
        if not isinstance(content, bytes) or not content:
            raise ValueError("Telegram document content must be non-empty bytes")
        if len(content) > MAX_DOCUMENT_BYTES:
            raise ValueError("Telegram document exceeds the configured size limit")
        clean_caption = str(caption).strip()
        if len(clean_caption) > MAX_CAPTION_CHARS:
            clean_caption = (
                clean_caption[: MAX_CAPTION_CHARS - len(_TRUNCATION_MARK)] + _TRUNCATION_MARK
            )
        response = await self._client.post(
            f"https://api.telegram.org/bot{self._token}/sendDocument",
            data={"chat_id": self._chat_id, "caption": clean_caption},
            files={"document": (filename, content, "application/pdf")},
        )
        raise_for_status = getattr(response, "raise_for_status", None)
        if callable(raise_for_status):
            raise_for_status()

    async def _send(self, text: str) -> None:
        payload = {"chat_id": self._chat_id, "text": text}
        response = await self._client.post(
            f"https://api.telegram.org/bot{self._token}/sendMessage", json=payload
        )
        raise_for_status = getattr(response, "raise_for_status", None)
        if callable(raise_for_status):
            raise_for_status()
