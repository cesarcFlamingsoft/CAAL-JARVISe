"""Native email provider tools for CAAL.

Uses stdlib IMAP/SMTP so custom-domain Zoho/generic accounts work without an
extra provider SDK. Public functions return the voice-tool contract:
{"message": str, "data": dict}.
"""

from __future__ import annotations

import email
import imaplib
import smtplib
from email.message import EmailMessage
from email.utils import getaddresses, parsedate_to_datetime
from typing import Any

from caal import settings as settings_module


def _result(message: str, data: dict[str, Any] | None = None, status: str = "ok") -> dict[str, Any]:
    return {"status": status, "message": message, "data": data or {}}


def _accounts() -> list[dict[str, Any]]:
    return [
        settings_module.apply_email_provider_preset(a)
        for a in settings_module.load_settings().get("email_accounts", [])
        if isinstance(a, dict)
    ]


def _resolve_account(account: str | None = None) -> dict[str, Any]:
    accounts = _accounts()
    if not accounts:
        raise ValueError("No email accounts are configured")

    if account:
        for candidate in accounts:
            if account in {
                str(candidate.get("id", "")),
                str(candidate.get("email", "")),
                str(candidate.get("display_name", "")),
            }:
                return candidate
        raise ValueError(f"Email account not found: {account}")

    for candidate in accounts:
        if candidate.get("default") or candidate.get("is_default"):
            return candidate
    return accounts[0]


def _imap_search_criteria(query: str) -> str:
    query = (query or "ALL").strip()
    if not query or query.upper() == "ALL":
        return "ALL"
    lowered = query.lower()
    if lowered.startswith("from:"):
        return f'(FROM "{query.split(":", 1)[1].strip()}")'
    if lowered.startswith("to:"):
        return f'(TO "{query.split(":", 1)[1].strip()}")'
    if lowered.startswith("subject:"):
        return f'(SUBJECT "{query.split(":", 1)[1].strip()}")'
    return f'(TEXT "{query}")'


def _connect_imap(account: dict[str, Any]):
    host = account.get("imap_host")
    if not host:
        raise ValueError("Selected email account is missing imap_host")
    port = int(account.get("imap_port") or (993 if account.get("imap_ssl", True) else 143))
    cls = imaplib.IMAP4_SSL if account.get("imap_ssl", True) else imaplib.IMAP4
    client = cls(host, port)
    username = account.get("imap_username") or account.get("email")
    password = account.get("imap_password") or account.get("password")
    if username and password:
        client.login(username, password)
    return client


def _parse_message(message_id: str, raw: bytes, include_body: bool = False) -> dict[str, Any]:
    msg = email.message_from_bytes(raw)
    subject = str(msg.get("Subject", ""))
    sender = str(msg.get("From", ""))
    recipients = [addr for _, addr in getaddresses(msg.get_all("To", []))]
    date_header = str(msg.get("Date", ""))
    parsed_date = None
    if date_header:
        try:
            parsed_date = parsedate_to_datetime(date_header).isoformat()
        except Exception:
            parsed_date = date_header

    parsed: dict[str, Any] = {
        "id": str(message_id),
        "subject": subject,
        "from": sender,
        "to": recipients,
        "date": parsed_date,
    }
    if include_body:
        parsed["body"] = _extract_body(msg)
    return parsed


def _extract_body(msg) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and not part.get_filename():
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode(
                        part.get_content_charset() or "utf-8", errors="replace"
                    ).strip()
        return ""
    payload = msg.get_payload(decode=True)
    if payload:
        return payload.decode(msg.get_content_charset() or "utf-8", errors="replace").strip()
    return str(msg.get_payload() or "").strip()


def search_email(account: str | None = None, query: str = "ALL", limit: int = 10) -> dict[str, Any]:
    selected = _resolve_account(account)
    limit = max(1, min(int(limit or 10), 50))
    client = _connect_imap(selected)
    try:
        client.select(selected.get("imap_mailbox", "INBOX"))
        status, data = client.search(None, _imap_search_criteria(query))
        if status != "OK":
            raise RuntimeError(f"IMAP search failed: {status}")
        ids = (data[0] if data else b"").split()[-limit:]
        messages = []
        for msg_id in reversed(ids):
            fetch_status, fetch_data = client.fetch(msg_id, "(RFC822)")
            if fetch_status == "OK" and fetch_data:
                raw = next((item[1] for item in fetch_data if isinstance(item, tuple)), None)
                if raw:
                    messages.append(
                        _parse_message(
                            msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id), raw
                        )
                    )
    finally:
        try:
            client.logout()
        except Exception:
            pass

    count = len(messages)
    plural = "email" if count == 1 else "emails"
    return _result(
        f"Found {count} {plural} matching {query}.",
        {"account": selected.get("id"), "messages": messages},
    )


def read_email(account: str | None = None, message_id: str = "") -> dict[str, Any]:
    if not message_id:
        raise ValueError("message_id is required")
    selected = _resolve_account(account)
    client = _connect_imap(selected)
    try:
        client.select(selected.get("imap_mailbox", "INBOX"))
        status, fetch_data = client.fetch(str(message_id), "(RFC822)")
        if status != "OK" or not fetch_data:
            raise RuntimeError(f"IMAP fetch failed: {status}")
        raw = next((item[1] for item in fetch_data if isinstance(item, tuple)), None)
        if not raw:
            raise RuntimeError("IMAP fetch returned no message bytes")
        parsed = _parse_message(str(message_id), raw, include_body=True)
    finally:
        try:
            client.logout()
        except Exception:
            pass

    return _result(
        f"Read email: {parsed.get('subject') or 'No subject'}.",
        {"account": selected.get("id"), "message": parsed},
    )


def send_email(
    account: str | None = None,
    to: list[str] | str | None = None,
    subject: str = "",
    body: str = "",
    cc: list[str] | str | None = None,
    bcc: list[str] | str | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    recipients = _normalize_addresses(to)
    cc_list = _normalize_addresses(cc)
    bcc_list = _normalize_addresses(bcc)
    if not recipients:
        raise ValueError("At least one recipient is required")
    selected = _resolve_account(account)

    if not confirmed:
        return _result(
            "Please confirm before I send email to "
            f"{', '.join(recipients)} with subject: {subject}.",
            {"to": recipients, "cc": cc_list, "bcc": bcc_list, "subject": subject},
            status="confirmation_required",
        )

    host = selected.get("smtp_host")
    if not host:
        raise ValueError("Selected email account is missing smtp_host")
    port = int(selected.get("smtp_port") or 587)
    username = selected.get("smtp_username") or selected.get("email")
    password = selected.get("smtp_password") or selected.get("password")

    msg = EmailMessage()
    msg["From"] = selected.get("from") or selected.get("email") or username
    msg["To"] = ", ".join(recipients)
    if cc_list:
        msg["Cc"] = ", ".join(cc_list)
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(host, port, timeout=30) as smtp:
        if selected.get("smtp_starttls", True):
            smtp.starttls()
        if username and password:
            smtp.login(username, password)
        smtp.send_message(msg)

    return _result(
        f"Email sent to {', '.join(recipients)}.",
        {
            "account": selected.get("id"),
            "to": recipients,
            "cc": cc_list,
            "bcc": bcc_list,
            "subject": subject,
        },
    )


def test_email_account(account: str | None = None) -> dict[str, Any]:
    selected = _resolve_account(account)
    checks: dict[str, str] = {}
    if selected.get("imap_host"):
        client = _connect_imap(selected)
        try:
            client.select(selected.get("imap_mailbox", "INBOX"))
            checks["imap"] = "ok"
        finally:
            try:
                client.logout()
            except Exception:
                pass
    if selected.get("smtp_host"):
        host = selected.get("smtp_host")
        port = int(selected.get("smtp_port") or 587)
        with smtplib.SMTP(host, port, timeout=15) as smtp:
            if selected.get("smtp_starttls", True):
                smtp.starttls()
            username = selected.get("smtp_username") or selected.get("email")
            password = selected.get("smtp_password") or selected.get("password")
            if username and password:
                smtp.login(username, password)
            checks["smtp"] = "ok"
    return _result(
        "Email account connection succeeded.", {"account": selected.get("id"), "checks": checks}
    )


def _normalize_addresses(value: list[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]
