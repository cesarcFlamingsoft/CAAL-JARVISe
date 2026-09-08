import email
import json

import pytest


def configure_settings(monkeypatch, tmp_path, settings_data):
    from caal import settings as settings_module

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps(settings_data))
    monkeypatch.setattr(settings_module, "SETTINGS_PATH", settings_path)
    settings_module._settings_cache = None
    return settings_module


class FakeIMAP:
    instances = []

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.selected = None
        FakeIMAP.instances.append(self)

    def login(self, username, password):
        self.username = username
        self.password = password
        return "OK", []

    def select(self, mailbox):
        self.selected = mailbox
        return "OK", []

    def search(self, charset, criteria):
        self.criteria = criteria
        return "OK", [b"1 2"]

    def fetch(self, message_id, parts):
        msg = email.message.EmailMessage()
        msg["Subject"] = "Hello"
        msg["From"] = "sender@example.com"
        msg["To"] = "cesar@example.com"
        msg["Date"] = "Sat, 30 May 2026 10:00:00 -0600"
        msg.set_content(
            f"Body for {message_id.decode() if isinstance(message_id, bytes) else message_id}"
        )
        return "OK", [(b"RFC822", msg.as_bytes())]

    def logout(self):
        self.logged_out = True


class FakeSMTP:
    instances = []

    def __init__(self, host, port, timeout=30):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.started_tls = False
        self.sent = []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def starttls(self):
        self.started_tls = True

    def login(self, username, password):
        self.username = username
        self.password = password

    def send_message(self, message):
        self.sent.append(message)


@pytest.mark.parametrize("account_arg", [None, "personal", "cesar@example.com"])
def test_email_search_uses_configured_imap_and_returns_speakable_contract(
    monkeypatch, tmp_path, account_arg
):
    configure_settings(
        monkeypatch,
        tmp_path,
        {
            "email_accounts": [
                {
                    "id": "personal",
                    "display_name": "Personal",
                    "email": "cesar@example.com",
                    "imap_host": "imap.example.com",
                    "imap_port": 993,
                    "imap_ssl": True,
                    "imap_username": "cesar@example.com",
                    "imap_password": "imap_secret",
                }
            ]
        },
    )
    from caal.tools import email_tools

    monkeypatch.setattr(email_tools.imaplib, "IMAP4_SSL", FakeIMAP)
    result = email_tools.search_email(account=account_arg, query="from:sender", limit=1)

    assert result["message"] == "Found 1 email matching from:sender."
    assert result["data"]["account"] == "personal"
    assert result["data"]["messages"][0]["subject"] == "Hello"
    assert FakeIMAP.instances[-1].host == "imap.example.com"
    assert FakeIMAP.instances[-1].username == "cesar@example.com"


def test_email_send_refuses_without_confirmation_and_sends_after_confirmation(
    monkeypatch, tmp_path
):
    configure_settings(
        monkeypatch,
        tmp_path,
        {
            "email_accounts": [
                {
                    "id": "personal",
                    "email": "cesar@example.com",
                    "smtp_host": "smtp.example.com",
                    "smtp_port": 587,
                    "smtp_starttls": True,
                    "smtp_username": "cesar@example.com",
                    "smtp_password": "smtp_secret",
                }
            ]
        },
    )
    from caal.tools import email_tools

    monkeypatch.setattr(email_tools.smtplib, "SMTP", FakeSMTP)
    blocked = email_tools.send_email(
        account="personal",
        to=["friend@example.com"],
        subject="Lunch",
        body="Want lunch?",
    )
    assert blocked["status"] == "confirmation_required"
    assert not FakeSMTP.instances

    sent = email_tools.send_email(
        account="personal",
        to=["friend@example.com"],
        subject="Lunch",
        body="Want lunch?",
        confirmed=True,
    )
    smtp = FakeSMTP.instances[-1]
    assert sent["message"] == "Email sent to friend@example.com."
    assert smtp.started_tls is True
    assert smtp.username == "cesar@example.com"
    assert smtp.sent[0]["Subject"] == "Lunch"


def test_calendar_lists_events_from_ics_source(monkeypatch, tmp_path):
    ics = """BEGIN:VCALENDAR
BEGIN:VEVENT
UID:1
SUMMARY:Standup
DTSTART:20260530T160000Z
DTEND:20260530T163000Z
LOCATION:Zoom
DESCRIPTION:Daily sync
END:VEVENT
END:VCALENDAR
"""
    ics_path = tmp_path / "calendar.ics"
    ics_path.write_text(ics)
    configure_settings(
        monkeypatch,
        tmp_path,
        {
            "calendar_sources": [
                {
                    "id": "work",
                    "provider": "ics",
                    "display_name": "Work",
                    "url": ics_path.as_uri(),
                    "writable": False,
                }
            ]
        },
    )
    from caal.tools.calendar_tools import list_calendar_events

    result = list_calendar_events(
        source="work",
        start="2026-05-30T00:00:00+00:00",
        end="2026-05-31T00:00:00+00:00",
    )

    assert result["message"] == "Found 1 calendar event."
    assert result["data"]["events"][0]["title"] == "Standup"
    assert result["data"]["events"][0]["source"] == "work"


def test_calendar_finds_free_time_between_occupied_events(monkeypatch, tmp_path):
    ics = """BEGIN:VCALENDAR
BEGIN:VEVENT
UID:morning
SUMMARY:Standup
DTSTART:20260530T090000Z
DTEND:20260530T093000Z
END:VEVENT
BEGIN:VEVENT
UID:afternoon
SUMMARY:Review
DTSTART:20260530T110000Z
DTEND:20260530T113000Z
END:VEVENT
END:VCALENDAR
"""
    ics_path = tmp_path / "calendar.ics"
    ics_path.write_text(ics)
    configure_settings(
        monkeypatch,
        tmp_path,
        {
            "calendar_sources": [
                {
                    "id": "work",
                    "provider": "ics",
                    "url": ics_path.as_uri(),
                    "writable": False,
                }
            ]
        },
    )
    from caal.tools.calendar_tools import find_free_time

    result = find_free_time(
        source="work",
        start="2026-05-30T08:00:00+00:00",
        end="2026-05-30T12:00:00+00:00",
        duration_minutes=60,
    )

    assert result["message"] == "Found 2 available time slots."
    assert result["data"]["slots"] == [
        {"start": "2026-05-30T08:00:00+00:00", "end": "2026-05-30T09:00:00+00:00"},
        {"start": "2026-05-30T09:30:00+00:00", "end": "2026-05-30T11:00:00+00:00"},
    ]


def test_calendar_free_time_tool_is_registered():
    from caal.tools.registry import create_default_registry

    free_time = create_default_registry().get("calendar.find_free_time")
    update = create_default_registry().get("calendar.update_event")
    delete = create_default_registry().get("calendar.delete_event")

    assert free_time.requires_confirmation is False
    assert free_time.handler.__name__ == "find_free_time"
    assert update.requires_confirmation is True
    assert update.handler.__name__ == "update_calendar_event"
    assert delete.requires_confirmation is True
    assert delete.handler.__name__ == "delete_calendar_event"


def test_calendar_create_event_requires_confirmation_and_writable_source(monkeypatch, tmp_path):
    configure_settings(
        monkeypatch,
        tmp_path,
        {
            "calendar_sources": [
                {
                    "id": "work",
                    "provider": "caldav",
                    "display_name": "Work",
                    "url": "https://cal.example.com/user/work/",
                    "username": "cesar",
                    "password": "cal_secret",
                    "writable": True,
                }
            ]
        },
    )
    from caal.tools import calendar_tools

    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))

        class Response:
            status_code = 201
            text = ""

            def raise_for_status(self):
                pass

        return Response()

    monkeypatch.setattr(calendar_tools.requests, "request", fake_request)
    blocked = calendar_tools.create_calendar_event(
        source="work",
        title="Dentist",
        start="2026-05-30T17:00:00+00:00",
        end="2026-05-30T18:00:00+00:00",
    )
    assert blocked["status"] == "confirmation_required"
    assert calls == []

    created = calendar_tools.create_calendar_event(
        source="work",
        title="Dentist",
        start="2026-05-30T17:00:00+00:00",
        end="2026-05-30T18:00:00+00:00",
        confirmed=True,
    )
    assert created["message"] == "Created calendar event: Dentist."
    assert calls[0][0] == "PUT"
    assert "BEGIN:VEVENT" in calls[0][2]["data"]


def test_calendar_update_and_delete_require_confirmation(monkeypatch, tmp_path):
    configure_settings(
        monkeypatch,
        tmp_path,
        {
            "calendar_sources": [
                {
                    "id": "work",
                    "provider": "caldav",
                    "url": "https://cal.example.com/user/work/",
                    "writable": True,
                }
            ]
        },
    )
    from caal.tools import calendar_tools

    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))

        class Response:
            def raise_for_status(self):
                pass

        return Response()

    monkeypatch.setattr(calendar_tools.requests, "request", fake_request)
    blocked = calendar_tools.update_calendar_event(
        source="work", event_id="dentist", title="Dentist", confirmed=False
    )
    assert blocked["status"] == "confirmation_required"
    assert calls == []

    updated = calendar_tools.update_calendar_event(
        source="work",
        event_id="dentist",
        title="Dentist rescheduled",
        start="2026-05-30T17:00:00+00:00",
        end="2026-05-30T18:00:00+00:00",
        confirmed=True,
    )
    assert updated["message"] == "Updated calendar event: Dentist rescheduled."
    assert calls[0][0] == "PUT"
    assert calls[0][1].endswith("/dentist.ics")

    deleted = calendar_tools.delete_calendar_event(
        source="work", event_id="dentist", confirmed=True
    )
    assert deleted["message"] == "Deleted calendar event: dentist."
    assert calls[1][0] == "DELETE"
    assert calls[1][1].endswith("/dentist.ics")


@pytest.mark.asyncio
async def test_native_tools_are_discovered_and_executed_by_llm_router(monkeypatch, tmp_path):
    configure_settings(
        monkeypatch,
        tmp_path,
        {
            "native_tools_enabled": True,
            "email_accounts": [
                {
                    "id": "personal",
                    "email": "cesar@example.com",
                    "imap_host": "imap.example.com",
                    "imap_port": 993,
                    "imap_ssl": True,
                    "imap_username": "cesar@example.com",
                    "imap_password": "imap_secret",
                }
            ],
        },
    )
    import importlib

    llm_node = importlib.import_module("caal.llm.llm_node")
    from caal.tools import email_tools

    monkeypatch.setattr(email_tools.imaplib, "IMAP4_SSL", FakeIMAP)

    class Agent:
        _llm_tools_cache = None

    agent = Agent()
    tools = await llm_node._discover_tools(agent)
    names = {tool["function"]["name"] for tool in tools}
    assert "email.search" in names

    result = await llm_node._execute_single_tool(
        agent, "email.search", {"account": "personal", "query": "from:sender", "limit": 1}
    )
    assert result["data"]["messages"][0]["subject"] == "Hello"
