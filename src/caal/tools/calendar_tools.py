"""Native calendar provider tools for CAAL.

Supports read-only ICS feeds/files and basic CalDAV REPORT/PUT operations using
requests. Google/Outlook can be configured as ICS feeds for reads until OAuth is
added; writable sources use CalDAV-compatible URLs.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import requests

from caal import settings as settings_module


def _result(message: str, data: dict[str, Any] | None = None, status: str = "ok") -> dict[str, Any]:
    return {"status": status, "message": message, "data": data or {}}


def _sources() -> list[dict[str, Any]]:
    return [
        s
        for s in settings_module.load_settings().get("calendar_sources", [])
        if isinstance(s, dict)
    ]


def _resolve_sources(
    source: str | None = None, writable: bool | None = None
) -> list[dict[str, Any]]:
    sources = _sources()
    if writable is not None:
        sources = [s for s in sources if bool(s.get("writable", False)) is writable]
    if not source or source == "all":
        if not sources:
            raise ValueError("No matching calendar sources are configured")
        return sources
    for candidate in sources:
        if source in {
            str(candidate.get("id", "")),
            str(candidate.get("display_name", "")),
        }:
            return [candidate]
    raise ValueError(f"Calendar source not found: {source}")


def _parse_datetime(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z") and "T" in text and "-" in text:
        text = text[:-1] + "+00:00"
    if re.match(r"^\d{8}T\d{6}Z$", text):
        return datetime.strptime(text, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    if re.match(r"^\d{8}T\d{6}$", text):
        return datetime.strptime(text, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    if re.match(r"^\d{8}$", text):
        return datetime.strptime(text, "%Y%m%d").replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _ical_datetime(dt: str) -> str:
    parsed = _parse_datetime(dt).astimezone(timezone.utc)
    return parsed.strftime("%Y%m%dT%H%M%SZ")


def _load_ics(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return Path(parsed.path).read_text()
    if parsed.scheme in {"http", "https"}:
        with urlopen(url, timeout=20) as response:  # noqa: S310 - user-configured calendar URL
            return response.read().decode("utf-8", errors="replace")
    return Path(url).read_text()


def _unfold_ics(text: str) -> list[str]:
    lines: list[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.startswith((" ", "\t")) and lines:
            lines[-1] += line[1:]
        elif line:
            lines.append(line)
    return lines


def _parse_ics_events(text: str, source_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current: dict[str, str] | None = None
    for line in _unfold_ics(text):
        if line == "BEGIN:VEVENT":
            current = {}
            continue
        if line == "END:VEVENT" and current is not None:
            start = current.get("DTSTART")
            end = current.get("DTEND")
            if start:
                events.append(
                    {
                        "id": current.get("UID"),
                        "source": source_id,
                        "title": current.get("SUMMARY", "Untitled event"),
                        "start": _parse_datetime(start).isoformat(),
                        "end": _parse_datetime(end).isoformat() if end else None,
                        "location": current.get("LOCATION", ""),
                        "notes": current.get("DESCRIPTION", ""),
                    }
                )
            current = None
            continue
        if current is not None and ":" in line:
            key, value = line.split(":", 1)
            current[key.split(";", 1)[0]] = _ical_unescape(value)
    return events


def _ical_escape(value: str) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace(",", "\\,")
        .replace(";", "\\;")
    )


def _ical_unescape(value: str) -> str:
    return (
        str(value)
        .replace("\\n", "\n")
        .replace("\\,", ",")
        .replace("\\;", ";")
        .replace("\\\\", "\\")
    )


def list_calendar_events(
    source: str | None = "all", start: str = "", end: str = ""
) -> dict[str, Any]:
    start_dt = _parse_datetime(start)
    end_dt = _parse_datetime(end)
    events: list[dict[str, Any]] = []
    for selected in _resolve_sources(source):
        provider = str(selected.get("provider", "ics")).lower()
        if provider in {"ics", "google", "outlook"} or str(selected.get("url", "")).endswith(
            ".ics"
        ):
            source_events = _parse_ics_events(
                _load_ics(str(selected.get("url", ""))), str(selected.get("id"))
            )
        elif provider in {"caldav", "zoho_caldav", "icloud_caldav"}:
            source_events = _list_caldav_events(selected, start_dt, end_dt)
        else:
            raise ValueError(f"Unsupported calendar provider for reads: {provider}")
        events.extend(_filter_events(source_events, start_dt, end_dt))

    events.sort(key=lambda e: e.get("start") or "")
    count = len(events)
    return _result(
        f"Found {count} calendar {'event' if count == 1 else 'events'}.",
        {"events": events},
    )


def find_free_time(
    source: str | None = "all",
    start: str = "",
    end: str = "",
    duration_minutes: int = 30,
) -> dict[str, Any]:
    """Return unoccupied time ranges that can fit the requested duration."""
    if duration_minutes <= 0:
        raise ValueError("duration_minutes must be greater than zero")
    start_dt = _parse_datetime(start)
    end_dt = _parse_datetime(end)
    if end_dt <= start_dt:
        raise ValueError("end must be after start")

    events = list_calendar_events(source, start, end)["data"]["events"]
    busy_ranges = sorted(
        (
            max(start_dt, _parse_datetime(event["start"])),
            min(end_dt, _parse_datetime(event["end"] or event["start"])),
        )
        for event in events
    )
    slots: list[dict[str, str]] = []
    cursor = start_dt
    minimum_duration = timedelta(minutes=duration_minutes)
    for busy_start, busy_end in busy_ranges:
        if busy_start - cursor >= minimum_duration:
            slots.append({"start": cursor.isoformat(), "end": busy_start.isoformat()})
        if busy_end > cursor:
            cursor = busy_end
    if end_dt - cursor >= minimum_duration:
        slots.append({"start": cursor.isoformat(), "end": end_dt.isoformat()})

    count = len(slots)
    return _result(
        f"Found {count} available time {'slot' if count == 1 else 'slots'}.",
        {"slots": slots},
    )


def _filter_events(
    events: list[dict[str, Any]], start: datetime, end: datetime
) -> list[dict[str, Any]]:
    filtered = []
    for event in events:
        event_start = _parse_datetime(str(event.get("start")))
        event_end = _parse_datetime(str(event.get("end") or event.get("start")))
        if event_start < end and event_end >= start:
            filtered.append(event)
    return filtered


def _list_caldav_events(
    source: dict[str, Any], start: datetime, end: datetime
) -> list[dict[str, Any]]:
    start_utc = start.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    end_utc = end.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    body = f"""<?xml version="1.0" encoding="utf-8" ?>
<c:calendar-query xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
  <d:prop><d:getetag/><c:calendar-data/></d:prop>
  <c:filter><c:comp-filter name="VCALENDAR"><c:comp-filter name="VEVENT">
    <c:time-range start="{start_utc}" end="{end_utc}"/>
  </c:comp-filter></c:comp-filter></c:filter>
</c:calendar-query>"""
    response = requests.request(
        "REPORT",
        str(source.get("url", "")),
        data=body,
        headers={"Depth": "1", "Content-Type": "application/xml; charset=utf-8"},
        auth=_auth(source),
        timeout=30,
    )
    response.raise_for_status()
    ics_blobs = re.findall(r"BEGIN:VCALENDAR.*?END:VCALENDAR", response.text, flags=re.DOTALL)
    events: list[dict[str, Any]] = []
    for blob in ics_blobs:
        events.extend(_parse_ics_events(blob, str(source.get("id"))))
    return events


def create_calendar_event(
    source: str | None = None,
    title: str = "",
    start: str = "",
    end: str = "",
    attendees: list[str] | None = None,
    location: str = "",
    notes: str = "",
    confirmed: bool = False,
) -> dict[str, Any]:
    selected = _resolve_sources(source, writable=True)[0]
    if not confirmed:
        return _result(
            f"Please confirm before I create calendar event: {title}.",
            {"source": selected.get("id"), "title": title, "start": start, "end": end},
            status="confirmation_required",
        )

    provider = str(selected.get("provider", "")).lower()
    if provider not in {"caldav", "zoho_caldav", "icloud_caldav"}:
        raise ValueError(f"Calendar source {selected.get('id')} is not writable through CalDAV")

    event_uid = f"{uuid.uuid4()}@caal"
    ics = _build_event_ics(event_uid, title, start, end, attendees or [], location, notes)
    base_url = str(selected.get("url", "")).rstrip("/")
    event_url = f"{base_url}/{event_uid}.ics"
    response = requests.request(
        "PUT",
        event_url,
        data=ics,
        headers={"Content-Type": "text/calendar; charset=utf-8"},
        auth=_auth(selected),
        timeout=30,
    )
    response.raise_for_status()
    return _result(
        f"Created calendar event: {title}.",
        {"source": selected.get("id"), "event_id": event_uid, "url": event_url, "title": title},
    )


def _writable_caldav_source(source: str | None) -> dict[str, Any]:
    selected = _resolve_sources(source, writable=True)[0]
    provider = str(selected.get("provider", "")).lower()
    if provider not in {"caldav", "zoho_caldav", "icloud_caldav"}:
        raise ValueError(f"Calendar source {selected.get('id')} is not writable through CalDAV")
    return selected


def _event_url(source: dict[str, Any], event_id: str) -> str:
    if not event_id or "/" in event_id or "\\" in event_id:
        raise ValueError("event_id must be a single calendar event identifier")
    filename = event_id if event_id.endswith(".ics") else f"{event_id}.ics"
    return f"{str(source.get('url', '')).rstrip('/')}/{filename}"


def update_calendar_event(
    source: str | None = None,
    event_id: str = "",
    title: str = "",
    start: str = "",
    end: str = "",
    attendees: list[str] | None = None,
    location: str = "",
    notes: str = "",
    confirmed: bool = False,
) -> dict[str, Any]:
    """Replace a CalDAV event after explicit confirmation."""
    selected = _writable_caldav_source(source)
    if not confirmed:
        return _result(
            f"Please confirm before I update calendar event: {event_id}.",
            {"source": selected.get("id"), "event_id": event_id},
            status="confirmation_required",
        )
    if not title or not start or not end:
        raise ValueError("title, start, and end are required to update a calendar event")
    response = requests.request(
        "PUT",
        _event_url(selected, event_id),
        data=_build_event_ics(
            event_id.removesuffix(".ics"), title, start, end, attendees or [], location, notes
        ),
        headers={"Content-Type": "text/calendar; charset=utf-8"},
        auth=_auth(selected),
        timeout=30,
    )
    response.raise_for_status()
    return _result(
        f"Updated calendar event: {title}.",
        {"source": selected.get("id"), "event_id": event_id, "title": title},
    )


def delete_calendar_event(
    source: str | None = None, event_id: str = "", confirmed: bool = False
) -> dict[str, Any]:
    """Delete a CalDAV event after explicit confirmation."""
    selected = _writable_caldav_source(source)
    if not confirmed:
        return _result(
            f"Please confirm before I delete calendar event: {event_id}.",
            {"source": selected.get("id"), "event_id": event_id},
            status="confirmation_required",
        )
    response = requests.request(
        "DELETE", _event_url(selected, event_id), auth=_auth(selected), timeout=30
    )
    response.raise_for_status()
    return _result(
        f"Deleted calendar event: {event_id.removesuffix('.ics')}.",
        {"source": selected.get("id"), "event_id": event_id},
    )


def _build_event_ics(
    event_uid: str,
    title: str,
    start: str,
    end: str,
    attendees: list[str],
    location: str,
    notes: str,
) -> str:
    attendee_lines = "\n".join(f"ATTENDEE:mailto:{_ical_escape(a)}" for a in attendees)
    parts = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//CAAL//Native Calendar//EN",
        "BEGIN:VEVENT",
        f"UID:{event_uid}",
        f"DTSTAMP:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        f"DTSTART:{_ical_datetime(start)}",
        f"DTEND:{_ical_datetime(end)}",
        f"SUMMARY:{_ical_escape(title)}",
    ]
    if location:
        parts.append(f"LOCATION:{_ical_escape(location)}")
    if notes:
        parts.append(f"DESCRIPTION:{_ical_escape(notes)}")
    if attendee_lines:
        parts.extend(attendee_lines.split("\n"))
    parts.extend(["END:VEVENT", "END:VCALENDAR", ""])
    return "\n".join(parts)


def _auth(source: dict[str, Any]) -> tuple[str, str] | None:
    username = source.get("username")
    password = source.get("password") or source.get("access_token")
    if username and password:
        return str(username), str(password)
    return None


def test_calendar_source(source: str | None = None) -> dict[str, Any]:
    selected = _resolve_sources(source)[0]
    provider = str(selected.get("provider", "ics")).lower()
    if provider in {"ics", "google", "outlook"} or str(selected.get("url", "")).endswith(".ics"):
        _load_ics(str(selected.get("url", "")))
    else:
        propfind_body = (
            '<?xml version="1.0"?>'
            '<d:propfind xmlns:d="DAV:">'
            "<d:prop><d:displayname/></d:prop>"
            "</d:propfind>"
        )
        response = requests.request(
            "PROPFIND",
            str(selected.get("url", "")),
            data=propfind_body,
            headers={"Depth": "0", "Content-Type": "application/xml"},
            auth=_auth(selected),
            timeout=20,
        )
        response.raise_for_status()
    return _result("Calendar source connection succeeded.", {"source": selected.get("id")})
