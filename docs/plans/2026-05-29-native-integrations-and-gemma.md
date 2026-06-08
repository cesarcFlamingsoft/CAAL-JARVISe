# Native Integrations and Gemma Model Migration Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Remove n8n as the assistant integration layer and replace it with native, configurable tools for SMTP/email, multi-source calendars, reminders, alarms, and a better Ollama model option such as Gemma.

**Architecture:** Add a first-class native tool layer inside CAAL with provider-specific adapters behind stable voice tool contracts. Keep Home Assistant and Friday direct integrations, but disable/remove n8n discovery and n8n UI as the default integration path. Store all end-user integration config in `settings.json` via the UI/settings API — never requiring Docker/server file edits for email, calendars, or reminders. Redact secrets from API responses, preserve redacted placeholders on save, and expose safe confirmation policies for side-effecting actions.

**Tech Stack:** Python FastAPI/LiveKit Agents, SMTP/IMAP for Zoho Mail/custom-domain mail and generic providers, Microsoft Graph, Google Calendar API, CalDAV/ICS, Apple EventKit or macOS CLI bridge, SQLite/APScheduler for alarms, Ollama Gemma-family model.

---

## Implementation Status — 2026-05-30

Completed in the deployed CAAL stack:

- Runtime settings layer for `email_accounts` and `calendar_sources`.
- Recursive secret redaction for nested settings and preservation of `********` placeholders on save.
- Zoho Mail/default IMAP+SMTP presets.
- Native tool registry wiring into the LLM tool loop.
- Email provider execution:
  - `email.search` via IMAP.
  - `email.read` via IMAP.
  - `email.send` via SMTP with confirmation-required contract.
- Calendar provider execution:
  - `calendar.list_events` via ICS feeds/files and CalDAV REPORT.
  - `calendar.create_event` via CalDAV PUT with confirmation-required contract.
- Setup test endpoints:
  - `POST /setup/test-email`.
  - `POST /setup/test-calendar`.
- Settings UI forms for email/calendar configuration plus test-connection buttons.

Remaining follow-up:

- Real-account credential setup in the UI.
- OAuth-native Google Calendar and Microsoft Graph providers; current first pass supports them as ICS/read-only URL sources unless configured through CalDAV-compatible URLs.
- Email reply, calendar free-time, and calendar delete tools.
- Full reminders/alarms execution layer beyond current UI/settings toggles.
- Prompt tuning for stronger native-tool selection and confirmation phrasing.

---

## Target Tool Surface

Native tools should be stable regardless of provider:

### Email
- `email_search(query, account=None, max_results=10)`
- `email_read(message_id, account=None)`
- `email_send(to, subject, body, cc=None, bcc=None, account=None, confirm=True)`
- `email_reply(message_id, body, account=None, confirm=True)`

### Calendar
- `calendar_list(start, end, sources=None)`
- `calendar_create_event(title, start, end, source=None, attendees=None, location=None, description=None, confirm=True)`
- `calendar_find_free_time(start, end, duration_minutes, sources=None)`
- `calendar_delete_event(event_id, source=None, confirm=True)`

### Reminders
- `reminder_create(title, due=None, list_name=None)`
- `reminder_list(range='today', list_name=None)`
- `reminder_complete(reminder_id)`

### Alarms and timers
- `timer_start(duration_seconds, label=None)`
- `alarm_set(time, label=None, recurrence=None)`
- `alarm_list()`
- `alarm_cancel(alarm_id)`

---

## Provider Strategy

### Email
Use SMTP for sending and IMAP for reading/searching. The first supported target account is Zoho Mail with a custom domain, configured entirely from the UI at runtime. No `.env`, Docker Compose, or server-file edits should be required after deployment.

UI fields should include:

- Account id / display name
- Email address, including custom-domain addresses such as `name@yourdomain.com`
- Provider preset: `zoho`, `generic_imap_smtp`
- IMAP host/port/SSL
- SMTP host/port/STARTTLS or SSL
- Username
- App password/token
- Default account toggle
- Test connection button for IMAP and SMTP

Zoho preset defaults:

```json
{
  "provider": "zoho",
  "imap_host": "imap.zoho.com",
  "imap_port": 993,
  "imap_ssl": true,
  "smtp_host": "smtp.zoho.com",
  "smtp_port": 587,
  "smtp_starttls": true
}
```

Stored settings example:

```json
{
  "email_accounts": [
    {
      "id": "personal",
      "provider": "zoho",
      "display_name": "Cesar",
      "email": "cesar@yourdomain.com",
      "smtp_host": "smtp.zoho.com",
      "smtp_port": 587,
      "smtp_starttls": true,
      "smtp_username": "cesar@yourdomain.com",
      "smtp_password": "...",
      "imap_host": "imap.zoho.com",
      "imap_port": 993,
      "imap_ssl": true,
      "imap_username": "cesar@yourdomain.com",
      "imap_password": "...",
      "default": true
    }
  ]
}
```

Important: `/settings` must recursively redact nested account passwords/tokens, and `POST /settings` must preserve existing nested secrets when the UI submits `********`. The frontend must not force users to paste JSON for normal setup; JSON can remain as an advanced/import mode.

### Calendars
Support multiple source types behind one interface, also configured live from the UI/settings API instead of Docker/env files. The UI should support adding/removing calendar sources, testing credentials, choosing a default writable calendar, and marking read-only subscriptions.

Provider targets:

1. Google Calendar: OAuth via Google Calendar API.
2. Outlook/Microsoft 365: Microsoft Graph OAuth.
3. Zoho Calendar: CalDAV if available, otherwise Zoho Calendar API OAuth.
4. Apple/iCloud: CalDAV with app-specific password; on macOS optionally EventKit/Calendar.app bridge later.
5. Generic ICS: read-only subscribed calendar feeds.

Minimum `calendar_sources` schema:

```json
{
  "calendar_sources": [
    {
      "id": "work",
      "provider": "zoho_caldav",
      "display_name": "Work Calendar",
      "url": "https://calendar.zoho.com/caldav/...",
      "username": "cesar@yourdomain.com",
      "password": "...",
      "default": true,
      "writable": true
    }
  ]
}
```

Use a normalized event model:

```python
@dataclass
class CalendarEvent:
    id: str
    source_id: str
    title: str
    start: datetime
    end: datetime
    timezone: str | None = None
    location: str | None = None
    description: str | None = None
    attendees: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)
```

### Alarms
Use local SQLite storage and an async scheduler in the CAAL agent process. When an alarm fires:

1. If a LiveKit room is active, call `session.say(...)`.
2. Also POST to `/announce` where possible.
3. If no active session, persist missed alarm and announce next connection.

---

## Task 1: Disable n8n as default discovery path

**Objective:** Stop CAAL from depending on n8n workflows for tools.

**Files:**
- Modify: `src/caal/settings.py`
- Modify: `src/caal/integrations/mcp_loader.py`
- Modify: `voice_agent.py`
- Modify: `frontend/components/setup/integrations-step.tsx`
- Modify: `frontend/components/livekit/agent-control-bar/reload-tools-button.tsx`

**Steps:**
1. Change `DEFAULT_SETTINGS["n8n_enabled"]` to `False` and mark n8n as legacy.
2. In `voice_agent.py`, guard all n8n workflow discovery behind `legacy_n8n_enabled` or remove from startup path.
3. Keep existing n8n code temporarily for rollback, but do not surface it as the main integration path.
4. Update frontend copy from “n8n Workflows” to “Legacy n8n Workflows”.
5. Verify: `curl http://localhost:8889/settings` shows n8n disabled by default.

---

## Task 2: Add recursive secret redaction to settings API

**Objective:** Prevent tokens/passwords/API keys from leaking via `/settings`, including nested email/calendar provider config submitted through the UI.

**Files:**
- Modify: `src/caal/settings.py`
- Test: `tests/test_settings_redaction.py`

**Steps:**
1. Add `SENSITIVE_KEY_PARTS = ("token", "key", "secret", "password", "credential")`.
2. Make `load_settings_safe()` recursively redact sensitive keys.
3. Ensure POST `/settings` still stores real values server-side.
4. Add tests for nested email/calendar account secrets.
5. Run: `uv run pytest tests/test_settings_redaction.py -v`.

---

## Task 3: Create native tool registry

**Objective:** Centralize tool definitions, handlers, and confirmation policy.

**Files:**
- Create: `src/caal/tools/registry.py`
- Modify: `src/caal/llm/llm_node.py`
- Modify: `voice_agent.py`
- Test: `tests/test_tool_registry.py`

**Implementation sketch:**

```python
@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., Awaitable[Any]]
    requires_confirmation: bool = False
    category: str = "general"
```

**Steps:**
1. Create registry and `register_tool()` / `get_tool_definitions()` / `execute_tool()`.
2. Move `hass_assist`, `friday`, and `web_search` into or through registry adapters.
3. Update `llm_node.py` to discover tools from registry first.
4. Verify existing tools still appear and execute.

---

## Task 4: Add UI-configurable Zoho SMTP/IMAP email integration

**Objective:** Allow end users to add, test, edit, and remove Zoho Mail/custom-domain email accounts from the UI without touching `.env`, Docker Compose, or server files.

**Files:**
- Create: `src/caal/tools/email.py`
- Modify: `src/caal/settings.py`
- Modify: `src/caal/webhooks.py`
- Modify: `voice_agent.py`
- Modify: `frontend/components/settings/settings-panel.tsx`
- Test: `tests/test_email_tools.py`
- Test: `tests/test_settings_and_tools.py`

**Dependencies:** Python stdlib `smtplib`, `imaplib`, `email`; no new dependency for first pass.

**Steps:**
1. Add provider preset metadata for `zoho` and `generic_imap_smtp`.
2. Replace the raw email JSON textarea with an account editor form, keeping JSON import/export as advanced mode only.
3. Add backend settings validation for email account ids, provider, host, port, TLS mode, usernames, and default account uniqueness.
4. Add `/setup/test-email` or `/tools/email/test-account` to verify IMAP login and SMTP connection using submitted UI values.
5. Implement account selection by `id` or default account.
6. Implement IMAP search/read with safe result limits.
7. Implement SMTP send.
8. Add confirmation-required metadata for send/reply.
9. Add tests with mocked SMTP/IMAP classes and nested secret preservation.
10. Verify by adding a Zoho custom-domain account from the UI, refreshing settings, and confirming secrets return as `********` while the account remains usable.

---

## Task 5: Add UI-configurable calendar provider interface

**Objective:** Normalize Google, Outlook, Zoho, Apple/iCloud, and ICS calendars behind one native calendar tool set, with all source setup editable from the UI at runtime.

**Files:**
- Create: `src/caal/tools/calendar/base.py`
- Create: `src/caal/tools/calendar/ics_provider.py`
- Create: `src/caal/tools/calendar/caldav_provider.py`
- Create: `src/caal/tools/calendar/google_provider.py`
- Create: `src/caal/tools/calendar/microsoft_provider.py`
- Create: `src/caal/tools/calendar/zoho_provider.py`
- Create: `src/caal/tools/calendar/tools.py`
- Modify: `src/caal/settings.py`
- Test: `tests/test_calendar_tools.py`

**Dependencies:**
- `httpx`
- `icalendar`
- `caldav`
- `google-api-python-client`, `google-auth-oauthlib` for Google
- Microsoft Graph via `httpx` OAuth first; MSAL optional later

**Steps:**
1. Add `calendar_sources` settings list with provider, credentials, read/write capability, default source, and display name.
2. Replace the raw calendar JSON textarea with a source editor form, keeping JSON import/export as advanced mode only.
3. Add backend validation for calendar source ids, provider types, URLs, credentials, and one default writable source.
4. Add `/setup/test-calendar` or `/tools/calendar/test-source` to verify provider connectivity from submitted UI values.
5. Implement read-only ICS provider first.
6. Implement CalDAV provider for Apple/iCloud and many Zoho setups.
7. Implement Google provider.
8. Implement Microsoft Graph provider.
9. Implement Zoho provider as CalDAV-first, API later if required.
10. Implement `calendar_list` aggregating and sorting events.
11. Implement `calendar_create_event` only for writable providers.
12. Add tests with provider fakes and nested secret preservation.

---

## Task 6: Add UI-configurable reminders

**Objective:** Add personal reminders independent of n8n, selectable by the end user from the settings UI without server edits.

**Files:**
- Create: `src/caal/tools/reminders.py`
- Modify: `src/caal/settings.py`
- Modify: `src/caal/webhooks.py`
- Modify: `frontend/components/settings/settings-panel.tsx`
- Test: `tests/test_reminders_tools.py`

**Strategy:**
- Provider choice is stored in `reminders_provider` and editable from the UI.
- On macOS: use `remindctl` if available for Apple Reminders sync.
- Fallback: local SQLite reminder store.

**Steps:**
1. Add UI selector for `reminders_provider`: `local` or `apple`.
2. Add a test/status endpoint that reports whether the selected provider is usable.
3. Implement Apple bridge using subprocess calls to `remindctl --json`.
4. Implement local fallback store.
5. Register `reminder_create`, `reminder_list`, `reminder_complete`.

---

## Task 7: Add alarms and timers

**Objective:** Add reliable timers and alarms that can speak through JARVIS.

**Files:**
- Create: `src/caal/tools/alarms.py`
- Create: `src/caal/scheduler.py`
- Modify: `voice_agent.py`
- Test: `tests/test_alarms.py`

**Steps:**
1. Add SQLite table for alarms/timers.
2. Start scheduler at agent startup.
3. Register timer/alarm tools.
4. On fire, announce to active session or queue missed notification.
5. Add tests for due alarm detection and cancellation.

---

## Task 8: Add model migration support for Gemma-family models

**Objective:** Make it easy to switch from `ministral-3:14b` to a Gemma-family Ollama model.

**Files:**
- Modify: `src/caal/settings.py`
- Modify: `frontend/components/setup/provider-step.tsx`
- Modify: `docs/APPLE-SILICON.md` or `README.md`

**Steps:**
1. Add recommended model dropdown entries: `gemma3:12b`, `gemma3:27b`, and whatever exact `gemma4` Ollama tag exists in the target environment.
2. Add model availability check using `/api/tags`.
3. Add a setup command in docs: `ollama pull <model-tag>`.
4. Verify from CAAL container that `ollama_host` is reachable.

---

## Task 9: Update prompt for native tools

**Objective:** Teach JARVIS to use native tools, not n8n.

**Files:**
- Modify: `prompt/custom.md`

**Steps:**
1. Remove “create new tools using n8n_create_caal_tool”.
2. Add direct rules for email, calendar, reminders, alarms.
3. Add confirmation rules for send/create/delete operations.
4. Keep responses voice-friendly and short.

---

## Task 10: End-to-end verification

**Objective:** Prove the assistant works without n8n.

**Commands:**

```bash
uv run pytest -v
uv run ruff check src/ voice_agent.py
curl http://localhost:8889/health
curl http://localhost:8889/settings
```

Manual voice tests:
- “What’s on my calendar today?”
- “Send an email draft to Cesar saying test.”
- “Remind me tomorrow at nine to check backups.”
- “Set a timer for five minutes.”
- “What model are you running?”

Expected: no n8n workflow calls required.
