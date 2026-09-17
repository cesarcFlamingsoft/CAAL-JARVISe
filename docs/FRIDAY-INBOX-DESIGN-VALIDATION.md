# FRIDAY inbox and design validation

No commit, push, or deployment. Existing uncommitted reader work was preserved.

## Behavior

Owned Gmail/Microsoft live details are fully validated before marking read with
existing writable grants. Gmail removes only UNREAD; Microsoft patches only
isRead. The response returns provider-confirmed state. Wrong owners, malformed
identifiers/details and failed reads never cause a write. Failed writes return
bounded errors. No provider payloads or tokens reach responses/logs. Writes share
the existing byte/time budgets and single 401 token refresh retry.

OAuth defaults/authentication are unchanged. Gmail requires gmail.modify (or full
mail scope); Microsoft requires Mail.ReadWrite. Existing read-only grants and Zoho
retain observed unread state. Operator scope overrides and reconsent are required
for read-only accounts; see [reader documentation](DASHBOARD-MESSAGE-READER.md).

The authenticated no-store BFF and backend owner checks remain intact. Inbox dots
and account counts update only from valid detail; subsequent live feeds are
authoritative. Text-only rendering and close/late-response cancellation remain.

## Design

Shared body-level FRIDAY tokens reach pages and portal dialogs: black/navy glass,
cyan accents, consistent borders/shadows/focus and typography. Account, settings,
setup, connections result, admin/company/satellite, Home Assistant result,
change-password, dashboard and email reader use that palette. Error and warning
states are preserved. No intentional green/emerald/lime UI occurrences remain in
production TSX/CSS or default app color configuration.

## TDD and checks

- Provider regression before changes: **3 failures, 4 passes (RED)**.
- Frontend reader/design regressions before changes: **5 failures, 7 passes (RED)**.
- Password-surface/default-accent guards also confirmed RED before fixes.
- Focused provider/endpoint suite after changes: **67 passed (GREEN)**.
- `.venv/bin/python -m pytest tests/test_provider_data.py tests/test_dashboard_api.py tests/test_oauth_providers.py tests/test_provider_connections.py -q`: **102 passed**.
- From frontend, `node --test 'lib/**/*.test.ts'`: **440 passed**.
- From frontend, `./node_modules/.bin/tsc --noEmit`: **passed**.
- From frontend, `JARVIS_BUILD_DIR=.next-friday-review npm run build`: **passed**, isolated output, no deployment.
- `.venv/bin/ruff check src/caal/provider_data.py tests/test_provider_data.py tests/test_dashboard_api.py`: **passed**.
- `git diff --check`: **passed**.

Build warnings: existing satellite hook dependency, inferred workspace root from
multiple lockfiles, and missing metadataBase. Tests mock provider HTTP and frontend
framework boundaries. No live-provider or authenticated browser screenshot/E2E
verification was performed.

## Files edited by this task

- `docs/DASHBOARD-MESSAGE-READER.md`
- `docs/FRIDAY-INBOX-DESIGN-VALIDATION.md`
- `frontend/app-config.ts`
- `frontend/app/(app)/account/page.tsx`
- `frontend/app/(app)/admin/page.tsx`
- `frontend/app/(app)/connections/result/page.tsx`
- `frontend/app/admin/company/page.tsx`
- `frontend/app/admin/satellite/page.tsx`
- `frontend/app/change-password/page.tsx`
- `frontend/app/home-assistant/result/page.tsx`
- `frontend/app/layout.tsx`
- `frontend/components/account/account-panel.tsx`
- `frontend/components/admin/admin-panel.tsx`
- `frontend/components/auth/change-password-form.tsx`
- `frontend/components/dashboard/hands-dock.tsx`
- `frontend/components/dashboard/message-reader.tsx`
- `frontend/components/dashboard/voice-dock.tsx`
- `frontend/components/dashboard/widgets/inbox-widget.tsx`
- `frontend/components/dashboard/widgets/work-widget.tsx`
- `frontend/components/home-assistant/access.tsx`
- `frontend/components/livekit/agent-control-bar/reload-tools-button.tsx`
- `frontend/components/livekit/agent-control-bar/server-wake-word-indicator.tsx`
- `frontend/components/livekit/agent-control-bar/tool-status-indicator.tsx`
- `frontend/components/livekit/agent-control-bar/wake-word-toggle.tsx`
- `frontend/components/settings/local-model.tsx`
- `frontend/components/settings/settings-modal.tsx`
- `frontend/components/settings/settings-panel.tsx`
- `frontend/components/setup/integrations-step.tsx`
- `frontend/components/setup/provider-step.tsx`
- `frontend/components/setup/setup-wizard.tsx`
- `frontend/lib/dashboard/design-system.test.ts`
- `frontend/lib/dashboard/message-reader.test.ts`
- `frontend/styles/globals.css`
- `src/caal/provider_data.py`
- `tests/test_dashboard_api.py`
- `tests/test_provider_data.py`

The backend dashboard endpoint, frontend BFF message route, auth backend helper,
feed guard and provider parser were inspected. Their pre-existing working-tree
changes remain; this task did not need to alter their behavior.
