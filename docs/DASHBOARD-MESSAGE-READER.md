# Dashboard message reader

Click a recent inbox message to open a portal dialog over the mounted FRIDAY
workspace. The reader traps focus, makes the background inert, restores focus
on close, and supports Escape, backdrop clicks, a close button, loading, retry,
and a scrollable text body. Closing cancels the browser request; stale responses
are ignored. After a successful owner-scoped live fetch and full text validation,
Gmail and Microsoft messages are marked read when the existing grant permits it.
The response carries the provider-confirmed unread state; the inbox updates its
unread dot and count only after that validated response. A refreshed feed becomes
authoritative again. Feed/list requests never modify messages.

The BFF `GET /api/dashboard/inbox/message?connectionId=…&messageId=…` uses
`requireUser` and `callAsUser`, a 30-second timeout, and no-store responses.
The backend path is `/users/me/dashboard/inbox/message` with the same query.
Both require exactly one of each identifier. Only the authenticated user's live
connection can be read; another user's, unknown, and revoked connections return
`404 not_found`. The backend never consults or updates the knowledge index.

The allowlisted contract is `id`, `connection_id`, `provider`, `subject`,
`sender`, `recipients`, `received_at`, `unread`, `body`, and optional `link`.
The BFF maps snake_case to camelCase. Bodies are limited to 65,536 characters;
recipients to 50 entries of 254 characters; subject/sender/link retain existing
summary bounds. Bodies retain newlines, remove control characters, and render
only as React text. No HTML rendering, attachment downloads, tracking images,
raw provider fields, or tokens reach the browser. Large bodies can be truncated;
provider responses over the transport byte budget return unavailable.

Provider details:

- [Gmail messages.get](https://developers.google.com/workspace/gmail/api/reference/rest/v1/users.messages/get):
  `format=full`, bounded MIME traversal (100 parts, depth 20), base64url decoding
  with the declared charset, preferring inline `text/plain`. HTML-only messages
  are converted using the standard-library HTML parser. Script/style/head and
  embedded-object contents are suppressed; attributes and URLs are discarded.
  Attachments and externally stored MIME parts are not downloaded. A message
  without an available inline text body returns unavailable.
- [Microsoft Graph message.get](https://learn.microsoft.com/en-us/graph/api/message-get?view=graph-rest-1.0):
  `/me/messages/{id}`, an explicit field selection including recipients, and
  `Prefer: outlook.body-content-type="text"`. Unexpected HTML is converted to
  text defensively. Only HTTPS provider links are offered.
- [Zoho email content](https://www.zoho.com/mail/help/api/get-email-content.html):
  resolves exactly one mail account from the grant, then locates the exact
  message and its folder ID in a bounded live `newMails` search (50 rows).
  Calls `/api/accounts/{accountId}/folders/{folderId}/messages/{messageId}/content`.
  Missing, ambiguous, moved, or out-of-window messages return unavailable;
  account/folder identifiers are never guessed or taken from a browser URL.

All reads reuse the existing OAuth scopes, token refresh and one-time 401 retry,
HTTPS-only transport, redirect refusal, 256 KiB default response budget,
10-second socket timeout and 20-second total provider budget. No additional
OAuth scopes or dependencies are introduced. Failures preserve the existing
bounded account states; exception text and message contents are not logged.

Focused verification:

```sh
.venv/bin/pytest tests/test_provider_data.py tests/test_dashboard_api.py -q
cd frontend
node --test lib/dashboard/message-reader.test.ts lib/dashboard/feed-guard.test.ts lib/dashboard/provider-data.test.ts
npx tsc --noEmit
```

Frontend tests execute the BFF and dialog logic with framework/DOM boundaries
substituted, using the existing TypeScript compiler. They are not browser E2E
or live-provider tests.


Read-state writes use [Gmail messages.modify](https://developers.google.com/workspace/gmail/api/reference/rest/v1/users.messages/modify)
with `removeLabelIds: ["UNREAD"]`, or [Microsoft message update](https://learn.microsoft.com/en-us/graph/api/message-update?view=graph-rest-1.0)
with `isRead: true`. They share the existing byte/time limits and single 401 refresh
retry. Malformed detail, ownership failures, and provider read errors never trigger
a write. Failed or malformed write responses return a bounded error, never a false
success. Already-read messages need no write.

OAuth/auth behavior and default scopes are unchanged. Existing read-only grants
retain their observed unread state. Enabling read-state updates requires an operator
scope override containing `https://www.googleapis.com/auth/gmail.modify` (or
`https://mail.google.com/`) for Google, or `Mail.ReadWrite` for Microsoft, plus any
other desired existing product scopes; users must reconnect to consent to them.
Zoho remains read-only with its observed state: no guessed mutation endpoint or scope.
