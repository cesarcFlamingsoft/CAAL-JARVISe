/**
 * The signed-in user own local reminders.
 *
 * Proxies the backend own reminder feed as the session user and reduces the
 * answer through the reminder parser before it reaches the browser, so a
 * payload that is not exactly the backend own shape can never be forwarded by
 * accident. Reminders are private: there is no unauthenticated path here, no
 * parameter that could name another owner, and nothing cacheable in the
 * response. Read-only.
 */
import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import { apiError, backendFailure, noStoreJson, requireUser } from '@/lib/auth/guard';
import { browserReminders } from '@/lib/dashboard/reminders';

export const dynamic = 'force-dynamic';

/** A local SQLite read; it does not wait on any provider. */
const REMINDERS_TIMEOUT_MS = 10_000;

export async function GET(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;

  const result = await callAsUser(auth.config, auth.user.userId, '/users/me/dashboard/reminders', {
    method: 'GET',
    timeoutMs: REMINDERS_TIMEOUT_MS,
  });
  if (!result.ok) return backendFailure(result.status, result.data);
  const feed = browserReminders(result.data);
  if (!feed) {
    // The backend answered and the answer could not be read: without this the
    // widget's one generic sentence is the only trace such a day leaves. The
    // method, the internal path, the backend status and the branch are the
    // whole of it -- never a body, an owner, a title or a time.
    console.error('[dashboard] GET /users/me/dashboard/reminders %d unreadable', result.status);
    return apiError(502, 'backend_unavailable');
  }
  return noStoreJson(feed);
}
