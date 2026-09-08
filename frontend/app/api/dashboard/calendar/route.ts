/**
 * Upcoming events from the signed-in user's connected accounts.
 *
 * Proxies the backend's calendar feed as the session user and reduces the
 * answer through the feed parser before it reaches the browser, so a token, a
 * body or a raw provider payload can never be forwarded by accident. Read-only,
 * uncacheable, and bounded by its own timeout: the backend may spend up to its
 * per-account budget talking to the providers.
 */
import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import { apiError, backendFailure, noStoreJson, requireUser } from '@/lib/auth/guard';
import { browserCalendarFeed, feedQuery } from '@/lib/dashboard/provider-data';

export const dynamic = 'force-dynamic';

/** The backend reads every account within a 20 s budget; allow for that plus overhead. */
const FEED_TIMEOUT_MS = 30_000;
const DEFAULTS = { days: 7, limit: 25 };

export async function GET(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;

  const query = feedQuery(req.nextUrl.searchParams, DEFAULTS);
  if (query === null) return apiError(422, 'invalid');

  const result = await callAsUser(
    auth.config,
    auth.user.userId,
    '/users/me/dashboard/calendar?' + query,
    { method: 'GET', timeoutMs: FEED_TIMEOUT_MS }
  );
  if (!result.ok) return backendFailure(result.status, result.data);
  const feed = browserCalendarFeed(result.data);
  return feed ? noStoreJson(feed) : apiError(502, 'backend_unavailable');
}
