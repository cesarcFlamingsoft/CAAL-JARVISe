/**
 * Places matching what someone typed, for the weather location setting.
 *
 * The query is bounded here before it becomes a backend call, so a mistyped
 * or oversized search never reaches the city lookup at all; the backend
 * caches each place name for a day on top of that. Read-only, uncacheable,
 * and reduced to the bounded search shape before the browser sees it.
 */
import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import { apiError, backendFailure, noStoreJson, requireUser } from '@/lib/auth/guard';
import { browserCitySearch, citySearchQuery } from '@/lib/dashboard/weather';

export const dynamic = 'force-dynamic';

const SEARCH_TIMEOUT_MS = 15_000;

export async function GET(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;

  const query = citySearchQuery(req.nextUrl.searchParams);
  if (query === null) return apiError(422, 'invalid');

  const result = await callAsUser(
    auth.config,
    auth.user.userId,
    '/users/me/weather/cities?' + query,
    { method: 'GET', timeoutMs: SEARCH_TIMEOUT_MS }
  );
  if (!result.ok) return backendFailure(result.status, result.data);
  const search = browserCitySearch(result.data);
  return search ? noStoreJson(search) : apiError(502, 'backend_unavailable');
}
