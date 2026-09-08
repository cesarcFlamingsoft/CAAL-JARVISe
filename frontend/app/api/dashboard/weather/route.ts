/**
 * The current weather where the signed-in user's weather is read from.
 *
 * Proxies the backend as the session user and reduces the answer through the
 * weather parser before it reaches the browser, so an upstream payload can
 * never be forwarded by accident. The browser never speaks to a weather
 * service itself: the backend holds the once-an-hour cap, and this route is
 * the only way in. Read-only and uncacheable.
 */
import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import { apiError, backendFailure, noStoreJson, requireUser } from '@/lib/auth/guard';
import { browserWeather } from '@/lib/dashboard/weather';

export const dynamic = 'force-dynamic';

/** The backend may spend a socket timeout on the upstream; allow for that. */
const WEATHER_TIMEOUT_MS = 15_000;

export async function GET(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;

  const result = await callAsUser(auth.config, auth.user.userId, '/users/me/weather', {
    method: 'GET',
    timeoutMs: WEATHER_TIMEOUT_MS,
  });
  if (!result.ok) return backendFailure(result.status, result.data);
  const weather = browserWeather(result.data);
  return weather ? noStoreJson(weather) : apiError(502, 'backend_unavailable');
}
