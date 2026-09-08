/**
 * Where the signed-in user has said their weather should come from.
 *
 * The settings panel reads this to show the current choice: a chosen city, or
 * a shared browser position and when it expires. Taking no reading and
 * naming no coordinate, it is a plain read: uncacheable, reduced through the
 * weather parser, and scoped to the session user by the backend.
 */
import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import { apiError, backendFailure, noStoreJson, requireUser } from '@/lib/auth/guard';
import { browserWeatherLocation } from '@/lib/dashboard/weather';

export const dynamic = 'force-dynamic';

export async function GET(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;

  const result = await callAsUser(auth.config, auth.user.userId, '/users/me/weather/location', {
    method: 'GET',
  });
  if (!result.ok) return backendFailure(result.status, result.data);
  const settings = browserWeatherLocation(result.data);
  return settings ? noStoreJson(settings) : apiError(502, 'backend_unavailable');
}
