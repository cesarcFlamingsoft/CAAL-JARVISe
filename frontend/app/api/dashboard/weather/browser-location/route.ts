/**
 * The position the signed-in user's browser offered, with their say-so.
 *
 * PUT forwards it only when the body carries a literal true consent flag and
 * coordinates that could be real; DELETE forgets it. The backend stores it
 * rounded to about a kilometre and expires it within hours, and never
 * answers it back, so nothing here -- request, response or log -- turns a
 * position into a record of where somebody is.
 */
import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import {
  apiError,
  backendFailure,
  guardMutation,
  noStoreJson,
  readJsonObject,
  requireUser,
} from '@/lib/auth/guard';
import { browserWeatherLocation, consentedPosition } from '@/lib/dashboard/weather';

export const dynamic = 'force-dynamic';

const SAVE_TIMEOUT_MS = 15_000;

export async function PUT(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;

  // No consent, no position: the flag is a decision made in the page, never
  // a default this route may assume.
  const position = consentedPosition(await readJsonObject(req));
  if (position === null) return apiError(422, 'invalid');

  const result = await callAsUser(
    auth.config,
    auth.user.userId,
    '/users/me/weather/browser-location',
    {
      method: 'PUT',
      body: position,
      timeoutMs: SAVE_TIMEOUT_MS,
    }
  );
  if (!result.ok) return backendFailure(result.status, result.data);
  const settings = browserWeatherLocation(result.data);
  return settings ? noStoreJson(settings) : apiError(502, 'backend_unavailable');
}

export async function DELETE(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;

  const result = await callAsUser(
    auth.config,
    auth.user.userId,
    '/users/me/weather/browser-location',
    { method: 'DELETE' }
  );
  if (!result.ok) return backendFailure(result.status, result.data);
  const settings = browserWeatherLocation(result.data);
  return settings ? noStoreJson(settings) : apiError(502, 'backend_unavailable');
}
