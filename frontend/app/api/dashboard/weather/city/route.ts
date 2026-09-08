/**
 * The signed-in user's hand-picked weather city.
 *
 * PUT saves one of the places the city search returned; DELETE goes back to
 * whatever position the browser last offered, or to nothing. The body is
 * reduced to exactly the three fields the backend takes, so a label a page
 * invented can never travel: the backend keeps only what its own lookup said
 * about the place. Both are mutations, so both carry the origin, CSRF and
 * per-user budget checks.
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
import { browserWeatherLocation, citySelection } from '@/lib/dashboard/weather';

export const dynamic = 'force-dynamic';

const SAVE_TIMEOUT_MS = 15_000;

export async function PUT(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;

  const chosen = citySelection(await readJsonObject(req));
  if (chosen === null) return apiError(422, 'invalid');

  const result = await callAsUser(auth.config, auth.user.userId, '/users/me/weather/city', {
    method: 'PUT',
    body: chosen,
    timeoutMs: SAVE_TIMEOUT_MS,
  });
  if (!result.ok) return backendFailure(result.status, result.data);
  const settings = browserWeatherLocation(result.data);
  return settings ? noStoreJson(settings) : apiError(502, 'backend_unavailable');
}

export async function DELETE(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;

  const result = await callAsUser(auth.config, auth.user.userId, '/users/me/weather/city', {
    method: 'DELETE',
  });
  if (!result.ok) return backendFailure(result.status, result.data);
  const settings = browserWeatherLocation(result.data);
  return settings ? noStoreJson(settings) : apiError(502, 'backend_unavailable');
}
