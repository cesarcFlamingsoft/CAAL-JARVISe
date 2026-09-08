/**
 * The signed-in user's own profile: read, and edit the display name only.
 * Role, status and the callback number are administrator-controlled.
 */
import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import {
  apiError,
  backendFailure,
  browserUser,
  guardMutation,
  isDisplayName,
  noStoreJson,
  readJsonObject,
  requireUser,
} from '@/lib/auth/guard';

export const dynamic = 'force-dynamic';

export async function GET(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;

  const result = await callAsUser(auth.config, auth.user.userId, '/users/me', { method: 'GET' });
  if (!result.ok) return backendFailure(result.status, result.data);
  const user = browserUser(result.data);
  return user ? noStoreJson({ user }) : apiError(502, 'backend_unavailable');
}

export async function PATCH(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;

  const body = await readJsonObject(req);
  if (!body || !isDisplayName(body.displayName) || Object.keys(body).length !== 1) {
    return apiError(422, 'invalid');
  }
  const result = await callAsUser(auth.config, auth.user.userId, '/users/me', {
    method: 'PATCH',
    body: { display_name: body.displayName.trim() },
  });
  if (!result.ok) return backendFailure(result.status, result.data);
  const user = browserUser(result.data);
  return user ? noStoreJson({ user }) : apiError(502, 'backend_unavailable');
}
