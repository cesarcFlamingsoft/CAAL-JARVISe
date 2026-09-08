/**
 * Administrator: approve or clear a user's single callback number.
 *
 * The number travels from the admin's browser to the backend exactly once and
 * is never returned, logged, or echoed in an error; responses only ever say
 * whether a number is set.
 */
import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import {
  apiError,
  backendFailure,
  browserUser,
  guardMutation,
  isUserId,
  noStoreJson,
  readJsonObject,
  requireAdmin,
} from '@/lib/auth/guard';

export const dynamic = 'force-dynamic';

type Params = { params: Promise<{ id: string }> };

const E164_LIKE = /^\+[0-9()\s.-]{7,31}$/;

export async function PUT(req: NextRequest, { params }: Params) {
  const auth = await requireAdmin(req);
  if (!auth.ok) return auth.response;
  const { id } = await params;
  if (!isUserId(id)) return apiError(404, 'not_found');
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;

  const body = await readJsonObject(req);
  const number = typeof body?.number === 'string' ? body.number.trim() : '';
  if (!body || Object.keys(body).length !== 1 || !E164_LIKE.test(number)) {
    return apiError(422, 'callback_number');
  }

  const result = await callAsUser(
    auth.config,
    auth.user.userId,
    `/admin/users/${id}/callback-number`,
    { method: 'PUT', body: { number } }
  );
  if (!result.ok) return backendFailure(result.status, result.data);
  const user = browserUser(result.data, { admin: true });
  return user ? noStoreJson({ user }) : apiError(502, 'backend_unavailable');
}

export async function DELETE(req: NextRequest, { params }: Params) {
  const auth = await requireAdmin(req);
  if (!auth.ok) return auth.response;
  const { id } = await params;
  if (!isUserId(id)) return apiError(404, 'not_found');
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;

  const result = await callAsUser(
    auth.config,
    auth.user.userId,
    `/admin/users/${id}/callback-number`,
    { method: 'DELETE' }
  );
  if (!result.ok) return backendFailure(result.status, result.data);
  const user = browserUser(result.data, { admin: true });
  return user ? noStoreJson({ user }) : apiError(502, 'backend_unavailable');
}
