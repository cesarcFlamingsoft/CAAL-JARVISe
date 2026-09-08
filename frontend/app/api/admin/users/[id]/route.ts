/**
 * Administrator: read and edit one user (display name, role, status).
 */
import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import {
  apiError,
  backendFailure,
  browserUser,
  guardMutation,
  isDisplayName,
  isUserId,
  noStoreJson,
  readJsonObject,
  requireAdmin,
} from '@/lib/auth/guard';

export const dynamic = 'force-dynamic';

type Params = { params: Promise<{ id: string }> };

export async function GET(req: NextRequest, { params }: Params) {
  const auth = await requireAdmin(req);
  if (!auth.ok) return auth.response;
  const { id } = await params;
  if (!isUserId(id)) return apiError(404, 'not_found');

  const result = await callAsUser(auth.config, auth.user.userId, `/admin/users/${id}`, {
    method: 'GET',
  });
  if (!result.ok) return backendFailure(result.status, result.data);
  const user = browserUser(result.data, { admin: true });
  return user ? noStoreJson({ user }) : apiError(502, 'backend_unavailable');
}

export async function PATCH(req: NextRequest, { params }: Params) {
  const auth = await requireAdmin(req);
  if (!auth.ok) return auth.response;
  const { id } = await params;
  if (!isUserId(id)) return apiError(404, 'not_found');
  const blocked = guardMutation(req, auth.config, auth.user.userId);
  if (blocked) return blocked;

  const body = await readJsonObject(req);
  if (!body || Object.keys(body).length === 0) return apiError(422, 'invalid');
  const update: Record<string, string> = {};
  for (const [key, value] of Object.entries(body)) {
    if (key === 'displayName' && isDisplayName(value)) {
      update.display_name = value.trim();
    } else if (key === 'role' && (value === 'admin' || value === 'member')) {
      update.role = value;
    } else if (key === 'status' && (value === 'active' || value === 'suspended')) {
      update.status = value;
    } else {
      return apiError(422, 'invalid');
    }
  }

  const result = await callAsUser(auth.config, auth.user.userId, `/admin/users/${id}`, {
    method: 'PATCH',
    body: update,
  });
  if (!result.ok) return backendFailure(result.status, result.data);
  const user = browserUser(result.data, { admin: true });
  return user ? noStoreJson({ user }) : apiError(502, 'backend_unavailable');
}
