import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import {
  apiError,
  backendFailure,
  guardMutation,
  isUserId,
  noStoreJson,
  readJsonObject,
  requireAdmin,
  requireUser,
} from '@/lib/auth/guard';
import { connectionId, parseAccess } from './contract';

export async function accessRoute(req: NextRequest, method: 'GET' | 'PUT', id?: string) {
  const auth = await (id ? requireAdmin(req) : requireUser(req));
  if (!auth.ok) return auth.response;
  if (id && !isUserId(id)) return apiError(404, 'not_found');
  let body: Record<string, unknown> | undefined;
  if (method === 'PUT') {
    const blocked = guardMutation(req, auth.config, auth.user.userId);
    if (blocked) return blocked;
    const data = await readJsonObject(req);
    if (
      !id ||
      !data ||
      Object.keys(data).some((k) => !['enabled', 'connection_id'].includes(k)) ||
      typeof data.enabled !== 'boolean' ||
      (data.connection_id !== null && !connectionId(data.connection_id))
    )
      return apiError(422, 'invalid');
    body = data;
  }
  const path = id ? `/admin/users/${id}/home-assistant` : '/users/me/home-assistant';
  const result = await callAsUser(auth.config, auth.user.userId, path, {
    method,
    ...(body ? { body } : {}),
  });
  if (!result.ok) return backendFailure(result.status, result.data);
  const data = parseAccess(result.data);
  return data ? noStoreJson(data) : apiError(502, 'backend_unavailable');
}
