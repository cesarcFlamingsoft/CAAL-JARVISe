import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import {
  apiError,
  backendFailure,
  guardMutation,
  noStoreJson,
  readJsonObject,
  requireAdmin,
} from '@/lib/auth/guard';
import {
  connectionId,
  entityId,
  parseEnrollment,
  parseStatus,
  satelliteId,
  scopeValue,
} from './contract';

export async function satelliteRoute(
  req: NextRequest,
  method: 'GET' | 'POST' | 'PUT' | 'DELETE',
  id?: string
) {
  const auth = await requireAdmin(req);
  if (!auth.ok) return auth.response;
  if (id && !satelliteId(id)) return apiError(404, 'not_found');
  if (method !== 'GET') {
    const blocked = guardMutation(req, auth.config, auth.user.userId);
    if (blocked) return blocked;
  }
  let body: Record<string, unknown> | undefined;
  if (method === 'POST' || method === 'PUT') {
    const parsed = await readJsonObject(req);
    const keys = method === 'POST' ? ['satellite_id', 'connection_id'] : ['connection_id', 'scope'];
    if (
      !parsed ||
      Object.keys(parsed).length !== 2 ||
      Object.keys(parsed).some((k) => !keys.includes(k)) ||
      !connectionId(parsed.connection_id) ||
      (method === 'POST' ? !entityId(parsed.satellite_id) : !scopeValue(parsed.scope))
    )
      return apiError(422, 'invalid');
    body = parsed;
  }
  const selected = new URL(req.url).searchParams.get('connection_id');
  if (selected && !connectionId(selected)) return apiError(422, 'invalid');
  const query =
    method === 'GET' && selected ? `?connection_id=${encodeURIComponent(selected)}` : '';
  const result = await callAsUser(
    auth.config,
    auth.user.userId,
    `/admin/satellites${id ? `/${id}` : ''}${query}`,
    {
      method,
      ...(body ? { body } : {}),
    }
  );
  if (!result.ok) return backendFailure(result.status, result.data);
  if (method === 'PUT') return noStoreJson({ status: 'configured' });
  if (method === 'DELETE') return noStoreJson({ status: 'revoked' });
  const data = method === 'POST' ? parseEnrollment(result.data) : parseStatus(result.data);
  return data ? noStoreJson(data) : apiError(502, 'backend_unavailable');
}
