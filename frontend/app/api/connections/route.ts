/**
 * The signed-in user's own provider connections: which are live, and which
 * providers the operator has configured. Names and booleans only; the backend
 * never returns a token and this route forwards nothing it does not recognise.
 */
import type { NextRequest } from 'next/server';
import { callAsUser } from '@/lib/auth/backend';
import { apiError, backendFailure, noStoreJson, requireUser } from '@/lib/auth/guard';
import { browserConnectionList } from '@/lib/connections/protocol';

export const dynamic = 'force-dynamic';

export async function GET(req: NextRequest) {
  const auth = await requireUser(req);
  if (!auth.ok) return auth.response;

  const result = await callAsUser(auth.config, auth.user.userId, '/users/me/connections', {
    method: 'GET',
  });
  if (!result.ok) return backendFailure(result.status, result.data);
  const list = browserConnectionList(result.data);
  return list ? noStoreJson(list) : apiError(502, 'backend_unavailable');
}
